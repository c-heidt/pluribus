"""Training server for multi-process CFR.

The :class:`Server` is the process-level orchestrator: it spawns a
pool of :class:`~poker_ai.blueprint.multiprocess.worker.Worker` subprocesses,
owns the shared :class:`~poker_ai.tables.cfr_tables.CFRTables`, dispatches
training jobs through a multiprocessing queue, and drives the
sync-cycle-based schedule (sync barriers, strategy updates,
discounting, checkpoints).  All per-step training logic (CFR
traversals, pruning decisions, the discount formula, LUT loading)
lives in :mod:`poker_ai.blueprint.training` so this class stays a thin
orchestrator.

Process and concurrency model
-----------------------------
- The server process owns the :class:`CFRTables`, the LMDB indexes,
  the shared-memory chunk files, and the
  :class:`~poker_ai.tables.checkpoint.CheckpointManager`.  Workers
  inherit everything via fork copy-on-write.
- Jobs travel over a bounded :class:`multiprocessing.JoinableQueue`.
  Sync barriers are implemented by draining the queue with
  :meth:`_join_queue`, broadcasting a ``sync`` job to every worker,
  and draining again.
- Fatal worker exceptions are surfaced via a shared
  :class:`multiprocessing.Event`.  :meth:`_join_queue` polls the
  event and raises :class:`WorkerError` so the server loop can stop
  cleanly without blocking on a dead worker.
- ``SIGTERM`` / ``SIGINT`` are handled by the
  :class:`CheckpointManager`, which sets an event that the main loop
  polls at every iteration.
"""

import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Union

from poker_ai.tables.checkpoint import CheckpointManager
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.tables.chunk_store import CHUNK_SIZE as _CHUNK_SIZE
from poker_ai.tables.warm_start import apply_warm_start
from poker_ai.blueprint.bias import BiasClass
from poker_ai.blueprint.multiprocess.worker import Worker
from poker_ai.blueprint.training import (
    DiscountState,
    at_sync_barrier,
    should_checkpoint,
    should_discount,
    should_update_strategy,
)
from information_abstraction import load_info_set_lut, prewarm_lut

log = logging.getLogger("sync.server")


def _startup_signal_handler(signum: int, frame) -> None:
    """Exit cleanly when SIGTERM/SIGINT arrives during server startup.

    Server startup includes LUT joblib deserialisation, LUT
    pre-warming, and LMDB env opens — operations that can run for
    minutes before :class:`CheckpointManager` is constructed and
    overrides the signal handlers.  Without this handler the default
    Python disposition (terminate) applies: SLURM's grace-period
    SIGTERM lands during startup, Python dies without logging, and
    after the grace window slurm logs the job as KILLED rather than
    cleanly TERMINATED.

    No training state has been mutated yet, so there is nothing to
    checkpoint.  We log the signal and call :func:`sys.exit` so
    Python unwinds normally (running ``atexit`` hooks and flushing
    log handlers) before the process exits.
    """
    sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    log.warning(
        f"{sig_name} received during server startup — exiting cleanly "
        "before training begins (no checkpoint needed)"
    )
    sys.exit(143 if signum == signal.SIGTERM else 130)


def _read_persisted_index_capacities(save_path) -> Optional[Dict[int, int]]:
    """Return the index-cache capacities saved in the latest checkpoint, if any.

    Read *before* :class:`CFRTables` is constructed so a resume rebuilds the
    same-size shm index cache the original run used (see
    :func:`poker_ai.tables.cfr_tables._cache_capacity`).  Returns ``None`` on a
    fresh run (no checkpoint) or an older checkpoint without the field.
    """
    import joblib

    checkpoints = sorted(Path(save_path).glob("checkpoint_[0-9]*"))
    for cp in reversed(checkpoints):
        state_file = cp / "server_state.pkl"
        if not state_file.exists():
            continue
        try:
            state = joblib.load(state_file)
        except Exception:
            continue
        caps = state.get("index_cache_capacity")
        if caps:
            return {int(k): int(v) for k, v in caps.items()}
        return None
    return None


class WorkerError(RuntimeError):
    """Raised by the server loop when a worker has reported a fatal error.

    Workers set a shared error event before breaking out of their
    dispatch loop; the server turns that event into this exception via
    :meth:`Server._join_queue` so the main loop can unwind cleanly and
    trigger an unsafe :meth:`Server.terminate`.
    """
    pass


class Server:
    """Coordinates a pool of :class:`Worker` processes running CFR.

    A single :class:`Server` instance is the entry point for a
    multi-process training run.  Constructing it loads the card-info
    LUT, opens the shared tables, restores from a checkpoint if one
    exists, and spawns the worker pool.  Calling :meth:`search` then
    drives the training loop until the wall-clock budget is
    exhausted or a shutdown signal is received.  The caller is
    responsible for calling :meth:`terminate` afterwards to tear down
    the pool.
    """

    def __init__(
        self,
        strategy_interval: int,
        max_runtime_hours: float,
        discount_duration_cycles: int,
        prune_threshold: int,
        c: int,
        n_players: int,
        update_threshold: int,
        save_path: Union[str, Path],
        lut_path: Union[str, Path] = ".",
        pickle_dir: bool = False,
        sync_interval: int = 10,
        discount_interval: int = 1,
        checkpoint_interval: int = 1,
        checkpoint_start_cycles: int = 0,
        start_timestep: int = 0,
        n_processes: Optional[int] = None,
        batch_size: Optional[int] = None,
        strategy_per_job: Optional[int] = None,
        bias: BiasClass = "none",
        bias_magnitude: float = 0.0,
        warm_start: Optional[Union[str, Path]] = None,
    ):
        """Initialise the server and spawn the worker pool.

        All interval and threshold parameters except ``prune_threshold``
        are counted in **sync cycles**.  A sync cycle is
        ``sync_interval`` raw training iterations.  ``prune_threshold``
        stays in raw iterations because it is checked inside the
        per-traversal pruning decision in :func:`cfr_step`.

        Parameters
        ----------
        strategy_interval : int
            Period (in sync cycles) between strategy-update passes.
        max_runtime_hours : float
            Wall-clock budget for this run.  Training stops when
            elapsed time reaches this bound.
        discount_duration_cycles : int
            Length of the LCFR discount window in sync cycles.
        prune_threshold : int
            Raw iteration at which CFR-P becomes eligible in
            :func:`poker_ai.blueprint.training.cfr_step`.
        c : int
            Regret threshold for CFR-P (see
            :func:`poker_ai.blueprint.cfr.cfrp`).
        n_players : int
            Number of players in the game.
        update_threshold : int
            Warm-up in sync cycles before strategy updates begin.
        save_path : str or Path
            Root directory for checkpoints, LMDB indexes, and logs.
        lut_path : str or Path, optional
            Directory holding the card-info LUT.
        pickle_dir : bool, optional
            Use the legacy pickle-directory LUT layout.  Defaults to
            ``False``.
        sync_interval : int, optional
            Number of traversals-per-player between sync barriers.  The
            base unit for every other cycle-based parameter.  Counts
            traversals (not loop ticks) so the same config produces
            equivalent training regardless of how many worker
            processes the hardware supports.
        discount_interval : int, optional
            Period (in sync cycles) between LCFR discount applications.
        checkpoint_interval : int, optional
            Period (in sync cycles) between checkpoint writes.  Every
            checkpoint is retained (never deleted) and serves as a post-flop
            average-strategy snapshot, so this also sets the snapshot cadence.
        checkpoint_start_cycles : int, optional
            Suppress scheduled checkpoints until ``sync_step`` reaches this
            many cycles (``0`` = checkpoint from the beginning, identical to
            having no gate).  The first checkpoint fires at the first
            ``checkpoint_interval`` multiple ``>=`` this value, so setting it
            equal to a multiple of the interval fires exactly there.  Used as
            the average-strategy warm-up so the retained snapshots skip the
            near-random early era.  The end-of-run / SIGTERM checkpoint
            ignores this gate so an orderly stop is always resumable.
        start_timestep : int, optional
            Initial traversals-per-player counter (``0`` on fresh
            runs).  Overridden on resume by the checkpoint manager.
        n_processes : int, optional
            Number of worker processes to spawn.  Defaults to
            ``SLURM_CPUS_PER_TASK - 1`` when running under SLURM or
            ``cpu_count() - 1`` otherwise.  The number of jobs dispatched
            per player per iteration is derived as
            ``max(1, n_processes // n_players)`` to keep every worker
            busy — batching does not change this because each worker
            processes one queue item at a time regardless of batch
            size.  Batching instead reduces the wall-clock *rate*
            of queue ops (each item now carries ``batch_size``
            traversals of work).
        batch_size : int, optional
            Number of CFR traversals executed per ``cfr`` queue item.
            Raising this reduces queue IPC traffic and dispatcher
            pressure at the cost of longer per-job wall time.
            Defaults to the ``PLURIBUS_CFR_BATCH_SIZE`` environment
            variable if set, else ``5``.
        strategy_per_job : int, optional
            Pre-flop UPDATE-STRATEGY playthroughs folded into each ``cfr``
            job (per player, after warm-up).  ``None`` (default) reads the
            ``PLURIBUS_STRATEGY_PER_JOB`` environment variable, falling back
            to ``1`` — one pass covers the whole pre-flop opponent tree per
            deal (full branching).  ``0`` disables the strategy pass entirely.
        """
        # Install a minimal SIGTERM/SIGINT handler immediately so a
        # signal that arrives during the slow startup phases (LUT
        # rsync from the LUT loader's perspective is already done,
        # but joblib deserialise + prewarm + LMDB env open can still
        # take minutes) causes a clean exit rather than the default
        # process-terminate.  No training state has been mutated at
        # this point, so there is nothing to checkpoint — we just
        # exit promptly so slurm logs the job as terminated rather
        # than waiting out the grace period and SIGKILL'ing us.
        # CheckpointManager installs the full "set event, drain
        # final checkpoint" handler later in this constructor,
        # overriding this one.
        signal.signal(signal.SIGTERM, _startup_signal_handler)
        signal.signal(signal.SIGINT, _startup_signal_handler)
        log.info("Early SIGTERM/SIGINT handler installed (startup phase)")

        if n_processes is None:
            slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
            if slurm_cpus is not None:
                n_processes = int(slurm_cpus) - 1
                log.info(f"Using {n_processes} processes (SLURM_CPUS_PER_TASK={slurm_cpus})")
            else:
                n_processes = mp.cpu_count() - 1
                log.info(f"Using {n_processes} processes (cpu_count={mp.cpu_count()})")

        if batch_size is None:
            batch_size = int(os.environ.get("PLURIBUS_CFR_BATCH_SIZE", 5))
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._batch_size = batch_size

        # workers_per_player saturates the worker pool: every worker
        # processes one queue item at a time regardless of batch_size,
        # so the number of outstanding items needed is independent of
        # how much work each item contains.  batch_size instead
        # reduces the *rate* at which items flow through the queue
        # (each item now holds ``batch_size`` traversals' worth of
        # work) — that is the dispatcher-pressure win, not fewer
        # outstanding items.
        self._workers_per_player = max(1, n_processes // n_players)
        traversals_per_loop = (
            self._workers_per_player * self._batch_size * n_players
        )
        log.info(
            f"batch_size={self._batch_size} "
            f"workers_per_player={self._workers_per_player} "
            f"(n_processes={n_processes}, n_players={n_players}) → "
            f"{self._workers_per_player * n_players} queue items per loop, "
            f"{traversals_per_loop} traversals per loop, "
            f"{self._workers_per_player * self._batch_size} per player"
        )

        # Number of pre-flop UPDATE-STRATEGY playthroughs folded into each
        # ``cfr`` job (per player, after warm-up).  The strategy pass is
        # interleaved with CFR and flushed alongside the regret delta at the
        # next sync — no separate barrier.  One pass covers the whole pre-flop
        # opponent tree per deal (full branching), so the default is 1; the old
        # auto-sizing existed to feed the abandoned post-flop average.
        if strategy_per_job is None:
            # Falsy (unset OR the empty string a ``${VAR:-}`` export produces)
            # → fall through to the default.
            env_spj = os.environ.get("PLURIBUS_STRATEGY_PER_JOB")
            strategy_per_job = int(env_spj) if env_spj else 1
        if strategy_per_job < 0:
            raise ValueError(
                f"strategy_per_job must be >= 0, got {strategy_per_job}"
            )
        self._strategy_per_job = strategy_per_job
        log.info(
            f"strategy_per_job={self._strategy_per_job} "
            f"(pre-flop UPDATE-STRATEGY, folded into each cfr job after warm-up)"
        )

        self._bias: BiasClass = bias
        self._bias_magnitude = float(bias_magnitude)
        self._strategy_interval = strategy_interval
        self._max_runtime_hours = max_runtime_hours
        self._prune_threshold = prune_threshold
        self._c = c
        self._n_players = n_players
        self._update_threshold = update_threshold
        self._save_path = Path(save_path)
        self._lut_path = lut_path
        self._pickle_dir = pickle_dir
        self._sync_interval = sync_interval
        self._discount_interval = discount_interval
        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_start_cycles = checkpoint_start_cycles
        self._start_t = start_timestep
        self._discount_state = DiscountState(
            duration_cycles=discount_duration_cycles,
            discount_interval=discount_interval,
        )
        if 0 < checkpoint_start_cycles < discount_duration_cycles:
            log.warning(
                "checkpoint_start_cycles=%d is inside the LCFR discount window "
                "(%d cycles); retained snapshots will include still-discounting "
                "iterates. Pluribus starts snapshotting after the discount "
                "window closes — consider raising it above "
                "discount_duration_cycles.",
                checkpoint_start_cycles,
                discount_duration_cycles,
            )

        # Load the LUT once in the parent; workers inherit the
        # deserialised object via fork copy-on-write, avoiding one
        # load per worker.  Memmap-backed streets (the river on
        # 52-card decks) are eagerly pre-warmed so per-traversal
        # lookups hit RAM instead of paging from disk.
        self._info_set_lut = load_info_set_lut(lut_path, pickle_dir)
        prewarm_lut(self._info_set_lut)

        self._job_queue: mp.JoinableQueue = mp.JoinableQueue(maxsize=n_processes)
        self._logging_queue: mp.Queue = mp.Queue()

        # Per-street shared-memory tables.  The LMDB map_size can be
        # overridden from the environment for very large runs.
        shm_dir = os.environ.get("PLURIBUS_SHM_DIR", "/dev/shm")
        from poker_ai.tables.index import lmdb_map_size_for_players
        from environment.action_space import MAX_ACTIONS_PER_STREET
        lmdb_map_size = int(
            os.environ.get(
                "PLURIBUS_LMDB_MAP_SIZE",
                lmdb_map_size_for_players(n_players),
            )
        )
        log.info(
            f"LMDB map_size={lmdb_map_size // 1024**3} GiB for {n_players} players"
        )

        # Stage a warm-start checkpoint into the save dir BEFORE
        # constructing CFRTables so the CheckpointManager finds it on
        # construction and restores chunks transparently.  No-op when
        # the save dir already contains a checkpoint (resume wins).
        if warm_start is not None:
            apply_warm_start(
                save_path=self._save_path,
                warm_start_path=Path(warm_start),
                expected_n_players=n_players,
            )
        # Optional node-local LMDB staging.  When PLURIBUS_LMDB_LOCAL_DIR
        # is set the runtime LMDB lives on fast scratch (avoids per-
        # lookup NFS lock-table latency), and CheckpointManager
        # mirrors it back to ``save_path/lmdb_index`` at every
        # checkpoint so the persistent copy stays current and the
        # job can resume from /pfs if the local scratch is lost.
        lmdb_persistent_dir = self._save_path / "lmdb_index"
        lmdb_local_env = os.environ.get("PLURIBUS_LMDB_LOCAL_DIR")
        if lmdb_local_env:
            lmdb_runtime_dir = Path(lmdb_local_env)
            lmdb_runtime_dir.mkdir(parents=True, exist_ok=True)
            log.info(
                f"LMDB runtime dir: {lmdb_runtime_dir} (local staging); "
                f"persistent mirror: {lmdb_persistent_dir}"
            )
        else:
            lmdb_runtime_dir = lmdb_persistent_dir
            log.info(f"LMDB runtime dir: {lmdb_runtime_dir} (no local staging)")
        self._lmdb_runtime_dir = lmdb_runtime_dir
        self._lmdb_persistent_dir = lmdb_persistent_dir

        # Shared-memory index cache (on by default): serves the per-node
        # info-set lookup from shm instead of an LMDB read txn.  On resume the
        # capacity persisted in the checkpoint is reused as a floor (so the
        # cache never shrinks below the original run and overflows), but
        # PLURIBUS_INDEX_CAPACITY can still raise a street further — the
        # per-street max of the two is used (see
        # CFRTables._build_index_caches); on a fresh run the size comes from
        # PLURIBUS_INDEX_CAPACITY / existing rows.
        enable_index_cache = os.environ.get("PLURIBUS_INDEX_CACHE", "1") == "1"
        persisted_caps = _read_persisted_index_capacities(self._save_path)
        self._tables = CFRTables(
            index_path=lmdb_runtime_dir,
            shm_dir=shm_dir,
            lmdb_map_size=lmdb_map_size,
            actions_per_street=MAX_ACTIONS_PER_STREET,
            enable_index_cache=enable_index_cache,
            index_capacities=persisted_caps,
        )
        if enable_index_cache:
            self._warn_index_cache_budget()
        self._locks: Dict[str, mp.synchronize.Lock] = {}
        self._error_event: mp.Event = mp.Event()  # type: ignore
        self._current_t: int = self._start_t

        # CheckpointManager registers signal handlers and restores
        # from an existing checkpoint before any worker is spawned so
        # workers observe the restored state.
        self._checkpoint_manager = CheckpointManager(
            self,
            self._save_path,
            lmdb_runtime_dir=lmdb_runtime_dir,
            lmdb_persistent_dir=lmdb_persistent_dir,
        )
        if os.environ.get("TESTING_SUITE"):
            n_processes = 4
        self._workers = self._start_workers(n_processes)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    # How often (in seconds) to emit a progress log line during training.
    _LOG_INTERVAL_SECS: float = 60.0

    def search(self):
        """Run the CFR training loop until time or signal stops it.

        Each iteration:

        1. Dispatches one ``cfr`` job per player into the worker pool.
           After warm-up each ``cfr`` job also folds in
           ``strategy_per_job`` average-strategy playthroughs for the
           same player, accumulated into the worker's persistent
           strategy delta.
        2. At sync barriers, drains the queue, broadcasts a ``sync``
           job so workers flush their accumulated regret *and* strategy
           deltas, then re-drains the queue.  The strategy pass is thus
           overlapped with CFR and needs no barrier of its own.
        3. At sync barriers that satisfy the discount schedule,
           applies an LCFR discount to the shared tables.
        4. At sync barriers that satisfy the checkpoint schedule,
           writes a checkpoint via the checkpoint manager.

        The loop exits when :attr:`max_runtime_hours` is reached or
        when the checkpoint manager's sigterm event is set.  After
        the loop exits, one final checkpoint is written so no work
        is lost.

        Progress is logged at most once every :attr:`_LOG_INTERVAL_SECS`
        seconds at sync barriers, showing elapsed time, estimated
        remaining time, and the current iteration count.  This produces
        structured, infrequent log lines that are easy to scan in HPC
        cluster log files without the noise of a live progress bar.
        """
        self._training_start = time.monotonic()
        max_runtime_secs = self._max_runtime_hours * 3600.0
        sigterm = self._checkpoint_manager.sigterm_event
        step = self._workers_per_player * self._batch_size
        t = self._start_t  # traversals-per-player completed so far
        _last_log_time = self._training_start
        try:
            while True:
                elapsed = time.monotonic() - self._training_start
                if elapsed >= max_runtime_secs:
                    log.info(
                        f"Time limit reached after {elapsed / 3600:.2f}h — "
                        f"{t} traversals-per-player completed"
                    )
                    break
                if sigterm.is_set():
                    break

                # Average-strategy playthroughs are folded into the cfr jobs
                # (``strat_batch`` per job, per player) so they overlap CFR and
                # flush with the regret delta at the next sync — no separate
                # strategy barrier.  Gated to 0 during warm-up by the same
                # ``should_update_strategy`` predicate the old barriered pass used.
                strat_batch = (
                    self._strategy_per_job
                    if should_update_strategy(
                        t // self._sync_interval,
                        self._strategy_interval,
                        self._update_threshold,
                    )
                    else 0
                )
                for i in range(self._n_players):
                    for _ in range(self._workers_per_player):
                        self._send_job(
                            "cfr", t=t, i=i, batch=self._batch_size,
                            strat_batch=strat_batch,
                        )
                t += step
                self._current_t = t

                if at_sync_barrier(t, self._sync_interval, step=step):
                    self._join_queue()
                    if sigterm.is_set():
                        break
                    self._broadcast_job("sync")
                    self._join_queue()
                    if sigterm.is_set():
                        break

                    sync_step = t // self._sync_interval

                    # Strategy updates are no longer a barrier here — they are
                    # folded into the cfr jobs above (``strat_batch``) and flushed
                    # with the regret delta by the ``sync`` broadcast, so the pool
                    # never stalls on a strategy-only phase.

                    if should_discount(sync_step, self._discount_interval):
                        self._discount_state.apply(self._tables, sync_step)

                    if (
                        sync_step >= self._checkpoint_start_cycles
                        and should_checkpoint(sync_step, self._checkpoint_interval)
                    ):
                        self._checkpoint_manager.checkpoint(t=t)

                    now = time.monotonic()
                    if now - _last_log_time >= self._LOG_INTERVAL_SECS:
                        elapsed = now - self._training_start
                        remaining = max_runtime_secs - elapsed
                        log.info(
                            f"[t={t}  sync_step={sync_step}]  "
                            f"elapsed={elapsed / 3600:.2f}h  "
                            f"remaining≈{max(remaining, 0) / 3600:.2f}h"
                        )
                        _last_log_time = now

            # Drain any jobs still in the queue, then write the final
            # checkpoint so the run can be resumed from its last iteration.
            self._join_queue()
            elapsed_total = (time.monotonic() - self._training_start) / 3600.0
            if sigterm.is_set():
                log.info("Signal received — writing final checkpoint before shutdown")
            else:
                log.info(
                    f"Training complete — {self._current_t} iters, "
                    f"{elapsed_total:.2f}h elapsed"
                )
            self._checkpoint_manager.checkpoint(t=self._current_t, wait=True)
            self._checkpoint_manager.shutdown()
        except WorkerError:
            log.error("A worker encountered a fatal error — terminating all workers")
            raise

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def terminate(self, safe: bool = True):
        """Shut down the worker pool and release shared resources.

        Two paths are supported:

        - **Safe shutdown** (default): drain the job queue, broadcast
          ``terminate`` to every worker, then join each worker with a
          timeout.  Workers that refuse to exit within the timeout
          are killed.  This is the normal exit path after
          :meth:`search` returns.
        - **Unsafe shutdown** (``safe=False``): kill every worker
          immediately without going through the queue.  Used when
          the queue may be deadlocked because a worker has already
          crashed and raised :class:`WorkerError`.

        In both cases the shared tables are closed and unlinked
        before the method returns.

        Parameters
        ----------
        safe : bool, optional
            Whether to attempt an orderly shutdown.  Defaults to
            ``True``.
        """
        SHUTDOWN_TIMEOUT_SECS = 60
        if not safe:
            # Emergency shutdown: kill all workers immediately without
            # waiting for the job queue, which may be deadlocked due
            # to a worker error.
            log.warning("Unsafe termination — killing all workers immediately")
            for worker in self._workers:
                if worker.is_alive():
                    worker.kill()
            for worker in self._workers:
                worker.join(timeout=SHUTDOWN_TIMEOUT_SECS)
                if worker.exitcode not in (0, None, -9):
                    log.warning(f"{worker.name} exited with code {worker.exitcode}")
            self._cleanup()
            return
        self._job_queue.join()
        self._broadcast_job("terminate")
        self._job_queue.join()
        for worker in self._workers:
            deadline = time.monotonic() + SHUTDOWN_TIMEOUT_SECS
            while worker.is_alive():
                self._drain_logging_queue(nowait=True)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.error(f"{worker.name} stuck after {SHUTDOWN_TIMEOUT_SECS}s — killing")
                    worker.kill()
                    break
                worker.join(timeout=min(0.5, remaining))
            if worker.exitcode not in (0, None):
                log.warning(f"{worker.name} exited with code {worker.exitcode}")
            else:
                log.info(f"worker {worker.name} joined.")
        self._drain_logging_queue(nowait=True)
        self._cleanup()

    def _cleanup(self):
        """Close and unlink shared tables after the worker pool exits."""
        self._checkpoint_manager.shutdown()
        self._tables.close()

    def _warn_index_cache_budget(self) -> None:
        """Log the shm index-cache footprint and warn if it is a large share
        of the node's memory budget.

        The caches plus the chunk mmaps share the node's RAM; an oversized
        capacity can OOM the run.  This surfaces the footprint at startup —
        *before* compute is committed — using ``SLURM_MEM_PER_NODE`` (MiB)
        when available.
        """
        total_bytes = self._tables.index_cache_total_bytes()
        mem_mb = os.environ.get("SLURM_MEM_PER_NODE")
        if mem_mb:
            frac = total_bytes / (float(mem_mb) * 1024 ** 2)
            msg = (
                f"Index caches use {total_bytes / 1024 ** 3:.2f} GiB "
                f"({frac:.0%} of the {float(mem_mb) / 1024:.1f} GiB node budget); "
                f"chunk mmaps need the rest."
            )
            if frac > 0.40:
                log.warning(
                    "%s — consider lowering PLURIBUS_INDEX_CAPACITY or raising "
                    "--mem so the chunk tables still fit.", msg
                )
            else:
                log.info(msg)
        else:
            log.info(
                "Index caches use %.2f GiB (set SLURM_MEM_PER_NODE for a "
                "budget check).", total_bytes / 1024 ** 3
            )

    def _start_workers(self, n_processes: int):
        """Construct and start *n_processes* worker processes.

        Each worker receives a reference to the shared tables, the
        job and logging queues, the training hyperparameters it
        needs, and the pre-loaded LUT so it does not have to repeat
        the deserialisation in the child process.
        """
        workers = []
        for _ in range(n_processes):
            worker = Worker(
                job_queue=self._job_queue,
                logging_queue=self._logging_queue,
                locks=self._locks,
                tables=self._tables,
                lut_path=self._lut_path,
                pickle_dir=self._pickle_dir,
                n_players=self._n_players,
                prune_threshold=self._prune_threshold,
                c=self._c,
                save_path=self._save_path,
                info_set_lut=self._info_set_lut,
                error_event=self._error_event,
                bias=self._bias,
                bias_magnitude=self._bias_magnitude,
            )
            workers.append(worker)
        # Prewarm the shm index caches from LMDB before the fork so every
        # worker inherits a warm, consistent cache (the mmap is shared, so
        # this happens exactly once).  Must precede close_envs() — it reads
        # each index's LMDB env.  No-op when the cache is disabled.
        self._tables.prewarm_caches()

        # Close every LMDB env in the parent immediately before
        # forking the workers.  python-lmdb (1.3) appears to hold
        # transaction state that survives env.close() / reopen in the
        # child — even with max_spare_txns=0 — and triggers
        # ``MDB_BAD_RSLOT`` on the first read txn after fork.  Forking
        # while the parent's envs are closed guarantees the child
        # inherits nothing, then each side reopens its own env.
        self._tables.close_envs()
        try:
            for worker in workers:
                worker.start()
                log.info(f"started worker {worker.name}")
        finally:
            self._tables.open_envs()
        return workers

    # ------------------------------------------------------------------
    # Checkpoint integration
    # ------------------------------------------------------------------

    def to_dict(self, t: Optional[int] = None) -> Dict[str, Union[str, float, int, None]]:
        """Serialise the server state into a JSON-compatible dict.

        The returned dict is persisted by the checkpoint manager as
        ``server_state.pkl``.  It contains every hyperparameter
        needed for the resume-time structural compatibility check plus
        the iteration counter and per-street chunk counts needed to
        locate the on-disk chunk files.

        Parameters
        ----------
        t : int, optional
            Iteration counter to embed in the snapshot.  When
            omitted, ``self._current_t`` is used (safe to call
            outside the training loop).

        Returns
        -------
        dict
            Flat dict with path-like values converted to absolute
            strings so the checkpoint is portable between CWDs.
        """
        from environment.poker_env import INFO_SET_ENCODING, action_grid_fingerprint

        t_val = t if t is not None else self._current_t
        config = dict(
            t=t_val,
            info_set_encoding=INFO_SET_ENCODING,
            action_grid_fingerprint=action_grid_fingerprint(self._n_players),
            strategy_interval=self._strategy_interval,
            max_runtime_hours=self._max_runtime_hours,
            discount_duration_cycles=self._discount_state.duration_cycles,
            discount_active=self._discount_state.active,
            prune_threshold=self._prune_threshold,
            c=self._c,
            n_players=self._n_players,
            update_threshold=self._update_threshold,
            save_path=self._save_path,
            lut_path=self._lut_path,
            pickle_dir=self._pickle_dir,
            sync_interval=self._sync_interval,
            discount_interval=self._discount_interval,
            checkpoint_interval=self._checkpoint_interval,
            # Not structural — snapshot-cadence gate, safe to change on resume.
            checkpoint_start_cycles=self._checkpoint_start_cycles,
            start_timestep=self._start_t,
            n_chunks_per_street=self._tables.n_chunks_per_street(),
            chunk_size=_CHUNK_SIZE,
            # Not structural — persisted so a resume rebuilds the same-size
            # shm index cache instead of auto-shrinking below this run's
            # capacity and overflowing as it keeps allocating.
            index_cache_capacity=self._tables.index_cache_capacities(),
        )
        return {
            k: os.path.abspath(str(v)) if isinstance(v, Path) else v
            for k, v in sorted(config.items())
        }

    def flush_all_workers(self) -> None:
        """Broadcast a ``sync`` job and wait for every worker to flush.

        Called by :meth:`CheckpointManager.checkpoint` before writing
        dirty chunks so no in-flight regret delta is left in a
        worker's local accumulator at the time the chunks are
        serialised.
        """
        self._broadcast_job("sync")
        self._join_queue()

    # ------------------------------------------------------------------
    # Queue plumbing
    # ------------------------------------------------------------------

    def _join_queue(self):
        """Block until the job queue drains, surfacing worker errors.

        Uses a dedicated daemon thread to drain the queue so the main
        thread can continue polling the shared ``_error_event``.  If
        a worker sets the event while the queue is being drained,
        this method raises :class:`WorkerError` immediately instead
        of blocking forever on a queue that will never empty.

        Raises
        ------
        WorkerError
            If a worker has signalled a fatal error during the wait.
        """
        t = threading.Thread(target=self._job_queue.join, daemon=True)
        t.start()
        while t.is_alive():
            if self._error_event.is_set():
                raise WorkerError("A worker encountered a fatal error")
            t.join(timeout=0.5)

    def _send_job(self, job_name: str, **kwargs):
        """Enqueue one job, retrying if the bounded queue is full.

        Retries until the put succeeds or the error event fires.  The
        bounded queue acts as back-pressure: if every worker is busy,
        this method blocks the server until a worker picks up a job,
        keeping the pipeline depth under control.

        Parameters
        ----------
        job_name : str
            Dispatch key read by :meth:`Worker.run`.
        **kwargs
            Keyword arguments forwarded to the worker method.

        Raises
        ------
        WorkerError
            If the error event fires before the put succeeds.
        """
        while True:
            if self._error_event.is_set():
                raise WorkerError("A worker encountered a fatal error")
            try:
                self._job_queue.put((job_name, kwargs), block=True, timeout=0.5)
                return
            except Exception:
                # Queue full — retry after checking the error event.
                pass

    def _broadcast_job(self, job_name: str, **kwargs):
        """Enqueue *job_name* once for every worker in the pool.

        Used for operations that every worker must perform exactly
        once — typically ``sync`` and ``terminate``.
        """
        for _ in self._workers:
            self._send_job(job_name, **kwargs)

    def _drain_logging_queue(self, nowait: bool = False) -> None:
        """Emit any pending worker log messages via the server logger.

        Workers may push human-readable status strings onto the shared
        logging queue for exceptional events; this helper drains them
        into the server's logger so they appear in the normal run log.
        During normal operation the queue is empty because per-flush
        messages are only emitted at ``DEBUG`` level from the workers.
        The method is still called during :meth:`terminate` to flush
        any late-arriving messages before the process exits.

        Parameters
        ----------
        nowait : bool, optional
            Use :meth:`Queue.get_nowait` instead of :meth:`Queue.get`.
            Defaults to ``False``.
        """
        while not self._logging_queue.empty():
            try:
                msg = (
                    self._logging_queue.get_nowait() if nowait
                    else self._logging_queue.get()
                )
            except Exception:
                return
            log.info(msg)
