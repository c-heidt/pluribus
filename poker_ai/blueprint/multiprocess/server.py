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
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Union

from poker_ai.tables.checkpoint import CheckpointManager
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.tables.chunk_store import CHUNK_SIZE as _CHUNK_SIZE
from poker_ai.blueprint.multiprocess.worker import Worker
from poker_ai.blueprint.training import (
    DiscountState,
    at_sync_barrier,
    should_checkpoint,
    should_discount,
    should_update_strategy,
)
from information_abstraction import load_info_set_lut

log = logging.getLogger("sync.server")


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
        start_timestep: int = 0,
        n_processes: Optional[int] = None,
        batch_size: Optional[int] = None,
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
            Period (in sync cycles) between checkpoint writes.
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
        """
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
        self._start_t = start_timestep
        self._discount_state = DiscountState(
            duration_cycles=discount_duration_cycles,
            discount_interval=discount_interval,
        )

        # Load the LUT once in the parent; workers inherit the
        # deserialised object via fork copy-on-write, avoiding one
        # load per worker.
        self._info_set_lut = load_info_set_lut(lut_path, pickle_dir)

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
        self._tables = CFRTables(
            index_path=self._save_path / "lmdb_index",
            shm_dir=shm_dir,
            lmdb_map_size=lmdb_map_size,
            actions_per_street=MAX_ACTIONS_PER_STREET,
        )
        self._locks: Dict[str, mp.synchronize.Lock] = {}
        self._error_event: mp.Event = mp.Event()  # type: ignore
        self._current_t: int = self._start_t

        # CheckpointManager registers signal handlers and restores
        # from an existing checkpoint before any worker is spawned so
        # workers observe the restored state.
        self._checkpoint_manager = CheckpointManager(self, self._save_path)
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
        2. At sync barriers, drains the queue, broadcasts a ``sync``
           job so workers flush their accumulated deltas, then
           re-drains the queue.
        3. At sync barriers that satisfy the strategy-interval
           schedule, dispatches one ``update_strategy`` job per
           player and waits for them all to complete.
        4. At sync barriers that satisfy the discount schedule,
           applies an LCFR discount to the shared tables.
        5. At sync barriers that satisfy the checkpoint schedule,
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

                for i in range(self._n_players):
                    for _ in range(self._workers_per_player):
                        self._send_job(
                            "cfr", t=t, i=i, batch=self._batch_size
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

                    if should_update_strategy(
                        sync_step, self._strategy_interval, self._update_threshold
                    ):
                        for i in range(self._n_players):
                            self._send_job("update_strategy", i=i)
                        self._join_queue()
                        if sigterm.is_set():
                            break

                    if should_discount(sync_step, self._discount_interval):
                        self._discount_state.apply(self._tables, sync_step)

                    if should_checkpoint(sync_step, self._checkpoint_interval):
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
            )
            workers.append(worker)
        for worker in workers:
            worker.start()
            log.info(f"started worker {worker.name}")
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
        t_val = t if t is not None else self._current_t
        config = dict(
            t=t_val,
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
            start_timestep=self._start_t,
            n_chunks_per_street=self._tables.n_chunks_per_street(),
            chunk_size=_CHUNK_SIZE,
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
