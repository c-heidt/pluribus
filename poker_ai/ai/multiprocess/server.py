"""Training server for multi-process CFR.

Owns the worker pool, the shared job queue, signal handling, the
checkpoint manager, and progress tracking.  All training-step logic
(CFR traversals, discount formula, schedule predicates, LUT loading)
lives in :mod:`poker_ai.ai.training`; this class is a thin orchestrator
around those primitives.
"""
import logging
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Union

import enlighten

from poker_ai.ai.checkpoint import CheckpointManager
from poker_ai.ai.cfr_tables import CFRTables
from poker_ai.ai.multiprocess.worker import Worker
from poker_ai.ai.training import (
    DiscountState,
    at_sync_barrier,
    load_info_set_lut,
    should_checkpoint,
    should_discount,
    should_update_strategy,
)

log = logging.getLogger("sync.server")


class WorkerError(RuntimeError):
    """Raised when a worker process encounters a fatal error."""
    pass


class Server:
    """Coordinates a pool of :class:`Worker` processes running CFR."""

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
        start_timestep: int = 1,
        n_processes: Optional[int] = None,
    ):
        """Set up the optimisation server.

        All interval/threshold parameters (``strategy_interval``,
        ``discount_interval``, ``checkpoint_interval``,
        ``discount_duration_cycles``, ``update_threshold``) are counted
        in **sync cycles** (= ``sync_interval`` iterations).  The only
        exception is ``prune_threshold``, which is checked per CFR call
        and therefore stays in raw iterations.
        """
        if n_processes is None:
            slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
            if slurm_cpus is not None:
                n_processes = int(slurm_cpus) - 1
                log.info(f"Using {n_processes} processes (SLURM_CPUS_PER_TASK={slurm_cpus})")
            else:
                n_processes = mp.cpu_count() - 1
                log.info(f"Using {n_processes} processes (cpu_count={mp.cpu_count()})")

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

        # Load LUT once in the parent; workers inherit via fork copy-on-write.
        self._info_set_lut = load_info_set_lut(lut_path, pickle_dir)

        self._job_queue: mp.JoinableQueue = mp.JoinableQueue(maxsize=n_processes)
        self._logging_queue: mp.Queue = mp.Queue()

        # Per-street shared-memory tables.
        shm_dir = os.environ.get("PLURIBUS_SHM_DIR", "/dev/shm")
        from poker_ai.ai.index import lmdb_map_size_for_players
        from poker_ai.ai.action_space import MAX_ACTIONS_PER_STREET
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

        # CheckpointManager registers signal handlers and restores from
        # checkpoint before workers are spawned.
        self._checkpoint_manager = CheckpointManager(self, self._save_path)
        if os.environ.get("TESTING_SUITE"):
            n_processes = 4
        self._workers = self._start_workers(n_processes)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def search(self):
        """Run the MCCFR training loop until time or signal stops it."""
        self._training_start = time.monotonic()
        progress_bar = enlighten.get_manager().counter(
            desc="Optimisation iterations", unit="iter"
        )
        sigterm = self._checkpoint_manager.sigterm_event
        t = self._start_t
        try:
            while True:
                elapsed_hours = (time.monotonic() - self._training_start) / 3600.0
                if elapsed_hours >= self._max_runtime_hours:
                    log.info(
                        f"Time limit reached after {elapsed_hours:.2f}h — {t - 1} iterations"
                    )
                    break
                if sigterm.is_set():
                    break

                self._current_t = t
                self._drain_logging_queue()

                for i in range(self._n_players):
                    self._send_job("cfr", t=t, i=i)

                if at_sync_barrier(t, self._sync_interval):
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

                progress_bar.update()
                t += 1

            # Drain any jobs still in the queue, then write the final checkpoint.
            self._join_queue()
            elapsed_total = (time.monotonic() - self._training_start) / 3600.0
            if sigterm.is_set():
                log.info("Signal received — writing final checkpoint before shutdown")
            else:
                log.info(
                    f"Training complete — {self._current_t} iters, {elapsed_total:.2f}h elapsed"
                )
            self._checkpoint_manager.checkpoint(t=self._current_t)
        except WorkerError:
            log.error("A worker encountered a fatal error — terminating all workers")
            raise

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def terminate(self, safe: bool = True):
        """Broadcast terminate to all workers and join them."""
        SHUTDOWN_TIMEOUT_SECS = 60
        if not safe:
            # Emergency shutdown: kill all workers immediately without waiting
            # for the job queue (it may be deadlocked due to the worker error).
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
        """Close and unlink all shared-memory tables and indexes."""
        self._tables.close()

    def _start_workers(self, n_processes: int):
        """Spawn *n_processes* worker processes."""
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
        """Serialise the server state for checkpointing.

        Parameters
        ----------
        t:
            Current training iteration to embed in the snapshot.  When
            omitted, ``self._current_t`` is used (safe to call outside
            the training loop).
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
        )
        return {
            k: os.path.abspath(str(v)) if isinstance(v, Path) else v
            for k, v in sorted(config.items())
        }

    def flush_all_workers(self) -> None:
        """Broadcast a sync job and wait until all workers have flushed.

        Called by :meth:`CheckpointManager.checkpoint` before writing
        dirty chunks so that no in-flight local deltas remain in worker
        buffers.
        """
        self._broadcast_job("sync")
        self._join_queue()

    # ------------------------------------------------------------------
    # Queue plumbing
    # ------------------------------------------------------------------

    def _join_queue(self):
        """Block until the job queue drains, raising WorkerError if a worker dies."""
        t = threading.Thread(target=self._job_queue.join, daemon=True)
        t.start()
        while t.is_alive():
            if self._error_event.is_set():
                raise WorkerError("A worker encountered a fatal error")
            t.join(timeout=0.5)

    def _send_job(self, job_name: str, **kwargs):
        """Send a single job of type *job_name* to the worker pool."""
        while True:
            if self._error_event.is_set():
                raise WorkerError("A worker encountered a fatal error")
            try:
                self._job_queue.put((job_name, kwargs), block=True, timeout=0.5)
                return
            except Exception:
                # Queue full — retry after checking error event.
                pass

    def _broadcast_job(self, job_name: str, **kwargs):
        """Send *job_name* to every worker in the pool (once each)."""
        for _ in self._workers:
            self._send_job(job_name, **kwargs)

    def _drain_logging_queue(self, nowait: bool = False) -> None:
        """Emit any pending worker log messages via the server logger."""
        while not self._logging_queue.empty():
            try:
                msg = (
                    self._logging_queue.get_nowait() if nowait
                    else self._logging_queue.get()
                )
            except Exception:
                return
            log.info(msg)
