import logging
import mmap as _mmap
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Union

from poker_ai.ai.checkpoint import CheckpointManager

import enlighten

from poker_ai.ai.agent import Agent
from poker_ai import utils
from poker_ai.ai.multiprocess.worker import Worker

log = logging.getLogger("sync.server")


class WorkerError(RuntimeError):
    """Raised when a worker process encounters a fatal error."""
    pass


class Server:
    """Server class to manage all workers optimising CFR algorithm."""

    def __init__(
        self,
        strategy_interval: int,
        max_runtime_hours: float,
        discount_duration_iters: int,
        prune_threshold: int,
        c: int,
        n_players: int,
        update_threshold: int,
        save_path: Union[str, Path],
        lut_path: Union[str, Path] = ".",
        pickle_dir: bool = False,
        sync_interval: int = 10,
        discount_interval: int = 1,
        checkpoint_interval: int = 1000,
        start_timestep: int = 1,
        n_processes: Optional[int] = None,
    ):
        """Set up the optimisation server."""
        # Determine number of processes to use
        if n_processes is None:
            # Check if running under Slurm
            slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
            if slurm_cpus is not None:
                n_processes = int(slurm_cpus) - 1
                log.info(f"Using {n_processes} processes (SLURM_CPUS_PER_TASK={slurm_cpus})")
            else:
                n_processes = mp.cpu_count() - 1
                log.info(f"Using {n_processes} processes (cpu_count={mp.cpu_count()})")

        self._strategy_interval = strategy_interval
        self._max_runtime_hours = max_runtime_hours
        self._discount_duration_iters = discount_duration_iters
        self._discounting_active = True
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
        self._info_set_lut = self._load_lut(lut_path, pickle_dir)
        self._job_queue: mp.JoinableQueue = mp.JoinableQueue(maxsize=n_processes)
        self._logging_queue: mp.Queue = mp.Queue()
        # Agent with per-street shared-memory tables
        shm_dir = os.environ.get("PLURIBUS_SHM_DIR", "/dev/shm")
        from poker_ai.ai.index import lmdb_map_size_for_players
        lmdb_map_size = int(
            os.environ.get(
                "PLURIBUS_LMDB_MAP_SIZE",
                lmdb_map_size_for_players(n_players),
            )
        )
        log.info(
            f"LMDB map_size={lmdb_map_size // 1024**3} GiB for {n_players} players "
            f"(sparse file — check real usage with: du -sh <save_path>/lmdb_index/data.mdb)"
        )
        self._agent = Agent(
            index_path=self._save_path / "lmdb_index",
            shm_dir=shm_dir,
            lmdb_map_size=lmdb_map_size,
        )
        self._locks: Dict[str, mp.synchronize.Lock] = {}
        self._error_event: mp.Event = mp.Event() # type: ignore
        self._current_t: int = self._start_t
        # CheckpointManager registers signal handlers and restores from
        # checkpoint before workers are spawned.
        self._checkpoint_manager = CheckpointManager(self, self._save_path)
        if os.environ.get("TESTING_SUITE"):
            n_processes = 4
        self._workers = self._start_workers(n_processes)

    def search(self):
        """Perform MCCFR and train the agent."""
        self._training_start = time.monotonic()
        progress_bar_manager = enlighten.get_manager()
        progress_bar = progress_bar_manager.counter(
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
                while not self._logging_queue.empty():
                    log.info(self._logging_queue.get())
                for i in range(self._n_players):
                    self._send_job("cfr", t=t, i=i)

                if t % self._sync_interval == 0:
                    self._join_queue()
                    if sigterm.is_set():
                        break
                    self._broadcast_job("sync")
                    self._join_queue()
                    if sigterm.is_set():
                        break

                    if t > self._update_threshold and t % self._strategy_interval == 0:
                        for i in range(self._n_players):
                            self._send_job("update_strategy", t=t, i=i)
                        self._join_queue()
                        if sigterm.is_set():
                            break

                    sync_step = t // self._sync_interval
                    if sync_step % self._discount_interval == 0:
                        self._apply_discount(t)

                if t % self._checkpoint_interval == 0:
                    self._checkpoint_manager.checkpoint(t=t)

                progress_bar.update()
                t += 1

            # Drain any jobs still in the queue, then flush all worker local deltas
            # before writing the final checkpoint.
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
                while not self._logging_queue.empty():
                    try:
                        log.info(self._logging_queue.get_nowait())
                    except Exception:
                        pass
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
        # Drain any remaining log messages after all workers have exited.
        while not self._logging_queue.empty():
            try:
                log.info(self._logging_queue.get_nowait())
            except Exception:
                pass
        self._cleanup()

    def _cleanup(self):
        """Close and unlink all shared-memory agent tables."""
        for r in range(4):
            self._agent.regret_tables[r].close()
            self._agent.strategy_tables[r].close()
            self._agent.regret_tables[r].unlink_all()
            self._agent.strategy_tables[r].unlink_all()
        self._agent._index.close()

    def to_dict(self, t: Optional[int] = None) -> Dict[str, Union[str, float, int, None]]:
        """Serialise the server object to save the progress of optimisation.

        Parameters
        ----------
        t:
            Current training iteration to embed in the snapshot.  When
            omitted, ``self._current_t`` is used (safe to call outside the
            training loop for testing).
        """
        t_val = t if t is not None else self._current_t
        config = dict(
            t=t_val,
            strategy_interval=self._strategy_interval,
            max_runtime_hours=self._max_runtime_hours,
            discount_duration_iters=self._discount_duration_iters,
            discount_active=self._discounting_active,
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
            n_chunks_per_street={
                r: self._agent.regret_tables[r].n_chunks for r in range(4)
            },
            n_strategy_chunks_per_street={
                r: self._agent.strategy_tables[r].n_chunks for r in range(4)
            },
        )
        return {
            k: os.path.abspath(str(v)) if isinstance(v, Path) else v
            for k, v in sorted(config.items())
        }

    @staticmethod
    def from_dict(config):
        """Load serialised server and return instance."""
        return Server(**config)

    def _join_queue(self):
        """Block until the job queue drains, raising WorkerError if a worker dies."""
        t = threading.Thread(target=self._job_queue.join, daemon=True)
        t.start()
        while t.is_alive():
            if self._error_event.is_set():
                raise WorkerError("A worker encountered a fatal error")
            t.join(timeout=0.5)

    def _send_job(self, job_name: str, **kwargs):
        """Send job of type ``job_name`` with arguments to worker pool."""
        while True:
            if self._error_event.is_set():
                raise WorkerError("A worker encountered a fatal error")
            try:
                self._job_queue.put((job_name, kwargs), block=True, timeout=0.5)
                return
            except Exception:
                # Queue full — retry after checking error event
                pass

    def _broadcast_job(self, job_name: str, **kwargs):
        """Send ``job_name`` to every worker in the pool."""
        for _ in self._workers:
            self._send_job(job_name, **kwargs)

    def flush_all_workers(self) -> None:
        """Broadcast a sync job and wait until all workers have flushed.

        Called by ``CheckpointManager.checkpoint()`` before writing dirty
        chunks so that no in-flight local deltas remain in worker buffers.
        """
        self._broadcast_job("sync")
        self._join_queue()

    def _apply_discount(self, t: int) -> None:
        """Apply LCFR discount directly to all shared-memory tables.

        Called by the server after every sync barrier while the discount window
        is active.  Running in the server process (not a worker job) guarantees
        the discount is applied exactly once per sync with no concurrency.
        """
        if not self._discounting_active:
            return
        if t >= self._discount_duration_iters:
            log.info(f"Discount window closed after {t} iters")
            self._discounting_active = False
            return
        from poker_ai.ai.index import CHUNK_SIZE
        discount_step = t // (self._sync_interval * self._discount_interval)
        discount_factor = discount_step / (discount_step + 1)
        log.info(
            f"[t={t}] Discounting regrets and strategy "
            f"(step={discount_step}, factor={discount_factor:.4f})"
        )
        for r in range(4):
            for table in (
                self._agent.regret_tables[r],
                self._agent.strategy_tables[r],
            ):
                n = table.n_allocated
                if n > 0:
                    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
                    for chunk_id in range(n_chunks):
                        table._ensure_chunk(chunk_id)
                table.set_sync_boundary(True)
                table.apply_discount(discount_factor)
                table.set_sync_boundary(False)


    def _load_lut(self, lut_path, pickle_dir):
        """Load LUT once in the parent process.

        Workers inherit the deserialized Python object via fork copy-on-write.
        Only one deserialization happens regardless of worker count.
        """
        if pickle_dir:
            return utils.io.load_info_set_lut(str(lut_path), pickle_dir)
        lut_file_path = os.path.join(str(lut_path), "card_info_lut.joblib")
        log.info(f"Loading LUT from {lut_file_path} ...")
        import joblib as _joblib
        lut = _joblib.load(lut_file_path)
        log.info("LUT loaded.")
        return lut

    def _start_workers(self, n_processes: int):
        """Begin the worker processes."""
        workers = []
        for _ in range(n_processes):
            worker = Worker(
                job_queue=self._job_queue,
                logging_queue=self._logging_queue,
                locks=self._locks,
                agent=self._agent,
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
