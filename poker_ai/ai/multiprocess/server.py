import logging
import mmap as _mmap
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, Optional, Union

from poker_ai.ai.checkpoint import CheckpointManager

import enlighten

from poker_ai.ai.agent import Agent
from poker_ai import utils
from poker_ai.games.short_deck import state
from poker_ai.ai.multiprocess.worker import Worker

log = logging.getLogger("sync.server")


class Server:
    """Server class to manage all workers optimising CFR algorithm."""

    def __init__(
        self,
        strategy_interval: int,
        max_runtime_hours: float,
        discount_interval: int,
        discount_duration_iters: int,
        prune_threshold: int,
        c: int,
        n_players: int,
        update_threshold: int,
        save_path: Union[str, Path],
        lut_path: Union[str, Path] = ".",
        pickle_dir: bool = False,
        sync_interval: int = 10,
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
        self._discount_interval = discount_interval
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
        self._locks: Dict[str, mp.synchronize.Lock] = dict(
            strategy_update_lock=mp.Lock()
        )
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
                self._job_queue.join()
                if sigterm.is_set():
                    break
                self._broadcast_job("sync")
                self._job_queue.join()
                if sigterm.is_set():
                    break

                if t > self._update_threshold and t % self._strategy_interval == 0:
                    for i in range(self._n_players):
                        self._send_job("update_strategy", t=t, i=i)
                    self._job_queue.join()
                    if sigterm.is_set():
                        break

                if self._discount_window_active(t):
                    self._broadcast_job("discount", t=t)
                    self._job_queue.join()
                    if sigterm.is_set():
                        break

            if t % self._checkpoint_interval == 0:
                self._checkpoint_manager.checkpoint(t=t)

            progress_bar.update()
            t += 1

        # Drain any jobs still in the queue, then flush all worker local deltas
        # before writing the final checkpoint.
        self._job_queue.join()
        elapsed_total = (time.monotonic() - self._training_start) / 3600.0
        if sigterm.is_set():
            log.info("Signal received — writing final checkpoint before shutdown")
        else:
            log.info(
                f"Training complete — {self._current_t} iters, {elapsed_total:.2f}h elapsed"
            )
        self._checkpoint_manager.checkpoint(t=self._current_t)

    def terminate(self, safe: bool = True):
        """Broadcast terminate to all workers and join them."""
        SHUTDOWN_TIMEOUT_SECS = 60
        if safe:
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
            discount_interval=self._discount_interval,
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

    def _send_job(self, job_name: str, **kwargs):
        """Send job of type ``job_name`` with arguments to worker pool."""
        self._job_queue.put((job_name, kwargs), block=True)

    def _broadcast_job(self, job_name: str, **kwargs):
        """Send ``job_name`` to every worker in the pool."""
        for _ in self._workers:
            self._job_queue.put((job_name, kwargs), block=True)

    def flush_all_workers(self) -> None:
        """Broadcast a sync job and wait until all workers have flushed.

        Called by ``CheckpointManager.checkpoint()`` before writing dirty
        chunks so that no in-flight local deltas remain in worker buffers.
        """
        self._broadcast_job("sync")
        self._job_queue.join()

    def _discount_window_active(self, t: int) -> bool:
        """Return True if a discount broadcast should fire at iteration t."""
        if not self._discounting_active:
            return False
        if t >= self._discount_duration_iters:
            log.info(f"Discount window closed after {t} iters")
            self._broadcast_job("sync")
            self._job_queue.join()
            self._discounting_active = False
            return False
        return t % self._discount_interval == 0

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
                discount_interval=self._discount_interval,
                save_path=self._save_path,
                info_set_lut=self._info_set_lut,
            )
            workers.append(worker)
        for worker in workers:
            worker.start()
            log.info(f"started worker {worker.name}")
        return workers
