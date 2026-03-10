import logging
import mmap as _mmap
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, Optional, Union

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
        n_iterations: int,
        lcfr_threshold: int,
        discount_interval: int,
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
        self._n_iterations = n_iterations
        self._lcfr_threshold = lcfr_threshold
        self._discount_interval = discount_interval
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
        self._agent = Agent(
            index_path=self._save_path / "lmdb_index",
            shm_dir=shm_dir,
        )
        self._locks: Dict[str, mp.synchronize.Lock] = dict(
            strategy_update_lock=mp.Lock()
        )
        self._maybe_resume(self._save_path)
        if os.environ.get("TESTING_SUITE"):
            n_processes = 4
        self._workers = self._start_workers(n_processes)

    def search(self):
        """Perform MCCFR and train the agent."""
        self._training_start = time.monotonic()
        progress_bar_manager = enlighten.get_manager()
        progress_bar = progress_bar_manager.counter(
            total=self._n_iterations, desc="Optimisation iterations", unit="iter"
        )
        for t in range(self._start_t, self._n_iterations + 1):
            while not self._logging_queue.empty():
                log.info(self._logging_queue.get())
            for i in range(self._n_players):
                self._send_job("cfr", t=t, i=i)

            if t % self._sync_interval == 0:
                self._job_queue.join()
                self._broadcast_job("sync")
                self._job_queue.join()

                if t > self._update_threshold and t % self._strategy_interval == 0:
                    for i in range(self._n_players):
                        self._send_job("update_strategy", t=t, i=i)
                    self._job_queue.join()

                # Discount window stub — always False until Phase 7
                if self._discount_window_active(t):
                    self._broadcast_job("discount", t=t)
                    self._job_queue.join()

            if t % self._checkpoint_interval == 0:
                log.info(f"[t={t}] Checkpoint stub — Phase 6 will implement full write")

            progress_bar.update()

    def terminate(self, safe: bool = True):
        """Broadcast terminate to all workers and join them."""
        if safe:
            self._job_queue.join()
        self._broadcast_job("terminate")
        self._job_queue.join()
        for worker in self._workers:
            while worker.is_alive():
                while not self._logging_queue.empty():
                    try:
                        log.info(self._logging_queue.get_nowait())
                    except Exception:
                        pass
                worker.join(timeout=0.5)
            log.info(f"worker {worker.name} joined.")
        for r in range(4):
            self._agent.regret_tables[r].close()
            self._agent.strategy_tables[r].close()
            self._agent.regret_tables[r].unlink_all()
            self._agent.strategy_tables[r].unlink_all()
        self._agent._index.close()

    def to_dict(self) -> Dict[str, Union[str, float, int, None]]:
        """Serialise the server object to save the progress of optimisation."""
        config = dict(
            strategy_interval=self._strategy_interval,
            n_iterations=self._n_iterations,
            lcfr_threshold=self._lcfr_threshold,
            discount_interval=self._discount_interval,
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

    def _maybe_resume(self, save_path: Path) -> None:
        """Stub — full resume logic wired in Phase 6."""
        if (save_path / "server_state.pkl").exists():
            log.info("Checkpoint found — resume wired in Phase 6")
        else:
            log.info("No checkpoint — starting fresh")

    def _discount_window_active(self, t: int) -> bool:
        """Stub — discount window logic implemented in Phase 7."""
        return False

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
