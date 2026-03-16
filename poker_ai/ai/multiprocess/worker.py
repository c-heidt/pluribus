import logging
import mmap as _mmap
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Tuple, Union

import joblib
import numpy as np

from poker_ai.ai import ai
from poker_ai.ai.agent import Agent
from poker_ai import utils
from poker_ai.games.short_deck import state

log = logging.getLogger("sync.worker")


class Worker(mp.Process):
    """Subclass of multiprocessing Process to handle agent optimisation."""

    def __init__(
        self,
        job_queue: mp.Queue,
        logging_queue: mp.Queue,
        locks: Dict[str, mp.synchronize.Lock],
        agent: Agent,
        lut_path: Union[str, Path],
        pickle_dir: bool,
        n_players: int,
        prune_threshold: int,
        c: int,
        discount_interval: int,
        save_path: Path,
        info_set_lut=None,
    ):
        """Construct the process, setup the state."""
        super().__init__(group=None, name=None, args=(), kwargs={}, daemon=None)
        self._job_queue: mp.Queue = job_queue
        self._logging_queue: mp.Queue = logging_queue
        self._locks = locks
        self._n_players = n_players
        self._prune_threshold = prune_threshold
        self._agent = agent
        self._c = c
        self._discount_interval = discount_interval
        self._save_path = Path(save_path)
        self._lut_path = str(lut_path)
        self._pickle_dir = pickle_dir
        if info_set_lut is not None:
            self._info_set_lut = info_set_lut
        # Per-traversal regret accumulator keyed by (betting_round, info_set).
        # Values are int64 delta arrays of length MAX_ACTIONS_PER_STREET[r].
        self._local_delta: Dict[Tuple[int, str], np.ndarray] = {}
        self._local_iteration_count: int = 0

    def run(self):
        """Load the LUT, seed RNG, then process jobs dispatched by the server."""
        # Reopen the LMDB environment after fork so this process gets its own
        # reader lock-table slot (avoids MDB_BAD_RSLOT).
        self._agent._index.reopen_after_fork()
        if not hasattr(self, "_info_set_lut"):
            if not self._pickle_dir:
                lut_file_path = os.path.join(self._lut_path, "card_info_lut.joblib")
                lut_file = open(lut_file_path, "rb")
                lut_mmap = _mmap.mmap(lut_file.fileno(), 0, access=_mmap.ACCESS_READ)
                self._info_set_lut = joblib.load(lut_mmap)
                lut_mmap.close()
                lut_file.close()
            else:
                self._info_set_lut = utils.io.load_info_set_lut(
                    self._lut_path, self._pickle_dir
                )
        self._set_seed()
        self._setup_new_game()
        while True:
            name, kwargs = self._job_queue.get(block=True)
            if name == "terminate":
                self._sync_to_master()
                self._job_queue.task_done()
                break
            elif name == "cfr":
                function = self._cfr
            elif name == "sync":
                function = self._sync_to_master
            elif name == "discount":
                function = self._discount
            elif name == "update_strategy":
                function = self._update_strategy
            else:
                raise ValueError(f"Unrecognised function name: {name}")
            function(**kwargs)
            self._job_queue.task_done()

    def _set_seed(self):
        """Lose all reproducability as we need unique streams per worker."""
        # NOTE(fedden): NumPy in particular has a problem with processes and
        #               seeds: https://github.com/numpy/numpy/issues/9650
        random_seed: int = int.from_bytes(os.urandom(4), byteorder="little")
        utils.random.seed(random_seed)

    def _sync_to_master(self) -> None:
        """Flush ``_local_delta`` into the agent's per-street regret tables.

        Routes each ``(betting_round, info_set)`` delta into the correct
        ``SparseRegretTable`` via ``merge_delta_row`` which holds the stripe
        lock internally.  After the flush ``_local_delta`` is cleared.
        """
        if not self._local_delta:
            return
        n_infosets = len(self._local_delta)
        ai.merge_local_delta(self._agent, self._local_delta)
        self._local_delta.clear()
        self._logging_queue.put(
            f"[worker={self.name}] Synced {n_infosets:,} infosets to master",
            block=True,
        )

    def _cfr(self, t, i):
        """Search over random game and calculate the strategy."""
        self._setup_new_game()
        use_pruning: bool = np.random.uniform() < 0.95
        pruning_allowed: bool = t > self._prune_threshold
        if pruning_allowed and use_pruning:
            ai.cfrp(self._agent, self._state, i, t, self._c, self._local_delta)
        else:
            ai.cfr(self._agent, self._state, i, t, self._local_delta)
        self._local_iteration_count += 1
        # Delta is flushed on explicit "sync" jobs dispatched by the server,
        # not after every traversal (Phase 5 decoupling).

    def _discount(self, t):
        """Apply LCFR discount to all regret and strategy tables."""
        discount_factor = (t / self._discount_interval) / (
            (t / self._discount_interval) + 1
        )
        self._logging_queue.put(
            f"[t={t}] Discounting regrets and strategy (factor={discount_factor:.4f})",
            block=True,
        )
        for r in range(4):
            self._agent.regret_tables[r].set_sync_boundary(True)
            self._agent.regret_tables[r].apply_discount(discount_factor)
            self._agent.regret_tables[r].set_sync_boundary(False)
            self._agent.strategy_tables[r].set_sync_boundary(True)
            self._agent.strategy_tables[r].apply_discount(discount_factor)
            self._agent.strategy_tables[r].set_sync_boundary(False)

    def _update_strategy(self, t, i):
        """Update strategy visit counts for all streets."""
        self._setup_new_game()
        self._locks["strategy_update_lock"].acquire()
        ai.update_strategy(self._agent, self._state, i, t)
        self._locks["strategy_update_lock"].release()

    def _setup_new_game(self):
        """Setup up new poker game."""
        self._state: state.ShortDeckPokerState = state.new_game(
            self._n_players, self._info_set_lut,
        )
