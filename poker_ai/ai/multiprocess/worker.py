import copy
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Union

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
        status_queue: mp.Queue,
        logging_queue: mp.Queue,
        locks: Dict[str, mp.synchronize.Lock],
        agent: Agent,
        info_set_lut: state.InfoSetLookupTable,
        n_players: int,
        prune_threshold: int,
        c: int,
        lcfr_threshold: int,
        discount_interval: int,
        update_threshold: int,
        dump_iteration: int,
        save_path: Path,
    ):
        """Construct the process, setup the state."""
        super().__init__(group=None, name=None, args=(), kwargs={}, daemon=None)
        self._job_queue: mp.Queue = job_queue
        self._status_queue: mp.Queue = status_queue
        self._logging_queue: mp.Queue = logging_queue
        self._locks = locks
        self._n_players = n_players
        self._prune_threshold = prune_threshold
        self._agent = agent
        self._c = c
        self._lcfr_threshold = lcfr_threshold
        self._discount_interval = discount_interval
        self._update_threshold = update_threshold
        self._dump_iteration = dump_iteration
        self._save_path = Path(save_path)
        self._info_set_lut: state.InfoSetLookupTable = info_set_lut
        self._setup_new_game()

    def run(self):
        """Compute the next job the server sent."""
        # Seed process.
        self._set_seed()
        # Start processing loop, that will block on a wait for the next job
        # that will be sent from the server to be consumed by the worker(s).
        while True:
            # Get the name of the method and the key word arguments needed for
            # the method.
            self._update_status("idle")
            name, kwargs = self._job_queue.get(block=True)
            if name == "terminate":
                break
            elif name == "cfr":
                function = self._cfr
            elif name == "discount":
                function = self._discount
            elif name == "update_strategy":
                function = self._update_strategy
            elif name == "serialise":
                function = self._serialise
            else:
                raise ValueError(f"Unrecognised function name: {name}")
            self._update_status(name)
            function(**kwargs)
            # Notify the job queue that the task is done.
            self._job_queue.task_done()

    def _set_seed(self):
        """Lose all reproducability as we need unique streams per worker."""
        # NOTE(fedden): NumPy in particular has a problem with processes and
        #               seeds: https://github.com/numpy/numpy/issues/9650
        random_seed: int = int.from_bytes(os.urandom(4), byteorder="little")
        utils.random.seed(random_seed)

    def _cfr(self, t, i):
        """Search over random game and calculate the strategy."""
        self._setup_new_game()
        local_delta: Dict[str, Dict[str, float]] = {}
        use_pruning: bool = np.random.uniform() < 0.95
        pruning_allowed: bool = t > self._prune_threshold
        if pruning_allowed and use_pruning:
            ai.cfrp(self._agent, self._state, i, t, self._c, local_delta)
        else:
            ai.cfr(self._agent, self._state, i, t, local_delta)
        # Merge the local regret delta into the shared regret table under lock.
        # One lock acquisition per cfr() call replaces the old per-infoset
        # lock pattern, keeping cfr() traversal completely lock-free.
        self._locks["regret"].acquire()
        ai.merge_local_delta(self._agent, local_delta)
        self._locks["regret"].release()

    def _discount(self, t):
        """Discount previous regrets and strategy."""
        # TODO(fedden): Is discount_interval actually set/managed in
        #               minutes here? In Algorithm 1 this should be managed
        #               in minutes using perhaps the time module, but here
        #               it appears to be being managed by the iterations
        #               count.
        discount_factor = (t / self._discount_interval) / (
            (t / self._discount_interval) + 1
        )
        self._logging_queue.put(
            f"[t={t}] Discounting regrets and strategy (factor={discount_factor:.4f})",
            block=True,
        )
        # Per Pluribus paper only regret is discounted (Bug 2 fix).
        # Discounting strategy degrades convergence and is incorrect.
        #
        # IMPORTANT: agent.regret is a manager.dict() proxy.  Accessing
        # agent.regret[info_set] returns a *copy* of the nested dict, not a
        # proxy into it.  Mutating that copy in-place (e.g.
        # regret[info_set][action] *= d) is a silent lost update — the change
        # is never written back to the shared manager process.  We must
        # explicitly read the row, mutate a local copy, then write it back.
        self._locks["regret"].acquire()
        for info_set in list(self._agent.regret.keys()):
            row = dict(self._agent.regret[info_set])
            for action in row:
                row[action] *= discount_factor
            self._agent.regret[info_set] = row
        self._locks["regret"].release()

    def _update_strategy(self, t, i):
        """Update the strategy."""
        # Deal a fresh hand so each strategy update traverses a different game
        # path, giving more diverse strategy samples.
        self._setup_new_game()
        # Acquire the pre_flop_strategy lock for the full traversal so that
        # concurrent workers performing strategy updates don't race on the
        # same infoset's read-modify-write.  The lock is *not* acquired inside
        # ai.update_strategy() (that function is recursive and must not try to
        # re-acquire a non-reentrant lock); it is acquired here at the single
        # entry point for the whole traversal.
        self._locks["pre_flop_strategy"].acquire()
        ai.update_strategy(self._agent, self._state, i, t)
        self._locks["pre_flop_strategy"].release()

    def _serialise(self, t: int, server_state: Dict[str, Union[str, float, int, None]]):
        """Write progress of optimising agent (and server state) to file."""
        self._logging_queue.put(
            f"[t={t}] Saving checkpoint to {self._save_path}", block=True
        )
        ai.serialise(
            agent=self._agent,
            save_path=self._save_path,
            t=t,
            server_state=server_state,
            locks=self._locks,
        )
        n_info_sets = len(self._agent.regret)
        self._logging_queue.put(
            f"[t={t}] Checkpoint saved — {n_info_sets:,} info sets in regret table",
            block=True,
        )

    def _update_status(self, status):
        """Update the status of this worker by posting it to the server."""
        self._status_queue.put((self.name, status), block=True)

    def _setup_new_game(self):
        """Setup up new poker game."""
        self._state: state.ShortDeckPokerState = state.new_game(
            self._n_players, self._info_set_lut,
        )
