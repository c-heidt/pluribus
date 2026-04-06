"""Worker process for multi-process CFR training.

Each :class:`Worker` is a long-lived ``multiprocessing.Process`` that
consumes jobs from a shared queue dispatched by the
:class:`poker_ai.ai.multiprocess.server.Server`.  All real training
logic lives in :mod:`poker_ai.ai.training`; this class is a thin
dispatch loop around those primitives.

Worker-specific concerns that stay here:

- Post-fork LMDB reopen (reader slots must not be shared across forks).
- Post-fork LUT attach (``_info_set_lut`` may have been inherited via
  copy-on-write or must be loaded from disk).
- Per-process RNG seeding (each worker needs an independent stream).
- The persistent ``_local_delta`` accumulator that batches regret
  updates across many CFR calls before flushing at a sync barrier.
"""
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from poker_ai import utils
from poker_ai.ai.cfr import merge_local_delta
from poker_ai.ai.cfr_tables import CFRTables
from poker_ai.ai.training import (
    cfr_step,
    load_info_set_lut,
    strategy_step,
)
from poker_ai.environment import poker_env as state

log = logging.getLogger("sync.worker")


class Worker(mp.Process):
    """Long-lived worker process running CFR jobs dispatched by the server."""

    def __init__(
        self,
        job_queue: mp.Queue,
        logging_queue: mp.Queue,
        locks: Dict[str, mp.synchronize.Lock],
        tables: CFRTables,
        lut_path: Union[str, Path],
        pickle_dir: bool,
        n_players: int,
        prune_threshold: int,
        c: int,
        save_path: Path,
        info_set_lut=None,
        error_event: Optional[mp.Event] = None,  # type: ignore
    ):
        super().__init__(group=None, name=None, args=(), kwargs={}, daemon=None)
        self._job_queue: mp.Queue = job_queue
        self._logging_queue: mp.Queue = logging_queue
        self._locks = locks
        self._n_players = n_players
        self._tables = tables
        self._c = c
        self._prune_threshold = prune_threshold
        self._save_path = Path(save_path)
        self._lut_path = str(lut_path)
        self._pickle_dir = pickle_dir
        self._error_event: Optional[mp.Event] = error_event  # type: ignore
        if info_set_lut is not None:
            self._info_set_lut = info_set_lut
        # Persistent regret accumulator keyed by (betting_round, info_set).
        # Batched across many CFR calls and flushed on explicit "sync" jobs
        # dispatched by the server (Phase 5 decoupling).
        self._local_delta: Dict[Tuple[int, str], np.ndarray] = {}

    def run(self):
        """Set up post-fork state, then process jobs from the queue."""
        # Reopen LMDB indexes so this process gets its own reader lock-table
        # slots (avoids MDB_BAD_RSLOT on concurrent transactions).
        self._tables.reopen_after_fork()
        if not hasattr(self, "_info_set_lut"):
            self._info_set_lut = load_info_set_lut(self._lut_path, self._pickle_dir)
        self._set_seed()

        while True:
            name, kwargs = self._job_queue.get(block=True)
            should_break = False
            try:
                if name == "terminate":
                    self._flush_delta()
                    should_break = True
                elif name == "cfr":
                    game_state = state.new_game(
                        self._n_players, self._info_set_lut,
                    )
                    cfr_step(
                        self._tables,
                        game_state,
                        kwargs["i"],
                        kwargs["t"],
                        self._prune_threshold,
                        self._c,
                        self._local_delta,
                    )
                elif name == "sync":
                    self._flush_delta()
                elif name == "update_strategy":
                    game_state = state.new_game(
                        self._n_players, self._info_set_lut,
                    )
                    strategy_step(self._tables, game_state, kwargs["i"])
                else:
                    raise ValueError(f"Unrecognised job name: {name}")
            except Exception:
                log.exception(
                    f"[worker={self.name}] Unhandled exception in job '{name}' — "
                    f"signaling shutdown"
                )
                if self._error_event is not None:
                    self._error_event.set()
                should_break = True
            finally:
                self._job_queue.task_done()
            if should_break:
                break

    def _set_seed(self):
        """Seed the RNG with an independent stream for this worker.

        NumPy in particular has a problem with processes and seeds:
        https://github.com/numpy/numpy/issues/9650
        """
        random_seed: int = int.from_bytes(os.urandom(4), byteorder="little")
        utils.random.seed(random_seed)

    def _flush_delta(self) -> None:
        """Flush ``_local_delta`` into the shared regret tables.

        Each ``(betting_round, info_set)`` delta is routed into the
        correct per-street regret table via ``merge_delta_row``, which
        acquires the stripe lock internally.  ``_local_delta`` is
        cleared after the flush.
        """
        if not self._local_delta:
            return
        n_infosets = len(self._local_delta)
        merge_local_delta(self._tables, self._local_delta)
        self._local_delta.clear()
        self._logging_queue.put(
            f"[worker={self.name}] Synced {n_infosets:,} infosets to master",
            block=True,
        )
