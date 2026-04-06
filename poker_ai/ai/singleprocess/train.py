"""Single-process CFR training loop — baseline for the multiprocess server.

This mode mirrors :class:`poker_ai.ai.multiprocess.server.Server` exactly:
the same sync-cycle-based schedule, the same LCFR discount formula, the
same strategy/discount windows.  It exists as a low-overhead reference
for debugging and correctness checks, not a different algorithm.

All non-trivial logic lives in :mod:`poker_ai.ai.training`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from tqdm import tqdm, trange

from poker_ai import utils
from poker_ai.ai.cfr import merge_local_delta
from poker_ai.ai.cfr_tables import CFRTables
from poker_ai.ai.training import (
    DiscountState,
    at_sync_barrier,
    cfr_step,
    should_discount,
    should_update_strategy,
    strategy_step,
)
from poker_ai.environment.poker_env import new_game, PokerEnv as PokerState


log = logging.getLogger("poker_ai.ai.singleprocess")


def print_strategy(strategy: Dict[str, Dict[str, int]]):
    """Print a normalised strategy dict."""
    for info_set, action_to_probabilities in sorted(strategy.items()):
        norm = sum(list(action_to_probabilities.values()))
        tqdm.write(f"{info_set}")
        for action, probability in action_to_probabilities.items():
            tqdm.write(f"  - {action}: {probability / norm:.2f}")


def simple_search(
    config: Dict[str, int],
    save_path: Path,
    lut_path: Union[str, Path],
    pickle_dir: bool,
    strategy_interval: int,
    n_iterations: int,
    discount_duration_cycles: int,
    prune_threshold: int,
    c: int,
    n_players: int,
    update_threshold: int,
    sync_interval: int,
    discount_interval: int,
):
    """Train a single-process CFR agent.

    All cycle-based parameters (``strategy_interval``, ``discount_interval``,
    ``discount_duration_cycles``, ``update_threshold``) are counted in
    **sync cycles** (= ``sync_interval`` raw iterations), matching the
    multi-process server.  ``prune_threshold`` stays in raw iterations
    because pruning is a per-traversal decision.
    """
    from poker_ai.ai.index import lmdb_map_size_for_players

    utils.random.seed(42)
    tables = CFRTables(
        index_path=save_path / "lmdb_index",
        lmdb_map_size=lmdb_map_size_for_players(n_players),
    )
    discount_state = DiscountState(
        duration_cycles=discount_duration_cycles,
        discount_interval=discount_interval,
    )
    card_info_lut: Dict = {}

    for t in trange(1, n_iterations + 1, desc="train iter"):
        if t == 2:
            logging.disable(logging.DEBUG)

        for i in range(n_players):
            state: PokerState = new_game(
                n_players,
                card_info_lut,
                lut_path=lut_path,
                pickle_dir=pickle_dir,
            )
            card_info_lut = state.card_info_lut
            local_delta: Dict[Tuple[int, str], np.ndarray] = {}
            cfr_step(tables, state, i, t, prune_threshold, c, local_delta)
            merge_local_delta(tables, local_delta)

        if at_sync_barrier(t, sync_interval):
            sync_step = t // sync_interval

            if should_update_strategy(sync_step, strategy_interval, update_threshold):
                for i in range(n_players):
                    state = new_game(
                        n_players,
                        card_info_lut,
                        lut_path=lut_path,
                        pickle_dir=pickle_dir,
                    )
                    card_info_lut = state.card_info_lut
                    strategy_step(tables, state, i)

            if should_discount(sync_step, discount_interval):
                discount_state.apply(tables, sync_step)
