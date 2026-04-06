"""Single-process CFR training loop — baseline for multiprocess server.

This mode mirrors the multi-process ``Server`` loop exactly: the same
sync-cycle-based discount, strategy-update, and (future) checkpoint
schedules.  It exists as a low-overhead reference for debugging and
correctness checks, not a different algorithm.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from tqdm import tqdm, trange

from poker_ai.ai.action_space import MAX_ACTIONS_PER_STREET
from poker_ai.ai.cfr import cfr, cfrp, merge_local_delta
from poker_ai.ai.cfr_tables import CFRTables
from poker_ai.ai.strategy import update_strategy
from poker_ai import utils
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
    utils.random.seed(42)
    from poker_ai.ai.index import lmdb_map_size_for_players
    tables = CFRTables(
        index_path=save_path / "lmdb_index",
        lmdb_map_size=lmdb_map_size_for_players(n_players),
        actions_per_street=MAX_ACTIONS_PER_STREET,
    )
    card_info_lut = {}
    discounting_active: bool = True

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
            if t > prune_threshold:
                if random.uniform(0, 1) < 0.05:
                    cfr(tables=tables, state=state, i=i, t=t, local_delta=local_delta)
                else:
                    cfrp(tables=tables, state=state, i=i, t=t, c=c, local_delta=local_delta)
            else:
                cfr(tables=tables, state=state, i=i, t=t, local_delta=local_delta)
            merge_local_delta(tables=tables, local_delta=local_delta)

        # Sync-cycle-based schedules — mirror Server.search().
        if t % sync_interval == 0:
            sync_step = t // sync_interval

            if (
                sync_step > update_threshold
                and sync_step % strategy_interval == 0
            ):
                for i in range(n_players):
                    state = new_game(
                        n_players,
                        card_info_lut,
                        lut_path=lut_path,
                        pickle_dir=pickle_dir,
                    )
                    card_info_lut = state.card_info_lut
                    update_strategy(tables=tables, state=state, i=i)

            if discounting_active and sync_step % discount_interval == 0:
                if sync_step >= discount_duration_cycles:
                    discounting_active = False
                    log.info(
                        f"Discount window closed after {sync_step} sync cycles"
                    )
                else:
                    discount_step = sync_step // discount_interval
                    d = discount_step / (discount_step + 1)
                    log.info(
                        f"[sync_step={sync_step}] Discounting "
                        f"(step={discount_step}, factor={d:.4f})"
                    )
                    tables.apply_discount(d)
