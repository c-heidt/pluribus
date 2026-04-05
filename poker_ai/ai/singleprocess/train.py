"""
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, Tuple, Union

import click
import joblib
import numpy as np
import yaml
from tqdm import tqdm, trange

from poker_ai.ai.action_space import MAX_ACTIONS_PER_STREET
from poker_ai.ai.cfr import cfr, cfrp, merge_local_delta
from poker_ai.ai.cfr_tables import CFRTables
from poker_ai.ai.strategy import update_strategy
from poker_ai import utils
from poker_ai.environment.poker_env import new_game, PokerEnv as PokerState


def print_strategy(strategy: Dict[str, Dict[str, int]]):
    """
    Print strategy.

    ...

    Parameters
    ----------
    strategy : Dict[str, Dict[str, int]]
        The preflop strategy for our agent.
    """
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
    lcfr_threshold: int,
    prune_threshold: int,
    c: int,
    n_players: int,
    dump_iteration: int,
    update_threshold: int,
):
    """
    Train agent.

    ...

    Parameters
    ----------
    config : Dict[str, int],
        Configurations for the simple search.
    save_path : str
        Path to save to.
    strategy_interval : int
        Iteration at which to update strategy.
    n_iterations : int
        Number of iterations.
    lcfr_threshold : int
        Iteration at which to begin linear CFR.
    prune_threshold : int
        Iteration at which to begin pruning.
    c : int
        Floor for regret at which we do not search a node.
    n_players : int
        Number of players.
    dump_iteration : int
        Iteration at which we begin serialization.
    update_threshold : int
        Iteration at which we begin updating strategy.
    """
    discount_step: int = 0
    utils.random.seed(42)
    from poker_ai.ai.index import lmdb_map_size_for_players
    tables = CFRTables(
        index_path=save_path / "lmdb_index",
        lmdb_map_size=lmdb_map_size_for_players(n_players),
        actions_per_street=MAX_ACTIONS_PER_STREET,
    )
    card_info_lut = {}
    for t in trange(1, n_iterations + 1, desc="train iter"):
        if t == 2:
            logging.disable(logging.DEBUG)
        for i in range(n_players):  # fixed position i
            # Create a new state.
            state: PokerState = new_game(
                n_players,
                card_info_lut,
                lut_path=lut_path,
                pickle_dir=pickle_dir
            )
            card_info_lut = state.card_info_lut
            if t > update_threshold and t % strategy_interval == 0:
                update_strategy(tables=tables, state=state, i=i, t=t)
            local_delta: Dict[Tuple[int, str], np.ndarray] = {}
            if t > prune_threshold:
                if random.uniform(0, 1) < 0.05:
                    cfr(tables=tables, state=state, i=i, t=t, local_delta=local_delta)
                else:
                    cfrp(tables=tables, state=state, i=i, t=t, c=c, local_delta=local_delta)
            else:
                cfr(tables=tables, state=state, i=i, t=t, local_delta=local_delta)
            merge_local_delta(tables=tables, local_delta=local_delta)
        if t < lcfr_threshold:
            discount_step += 1
            d = discount_step / (discount_step + 1)
            tables.apply_discount(d)


if __name__ == "__main__":
    simple_search()
