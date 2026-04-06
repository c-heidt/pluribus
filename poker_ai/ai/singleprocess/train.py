"""Single-process CFR training loop.

Executes the same CFR schedule as
:class:`poker_ai.ai.multiprocess.server.Server` but in a single
Python process with no worker pool, no job queue, and no sync
barriers across processes.  All non-trivial per-traversal and
per-schedule logic is shared with the multi-process server through
:mod:`poker_ai.ai.training`, so the two modes are guaranteed to run
the same algorithm with the same hyperparameter semantics.

The single-process mode is intended as a low-overhead reference
implementation for debugging, correctness checks, and small
validation runs where the multiprocessing machinery would add more
noise than value.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from tqdm import tqdm, trange

from poker_ai import utils
from poker_ai.ai.action_space import MAX_ACTIONS_PER_STREET
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
    """Pretty-print a normalised strategy dict to stdout.

    Each information-set header is followed by its per-action
    probabilities normalised to sum to 1.  Intended for interactive
    use — training itself writes no human-readable output through
    this function.

    Parameters
    ----------
    strategy : dict[str, dict[str, int]]
        Mapping from information-set string to ``{action: count}``
        dicts.  Counts are normalised per row before printing.
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
    discount_duration_cycles: int,
    prune_threshold: int,
    c: int,
    n_players: int,
    update_threshold: int,
    sync_interval: int,
    discount_interval: int,
):
    """Run a single-process CFR training loop for *n_iterations* iterations.

    The loop mirrors :meth:`Server.search` exactly: one CFR traversal
    per player per iteration, a sync barrier every ``sync_interval``
    iterations, strategy updates and discounting scheduled by
    sync-cycle predicates, and the same
    :func:`poker_ai.ai.training.cfr_step` pruning decision.  The only
    structural differences are that local deltas are merged
    immediately after each traversal (there is no worker pool to
    batch for) and the stop condition is a fixed iteration count
    instead of a wall-clock budget.

    All cycle-based parameters (``strategy_interval``,
    ``discount_interval``, ``discount_duration_cycles``,
    ``update_threshold``) are counted in **sync cycles**, i.e.
    ``sync_interval`` raw iterations.  ``prune_threshold`` stays in
    raw iterations because pruning is a per-traversal decision.

    Parameters
    ----------
    config : dict
        Full hyperparameter dictionary, kept only for consistency
        with the multi-process entry point; not read directly here.
    save_path : Path
        Root directory for the LMDB index and any other on-disk
        state produced by the run.
    lut_path : str or Path
        Directory containing the card-info LUT.
    pickle_dir : bool
        Use the legacy pickle-directory LUT layout.
    strategy_interval : int
        Period (in sync cycles) between strategy-update passes.
    n_iterations : int
        Total number of training iterations to run.
    discount_duration_cycles : int
        Length of the LCFR discount window in sync cycles.
    prune_threshold : int
        Raw iteration after which CFR-P becomes eligible.
    c : int
        CFR-P regret threshold.
    n_players : int
        Number of players in the game.
    update_threshold : int
        Warm-up in sync cycles before strategy updates begin.
    sync_interval : int
        Raw iterations per sync cycle; base unit for every
        cycle-based parameter.
    discount_interval : int
        Period (in sync cycles) between LCFR discount applications.
    """
    from poker_ai.ai.index import lmdb_map_size_for_players

    utils.random.seed(42)
    tables = CFRTables(
        index_path=save_path / "lmdb_index",
        lmdb_map_size=lmdb_map_size_for_players(n_players),
        actions_per_street=MAX_ACTIONS_PER_STREET,
    )
    discount_state = DiscountState(
        duration_cycles=discount_duration_cycles,
        discount_interval=discount_interval,
    )
    card_info_lut: Dict = {}

    for t in trange(1, n_iterations + 1, desc="train iter"):
        if t == 2:
            # Silence DEBUG-level CFR traces after the first iteration so the
            # interactive progress bar stays readable.
            logging.disable(logging.DEBUG)

        for i in range(n_players):
            state: PokerState = new_game(
                n_players,
                card_info_lut,
                lut_path=lut_path,
                pickle_dir=pickle_dir,
            )
            # ``new_game`` caches the LUT on the returned state so we
            # can reuse it on subsequent calls without re-loading.
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
