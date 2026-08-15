"""Single-process CFR training loop.

Executes the same CFR schedule as
:class:`poker_ai.blueprint.multiprocess.server.Server` but in a single
Python process with no worker pool, no job queue, and no sync
barriers across processes.  All non-trivial per-traversal and
per-schedule logic is shared with the multi-process server through
:mod:`poker_ai.blueprint.training`, so the two modes are guaranteed to run
the same algorithm with the same hyperparameter semantics.

The single-process mode is intended as a low-overhead reference
implementation for debugging, correctness checks, and small
validation runs where the multiprocessing machinery would add more
noise than value.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from environment.action_space import MAX_ACTIONS_PER_STREET
from poker_ai.blueprint.bias import BiasClass
from poker_ai.blueprint.cfr import merge_local_delta
from poker_ai.blueprint.core_runner import CoreDriver, core_enabled
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.tables.warm_start import (
    apply_warm_start_to_tables,
    stage_warm_start_lmdb,
)
from poker_ai.blueprint.training import (
    DiscountState,
    at_sync_barrier,
    cfr_step,
    pin_blas_threads,
    seed,
    should_discount,
    should_update_strategy,
    strategy_step,
)
from environment.poker_env import new_game, PokerEnv as PokerState


log = logging.getLogger("poker_ai.blueprint.singleprocess")


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
        log.info(info_set)
        for action, probability in action_to_probabilities.items():
            log.info(f"  - {action}: {probability / norm:.2f}")


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
    bias: BiasClass = "none",
    bias_magnitude: float = 0.0,
    warm_start: Optional[Union[str, Path]] = None,
    strategy_per_job: Optional[int] = None,
):
    """Run a single-process CFR training loop for *n_iterations* iterations.

    The loop mirrors :meth:`Server.search` exactly: one CFR traversal
    per player per iteration, a sync barrier every ``sync_interval``
    iterations, strategy updates and discounting scheduled by
    sync-cycle predicates, and the same
    :func:`poker_ai.blueprint.training.cfr_step` pruning decision.  The only
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
    strategy_per_job : int, optional
        Pre-flop UPDATE-STRATEGY playthroughs per player per strategy-update
        firing — the same knob as the multi-process server's
        ``strategy_per_job``.  Defaults to the ``PLURIBUS_STRATEGY_PER_JOB``
        environment variable if set, else ``1`` (one pass covers the whole
        pre-flop opponent tree per deal via full branching).
    """
    import os

    from poker_ai.tables.index import lmdb_map_size_for_players
    from information_abstraction import load_info_set_lut

    pin_blas_threads(1)

    _LOG_INTERVAL_SECS = 60.0

    if strategy_per_job is None:
        env_spj = os.environ.get("PLURIBUS_STRATEGY_PER_JOB")
        strategy_per_job = int(env_spj) if env_spj else 1
    if strategy_per_job < 1:
        raise ValueError(
            f"strategy_per_job must be >= 1, got {strategy_per_job}"
        )

    seed(42)
    shm_dir = save_path / "shm"
    shm_dir.mkdir(parents=True, exist_ok=True)
    warm_start_staged = False
    if warm_start is not None:
        warm_start_staged = stage_warm_start_lmdb(
            save_path=save_path,
            warm_start_path=Path(warm_start),
            expected_n_players=n_players,
        )
    enable_index_cache = os.environ.get("PLURIBUS_INDEX_CACHE", "1") == "1"
    tables = CFRTables(
        index_path=save_path / "lmdb_index",
        shm_dir=str(shm_dir),
        lmdb_map_size=lmdb_map_size_for_players(n_players),
        actions_per_street=MAX_ACTIONS_PER_STREET,
        enable_index_cache=enable_index_cache,
    )
    # Only restore chunks when we actually staged the LMDB this run —
    # otherwise the loaded chunks would belong to the warm-start's
    # info-set mapping but the tables would be reading the existing
    # destination LMDB's mapping, silently mis-routing every row.
    if warm_start is not None and warm_start_staged:
        apply_warm_start_to_tables(
            tables=tables,
            warm_start_path=Path(warm_start),
            expected_n_players=n_players,
        )
    # Prewarm the shm index caches from LMDB now that any warm-start /
    # resume mapping is in place (single process → no fork to precede).
    tables.prewarm_caches()
    # Optional compiled-core CFR (PLURIBUS_CFR_CORE=1).  Built after the
    # caches are warm so the pure-shm read path sees a complete mirror;
    # falls back to the Python path (core stays None) when unset or biased.
    core = None
    if core_enabled(bias):
        core = CoreDriver(tables, n_players)
        log.info("PLURIBUS_CFR_CORE=1 — driving CFR through the compiled core")
    discount_state = DiscountState(
        duration_cycles=discount_duration_cycles,
        discount_interval=discount_interval,
    )
    card_info_lut = load_info_set_lut(lut_path, pickle_dir)

    _start_time = time.monotonic()
    _last_log_time = _start_time
    log.info(f"Training started — {n_iterations} iterations, {n_players} players")

    for t in range(1, n_iterations + 1):
        for i in range(n_players):
            state: PokerState = new_game(n_players, card_info_lut)
            local_delta: Dict[Tuple[int, str], np.ndarray] = {}
            cfr_step(
                tables, state, i, t, prune_threshold, c, local_delta,
                bias=bias, bias_magnitude=bias_magnitude, core=core,
            )
            merge_local_delta(tables, local_delta)

        if at_sync_barrier(t, sync_interval):
            sync_step = t // sync_interval

            if should_update_strategy(sync_step, strategy_interval, update_threshold):
                for i in range(n_players):
                    for _ in range(strategy_per_job):
                        state = new_game(n_players, card_info_lut)
                        strategy_step(tables, state, i)

            if should_discount(sync_step, discount_interval):
                discount_state.apply(tables, sync_step)

            now = time.monotonic()
            if now - _last_log_time >= _LOG_INTERVAL_SECS:
                elapsed = now - _start_time
                iters_per_sec = t / elapsed if elapsed > 0 else 0.0
                remaining_iters = n_iterations - t
                remaining_secs = (
                    remaining_iters / iters_per_sec if iters_per_sec > 0 else float("inf")
                )
                log.info(
                    f"[t={t}/{n_iterations}  sync_step={sync_step}]  "
                    f"elapsed={elapsed:.0f}s  "
                    f"remaining≈{remaining_secs:.0f}s  "
                    f"({iters_per_sec:.1f} iter/s)"
                )
                _last_log_time = now

    # In deferred-allocation mode the shm cache is the live row authority and
    # LMDB lags; make the on-disk index current so a reopen / warm-start (and the
    # golden-trace regression) sees every allocated row.  No-op otherwise.
    tables.persist_indexes()
