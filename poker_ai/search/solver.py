"""Depth-limited subgame solver — public entry point (§6.5).

:func:`solve` runs one real-time search over a subgame root and returns a
:class:`SearchResult`.  It is a **thin orchestrator** over the composition pieces:
:class:`~poker_ai.search.solver_state.SolverState` holds the shared CFR tables,
one regime class owns the iteration body (MCCFR now; the vector regime is a
documented seam, :mod:`poker_ai.search.vector`), and :class:`SearchPolicy` reads
the result.  The orchestrator only selects the regime, runs the iteration loop
with the Linear-CFR discount cadence and the dual stop, and packages the result.

``SolverConfig`` / ``SolverState`` are re-exported here so callers have a single
public surface: ``from poker_ai.search.solver import solve, SolverConfig,
SolverState, SearchResult``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from poker_ai.search.context import SubgameContext
from poker_ai.search.mccfr import _MCCFRSolver
from poker_ai.search.parallel import (
    plan_workers,
    resolve_workers,
    run_loop,
    run_parallel,
)
from poker_ai.search.policy import SearchPolicy
from poker_ai.search.solver_state import SolverConfig, SolverState
from poker_ai.search.vector import _VectorSolver

__all__ = [
    "solve",
    "SolverConfig",
    "SolverState",
    "SearchResult",
    "SearchPolicy",
]


@dataclass
class SearchResult:
    """Outcome of one :func:`solve` call.

    Attributes
    ----------
    policy : SearchPolicy
        Final-iteration strategy — the policy the bot plays at its actual hand.
    average_policy : SearchPolicy
        Weighted-average strategy — feeds the next round's belief update.
    state : SolverState
        The solved tables; the warm-start carrier and freeze store for re-search.
    iterations_run : int
        Number of CFR iterations actually executed (≤ ``cfg.max_iterations``).
    wall_seconds : float
        Wall-clock time spent in the iteration loop.
    """

    policy: SearchPolicy
    average_policy: SearchPolicy
    state: SolverState
    iterations_run: int
    wall_seconds: float


def _select_regime(ctx: SubgameContext) -> str:
    """Regime for ``ctx`` (§6.5): vector iff heads-up turn/river, else MCCFR.

    Vector-form CFR is used for the *small / late* subgames — heads-up
    (two live seats) rooted on the turn or river.  Everything else (round 1,
    all of round 2, any multiway later subgame) uses external-sampling MCCFR.
    """
    if len(ctx.ranges) == 2 and ctx.street_at_root in (2, 3):
        return "vector"
    return "mccfr"


def solve(
    root_env,
    ctx: SubgameContext,
    cfg: SolverConfig,
    warm_start: Optional[SolverState] = None,
) -> SearchResult:
    """Search ``root_env`` and return the bot's strategy plus the solved state.

    Parameters
    ----------
    root_env : PokerEnv
        Subgame root; **deepcopied by the caller** (the solver walks it with
        make/undo and per-iteration ``with_hole_cards`` copies, never mutating
        the caller's env irrecoverably).
    ctx : SubgameContext
        Per-search inputs — ranges (incl. the bot), folded ranges, board mask,
        depth limit, leaf config, and the RNG seat.
    cfg : SolverConfig
        Hyperparameters (iteration / wall-clock budget, discount cadence, leaf
        config).
    warm_start : SolverState, optional
        A prior state to re-search in place (reusing rows, growing widened
        nodes, carrying the freeze map).  ``None`` starts fresh.

    Returns
    -------
    SearchResult
    """
    regime = _select_regime(ctx)
    workers = resolve_workers(getattr(cfg, "workers", 1))
    # Both regimes parallelize the same way (§6.7 row 11): W independent replicas,
    # merged once.  The vector regime is chance-sampled (one river per iteration),
    # so its replicas draw independent river substreams and summing their regrets
    # multiplies the effective samples per river by W — directly buying back the
    # per-river sampling variance.  (No nested intra-replica parallelism: the
    # showdown is ~25% of an iteration and fine-grained, so replica-level scaling
    # of the whole iteration dominates — see §6.7.)

    if workers > 1:
        # Parallel: W independent replicas, merged once at the end (§6.7 row 11).
        # The base seed is drawn from ctx.rng so a given (ctx.rng state, workers)
        # is reproducible; the staggered traverser rotation balances across the
        # live players (len(ctx.ranges)).
        base_seed = int(ctx.rng.integers(0, 2 ** 63 - 1))
        plan = plan_workers(workers, len(ctx.ranges), base_seed=base_seed)
        state, iterations, wall = run_parallel(
            root_env, ctx, cfg, warm_start, plan, regime
        )
    else:
        # Serial: the original single-thread loop — bit-for-bit unchanged.
        state = warm_start if warm_start is not None else SolverState.empty()
        if regime == "vector":
            solver = _VectorSolver(root_env, state, ctx, cfg, ctx.rng)
        else:
            solver = _MCCFRSolver(root_env, state, ctx, cfg, ctx.rng)
        start = time.perf_counter()
        iterations = run_loop(solver, state, cfg)
        wall = time.perf_counter() - start

    return SearchResult(
        policy=SearchPolicy(state, use_average=False),
        average_policy=SearchPolicy(state, use_average=True),
        state=state,
        iterations_run=iterations,
        wall_seconds=wall,
    )
