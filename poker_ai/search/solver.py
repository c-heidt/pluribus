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
    state = warm_start if warm_start is not None else SolverState.empty()
    regime = _select_regime(ctx)
    if regime == "vector":
        solver = _VectorSolver(root_env, state, ctx, cfg, ctx.rng)
    else:
        solver = _MCCFRSolver(root_env, state, ctx, cfg, ctx.rng)

    start = time.perf_counter()
    delta = cfg.discount_interval
    iterations = 0
    for t in range(1, cfg.max_iterations + 1):
        solver.iterate()
        iterations = t
        # Linear-CFR discount on the cadence: d = (t/Δ)/(t/Δ + 1) (§6.5).
        if delta > 0 and t % delta == 0:
            k = t / delta
            state.discount(k / (k + 1.0))
        if time.perf_counter() - start >= cfg.max_wall_seconds:
            break
    wall = time.perf_counter() - start

    return SearchResult(
        policy=SearchPolicy(state, use_average=False),
        average_policy=SearchPolicy(state, use_average=True),
        state=state,
        iterations_run=iterations,
        wall_seconds=wall,
    )
