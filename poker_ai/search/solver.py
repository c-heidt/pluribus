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

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from poker_ai.search.budget import iteration_budget
from poker_ai.search.context import SubgameContext
from poker_ai.search.mccfr import _MCCFRSolver
from poker_ai.search.parallel import run_loop
from poker_ai.search.policy import SearchPolicy
from poker_ai.search.solver_state import SearchStats, SolverConfig, SolverState
from poker_ai.search.vector import _VectorSolver, ox_enter_prob

__all__ = [
    "solve",
    "SolverConfig",
    "SolverState",
    "SearchResult",
    "SearchStats",
    "SearchPolicy",
    "config_fingerprint",
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
        Number of CFR iterations actually executed (≤ ``cfg.max_iterations``); the
        pooled sum across replicas on the parallel path.
    wall_seconds : float
        Wall-clock time spent in the iteration loop (excludes result packaging).
    regime : str
        Which CFR regime ran: ``'mccfr'`` or ``'vector'`` (eval doc §6).
    n_live : int
        Number of live ranges the solver sized the subgame on (``len(ctx.ranges)``)
        — the exact axis the MCCFR budget scales by (and the vector/MCCFR split keys
        on).  Logged per decision so calibration can group throughput/budget by the
        live-player count; distinct from the table-active ``num_live`` (all-in
        contestants keep a range here but are not table-active).
    stop_reason : str
        Which cap ended the search: ``'iteration_cap'`` (ran the full structural
        iteration budget — the normal case) or ``'wall_cap'`` (the wall backstop
        broke early).
    stats : SearchStats
        Walk/cache instrumentation counters (node/tree size, cache hit/miss),
        snapshotted for the ``decisions`` grain (eval doc §9.1).
    """

    policy: SearchPolicy
    average_policy: SearchPolicy
    state: SolverState
    iterations_run: int
    wall_seconds: float
    regime: str
    n_live: int
    stop_reason: str
    stats: SearchStats = field(default_factory=SearchStats)
    # OX-Search (Approach B, §11.3): the opt-out saturation metric (mean ENTER-prob
    # over feasible opponent infosets) when the gadget ran, else ``None``.  Non-``None``
    # is ALSO the "OX was active" signal — the agent plays the weighted-average (not the
    # final iterate) exactly when this is set (decision 5), and the eval logs it.
    ox_enter_prob: Optional[float] = None
    # Calibration root-value signal (eval doc §9): the hero's root counterfactual value
    # for its ACTUAL hand, as the played hand's conditional EV vs the belief opponent
    # (chips), linearly averaged over iterations and pooled across replicas.  ``None``
    # when no played-combo value was tracked (bot not a live seat / hand not in range /
    # zero iterations).  A value-based, equilibrium-invariant convergence signal — the
    # calibration reads it in place of full-policy L1 (which never vanishes at an
    # indifferent infoset and over-weights cold, never-played mass).
    root_value: Optional[float] = None


def _select_regime(ctx: SubgameContext) -> str:
    """Regime for ``ctx`` (§6.5): vector iff heads-up **turn/river**, else MCCFR.

    Vector-form CFR is the *small / late* path — a heads-up (two live seats)
    subgame with **at most one future chance node** left to resolve: a **turn** root
    (river ahead) or a **river** root (nothing ahead).  There it is full-width and
    cheap, and its exactness-per-iteration is a real quality win.

    A heads-up **flop** root, though, still has **two** future chance nodes
    (turn *and* river): the vector walk is full-width across the whole flop→turn→river
    betting tree (~10^5 nodes/iteration), which is ~1 iteration/second even on the
    compiled core — its per-replica budget (1500) would need ~minutes/replica, and the
    budget is *not* divisible across workers (each full-width replica needs the whole
    horizon), so more cores do not shorten it.  So the flop goes to **external-sampling
    MCCFR** instead: sampled opponent actions make each iteration ~100× cheaper, and the
    MCCFR budget is a global pool *divided* across replicas, so it scales down with the
    worker count on the 64-core target.  (The vector regime still *supports* a flop root
    — the differential/oracle harness drives it directly — it is just no longer routed
    there in production.)

    Everything else — the pre-flop root, and any multiway subgame — is MCCFR too.
    """
    if len(ctx.ranges) == 2 and ctx.street_at_root in (2, 3):
        return "vector"
    return "mccfr"



def config_fingerprint(
    cfg: SolverConfig, table_policy: Optional[Any] = None
) -> str:
    """Stable short hash of the solver identity (eval doc §6, §9.1).

    Canonicalises the ``SolverConfig`` + its ``LeafConfig`` scalar knobs (and, when
    the runner supplies one, the ``table_policy``) and returns a 16-hex-char SHA-256
    digest, so records group across runs even as unrelated settings are tweaked.
    Sorted keys + JSON's fixed float formatting give the canonicalisation the risk
    note (§11) calls for.  The un-hashable ``LeafConfig.policies`` fleet is
    represented by its set of bias-class keys (the fleet's *shape*, not object
    identity — two runs with the same four §4 variants fingerprint alike).
    """
    payload = {
        "solver": {
            "max_iterations": cfg.max_iterations,
            "max_wall_seconds": cfg.max_wall_seconds,
            "discount_interval": cfg.discount_interval,
            "beta": getattr(cfg, "beta", None),
        },
        "leaf": {
            "policies": sorted(str(k) for k in cfg.leaf.policies),
        },
        "table_policy": table_policy,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def solve(
    root_env,
    ctx: SubgameContext,
    cfg: SolverConfig,
    warm_start: Optional[SolverState] = None,
    regime_override: Optional[str] = None,
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
    regime_override : {'vector', 'mccfr'}, optional
        Force the CFR regime instead of the ``_select_regime`` routing.  For
        **A/B harnesses** (e.g. the calibration's turn regime comparison), where
        the same root is solved under both regimes at equal budget to compare
        results — both regimes are street-general, production just routes each
        street to one of them.  ``None`` (default) uses the production routing, so
        every existing caller is unchanged.  **OX-Search note:** the gadget root
        (``cfg.beta`` set) lives only in the vector regime, so forcing ``'mccfr'``
        with ``beta`` set would silently solve the *vanilla* tree (no gadget); the
        caller must not do that (the calibration gates the A/B off when ``beta`` is
        set).  With ``auto_budget`` on, the structural budget still derives the
        per-stage count from ``ctx`` (not the override); the A/B path forces
        ``auto_budget=False`` so the two regimes run an identical per-replica count.

    Returns
    -------
    SearchResult
    """
    if regime_override is not None and regime_override not in ("vector", "mccfr"):
        raise ValueError(
            f"regime_override must be 'vector', 'mccfr', or None; got {regime_override!r}."
        )
    if regime_override == "mccfr" and getattr(cfg, "beta", None) is not None:
        # The OX-Search gadget root lives only in the vector regime (_VectorSolver).
        # Forcing 'mccfr' with beta set would silently solve the vanilla/DBR tree —
        # no gadget, no opt-out row, ox_enter_prob stays None — with no error, while
        # the caller believes it configured OX-Search. Documented as a caller
        # obligation; enforced here too so a future/alternate caller can't hit it
        # silently.
        raise ValueError(
            "solve(): regime_override='mccfr' is incompatible with cfg.beta set "
            "(OX-Search gadget requires the vector regime); leave regime_override "
            "None or clear cfg.beta."
        )
    regime = regime_override if regime_override is not None else _select_regime(ctx)
    # Structural iteration budget (§6.5): replace ``max_iterations`` with the per-replica
    # count derived from the subgame's structure — vector = per-stage constant, MCCFR =
    # ``base[street] * n_live``.  The primary, machine-independent stop; the original
    # ``max_iterations`` stays the absolute ceiling.  ``auto_budget=False`` leaves it
    # untouched.
    cfg = dataclasses.replace(cfg, max_iterations=iteration_budget(ctx, cfg))

    # One serial search (production runs one hand per core — per-hand parallelism — so a
    # single search never forks a replica pool).
    state = warm_start if warm_start is not None else SolverState.empty()
    if warm_start is not None:
        # Re-search: the reused state carries the prior solve's cumulative counters —
        # zero them so this invocation's stats are its own (§9.1).
        state.reset_counters()
    if regime == "vector":
        solver = _VectorSolver(root_env, state, ctx, cfg, ctx.rng)
    else:
        solver = _MCCFRSolver(root_env, state, ctx, cfg, ctx.rng)
    start = time.perf_counter()
    iterations, stop_reason = run_loop(solver, state, cfg)
    # The MCCFR regime walks ``root_env`` in place (reseat per iteration) rather than
    # deepcopying it, so rewind it to pristine — solve() must not mutate ``root_env``
    # (warm re-search reuses it).  The vector regime walks via make/undo balance (or a
    # separate FastState) and already leaves it clean.
    if regime != "vector":
        solver.restore_root()
    wall = time.perf_counter() - start
    stats = state.stats_snapshot()

    # OX-Search saturation metric (Approach B): read off the solved/merged state's
    # opt-out row, and only in the regime that runs the gadget.  Non-None is the
    # "OX was active" signal the agent uses to play the weighted-average (decision 5).
    ox = (ox_enter_prob(state, root_env, ctx.board_compatible)
          if getattr(cfg, "beta", None) is not None and regime == "vector" else None)

    # Root-value convergence signal (calibration): read the pooled linear estimate off
    # the solved/merged state.  ``den == 0`` ⇒ nothing was tracked (bot not live, hand
    # not in range, or zero iterations) ⇒ leave it ``None``.
    rv_den = getattr(state, "root_value_den", 0.0)
    root_value = (state.root_value_num / rv_den) if rv_den > 0.0 else None

    return SearchResult(
        policy=SearchPolicy(state, use_average=False),
        average_policy=SearchPolicy(state, use_average=True),
        state=state,
        iterations_run=iterations,
        wall_seconds=wall,
        regime=regime,
        n_live=len(ctx.ranges),
        stop_reason=stop_reason,
        stats=stats,
        ox_enter_prob=ox,
        root_value=root_value,
    )
