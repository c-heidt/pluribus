"""Structural per-subgame iteration budget for the real-time solver (§6.5).

The real game is far too large for any single sampled replica to reach a tight
equilibrium *online* (a huge tree, one trajectory sampled per iteration, and W
independent replicas that are merged only at the end — pooling cuts the merged
variance, not the per-replica iteration count).  So the search does **not** run an
online convergence test.  Instead the per-subgame iteration count is a function of
the subgame's *structure*, which — unlike a wall-clock cap — is machine-independent
and known up front.  This is the primary stop; ``SolverConfig.max_iterations`` is an
absolute safety ceiling and ``max_wall_seconds`` a loose backstop.

Two regimes, sized on different principles (they differ by design, §6.5):

- **Vector** (heads-up flop/turn/river) is **full-width** — every iteration updates
  every infoset over the whole range.  Iterations-to-converge is therefore driven by
  tree *depth* (streets left to resolve: flop > turn > river), essentially *not* by
  the infoset count.  So the budget is a per-stage constant; the infoset count
  (~746k flop / ~89k turn / ~26k river at 200 buckets/street) only bounds the
  per-iteration *wall* (flop ≈ 1500 it × 746k rows ≈ tens of s/replica, i.e. the
  paper's 1–33 s envelope).

- **MCCFR** (multiway, or the heads-up pre-flop root) is **sampled**, and only the
  **hot path** must converge — nodes the solved tree never covers fall back to the
  blueprint at play time (a hard fallback, no blend), so they need no search refinement.  Its
  per-replica budget grows ~linearly with the live-player count (a bigger hot path),
  **not** the exponential full-tree size: ``base[street] × n_live``, clamped to
  ``max_iterations``.

Both regimes yield a **per-replica** count.  Production always runs a single replica
(``workers=1`` — one hand per core, per-hand parallelism), so the budget IS the work.
Extra replicas would only add samples to the merged average (variance reduction); they
never divide the budget.

The regime split mirrors :func:`poker_ai.search.solver._select_regime` exactly.
"""

from __future__ import annotations

from poker_ai.search.context import SubgameContext
from poker_ai.search.solver_state import SolverConfig


def _is_vector(ctx: SubgameContext) -> bool:
    """Vector iff heads-up (two live seats) on the **turn/river** — §6.5, mirrors
    ``_select_regime``.

    A heads-up **flop** root has two future chance nodes left (turn+river) and is
    routed to sampled MCCFR instead — the full-width vector walk over the whole
    flop→turn→river tree is ~1 iter/s and its budget does not divide across workers,
    so it does not scale on the 64-core target.  Turn (one chance node ahead) and
    river (none) stay vector.
    """
    return len(ctx.ranges) == 2 and ctx.street_at_root in (2, 3)


def iteration_budget(ctx: SubgameContext, cfg: SolverConfig,
                     regime_override: "str | None" = None) -> int:
    """Per-replica iterations to run for the subgame ``ctx`` under ``cfg`` (§6.5).

    Returns ``cfg.max_iterations`` verbatim when ``cfg.auto_budget`` is off (the escape
    hatch for tests that pin an exact iteration count).  Otherwise returns the
    structural **per-replica** budget for the selected regime, clamped to at most
    ``cfg.max_iterations`` (the absolute ceiling) and at least 1:

    - **vector** — a per-stage constant indexed (flop, turn, river);
    - **MCCFR** — ``base[street] × n_live`` (the hot path grows ~linearly with the live
      players), clamped to ``max_iterations``.

    Both are per-replica: production runs one replica (``workers=1``); extra replicas
    only reduce variance, they never divide the budget.

    ``regime_override`` (``"vector"`` / ``"mccfr"``) forces the regime instead of the
    ``_select_regime`` routing — used by the calibration A/B, which solves the same root
    under BOTH regimes and needs each regime's *own* production budget as the ladder
    centre (the forced-mccfr HU-turn arm would otherwise read the vector budget).
    """
    if not getattr(cfg, "auto_budget", True):
        return cfg.max_iterations

    is_vec = (regime_override == "vector" if regime_override is not None
              else _is_vector(ctx))
    if is_vec:
        # Per-stage constant, indexed (flop, turn, river) = street 1, 2, 3.
        flop, turn, river = cfg.vector_budget_by_street
        budget = {1: flop, 2: turn, 3: river}[ctx.street_at_root]
    else:
        # Hot-path budget, ~linear in the live-player count (a deep multiway flop needs
        # more sampled work than a river).  base indexed by ``street_at_root``
        # (0=preflop … 3=river).
        n_live = max(2, len(ctx.ranges))
        budget = cfg.mccfr_per_player_by_street[ctx.street_at_root] * n_live
        # DBR needs more iterations than vanilla for the same VALUE convergence (its
        # tail-driven objective inflates the sampled-value variance), so scale UP only
        # when the subgame carries opponent models.  Vanilla (no models) is byte-untouched.
        if getattr(ctx, "models", None):
            budget = budget * float(getattr(cfg, "dbr_mccfr_scale", 1.0))

    return max(1, min(int(budget), int(cfg.max_iterations)))
