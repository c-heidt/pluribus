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
  the infoset count, which only bounds the per-iteration *wall*.

- **MCCFR** (multiway, or the heads-up pre-flop root) is **sampled**, and only the
  **hot path** must converge — nodes the solved tree never covers fall back to the
  blueprint at play time (a hard fallback, no blend), so they need no search refinement.

The budget itself is **looked up, not computed**: an explicit number per
``(approach, street, n_live)`` in :data:`~poker_ai.search.solver_state.MCCFR_BUDGET` /
:data:`~poker_ai.search.solver_state.VECTOR_BUDGET`.  There is no per-live-player base
multiplied out and no per-approach scale factor.  Those formulas implied a regularity the
measurements do not have — DBR's cost is not a fixed multiple of vanilla's, and its
convergence point differs per street — and they coupled unrelated cells, so retuning one
wall-bound cell silently moved every cell sharing the base.  See the tables for how each
number was derived and which are wall-clipped rather than converged.

Both regimes yield a **per-replica** count.  Production always runs a single replica
(``workers=1`` — one hand per core, per-hand parallelism), so the budget IS the work.
Extra replicas would only add samples to the merged average (variance reduction); they
never divide the budget.

The regime split mirrors :func:`poker_ai.search.solver._select_regime` exactly.
"""

from __future__ import annotations

import logging

from poker_ai.search.context import SubgameContext
from poker_ai.search.solver_state import DBR, OX, VANILLA, SolverConfig

logger = logging.getLogger(__name__)


def search_approach(ctx: SubgameContext, cfg: SolverConfig) -> str:
    """Which approach this solve is — ``'vanilla'`` | ``'dbr'`` | ``'ox'``.

    Inferred from the solve's own inputs rather than passed in, so no caller can label a
    solve one thing and configure it another:

    - **DBR** is the approach that carries opponent models (``ctx.models``);
    - **OX-Search** is the one that sets ``cfg.beta`` (the gadget root);
    - neither ⇒ the **vanilla** paper baseline.

    The two are mutually exclusive by construction — ``evaluation.runner.for_condition``
    rejects a model on an OX arm and a ``beta`` on a DBR arm — so the order here cannot
    mask a real configuration.  A solve that somehow carries both is a config bug, not a
    fourth approach: prefer DBR (the models genuinely change the walk, whereas ``beta``
    outside the vector regime is already inert) and say so loudly.
    """
    has_models = bool(getattr(ctx, "models", None))
    has_beta = getattr(cfg, "beta", None) is not None
    if has_models and has_beta:
        logger.warning(
            "solve carries BOTH opponent models and beta=%s — these are mutually "
            "exclusive approaches; budgeting it as DBR.", cfg.beta,
        )
    if has_models:
        return DBR
    return OX if has_beta else VANILLA


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


def _lookup(table, street: int, n_live: int, approach: str, regime: str) -> int:
    """The explicit budget for ``(street, n_live)``, or the nearest larger-table entry.

    The tables cover the cells production actually reaches (up to 4 live players, the
    blueprint's table size).  A deeper game — a 6p table, where ``n_live`` can reach 6 —
    would otherwise KeyError mid-hand and abort the search, so an uncovered live-count
    falls back to the LARGEST covered one for that street and says so once.  That is a
    deliberate under-budget, not a silent extrapolation: the hot path grows with the live
    count, so the fallback is too small, and the log line is the signal to measure the
    cell and add it rather than to trust the number.
    """
    hit = table.get((street, n_live))
    if hit is not None:
        return int(hit)
    covered = sorted(nl for (st, nl) in table if st == street)
    if not covered:
        raise KeyError(
            f"no {regime} budget for street {street} under approach {approach!r}; "
            f"the tables in poker_ai.search.solver_state need an entry for it"
        )
    fallback = covered[-1]
    logger.warning(
        "no %s budget for (street=%d, n_live=%d) under %r — falling back to the "
        "largest covered live-count (n_live=%d, %d iters). This UNDER-budgets the "
        "search; measure the cell and add it to solver_state.",
        regime, street, n_live, approach, fallback, table[(street, fallback)],
    )
    return int(table[(street, fallback)])


def iteration_budget(ctx: SubgameContext, cfg: SolverConfig,
                     regime_override: "str | None" = None) -> int:
    """Per-replica iterations to run for the subgame ``ctx`` under ``cfg`` (§6.5).

    Returns ``cfg.max_iterations`` verbatim when ``cfg.auto_budget`` is off (the escape
    hatch for tests that pin an exact iteration count).  Otherwise returns the
    structural **per-replica** budget for the selected regime, clamped to at most
    ``cfg.max_iterations`` (the absolute ceiling) and at least 1:

    - **vector** — the ``(street, n_live=2)`` entry of this approach's vector table;
    - **MCCFR** — the ``(street, n_live)`` entry of this approach's MCCFR table.

    Both are per-replica: production runs one replica (``workers=1``); extra replicas
    only reduce variance, they never divide the budget.

    ``regime_override`` (``"vector"`` / ``"mccfr"``) forces the regime instead of the
    ``_select_regime`` routing — used by the calibration A/B, which solves the same root
    under BOTH regimes and needs each regime's *own* production budget as the ladder
    centre (the forced-mccfr HU-turn arm would otherwise read the vector budget).
    """
    if not getattr(cfg, "auto_budget", True):
        return cfg.max_iterations

    approach = search_approach(ctx, cfg)
    n_live = max(2, len(ctx.ranges))
    street = int(ctx.street_at_root)
    is_vec = (regime_override == "vector" if regime_override is not None
              else _is_vector(ctx))
    table = (cfg.vector_budget if is_vec else cfg.mccfr_budget)[approach]
    budget = _lookup(table, street, n_live, approach,
                     "vector" if is_vec else "mccfr")
    return max(1, min(int(budget), int(cfg.max_iterations)))
