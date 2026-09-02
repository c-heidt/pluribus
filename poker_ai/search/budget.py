"""Structural per-subgame iteration budget for the real-time solver (§6.5).

The search runs **no online convergence test**: the per-subgame iteration count is a
function of the subgame's *structure*, which — unlike a wall-clock cap — is
machine-independent and known up front.  This is the primary stop;
``SolverConfig.max_iterations`` is an absolute ceiling and ``max_wall_seconds`` a
loose backstop.

Two regimes, sized on different principles:

- **Vector** (heads-up turn/river) is full-width, so iterations-to-converge is driven
  by tree *depth*, not by the infoset count (which only bounds the per-iteration wall).
- **MCCFR** (multiway, or the heads-up pre-flop root) is sampled, and only the **hot
  path** must converge — nodes the solved tree never covers fall back to the blueprint
  at play time.

The budget is **looked up, not computed**: one explicit number per
``(approach, street, n_live)`` in :data:`~poker_ai.search.solver_state.MCCFR_BUDGET` /
:data:`~poker_ai.search.solver_state.VECTOR_BUDGET`.  No per-live-player base and no
per-approach scale factor — DBR's cost is not a fixed multiple of vanilla's, and a
shared base would couple unrelated cells.

Both regimes yield a **per-replica** count, and production runs one replica
(``workers=1``), so the budget IS the work.  The regime split mirrors
:func:`poker_ai.search.solver._select_regime` exactly.
"""

from __future__ import annotations

import logging

from poker_ai.search.context import SubgameContext
from poker_ai.search.solver_state import DBR, OX, VANILLA, SolverConfig

logger = logging.getLogger(__name__)


def search_approach(ctx: SubgameContext, cfg: SolverConfig) -> str:
    """Which approach this solve is — ``'vanilla'`` | ``'dbr'`` | ``'ox'``.

    Inferred from the solve's own inputs, so no caller can label a solve one thing and
    configure it another: opponent models (``ctx.models``) ⇒ DBR, ``cfg.beta`` OR
    ``cfg.ox_kbeta`` (the deck-agnostic form of the same knob) ⇒ OX, neither ⇒ the
    vanilla baseline.  ``runner.for_condition`` makes the two mutually
    exclusive, so a solve carrying both is a config bug — prefer DBR (models change the
    walk; ``beta`` outside the vector regime is inert) and warn.
    """
    has_models = bool(getattr(ctx, "models", None))
    has_beta = (getattr(cfg, "beta", None) is not None
                or getattr(cfg, "ox_kbeta", None) is not None)
    if has_models and has_beta:
        logger.warning(
            "solve carries BOTH opponent models and beta=%s — these are mutually "
            "exclusive approaches; budgeting it as DBR.", cfg.beta,
        )
    if has_models:
        return DBR
    return OX if has_beta else VANILLA


def _is_vector(ctx: SubgameContext) -> bool:
    """Vector iff heads-up (two live seats) on the **turn/river**; mirrors
    ``_select_regime``.

    A heads-up **flop** root routes to sampled MCCFR instead: the full-width vector walk
    over the flop-turn-river tree runs at ~1 iter/s and its budget does not divide
    across workers, so it does not scale.
    """
    return len(ctx.ranges) == 2 and ctx.street_at_root in (2, 3)


def _lookup(table, street: int, n_live: int, approach: str, regime: str) -> int:
    """The explicit budget for ``(street, n_live)``, or the nearest larger-table entry.

    The tables cover up to 4 live players.  An uncovered live-count (a 6p table) would
    otherwise KeyError mid-hand, so it falls back to the LARGEST covered one for that
    street and warns once.  The hot path grows with the live count, so that fallback is
    a deliberate UNDER-budget — the log line means "measure the cell and add it", not
    "trust this number".
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
    hatch for tests pinning an exact count).  Otherwise the structural budget for the
    selected regime — the ``(street, n_live)`` entry of this approach's vector or MCCFR
    table — clamped to ``[1, cfg.max_iterations]``.

    ``regime_override`` (``"vector"`` / ``"mccfr"``) forces the regime instead of the
    ``_select_regime`` routing, for the calibration A/B that solves one root under both
    and needs each regime's own production budget.
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
