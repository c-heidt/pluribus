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
  **hot path** must converge — rarely-reached infosets fall back to the blueprint via
  the ``blueprint_prior_kappa`` shrinkage, so they need no search refinement.  Its
  budget is a **GLOBAL** pooled iteration count (total across all W replicas), *split*
  among them — ``per_replica = max(min_per_replica, ceil(global / workers))`` — so more
  workers shorten the wall at ~constant total work, down to a per-replica *learning
  floor* (below which a starved replica would pollute the merged average, so the
  effective global rises to ``floor × workers``).  The merged average pools every
  replica's samples, so bounding *total* sampled work is what matters; the global count
  grows ~linearly with the live-player count (bigger hot path), **not** the exponential
  full-tree size.
  (Vector, being full-width, cannot be split this way — each replica needs the whole
  learning horizon — so it stays a per-replica constant.)

The regime split mirrors :func:`poker_ai.search.solver._select_regime` exactly.
"""

from __future__ import annotations

from math import ceil

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


def iteration_budget(ctx: SubgameContext, cfg: SolverConfig, workers: int = 1) -> int:
    """Per-replica iterations to run for the subgame ``ctx`` under ``cfg`` (§6.5).

    ``workers`` is the resolved replica count (:func:`resolve_workers`).  Returns
    ``cfg.max_iterations`` verbatim when ``cfg.auto_budget`` is off (the escape hatch
    for tests that pin an exact iteration count).  Otherwise returns the structural
    per-replica budget for the selected regime, clamped to at most
    ``cfg.max_iterations`` (the absolute per-replica ceiling) and at least 1:

    - **vector** — a per-stage constant, *independent of* ``workers`` (each full-width
      replica needs the whole learning horizon; more workers reduce variance, so total
      work grows with W);
    - **MCCFR** — a global pooled budget split across replicas, so per-replica shrinks
      as ``workers`` grows (total work ~constant, wall drops with W).
    """
    if not getattr(cfg, "auto_budget", True):
        return cfg.max_iterations

    w = max(1, int(workers))
    if _is_vector(ctx):
        # Per-stage constant, indexed (flop, turn, river) = street 1, 2, 3.  Per
        # replica — NOT divided by workers.
        flop, turn, river = cfg.vector_budget_by_street
        budget = {1: flop, 2: turn, 3: river}[ctx.street_at_root]
    else:
        # Global pooled budget (hot-path, ~linear in live players), split across the W
        # replicas → per-replica = ceil(global / workers), but never below the learning
        # floor (so a replica still learns properly at large W — the effective global
        # then rises to floor × W).  Both the per-player base and the floor are indexed
        # by ``street_at_root`` (0=preflop … 3=river): a deep multiway flop needs more
        # sampled work than a river.  Under per-hand parallelism (production eval: one
        # hand per core, search ``workers=1``) ``global / 1`` ≫ the floor, so the GLOBAL
        # budget binds; the floor binds only under within-search parallelism (W>1).
        n_live = max(2, len(ctx.ranges))
        street = ctx.street_at_root
        global_budget = cfg.mccfr_global_per_player_by_street[street] * n_live
        global_budget = max(cfg.mccfr_global_min,
                            min(cfg.mccfr_global_max, global_budget))
        floor = cfg.mccfr_min_per_replica_by_street[street]
        budget = max(floor, ceil(global_budget / w))

    return max(1, min(int(budget), int(cfg.max_iterations)))
