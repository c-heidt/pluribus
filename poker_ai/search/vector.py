"""Vector-form Linear CFR regime for the subgame solver (§6.5) — **seam only**.

The vector regime is the *small / late* path: **heads-up turn/river** subgames.
It is the designed-in counterpart to :mod:`mccfr` and is **not implemented yet**;
this module fixes the interface and records the design so the follow-up drops in
without touching the orchestrator, :class:`SolverState`, or :class:`SearchPolicy`.

Planned design (per §6.5):

- Carry a **per-combo reach vector per player** (``reach[p] = ctx.ranges[p]`` at
  the root), not a sampled concrete hand.
- **Expand every action at every decision node** (no action sampling); **sample
  one board runout per iteration** at the chance nodes, leaving the tree
  deterministic for that iteration.
- Regret / strategy-sum updates are **reach-weighted per combo** (float64), with
  opponent reach on combos conflicting with the acting combo zeroed (card
  removal, the ``poker_ai.search.ranges._zero_conflicting`` idiom).
- Terminals are **showdowns** (these subgames run to the end of the game),
  scored by the vectorised F2 showdown
  (:func:`poker_ai.search.showdown.showdown_values` /
  :func:`~poker_ai.search.showdown.showdown_cfv`) with ``stake`` derived from the
  env's matched contribution.  ``rank_combos_on_board`` is **reach-independent**,
  so the ≤~46 river rankings are precomputed once and reused across iterations
  rather than re-ranked per iteration.  Heads-up only — no opponent-opponent
  removal term, no side pots; ``env.payout`` is not used here.

These per-combo rows persist into the **same** :class:`SolverState`, keyed
``(public_key, combo_index)`` — the dict-of-rows shape already accommodates them —
so :class:`SearchPolicy` reads a vector-regime result with no changes.
"""

from __future__ import annotations

import numpy as np

from poker_ai.search.context import SubgameContext
from poker_ai.search.solver_state import SolverConfig, SolverState


class _VectorSolver:
    """Heads-up turn/river vector-form Linear CFR (not yet implemented).

    Constructed with the same ``(root_env, state, ctx, cfg, rng)`` surface as
    :class:`poker_ai.search.mccfr._MCCFRSolver` so :func:`solve` can dispatch on
    the selected regime uniformly.  :meth:`iterate` raises until the body lands.
    """

    def __init__(self, root_env, state: SolverState, ctx: SubgameContext,
                 cfg: SolverConfig, rng: np.random.Generator) -> None:
        self.root_env = root_env
        self.state = state
        self.ctx = ctx
        self.cfg = cfg
        self.rng = rng

    def iterate(self) -> None:
        raise NotImplementedError(
            "vector regime (heads-up turn/river) is a documented follow-up to "
            "the MCCFR path; see poker_ai/search/vector.py for the design."
        )
