"""Structural per-subgame iteration budget (§6.5, :mod:`poker_ai.search.budget`).

The real-time search has **no online convergence test** (a real subgame is too large
for a single sampled replica to reach a tight equilibrium online).  The stop is a
per-subgame iteration count derived from the subgame's structure, and it is a
**per-replica** count for both regimes (production runs one replica per hand —
``workers=1`` — so the budget IS the work; extra replicas would only reduce variance):

- vector (HU turn/river) — full-width, so a per-stage constant (depth-driven);
- MCCFR (multiway / HU preflop / HU flop) — sampled, hot-path-driven: ``base * n_live``.

Both are clamped to the single ``max_iterations`` ceiling.  These lock in the model,
the ``solve`` wiring (``auto_budget``), and — the "verify with a test" the design
called for — that the vector river budget really does drive the average strategy to a
small best-response gap on the exact oracle.
"""

import numpy as np
import pytest

from poker_ai.search.budget import iteration_budget
from poker_ai.search.context import SubgameContext
from poker_ai.search.leaf import LeafConfig
from poker_ai.search.solver import solve
from poker_ai.search.solver_state import SolverConfig

from test.search._helpers import _ctx, _late_env, _policies


def _cfg(**kw):
    leaf = LeafConfig(policies=_policies(), n_rollouts=1)
    return SolverConfig(leaf=leaf, **kw)


def _ctx_for(street, n_players):
    leaf = LeafConfig(policies=_policies(), n_rollouts=1)
    ranges = {s: np.ones(4, np.float32) for s in range(n_players)}
    return SubgameContext(
        my_seat=0, my_hole=(0, 1), ranges=ranges, folded_ranges={},
        board_compatible=np.ones(4, bool), street_at_root=street,
        depth_limit=None, leaf=leaf, rng=np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# Model units
# --------------------------------------------------------------------------- #

# Only turn/river are vector now — a HU flop root has two future chance nodes left
# and routes to sampled MCCFR (see ``test_hu_flop_uses_mccfr_budget`` below).  Turn/river
# are the v4 calibration convergence points (turn 1350, river 850).
@pytest.mark.parametrize("street,expected", [(2, 1350), (3, 850)])
def test_vector_budget_is_per_stage_constant(street, expected):
    cfg = _cfg(auto_budget=True)
    assert iteration_budget(_ctx_for(street, 2), cfg) == expected


def test_hu_flop_uses_mccfr_budget():
    """HU flop is no longer vector: its budget is the MCCFR per-replica budget
    (base * n_live), not the vector per-stage constant (1500).  Mirrors
    ``_select_regime``."""
    cfg = _cfg(auto_budget=True, max_iterations=1_000_000)
    ctx = _ctx_for(1, 2)                                     # HU flop
    assert iteration_budget(ctx, cfg) == 12000               # flop base 6000 * 2 live


@pytest.mark.parametrize("n_players,expected", [(2, 6000), (3, 9000), (6, 18000)])
def test_mccfr_budget_scales_with_players(n_players, expected):
    # Per-replica MCCFR budget = base[street] * n_live.  Uses the preflop base (3000)
    # so the n_live scaling is clean and unclamped; the per-street *base* differentiation
    # is covered by ``test_mccfr_budget_is_per_street``.
    cfg = _cfg(auto_budget=True, max_iterations=1_000_000)
    assert iteration_budget(_ctx_for(0, n_players), cfg) == expected


def test_mccfr_clamped_to_max_iterations():
    """``max_iterations`` is the single ceiling: a deep multiway flop that would request
    more (flop base 6000 * 20 live = 120000) is clamped down to it."""
    cfg = _cfg(auto_budget=True, max_iterations=30_000)
    assert iteration_budget(_ctx_for(0, 2), cfg) == 6000           # 3000 * 2, unclamped
    assert iteration_budget(_ctx_for(1, 20), cfg) == 30_000        # 6000 * 20 → ceiling


def test_mccfr_budget_is_per_street():
    """The MCCFR base is indexed by ``street_at_root`` — a deep multiway flop gets more
    per-player sampled work than a river."""
    cfg = _cfg(
        auto_budget=True, max_iterations=1_000_000,
        mccfr_per_player_by_street=(3000, 9000, 3000, 3000),
    )
    assert iteration_budget(_ctx_for(1, 4), cfg) == 36000   # flop base 9000 * 4 live
    assert iteration_budget(_ctx_for(3, 4), cfg) == 12000   # river base 3000 * 4 live


def _ctx_with_models(street, n_players):
    """A ``_ctx_for`` carrying a (dummy) opponent model ⇒ the DBR path."""
    base = _ctx_for(street, n_players)
    return SubgameContext(
        my_seat=base.my_seat, my_hole=base.my_hole, ranges=base.ranges,
        folded_ranges={}, board_compatible=base.board_compatible,
        street_at_root=street, depth_limit=None, leaf=base.leaf,
        rng=np.random.default_rng(0), models={1: object()})


def test_dbr_budget_scale_is_models_only():
    """The ``dbr_mccfr_scale`` lifts the MCCFR budget ONLY when the subgame carries
    opponent models (DBR); vanilla (no models) is byte-for-byte the paper baseline."""
    cfg = _cfg(auto_budget=True, max_iterations=1_000_000, dbr_mccfr_scale=1.5)
    # HU flop: vanilla 6000*2 = 12000; DBR ×1.5 = 18000.
    assert iteration_budget(_ctx_for(1, 2), cfg) == 12000
    assert iteration_budget(_ctx_with_models(1, 2), cfg) == 18000
    # A scale of 1.0 (or the vanilla context) leaves the budget identical.
    cfg1 = _cfg(auto_budget=True, max_iterations=1_000_000, dbr_mccfr_scale=1.0)
    assert iteration_budget(_ctx_with_models(1, 2), cfg1) == 12000


def test_dbr_budget_scale_still_clamped_to_ceiling():
    """The DBR scale never escapes the single ``max_iterations`` ceiling: a 4-way flop
    (6000*4 = 24000, ×1.5 = 36000) caps at 30000."""
    cfg = _cfg(auto_budget=True, max_iterations=30_000, dbr_mccfr_scale=1.5)
    assert iteration_budget(_ctx_with_models(1, 4), cfg) == 30_000
    assert iteration_budget(_ctx_for(1, 4), cfg) == 24_000          # vanilla, unclamped


def test_auto_budget_off_returns_flat_max_iterations():
    cfg = _cfg(auto_budget=False, max_iterations=77)
    assert iteration_budget(_ctx_for(1, 2), cfg) == 77
    assert iteration_budget(_ctx_for(0, 6), cfg) == 77


def test_max_iterations_is_an_absolute_per_replica_ceiling():
    # Vector flop wants 1500 but the per-replica ceiling caps it.
    cfg = _cfg(auto_budget=True, max_iterations=300)
    assert iteration_budget(_ctx_for(1, 2), cfg) == 300


# --------------------------------------------------------------------------- #
# solve() wiring
# --------------------------------------------------------------------------- #


def test_solve_runs_the_structural_budget():
    """A HU river subgame (vector) runs the per-stage river budget on the one serial
    search (production runs one hand per core), so ``iterations_run`` == the budget."""
    env = _late_env(3)
    ctx = _ctx(env, seed=7)
    res = solve(env, ctx, _cfg(auto_budget=True, max_iterations=5000,
                               max_wall_seconds=1e9))
    assert res.regime == "vector"
    assert res.iterations_run == 850            # river budget
    assert res.stop_reason == "iteration_cap"


def test_solve_auto_budget_off_honours_flat_count():
    env = _late_env(3)
    ctx = _ctx(env, seed=7)
    res = solve(env, ctx, _cfg(auto_budget=False, max_iterations=120,
                               max_wall_seconds=1e9))
    assert res.iterations_run == 120


# --------------------------------------------------------------------------- #
# Verification: the vector budget actually converges (exact oracle)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_vector_river_budget_reaches_low_exploitability(_seeded):
    """At the calibrated river budget the average strategy is near-equilibrium."""
    from test.search.brute_force_cfr import build_subgame, exploitability
    from test.search.test_equilibrium_oracle import _river_subgame, _solver_sigma

    env, r0, r1, s0, s1 = _river_subgame(_seeded)
    sub = build_subgame(env, r0, r1, s0, s1)
    scale = max(abs(v) for leaf in sub.payoff.values() for v in leaf.values())

    ranges = {0: r0.astype(np.float32), 1: r1.astype(np.float32)}
    ctx = _ctx(env, ranges=ranges, seed=7)
    res = solve(env, ctx, _cfg(auto_budget=True, max_iterations=5000,
                               max_wall_seconds=1e9, discount_interval=100))
    assert res.regime == "vector" and res.iterations_run == 850

    expl = exploitability(sub, _solver_sigma(res.state, env, sub))
    assert expl < 0.03 * scale, f"river budget under-converged: expl={expl:.4f}"
