"""Structural per-subgame iteration budget (§6.5, :mod:`poker_ai.search.budget`).

The real-time search has **no online convergence test** (a real subgame is too large
for a single sampled replica to reach a tight equilibrium online).  The stop is a
per-subgame iteration count derived from the subgame's structure:

- vector (HU flop/turn/river) — full-width, so a per-stage constant (depth-driven);
- MCCFR (multiway / HU preflop) — sampled, hot-path-driven, ~linear in live players.

These lock in the model, the clamps, the ``solve`` wiring (``auto_budget``), and —
the "verify with a test" the design called for — that the vector river budget really
does drive the average strategy to a small best-response gap on the exact oracle.
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
# and routes to sampled MCCFR (see ``test_hu_flop_uses_mccfr_budget`` below).
@pytest.mark.parametrize("street,expected", [(2, 1000), (3, 500)])
def test_vector_budget_is_per_stage_constant(street, expected):
    cfg = _cfg(auto_budget=True)
    assert iteration_budget(_ctx_for(street, 2), cfg) == expected


def test_hu_flop_uses_mccfr_budget():
    """HU flop is no longer vector: its budget is the MCCFR global pool (÷ workers),
    not the vector per-stage constant (1500).  Mirrors ``_select_regime``."""
    cfg = _cfg(auto_budget=True, max_iterations=1_000_000)
    ctx = _ctx_for(1, 2)                                    # HU flop
    assert iteration_budget(ctx, cfg, workers=1) == 6000   # global = 3000 * 2 live
    assert iteration_budget(ctx, cfg, workers=6) == 1000   # ceil(6000 / 6), split by W


def test_vector_budget_is_independent_of_workers():
    """Full-width vector: each replica needs the whole horizon → not divided by W."""
    cfg = _cfg(auto_budget=True)
    for w in (1, 6, 63):
        assert iteration_budget(_ctx_for(3, 2), cfg, workers=w) == 500  # river


@pytest.mark.parametrize("n_players,expected_global", [(2, 6000), (3, 9000), (6, 18000)])
def test_mccfr_global_budget_scales_with_players(n_players, expected_global):
    # workers=1 → per-replica == the global pooled budget (= 3000 * n_live, clamped).
    cfg = _cfg(auto_budget=True, max_iterations=1_000_000)
    street = 0 if n_players == 2 else 1
    assert iteration_budget(_ctx_for(street, n_players), cfg, workers=1) == expected_global


def test_mccfr_global_budget_splits_across_workers():
    """MCCFR global budget is divided among replicas → per-replica shrinks with W,
    total pooled work (~global) stays constant → wall drops with more workers, while
    the plain division stays above the learning floor."""
    cfg = _cfg(auto_budget=True, max_iterations=1_000_000,
               mccfr_min_per_replica_by_street=(750, 750, 750, 750))
    ctx = _ctx_for(0, 2)  # HU pre-flop, global = 6000
    assert iteration_budget(ctx, cfg, workers=1) == 6000
    assert iteration_budget(ctx, cfg, workers=4) == 1500   # ceil(6000/4)
    assert iteration_budget(ctx, cfg, workers=6) == 1000   # ceil(6000/6)
    # Pooled total (per-replica × W) stays ~constant at the global budget while the
    # per-replica count is above the floor (750).
    for w in (1, 3, 4, 6, 8):
        assert 6000 <= iteration_budget(ctx, cfg, workers=w) * w < 6000 + w


def test_mccfr_learning_floor_kicks_in_at_high_worker_counts():
    """At large W the plain division would starve a replica, so it floors — each
    replica still learns properly and the effective global rises to floor × W."""
    cfg = _cfg(auto_budget=True, max_iterations=1_000_000,
               mccfr_min_per_replica_by_street=(750, 750, 750, 750))
    ctx = _ctx_for(0, 2)  # global = 6000; 6000/W < 750 once W > 8
    assert iteration_budget(ctx, cfg, workers=16) == 750   # floored, not ceil(6000/16)=375
    assert iteration_budget(ctx, cfg, workers=63) == 750   # still the floor
    # Effective global (pooled) has risen to floor × W (each worker learns properly).
    assert iteration_budget(ctx, cfg, workers=63) * 63 == 750 * 63


def test_mccfr_global_clamped_to_min_and_max():
    cfg = _cfg(auto_budget=True, max_iterations=1_000_000)
    # 2 players * 3000 = 6000 == the floor; 20 players * 3000 = 60000 → max 30000.
    assert iteration_budget(_ctx_for(0, 2), cfg, workers=1) == 6000
    assert iteration_budget(_ctx_for(1, 20), cfg, workers=1) == 30000


def test_mccfr_budget_is_per_street():
    """The MCCFR global base and learning floor are indexed by ``street_at_root``.

    At a large worker count the per-street **floor** is what binds (global/W falls
    below it), so distinct per-street floors yield distinct per-replica budgets —
    the dial the cluster calibration actually sets.
    """
    cfg = _cfg(
        auto_budget=True, max_iterations=1_000_000,
        # flop (street 1) gets a big floor, river (street 3) a small one.
        mccfr_min_per_replica_by_street=(750, 1200, 750, 400),
    )
    # W=63: global/63 << every floor, so the per-street floor governs directly.
    assert iteration_budget(_ctx_for(1, 4), cfg, workers=63) == 1200   # flop floor
    assert iteration_budget(_ctx_for(3, 3), cfg, workers=63) == 400    # river floor
    # Per-street global base also independently differentiates at low W (floor inert).
    cfg2 = _cfg(
        auto_budget=True, max_iterations=1_000_000, mccfr_global_max=1_000_000,
        mccfr_global_per_player_by_street=(3000, 9000, 3000, 3000),
    )
    assert iteration_budget(_ctx_for(1, 4), cfg2, workers=1) == 36000  # 9000 * 4 live
    assert iteration_budget(_ctx_for(3, 4), cfg2, workers=1) == 12000  # 3000 * 4 live


def test_auto_budget_off_returns_flat_max_iterations():
    cfg = _cfg(auto_budget=False, max_iterations=77)
    assert iteration_budget(_ctx_for(1, 2), cfg, workers=6) == 77
    assert iteration_budget(_ctx_for(0, 6), cfg, workers=6) == 77


def test_max_iterations_is_an_absolute_per_replica_ceiling():
    # Vector flop wants 1500 but the per-replica ceiling caps it.
    cfg = _cfg(auto_budget=True, max_iterations=300)
    assert iteration_budget(_ctx_for(1, 2), cfg, workers=6) == 300


# --------------------------------------------------------------------------- #
# solve() wiring
# --------------------------------------------------------------------------- #

_WORKERS = 6  # this machine's sane replica count (LMDB-fork safe with UniformPolicy)


def test_solve_runs_the_structural_budget():
    """A HU river subgame (vector): each of W replicas runs the per-stage river budget.

    ``iterations_run`` is the pooled sum across replicas (§6.7), so it is the
    per-replica budget times the worker count — confirming the structural budget is
    applied on the production parallel path, not just the serial one.
    """
    env = _late_env(3)
    ctx = _ctx(env, seed=7)
    res = solve(env, ctx, _cfg(auto_budget=True, max_iterations=5000,
                               max_wall_seconds=1e9, workers=_WORKERS))
    assert res.regime == "vector"
    assert res.iterations_run == 500 * _WORKERS  # river budget (500) per replica
    assert res.stop_reason == "iteration_cap"


def test_solve_auto_budget_off_honours_flat_count():
    env = _late_env(3)
    ctx = _ctx(env, seed=7)
    res = solve(env, ctx, _cfg(auto_budget=False, max_iterations=120,
                               max_wall_seconds=1e9, workers=_WORKERS))
    assert res.iterations_run == 120 * _WORKERS


def test_solve_mccfr_pooled_budget_constant_across_workers():
    """HU pre-flop (MCCFR): the global pooled budget is split across replicas, so the
    total iterations run is ~constant in W (wall drops, work doesn't)."""
    from test.search._helpers import _preflop_env
    # Small global budget keeps the test fast; low learning floor so the plain
    # division (not the floor) governs at these worker counts.
    kw = dict(auto_budget=True, max_iterations=1_000_000, max_wall_seconds=1e9,
              mccfr_global_per_player_by_street=(300, 300, 300, 300),
              mccfr_global_min=600, mccfr_global_max=6000,
              mccfr_min_per_replica_by_street=(50, 50, 50, 50))

    def run(workers):
        env = _preflop_env()
        res = solve(env, _ctx(env, seed=7), _cfg(workers=workers, **kw))
        assert res.regime == "mccfr"
        return res.iterations_run

    # global = max(600, 300*2) = 600; pooled = ceil(600/W)*W ≈ 600 for every W.
    assert run(2) == 600            # ceil(600/2)*2
    assert run(3) == 600            # ceil(600/3)*3
    assert run(4) == 600            # ceil(600/4)*4 = 150*4


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
                               max_wall_seconds=1e9, discount_interval=100,
                               workers=_WORKERS))
    assert res.regime == "vector" and res.iterations_run == 500 * _WORKERS

    expl = exploitability(sub, _solver_sigma(res.state, env, sub))
    assert expl < 0.03 * scale, f"river budget under-converged: expl={expl:.4f}"
