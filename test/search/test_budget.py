"""Per-subgame iteration budget (§6.5, :mod:`poker_ai.search.budget`).

The real-time search has **no online convergence test** (a real subgame is too large
for a single sampled replica to reach a tight equilibrium online).  The stop is an
explicit per-cell iteration count — one number per ``(approach, street, n_live)`` in
``solver_state.MCCFR_BUDGET`` / ``VECTOR_BUDGET`` — and it is a **per-replica** count
for both regimes (production runs one replica per hand, ``workers=1``, so the budget IS
the work; extra replicas only reduce variance).

What is pinned here is the *lookup behaviour*, not the calibrated numbers themselves —
those move every recalibration, and a test that copies them just has to be edited in
lockstep.  So: which table a solve reads (regime x approach), that the approach is
inferred from the solve's own inputs, that an uncovered cell degrades loudly instead of
crashing, that ``max_iterations`` still bounds everything, and — the "verify with a
test" the design called for — that the shipped vector river budget really does drive the
average strategy to a small best-response gap on the exact oracle.
"""

import numpy as np
import pytest

from poker_ai.search.budget import iteration_budget, search_approach
from poker_ai.search.context import SubgameContext
from poker_ai.search.leaf import LeafConfig
from poker_ai.search.solver import solve
from poker_ai.search.solver_state import (
    APPROACHES,
    DBR,
    OX,
    VANILLA,
    SolverConfig,
)

from test.search._helpers import _ctx, _late_env, _policies


def _cfg(**kw):
    leaf = LeafConfig(policies=_policies())
    return SolverConfig(leaf=leaf, **kw)


def _ctx_for(street, n_players):
    leaf = LeafConfig(policies=_policies())
    ranges = {s: np.ones(4, np.float32) for s in range(n_players)}
    return SubgameContext(
        my_seat=0, my_hole=(0, 1), ranges=ranges, folded_ranges={},
        board_compatible=np.ones(4, bool), street_at_root=street,
        depth_limit=None, leaf=leaf, rng=np.random.default_rng(0))


def _ctx_with_models(street, n_players):
    """A ``_ctx_for`` carrying a (dummy) opponent model ⇒ the DBR path."""
    base = _ctx_for(street, n_players)
    return SubgameContext(
        my_seat=base.my_seat, my_hole=base.my_hole, ranges=base.ranges,
        folded_ranges={}, board_compatible=base.board_compatible,
        street_at_root=street, depth_limit=None, leaf=base.leaf,
        rng=np.random.default_rng(0), models={1: object()})


# --------------------------------------------------------------------------- #
# Which approach a solve is — inferred, never declared
# --------------------------------------------------------------------------- #
def test_approach_is_inferred_from_the_solve_inputs():
    """Models ⇒ DBR, ``beta`` ⇒ OX, neither ⇒ vanilla.  Inferring it (rather than taking
    a caller's label) is what stops a solve being budgeted as one approach while
    configured as another."""
    plain = _cfg(auto_budget=True)
    assert search_approach(_ctx_for(1, 2), plain) == VANILLA
    assert search_approach(_ctx_with_models(1, 2), plain) == DBR
    assert search_approach(_ctx_for(3, 2), _cfg(auto_budget=True, beta=3.0)) == OX


def test_conflicting_inputs_prefer_dbr_and_warn(caplog):
    """Models AND beta together is a config bug (the runner rejects it), not a fourth
    approach — it must not silently pick one."""
    cfg = _cfg(auto_budget=True, beta=3.0)
    with caplog.at_level("WARNING"):
        assert search_approach(_ctx_with_models(1, 2), cfg) == DBR
    assert "mutually exclusive" in caplog.text


# --------------------------------------------------------------------------- #
# The lookup: regime picks the table, approach picks the row
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("street", [2, 3])
def test_vector_cells_read_the_vector_table(street):
    """HU turn/river route to vector, so the budget comes from the vector table."""
    cfg = _cfg(auto_budget=True, max_iterations=1_000_000)
    assert iteration_budget(_ctx_for(street, 2), cfg) == \
        cfg.vector_budget[VANILLA][(street, 2)]


def test_hu_flop_reads_the_mccfr_table():
    """A HU flop has two future chance nodes left and routes to sampled MCCFR, so it must
    NOT pick up the vector table's oracle-only flop entry.  Mirrors ``_select_regime``."""
    cfg = _cfg(auto_budget=True, max_iterations=1_000_000)
    ctx = _ctx_for(1, 2)
    assert iteration_budget(ctx, cfg) == cfg.mccfr_budget[VANILLA][(1, 2)]
    assert iteration_budget(ctx, cfg) != cfg.vector_budget[VANILLA][(1, 2)]


@pytest.mark.parametrize("street,n_live", [(1, 2), (1, 3), (2, 3), (3, 3)])
def test_dbr_reads_its_own_row_not_vanillas(street, n_live):
    """The DBR budget is an independent number per cell, not a transform of vanilla's —
    on some cells it is LOWER (its slower iteration hits the wall sooner) and on others
    HIGHER (it genuinely needs more).  A test asserting a fixed ratio would re-import the
    multiplier this design removed."""
    cfg = _cfg(auto_budget=True, max_iterations=1_000_000)
    got = iteration_budget(_ctx_with_models(street, n_live), cfg)
    assert got == cfg.mccfr_budget[DBR][(street, n_live)]


def test_ox_is_sized_like_dbr_in_vector_and_vanilla_in_mccfr():
    """OX's gadget root exists ONLY in the vector regime, so that is the only place it
    costs like a modelled solve.  In MCCFR ``beta`` is inert — ``solve`` warns such a
    search is "an ordinary best response, NOT adaptation-safe" — so an OX-labelled MCCFR
    solve is a vanilla solve and must not be handed DBR's wall for work it never does."""
    cfg = _cfg(auto_budget=True, max_iterations=1_000_000, beta=3.0)
    assert cfg.vector_budget[OX] == cfg.vector_budget[DBR]
    assert cfg.mccfr_budget[OX] == cfg.mccfr_budget[VANILLA]
    for street in (2, 3):                       # HU turn/river route to vector
        assert iteration_budget(_ctx_for(street, 2), cfg) == \
            cfg.vector_budget[DBR][(street, 2)]
    for cell in ((1, 2), (2, 3), (3, 3)):       # flop / multiway route to MCCFR
        assert iteration_budget(_ctx_for(*cell), cfg) == \
            cfg.mccfr_budget[VANILLA][cell]


def test_ox_and_dbr_vector_budgets_are_not_accidentally_vanillas():
    """The split only means something if DBR and vanilla actually differ in the vector
    regime — otherwise this test would pass on a table where OX fell through to vanilla."""
    cfg = _cfg()
    assert cfg.vector_budget[DBR] != cfg.vector_budget[VANILLA]
    assert cfg.mccfr_budget[DBR] != cfg.mccfr_budget[VANILLA]


def test_every_approach_covers_the_same_cells():
    """A cell present for one approach and missing for another would silently route that
    approach through the uncovered-cell fallback — a quiet under-budget on one arm only,
    which is exactly the kind of asymmetry that invalidates a cross-arm evaluation."""
    cfg = _cfg()
    for table in (cfg.mccfr_budget, cfg.vector_budget):
        cells = {a: set(table[a]) for a in APPROACHES}
        assert len(set(map(frozenset, cells.values()))) == 1, cells


# --------------------------------------------------------------------------- #
# Uncovered cells degrade loudly, they do not crash
# --------------------------------------------------------------------------- #
def test_uncovered_live_count_falls_back_and_warns(caplog):
    """A 6-player table reaches ``n_live`` the tables do not cover.  That must not
    KeyError mid-hand; it falls back to the largest covered live-count and logs, because
    the fallback is an UNDER-budget (the hot path grows with the live count), not a safe
    extrapolation."""
    cfg = _cfg(auto_budget=True, max_iterations=10_000_000)
    covered = max(nl for (st, nl) in cfg.mccfr_budget[VANILLA] if st == 1)
    with caplog.at_level("WARNING"):
        got = iteration_budget(_ctx_for(1, covered + 2), cfg)
    assert got == cfg.mccfr_budget[VANILLA][(1, covered)]
    assert "UNDER-budget" in caplog.text


def test_missing_street_is_an_error_not_a_guess():
    """No entry at all for a street is a table bug — fail loudly rather than invent one."""
    cfg = _cfg(auto_budget=True,
               mccfr_budget={a: {(1, 2): 100} for a in APPROACHES})
    with pytest.raises(KeyError, match="no mccfr budget for street 3"):
        iteration_budget(_ctx_for(3, 3), cfg)


# --------------------------------------------------------------------------- #
# The ceiling and the escape hatch
# --------------------------------------------------------------------------- #
def test_max_iterations_is_an_absolute_per_replica_ceiling():
    cfg = _cfg(auto_budget=True, max_iterations=300)
    assert iteration_budget(_ctx_for(1, 2), cfg) == 300
    assert iteration_budget(_ctx_with_models(1, 2), cfg) == 300


def test_shipped_budgets_all_fit_under_the_ceiling():
    """Every shipped number must be reachable — one silently clamped by ``max_iterations``
    would read as a calibrated budget while running a different, smaller one."""
    cfg = _cfg()
    for table in (cfg.mccfr_budget, cfg.vector_budget):
        for approach in APPROACHES:
            for cell, iters in table[approach].items():
                assert iters <= cfg.max_iterations, (approach, cell, iters)


def test_auto_budget_off_returns_flat_max_iterations():
    cfg = _cfg(auto_budget=False, max_iterations=77)
    assert iteration_budget(_ctx_for(1, 2), cfg) == 77
    assert iteration_budget(_ctx_for(0, 6), cfg) == 77


# --------------------------------------------------------------------------- #
# solve() wiring
# --------------------------------------------------------------------------- #
def test_solve_runs_the_structural_budget():
    """A HU river subgame (vector) runs its cell's budget on the one serial search
    (production runs one hand per core), so ``iterations_run`` == the budget."""
    env = _late_env(3)
    ctx = _ctx(env, seed=7)
    cfg = _cfg(auto_budget=True, max_iterations=5000, max_wall_seconds=1e9)
    res = solve(env, ctx, cfg)
    assert res.regime == "vector"
    assert res.iterations_run == cfg.vector_budget[VANILLA][(3, 2)]
    assert res.stop_reason == "iteration_cap"


def test_solve_auto_budget_off_honours_flat_count():
    env = _late_env(3)
    ctx = _ctx(env, seed=7)
    res = solve(env, ctx, _cfg(auto_budget=False, max_iterations=120,
                               max_wall_seconds=1e9))
    assert res.iterations_run == 120


# --------------------------------------------------------------------------- #
# Verification: the vector river budget actually converges (exact oracle)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_vector_river_budget_reaches_low_exploitability(_seeded):
    """At the shipped river budget the average strategy is near-equilibrium.  This is the
    one cell where that can be checked exactly — the river has no chance node left, so the
    brute-force oracle is tractable and the vector solve is deterministic."""
    from test.search.brute_force_cfr import build_subgame, exploitability
    from test.search.test_equilibrium_oracle import _river_subgame, _solver_sigma

    env, r0, r1, s0, s1 = _river_subgame(_seeded)
    sub = build_subgame(env, r0, r1, s0, s1)
    scale = max(abs(v) for leaf in sub.payoff.values() for v in leaf.values())

    ranges = {0: r0.astype(np.float32), 1: r1.astype(np.float32)}
    ctx = _ctx(env, ranges=ranges, seed=7)
    cfg = _cfg(auto_budget=True, max_iterations=5000, max_wall_seconds=1e9,
               discount_interval=100)
    res = solve(env, ctx, cfg)
    assert res.regime == "vector"
    assert res.iterations_run == cfg.vector_budget[VANILLA][(3, 2)]

    expl = exploitability(sub, _solver_sigma(res.state, env, sub))
    assert expl < 0.03 * scale, f"river budget under-converged: expl={expl:.4f}"
