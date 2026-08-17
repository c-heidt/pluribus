"""Mid-search snapshots (``solve(snapshot_at=..., on_snapshot=...)``).

The calibration sweeps a *ladder* of iteration budgets per cell.  It used to run one
independent solve per rung, which re-walked the same trajectory ~3.8x over: a solve's
trajectory depends only on ``(env, seeded ctx.rng, cfg)`` and NOT on ``max_iterations``,
so the strategy after ``t`` iterations of a longer search is the same object a solve
capped at ``t`` returns.  These pin exactly that equivalence — it is the correctness
claim the whole optimisation rests on — plus the hook's inertness when unused.
"""

import dataclasses

import numpy as np
import pytest

from poker_ai.search.leaf import LeafConfig
from poker_ai.search.solver import solve
from poker_ai.search.solver_state import SolverConfig

from test.search._helpers import _ctx, _late_env, _policies


def _cfg(iters, **kw):
    leaf = LeafConfig(policies=_policies())
    return SolverConfig(leaf=leaf, max_iterations=iters, max_wall_seconds=1e9,
                        auto_budget=False, **kw)


def _root_row(res, env):
    """The hero's root average-strategy row from a finished solve."""
    pk = env.public_key
    legal = [a for a in env.legal_actions if a is not None]
    hr = env.combo_index[tuple(sorted(int(c) for c in env.players[0].cards))]
    if pk not in res.state.legal_at:
        return None
    return np.asarray(res.average_policy.strategy_for(pk, hr, legal), dtype=np.float64)


def _snapshots(env, seed, rungs, **cfgkw):
    """``{t: sigma}`` captured mid-search from ONE solve to ``max(rungs)``."""
    pk = env.public_key
    legal = [a for a in env.legal_actions if a is not None]
    hr = env.combo_index[tuple(sorted(int(c) for c in env.players[0].cards))]
    out = {}

    def grab(t, avg, elapsed):
        if pk in avg._state.legal_at:
            out[int(t)] = np.array(avg.strategy_for(pk, hr, legal), dtype=np.float64)

    res = solve(env, _ctx(env, seed=seed), _cfg(max(rungs), **cfgkw),
                snapshot_at=rungs, on_snapshot=grab)
    return out, res


# --------------------------------------------------------------------------- #
# The equivalence: snapshot at t == a separate solve capped at t
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("street,regime", [(3, "vector"), (1, "mccfr")])
def test_snapshot_matches_independent_solve(street, regime):
    rungs = [7, 11, 17, 25]
    snaps, _res = _snapshots(_late_env(street), 5, rungs)
    assert set(snaps) == set(rungs), "a rung was not snapshotted"
    for t in rungs:
        env = _late_env(street)          # a pristine root, as the sweep deepcopies per solve
        ref = solve(env, _ctx(env, seed=5), _cfg(t), regime_override=regime)
        got = _root_row(ref, env)
        assert got is not None
        np.testing.assert_allclose(
            snaps[t], got, rtol=0, atol=0,
            err_msg=f"snapshot at {t} differs from a solve capped at {t} ({regime})")


def test_snapshot_at_top_matches_the_solves_own_result():
    rungs = [9, 21]
    env = _late_env(3)
    snaps, res = _snapshots(env, 3, rungs)
    np.testing.assert_allclose(snaps[max(rungs)], _root_row(res, env), rtol=0, atol=0)


def test_snapshots_differ_across_rungs():
    """Sanity: the strategy really is still moving, so the rungs are not trivially equal."""
    snaps, _ = _snapshots(_late_env(1), 7, [5, 40])
    assert not np.allclose(snaps[5], snaps[40]), "rungs identical — nothing to calibrate"


# --------------------------------------------------------------------------- #
# Inertness when unused (production path must be untouched)
# --------------------------------------------------------------------------- #
def test_no_hook_is_byte_identical():
    a_env = _late_env(3)
    a = solve(a_env, _ctx(a_env, seed=11), _cfg(15))
    b_env = _late_env(3)
    b = solve(b_env, _ctx(b_env, seed=11), _cfg(15),
              snapshot_at=None, on_snapshot=None)
    np.testing.assert_allclose(_root_row(a, a_env), _root_row(b, b_env), rtol=0, atol=0)
    assert a.iterations_run == b.iterations_run


def test_hook_does_not_change_the_result():
    """Snapshotting must not perturb the search it observes (no RNG draw, no mutation)."""
    plain_env = _late_env(1)
    plain = solve(plain_env, _ctx(plain_env, seed=4), _cfg(30))
    snaps, withhook = _snapshots(_late_env(1), 4, [10, 30])
    np.testing.assert_allclose(_root_row(withhook, _late_env(1)),
                               _root_row(plain, plain_env), rtol=0, atol=0)


def test_rungs_outside_the_run_are_simply_absent():
    """A requested rung above ``max_iterations`` is never reached, so it yields no row
    (the sweep then has no reference for that rung — it does not fabricate one)."""
    env = _late_env(3)
    pk = env.public_key
    legal = [a for a in env.legal_actions if a is not None]
    hr = env.combo_index[tuple(sorted(int(c) for c in env.players[0].cards))]
    seen = {}

    def grab(t, avg, elapsed):
        seen[int(t)] = True

    res = solve(env, _ctx(env, seed=2), _cfg(10),          # cap BELOW the 999 rung
                snapshot_at=[5, 10, 999], on_snapshot=grab)
    assert set(seen) == {5, 10}
    assert res.iterations_run == 10
