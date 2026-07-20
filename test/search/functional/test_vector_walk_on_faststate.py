"""Phase-3c gate: the vector walk on the compiled ``FastState`` (via the adapter)
produces ``vregret`` / ``vstrat`` **byte-identical** to the ``PokerEnv`` walk.

The Python ``_VectorSolver._walk`` is unchanged; only the env it walks differs
(a ``PokerEnv`` root vs a :class:`~poker_ai.search.fast_env.FastEnvAdapter` over a
``FastState``).  Driving both from the same empty state with the **same** river
draw (a fresh identically-seeded RNG each) must yield identical tables to the last
ULP — the make/undo engine + public_key keying + terminal ``vector_payout`` all
agree.  Complements the golden-digest gate (run separately under the flag).

Skips cleanly when the compiled core is not built.
"""

import numpy as np
import pytest

from poker_ai import _core

pytestmark = pytest.mark.skipif(
    not _core.CORE_AVAILABLE, reason="compiled core extension not built"
)

if _core.CORE_AVAILABLE:
    from poker_ai.search.fast_env import FastEnvAdapter, build_fast_walk_env
    from poker_ai.search.solver_state import SolverConfig, SolverState
    from poker_ai.search.vector import _VectorSolver

    from test.search import core_diff
    from test.search._helpers import _ctx, _late_env


def _cfg(leaf):
    return SolverConfig(
        leaf=leaf, max_iterations=200, max_wall_seconds=60.0,
        discount_interval=50, workers=1,
    )


def _run(env, ctx, *, use_core, seed, iters):
    """Run ``iters`` vector iterations from empty state; return the SolverState.

    A fresh identically-seeded RNG makes the per-iteration river draw identical
    across the two runs, so the passes are deterministic and directly comparable.
    """
    state = SolverState.empty()
    rng = np.random.default_rng(seed)
    solver = _VectorSolver(env, state, ctx, _cfg(ctx.leaf), rng)
    if use_core:
        adapter = build_fast_walk_env(env)
        assert adapter is not None, "FastState adapter unexpectedly unavailable"
        assert isinstance(adapter, FastEnvAdapter)
        solver._walk_env = adapter
    else:
        solver._walk_env = env  # pin the PokerEnv walk regardless of env var
    for _ in range(iters):
        solver.iterate()
    return state


@pytest.mark.parametrize(
    "target_round,stacks",
    [(2, (200, 200)), (2, (150, 400)), (3, (200, 200)), (3, (300, 120))],
)
def test_vector_walk_byte_identical(target_round, stacks):
    """Turn (with river chance) and river subgames, equal + unequal stacks: the
    FastState-driven walk matches the PokerEnv walk table-for-table."""
    env = _late_env(target_round, stacks=stacks, seed=0)
    ctx = _ctx(env, seed=1)

    py = _run(env, ctx, use_core=False, seed=7, iters=40)
    core = _run(env, ctx, use_core=True, seed=7, iters=40)

    assert py.vregret, "vector pass allocated no nodes — vacuous"
    core_diff.assert_tables_equal(
        core_diff.snapshot(py.vregret), core_diff.snapshot(core.vregret)
    )
    core_diff.assert_tables_equal(
        core_diff.snapshot(py.vstrat), core_diff.snapshot(core.vstrat)
    )


def test_adapter_falls_back_when_overlay_present():
    """An injected off-tree action makes the histories un-representable in the
    byte-code engine — ``build_fast_walk_env`` must return ``None`` (→ Python walk),
    never a FastState that would mis-key the node."""
    env = _late_env(2, stacks=(200, 200), seed=0)
    assert build_fast_walk_env(env) is not None  # clean root → adapter
    # Simulate a re-search that injected an off-tree raise somewhere in the tree
    # (a non-empty overlay is exactly the guard's trigger; the byte-code engine
    # cannot represent the off-tree token, so the walk must stay in Python).
    env._extra_legal_actions[env.public_key] = frozenset({"raise:1.5"})
    assert build_fast_walk_env(env) is None      # overlay non-empty → fall back
