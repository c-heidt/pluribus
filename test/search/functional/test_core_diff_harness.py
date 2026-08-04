"""Self-tests for the search record/replay harness (:mod:`test.search.core_diff`).

Proves the differential machinery is sound: a vector pass run twice from an
identical baseline reproduces the ``SolverState`` matrices byte-for-byte (the
determinism the compiled-core differential relies on).  The scalar MCCFR
record/replay path retired with the traverser-vectorized walk (there is no longer
a separate regret/strategy two-pass to record); only the vector determinism
self-test remains.
"""

from poker_ai.search.solver_state import SolverConfig, SolverState
from poker_ai.search.vector import _VectorSolver

from test.search import core_diff
from test.search._helpers import _ctx, _late_env


def _cfg(leaf):
    return SolverConfig(
        leaf=leaf, max_iterations=200, max_wall_seconds=60.0,
        discount_interval=50,
    )


class TestVectorDeterministicPass:
    def _vector_baseline(self):
        env = _late_env(2, seed=0)
        ctx = _ctx(env, seed=1)
        return env, ctx

    def test_pass_deterministic_given_river(self):
        env, ctx = self._vector_baseline()
        s1 = _VectorSolver(env, SolverState.empty(), ctx, _cfg(ctx.leaf), ctx.rng)
        vr1, vs1 = core_diff.run_vector_pass(s1)
        assert vr1, "vector pass allocated no nodes — vacuous"

        s2 = _VectorSolver(env, SolverState.empty(), ctx, _cfg(ctx.leaf), ctx.rng)
        vr2, vs2 = core_diff.run_vector_pass(s2)
        core_diff.assert_tables_equal(vr1, vr2)
        core_diff.assert_tables_equal(vs1, vs2)
