"""Self-tests for the search record/replay harness (:mod:`test.search.core_diff`).

Proves the Phase-0 differential machinery is sound *before* Phases 3-4 rely on it:
a Python walk recorded and then replayed into a second Python walk must reproduce
the ``SolverState`` tables byte-for-byte, the replay must be fully consumed, and a
truncated replay must fail loudly (the divergence detector).  The compiled-core
replay plugs into these same helpers in Phases 3-4 (``core`` in place of the second
Python walk).
"""

import numpy as np
import pytest

from poker_ai.search.mccfr import _MCCFRSolver
from poker_ai.search.solver_state import SolverConfig, SolverState
from poker_ai.search.vector import _VectorSolver

from test.search import core_diff
from test.search.test_solver import _ctx, _flop_env, _late_env


def _clone_state(src: SolverState) -> SolverState:
    """Fresh ``SolverState`` carrying copies of ``src``'s tables + node structure.

    Explicit (rather than ``deepcopy``) so record and replay start from an
    identical baseline without dragging the un-needed ``_CountingCache`` fields.
    """
    dst = SolverState.empty()
    dst.regret = {k: v.copy() for k, v in src.regret.items()}
    dst.strat_sum = {k: v.copy() for k, v in src.strat_sum.items()}
    dst.legal_at = dict(src.legal_at)
    dst.actor_at = dict(src.actor_at)
    dst.frozen = {k: v.copy() for k, v in src.frozen.items()}
    dst.vregret = {k: v.copy() for k, v in src.vregret.items()}
    dst.vstrat = {k: v.copy() for k, v in src.vstrat.items()}
    return dst


def _cfg(leaf):
    return SolverConfig(
        leaf=leaf, max_iterations=200, max_wall_seconds=60.0,
        discount_interval=50, workers=1,
    )


def _mccfr_baseline():
    """A lightly pre-trained MCCFR state + a fixed disjoint ``holes`` for the diff.

    Pre-training makes the regret-matched sigmas non-uniform, so the recorded
    opponent choices track a real strategy (not the degenerate uniform walk).
    """
    env = _flop_env(seed=0)
    ctx = _ctx(env, seed=1)
    solver = _MCCFRSolver(env, SolverState.empty(), ctx, _cfg(ctx.leaf), ctx.rng)
    for _ in range(30):
        solver.iterate()
    holes = solver._sample_root_holes()
    return env, ctx, solver.state, holes


def _mccfr_solver(env, ctx, state):
    return _MCCFRSolver(env, state, ctx, _cfg(ctx.leaf), ctx.rng)


class TestMCCFRRecordReplay:
    def test_regret_pass_byte_identical(self):
        env, ctx, base, holes = _mccfr_baseline()
        i = 0
        rec = _mccfr_solver(env, ctx, _clone_state(base))
        snap_rec, choices = core_diff.run_traverse_recording(rec, env, i, holes, seed=3)
        assert choices, "regret pass recorded no opponent decisions — vacuous"

        rep = _mccfr_solver(env, ctx, _clone_state(base))
        snap_rep = core_diff.run_traverse_replay(rep, env, i, holes, choices)
        core_diff.assert_tables_equal(snap_rec, snap_rep)

    def test_regret_pass_mutates_state(self):
        """Non-vacuity: the pass actually writes regret rows (not a no-op diff)."""
        env, ctx, base, holes = _mccfr_baseline()
        before = {k: v.copy() for k, v in base.regret.items()}
        rec = _mccfr_solver(env, ctx, _clone_state(base))
        snap_rec, _ = core_diff.run_traverse_recording(rec, env, 0, holes, seed=3)
        # Some row changed relative to the baseline (regret accumulated).
        changed = any(
            k not in before or not np.array_equal(before[k], snap_rec[k])
            for k in snap_rec
        )
        assert changed

    def test_strategy_pass_byte_identical(self):
        env, ctx, base, holes = _mccfr_baseline()
        i = 1
        rec = _mccfr_solver(env, ctx, _clone_state(base))
        snap_rec, choices = core_diff.run_strategy_recording(rec, env, i, holes, seed=5)
        assert choices, "strategy pass recorded no decisions — vacuous"

        rep = _mccfr_solver(env, ctx, _clone_state(base))
        snap_rep = core_diff.run_strategy_replay(rep, env, i, holes, choices)
        core_diff.assert_tables_equal(snap_rec, snap_rep)

    def test_truncated_replay_raises(self):
        """A short replay is detected (the walk visits more nodes than recorded)."""
        env, ctx, base, holes = _mccfr_baseline()
        rec = _mccfr_solver(env, ctx, _clone_state(base))
        _, choices = core_diff.run_traverse_recording(rec, env, 0, holes, seed=3)
        assert len(choices) >= 1
        rep = _mccfr_solver(env, ctx, _clone_state(base))
        with pytest.raises(AssertionError):
            core_diff.run_traverse_replay(rep, env, 0, holes, choices[:-1])


class TestVectorDeterministicPass:
    def _vector_baseline(self):
        env = _late_env(2, seed=0)
        ctx = _ctx(env, seed=1)
        return env, ctx

    def test_pass_deterministic_given_river(self):
        env, ctx = self._vector_baseline()
        s1 = _VectorSolver(env, SolverState.empty(), ctx, _cfg(ctx.leaf), ctx.rng)
        vr1, vs1 = core_diff.run_vector_pass(s1, sampled_k=0)
        assert vr1, "vector pass allocated no nodes — vacuous"

        s2 = _VectorSolver(env, SolverState.empty(), ctx, _cfg(ctx.leaf), ctx.rng)
        vr2, vs2 = core_diff.run_vector_pass(s2, sampled_k=0)
        core_diff.assert_tables_equal(vr1, vr2)
        core_diff.assert_tables_equal(vs1, vs2)
