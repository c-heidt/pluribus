"""The vector regime conditions the WHOLE pass on one sampled board completion.

:meth:`_VectorSolver.iterate` draws the runout once per iteration and masks both
root reaches by its feasibility, so a combo holding a completion card does not
exist in that sample anywhere in the tree.  Every terminal must agree with that
measure.  A showdown does automatically — the env ranks against the *complete*
board, so such a combo is board-incompatible.  A fold does **not**: the env masks
the fold path against the board the hand actually reached (four cards for a
turn-side fold), which is the right general-purpose semantics but leaves the
completion card unmasked.  Those rows carry zero reach in the current walk, so
nothing reads them today; the mask is what keeps that an invariant rather than a
coincidence.
"""

import numpy as np
import pytest

from poker_ai.search.solver_state import SolverConfig, SolverState
from poker_ai.search.vector import _VectorSolver
from test.search._helpers import _ctx, _late_env


def _turn_solver(seed=3):
    env = _late_env(2)                       # turn root: four community cards
    ctx = _ctx(env, seed=seed)
    cfg = SolverConfig(leaf=ctx.leaf, max_iterations=1, max_wall_seconds=5.0)
    return env, ctx, _VectorSolver(env, SolverState.empty(), ctx, cfg, ctx.rng)


def _pin_completion(solver, which=0):
    comp = (int(solver._cmaps.avail[which]),)
    solver._completion = comp
    solver._cmaps.refresh(comp)
    return comp


class TestFoldTerminalConditioning:

    def test_fold_terminal_zeroes_completion_holding_combos(self):
        """A turn-side fold must value a combo holding the sampled river at 0."""
        env, ctx, solver = _turn_solver()
        _pin_completion(solver)
        ff = solver._cmaps.feas_full
        assert ff is not None and (ff == 0.0).any(), "fixture must remove some combos"

        walk_env = solver._walk_env
        p = walk_env.player_i                 # the folder; its own combos index v
        pi = np.asarray(ctx.board_compatible, dtype=np.float64)
        token = walk_env.step_in_place("fold", settle_winners=False)
        try:
            v = solver._child(walk_env, p, pi.copy(), pi.copy(), parent_street=2)
        finally:
            walk_env.undo(token)

        # Meaningful: the fold does pay the surviving combos (else the mask is
        # trivially satisfied and the test proves nothing).
        assert np.any(v[ff > 0.0] != 0.0), "fold terminal returned an all-zero vector"
        infeasible = ff == 0.0
        assert np.all(v[infeasible] == 0.0), (
            "fold terminal priced combos holding the sampled river card: "
            f"{np.abs(v[infeasible]).max():.4f} on {int(infeasible.sum())} combos"
        )

    def test_showdown_terminal_already_conditioned(self):
        """The showdown path gets the same invariant from the env's board mask."""
        env, ctx, solver = _turn_solver()
        _pin_completion(solver)
        ff = solver._cmaps.feas_full

        walk_env = solver._walk_env
        p = walk_env.player_i
        pi = np.asarray(ctx.board_compatible, dtype=np.float64)
        token = walk_env.step_in_place("all_in", settle_winners=False)
        token2 = walk_env.step_in_place("call", settle_winners=False)
        try:
            v = solver._child(walk_env, p, pi.copy(), pi.copy(), parent_street=2)
        finally:
            walk_env.undo(token2)
            walk_env.undo(token)

        assert np.any(v[ff > 0.0] != 0.0), "showdown terminal returned an all-zero vector"
        assert np.all(v[ff == 0.0] == 0.0)
