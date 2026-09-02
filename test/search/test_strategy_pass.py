"""The MCCFR root-street average-strategy pass (opponent actions enumerated).

External sampling samples the OPPONENT's action, so a strategy sum fused into the
regret walk accrues only on the sampled trajectory and carries an extra
``pi_{-i}(I)`` factor the CFR average does not want.  It cancels on normalisation
(constant across a row), but what survives is that iteration ``t`` is kept only with
probability ``pi_{-i}^t(I)`` — measured on a real-LUT heads-up flop root, the median
root-street node accrued on ~4% of its traverser's iterations.

``_MCCFRSolver.accumulate_strategy`` replaces that with a root-street pass that
branches every opponent action, which is Pluribus's Algorithm 1 ``UPDATE-STRATEGY``
fix and the one :mod:`poker_ai.blueprint.strategy` already applies to training.  The
vector regime needs none of it: its walk already enumerates opponent actions.
"""

import numpy as np
import pytest

from poker_ai.search.solver import solve
from poker_ai.search.solver_state import DEFAULT_STRATEGY_INTERVAL, SolverConfig
from test.search._helpers import _ctx, _late_env, _real_lut_env


def _flop_solve(**kw):
    env = _real_lut_env(1, stacks=(10000, 10000), seed=0)
    ctx = _ctx(env, seed=7)
    cfg = SolverConfig(leaf=ctx.leaf, max_iterations=600, auto_budget=False,
                       max_wall_seconds=600.0, discount_interval=100, **kw)
    return env, ctx, solve(env, ctx, cfg).state


def _root_nodes(state):
    return [pk for pk, sp in state.vrow_space.items() if sp == "combo"]


class TestCadenceResolution:

    def test_default_is_independent_of_the_discount(self):
        """Widening the discount must not silently starve the average.

        Measured on the river oracle gate, "follow the discount" at 2000 gave 25
        passes and trained exploitability 34.14 against a 30.0 tolerance; the fixed
        100 gives 215 passes and 2.57.  The two knobs size different things.
        """
        for discount in (20, 100, 2000, 0):
            cfg = SolverConfig(leaf=None, discount_interval=discount)
            assert cfg.resolved_strategy_interval() == DEFAULT_STRATEGY_INTERVAL

    def test_explicit_value_wins(self):
        cfg = SolverConfig(leaf=None, discount_interval=100, strategy_interval=25)
        assert cfg.resolved_strategy_interval() == 25

    def test_zero_disables(self):
        cfg = SolverConfig(leaf=None, discount_interval=100, strategy_interval=0)
        assert cfg.resolved_strategy_interval() == 0


@pytest.mark.slow
class TestMCCFRRootStreetPass:

    def test_shallow_nodes_cover_every_reachable_combo(self):
        """Where own reach is unmodified, EVERY board-compatible row must accumulate.

        A sampled accrual leaves holes here; an enumerated pass cannot.  Deeper nodes
        legitimately taper as the traverser's own strategy zeroes combos out — that is
        the CFR average being undefined at zero own reach, not a coverage gap.
        """
        env, ctx, state = _flop_solve()
        board_ok = np.asarray(ctx.board_compatible, dtype=np.float64) > 0.0
        n_reach = int(board_ok.sum())
        assert n_reach > 0

        nodes = _root_nodes(state)
        assert nodes, "expected root-street (combo-keyed) nodes"
        best = max(int(((state.vstrat[pk].sum(axis=1) > 0.0) & board_ok).sum())
                   for pk in nodes)
        assert best == n_reach, (
            f"no root-street node covers all {n_reach} board-compatible combos "
            f"(best {best}) — the pass is not enumerating"
        )

    def test_never_accumulates_on_a_board_incompatible_combo(self):
        env, ctx, state = _flop_solve()
        board_ok = np.asarray(ctx.board_compatible, dtype=np.float64) > 0.0
        for pk in _root_nodes(state):
            leaked = state.vstrat[pk].sum(axis=1)[~board_ok]
            assert not np.any(leaked > 0.0), f"strategy mass on an impossible combo at {pk}"

    def test_disabling_the_pass_leaves_root_rows_empty_but_regrets_intact(self):
        """Pins the no-double-count invariant.

        The fused walk must no longer write root-street ``strat`` — with the pass off,
        those rows stay empty.  ``vregret`` must be untouched by that suppression:
        ``write_strat=False`` skips only the average, never the regret update.
        """
        _env, _ctx_, state = _flop_solve(strategy_interval=0)
        nodes = _root_nodes(state)
        assert nodes
        strat_mass = sum(float(state.vstrat[pk].sum()) for pk in nodes)
        regret_mass = sum(float(np.abs(state.vregret[pk]).sum()) for pk in nodes)
        assert strat_mass == 0.0, (
            "root-street strategy accrued with the pass disabled — the fused walk is "
            "still writing it, so enabling the pass double-counts"
        )
        assert regret_mass > 0.0, "suppressing the average also suppressed regrets"


@pytest.mark.slow
class TestVectorRegimeNeedsNoPass:

    def test_fused_accrual_already_covers_every_root_row(self):
        """The vector walk enumerates opponent actions, so its fused accrual IS the average.

        That is why only MCCFR gets a pass.  Pinned by the property that makes it true:
        after a handful of iterations every board-compatible root row already carries
        strategy mass, with no separate pass anywhere.
        """
        from poker_ai.search.solver_state import SolverState
        from poker_ai.search.vector import _VectorSolver

        env = _late_env(2)                                  # heads-up turn root
        ctx = _ctx(env, seed=3)
        cfg = SolverConfig(leaf=ctx.leaf, max_iterations=50, auto_budget=False,
                           max_wall_seconds=60.0)
        solver = _VectorSolver(env, SolverState.empty(), ctx, cfg, ctx.rng)
        assert not hasattr(solver, "accumulate_strategy"), (
            "the vector regime must not carry a strategy pass — its walk already "
            "enumerates opponent actions"
        )
        for _ in range(20):
            solver.iterate()

        board_ok = np.asarray(ctx.board_compatible, dtype=np.float64) > 0.0
        root_pk = env.public_key
        mat = solver.state.vstrat[root_pk]
        covered = int(((mat.sum(axis=1) > 0.0) & board_ok).sum())
        assert covered == int(board_ok.sum())
