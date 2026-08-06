"""Low-variance AIVAT root value for calibration (``calibrate.aivat_root_value``).

The sweep's convergence metric is the hero root-EV gap.  For MCCFR the solver's raw
internal accumulator is a high-variance single-combo MC estimate; ``aivat_root_value``
replaces it with a control-variate estimate of the SAME quantity — the hero's chip EV
playing the solved σ from the root vs its belief-sampled opponents, with an AIVAT term
at every action node and the exact runout chance-correction at all-in terminals.

These pin the mechanism (finite, deterministic per seed, uses the ctx belief), not the
variance magnitude — that needs a trained blueprint, which the stub (uniform) lacks.
Also covers the AIVAT ``max_runout_cards`` knob that lets the estimate cover a pre-flop
all-in (enhancement 2), while the played-game default (2) stays unchanged.
"""

import collections
import copy
import dataclasses

import numpy as np
import pytest

from evaluation.calibrate import aivat_root_value, construct_roots
from evaluation.runner import EvalConfig, EvalSession
from poker_ai.search.leaf import LeafConfig
from poker_ai.search.mccfr import _BIAS_CLASSES
from poker_ai.search.policy import Policy
from poker_ai.search.solver import solve
from poker_ai.search.solver_state import SolverConfig


class _Uniform(Policy):
    def strategy(self, state, bias="none"):
        n = len(state.legal_actions)
        return np.full(n, 1.0 / n, np.float32) if n else np.array([], np.float32)


def _session(n_players=3):
    leaf = LeafConfig(policies={c: _Uniform() for c in _BIAS_CLASSES}, n_rollouts=1)
    scfg = SolverConfig(leaf=leaf, max_iterations=4, max_wall_seconds=30.0,
                        discount_interval=20)
    lut = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    cfg = EvalConfig(run_id="t", run_seed=7, table_policy="all_blueprint", fixed_seats=None,
                     n_players=n_players, time_budget_hours=0.0, big_blind=100,
                     small_blind=50, starting_stack=600, low_card_rank=11, high_card_rank=14)
    return EvalSession(config=cfg, solver_cfg=scfg,
                       blueprint_policy=_Uniform(), card_info_lut=lut), cfg, scfg


def _solved_mccfr_root(street):
    session, cfg, scfg = _session(3)
    out = construct_roots(session, cfg, scfg, "vanilla", per_cell=1, run_seed=7)
    cell = next(c for c in out if c[1] == "mccfr" and c[2] == street and c[3] == 2)
    s = out[cell][0]
    env = copy.deepcopy(s.env)
    ctx = dataclasses.replace(s.ctx, rng=np.random.default_rng(0))
    solve_cfg = dataclasses.replace(scfg, auto_budget=False, max_iterations=300,
                                    max_wall_seconds=1e9)
    res = solve(env, ctx, solve_cfg)
    return res, env, ctx


def test_returns_finite_value():
    res, env, ctx = _solved_mccfr_root(street=1)          # flop mccfr HU
    v = aivat_root_value(res, env, ctx, rollouts=12, hole_samples=6,
                         rng=np.random.default_rng(1))
    assert v is not None and np.isfinite(v)


def test_deterministic_same_seed():
    res, env, ctx = _solved_mccfr_root(street=1)
    a = aivat_root_value(res, env, ctx, rollouts=12, hole_samples=6,
                         rng=np.random.default_rng(3))
    b = aivat_root_value(res, env, ctx, rollouts=12, hole_samples=6,
                         rng=np.random.default_rng(3))
    assert a == b


def test_different_seed_differs():
    res, env, ctx = _solved_mccfr_root(street=1)
    a = aivat_root_value(res, env, ctx, rollouts=12, hole_samples=6,
                         rng=np.random.default_rng(1))
    b = aivat_root_value(res, env, ctx, rollouts=12, hole_samples=6,
                         rng=np.random.default_rng(2))
    assert a != b


def test_preserves_global_rng():
    # CRN safety: the estimate must not perturb the caller's global numpy stream.
    res, env, ctx = _solved_mccfr_root(street=1)
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    aivat_root_value(res, env, ctx, rollouts=8, hole_samples=6,
                     rng=np.random.default_rng(0))
    assert np.array_equal(np.random.get_state()[1], before)


# --------------------------------------------------------------------------- #
# AIVAT runout-coverage knob (enhancement 2): max_runout_cards
# --------------------------------------------------------------------------- #

def test_aivat_max_runout_cards_default_unchanged():
    from evaluation.aivat import AivatAccumulator, _MAX_RUNOUT_CARDS
    acc = AivatAccumulator(0, object(), np.random.default_rng(0))
    assert acc._max_runout_cards == _MAX_RUNOUT_CARDS == 2   # played-game default

    class _T:
        is_decision_free = True
        terminal_board_len = 0                              # pre-flop all-in (5 to come)
    assert acc._cheap_runout(_T()) is False                 # skipped at default

    acc5 = AivatAccumulator(0, object(), np.random.default_rng(0), max_runout_cards=5)
    assert acc5._cheap_runout(_T()) is True                 # covered when raised
