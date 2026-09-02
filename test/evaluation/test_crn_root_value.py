"""Common-random-numbers root value for calibration (``calibrate.crn_root_value``).

The sweep's convergence metric is the hero root-EV gap.  For MCCFR the solver's raw
internal accumulator is a single-combo value against belief-*sampled* opponents; the
opponent/board sampling is the dominant noise in a ~100 bb subgame.  ``crn_root_value``
replaces it with an average of the hero's chip result of playing the solved σ over a
fixed set of card-worlds keyed to a ``seed`` — so two replicas that share the seed (the
sweep keys it to the ROOT, identical across replicas and budgets) evaluate the SAME
cards, and that sampling noise is common-mode and cancels in replica_spread.

These tests pin the mechanism: finite, deterministic per seed, and — the property that
makes the cancellation work — the value depends ONLY on the seed and the solved σ, never
on ambient/global RNG state.  The stub (uniform) policy can't show the variance
magnitude, only that the estimator is well-formed and CRN-safe.
"""

import collections
import copy
import dataclasses

import numpy as np

from evaluation.calibrate import crn_root_value, construct_roots
from evaluation.runner import EvalConfig, EvalSession
from poker_ai.search.leaf import LeafConfig
from poker_ai.search.mccfr import _BIAS_CLASSES
from poker_ai.search.policy import Policy
from poker_ai.search.solver import solve
from poker_ai.search.solver_state import SolverConfig
from test.lut_helpers import cluster_lut as _cluster_lut, install_cluster_lut


class _Uniform(Policy):
    def strategy(self, state, bias="none"):
        n = len(state.legal_actions)
        return np.full(n, 1.0 / n, np.float32) if n else np.array([], np.float32)


def _session(n_players=3):
    leaf = LeafConfig(policies={c: _Uniform() for c in _BIAS_CLASSES})
    scfg = SolverConfig(leaf=leaf, max_iterations=4, max_wall_seconds=30.0,
                        discount_interval=20)
    lut = _cluster_lut()
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
    v = crn_root_value(res, env, ctx, worlds=16, seed=1)
    assert v is not None and np.isfinite(v)


def test_deterministic_same_seed():
    res, env, ctx = _solved_mccfr_root(street=1)
    a = crn_root_value(res, env, ctx, worlds=16, seed=3)
    b = crn_root_value(res, env, ctx, worlds=16, seed=3)
    assert a == b


def test_different_seed_differs():
    res, env, ctx = _solved_mccfr_root(street=1)
    a = crn_root_value(res, env, ctx, worlds=16, seed=1)
    b = crn_root_value(res, env, ctx, worlds=16, seed=2)
    assert a != b


def test_ignores_ambient_randomness():
    # The CRN property: the estimate depends ONLY on the seed (and σ), never on ambient
    # global RNG state — which is exactly why sharing the seed across replicas cancels
    # the sampling noise.  Perturbing the global stream between calls changes nothing.
    res, env, ctx = _solved_mccfr_root(street=1)
    a = crn_root_value(res, env, ctx, worlds=16, seed=5)
    np.random.seed(999)
    _ = np.random.random(1000)
    b = crn_root_value(res, env, ctx, worlds=16, seed=5)
    assert a == b


def test_preserves_global_rng():
    # CRN safety: the estimate must not perturb the caller's global numpy stream.
    res, env, ctx = _solved_mccfr_root(street=1)
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    crn_root_value(res, env, ctx, worlds=8, seed=0)
    assert np.array_equal(np.random.get_state()[1], before)
