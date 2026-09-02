"""VR-MCCFR variance reduction on the DBR MCCFR path (opponent_modeling §5.5).

Three levels:

- **Estimator (exact/statistical units)** — :func:`vr_baseline_estimate` is unbiased for
  ANY baseline, reduces variance with a good baseline, and reduces to the plain
  single-sample value when the baseline is zero.
- **Gate** — the solver's ``_vr`` flag is on ONLY when ``cfg.variance_reduction`` **and**
  ``ctx.models`` are both set (DBR-only), so a vanilla solve is never affected.
- **Vanilla byte-identity** — a no-model solve is bit-for-bit identical with the flag on
  or off (the paper baseline stays untouched, the load-bearing guarantee).
"""

import collections
import dataclasses

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv
from poker_ai.search.context import SubgameContext
from poker_ai.search.leaf import LeafConfig
from poker_ai.search.mccfr import _BIAS_CLASSES, _MCCFRSolver
from poker_ai.search.policy import Policy
from poker_ai.search.solver_state import SolverConfig, SolverState
from poker_ai.search.vform import vr_baseline_estimate
from test.abstraction_helpers import passive_action
from test.lut_helpers import install_cluster_lut


# --------------------------------------------------------------------------- #
# Estimator units — the correctness + benefit of the control variate.
# --------------------------------------------------------------------------- #

def _sigma(rng, n):
    s = rng.random(n) + 1e-3
    return s / s.sum()


def test_vr_estimate_zero_baseline_is_plain():
    """b = 0 ⇒ ``Σ_a σ(a)·0 + (v − 0) = v``: the plain single-sample value exactly, so
    VR can only ever *reduce* variance as the baseline learns (never adds any)."""
    rng = np.random.default_rng(0)
    n_a, n_c = 4, 6
    sigma = _sigma(rng, n_a)
    v_all = rng.standard_normal((n_a, n_c)) * 10.0
    b = np.zeros((n_a, n_c))
    for a in range(n_a):
        assert np.array_equal(vr_baseline_estimate(sigma, b, a, v_all[a]), v_all[a])


@pytest.mark.parametrize("baseline_kind", ["zero", "random", "perfect"])
def test_vr_estimate_is_unbiased(baseline_kind):
    """``E_{a*~σ}[Σ_a σ(a)·b(a) + (v(a*) − b(a*))] = Σ_a σ(a)·v(a)`` for ANY fixed
    baseline — the property that keeps DBR converging to the same best response."""
    rng = np.random.default_rng(1)
    n_a, n_c = 4, 5
    sigma = _sigma(rng, n_a)
    v_all = rng.standard_normal((n_a, n_c)) * 10.0
    true_ev = sigma @ v_all
    b = {
        "zero": np.zeros((n_a, n_c)),
        "random": rng.standard_normal((n_a, n_c)) * 5.0,
        "perfect": v_all.copy(),
    }[baseline_kind]

    N = 60_000
    acc = np.zeros(n_c)
    for _ in range(N):
        a = int(rng.choice(n_a, p=sigma))
        acc += vr_baseline_estimate(sigma, b, a, v_all[a])   # fixed b (no update)
    mean = acc / N
    # A perfect baseline is zero-variance ⇒ exact; else within the MC standard error.
    atol = 1e-9 if baseline_kind == "perfect" else 0.3
    assert np.allclose(mean, true_ev, atol=atol), f"{mean} vs {true_ev}"


def test_vr_reduces_variance_with_a_good_baseline():
    """A baseline close to the true action values collapses the estimator's variance —
    the whole point.  Perfect baseline ⇒ ~0 variance; plain ⇒ the full spread over a."""
    rng = np.random.default_rng(2)
    n_a, n_c = 4, 5
    sigma = _sigma(rng, n_a)
    v_all = rng.standard_normal((n_a, n_c)) * 10.0
    b_good = v_all + rng.standard_normal((n_a, n_c)) * 0.1   # baseline ≈ true values

    N = 20_000
    plain = np.empty((N, n_c))
    vr = np.empty((N, n_c))
    for t in range(N):
        a = int(rng.choice(n_a, p=sigma))
        plain[t] = v_all[a]                                   # plain single-sample
        vr[t] = vr_baseline_estimate(sigma, b_good, a, v_all[a])
    plain_var = plain.var(axis=0).mean()
    vr_var = vr.var(axis=0).mean()
    assert vr_var < 0.05 * plain_var, f"vr_var={vr_var:.3f} plain_var={plain_var:.3f}"


# --------------------------------------------------------------------------- #
# Solver fixture (3p turn root, stub LUT — mirrors test_traverser_vectorized_walk).
# --------------------------------------------------------------------------- #

class _UniformPolicy(Policy):
    def strategy(self, state, bias="none"):
        n = len(state.legal_actions)
        return np.full(n, 1.0 / n, np.float32) if n else np.array([], np.float32)


class _StubModel:
    """A minimal opponent model (point mass on the first action) — enough to make
    ``ctx.models`` non-empty so the DBR gate activates."""

    def strategy(self, state, bias="none"):
        n = len(state.legal_actions)
        r = np.zeros(n, np.float32)
        if n:
            r[0] = 1.0
        return r

    def confidence(self, state):
        return 1.0


def _turn_env_3p(seed, low=9, high=14, stacks=(2000, 2000, 2000)):
    np.random.seed(seed)
    env = PokerEnv(players=[Player(i, s) for i, s in enumerate(stacks)],
                   low_card_rank=low, high_card_rank=high)
    install_cluster_lut(env)
    g = 0
    while env.betting_round < 2 and not env.is_terminal and g < 80:
        env.step_in_place(passive_action(env))
        g += 1
    return env


def _ctx(env, seed):
    ranges = {s: np.ones(env.n_combos, np.float32) / env.n_combos for s in range(3)}
    leaf = LeafConfig(policies={c: _UniformPolicy() for c in _BIAS_CLASSES})
    return SubgameContext.from_runtime(
        env=env, my_seat=0,
        my_hole=tuple(int(c) for c in env.players[0].cards),
        ranges=ranges, folded_ranges={}, leaf=leaf,
        rng=np.random.default_rng(seed),
    )


def _cfg(ctx, *, vr, iters=30):
    return SolverConfig(leaf=ctx.leaf, max_iterations=iters, max_wall_seconds=60.0,
                        discount_interval=0, variance_reduction=vr)


def _digest(state):
    h = []
    for pk in sorted(state.vregret, key=lambda k: repr(k)):
        h.append(repr(pk).encode())
        h.append(np.asarray(state.vregret[pk]).tobytes())
        h.append(np.asarray(state.vstrat[pk]).tobytes())
    return b"".join(h)


def _solve_digest(seed, *, vr, models=None, iters=30):
    env = _turn_env_3p(seed)
    ctx = _ctx(env, seed)
    if models is not None:
        ctx = dataclasses.replace(ctx, models=models)
    state = SolverState.empty()
    solver = _MCCFRSolver(env, state, ctx, _cfg(ctx, vr=vr, iters=iters), ctx.rng)
    for _ in range(iters):
        solver.iterate()
    solver.restore_root()
    return solver, state


# --------------------------------------------------------------------------- #
# Gate — DBR-only activation.
# --------------------------------------------------------------------------- #

def test_vr_gated_on_flag_and_models():
    """``_vr`` is True ONLY with the flag AND opponent models — so vanilla (no models)
    is inert regardless of the flag."""
    env = _turn_env_3p(0)
    ctx = _ctx(env, 0)
    ctx_m = dataclasses.replace(ctx, models={1: _StubModel()})

    def vr_of(c, flag):
        s = _MCCFRSolver(env, SolverState.empty(), c, _cfg(c, vr=flag), c.rng)
        return s._vr

    assert vr_of(ctx, False) is False
    assert vr_of(ctx, True) is False      # flag on, NO models ⇒ still off (vanilla safe)
    assert vr_of(ctx_m, False) is False   # models, flag off ⇒ off
    assert vr_of(ctx_m, True) is True     # DBR + flag ⇒ on


# --------------------------------------------------------------------------- #
# Vanilla byte-identity — the paper baseline is untouched by the flag.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vanilla_solve_byte_identical_with_flag(seed):
    """No models ⇒ the VR flag is a no-op: bit-for-bit identical vregret/vstrat."""
    _, s_off = _solve_digest(seed, vr=False)
    _, s_on = _solve_digest(seed, vr=True)
    assert _digest(s_off) == _digest(s_on)
