"""A2 — the opponent-model solver clamp (opponent_modeling §5.1–5.3, §8.1–8.2).

The clamp blends a modeled seat's *realized* strategy toward its model,
``σ̃ = c·σ̂ + (1 − c)·x`` (Data Biased Response), at the one strategy-producing
seam both regimes share (:func:`poker_ai.search.vform.node_sigma` →
:func:`~poker_ai.search.vform.apply_model_clamp`).

Two gates:

1. **Baseline equivalence (load-bearing, §8.1)** — an empty ``ctx.models`` must be
   *bit-for-bit* the pre-change solver, in **both** regimes.  This is what keeps
   vanilla Pluribus byte-identical and makes condition B0 "A with no models"
   by construction, with no separate code path.
2. **Blend math (§8.2)** — ``c = 0`` returns the regret-matched σ exactly, ``c = 1``
   returns the model row, intermediate ``c`` interpolates, off-tree/overlay actions
   carry zero model mass, and the bot's frozen row never blends.
"""

import numpy as np
import pytest

from poker_ai.search.solver import solve
from poker_ai.search.vform import apply_model_clamp

from test.search._helpers import _ctx, _late_env, _preflop_env, _policies
from test.search.test_budget import _cfg


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class _FixedModel:
    """An ``OpponentModel`` returning a fixed row + constant confidence."""

    def __init__(self, row=None, c=1.0):
        self._row, self._c = row, c

    def strategy(self, state):
        n = len(state.legal_actions)
        if self._row is None:                      # point mass on the first action
            r = np.zeros(n, dtype=np.float32)
            r[0] = 1.0
            return r
        r = np.asarray(self._row, dtype=np.float32)[:n]
        return r / r.sum()

    def confidence(self, state):
        return self._c


class _Ctx:
    """Minimal ctx stand-in for the unit-level blend tests."""

    def __init__(self, models):
        self.models = models


class _State:
    def __init__(self):
        from poker_ai.search.solver_state import _CountingCache
        self.model_sigma_cache = _CountingCache()


def _digest(state):
    """Byte-level digest of everything a solve accumulates."""
    h = []
    for pk in sorted(state.vregret, key=lambda k: repr(k)):
        h.append(repr(pk).encode())
        h.append(np.asarray(state.vregret[pk]).tobytes())
        h.append(np.asarray(state.vstrat[pk]).tobytes())
    return b"".join(h)


# --------------------------------------------------------------------------- #
# 1. Baseline equivalence — the load-bearing gate, BOTH regimes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("env_fn,regime", [
    (lambda: _late_env(3), "vector"),     # HU river  → vector
    (_preflop_env, "mccfr"),              # HU preflop → MCCFR
])
def test_empty_models_is_bitwise_identical(env_fn, regime):
    """Empty ``ctx.models`` ⇒ byte-identical solver state (vanilla untouched)."""
    def run(models):
        env = env_fn()
        ctx = _ctx(env, seed=7)
        if models is not None:
            ctx = __import__("dataclasses").replace(ctx, models=models)
        res = solve(env, ctx, _cfg(auto_budget=False, max_iterations=40,
                                   max_wall_seconds=1e9, workers=1))
        assert res.regime == regime
        return _digest(res.state)

    assert run(None) == run({}), f"{regime}: empty models perturbed the solve"


@pytest.mark.parametrize("env_fn", [lambda: _late_env(3), _preflop_env])
def test_empty_models_leaves_cache_untouched(env_fn):
    """The clamp early-outs *before* the cache, so an unmodeled solve never
    reads or writes it — counters stay at zero."""
    env = env_fn()
    res = solve(env, _ctx(env, seed=7), _cfg(auto_budget=False, max_iterations=20,
                                             max_wall_seconds=1e9, workers=1))
    cache = res.state.model_sigma_cache
    assert len(cache) == 0
    assert cache.hits == 0 and cache.misses == 0


# --------------------------------------------------------------------------- #
# 2. Blend math
# --------------------------------------------------------------------------- #

def _sigma(n_combos=4, width=3):
    s = np.tile(np.arange(1.0, width + 1.0), (n_combos, 1))
    return s / s.sum(axis=1, keepdims=True)


def test_no_models_returns_sigma_object_unchanged():
    sigma = _sigma()
    out = apply_model_clamp(sigma, _Ctx({}), _State(), None, "pk", 1, 3, None)
    assert out is sigma          # identity, not just equality — no allocation


def test_unmodeled_actor_returns_sigma_unchanged():
    sigma = _sigma()
    ctx = _Ctx({0: _FixedModel()})
    out = apply_model_clamp(sigma, ctx, _State(), None, "pk", 1, 3, None)
    assert out is sigma          # seat 1 has no model


@pytest.mark.parametrize("c", [0.0, 0.25, 0.5, 1.0])
def test_blend_interpolates_between_free_and_model(c, monkeypatch):
    """``σ̃ = c·σ̂ + (1−c)·x`` exactly; c=0 ⇒ free strategy, c=1 ⇒ model row."""
    import poker_ai.search.vform as vform

    sigma = _sigma()
    n, width = sigma.shape
    model_row = np.zeros((n, width)); model_row[:, 0] = 1.0     # point mass, action 0
    monkeypatch.setattr(vform, "_model_rows",
                        lambda *a, **k: (model_row, np.full((n, 1), c)))

    ctx = _Ctx({1: _FixedModel(c=c)})
    out = apply_model_clamp(sigma, ctx, _State(), None, "pk", 1, width, None)

    np.testing.assert_allclose(out, c * model_row + (1.0 - c) * sigma)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)           # still a distribution
    if c == 0.0:
        np.testing.assert_array_equal(out, sigma)
    if c == 1.0:
        np.testing.assert_array_equal(out, model_row)


def test_model_rows_are_cached_per_seat_and_node(monkeypatch):
    """One build per ``(seat, public_key)``; revisits hit the cache."""
    import poker_ai.search.vform as vform

    sigma = _sigma()
    n, width = sigma.shape
    calls = []
    monkeypatch.setattr(vform, "_model_rows", lambda *a, **k: (
        calls.append(1), (np.zeros((n, width)), np.zeros((n, 1))))[1])

    ctx, state = _Ctx({1: _FixedModel()}), _State()
    for _ in range(3):
        apply_model_clamp(sigma, ctx, state, None, "pk", 1, width, None)
    apply_model_clamp(sigma, ctx, state, None, "other_pk", 1, width, None)

    # 4 queries over 2 distinct keys ⇒ 2 builds (misses) + 2 revisits (hits).
    assert len(calls) == 2                    # one build per distinct public key
    assert state.model_sigma_cache.misses == 2
    assert state.model_sigma_cache.hits == 2
    assert len(state.model_sigma_cache) == 2


def test_model_rows_zero_fill_overlay_columns():
    """A model row narrower than the node's legal width (an overlay/off-tree action
    was injected) is zero-filled by ``_model_rows``, so the blend can never invent
    mass on an injected size."""
    from poker_ai.search.vform import _model_rows

    class _FakeState:
        legal_actions = ("fold", "call", "raise:1.0")

    seen = {}

    class _FakeEnv:
        def policy_public_fields(self):
            return None

        def policy_state_for(self, combo, for_blueprint=False, public=None):
            seen["for_blueprint"] = for_blueprint
            return _FakeState()

    n, width = 3, 5          # node has 5 legal actions; model knows only 3
    m, c = _model_rows(_FixedModel(row=[0.5, 0.25, 0.25], c=0.7),
                       _FakeEnv(), width, np.zeros((n, 2), dtype=int))

    assert m.shape == (n, width) and c.shape == (n, 1)
    assert np.all(m[:, 3:] == 0.0)                       # overlay columns: no mass
    np.testing.assert_allclose(m[:, :3], np.tile([0.5, 0.25, 0.25], (n, 1)))
    np.testing.assert_allclose(c, 0.7)
    # The clamp canonicalises the history exactly as the blueprint read and the
    # §6.3 belief swap do, so all three query one and the same info-set key.
    assert seen["for_blueprint"] is True
