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

from test.search._helpers import _ctx, _policies, _real_lut_env
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

@pytest.mark.requires_lut
@pytest.mark.parametrize("env_fn,regime", [
    (lambda: _real_lut_env(3), "vector"),     # HU river  → vector
    (lambda: _real_lut_env(0), "mccfr"),      # HU preflop → MCCFR
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


@pytest.mark.requires_lut
@pytest.mark.parametrize("env_fn", [lambda: _real_lut_env(3), lambda: _real_lut_env(0)])
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
    out = apply_model_clamp(sigma, _Ctx({}), _State(), None, "pk", 1, 3, None, None, 4, True)
    assert out is sigma          # identity, not just equality — no allocation


def test_unmodeled_actor_returns_sigma_unchanged():
    sigma = _sigma()
    ctx = _Ctx({0: _FixedModel()})
    out = apply_model_clamp(sigma, ctx, _State(), None, "pk", 1, 3, None, None, 4, True)
    assert out is sigma          # seat 1 has no model




def _entry(n_rows, width):
    return (np.zeros((n_rows, width)), np.zeros((n_rows, 1)),
            np.zeros(n_rows, dtype=bool))


def _stub_fill(monkeypatch, m_row, c_val, counter=None):
    """Replace the row builder with one that fills every row with a fixed value."""
    import poker_ai.search.vform as vform

    def fake(entry, model, env, width, combo_cards, cof, is_root, rcof=None):
        if counter is not None:
            counter.append(1)
        m_rows, c_rows, filled = entry
        m_rows[:] = m_row
        c_rows[:] = c_val
        filled[:] = True

    monkeypatch.setattr(vform, "_fill_model_rows", fake)


@pytest.mark.parametrize("c", [0.0, 0.25, 0.5, 1.0])
def test_blend_interpolates_between_free_and_model(c, monkeypatch):
    """``σ̃ = c·σ̂ + (1−c)·x`` exactly; c=0 ⇒ free strategy, c=1 ⇒ model row."""
    sigma = _sigma()
    n, width = sigma.shape
    model_row = np.zeros(width); model_row[0] = 1.0       # point mass, action 0
    _stub_fill(monkeypatch, model_row, c)

    ctx = _Ctx({1: _FixedModel(c=c)})
    # The clamp blends IN PLACE (P2 kernel), so snapshot the free strategy first —
    # `sigma` is mutated by the call.  Safe in production: every caller's `sigma`
    # is freshly allocated by `node_sigma` and nobody holds the pre-clamp array.
    free = sigma.copy()
    out = apply_model_clamp(sigma, ctx, _State(), None, "pk", 1, width, None,
                            None, n, True)

    expected = c * np.tile(model_row, (n, 1)) + (1.0 - c) * free
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)      # still a distribution
    if c == 0.0:
        np.testing.assert_allclose(out, free)
    if c == 1.0:
        np.testing.assert_allclose(out, np.tile(model_row, (n, 1)))


def test_entry_is_cached_per_seat_and_node(monkeypatch):
    """One cache entry per ``(seat, public_key)``; revisits hit it."""
    sigma = _sigma()
    n, width = sigma.shape
    calls = []
    _stub_fill(monkeypatch, np.zeros(width), 0.0, counter=calls)

    ctx, state = _Ctx({1: _FixedModel()}), _State()
    for _ in range(3):
        apply_model_clamp(sigma, ctx, state, None, "pk", 1, width, None, None, n, True)
    apply_model_clamp(sigma, ctx, state, None, "other_pk", 1, width, None, None, n, True)

    # 4 queries over 2 distinct keys ⇒ 2 entry allocations + 2 revisits.
    assert state.model_sigma_cache.misses == 2
    assert state.model_sigma_cache.hits == 2
    assert len(state.model_sigma_cache) == 2


# --------------------------------------------------------------------------- #
# 3. Row-space correctness — the board-staleness regression
# --------------------------------------------------------------------------- #

class _InfoSetModel:
    """Row depends on the queried info-set, so a stale row is detectable."""

    def __init__(self):
        self.seen = []

    def strategy(self, state):
        self.seen.append(bytes(state.info_set))
        n = len(state.legal_actions)
        r = np.zeros(n, dtype=np.float64)
        r[hash(bytes(state.info_set)) % n] = 1.0
        return r

    def confidence(self, state):
        return 1.0


def test_rows_are_filled_per_cluster_and_gathered_with_the_live_board():
    """``_fill_model_rows`` builds ONE row per cluster row and only for rows the
    current board reaches; a later board fills the rows it newly exposes.

    This is the board-staleness regression: ``public_key`` carries no board, and
    both regimes re-sample the completion every iteration, so caching *combo*-space
    rows under ``(seat, public_key)`` would reuse the first board's rows forever.
    """
    from poker_ai.search.vform import _fill_model_rows

    class _FakeState:
        legal_actions = ("fold", "call")

        def __init__(self, tag):
            self.info_set = tag

    class _FakeEnv:
        def policy_public_fields(self):
            return None

        def policy_state_for(self, combo, for_blueprint=False, public=None):
            return _FakeState(bytes(np.asarray(combo).tobytes()))

    n_rows, width = 4, 2
    combo_cards = np.arange(12, dtype=np.int64).reshape(6, 2)
    model = _InfoSetModel()
    entry = _fill_model_rows_entry = _entry(n_rows, width)

    # Board A reaches cluster rows {0, 1} only.
    cof_a = np.array([0, 0, 1, 1, -1, -1])
    _fill_model_rows(entry, model, _FakeEnv(), width, combo_cards, cof_a, False)
    assert list(entry[2]) == [True, True, False, False]
    assert len(model.seen) == 2, "one query per cluster row, not per combo"

    # Board B reaches {1, 2, 3}: row 1 is reused, rows 2/3 are newly built.
    cof_b = np.array([1, 1, 2, 2, 3, 3])
    _fill_model_rows(entry, model, _FakeEnv(), width, combo_cards, cof_b, False)
    assert list(entry[2]) == [True, True, True, True]
    assert len(model.seen) == 4, "row 1 rebuilt (should have been reused)"


def test_confidence_is_clamped_into_the_unit_interval():
    """A third-party model returning c outside [0,1] would push the mixture off
    the simplex; the builder clamps it."""
    from poker_ai.search.vform import _fill_model_rows

    class _FakeState:
        legal_actions = ("fold", "call")
        info_set = b"k"

    class _FakeEnv:
        def policy_public_fields(self):
            return None

        def policy_state_for(self, combo, for_blueprint=False, public=None):
            return _FakeState()

    class _WildModel:
        def __init__(self, c):
            self._c = c

        def strategy(self, state):
            return np.array([1.0, 0.0])

        def confidence(self, state):
            return self._c

    for c, want in ((-2.0, 0.0), (5.0, 1.0), (0.3, 0.3)):
        entry = _entry(2, 2)
        _fill_model_rows(entry, _WildModel(c), _FakeEnv(), 2,
                         np.zeros((2, 2), dtype=np.int64), None, True)
        np.testing.assert_allclose(entry[1], want)


def test_model_rows_zero_fill_overlay_columns():
    """A model row narrower than the node's legal width (an off-tree action was
    injected) is zero-filled, so the blend can never invent mass on it."""
    from poker_ai.search.vform import _fill_model_rows

    class _FakeState:
        legal_actions = ("fold", "call", "raise:1.0")
        info_set = b"k"

    seen = {}

    class _FakeEnv:
        def policy_public_fields(self):
            return None

        def policy_state_for(self, combo, for_blueprint=False, public=None):
            seen["for_blueprint"] = for_blueprint
            return _FakeState()

    n, width = 3, 5          # node has 5 legal actions; model knows only 3
    entry = _entry(n, width)
    _fill_model_rows(entry, _FixedModel(row=[0.5, 0.25, 0.25], c=0.7),
                     _FakeEnv(), width, np.zeros((n, 2), dtype=np.int64), None, True)
    m, c = entry[0], entry[1]

    assert np.all(m[:, 3:] == 0.0)                       # overlay columns: no mass
    np.testing.assert_allclose(m[:, :3], np.tile([0.5, 0.25, 0.25], (n, 1)))
    np.testing.assert_allclose(c, 0.7)
    # The clamp canonicalises the history exactly as the blueprint read and the
    # §6.3 belief swap do, so all three query one and the same info-set key.
    assert seen["for_blueprint"] is True


# --------------------------------------------------------------------------- #
# 4. Fork safety — models must be in the LMDB reopen sweep
# --------------------------------------------------------------------------- #

def test_models_are_reopened_after_fork():
    """``ctx.models`` rides the same post-fork LMDB repair as the leaf fleet.

    A modeled seat's σ̂ is typically blueprint-backed and is queried *inside* the
    forked replica, so omitting it would trip ``MDB_BAD_RSLOT`` on first query.
    """
    from poker_ai.search.parallel import _reopen_leaf_fleet_lmdb

    class _Reopenable:
        def __init__(self): self.n = 0
        def reopen_after_fork(self): self.n += 1

    leaf_pol, model_pol, shared = _Reopenable(), _Reopenable(), _Reopenable()

    class _Leaf:
        policies = {"none": leaf_pol, "fold": shared}

    class _Ctx2:
        leaf = _Leaf()
        models = {1: model_pol, 2: shared}      # `shared` reachable both ways

    _reopen_leaf_fleet_lmdb(_Ctx2())
    assert leaf_pol.n == 1
    assert model_pol.n == 1, "opponent model was not reopened after fork"
    assert shared.n == 1, "shared blueprint reopened more than once"


def test_synthetic_model_delegates_reopen_to_its_policy():
    from poker_ai.modeling.model import SyntheticOpponentModel

    class _Pol:
        def __init__(self): self.n = 0
        def reopen_after_fork(self): self.n += 1
        def strategy(self, state, bias="none"): return np.array([1.0])

    pol = _Pol()
    SyntheticOpponentModel(pol).reopen_after_fork()
    assert pol.n == 1

    # In-memory policies expose no hook — must be a silent no-op, not a crash.
    class _Bare:
        def strategy(self, state, bias="none"): return np.array([1.0])
    SyntheticOpponentModel(_Bare()).reopen_after_fork()


# --------------------------------------------------------------------------- #
# P2 — the fused clamp kernel must be byte-identical to its Python oracle
# --------------------------------------------------------------------------- #

def _rand_blend_inputs(rng, n, w, n_rows, clustered):
    sigma = rng.random((n, w)); sigma /= sigma.sum(1, keepdims=True)
    m = rng.random((n_rows, w)); m /= m.sum(1, keepdims=True)
    c = rng.random((n_rows, 1))
    gof = rng.integers(0, n_rows, size=n).astype(np.int64) if clustered else None
    return np.ascontiguousarray(sigma), np.ascontiguousarray(m), \
        np.ascontiguousarray(c), gof


@pytest.mark.parametrize("clustered", [False, True])
def test_clamp_kernel_is_byte_identical_to_the_oracle(clustered):
    """The compiled fused gather+blend must match ``_clamp_sigma_py`` bit-for-bit."""
    from poker_ai.search import vform
    core = pytest.importorskip("poker_ai._core._clamp")

    rng = np.random.default_rng(0)
    # At a root node rows ARE combos, so n_rows must equal n; only a clustered
    # node has fewer rows than combos.
    shapes = ([(190, 3, 37), (64, 5, 12)] if clustered
              else [(190, 3, 190), (64, 5, 64)])
    for n, w, n_rows in shapes:
        for _ in range(20):
            s, m, c, gof = _rand_blend_inputs(rng, n, w, n_rows, clustered)
            got = core.clamp_sigma(s.copy(), m, c, gof)
            want = vform._clamp_sigma_py(s.copy(), m, c, gof)
            assert got.tobytes() == want.tobytes(), "kernel diverged from oracle"


@pytest.mark.parametrize("c_val", [0.0, 1.0])
def test_clamp_kernel_endpoints_are_exact(c_val):
    """``c=0`` leaves the free strategy untouched; ``c=1`` returns the model row."""
    from poker_ai.search.vform import clamp_sigma
    rng = np.random.default_rng(1)
    s, m, _, gof = _rand_blend_inputs(rng, 40, 4, 40, False)
    c = np.full((40, 1), c_val)
    out = clamp_sigma(s.copy(), m, c, gof)
    np.testing.assert_array_equal(out, s if c_val == 0.0 else m)


def test_clamp_kernel_falls_back_on_odd_layouts():
    """A non-float64 / non-contiguous input must take the oracle, not mis-type."""
    from poker_ai.search.vform import clamp_sigma
    rng = np.random.default_rng(2)
    s, m, c, _ = _rand_blend_inputs(rng, 8, 3, 8, False)
    out = clamp_sigma(s.astype(np.float32), m, c, None)   # float32 sigma
    assert out.shape == s.shape
