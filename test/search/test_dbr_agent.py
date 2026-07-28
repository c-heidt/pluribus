"""A6 — the DBR agent (opponent_modeling §6, §8.5).

A6 is deliberately thin: A2 gave the clamp and A4 gave the per-hand snapshot + the
belief swap, so the "agent" is mostly a set of *invariants that must still hold* once
models are attached.  Those invariants are the point of this file:

- **Determinism** (§8.5) — a seeded solve with models is reproducible, and model rows
  are first-draw only.
- **Multi-opponent** — one model per live opponent seat, each applied to its own seat.
- **Regime selection is unchanged** — no ``force_mccfr_when_modeled``.  Regime is
  model-**independent** (a HU turn/river takes vector, a HU flop/preflop takes MCCFR,
  regardless of models), so vanilla and DBR use the *identical* regime per subgame —
  which is what keeps the paired DBR-vs-vanilla comparison free of a regime confound.
- **The leaf stays blueprint** (§5.4, deferred for safety) — exploitation is confined
  to the searched subtree.
- **Hero is never modeled**, enforced at the clamp as well as at the agent.
"""

import numpy as np
import pytest

from poker_ai.search.agent import SearchAgent
from poker_ai.search.solver import _select_regime, solve
from poker_ai.search.vform import apply_model_clamp

from test.search._helpers import _ctx, _policies, _real_lut_env
from test.search.test_budget import _cfg
from test.search.test_belief_swap import _RecordingModel, _agent
from test.search.test_model_clamp import _State, _digest, _sigma


def _modeled(ctx, models):
    import dataclasses
    return dataclasses.replace(ctx, models=models)


# --------------------------------------------------------------------------- #
# Determinism (§8.5)
# --------------------------------------------------------------------------- #

@pytest.mark.requires_lut
@pytest.mark.parametrize("env_fn,regime", [
    (lambda: _real_lut_env(3), "vector"),
    (lambda: _real_lut_env(0), "mccfr"),
])
def test_seeded_solve_with_models_is_reproducible(env_fn, regime):
    """Same seed + same models ⇒ byte-identical solver state."""
    def run():
        env = env_fn()
        ctx = _modeled(_ctx(env, seed=7), {1: _RecordingModel(c=0.4)})
        res = solve(env, ctx, _cfg(auto_budget=False, max_iterations=30,
                                   max_wall_seconds=1e9, workers=1))
        assert res.regime == regime
        return _digest(res.state)

    assert run() == run()


@pytest.mark.requires_lut
def test_model_rows_are_first_draw_only():
    """Each info-set is queried from the model exactly once; later visits reuse it.

    Stated per *info-set*, not per filled row: since P1 the root street queries once
    per distinct **cluster** and broadcasts to every row in it, so queries are
    strictly fewer than rows.  On the real LUT distinct clusters give distinct
    info-sets, so a repeat in ``seen`` means a genuine rebuild.
    """
    env = _real_lut_env(3)
    model = _RecordingModel(c=0.5)
    ctx = _modeled(_ctx(env, seed=7), {1: model})
    res = solve(env, ctx, _cfg(auto_budget=False, max_iterations=40,
                               max_wall_seconds=1e9, workers=1))

    cache = res.state.model_sigma_cache
    filled_total = sum(int(cache[k][2].sum()) for k in cache)
    assert filled_total > 0, "the clamp never ran — test is vacuous"
    seen = [bytes(s) for s in model.seen]
    assert len(seen) == len(set(seen)), "an info-set was queried twice (rebuild)"
    # P1's dedup: strictly fewer queries than rows filled.
    assert len(seen) < filled_total, (
        f"{len(seen)} queries for {filled_total} rows — cluster dedup not active"
    )


@pytest.mark.requires_lut
def test_models_change_the_solve():
    """Sanity: a confident model must actually move the strategy, else every
    'no-op' guarantee above would be vacuously true."""
    env = _real_lut_env(3)
    base = solve(env, _ctx(env, seed=7),
                 _cfg(auto_budget=False, max_iterations=30,
                      max_wall_seconds=1e9, workers=1))
    env2 = _real_lut_env(3)
    clamped = solve(env2, _modeled(_ctx(env2, seed=7), {1: _RecordingModel(c=1.0)}),
                    _cfg(auto_budget=False, max_iterations=30,
                         max_wall_seconds=1e9, workers=1))
    assert _digest(base.state) != _digest(clamped.state)


# --------------------------------------------------------------------------- #
# Multi-opponent
# --------------------------------------------------------------------------- #

@pytest.mark.requires_lut
def test_agent_snapshot_holds_one_model_per_opponent_seat():
    env = _real_lut_env(0)
    m1, m2 = _RecordingModel(row=(0.9, 0.1)), _RecordingModel(row=(0.2, 0.8))
    ag = _agent(models={1: m1, 2: m2})
    ag.on_hand_start(env, my_seat=0)
    assert dict(ag._models) == {1: m1, 2: m2}


@pytest.mark.requires_lut
def test_each_seat_gets_its_own_model():
    """Two modeled seats blend toward *different* rows — no cross-talk."""
    sigma = _sigma(n_combos=3, width=2)
    n, width = sigma.shape
    m1 = _RecordingModel(row=(1.0, 0.0), c=1.0)
    m2 = _RecordingModel(row=(0.0, 1.0), c=1.0)

    class _FakeEnv:
        def policy_public_fields(self): return None

        def policy_state_for(self, combo, for_blueprint=False, public=None):
            class _S:
                legal_actions = ("fold", "call")
                info_set = b"stub"
            return _S()

    ctx = type("C", (), {"models": {1: m1, 2: m2}, "my_seat": 0})()
    state, combos = _State(), np.zeros((n, 2), dtype=int)
    # The clamp blends IN PLACE, so each call needs its own `sigma` — otherwise the
    # second call would blend on top of the first's result and both `out` handles
    # would alias the same mutated array.
    out1 = apply_model_clamp(sigma.copy(), ctx, state, _FakeEnv(), "pk", 1, width,
                             combos, None, n, True)
    out2 = apply_model_clamp(sigma.copy(), ctx, state, _FakeEnv(), "pk", 2, width,
                             combos, None, n, True)
    np.testing.assert_allclose(out1, np.tile([1.0, 0.0], (n, 1)))
    np.testing.assert_allclose(out2, np.tile([0.0, 1.0], (n, 1)))


# --------------------------------------------------------------------------- #
# Regime selection is UNCHANGED (no force_mccfr_when_modeled)
# --------------------------------------------------------------------------- #

@pytest.mark.requires_lut
@pytest.mark.parametrize("env_fn,expected", [
    (lambda: _real_lut_env(3), "vector"),    # HU river  → vector, modeled or not
    (lambda: _real_lut_env(2), "vector"),    # HU turn   → vector
    (lambda: _real_lut_env(0), "mccfr"),             # HU preflop → MCCFR
])
def test_regime_is_identical_with_and_without_models(env_fn, expected):
    """The confound-killer: attaching a model must not reroute the subgame."""
    env = env_fn()
    plain = _ctx(env, seed=7)
    assert _select_regime(plain) == expected
    assert _select_regime(_modeled(plain, {1: _RecordingModel()})) == expected


@pytest.mark.requires_lut
def test_solve_reports_the_same_regime_when_modeled():
    env = _real_lut_env(3)
    res = solve(env, _modeled(_ctx(env, seed=7), {1: _RecordingModel()}),
                _cfg(auto_budget=False, max_iterations=10,
                     max_wall_seconds=1e9, workers=1))
    assert res.regime == "vector"


# --------------------------------------------------------------------------- #
# The leaf stays blueprint (§5.4, deferred for safety)
# --------------------------------------------------------------------------- #

@pytest.mark.requires_lut
def test_leaf_config_has_no_per_seat_model_policies():
    """Exploitation is confined to the searched subtree: no seat_policies exists,
    and attaching models does not add one."""
    env = _real_lut_env(0)
    ag = _agent(models={1: _RecordingModel()})
    ag.on_hand_start(env, my_seat=0)
    ag._solve_and_store(ag._root_env)
    leaf = ag._ctx.leaf
    assert not hasattr(leaf, "seat_policies")
    # The leaf fleet is the blueprint fleet, unchanged by modelling.
    assert set(leaf.policies) == set(_policies())


# --------------------------------------------------------------------------- #
# Hero is never modeled — enforced at the clamp too, not only at the agent
# --------------------------------------------------------------------------- #

def test_clamp_refuses_to_blend_heros_own_rows():
    """Even a hand-built ctx that wrongly lists my_seat must not blend hero."""
    sigma = _sigma()
    n, width = sigma.shape
    ctx = type("C", (), {"models": {0: _RecordingModel(c=1.0)}, "my_seat": 0})()
    out = apply_model_clamp(sigma, ctx, _State(), None, "pk", 0, width,
                            np.zeros((n, 2), dtype=int), None, n, True)
    assert out is sigma


@pytest.mark.requires_lut
def test_agent_drops_hero_even_if_supplied():
    ag = _agent(models={0: _RecordingModel(), 1: _RecordingModel()})
    ag.on_hand_start(_real_lut_env(0), my_seat=0)
    assert set(ag._models) == {1}


# --------------------------------------------------------------------------- #
# A modeled solve must SUCCEED under the search core
# --------------------------------------------------------------------------- #

@pytest.mark.requires_lut
@pytest.mark.parametrize("env_fn,regime", [
    (lambda: _real_lut_env(3), "vector"),
    (lambda: _real_lut_env(0), "mccfr"),
])
def test_modeled_solve_runs_under_the_search_core(monkeypatch, env_fn, regime):
    """Regression: the clamp queries the model at *hypothetical* holdings, which
    used to go through the ``PokerEnv``-only ``policy_state_for``.  With the core
    enabled a modeled solve raised ``AttributeError`` — and because
    ``SearchAgent._solve_and_store`` swallows solve exceptions, that surfaced as a
    silent blueprint fallback: **zero exploitation, DBR reads as ≈ vanilla
    Pluribus**.  A modeled solve must therefore complete, and must actually have clamped.

    Keying the model by *cluster* removed the fallback entirely (P1b); the engine
    equivalence itself is gated in ``test_model_infoset_seam.py``.
    """
    monkeypatch.setenv("PLURIBUS_SEARCH_CORE", "1")
    env = env_fn()
    model = _RecordingModel(c=0.6)
    res = solve(env, _modeled(_ctx(env, seed=7), {1: model}),
                _cfg(auto_budget=False, max_iterations=12,
                     max_wall_seconds=1e9, workers=1))
    assert res.regime == regime
    assert model.seen, "the clamp never queried the model under the core"
    assert len(res.state.model_sigma_cache) > 0


@pytest.mark.requires_lut
def test_unmodeled_solve_still_uses_the_core(monkeypatch):
    """**Both** conditions keep the compiled walk.

    This used to assert the opposite for the modeled solver — a modeled solve was
    forced onto the ``PokerEnv`` walk because the clamp needed ``policy_state_for``.
    Since the clamp keys the model by cluster (P1b), both engines serve it, so DBR
    and vanilla now run on the same engine and the DBR-vs-vanilla comparison carries
    no wall-clock asymmetry.  Byte-identity across engines is gated in
    ``test_model_infoset_seam``.
    """
    from poker_ai.search.vector import _VectorSolver
    from poker_ai.search.solver_state import SolverState

    monkeypatch.setenv("PLURIBUS_SEARCH_CORE", "1")
    env = _real_lut_env(3)
    pytest.importorskip("poker_ai._core._state")

    plain = _VectorSolver(env, SolverState(), _ctx(env, seed=7),
                          _cfg(auto_budget=False, max_iterations=2,
                               max_wall_seconds=1e9, workers=1),
                          np.random.default_rng(0))
    modeled = _VectorSolver(env, SolverState(),
                            _modeled(_ctx(env, seed=7), {1: _RecordingModel()}),
                            _cfg(auto_budget=False, max_iterations=2,
                                 max_wall_seconds=1e9, workers=1),
                            np.random.default_rng(0))
    assert plain._walk_env is not env, "baseline lost the compiled walk"
    assert modeled._walk_env is not env, "modeled solve lost the compiled walk"
