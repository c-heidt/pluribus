"""A4 — the belief-likelihood swap (opponent_modeling §6.3).

For a **modeled** opponent seat, the Bayesian range update's likelihood becomes the
model ``σ̂`` instead of the baseline (last search's average / blueprint).  Crucially
it is ``σ̂``, **not** the solver's mixture ``σ̃ = c·σ̂ + (1−c)·x``: beliefs estimate
what the opponent *actually does*, while the mixture is only the solver's hedge.

Gates here:

1. **Baseline untouched** — with no models the closure is byte-for-byte the old
   path (vanilla Pluribus unaffected).
2. **Per-seat** — only modeled seats swap; unmodeled seats and the bot keep the
   baseline path.
3. **σ̂ not σ̃** — the likelihood ignores confidence entirely.
4. **The §6.3 invariant** — the belief tracker and the solver clamp read the *same*
   hand-start frozen snapshot, and the bot's own seat is never modeled.
"""

import numpy as np
import pytest

from poker_ai.search.agent import SearchAgent

from test.search._helpers import _policies, _preflop_env
from test.search.test_budget import _cfg


class _RecordingModel:
    """Returns a fixed row and records the info-sets it was queried at."""

    def __init__(self, row=(0.75, 0.25), c=0.3):
        self._row, self._c, self.seen = row, c, []

    def strategy(self, state):
        self.seen.append(state.info_set)
        n = len(state.legal_actions)
        r = np.zeros(n, dtype=np.float64)
        r[: len(self._row)] = self._row[:n]
        return r / r.sum()

    def confidence(self, state):
        return self._c


def _agent(models=None, seed=0):
    return SearchAgent(
        leaf_policies=_policies(),
        blueprint_policy=_policies()["none"],
        solver_cfg=_cfg(auto_budget=False, max_iterations=4, max_wall_seconds=1e9,
                        workers=1),
        rng=np.random.default_rng(seed),
        models=models,
    )


# --------------------------------------------------------------------------- #
# 1. Baseline untouched
# --------------------------------------------------------------------------- #

def test_no_models_keeps_the_baseline_closure():
    """Without models the per-seat argument changes nothing — vanilla path."""
    env = _preflop_env()
    ag = _agent()
    ag.on_hand_start(env, my_seat=0)

    base = ag._make_sigma_for_combo(env)          # old call shape (seat=None)
    per_seat = ag._make_sigma_for_combo(env, 1)   # new call shape, unmodeled seat
    for h in range(min(8, env.combo_cards.shape[0])):
        np.testing.assert_array_equal(base(h), per_seat(h))


def test_models_default_to_empty_and_agent_is_inert():
    ag = _agent()
    ag.on_hand_start(_preflop_env(), my_seat=0)
    assert dict(ag._models) == {}


# --------------------------------------------------------------------------- #
# 2/3. Per-seat swap, and it is σ̂ (never the mixture)
# --------------------------------------------------------------------------- #

def test_modeled_seat_uses_the_model_unmodeled_seat_does_not():
    env = _preflop_env()
    model = _RecordingModel()
    ag = _agent(models={1: model})
    ag.on_hand_start(env, my_seat=0)

    modeled = ag._make_sigma_for_combo(env, 1)
    unmodeled = ag._make_sigma_for_combo(env, 0)

    row = modeled(0)
    assert model.seen, "modeled seat did not query the model"
    # The model's row, renormalized over the node's legal width.
    np.testing.assert_allclose(row[:2] / row[:2].sum(), [0.75, 0.25])

    before = len(model.seen)
    unmodeled(0)
    assert len(model.seen) == before, "unmodeled seat queried the model"


@pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
def test_likelihood_is_sigma_hat_not_the_mixture(c):
    """Confidence must not enter the belief likelihood at all (§6.3): the same
    ``σ̂`` comes back regardless of ``c``."""
    env = _preflop_env()
    rows = []
    for conf in (c, 1.0 - c):
        ag = _agent(models={1: _RecordingModel(c=conf)})
        ag.on_hand_start(env, my_seat=0)
        rows.append(ag._make_sigma_for_combo(env, 1)(0))
    np.testing.assert_array_equal(rows[0], rows[1])


def test_likelihood_row_is_a_distribution_aligned_to_legal():
    env = _preflop_env()
    ag = _agent(models={1: _RecordingModel()})
    ag.on_hand_start(env, my_seat=0)
    legal = [a for a in env.legal_actions if a is not None]
    row = ag._make_sigma_for_combo(env, 1)(0)
    assert row.shape == (len(legal),)
    assert row.dtype == np.float64
    np.testing.assert_allclose(row.sum(), 1.0)


# --------------------------------------------------------------------------- #
# 4. The §6.3 invariant: one frozen snapshot, hero never modeled
# --------------------------------------------------------------------------- #

def test_hero_seat_is_dropped_from_the_model_snapshot():
    """Hero is never modeled — the clamp must never blend the bot's own rows."""
    m = _RecordingModel()
    ag = _agent(models={0: m, 1: m})
    ag.on_hand_start(_preflop_env(), my_seat=0)
    assert set(ag._models) == {1}


def test_snapshot_is_frozen_per_hand_and_shared_with_the_solver():
    """The tracker's likelihood and the solver's ctx read the SAME mapping, and a
    later mutation of the caller's dict does not leak into the live hand."""
    env = _preflop_env()
    src = {1: _RecordingModel()}
    ag = _agent(models=src)
    ag.on_hand_start(env, my_seat=0)
    snap = ag._models

    src[2] = _RecordingModel()          # caller mutates after the hand started
    assert set(ag._models) == {1}, "snapshot was not frozen at hand start"

    ag._solve_and_store(ag._root_env)
    assert ag._ctx is not None
    # Same models reach the solver clamp as fed the belief likelihood.
    assert dict(ag._ctx.models) == dict(snap)


def test_snapshot_refreshes_on_the_next_hand():
    src = {1: _RecordingModel()}
    ag = _agent(models=src)
    ag.on_hand_start(_preflop_env(), my_seat=0)
    src[2] = _RecordingModel()
    ag.on_hand_start(_preflop_env(), my_seat=0)   # new hand → re-freeze
    assert set(ag._models) == {1, 2}
