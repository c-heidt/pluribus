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

import hashlib

import numpy as np
import pytest

from poker_ai.search.agent import SearchAgent

from test.search._helpers import _policies, _real_lut_env
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

@pytest.mark.requires_lut
def test_no_models_keeps_the_baseline_closure():
    """Without models the per-seat argument changes nothing — vanilla path."""
    env = _real_lut_env(0)
    ag = _agent()
    ag.on_hand_start(env, my_seat=0)

    base = ag._make_sigma_for_combo(env)          # old call shape (seat=None)
    per_seat = ag._make_sigma_for_combo(env, 1)   # new call shape, unmodeled seat
    for h in range(min(8, env.combo_cards.shape[0])):
        np.testing.assert_array_equal(base(h), per_seat(h))


@pytest.mark.requires_lut
def test_models_default_to_empty_and_agent_is_inert():
    ag = _agent()
    ag.on_hand_start(_real_lut_env(0), my_seat=0)
    assert dict(ag._models) == {}


# --------------------------------------------------------------------------- #
# 2/3. Per-seat swap, and it is σ̂ (never the mixture)
# --------------------------------------------------------------------------- #

@pytest.mark.requires_lut
def test_modeled_seat_uses_the_model_unmodeled_seat_does_not():
    env = _real_lut_env(0)
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


@pytest.mark.requires_lut
@pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
def test_likelihood_is_sigma_hat_not_the_mixture(c):
    """Confidence must not enter the belief likelihood at all (§6.3): the same
    ``σ̂`` comes back regardless of ``c``."""
    env = _real_lut_env(0)
    rows = []
    for conf in (c, 1.0 - c):
        ag = _agent(models={1: _RecordingModel(c=conf)})
        ag.on_hand_start(env, my_seat=0)
        rows.append(ag._make_sigma_for_combo(env, 1)(0))
    np.testing.assert_array_equal(rows[0], rows[1])


@pytest.mark.requires_lut
def test_likelihood_row_is_a_distribution_aligned_to_legal():
    env = _real_lut_env(0)
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

@pytest.mark.requires_lut
def test_hero_seat_is_dropped_from_the_model_snapshot():
    """Hero is never modeled — the clamp must never blend the bot's own rows."""
    m = _RecordingModel()
    ag = _agent(models={0: m, 1: m})
    ag.on_hand_start(_real_lut_env(0), my_seat=0)
    assert set(ag._models) == {1}


@pytest.mark.requires_lut
def test_snapshot_is_frozen_per_hand_and_shared_with_the_solver():
    """The tracker's likelihood and the solver's ctx read the SAME mapping, and a
    later mutation of the caller's dict does not leak into the live hand."""
    env = _real_lut_env(0)
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


@pytest.mark.requires_lut
def test_snapshot_refreshes_on_the_next_hand():
    src = {1: _RecordingModel()}
    ag = _agent(models=src)
    ag.on_hand_start(_real_lut_env(0), my_seat=0)
    src[2] = _RecordingModel()
    ag.on_hand_start(_real_lut_env(0), my_seat=0)   # new hand → re-freeze
    assert set(ag._models) == {1, 2}


# --------------------------------------------------------------------------- #
# 5. Cluster-dedup of the belief sweep (the vanilla-and-DBR time win)
# --------------------------------------------------------------------------- #

def _info_set_row(state) -> np.ndarray:
    """A deterministic, DISTINCT row per info-set (collision-free in practice).

    The belief sweep's blueprint / model read is a function of the info-set
    ``(cluster, history)`` alone, so a correct cluster-dedup may only ever hand a
    combo the row of another combo sharing its info-set.  Seeding the row off the
    info-set bytes turns any mis-grouping into a value mismatch the test catches.
    """
    key = bytes(state.info_set)
    seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
    n = len(state.legal_actions)
    r = np.random.default_rng(seed).random(n) + 0.1
    return r / r.sum()


@pytest.mark.requires_lut
@pytest.mark.parametrize("target_round", [0, 1, 2, 3])
def test_dedup_by_cluster_matches_per_combo(target_round):
    """``_dedupe_by_cluster`` is byte-identical to the per-combo sweep it replaces.

    Grouping must be EXACT: ``clusters_for_board`` (the dedup's key) has to agree
    with the cluster ``policy_state_for`` embeds in the info-set, or a combo would
    receive a neighbour's row.  Checked over every feasible combo (the ones the real
    sweep actually queries — board-compatible, so nonzero range) at each street.
    """
    env = _real_lut_env(target_round)
    ag = _agent()
    cc = env.combo_cards
    public = env.policy_public_fields()

    def row_fn(h: int) -> np.ndarray:
        state = env.policy_state_for(
            tuple(int(c) for c in cc[h]), for_blueprint=True, public=public
        )
        return _info_set_row(state)

    deduped = ag._dedupe_by_cluster(env, cc, row_fn)
    board = np.asarray(env.community_cards, dtype=np.int64)
    feasible = np.flatnonzero(
        ~(np.isin(cc[:, 0], board) | np.isin(cc[:, 1], board))
    )
    assert feasible.size > 0
    for h in feasible.tolist():
        np.testing.assert_array_equal(deduped(int(h)), row_fn(int(h)))


@pytest.mark.requires_lut
def test_dedup_actually_collapses_queries():
    """The dedup must issue one read per distinct cluster, not one per combo — the
    whole point.  On the flop root the 20-card LUT has far fewer clusters than
    feasible combos, so the call count drops well below ``n_combos``."""
    env = _real_lut_env(1)
    ag = _agent()
    cc = env.combo_cards
    calls = {"n": 0}

    def row_fn(h: int) -> np.ndarray:
        calls["n"] += 1
        return np.array([1.0, 0.0], dtype=np.float64)

    deduped = ag._dedupe_by_cluster(env, cc, row_fn)
    board = np.asarray(env.community_cards, dtype=np.int64)
    feasible = np.flatnonzero(
        ~(np.isin(cc[:, 0], board) | np.isin(cc[:, 1], board))
    )
    for h in feasible.tolist():
        deduped(int(h))
    # One read per distinct cluster among the feasible combos.
    from information_abstraction.lookup import clusters_for_board
    from poker_ai.search.cluster_maps import _STREET_NAME
    clusters = clusters_for_board(
        env.card_info_lut[_STREET_NAME[env.betting_round]], cc, board
    )
    n_clusters = len(set(int(clusters[h]) for h in feasible.tolist()))
    assert calls["n"] == n_clusters < feasible.size
