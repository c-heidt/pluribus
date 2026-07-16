"""Tests for BayesOpponentModel (poker_ai/modeling/model.py, §3 model form).

The learned posterior: per-state blueprint prior + coarse-count Dirichlet blend +
expand back to legal width by the blueprint's within-class sizing.  Pins the three
properties §3/§8 specify — ``n=0`` recovers the blueprint, ``n→∞`` recovers the
empirical class frequencies, and the raise-class mass is split by the blueprint —
plus the confidence law and the ModelStore snapshot wiring.
"""

import numpy as np
import pytest

from environment.poker_env import PolicyState, encode_info_set
from poker_ai.modeling.counts import CountsTable, model_key
from poker_ai.modeling.model import BayesOpponentModel
from poker_ai.modeling.store import ModelStore
from poker_ai.modeling.tiers import build_tiers_from_centroids


def _toy_centroids():
    river = np.array(
        [[0.1, 0.9, 0.0], [0.4, 0.6, 0.0], [0.6, 0.4, 0.0], [0.9, 0.1, 0.0]]
    )
    turn = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    flop = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    return {"river": river, "turn": turn, "flop": flop}


def _tiers():
    return build_tiers_from_centroids(_toy_centroids(), n_tiers=4)


# A 5-action river node: fold, call, two raise sizes, all-in.
_LEGAL = ("fold", "call", "raise:0.5", "raise:1.0", "all_in")
_BP = np.array([0.2, 0.3, 0.3, 0.1, 0.1], dtype=np.float32)   # classes: 0,1,2,2,3


class FakeBlueprint:
    def __init__(self, dist):
        self._d = np.asarray(dist, dtype=np.float32)

    def strategy(self, state, bias="none"):
        return self._d


def _state():
    info_set = encode_info_set(3, [("river", [])])          # river cluster 3
    return PolicyState(
        player_i=0,
        betting_round=3,
        info_set=info_set,
        valid_mask=np.ones(len(_LEGAL), dtype=bool),
        legal_actions=_LEGAL,
    )


def _counts_with(key, masses):
    """A frozen CountsView carrying ``masses`` (a 4-vector) at ``key``."""
    t = CountsTable()
    for cls, m in enumerate(masses):
        if m:
            t.buffer_observation(key, cls, float(m))
    t.commit()
    return t.snapshot()


class TestBayesStrategy:

    def test_n_zero_recovers_the_exact_blueprint(self):
        st = _state()
        m = BayesOpponentModel(FakeBlueprint(_BP), CountsTable().snapshot(), _tiers(), tau=50)
        assert np.allclose(m.strategy(st), _BP, atol=1e-6)

    def test_blend_and_expand_matches_hand_computation(self):
        st = _state()
        key = model_key(st, _tiers())
        # n = [fold 10, call 0, raise 40, all_in 0], total 50, tau 50.
        view = _counts_with(key, [10, 0, 40, 0])
        m = BayesOpponentModel(FakeBlueprint(_BP), view, _tiers(), tau=50)
        # prior=[.2,.3,.4,.1]; sigma_coarse=(50*prior+n)/100=[.2,.15,.6,.05];
        # raise .6 split by bp[.3,.1] → .45/.15.
        expected = np.array([0.2, 0.15, 0.45, 0.15, 0.05])
        assert np.allclose(m.strategy(st), expected, atol=1e-6)

    def test_n_large_recovers_empirical_class_freqs(self):
        st = _state()
        key = model_key(st, _tiers())
        view = _counts_with(key, [1_000_000, 0, 1_000_000, 0])   # class freqs .5/0/.5/0
        m = BayesOpponentModel(FakeBlueprint(_BP), view, _tiers(), tau=50)
        # raise .5 split by bp[.3,.1] → .375/.125; fold .5; call/all_in 0.
        expected = np.array([0.5, 0.0, 0.375, 0.125, 0.0])
        assert np.allclose(m.strategy(st), expected, atol=1e-3)

    def test_tau_zero_no_data_degrades_to_blueprint_not_nan(self):
        # Degenerate tau=0 with no counts must not produce 0/0 = NaN.
        st = _state()
        out = BayesOpponentModel(
            FakeBlueprint(_BP), CountsTable().snapshot(), _tiers(), tau=0.0
        ).strategy(st)
        assert np.isfinite(out).all()
        assert np.allclose(out, _BP, atol=1e-6)

    def test_strategy_is_a_distribution(self):
        st = _state()
        view = _counts_with(model_key(st, _tiers()), [3, 7, 2, 1])
        out = BayesOpponentModel(FakeBlueprint(_BP), view, _tiers()).strategy(st)
        assert np.isclose(out.sum(), 1.0, atol=1e-6) and (out >= 0).all()

    def test_coarse_class_with_no_legal_action_is_dropped(self):
        # A 2-action node (fold/call) but counts carry raise mass → raise is dropped
        # and the row renormalises over the legal set.
        legal = ("fold", "call")
        bp = np.array([0.5, 0.5], dtype=np.float32)
        info_set = encode_info_set(3, [("river", [])])
        st = PolicyState(0, 3, info_set, np.ones(2, bool), legal)
        view = _counts_with(model_key(st, _tiers()), [0, 0, 100, 0])   # all "raise"
        out = BayesOpponentModel(FakeBlueprint(bp), view, _tiers(), tau=1).strategy(st)
        assert np.isclose(out.sum(), 1.0) and (out >= 0).all()


class TestBayesConfidence:

    def test_confidence_law_and_cap(self):
        st = _state()
        key = model_key(st, _tiers())
        m0 = BayesOpponentModel(FakeBlueprint(_BP), CountsTable().snapshot(), _tiers(),
                                tau=50, p_max=0.8)
        assert m0.confidence(st) == 0.0                       # no data
        m1 = BayesOpponentModel(FakeBlueprint(_BP), _counts_with(key, [25, 25, 0, 0]),
                                _tiers(), tau=50, p_max=0.8)
        assert m1.confidence(st) == pytest.approx(0.5)        # 50/(50+50)
        m2 = BayesOpponentModel(FakeBlueprint(_BP), _counts_with(key, [100, 100, 0, 0]),
                                _tiers(), tau=50, p_max=0.8)
        assert m2.confidence(st) == pytest.approx(0.8)        # 200/250=0.8, capped

    def test_confidence_is_monotone_in_counts(self):
        st = _state()
        key = model_key(st, _tiers())
        cs = [
            BayesOpponentModel(FakeBlueprint(_BP), _counts_with(key, [n, 0, 0, 0]),
                               _tiers(), tau=50, p_max=1.0).confidence(st)
            for n in (0, 10, 50, 200, 1000)
        ]
        assert all(a <= b for a, b in zip(cs, cs[1:]))


class TestStoreSnapshot:

    def test_store_snapshot_builds_a_working_model(self):
        st = _state()
        store = ModelStore(blueprint_policy=FakeBlueprint(_BP), tiers=_tiers(), tau=50)
        # No data yet → snapshot model returns the blueprint.
        assert np.allclose(store.snapshot("opp").strategy(st), _BP, atol=1e-6)
        # Buffer + commit an observation, then a fresh snapshot reflects it.
        store.buffer_observation("opp", model_key(st, _tiers()), 0, 50.0)  # fold class
        store.commit_hand()
        assert store.snapshot("opp").confidence(st) == pytest.approx(0.5)

    def test_snapshot_requires_blueprint_and_tiers(self):
        with pytest.raises(ValueError, match="blueprint_policy and tiers"):
            ModelStore().snapshot("opp")
