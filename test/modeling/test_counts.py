"""Tests for coarse-bucket counts + the model key (poker_ai/modeling/counts.py, §4.2).

Covers the ``π`` projection (street/tier/ctx from a PolicyState), the 4-class action
map, and the buffer/commit freeze semantics of the counts table + ModelStore.
"""

import numpy as np
import pytest

from environment.poker_env import PolicyState, encode_info_set
from poker_ai.modeling.counts import (
    CountsTable,
    action_class,
    model_key,
)
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


def _state(cluster, history, legal, r):
    info_set = encode_info_set(cluster, history)
    return PolicyState(
        player_i=0,
        betting_round=r,
        info_set=info_set,
        valid_mask=np.ones(len(legal), dtype=bool),
        legal_actions=tuple(legal),
    )


class TestActionClass:

    def test_maps_each_token(self):
        assert action_class("fold") == 0
        assert action_class("call") == 1
        assert action_class("check") == 1
        assert action_class("raise:0.5") == 2
        assert action_class("raise:1.0") == 2
        assert action_class("all_in") == 3

    def test_unclassifiable_raises(self):
        with pytest.raises(ValueError):
            action_class("teleport")


class TestModelKey:

    def test_postflop_key_uses_tier_and_ctx(self):
        # River, cluster 3 (highest equity → tier 3), one raise this street, facing a bet.
        st = _state(3, [("river", ["raise:0.5"])], ["fold", "call", "raise:1.0"], r=3)
        assert model_key(st, _tiers()) == (3, 3, 1, 1)

    def test_facing_bet_is_zero_when_check_available(self):
        st = _state(0, [("turn", [])], ["check", "raise:1.0"], r=2)
        # cluster 0 on turn → tier 0; no aggression yet → bucket 0; check ⇒ not facing.
        assert model_key(st, _tiers()) == (2, 0, 0, 0)

    def test_raise_bucket_saturates_at_two_plus(self):
        hist = [("flop", ["raise:0.5", "raise:1.0", "raise:1.0"])]
        st = _state(2, hist, ["fold", "call"], r=1)     # flop street index 1
        # 3 aggressive actions → bucket 2 (2+).
        assert model_key(st, _tiers())[2] == 2

    def test_preflop_is_lossless_cluster(self):
        st = _state(137, [("pre_flop", ["call"])], ["fold", "call", "raise:1.0"], r=0)
        key = model_key(st, _tiers())
        assert key[0] == 0 and key[1] == 137        # raw cluster, not a tier


class TestCountsTable:

    def test_buffer_is_invisible_until_commit(self):
        t = CountsTable()
        snap0 = t.snapshot()
        t.buffer_observation((3, 3, 1, 1), 2, 0.7)      # a raise, 0.7 mass
        # Mid-hand: the snapshot (and a fresh one) still see nothing.
        assert snap0.total((3, 3, 1, 1)) == 0.0
        assert t.snapshot().total((3, 3, 1, 1)) == 0.0
        t.commit()
        assert t.snapshot().total((3, 3, 1, 1)) == pytest.approx(0.7)

    def test_prior_snapshot_stays_frozen_across_a_later_commit(self):
        t = CountsTable()
        t.buffer_observation((0, 5, 0, 0), 1, 1.0)
        t.commit()
        frozen = t.snapshot()
        assert frozen.total((0, 5, 0, 0)) == pytest.approx(1.0)
        # A subsequent hand's commit must not mutate the earlier frozen view.
        t.buffer_observation((0, 5, 0, 0), 1, 2.0)
        t.commit()
        assert frozen.total((0, 5, 0, 0)) == pytest.approx(1.0)          # unchanged
        assert t.snapshot().total((0, 5, 0, 0)) == pytest.approx(3.0)    # new view

    def test_soft_mass_accumulates_per_class(self):
        t = CountsTable()
        k = (1, 2, 0, 1)
        t.buffer_observation(k, 0, 0.3)     # fold 0.3
        t.buffer_observation(k, 2, 0.7)     # raise 0.7
        t.commit()
        row = t.snapshot().count(k)
        assert row[0] == pytest.approx(0.3) and row[2] == pytest.approx(0.7)
        assert t.snapshot().total(k) == pytest.approx(1.0)


class TestModelStore:

    def test_per_opponent_isolation_and_commit(self):
        store = ModelStore()
        store.buffer_observation("alice", (3, 1, 0, 0), 1, 1.0)
        store.buffer_observation("bob", (3, 1, 0, 0), 2, 1.0)
        store.commit_hand()
        assert store.counts_snapshot("alice").count((3, 1, 0, 0))[1] == pytest.approx(1.0)
        assert store.counts_snapshot("bob").count((3, 1, 0, 0))[2] == pytest.approx(1.0)
        assert store.counts_snapshot("alice").count((3, 1, 0, 0))[2] == 0.0

    def test_save_load_round_trip(self, tmp_path):
        store = ModelStore()
        store.buffer_observation("alice", (3, 1, 0, 0), 1, 1.5)
        store.buffer_observation("alice", (0, 42, 2, 1), 3, 0.25)
        store.buffer_observation("bob", (2, 0, 1, 0), 0, 2.0)
        store.commit_hand()
        p = tmp_path / "models.npz"
        store.save(p)
        back = ModelStore.load(p)
        assert set(back.opponent_ids()) == {"alice", "bob"}
        assert back.counts_snapshot("alice").count((3, 1, 0, 0))[1] == pytest.approx(1.5)
        assert back.counts_snapshot("alice").count((0, 42, 2, 1))[3] == pytest.approx(0.25)
        assert back.counts_snapshot("bob").count((2, 0, 1, 0))[0] == pytest.approx(2.0)

    def test_save_load_empty_store(self, tmp_path):
        p = tmp_path / "empty.npz"
        ModelStore().save(p)
        assert ModelStore.load(p).opponent_ids() == []
