"""Tests for opponent agents + table-composition assignment (doc §10.1)."""

import collections
import json

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv, PolicyState
from evaluation.opponents import (
    HERO_LABEL,
    LABEL_TO_BIAS,
    OPPONENT_LABELS,
    BlueprintOpponent,
    ModelSpec,
    assign_seats,
    synthetic_models_for,
)
from test.search._helpers import UniformPolicy


def _env(n_players=2, low=11, high=14):
    env = PokerEnv(
        players=[Player(i, 200) for i in range(n_players)],
        low_card_rank=low,
        high_card_rank=high,
    )
    env.card_info_lut = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0)
    )
    return env


# --------------------------------------------------------------------------- #
# Seat assignment
# --------------------------------------------------------------------------- #

class TestAssignSeats:

    def test_all_blueprint(self):
        labels = assign_seats("all_blueprint", hero_seat=2, n_players=6,
                              rng=np.random.default_rng(0))
        assert labels[2] == HERO_LABEL
        assert all(labels[s] == "bp" for s in range(6) if s != 2)
        assert len(labels) == 6

    def test_random_draws_from_variants_and_is_reproducible(self):
        a = assign_seats("random", 0, 6, np.random.default_rng(42))
        b = assign_seats("random", 0, 6, np.random.default_rng(42))
        assert a == b                                  # deterministic per rng
        assert a[0] == HERO_LABEL
        assert all(a[s] in OPPONENT_LABELS for s in range(1, 6))

    def test_fixed_assigns_all_identities_to_non_hero_seats(self):
        fixed = ["bp", "bp_call", "bp_raise", "bp_fold", "bp"]
        labels = assign_seats("fixed", hero_seat=3, n_players=6,
                              rng=np.random.default_rng(0), fixed_seats=fixed)
        assert labels[3] == HERO_LABEL
        # Every identity is present among the non-hero seats (order is shuffled
        # per hand, see test_fixed_shuffles_seat_order_per_hand below).
        assert sorted(labels[s] for s in range(6) if s != 3) == sorted(fixed)

    def test_fixed_identities_stick_to_opponents_as_hero_rotates(self):
        # Like real poker: only 3 identities needed for a 4-player game, and every
        # opponent plays every hand — just from a different seat as the hero moves.
        fixed = ["bp_fold", "bp_call", "bp_raise"]
        for hero_seat in range(4):
            labels = assign_seats("fixed", hero_seat=hero_seat, n_players=4,
                                  rng=np.random.default_rng(0), fixed_seats=fixed)
            assert labels[hero_seat] == HERO_LABEL
            opp_labels = [labels[s] for s in range(4) if s != hero_seat]
            assert sorted(opp_labels) == sorted(fixed)  # all 3 present every hand

    def test_fixed_shuffles_seat_order_per_hand(self):
        # The physical seat each identity lands in is randomized (via the per-hand
        # rng), not pinned to ascending order — so seat/position never systematically
        # correlates with a given bias across the sample.
        fixed = ["bp_fold", "bp_call", "bp_raise", "bp"]
        orders = {
            tuple(assign_seats("fixed", 0, 5, np.random.default_rng(seed),
                               fixed_seats=fixed)[s] for s in range(1, 5))
            for seed in range(10)
        }
        assert len(orders) > 1                          # varies across hands
        assert all(sorted(o) == sorted(fixed) for o in orders)  # always complete

    def test_fixed_reproducible_given_same_rng(self):
        fixed = ["bp_fold", "bp_call", "bp_raise"]
        a = assign_seats("fixed", 1, 4, np.random.default_rng(7), fixed_seats=fixed)
        b = assign_seats("fixed", 1, 4, np.random.default_rng(7), fixed_seats=fixed)
        assert a == b

    def test_fixed_wrong_length_raises(self):
        with pytest.raises(ValueError):
            assign_seats("fixed", 0, 3, np.random.default_rng(0),
                        fixed_seats=["bp"])          # n_players=3 needs 2 labels

    def test_fixed_unknown_label_raises(self):
        with pytest.raises(ValueError):
            assign_seats("fixed", 0, 3, np.random.default_rng(0),
                        fixed_seats=["bp", "who?"])

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError):
            assign_seats("nonsense", 0, 6, np.random.default_rng(0))

    def test_label_vocabulary(self):
        assert set(OPPONENT_LABELS) == {"bp", "bp_fold", "bp_call", "bp_raise"}
        assert LABEL_TO_BIAS["bp"] == "none"
        assert LABEL_TO_BIAS["bp_raise"] == "raise"


# --------------------------------------------------------------------------- #
# BlueprintOpponent
# --------------------------------------------------------------------------- #

class TestBlueprintOpponent:

    def test_sample_returns_legal_action(self):
        env = _env()
        opp = BlueprintOpponent("bp", UniformPolicy())
        seat = env.player_i
        action = opp.sample(env, seat, np.random.default_rng(0))
        assert action in [a for a in env.legal_actions if a is not None]

    def test_action_probs_normalised(self):
        env = _env()
        opp = BlueprintOpponent("bp_call", UniformPolicy())
        legal, probs = opp.action_probs(env, env.player_i)
        assert len(legal) == len(probs)
        assert abs(float(probs.sum()) - 1.0) < 1e-9

    def test_unknown_label_rejected(self):
        with pytest.raises(ValueError):
            BlueprintOpponent("gto_god", UniformPolicy())


# --------------------------------------------------------------------------- #
# ModelSpec — scalar knobs and JSON-string schedules (DBR sweep wiring, A7)
# --------------------------------------------------------------------------- #

def _pstate(betting_round=3, info_set=b"k", legal=("fold", "call", "raise:1.0")):
    n = len(legal)
    return PolicyState(
        player_i=0,
        betting_round=betting_round,
        info_set=info_set,
        valid_mask=np.ones(n, dtype=bool),
        legal_actions=tuple(legal),
    )


class TestModelSpec:

    def test_scalar_resolve_is_constant(self):
        err, conf = ModelSpec(error=0.2, confidence=0.7).resolve()
        assert err == 0.2 and conf == 0.7                    # scalars, not callables

    def test_as_json_scalar_has_no_schedule_keys(self):
        j = ModelSpec(p_max=0.8, error=0.1, confidence=0.9, seed=3).as_json()
        assert j == {"p_max": 0.8, "error": 0.1, "confidence": 0.9, "seed": 3}

    def test_error_schedule_overrides_scalar(self):
        spec = ModelSpec(
            error=0.99,                                       # ignored once schedule is set
            error_schedule=json.dumps(
                {"kind": "street", "by_round": {"0": 0.0, "3": 0.4}}
            ),
        )
        err, _ = spec.resolve()
        assert callable(err)
        assert err(_pstate(betting_round=0)) == 0.0
        assert err(_pstate(betting_round=3)) == 0.4

    def test_confidence_schedule_references_error_schedule(self):
        # calibrated confidence must track the resolved error schedule.
        spec = ModelSpec(
            error_schedule=json.dumps(
                {"kind": "street", "by_round": {"0": 0.05, "3": 0.45}}
            ),
            confidence_schedule=json.dumps({"kind": "calibrated", "gain": 1.0}),
        )
        _, conf = spec.resolve()
        assert conf(_pstate(betting_round=0)) > conf(_pstate(betting_round=3))
        assert np.isclose(conf(_pstate(betting_round=0)), 0.95)

    def test_as_json_includes_parsed_schedules(self):
        spec = ModelSpec(
            p_max=1.0,
            error_schedule=json.dumps({"kind": "uniform", "e": 0.2}),
            confidence_schedule=json.dumps({"kind": "flat", "c": 0.5}),
        )
        j = spec.as_json()
        assert j["error_schedule"] == {"kind": "uniform", "e": 0.2}
        assert j["confidence_schedule"] == {"kind": "flat", "c": 0.5}

    def test_malformed_schedule_raises_on_resolve(self):
        with pytest.raises(ValueError):
            ModelSpec(error_schedule=json.dumps({"kind": "nope"})).resolve()


class TestSyntheticModelsFor:

    def test_builds_one_model_per_non_hero_seat(self):
        labels = {0: HERO_LABEL, 1: "bp", 2: "bp_fold"}
        models = synthetic_models_for(labels, UniformPolicy(), ModelSpec())
        assert set(models) == {1, 2}                          # hero seat skipped

    def test_scheduled_confidence_reaches_the_built_model(self):
        labels = {0: HERO_LABEL, 1: "bp"}
        spec = ModelSpec(
            error_schedule=json.dumps(
                {"kind": "street", "by_round": {"0": 0.05, "3": 0.45}}
            ),
            confidence_schedule=json.dumps({"kind": "calibrated", "gain": 1.0}),
            p_max=1.0,
        )
        m = synthetic_models_for(labels, UniformPolicy(), spec)[1]
        # The model's confidence tracks the scheduled error (calibrated).
        assert m.confidence(_pstate(betting_round=0)) > m.confidence(_pstate(betting_round=3))
