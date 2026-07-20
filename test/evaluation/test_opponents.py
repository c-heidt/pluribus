"""Tests for opponent agents + table-composition assignment (doc §10.1)."""

import collections

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv
from evaluation.opponents import (
    HERO_LABEL,
    LABEL_TO_BIAS,
    OPPONENT_LABELS,
    BlueprintOpponent,
    assign_seats,
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

    def test_fixed_uses_map_and_ignores_hero_seat_entry(self):
        fixed = {0: "bp", 1: "bp_call", 2: "bp_raise", 3: "bp_fold", 4: "bp",
                 5: "bp_call"}
        labels = assign_seats("fixed", hero_seat=3, n_players=6,
                              rng=np.random.default_rng(0), fixed_seats=fixed)
        assert labels[3] == HERO_LABEL                 # hero's own entry ignored
        assert labels[0] == "bp" and labels[2] == "bp_raise"

    def test_fixed_missing_seat_raises(self):
        with pytest.raises(ValueError):
            assign_seats("fixed", 0, 3, np.random.default_rng(0),
                        fixed_seats={1: "bp"})          # seat 2 missing

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
