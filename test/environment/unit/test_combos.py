"""Tests for the env's hole-combo enumeration."""

from environment.poker_env import new_game
from environment.player import Player
from environment.poker_env import PokerEnv


def _env(low: int = 2, high: int = 14, n_players: int = 2) -> PokerEnv:
    return PokerEnv(
        players=[Player(i, 10000) for i in range(n_players)],
        low_card_rank=low,
        high_card_rank=high,
    )


class TestComboEnumeration:

    def test_full_deck_n_combos(self):
        env = _env(2, 14)
        assert env.n_combos == 1326  # C(52, 2)

    def test_short_deck_n_combos(self):
        env = _env(10, 14)
        assert env.n_combos == 190  # C(20, 2)

    def test_combo_cards_shape_and_dtype(self):
        env = _env(2, 14)
        assert env.combo_cards.shape == (1326, 2)
        assert env.combo_cards.dtype.name == "int32"

    def test_combo_cards_sorted_within_row(self):
        env = _env(2, 14)
        assert (env.combo_cards[:, 0] < env.combo_cards[:, 1]).all()

    def test_combo_cards_unique(self):
        env = _env(10, 14)
        import numpy as np
        assert np.unique(env.combo_cards, axis=0).shape[0] == env.n_combos

    def test_combo_index_roundtrip(self):
        env = _env(10, 14)
        for i, (c0, c1) in enumerate(env.combo_cards):
            assert env.combo_index[(int(c0), int(c1))] == i

    def test_cache_shared_across_instances(self):
        env_a = _env(2, 14)
        env_b = _env(2, 14)
        # lru_cache returns identical objects for identical args.
        assert env_a.combo_cards is env_b.combo_cards
        assert env_a.combo_index is env_b.combo_index

    def test_different_decks_get_different_arrays(self):
        env_full = _env(2, 14)
        env_short = _env(10, 14)
        assert env_full.n_combos != env_short.n_combos
        assert env_full.combo_cards is not env_short.combo_cards
