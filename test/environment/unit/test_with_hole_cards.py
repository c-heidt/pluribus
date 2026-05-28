"""Tests for ``PokerEnv.with_hole_cards``."""

from environment.player import Player
from environment.poker_env import PokerEnv


def _env(n_players: int = 2):
    return PokerEnv(players=[Player(i, 10000) for i in range(n_players)])


class TestWithHoleCards:

    def test_returns_independent_env(self):
        env = _env()
        original_cards = env.players[0].cards
        replacement = (env.combo_cards[0, 0], env.combo_cards[0, 1])
        new_env = env.with_hole_cards(0, replacement)
        assert new_env is not env
        # Mutating the returned env's bet/pot state must not affect original.
        assert new_env.pot_size == env.pot_size
        # And the original's hole cards are untouched.
        assert env.players[0].cards == original_cards

    def test_seat_cards_replaced(self):
        env = _env()
        replacement = (env.combo_cards[5, 0], env.combo_cards[5, 1])
        new_env = env.with_hole_cards(0, replacement)
        assert tuple(new_env.players[0].cards) == (
            int(replacement[0]),
            int(replacement[1]),
        )

    def test_other_seats_unchanged(self):
        env = _env(n_players=3)
        original_seat_1 = env.players[1].cards
        original_seat_2 = env.players[2].cards
        replacement = (env.combo_cards[0, 0], env.combo_cards[0, 1])
        new_env = env.with_hole_cards(0, replacement)
        assert new_env.players[1].cards == original_seat_1
        assert new_env.players[2].cards == original_seat_2

    def test_card_info_lut_preserved(self):
        env = _env()
        env.card_info_lut = {"pre_flop": {"some_key": 42}}  # sentinel
        replacement = (env.combo_cards[0, 0], env.combo_cards[0, 1])
        new_env = env.with_hole_cards(0, replacement)
        assert new_env.card_info_lut is env.card_info_lut

    def test_overlay_shared(self):
        env = _env()
        env.inject_action("raise:0.42")
        replacement = (env.combo_cards[0, 0], env.combo_cards[0, 1])
        new_env = env.with_hole_cards(0, replacement)
        # Overlay is shared by reference (same dict object across the lineage).
        assert new_env.has_overlay_at_current_node
