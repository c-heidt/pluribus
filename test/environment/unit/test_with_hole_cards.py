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
        env.inject_action("raise:1.1")
        replacement = (env.combo_cards[0, 0], env.combo_cards[0, 1])
        new_env = env.with_hole_cards(0, replacement)
        # Overlay is shared by reference (same dict object across the lineage).
        assert new_env.has_overlay_at_current_node

    def test_info_set_reflects_replaced_cards(self):
        # The central use of with_hole_cards: the range tracker patches
        # an opponent's hole, then queries the policy via env.info_set.
        # If info_set didn't pick up the patch, the per-combo Bayes
        # update would collapse to one value across all candidate
        # hands.  Build a tiny LUT mapping two distinct combos to two
        # different cluster ids and verify info_set reflects the patch.
        env = _env()
        actual_hole = tuple(sorted(int(c) for c in env.players[env.player_i]._cards))
        # Pick a replacement combo distinct from the dealt cards.
        replacement = None
        for i in range(env.n_combos):
            combo = tuple(sorted(int(c) for c in env.combo_cards[i]))
            if combo != actual_hole:
                # Also avoid clashing with the other seat's hole cards
                # (would make the lookup_cards tuple have duplicates).
                other = set(int(c) for c in env.players[1 - env.player_i]._cards)
                if not (set(combo) & other):
                    replacement = combo
                    break
        assert replacement is not None
        env.card_info_lut = {
            "pre_flop": {
                actual_hole: 111,
                replacement: 222,
            }
        }
        info_before = env.info_set
        new_env = env.with_hole_cards(env.player_i, replacement)
        info_after = new_env.info_set
        assert info_before != info_after
        # Sanity: both info_sets contain the expected cluster ids.
        assert '"cards_cluster":111' in info_before
        assert '"cards_cluster":222' in info_after
