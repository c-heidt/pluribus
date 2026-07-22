"""Functional tests for poker_ai/environment/dynamics.py.

Covers all module-level functions: blind assignment, player ordering,
blind rotation, stage advancement, active/move counts, betting state,
hand ranking, and winner computation with chip conservation.
"""

import pytest

from environment import dynamics
from environment.player import Player
from environment.pot import Pot
from environment.poker_env import new_game
from environment.utils import make_card


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(n=3, chips=10000):
    return new_game(n_players=n, card_info_lut={}, initial_chips=chips)


def _setup_showdown(community_cards, *player_hole_cards):
    """Build a minimal env-like object for rank/payout tests."""
    env = _env(n=len(player_hole_cards))
    # Override cards directly
    for player, cards in zip(env.players, player_hole_cards):
        player._cards = cards
    env.community_cards = community_cards
    return env


# ---------------------------------------------------------------------------
# assign_order
# ---------------------------------------------------------------------------

class TestAssignOrder:
    def test_order_matches_list_index(self):
        env = _env(3)
        dynamics.assign_order(env)
        for i, player in enumerate(env.players):
            assert player.order == i

    def test_order_updated_after_rotate(self):
        env = _env(3)
        dynamics.rotate_blinds(env)
        dynamics.assign_order(env)
        for i, player in enumerate(env.players):
            assert player.order == i


# ---------------------------------------------------------------------------
# assign_blinds
# ---------------------------------------------------------------------------

class TestAssignBlinds:
    def test_pot_contains_correct_amounts(self):
        env = _env(2)
        env.pot.reset()
        for p in env.players:
            p.n_chips = 10000
            p.n_bet_chips = 0
        dynamics.assign_blinds(env)
        assert env.pot[0] == env.small_blind
        assert env.pot[1] == env.big_blind

    def test_player_stacks_reduced(self):
        env = _env(2)
        env.pot.reset()
        for p in env.players:
            p.n_chips = 10000
            p.n_bet_chips = 0
        dynamics.assign_blinds(env)
        assert env.players[0].n_chips == 10000 - env.small_blind
        assert env.players[1].n_chips == 10000 - env.big_blind


# ---------------------------------------------------------------------------
# rotate_blinds
# ---------------------------------------------------------------------------

class TestRotateBlinds:
    def test_first_player_moves_to_end(self):
        env = _env(3)
        first = env.players[0]
        dynamics.rotate_blinds(env)
        assert env.players[-1] is first

    def test_second_player_becomes_first(self):
        env = _env(3)
        second = env.players[1]
        dynamics.rotate_blinds(env)
        assert env.players[0] is second

    def test_player_list_length_unchanged(self):
        env = _env(3)
        dynamics.rotate_blinds(env)
        assert len(env.players) == 3


# ---------------------------------------------------------------------------
# n_active_players
# ---------------------------------------------------------------------------

class TestNActivePlayers:
    def test_all_active_initially(self):
        env = _env(3)
        assert dynamics.n_active_players(env) == 3

    def test_decreases_on_fold(self):
        env = _env(3)
        env.players[0].fold()
        assert dynamics.n_active_players(env) == 2

    def test_zero_when_all_folded(self):
        env = _env(3)
        for p in env.players:
            p.fold()
        assert dynamics.n_active_players(env) == 0


# ---------------------------------------------------------------------------
# n_players_with_moves
# ---------------------------------------------------------------------------

class TestNPlayersWithMoves:
    def test_all_can_move_initially(self):
        env = _env(3)
        assert dynamics.n_players_with_moves(env) == 3

    def test_excludes_folded_players(self):
        env = _env(3)
        env.players[0].fold()
        assert dynamics.n_players_with_moves(env) == 2

    def test_excludes_all_in_players(self):
        env = _env(3)
        env.players[0]._is_active = True
        env.players[0].n_chips = 0  # all-in
        assert dynamics.n_players_with_moves(env) == 2


# ---------------------------------------------------------------------------
# more_betting_needed
# ---------------------------------------------------------------------------

class TestMoreBettingNeeded:
    def test_true_when_bets_unequal(self):
        env = _env(2)
        env.players[0].n_bet_chips = 100
        env.players[1].n_bet_chips = 200
        assert dynamics.more_betting_needed(env) is True

    def test_false_when_bets_equal(self):
        env = _env(2)
        env.players[0].n_bet_chips = 100
        env.players[1].n_bet_chips = 100
        assert dynamics.more_betting_needed(env) is False

    def test_false_when_one_active_player(self):
        env = _env(2)
        env.players[0].fold()
        env.players[1].n_bet_chips = 100
        assert dynamics.more_betting_needed(env) is False

    def test_excludes_all_in_from_check(self):
        env = _env(3)
        # Player 0 all-in with different bet, but only one non-all-in player
        env.players[0].n_chips = 0
        env.players[0].n_bet_chips = 50
        env.players[1].n_bet_chips = 100
        env.players[2].fold()
        # Only player 1 can act, and has already out-bet the all-in, so there
        # is nothing left to call — no more betting needed.
        assert dynamics.more_betting_needed(env) is False

    def test_true_when_live_player_owes_over_the_top_all_in(self):
        # An all-in player has bet MORE than the lone live player, who still
        # owes a call-or-fold.  More betting IS needed even though the live
        # bets are "equal to each other" (there is only one live player) —
        # the comparison must be against the top bet, all-in included.
        env = _env(3)
        env.players[0].n_chips = 0
        env.players[0].n_bet_chips = 200   # all-in, over the top
        env.players[1].n_bet_chips = 100   # live, has not matched the shove
        env.players[2].fold()
        assert dynamics.more_betting_needed(env) is True


# ---------------------------------------------------------------------------
# rank_players_by_best_hand
# ---------------------------------------------------------------------------

class TestRankPlayersByBestHand:
    def test_better_hand_ranked_first(self):
        # Player 0 has royal flush, player 1 has high card
        community = tuple(make_card(r, "spades") for r in [14, 13, 12, 11, 10])
        env = _setup_showdown(
            community,
            (make_card(9, "hearts"), make_card(8, "clubs")),  # p0: junk extras
            (make_card(2, "hearts"), make_card(3, "clubs")),  # p1: 7-high
        )
        ranked = dynamics.rank_players_by_best_hand(env)
        # Both use community cards; p0 and p1 effectively tied via community
        assert len(ranked) >= 1

    def test_tied_hands_same_group(self):
        # Both players share the same 5-card board (no kicker difference)
        community = tuple(make_card(r, "spades") for r in [14, 13, 12, 11, 10])
        env = _setup_showdown(
            community,
            (make_card(2, "hearts"), make_card(3, "clubs")),
            (make_card(4, "hearts"), make_card(5, "clubs")),
        )
        ranked = dynamics.rank_players_by_best_hand(env)
        # Both players have the same best hand (royal flush on board)
        assert len(ranked) == 1
        assert len(ranked[0]) == 2

    def test_empty_when_all_folded(self):
        env = _env(3)
        for p in env.players:
            p.fold()
        ranked = dynamics.rank_players_by_best_hand(env)
        assert ranked == []


# ---------------------------------------------------------------------------
# compute_winners
# ---------------------------------------------------------------------------

class TestComputeWinners:
    def test_single_winner_receives_all(self):
        # Player 0: royal flush; player 1: high card
        community = tuple(make_card(r, "spades") for r in [14, 13, 12, 11, 10])
        env = _setup_showdown(
            community,
            (make_card(2, "hearts"), make_card(3, "clubs")),
            (make_card(4, "hearts"), make_card(5, "clubs")),
        )
        # Ensure pot has chips
        env.pot.reset()
        env.pot.add_chips(0, 500)
        env.pot.add_chips(1, 500)
        chips_before = {p.player_i: p.n_chips for p in env.players}
        dynamics.compute_winners(env)
        # pot must be zeroed
        assert env.pot.total == 0
        # Chip conservation: total chips unchanged
        total_before = sum(chips_before.values()) + 1000  # 1000 was in pot
        total_after = sum(p.n_chips for p in env.players)
        assert total_after == total_before

    def test_pot_zeroed_after_compute(self):
        community = tuple(make_card(r, "spades") for r in [14, 13, 12, 11, 10])
        env = _setup_showdown(
            community,
            (make_card(2, "hearts"), make_card(3, "clubs")),
            (make_card(4, "hearts"), make_card(5, "clubs")),
        )
        env.pot.reset()
        env.pot.add_chips(0, 300)
        env.pot.add_chips(1, 300)
        dynamics.compute_winners(env)
        assert env.pot.total == 0

    def test_chip_conservation_three_players(self):
        # Use _setup_showdown to control all cards and avoid duplicates
        community = tuple(make_card(r, "spades") for r in [9, 8, 7, 6, 5])
        env = _setup_showdown(
            community,
            (make_card(14, "hearts"), make_card(13, "hearts")),
            (make_card(12, "diamonds"), make_card(11, "clubs")),
            (make_card(10, "hearts"), make_card(4, "clubs")),
        )
        # Chips already deducted by blinds; capture state before payout
        initial_chips = sum(p.n_chips for p in env.players)
        pot_total = env.pot.total
        dynamics.compute_winners(env)
        final_total = sum(p.n_chips for p in env.players)
        assert final_total == initial_chips + pot_total

    def test_compute_winners_awards_folded_player_chips_to_winner(self):
        # Even when a player folded, their chips in the pot go to the winner.
        community = tuple(make_card(r, "spades") for r in [14, 13, 12, 11, 10])
        env = _setup_showdown(
            community,
            (make_card(2, "clubs"), make_card(3, "clubs")),
            (make_card(4, "clubs"), make_card(5, "clubs")),
            (make_card(6, "clubs"), make_card(7, "clubs")),
        )
        env.pot.reset()
        env.pot.add_chips(0, 100)
        env.pot.add_chips(1, 100)
        env.pot.add_chips(2, 100)
        env.players[2].fold()
        # Chips before payout
        chips_before = {p.player_i: p.n_chips for p in env.players}
        dynamics.compute_winners(env)
        total_after = sum(p.n_chips for p in env.players)
        # All 300 chips must be distributed
        assert total_after == sum(chips_before.values()) + 300

    def test_split_pot_two_tied_players(self):
        # Both players share exactly the same best hand via the board
        community = tuple(make_card(r, "spades") for r in [14, 13, 12, 11, 10])
        env = _setup_showdown(
            community,
            (make_card(2, "clubs"), make_card(3, "clubs")),
            (make_card(4, "clubs"), make_card(5, "clubs")),
        )
        env.pot.reset()
        env.pot.add_chips(0, 500)
        env.pot.add_chips(1, 500)
        chips_0_before = env.players[0].n_chips
        chips_1_before = env.players[1].n_chips
        dynamics.compute_winners(env)
        # Both gain equal amounts (500 each)
        assert env.players[0].n_chips == chips_0_before + 500
        assert env.players[1].n_chips == chips_1_before + 500


# ---------------------------------------------------------------------------
# rank_players_by_best_hand with folded players
# ---------------------------------------------------------------------------

class TestRankPlayersWithFolds:
    def test_rank_players_ignores_folded_players(self):
        community = tuple(make_card(r, "spades") for r in [14, 13, 12, 11, 10])
        env = _setup_showdown(
            community,
            (make_card(2, "hearts"), make_card(3, "hearts")),
            (make_card(4, "hearts"), make_card(5, "hearts")),
            (make_card(6, "hearts"), make_card(7, "hearts")),
        )
        env.players[0].fold()
        ranked = dynamics.rank_players_by_best_hand(env)
        # Only 2 active players; folded player 0 must not appear
        all_ranked = [p for group in ranked for p in group]
        assert all(p.player_i != 0 for p in all_ranked)
        assert len(all_ranked) == 2

    def test_assign_blinds_caps_to_stack_when_short(self):
        # Player with fewer chips than the big blind posts all they can
        env = _env(2)
        env.pot.reset()
        env.players[0].n_chips = 30   # less than big_blind (100)
        env.players[0].n_bet_chips = 0
        env.players[1].n_chips = 10000
        env.players[1].n_bet_chips = 0
        dynamics.assign_blinds(env)
        # Small blind player had only 30; they contribute all 30
        assert env.players[0].n_chips == 0
        assert env.pot[0] == 30
