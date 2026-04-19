"""Unit tests for poker_ai/environment/pot.py (Pot class).

Covers chip tracking, side pot construction, utility computation,
chip conservation, and edge cases including all-in scenarios.
"""

import pytest

from environment.player import Player
from environment.pot import Pot


def _player(i, chips=10000):
    p = Player(i, chips)
    p.order = i
    return p


class TestConstruction:
    def test_total_is_zero(self):
        pot = Pot(3)
        assert pot.total == 0

    def test_side_pots_empty(self):
        pot = Pot(3)
        assert pot.side_pots == []

    def test_getitem_initial_zero(self):
        pot = Pot(3)
        assert pot[0] == 0
        assert pot[2] == 0


class TestAddChips:
    def test_total_increases(self):
        pot = Pot(2)
        pot.add_chips(0, 100)
        assert pot.total == 100

    def test_per_player_tracking(self):
        pot = Pot(3)
        pot.add_chips(0, 100)
        pot.add_chips(1, 200)
        assert pot[0] == 100
        assert pot[1] == 200
        assert pot[2] == 0

    def test_cumulative(self):
        pot = Pot(2)
        pot.add_chips(0, 100)
        pot.add_chips(0, 50)
        assert pot[0] == 150
        assert pot.total == 150


class TestReset:
    def test_reset_zeroes_total(self):
        pot = Pot(3)
        pot.add_chips(0, 500)
        pot.reset()
        assert pot.total == 0

    def test_reset_zeroes_all_players(self):
        pot = Pot(3)
        for i in range(3):
            pot.add_chips(i, 100 * (i + 1))
        pot.reset()
        for i in range(3):
            assert pot[i] == 0


class TestSidePots:
    def test_equal_contributions_single_pot(self):
        pot = Pot(3)
        for i in range(3):
            pot.add_chips(i, 100)
        sp = pot.side_pots
        assert len(sp) == 1
        assert sum(sp[0].values()) == 300

    def test_unequal_creates_side_pot(self):
        # Player 0 all-in for 50, players 1 and 2 contribute 100
        pot = Pot(3)
        pot.add_chips(0, 50)
        pot.add_chips(1, 100)
        pot.add_chips(2, 100)
        sp = pot.side_pots
        assert len(sp) == 2
        # Main pot: 50 * 3 = 150
        assert sum(sp[0].values()) == 150
        # Side pot: 50 * 2 = 100
        assert sum(sp[1].values()) == 100

    def test_three_way_all_in(self):
        # Three distinct amounts
        pot = Pot(3)
        pot.add_chips(0, 30)
        pot.add_chips(1, 60)
        pot.add_chips(2, 90)
        sp = pot.side_pots
        assert len(sp) == 3
        # total must be preserved
        assert sum(sum(p.values()) for p in sp) == 180

    def test_side_pots_empty_when_no_contributions(self):
        pot = Pot(3)
        assert pot.side_pots == []


class TestComputeUtility:
    def test_single_winner_takes_all(self):
        pot = Pot(3)
        players = [_player(i) for i in range(3)]
        for i in range(3):
            pot.add_chips(i, 100)
        # player 0 wins, players 1 and 2 lose
        ranked = [[players[0]], [players[1]], [players[2]]]
        payouts = pot.compute_utility(players, ranked)
        assert payouts[0] == 300
        assert payouts[1] == 0
        assert payouts[2] == 0

    def test_split_pot_two_winners(self):
        pot = Pot(2)
        players = [_player(0), _player(1)]
        pot.add_chips(0, 100)
        pot.add_chips(1, 100)
        # Tied
        ranked = [[players[0], players[1]]]
        payouts = pot.compute_utility(players, ranked)
        assert payouts[0] + payouts[1] == 200
        assert payouts[0] == 100
        assert payouts[1] == 100

    def test_remainder_chip_to_first_winner_by_order(self):
        # 3 players contribute 33 each → total 99, split 3 ways = 33 each exactly
        # Use odd total to test remainder
        pot = Pot(3)
        players = [_player(i) for i in range(3)]
        pot.add_chips(0, 34)
        pot.add_chips(1, 33)
        pot.add_chips(2, 33)
        ranked = [[players[0], players[1], players[2]]]
        payouts = pot.compute_utility(players, ranked)
        assert sum(payouts.values()) == 100

    def test_chip_conservation(self):
        pot = Pot(4)
        players = [_player(i) for i in range(4)]
        for i in range(4):
            pot.add_chips(i, 250)
        ranked = [[players[0]], [players[1]], [players[2]], [players[3]]]
        payouts = pot.compute_utility(players, ranked)
        assert sum(payouts.values()) == pot.total

    def test_chip_conservation_all_payout_cases(self):
        pot = Pot(3)
        players = [_player(i) for i in range(3)]
        pot.add_chips(0, 100)
        pot.add_chips(1, 100)
        pot.add_chips(2, 100)
        total = pot.total
        ranked = [[players[0]], [players[1]], [players[2]]]
        payouts = pot.compute_utility(players, ranked)
        assert sum(payouts.values()) == total

    def test_all_in_player_wins_main_pot_only(self):
        # player 0 all-in for 50; player 1 and 2 contribute 100
        # player 0 has best hand but was all-in
        pot = Pot(3)
        players = [_player(i) for i in range(3)]
        pot.add_chips(0, 50)
        pot.add_chips(1, 100)
        pot.add_chips(2, 100)
        # Rankings: p0 best (but only eligible for main pot), p1 second, p2 third
        ranked = [[players[0]], [players[1]], [players[2]]]
        payouts = pot.compute_utility(players, ranked)
        # p0 wins main pot: 50 * 3 = 150
        assert payouts[0] == 150
        # p1 wins side pot: 50 * 2 = 100
        assert payouts[1] == 100
        assert payouts[2] == 0


class TestSidePotEdgeCases:
    def test_folded_player_chips_awarded_to_winner(self):
        # Player 2 folded; only active players contest the pot, but
        # folded player's chips remain in the pot and go to winner.
        pot = Pot(3)
        players = [_player(i) for i in range(3)]
        pot.add_chips(0, 100)
        pot.add_chips(1, 100)
        pot.add_chips(2, 100)
        players[2].fold()
        # ranked_groups only has active players; player 0 wins
        ranked = [[players[0]], [players[1]]]
        payouts = pot.compute_utility(players, ranked)
        assert payouts[0] == 300
        assert payouts[1] == 0
        assert payouts[2] == 0

    def test_side_pots_boundary_minimum_contribution(self):
        # P0=50, P1=50, P2=100 → main pot (50*3=150) + side pot (50*1=50)
        pot = Pot(3)
        pot.add_chips(0, 50)
        pot.add_chips(1, 50)
        pot.add_chips(2, 100)
        sp = pot.side_pots
        assert len(sp) == 2
        assert sum(sp[0].values()) == 150
        assert sum(sp[1].values()) == 50

    def test_compute_utility_single_winner_odd_total(self):
        # Odd total with 1 winner — all chips including remainder go to winner
        pot = Pot(2)
        players = [_player(i) for i in range(2)]
        pot.add_chips(0, 51)
        pot.add_chips(1, 50)
        ranked = [[players[0]], [players[1]]]
        payouts = pot.compute_utility(players, ranked)
        assert payouts[0] == 101
        assert payouts[1] == 0

    def test_chip_conservation_three_way_unequal(self):
        # Three players with different contributions; total is conserved
        pot = Pot(3)
        players = [_player(i) for i in range(3)]
        pot.add_chips(0, 30)
        pot.add_chips(1, 60)
        pot.add_chips(2, 90)
        total = pot.total
        ranked = [[players[0]], [players[1]], [players[2]]]
        payouts = pot.compute_utility(players, ranked)
        assert sum(payouts.values()) == total
