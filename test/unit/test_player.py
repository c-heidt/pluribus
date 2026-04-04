"""Unit tests for poker_ai/environment/player.py (Player class).

Covers construction defaults, chip management, betting actions,
and all properties and flags.
"""

import pytest

from poker_ai.environment.player import Player
from poker_ai.environment.pot import Pot
from poker_ai.environment.utils import make_card


@pytest.fixture
def player():
    return Player(player_i=0, initial_chips=10000)


@pytest.fixture
def pot():
    return Pot(2)


class TestConstruction:
    def test_default_name(self):
        p = Player(0)
        assert p.name == "player_0"

    def test_custom_name(self):
        p = Player(3, name="Alice")
        assert p.name == "Alice"

    def test_initial_chips(self):
        p = Player(0, initial_chips=5000)
        assert p.n_chips == 5000

    def test_n_bet_chips_zero(self, player):
        assert player.n_bet_chips == 0

    def test_is_active_true(self, player):
        assert player.is_active is True

    def test_all_flags_false(self, player):
        assert player.is_small_blind is False
        assert player.is_big_blind is False
        assert player.is_dealer is False
        assert player.is_turn is False

    def test_cards_empty(self, player):
        assert player.cards == ()

    def test_player_i(self):
        p = Player(7)
        assert p.player_i == 7

    def test_repr_contains_name(self, player):
        assert "player_0" in repr(player)

    def test_repr_contains_chips(self, player):
        assert "10000" in repr(player)


class TestAddChips:
    def test_add_chips_increases_stack(self, player):
        player.add_chips(500)
        assert player.n_chips == 10500

    def test_add_chips_zero(self, player):
        player.add_chips(0)
        assert player.n_chips == 10000


class TestAddToPot:
    def test_normal_add(self, player, pot):
        returned = player.add_to_pot(pot, 100)
        assert returned == 100
        assert player.n_chips == 9900
        assert player.n_bet_chips == 100
        assert pot.total == 100

    def test_zero_add_no_state_change(self, player, pot):
        returned = player.add_to_pot(pot, 0)
        assert returned == 0
        assert player.n_chips == 10000
        assert player.n_bet_chips == 0
        assert pot.total == 0

    def test_all_in_cap(self, pot):
        p = Player(0, initial_chips=200)
        returned = p.add_to_pot(pot, 500)
        assert returned == 200
        assert p.n_chips == 0
        assert p.n_bet_chips == 200

    def test_negative_raises(self, player, pot):
        with pytest.raises(ValueError):
            player.add_to_pot(pot, -1)

    def test_cumulative_n_bet_chips(self, player, pot):
        player.add_to_pot(pot, 100)
        player.add_to_pot(pot, 200)
        assert player.n_bet_chips == 300


class TestFold:
    def test_fold_deactivates(self, player):
        player.fold()
        assert player.is_active is False

    def test_fold_keeps_cards(self, player):
        player._cards = (make_card(14, "spades"),)
        player.fold()
        assert len(player.cards) == 1


class TestCall:
    def test_call_equalizes_bet(self):
        players = [Player(0, 10000), Player(1, 10000)]
        pot = Pot(2)
        # Player 1 has already bet 200
        players[1].add_to_pot(pot, 200)
        # Player 0 calls
        players[0].call(players, pot)
        assert players[0].n_bet_chips == 200
        assert players[0].n_chips == 9800

    def test_call_all_in_is_noop(self):
        p = Player(0, initial_chips=0)
        p._is_active = True
        pot = Pot(2)
        p_other = Player(1, 10000)
        p_other.add_to_pot(pot, 500)
        chips_before = p.n_chips
        bet_before = p.n_bet_chips
        p.call([p, p_other], pot)
        assert p.n_chips == chips_before
        assert p.n_bet_chips == bet_before


class TestRaiseTo:
    def test_raise_to_adds_chips(self):
        p = Player(0, 10000)
        pot = Pot(1)
        p.raise_to(pot, 300)
        assert p.n_bet_chips == 300
        assert p.n_chips == 9700


class TestProperties:
    def test_is_active_setter(self, player):
        player.is_active = False
        assert player._is_active is False

    def test_is_all_in_true_when_active_and_broke(self):
        p = Player(0, initial_chips=0)
        p._is_active = True
        assert p.is_all_in is True

    def test_is_all_in_false_when_inactive(self):
        p = Player(0, initial_chips=0)
        p._is_active = False
        assert p.is_all_in is False

    def test_is_all_in_false_when_has_chips(self, player):
        assert player.is_all_in is False

    def test_cards_property_returns_tuple(self, player):
        player._cards = (make_card(14, "spades"), make_card(13, "hearts"))
        assert isinstance(player.cards, tuple)
        assert len(player.cards) == 2
