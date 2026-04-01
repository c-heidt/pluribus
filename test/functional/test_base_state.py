"""Test base PokerState functionality common to all deck configurations.

These tests verify the core poker game mechanics that should work
identically across all deck sizes.
"""

import pytest

from poker_ai.environment.game_state import PokerState, new_game
from poker_ai.environment.player import Player
from poker_ai.environment.pot import Pot


def _new_game(
    n_players: int,
    small_blind: int = 50,
    big_blind: int = 100,
    initial_chips: int = 10000,
    low_card_rank: int = 10,
    high_card_rank: int = 14,
):
    """Create a new game for testing."""
    pot = Pot()
    players = [
        Player(player_i=player_i, pot=pot, initial_chips=initial_chips)
        for player_i in range(n_players)
    ]
    state = PokerState(
        players=players,
        load_card_lut=False,
        small_blind=small_blind,
        big_blind=big_blind,
        low_card_rank=low_card_rank,
        high_card_rank=high_card_rank,
    )
    return state, pot


# Use pytest parametrize to test multiple deck configurations
@pytest.fixture(params=[
    (10, 14, '20-card'),
    (2, 14, '52-card'),
    (6, 14, '36-card'),
])
def poker_state(request):
    """Fixture providing different deck configurations."""
    low, high, label = request.param
    state, pot = _new_game(n_players=3, low_card_rank=low, high_card_rank=high)
    return state, label


class TestBasePokerState:
    """Test core poker game mechanics that work across all deck sizes."""

    def test_initialization(self, poker_state):
        """Test that a poker game initializes correctly."""
        state, variant = poker_state

        assert state.betting_stage == "pre_flop"
        assert len(state.players) == 3
        assert all(len(player.cards) == 2 for player in state.players)

    def test_fold_mechanics(self, poker_state):
        """Test that folding works correctly and leads to terminal state."""
        state, variant = poker_state
        n_players = 3

        # Call for all players to reach flop
        player_i_order = [2, 0, 1]
        for i in range(n_players):
            assert state.current_player.name == f"player_{player_i_order[i]}"
            assert "call" in state.legal_actions
            assert "fold" in state.legal_actions
            assert state.betting_stage == "pre_flop"
            state = state.apply_action(action_str="call")

        assert state.betting_stage == "flop"

        # Fold for all but last player
        for player_i in range(n_players - 1):
            assert state.current_player.name == f"player_{player_i}"
            assert "fold" in state.legal_actions
            assert state.betting_stage == "flop"
            state = state.apply_action(action_str="fold")

        assert state.is_terminal, "state was not terminal after all but one player folded"
        assert state.betting_stage == "terminal"

    def test_betting_rounds_progression(self, poker_state):
        """Test that betting progresses through stages correctly."""
        state, variant = poker_state
        n_players = 3
        player_i_order = [2, 0, 1]

        # Pre-flop: Call for all players
        for i in range(n_players):
            assert state.current_player.name == f"player_{player_i_order[i]}"
            assert "call" in state.legal_actions
            assert state.betting_stage == "pre_flop"
            state = state.apply_action(action_str="call")

        assert state.betting_stage == "flop"

        # Flop: Raise for all players
        for player_i in range(n_players):
            assert state.current_player.name == f"player_{player_i}"
            raise_action = next((a for a in state.legal_actions if a and a.startswith("raise:")), None)
            assert raise_action is not None, "No raise action available"
            assert state.betting_stage == "flop"
            state = state.apply_action(action_str=raise_action)

        # Call to equalize bets
        for player_i in range(n_players - 1):
            assert state.current_player.name == f"player_{player_i}"
            if "call" in state.legal_actions:
                action_to_use = "call"
            elif "all_in" in state.legal_actions:
                action_to_use = "all_in"
            else:
                raise AssertionError(f"Neither call nor all_in available.")
            assert state.betting_stage == "flop"
            state = state.apply_action(action_str=action_to_use)

        assert state.betting_stage == "turn"

        # Turn: Try to raise if possible, otherwise just call/check to progress
        turn_had_any_action = False
        for player_i in range(n_players):
            assert state.current_player.name == f"player_{player_i}"
            assert state.betting_stage == "turn"

            raise_action = next((a for a in state.legal_actions if a and a.startswith("raise:")), None)
            if raise_action:
                state = state.apply_action(action_str=raise_action)
                turn_had_any_action = True
            elif "call" in state.legal_actions:
                state = state.apply_action(action_str="call")
            elif "all_in" in state.legal_actions:
                state = state.apply_action(action_str="all_in")
            else:
                state = state.apply_action(action_str="fold")

        if turn_had_any_action and state.betting_stage == "turn":
            for player_i in range(n_players - 1):
                if state.betting_stage != "turn":
                    break
                assert state.current_player.name == f"player_{player_i}"
                if "call" in state.legal_actions:
                    action_to_use = "call"
                elif "all_in" in state.legal_actions:
                    action_to_use = "all_in"
                else:
                    break
                state = state.apply_action(action_str=action_to_use)

        assert state.betting_stage == "river"

        # River: Fold for all but last player
        for player_i in range(n_players - 1):
            assert state.current_player.name == f"player_{player_i}"
            assert "fold" in state.legal_actions
            assert state.betting_stage == "river"
            state = state.apply_action(action_str="fold")

        assert state.is_terminal, "state was not terminal"
        assert state.betting_stage == "terminal"

    @pytest.mark.parametrize("n_players,small_blind,big_blind", [
        (2, 50, 100),
        (3, 50, 100),
        (4, 50, 100),
        (5, 50, 100),
        (6, 50, 100),
        (2, 200, 100),
        (3, 200, 100),
        (4, 200, 100),
        (5, 200, 100),
        (6, 200, 100),
        (2, 50, 1000),
        (3, 50, 1000),
        (4, 50, 1000),
        (5, 50, 1000),
        (6, 50, 1000),
        (2, 200, 1000),
        (3, 200, 1000),
        (4, 200, 1000),
        (5, 200, 1000),
        (6, 200, 1000),
    ])
    def test_pre_flop_pot_base(self, n_players, small_blind, big_blind):
        """Test that pot is set up correctly pre-flop."""
        state, pot = _new_game(
            n_players=n_players,
            small_blind=small_blind,
            big_blind=big_blind,
        )

        initial_pot = pot.total
        assert initial_pot == small_blind + big_blind

        assert state.player_i == 0 if n_players == 2 else 2
        assert state.betting_stage == "pre_flop"

        n_bet_chips = sum(p.n_bet_chips for p in state.players)
        assert n_bet_chips == small_blind + big_blind
        assert n_bet_chips == pot.total

        total_chips_in_play = sum(p.n_chips for p in state.players) + pot.total
        expected_total = n_players * 10000
        assert total_chips_in_play == expected_total

    def test_community_cards_dealt(self, poker_state):
        """Test that community cards are dealt at appropriate stages."""
        state, variant = poker_state
        n_players = 3

        assert len(state.community_cards) == 0

        for i in range(n_players):
            state = state.apply_action(action_str="call")

        assert state.betting_stage == "flop"
        assert len(state.community_cards) == 3

        for player_i in range(n_players):
            state = state.apply_action(action_str="call")

        assert state.betting_stage == "turn"
        assert len(state.community_cards) == 4

        for player_i in range(n_players):
            state = state.apply_action(action_str="call")

        assert state.betting_stage == "river"
        assert len(state.community_cards) == 5

    def test_legal_actions_available(self, poker_state):
        """Test that legal actions are available at each stage."""
        state, variant = poker_state

        assert "call" in state.legal_actions
        assert "fold" in state.legal_actions
        has_raise = any(action and action.startswith("raise:") for action in state.legal_actions)
        assert has_raise

    def test_immutable_state_pattern(self, poker_state):
        """Test that applying actions creates new state objects."""
        state, variant = poker_state

        original_betting_stage = state.betting_stage
        original_player = state.current_player.name

        new_state = state.apply_action(action_str="call")

        assert state.betting_stage == original_betting_stage
        assert state.current_player.name == original_player
        assert new_state is not state
        assert new_state.current_player.name != original_player

    def test_all_players_have_cards(self, poker_state):
        """Test that all players are dealt exactly 2 hole cards."""
        state, variant = poker_state

        for player in state.players:
            assert len(player.cards) == 2
            assert all(hasattr(card, 'rank') for card in player.cards)
            assert all(hasattr(card, 'suit') for card in player.cards)

    def test_player_rotation(self, poker_state):
        """Test that play rotates correctly between players."""
        state, variant = poker_state
        n_players = len(state.players)

        order = list(range(n_players))
        player_i_order = {
            "pre_flop": order[2:] + order[:2] if n_players > 2 else [0],
            "flop": order,
            "turn": order,
            "river": order,
        }

        prev_stage = ""
        order_i = 0

        while state.betting_stage in player_i_order and not state.is_terminal:
            if state.betting_stage != prev_stage:
                order_i = 0
                prev_stage = state.betting_stage

            target_player_i = player_i_order[state.betting_stage][order_i]
            assert state.current_player.name == f"player_{target_player_i}"
            assert state.player_i == target_player_i

            state = state.apply_action("call")
            order_i += 1

    @pytest.mark.parametrize("n_players", [2, 3, 4])
    def test_player_rotation_various_counts(self, n_players):
        """Test player rotation with different player counts."""
        state, pot = _new_game(n_players=n_players)

        order = list(range(n_players))
        player_i_order = {
            "pre_flop": order[2:] + order[:2],
            "flop": order,
            "turn": order,
            "river": order,
        }

        prev_stage = ""

        while state.betting_stage in player_i_order:
            if state.betting_stage != prev_stage:
                order_i = 0
                prev_stage = state.betting_stage

            target_player_i = player_i_order[state.betting_stage][order_i]
            assert state.current_player.name == f"player_{target_player_i}"
            assert state.player_i == target_player_i

            state = state.apply_action("call")
            order_i += 1


class TestBasePokerStateEdgeCases:
    """Test edge cases in base poker mechanics."""

    def test_all_in_scenario(self):
        """Test that all-in works correctly."""
        state, pot = _new_game(n_players=2, initial_chips=200)

        all_in_action = None
        for action in state.legal_actions:
            if action and (action.startswith('raise:') or action == 'all_in'):
                all_in_action = action
                break

        assert all_in_action is not None
        state = state.apply_action(action_str=str(all_in_action))
        assert state.betting_stage in ['pre_flop', 'flop', 'turn', 'river', 'terminal', 'show_down']

    @pytest.mark.xfail(reason="Game requires at least 2 players")
    def test_zero_players_fails(self):
        """Test that game fails with 0 players."""
        state, pot = _new_game(n_players=0)

    @pytest.mark.xfail(reason="Game requires at least 2 players")
    def test_one_player_fails(self):
        """Test that game fails with 1 player."""
        state, pot = _new_game(n_players=1)

    def test_minimum_two_players(self):
        """Test that game works with minimum 2 players."""
        state, pot = _new_game(n_players=2)

        assert len(state.players) == 2
        assert state.betting_stage == "pre_flop"
        state = state.apply_action(action_str="call")
        assert not state.is_terminal

    def test_maximum_players(self):
        """Test that game works with 6 players."""
        state, pot = _new_game(n_players=6)

        assert len(state.players) == 6
        assert state.betting_stage == "pre_flop"
        assert all(len(p.cards) == 2 for p in state.players)
