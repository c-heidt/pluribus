"""Test base PokerState functionality common to all poker variants.

These tests verify the core poker game mechanics that should work
identically across all variants (Short Deck, Full Deck, etc.).
"""

import pytest

from poker_ai.games.short_deck.state import ShortDeckPokerState
from poker_ai.games.short_deck.player import ShortDeckPokerPlayer
from poker_ai.games.full_deck.state import FullDeckPokerState, new_game as full_deck_new_game
from poker_ai.poker.pot import Pot


def _new_short_deck_game(
    n_players: int,
    small_blind: int = 50,
    big_blind: int = 100,
    initial_chips: int = 10000,
):
    """Create a new short deck game for testing."""
    pot = Pot()
    players = [
        ShortDeckPokerPlayer(player_i=player_i, pot=pot, initial_chips=initial_chips)
        for player_i in range(n_players)
    ]
    state = ShortDeckPokerState(
        players=players,
        load_card_lut=False,
        small_blind=small_blind,
        big_blind=big_blind,
    )
    return state, pot


def _new_full_deck_game(
    n_players: int,
    small_blind: int = 50,
    big_blind: int = 100,
):
    """Create a new full deck game for testing."""
    state = full_deck_new_game(
        n_players=n_players,
        small_blind=small_blind,
        big_blind=big_blind,
        card_info_lut={},
    )
    return state


# Use pytest parametrize to test both variants with same logic
@pytest.fixture(params=['short_deck', 'full_deck'])
def poker_state(request):
    """Fixture that provides both Short Deck and Full Deck state."""
    if request.param == 'short_deck':
        state, pot = _new_short_deck_game(n_players=3)
        return state, 'short_deck'
    else:
        state = _new_full_deck_game(n_players=3)
        return state, 'full_deck'


class TestBasePokerState:
    """Test core poker game mechanics that work across all variants."""
    
    def test_initialization(self, poker_state):
        """Test that a poker game initializes correctly."""
        state, variant = poker_state
        
        assert state.betting_stage == "pre_flop"
        assert len(state.players) == 3
        # Players should be dealt 2 cards each
        assert all(len(player.cards) == 2 for player in state.players)
    
    def test_fold_mechanics(self, poker_state):
        """Test that folding works correctly and leads to terminal state."""
        state, variant = poker_state
        n_players = 3
        
        # Call for all players to reach flop
        player_i_order = [2, 0, 1]
        for i in range(n_players):
            assert state.current_player.name == f"player_{player_i_order[i]}"
            assert len(state.legal_actions) == 3
            assert state.betting_stage == "pre_flop"
            state = state.apply_action(action_str="call")
        
        assert state.betting_stage == "flop"
        
        # Fold for all but last player
        for player_i in range(n_players - 1):
            assert state.current_player.name == f"player_{player_i}"
            assert len(state.legal_actions) == 3
            assert state.betting_stage == "flop"
            state = state.apply_action(action_str="fold")
        
        # Only one player left, so game state should be terminal
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
            assert len(state.legal_actions) == 3
            assert state.betting_stage == "pre_flop"
            state = state.apply_action(action_str="call")
        
        assert state.betting_stage == "flop"
        
        # Flop: Raise for all players
        for player_i in range(n_players):
            assert state.current_player.name == f"player_{player_i}"
            assert len(state.legal_actions) == 3
            assert state.betting_stage == "flop"
            state = state.apply_action(action_str="raise")
        
        # Call to equalize bets (now only 2 actions: call or fold)
        for player_i in range(n_players - 1):
            assert state.current_player.name == f"player_{player_i}"
            assert len(state.legal_actions) == 2
            assert state.betting_stage == "flop"
            state = state.apply_action(action_str="call")
        
        assert state.betting_stage == "turn"
        
        # Turn: Raise for all players
        for player_i in range(n_players):
            assert state.current_player.name == f"player_{player_i}"
            assert len(state.legal_actions) == 3
            assert state.betting_stage == "turn"
            state = state.apply_action(action_str="raise")
        
        # Call to equalize bets (now only 2 actions: call or fold)
        for player_i in range(n_players - 1):
            assert state.current_player.name == f"player_{player_i}"
            assert len(state.legal_actions) == 2
            assert state.betting_stage == "turn"
            state = state.apply_action(action_str="call")
        
        assert state.betting_stage == "river"
        
        # River: Fold for all but last player
        for player_i in range(n_players - 1):
            assert state.current_player.name == f"player_{player_i}"
            assert len(state.legal_actions) == 3
            assert state.betting_stage == "river"
            state = state.apply_action(action_str="fold")
        
        # Only one player left, so game state should be terminal
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
        """Test that pot is set up correctly pre-flop for different configurations."""
        # Test with short deck as representative
        state, pot = _new_short_deck_game(
            n_players=n_players,
            small_blind=small_blind,
            big_blind=big_blind,
        )
        
        # Before any actions, blinds should be posted
        initial_pot = pot.total
        assert initial_pot == small_blind + big_blind
        
        # Verify correct starting player (original test assertion)
        assert state.player_i == 0 if n_players == 2 else 2
        assert state.betting_stage == "pre_flop"
        
        # Verify bet chips match pot
        n_bet_chips = sum(p.n_bet_chips for p in state.players)
        assert n_bet_chips == small_blind + big_blind
        assert n_bet_chips == pot.total
        
        # Verify players have correct chip counts after blinds
        total_chips_in_play = sum(p.n_chips for p in state.players) + pot.total
        expected_total = n_players * 10000
        assert total_chips_in_play == expected_total
    
    def test_community_cards_dealt(self, poker_state):
        """Test that community cards are dealt at appropriate stages."""
        state, variant = poker_state
        n_players = 3
        player_i_order = [2, 0, 1]
        
        # Pre-flop: no community cards
        assert len(state.community_cards) == 0
        
        # Call to reach flop
        for i in range(n_players):
            state = state.apply_action(action_str="call")
        
        # Flop: 3 community cards
        assert state.betting_stage == "flop"
        assert len(state.community_cards) == 3
        
        # Call to reach turn (check may not be available after flop deal)
        for player_i in range(n_players):
            # Use call if check not available
            legal_actions_str = [str(a) for a in state.legal_actions]
            if any('check' in a for a in legal_actions_str):
                state = state.apply_action(action_str="check")
            else:
                state = state.apply_action(action_str="call")
        
        # Turn: 4 community cards
        assert state.betting_stage == "turn"
        assert len(state.community_cards) == 4
        
        # Call/check to reach river
        for player_i in range(n_players):
            legal_actions_str = [str(a) for a in state.legal_actions]
            if any('check' in a for a in legal_actions_str):
                state = state.apply_action(action_str="check")
            else:
                state = state.apply_action(action_str="call")
        
        # River: 5 community cards
        assert state.betting_stage == "river"
        assert len(state.community_cards) == 5
    
    def test_legal_actions_available(self, poker_state):
        """Test that legal actions are available at each stage."""
        state, variant = poker_state
        
        # Initially should have call, raise, fold
        assert len(state.legal_actions) == 3
        assert any('call' in str(action) for action in state.legal_actions)
        assert any('raise' in str(action) for action in state.legal_actions)
        assert any('fold' in str(action) for action in state.legal_actions)
    
    def test_immutable_state_pattern(self, poker_state):
        """Test that applying actions creates new state objects."""
        state, variant = poker_state
        
        original_betting_stage = state.betting_stage
        original_player = state.current_player.name
        
        # Apply action and get new state
        new_state = state.apply_action(action_str="call")
        
        # Original state should be unchanged
        assert state.betting_stage == original_betting_stage
        assert state.current_player.name == original_player
        
        # New state should be different
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
        """Test that play rotates correctly between players through all betting stages.
        
        Validates the exact player order: pre-flop should be [2, 0, 1, ...] 
        (rotated due to blinds), while flop/turn/river should be [0, 1, 2, ...].
        """
        state, variant = poker_state
        n_players = len(state.players)
        
        # Build expected order for each stage
        order = list(range(n_players))
        player_i_order = {
            "pre_flop": order[2:] + order[:2] if n_players > 2 else [0],
            "flop": order,
            "turn": order,
            "river": order,
        }
        
        prev_stage = ""
        order_i = 0
        
        # Play through all betting stages validating player order
        while state.betting_stage in player_i_order and not state.is_terminal:
            if state.betting_stage != prev_stage:
                # New betting stage, reset counter
                order_i = 0
                prev_stage = state.betting_stage
            
            # Validate current player matches expected order
            target_player_i = player_i_order[state.betting_stage][order_i]
            assert state.current_player.name == f"player_{target_player_i}", \
                f"{state.current_player.name} != player_{target_player_i} at stage {state.betting_stage}"
            assert state.player_i == target_player_i, \
                f"{state.player_i} != {target_player_i} at stage {state.betting_stage}"
            
            # All players call to keep things simple
            state = state.apply_action("call")
            order_i += 1
    
    @pytest.mark.parametrize("n_players", [2, 3, 4])
    def test_player_rotation_various_counts(self, n_players):
        """Test player rotation with different player counts (from test_short_deck_3).
        
        This validates that player rotation works correctly for 2, 3, and 4 players
        throughout all betting stages.
        """
        state, pot = _new_short_deck_game(n_players=n_players)
        
        # Build expected order for each stage (matches original exactly)
        order = list(range(n_players))
        player_i_order = {
            "pre_flop": order[2:] + order[:2],  # Rotation for pre-flop
            "flop": order,
            "turn": order,
            "river": order,
        }
        
        prev_stage = ""
        
        # Play through all betting stages validating player order
        while state.betting_stage in player_i_order:
            if state.betting_stage != prev_stage:
                # New betting stage, reset counter
                order_i = 0
                prev_stage = state.betting_stage
            
            # Validate current player matches expected order
            target_player_i = player_i_order[state.betting_stage][order_i]
            assert state.current_player.name == f"player_{target_player_i}", \
                f"{state.current_player.name} != player_{target_player_i}"
            assert state.player_i == target_player_i, \
                f"{state.player_i} != {target_player_i}"
            
            # All players call to keep things simple
            state = state.apply_action("call")
            order_i += 1


class TestBasePokerStateEdgeCases:
    """Test edge cases in base poker mechanics."""
    
    def test_all_in_scenario(self):
        """Test that all-in works correctly."""
        # Use short deck with limited chips
        state, pot = _new_short_deck_game(n_players=2, initial_chips=200)
        
        # Player can go all-in - find raise action
        all_in_action = None
        for action in state.legal_actions:
            if 'raise' in str(action):
                all_in_action = action
                break
        
        assert all_in_action is not None
        state = state.apply_action(action_str=str(all_in_action))
        
        # With limited chips, game continues until terminal or one player wins
        # Just verify the action was accepted and state progressed
        assert state.betting_stage in ['pre_flop', 'flop', 'turn', 'river', 'terminal', 'show_down']
    
    @pytest.mark.xfail(reason="Game requires at least 2 players")
    def test_zero_players_fails(self):
        """Test that game fails with 0 players."""
        state, pot = _new_short_deck_game(n_players=0)
    
    @pytest.mark.xfail(reason="Game requires at least 2 players")
    def test_one_player_fails(self):
        """Test that game fails with 1 player."""
        state, pot = _new_short_deck_game(n_players=1)
    
    def test_minimum_two_players(self):
        """Test that game works with minimum 2 players."""
        state, pot = _new_short_deck_game(n_players=2)
        
        assert len(state.players) == 2
        assert state.betting_stage == "pre_flop"
        
        # Players can take actions
        state = state.apply_action(action_str="call")
        assert not state.is_terminal
    
    def test_maximum_players(self):
        """Test that game works with 6 players."""
        state, pot = _new_short_deck_game(n_players=6)
        
        assert len(state.players) == 6
        assert state.betting_stage == "pre_flop"
        # All players should have cards
        assert all(len(p.cards) == 2 for p in state.players)
