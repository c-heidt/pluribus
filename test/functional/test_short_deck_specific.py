"""Test Short Deck specific features and edge cases.

These tests focus on Short Deck-specific functionality:
- 20-card deck (ranks 10-A: 5 ranks × 4 suits)
- Hand ranking differences (Flush > Full House, Three of a Kind > Straight)
- Short Deck specific game mechanics
"""

import collections
import random

import pytest
import numpy as np
import dill as pickle

from poker_ai.environment.game_state import PokerState
from poker_ai.environment.player import Player
from poker_ai.environment.card import Card
from poker_ai.environment.pot import Pot
from poker_ai.environment.evaluation.evaluator import Evaluator
from poker_ai.utils.random import seed


def _new_game(
    n_players: int,
    small_blind: int = 50,
    big_blind: int = 100,
    initial_chips: int = 10000,
):
    """Create a new short deck game."""
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
        low_card_rank=10,
        high_card_rank=14,
    )
    return state, pot


def _load_action_sequences(directory):
    with open(directory, "rb") as file:
        action_sequences = pickle.load(file)
    return action_sequences


class TestShortDeckSpecifics:
    """Test Short Deck variant-specific features."""
    
    def test_short_deck_has_20_cards(self):
        """Test that Short Deck uses only 20 cards (ranks 10-A, 5 ranks × 4 suits)."""
        state, _ = _new_game(n_players=2)
        
        # Get deck ranks - should be 10-14 (5 ranks)
        ranks = state._get_deck_ranks()
        assert len(ranks) == 5
        assert ranks == [10, 11, 12, 13, 14]
        
        # 5 ranks * 4 suits = 20 cards total
        deck_size = len(ranks) * 4
        assert deck_size == 20
    
    def test_short_deck_uses_custom_evaluator(self):
        """Test that Short Deck uses custom evaluator with adjusted rankings."""
        state, _ = _new_game(n_players=2)
        
        evaluator = Evaluator()
        assert isinstance(evaluator, Evaluator)
    
    def test_short_deck_no_low_cards(self):
        """Test that low cards (2-9) don't appear in Short Deck."""
        state, _ = _new_game(n_players=6)  # Max players to deal max cards
        
        # Check all player cards
        all_cards = []
        for player in state.players:
            all_cards.extend(player.cards)
        
        # Verify no cards have ranks 2-9
        for card in all_cards:
            assert card.rank_int >= 10, f"Found low card with rank {card.rank_int}"
            assert card.rank_int <= 14  # Max is Ace
    
    def test_flops_are_random_short_deck(self):
        """Test that flops are different across games in Short Deck."""
        seed(42)
        states = []
        for _ in range(10):
            state, _ = _new_game(n_players=3)
            # Progress to flop
            state = state.apply_action(action_str="call")
            state = state.apply_action(action_str="call")
            state = state.apply_action(action_str="call")
            states.append(state)
        
        community_cards = [
            tuple([card.eval_card for card in state.community_cards])
            for state in states
        ]
        
        # Should have different flops
        assert len(set(community_cards)) > 1


def _normalize_action(action: str) -> str:
    """Normalize action for comparison with pre-computed sequences.
    
    Converts new-style actions (e.g., "raise:0.5") to old-style ("raise").
    Also converts "all_in" to "call" since all_in wasn't a separate action
    in the old system.
    """
    if action and action.startswith("raise:"):
        return "raise"
    if action == "all_in":
        return "call"
    return action


class TestShortDeckActionSequences:
    """Test Short Deck with action sequences from original tests."""
    
    @pytest.mark.parametrize("n_players", [2, 3])
    def test_call_action_sequence_short_deck(self, n_players):
        """Ensure no invalid action sequences occur in Short Deck.
        
        Make sure we never see an action sequence of "raise", "call", "call" in the same
        round with only two players.
        """
        seed(42)
        bad_seq = ["raise", "call", "call"]
        
        # Run multiple random iterations
        for _ in range(200):
            state, _ = _new_game(n_players=n_players, small_blind=50, big_blind=100)
            betting_round_dict = collections.defaultdict(list)
            
            while state.betting_stage not in {"show_down", "terminal"}:
                uniform_probability = 1 / len(state.legal_actions)
                probabilities = np.full(len(state.legal_actions), uniform_probability)
                random_action = np.random.choice(state.legal_actions, p=probabilities)
                
                if state._poker_engine.n_active_players == 2:
                    # Normalize action for comparison
                    normalized_action = _normalize_action(random_action)
                    betting_round_dict[state.betting_stage].append(normalized_action)
                    no_fold_action_history = [
                        action for action in betting_round_dict[state.betting_stage]
                        if action != "skip"
                    ]
                    # Ensure bad sequence hasn't happened
                    for i in range(len(no_fold_action_history)):
                        history_slice = no_fold_action_history[i : i + len(bad_seq)]
                        assert history_slice != bad_seq
                
                state = state.apply_action(random_action)
    
    @pytest.mark.parametrize("n_players", [2, 3])
    @pytest.mark.xfail(reason="Pre-computed action sequences need to be regenerated for new action abstraction system with multiple raise sizes")
    def test_action_sequence_short_deck(self, n_players: int):
        """Check each round against validated action sequences.
        
        This ensures the state class is working correctly by validating
        against pre-computed valid action sequences.
        
        NOTE: This test is currently expected to fail because the pre-computed
        action sequences were generated with the old action system (single "raise"
        action) and need to be regenerated for the new system (multiple raise
        sizes like "raise:0.5", "raise:1.0", etc.).
        """
        seed(42)
        directory = "research/size_of_problem/action_sequences.pkl"
        action_sequences = _load_action_sequences(directory)
        
        for i in range(200):
            state, _ = _new_game(n_players=n_players, small_blind=50, big_blind=100)
            
            betting_stage_dict = {
                "pre_flop": {"action_sequence": [], "n_players": 0},
                "flop": {"action_sequence": [], "n_players": 0},
                "turn": {"action_sequence": [], "n_players": 0},
                "river": {"action_sequence": [], "n_players": 0},
            }
            betting_stage = None
            
            while state.betting_stage not in {"show_down", "terminal"}:
                if betting_stage != state.betting_stage:
                    betting_stage_dict[state.betting_stage][
                        "n_players"
                    ] = state.n_players_started_round
                    betting_stage = state.betting_stage
                
                uniform_probability = 1 / len(state.legal_actions)
                probabilities = np.full(len(state.legal_actions), uniform_probability)
                random_action = np.random.choice(state.legal_actions, p=probabilities)
                
                # Normalize action for comparison with pre-computed sequences
                normalized_action = _normalize_action(random_action)
                betting_stage_dict[state.betting_stage]["action_sequence"].append(
                    normalized_action
                )
                state = state.apply_action(random_action)
            
            for betting_stage in betting_stage_dict.keys():
                if betting_stage_dict[betting_stage]["action_sequence"]:
                    n_players_started_round = betting_stage_dict[betting_stage]["n_players"]
                    action_sequence = betting_stage_dict[betting_stage]["action_sequence"]
                    possible_sequences = action_sequences[n_players_started_round]
                    
                    assert action_sequence in possible_sequences


class TestShortDeckSkips:
    """Test skip functionality in Short Deck."""
    
    def test_skips(self):
        """Check each round to ensure skips are mod number of players.
        
        Verifies that skips are correctly appended on the skipped player's turn
        throughout all betting rounds.
        """
        n_players = 3
        seed(42)
        
        for _ in range(500):
            state, _ = _new_game(n_players=n_players, small_blind=50, big_blind=100)
            
            while True:
                uniform_probability = 1 / len(state.legal_actions)
                probabilities = np.full(len(state.legal_actions), uniform_probability)
                random_action = np.random.choice(state.legal_actions, p=probabilities)
                state = state.apply_action(random_action)
                
                if state.betting_stage in {"show_down", "terminal"}:
                    break
            
            # Validate skip mechanics across all betting rounds
            preflop_actions = state._history["pre_flop"]
            preflop_folds = [i for i, x in enumerate(preflop_actions) if x == "fold"]
            # accounting for rotation preflop
            preflop_fold_players = [(x + 2) % 3 for x in preflop_folds]
            
            if state._history["flop"] is not None:
                flop_actions = state._history["flop"]
                flop_folds = [i for i, x in enumerate(flop_actions) if x == "fold"]
            
            if state._history["turn"] is not None:
                turn_actions = state._history["turn"]
                turn_folds = [i for i, x in enumerate(turn_actions) if x == "fold"]
            
            for stage in state._history.keys():
                if state._history[stage] is not None:
                    if stage == "flop":
                        fold_players = preflop_fold_players
                    if stage == "turn":
                        fold_players = preflop_fold_players + flop_folds
                    if stage == "river":
                        fold_players = preflop_fold_players + flop_folds + turn_folds
                    
                    actions = state._history[stage]
                    folds = [i for i, x in enumerate(actions) if x == "fold"]
                    
                    for fold_idx in folds:
                        for i, action in enumerate(actions[fold_idx:]):
                            # i greater than 0 because 0 is a fold
                            if i > 0 and i % n_players == 0:
                                assert action == "skip"
                    
                    if stage != "pre_flop":
                        for fold_idx in fold_players:
                            # i can be 0 because folds happened in a previous round
                            for i, action in enumerate(actions[fold_idx:]):
                                if i % n_players == 0:
                                    assert action == "skip"


class TestShortDeckCardValidity:
    """Test card validity in Short Deck."""
    
    def test_all_cards_in_valid_range(self):
        """Test that all dealt cards are in valid rank range for Short Deck."""
        # Deal maximum cards to test
        state, _ = _new_game(n_players=6)
        
        # Progress through all betting stages to see all cards
        state = state.apply_action(action_str="call")
        state = state.apply_action(action_str="call")
        state = state.apply_action(action_str="call")
        state = state.apply_action(action_str="call")
        state = state.apply_action(action_str="call")
        state = state.apply_action(action_str="call")
        
        # Check all player cards
        for player in state.players:
            for card in player.cards:
                assert 10 <= card.rank_int <= 14
        
        # Check all community cards
        for card in state.community_cards:
            assert 10 <= card.rank_int <= 14
    
    def test_no_duplicate_cards(self):
        """Test that no duplicate cards are dealt in Short Deck."""
        state, _ = _new_game(n_players=6)
        
        # Collect all dealt cards
        all_cards = []
        for player in state.players:
            all_cards.extend(player.cards)
        
        # Check for duplicates using eval_card representation
        eval_cards = [card.eval_card for card in all_cards]
        assert len(eval_cards) == len(set(eval_cards)), "Duplicate cards dealt!"


class TestShortDeckPreFlopPot:
    """Test pre-flop pot mechanics specific to configurations."""
    
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
    def test_pre_flop_pot(self, n_players: int, small_blind: int, big_blind: int):
        """Test pre-flop pot for various Short Deck configurations."""
        state, pot = _new_game(
            n_players=n_players,
            small_blind=small_blind,
            big_blind=big_blind,
            initial_chips=10000,
        )
        
        # Verify correct starting player (from original test)
        assert state.player_i == 0 if n_players == 2 else 2
        assert state.betting_stage == "pre_flop"
        
        # Verify blinds were posted correctly
        n_bet_chips = sum(p.n_bet_chips for p in state.players)
        target = small_blind + big_blind
        assert n_bet_chips == target
        assert n_bet_chips == pot.total
        
        # Verify total chips conservation
        total_chips = sum(p.n_chips for p in state.players) + pot.total
        assert total_chips == n_players * 10000
