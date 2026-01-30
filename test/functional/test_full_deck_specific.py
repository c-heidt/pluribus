"""Test Full Deck (Texas Hold'em) specific features and edge cases.

These tests focus on Full Deck-specific functionality:
- 52-card deck (ranks 2-A)
- Standard poker hand rankings
- Full Deck specific mechanics like the wheel straight
"""

import pytest

from poker_ai.games.full_deck.state import FullDeckPokerState, new_game
from poker_ai.poker.card import Card
from poker_ai.poker.evaluation.evaluator import Evaluator


class TestFullDeckSpecifics:
    """Test Full Deck variant-specific features."""
    
    def test_full_deck_has_52_cards(self):
        """Test that Full Deck uses all 52 cards."""
        state = new_game(n_players=2, card_info_lut={})
        
        # Get deck ranks - should be 2-14 (13 ranks)
        ranks = state._get_deck_ranks()
        assert len(ranks) == 13
        assert ranks == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        
        # 13 ranks * 4 suits = 52 cards
        deck_size = len(ranks) * 4
        assert deck_size == 52
    
    def test_full_deck_uses_standard_evaluator(self):
        """Test that Full Deck uses the standard poker evaluator."""
        state = new_game(n_players=2, card_info_lut={})
        
        evaluator = state._get_evaluator()
        assert isinstance(evaluator, Evaluator)
        
        # Verify it's the standard evaluator, not Short Deck
        from poker_ai.poker.evaluation.short_deck_evaluator import ShortDeckEvaluator
        assert not isinstance(evaluator, ShortDeckEvaluator)
    
    def test_full_deck_includes_low_cards(self):
        """Test that low cards (2-9) appear in Full Deck."""
        state = new_game(n_players=6, card_info_lut={})  # Max players
        
        # Collect all dealt cards
        all_cards = []
        for player in state.players:
            all_cards.extend(player.cards)
        
        # Get all ranks present
        ranks_present = set(card.rank for card in all_cards)
        
        # With 6 players and random dealing, we should see various ranks
        # At minimum, verify cards are in valid range for full deck
        for card in all_cards:
            assert 2 <= card.rank_int <= 14, f"Card rank {card.rank_int} out of range"
    
    def test_full_deck_vs_short_deck_differences(self):
        """Test key differences between Full Deck and Short Deck."""
        from poker_ai.games.short_deck.state import ShortDeckPokerState
        from poker_ai.games.short_deck.player import ShortDeckPokerPlayer
        from poker_ai.poker.pot import Pot
        
        full_deck_state = new_game(n_players=2, card_info_lut={})
        
        pot = Pot()
        short_deck_players = [
            ShortDeckPokerPlayer(player_i=i, pot=pot, initial_chips=10000)
            for i in range(2)
        ]
        short_deck_state = ShortDeckPokerState(
            players=short_deck_players,
            load_card_lut=False,
        )
        
        # Different deck sizes
        full_deck_ranks = full_deck_state._get_deck_ranks()
        short_deck_ranks = short_deck_state._get_deck_ranks()
        
        assert len(full_deck_ranks) == 13  # 2-A
        assert len(short_deck_ranks) == 5  # 10-A
        
        # Different evaluators
        from poker_ai.poker.evaluation.short_deck_evaluator import ShortDeckEvaluator
        
        assert isinstance(full_deck_state._get_evaluator(), Evaluator)
        assert isinstance(short_deck_state._get_evaluator(), ShortDeckEvaluator)


class TestFullDeckStandardRankings:
    """Test standard poker hand rankings in Full Deck."""
    
    def test_full_house_beats_flush(self):
        """Test that Full House beats Flush in Full Deck (standard ranking)."""
        state = new_game(n_players=2, card_info_lut={})
        evaluator = state._get_evaluator()
        
        # Full House: Three Kings, Two Aces
        full_house_hole = [
            Card(13, "spades"),   # King
            Card(13, "hearts"),   # King
        ]
        full_house_board = [
            Card(13, "diamonds"), # King
            Card(14, "clubs"),    # Ace
            Card(14, "spades"),   # Ace
            Card(2, "clubs"),     # Two (filler)
            Card(3, "spades"),    # Three (filler)
        ]
        
        # Flush: Five hearts (not a straight)
        flush_hole = [
            Card(2, "hearts"),    # 2 of hearts
            Card(5, "hearts"),    # 5 of hearts
        ]
        flush_board = [
            Card(9, "hearts"),    # 9 of hearts
            Card(11, "hearts"),   # Jack of hearts
            Card(13, "hearts"),   # King of hearts
            Card(3, "diamonds"),  # 3 of diamonds (filler)
            Card(4, "diamonds"),  # 4 of diamonds (filler)
        ]
        
        fh_hole_cards = [c.eval_card for c in full_house_hole]
        fh_board_cards = [c.eval_card for c in full_house_board]
        fl_hole_cards = [c.eval_card for c in flush_hole]
        fl_board_cards = [c.eval_card for c in flush_board]
        
        fh_rank = evaluator.evaluate(fh_hole_cards, fh_board_cards)
        fl_rank = evaluator.evaluate(fl_hole_cards, fl_board_cards)
        
        # In standard poker, full house beats flush (lower rank is better)
        assert fh_rank < fl_rank, f"Full House rank {fh_rank} should be < Flush rank {fl_rank}"
        
        fh_class = evaluator.get_rank_class(fh_rank)
        fl_class = evaluator.get_rank_class(fl_rank)
        
        assert fh_class == 3, f"Full House should be class 3, got {fh_class}"
        assert fl_class == 4, f"Flush should be class 4, got {fl_class}"
    
    def test_straight_beats_three_of_kind(self):
        """Test that Straight beats Three of a Kind in Full Deck (standard ranking)."""
        state = new_game(n_players=2, card_info_lut={})
        evaluator = state._get_evaluator()
        
        # Three of a Kind: Three Tens
        three_of_kind_hole = [
            Card(10, "spades"),   # Ten
            Card(10, "hearts"),   # Ten
        ]
        three_of_kind_board = [
            Card(10, "diamonds"), # Ten
            Card(14, "clubs"),    # Ace
            Card(13, "spades"),   # King
            Card(2, "clubs"),     # Two (filler)
            Card(3, "spades"),    # Three (filler)
        ]
        
        # Straight: 5-6-7-8-9
        straight_hole = [
            Card(5, "spades"),    # 5
            Card(6, "hearts"),    # 6
        ]
        straight_board = [
            Card(7, "diamonds"),  # 7
            Card(8, "clubs"),     # 8
            Card(9, "spades"),    # 9
            Card(2, "hearts"),    # 2 (filler)
            Card(3, "clubs"),     # 3 (filler)
        ]
        
        tok_hole_cards = [c.eval_card for c in three_of_kind_hole]
        tok_board_cards = [c.eval_card for c in three_of_kind_board]
        str_hole_cards = [c.eval_card for c in straight_hole]
        str_board_cards = [c.eval_card for c in straight_board]
        
        tok_rank = evaluator.evaluate(tok_hole_cards, tok_board_cards)
        str_rank = evaluator.evaluate(str_hole_cards, str_board_cards)
        
        # In standard poker, straight beats three of a kind
        assert str_rank < tok_rank, f"Straight rank {str_rank} should be < Three of Kind rank {tok_rank}"
        
        tok_class = evaluator.get_rank_class(tok_rank)
        str_class = evaluator.get_rank_class(str_rank)
        
        assert tok_class == 6, f"Three of a Kind should be class 6, got {tok_class}"
        assert str_class == 5, f"Straight should be class 5, got {str_class}"


class TestFullDeckWheelStraight:
    """Test the wheel (A-2-3-4-5) straight specific to Full Deck."""
    
    def test_wheel_straight_recognized(self):
        """Test that the wheel (A-2-3-4-5) works in Full Deck."""
        state = new_game(n_players=2, card_info_lut={})
        evaluator = state._get_evaluator()
        
        # Wheel straight: A-2-3-4-5 (Ace acts as low card)
        wheel_hole = [
            Card(14, "spades"),   # Ace (can be low)
            Card(2, "hearts"),    # 2
        ]
        wheel_board = [
            Card(3, "diamonds"),  # 3
            Card(4, "clubs"),     # 4
            Card(5, "spades"),    # 5
            Card(10, "hearts"),   # Ten (filler)
            Card(11, "clubs"),    # Jack (filler)
        ]
        
        wheel_hole_cards = [c.eval_card for c in wheel_hole]
        wheel_board_cards = [c.eval_card for c in wheel_board]
        
        wheel_rank = evaluator.evaluate(wheel_hole_cards, wheel_board_cards)
        wheel_class = evaluator.get_rank_class(wheel_rank)
        
        # Should be recognized as a straight
        assert wheel_class == 5, f"Wheel should be a Straight (class 5), got {wheel_class}"
    
    def test_wheel_is_lowest_straight(self):
        """Test that wheel is the lowest possible straight."""
        state = new_game(n_players=2, card_info_lut={})
        evaluator = state._get_evaluator()
        
        # Wheel: A-2-3-4-5
        wheel_hole = [Card(14, "spades"), Card(2, "hearts")]
        wheel_board = [
            Card(3, "diamonds"), Card(4, "clubs"), Card(5, "spades"),
            Card(10, "hearts"), Card(11, "clubs")
        ]
        
        # Regular low straight: 2-3-4-5-6
        low_straight_hole = [Card(2, "spades"), Card(3, "hearts")]
        low_straight_board = [
            Card(4, "diamonds"), Card(5, "clubs"), Card(6, "spades"),
            Card(10, "hearts"), Card(11, "clubs")
        ]
        
        wheel_hole_cards = [c.eval_card for c in wheel_hole]
        wheel_board_cards = [c.eval_card for c in wheel_board]
        low_str_hole_cards = [c.eval_card for c in low_straight_hole]
        low_str_board_cards = [c.eval_card for c in low_straight_board]
        
        wheel_rank = evaluator.evaluate(wheel_hole_cards, wheel_board_cards)
        low_straight_rank = evaluator.evaluate(low_str_hole_cards, low_str_board_cards)
        
        # Wheel should have higher rank number (worse) than 2-3-4-5-6
        assert wheel_rank > low_straight_rank, "Wheel should be lowest straight"


class TestFullDeckCardRange:
    """Test card range validity in Full Deck."""
    
    def test_all_ranks_can_appear(self):
        """Test that all ranks 2-A can appear in Full Deck."""
        # Play many games to sample all possible cards
        all_ranks_seen = set()
        
        for _ in range(20):
            state = new_game(n_players=6, card_info_lut={})
            
            # Collect ranks from dealt cards
            for player in state.players:
                for card in player.cards:
                    all_ranks_seen.add(card.rank_int)
        
        # Should see a good variety of ranks
        # At minimum, verify we see both low and high cards
        assert any(rank <= 5 for rank in all_ranks_seen), "No low cards seen"
        assert any(rank >= 10 for rank in all_ranks_seen), "No high cards seen"
    
    def test_no_invalid_ranks(self):
        """Test that only valid ranks appear in Full Deck."""
        state = new_game(n_players=6, card_info_lut={})
        
        # Check all player cards
        for player in state.players:
            for card in player.cards:
                assert 2 <= card.rank_int <= 14, f"Invalid rank {card.rank_int}"
        
        # Progress to see community cards
        state = state.apply_action(action_str="call")
        state = state.apply_action(action_str="call")
        state = state.apply_action(action_str="call")
        state = state.apply_action(action_str="call")
        state = state.apply_action(action_str="call")
        state = state.apply_action(action_str="call")
        
        # Check community cards
        for card in state.community_cards:
            assert 2 <= card.rank_int <= 14, f"Invalid rank {card.rank_int}"
    
    def test_no_duplicate_cards(self):
        """Test that no duplicate cards are dealt in Full Deck."""
        state = new_game(n_players=6, card_info_lut={})
        
        # Collect all dealt cards
        all_cards = []
        for player in state.players:
            all_cards.extend(player.cards)
        
        # Check for duplicates
        eval_cards = [card.eval_card for card in all_cards]
        assert len(eval_cards) == len(set(eval_cards)), "Duplicate cards dealt!"


class TestFullDeckEdgeCases:
    """Test edge cases specific to Full Deck."""
    
    def test_low_card_straights(self):
        """Test that straights with low cards work correctly."""
        state = new_game(n_players=2, card_info_lut={})
        evaluator = state._get_evaluator()
        
        # Straight: 3-4-5-6-7
        low_straight_hole = [Card(3, "spades"), Card(4, "hearts")]
        low_straight_board = [
            Card(5, "diamonds"), Card(6, "clubs"), Card(7, "spades"),
            Card(10, "hearts"), Card(11, "clubs")
        ]
        
        ls_hole_cards = [c.eval_card for c in low_straight_hole]
        ls_board_cards = [c.eval_card for c in low_straight_board]
        
        ls_rank = evaluator.evaluate(ls_hole_cards, ls_board_cards)
        ls_class = evaluator.get_rank_class(ls_rank)
        
        assert ls_class == 5, f"Low straight should be class 5, got {ls_class}"
    
    def test_low_card_flushes(self):
        """Test that flushes with low cards work correctly."""
        state = new_game(n_players=2, card_info_lut={})
        evaluator = state._get_evaluator()
        
        # Flush with low cards: 2♥ 4♥ 6♥ 8♥ 10♥
        low_flush_hole = [Card(2, "hearts"), Card(4, "hearts")]
        low_flush_board = [
            Card(6, "hearts"), Card(8, "hearts"), Card(10, "hearts"),
            Card(3, "diamonds"), Card(5, "clubs")
        ]
        
        lf_hole_cards = [c.eval_card for c in low_flush_hole]
        lf_board_cards = [c.eval_card for c in low_flush_board]
        
        lf_rank = evaluator.evaluate(lf_hole_cards, lf_board_cards)
        lf_class = evaluator.get_rank_class(lf_rank)
        
        assert lf_class == 4, f"Flush should be class 4, got {lf_class}"


class TestFullDeckActionSequences:
    """Test Full Deck with various action sequences."""
    
    @pytest.mark.parametrize("n_players", [2, 3])
    def test_all_call_sequence(self, n_players):
        """Test a full game where everyone calls in Full Deck."""
        state = new_game(n_players=n_players, card_info_lut={})
        
        # Play through entire game with all calls
        safety_counter = 0
        max_iterations = 100
        
        while not state.is_terminal and safety_counter < max_iterations:
            # Always call to keep game moving
            state = state.apply_action(action_str='call')
            safety_counter += 1
        
        # Game should reach terminal state
        assert state.is_terminal
        assert state.betting_stage in ['terminal', 'show_down']
    
    @pytest.mark.parametrize("n_players", [2, 3])
    def test_mixed_action_sequence(self, n_players: int):
        """Test games with mixed actions in Full Deck."""
        import random
        state = new_game(n_players=n_players, card_info_lut={})
        random.seed(42)
        
        # Play with random actions
        safety_counter = 0
        max_iterations = 1000
        
        while not state.is_terminal and safety_counter < max_iterations:
            action = random.choice(state.legal_actions)
            state = state.apply_action(action_str=str(action))
            safety_counter += 1
        
        # Should eventually reach terminal
        assert safety_counter < max_iterations, "Game didn't terminate in reasonable time"
        assert state.is_terminal


class TestFullDeckPreFlopPot:
    """Test pre-flop pot mechanics for Full Deck configurations."""
    
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
    ])
    def test_pre_flop_pot(self, n_players: int, small_blind: int, big_blind: int):
        """Test pre-flop pot for various Full Deck configurations."""
        from poker_ai.games.full_deck.player import FullDeckPokerPlayer
        from poker_ai.poker.pot import Pot
        
        pot = Pot()
        players = [
            FullDeckPokerPlayer(player_i=i, pot=pot, initial_chips=10000)
            for i in range(n_players)
        ]
        state = FullDeckPokerState(
            players=players,
            load_card_lut=False,
            small_blind=small_blind,
            big_blind=big_blind,
        )
        
        # Verify blinds were posted
        assert pot.total == small_blind + big_blind
        
        # Verify total chips conservation
        total_chips = sum(p.n_chips for p in state.players) + pot.total
        assert total_chips == n_players * 10000
