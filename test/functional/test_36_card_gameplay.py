"""Integration test for 36-card Short Deck gameplay."""

import pytest
from poker_ai.games.short_deck.state import new_game
from poker_ai.poker.card import Card


def test_36_card_deck_gameplay():
    """Test that 36-card deck works in actual gameplay."""
    # Create a 36-card deck game
    state = new_game(n_players=3, deck_size=36, load_card_lut=False)
    
    # Verify initial setup
    assert len(state._table.dealer.deck) == 36
    assert state._get_deck_ranks() == [6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    # Verify players have cards
    for player in state.players:
        assert len(player.cards) == 2
    
    # Verify we can play through a hand
    # Players should be able to act
    assert state.current_player is not None
    print(f"✓ 36-card deck initialized: {len(state._table.dealer.deck)} cards")
    print(f"  Players: {len(state.players)}")
    print(f"  Cards dealt: {sum(len(p.cards) for p in state.players)}")


def test_36_card_flush_possibility():
    """Test that with 36 cards, regular flushes (non-straight) are possible."""
    # With 9 ranks per suit, you can have a flush without a straight
    # For example: 6♠ 7♠ 9♠ J♠ K♠ is a flush but not a straight
    
    # Create cards that form a non-straight flush
    from poker_ai.poker.evaluation.short_deck_evaluator import ShortDeckEvaluator
    
    evaluator = ShortDeckEvaluator()
    
    # Non-straight flush: 6, 7, 9, J, K of spades (gaps prevent straight)
    # Split into hole cards and board
    hole_cards_flush = [
        Card(6, 'spades'),
        Card(7, 'spades'),
    ]
    board_flush = [
        Card(9, 'spades'),
        Card(11, 'spades'),  # Jack
        Card(13, 'spades'),  # King
    ]
    
    # Straight: 6-7-8-9-10 (mixed suits)
    hole_cards_straight = [
        Card(6, 'hearts'),
        Card(7, 'diamonds'),
    ]
    board_straight = [
        Card(8, 'clubs'),
        Card(9, 'spades'),
        Card(10, 'hearts'),
    ]
    
    flush_rank = evaluator.evaluate(hole_cards_flush, board_flush)
    straight_rank = evaluator.evaluate(hole_cards_straight, board_straight)
    
    # In Short Deck, flush should beat straight (lower rank = better)
    assert flush_rank < straight_rank, f"Flush ({flush_rank}) should beat straight ({straight_rank})"
    print(f"✓ 36-card: Non-straight flush correctly beats straight")
    print(f"  Flush rank: {flush_rank}, Straight rank: {straight_rank}")


def test_both_deck_sizes_supported():
    """Test that both 20 and 36 card configurations work side by side."""
    # Create both deck types
    state_20 = new_game(n_players=2, deck_size=20, load_card_lut=False)
    state_36 = new_game(n_players=2, deck_size=36, load_card_lut=False)
    
    # Verify they're independent
    assert len(state_20._table.dealer.deck) == 20
    assert len(state_36._table.dealer.deck) == 36
    
    assert state_20._get_deck_ranks() == [10, 11, 12, 13, 14]
    assert state_36._get_deck_ranks() == [6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    print("✓ Both 20-card and 36-card decks work independently")


def test_36_card_three_of_kind_beats_straight():
    """Test that Three of a Kind beats Straight in 36-card deck."""
    from poker_ai.poker.evaluation.short_deck_evaluator import ShortDeckEvaluator
    
    evaluator = ShortDeckEvaluator()
    
    # Three of a Kind: 8-8 hole cards, 8-6-7 on board
    hole_three = [
        Card(8, 'hearts'),
        Card(8, 'diamonds'),
    ]
    board_three = [
        Card(8, 'clubs'),
        Card(6, 'spades'),
        Card(7, 'hearts'),
    ]
    
    # Straight: 6-7 hole cards, 8-9-10 on board makes 6-7-8-9-10
    hole_straight = [
        Card(6, 'hearts'),
        Card(7, 'diamonds'),
    ]
    board_straight = [
        Card(8, 'clubs'),
        Card(9, 'spades'),
        Card(10, 'hearts'),
    ]
    
    three_rank = evaluator.evaluate(hole_three, board_three)
    straight_rank = evaluator.evaluate(hole_straight, board_straight)
    
    # Lower rank = better hand
    assert three_rank < straight_rank, f"Three of a Kind ({three_rank}) should beat Straight ({straight_rank})"
    print(f"✓ 36-card: Three of a Kind correctly beats Straight")
    print(f"  3oK rank: {three_rank}, Straight rank: {straight_rank}")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing 36-Card Short Deck Gameplay")
    print("=" * 70)
    
    test_36_card_deck_gameplay()
    print()
    test_36_card_flush_possibility()
    print()
    test_both_deck_sizes_supported()
    print()
    test_36_card_three_of_kind_beats_straight()
    
    print("\n" + "=" * 70)
    print("All 36-card deck tests passed! ✓")
    print("=" * 70)
