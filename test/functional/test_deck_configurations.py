"""Test script to verify both 20-card and 36-card Short Deck configurations."""

from poker_ai.games.short_deck.state import new_game


def test_20_card_deck():
    """Test 20-card deck configuration (ranks 10-A)."""
    print("Testing 20-card deck configuration...")
    state = new_game(n_players=3, deck_size=20, load_card_lut=False)
    
    # Check deck has correct ranks
    deck_ranks = state._get_deck_ranks()
    assert deck_ranks == [10, 11, 12, 13, 14], f"Expected [10, 11, 12, 13, 14], got {deck_ranks}"
    
    # Check deck has 20 cards total (including dealt cards)
    total_cards = len(state._table.dealer.deck)
    assert total_cards == 20, f"Expected 20 cards, got {total_cards}"
    
    # Verify each rank appears 4 times (one per suit) in all cards
    from collections import Counter
    all_cards = state._table.dealer.deck._cards_in_deck + state._table.dealer.deck._dealt_cards
    rank_counts = Counter([card.rank_int for card in all_cards])
    for rank in [10, 11, 12, 13, 14]:
        assert rank_counts[rank] == 4, f"Rank {rank} appears {rank_counts[rank]} times, expected 4"
    
    print("✓ 20-card deck: OK (5 ranks × 4 suits = 20 cards)")


def test_36_card_deck():
    """Test 36-card deck configuration (ranks 6-A)."""
    print("\nTesting 36-card deck configuration...")
    state = new_game(n_players=3, deck_size=36, load_card_lut=False)
    
    # Check deck has correct ranks
    deck_ranks = state._get_deck_ranks()
    assert deck_ranks == [6, 7, 8, 9, 10, 11, 12, 13, 14], f"Expected [6, 7, 8, 9, 10, 11, 12, 13, 14], got {deck_ranks}"
    
    # Check deck has 36 cards total
    total_cards = len(state._table.dealer.deck)
    assert total_cards == 36, f"Expected 36 cards, got {total_cards}"
    
    # Verify each rank appears 4 times (one per suit)
    from collections import Counter
    all_cards = state._table.dealer.deck._cards_in_deck + state._table.dealer.deck._dealt_cards
    rank_counts = Counter([card.rank_int for card in all_cards])
    for rank in [6, 7, 8, 9, 10, 11, 12, 13, 14]:
        assert rank_counts[rank] == 4, f"Rank {rank} appears {rank_counts[rank]} times, expected 4"
    
    print("✓ 36-card deck: OK (9 ranks × 4 suits = 36 cards)")


def test_invalid_deck_size():
    """Test that invalid deck sizes raise an error."""
    print("\nTesting invalid deck size...")
    try:
        state = new_game(n_players=3, deck_size=52, load_card_lut=False)
        assert False, "Should have raised ValueError for deck_size=52"
    except ValueError as e:
        assert "deck_size must be 20 or 36" in str(e)
        print(f"✓ Invalid deck size correctly rejected: {e}")


def test_default_deck_size():
    """Test that default deck size is 20."""
    print("\nTesting default deck size...")
    state = new_game(n_players=3, load_card_lut=False)
    
    deck_ranks = state._get_deck_ranks()
    total_cards = len(state._table.dealer.deck)
    
    assert deck_ranks == [10, 11, 12, 13, 14], "Default should be 20-card deck"
    assert total_cards == 20, "Default should have 20 cards"
    
    print("✓ Default deck size: OK (defaults to 20 cards)")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Short Deck Dual Configuration Support")
    print("=" * 60)
    
    test_20_card_deck()
    test_36_card_deck()
    test_invalid_deck_size()
    test_default_deck_size()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nSummary:")
    print("  • 20-card deck: Ranks 10-A (5 ranks × 4 suits)")
    print("  • 36-card deck: Ranks 6-A (9 ranks × 4 suits)")
    print("  • Invalid sizes are properly rejected")
    print("  • Default is 20-card deck")
