"""Test script to verify various deck configurations."""

from poker_ai.environment.game_state import new_game


def test_20_card_deck():
    """Test 20-card deck configuration (ranks 10-A)."""
    state = new_game(n_players=3, low_card_rank=10, high_card_rank=14, card_info_lut={})

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


def test_36_card_deck():
    """Test 36-card deck configuration (ranks 6-A)."""
    state = new_game(n_players=3, low_card_rank=6, high_card_rank=14, card_info_lut={})

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


def test_invalid_deck_size():
    """Test that invalid rank configurations raise an error."""
    import pytest

    # low_card_rank > high_card_rank should raise ValueError
    with pytest.raises(ValueError):
        new_game(n_players=3, low_card_rank=14, high_card_rank=10, card_info_lut={})

    # Deck too small for the number of players should raise ValueError
    # 2 ranks x 4 suits = 8 cards, but 6 players need 12 hole cards + 5 community = 17
    with pytest.raises(ValueError):
        new_game(n_players=6, low_card_rank=13, high_card_rank=14, card_info_lut={})


def test_default_deck_size():
    """Test that defaults (no low/high params) give a 52-card deck (ranks 2-14)."""
    state = new_game(n_players=3, card_info_lut={})

    deck_ranks = state._get_deck_ranks()
    total_cards = len(state._table.dealer.deck)

    assert deck_ranks == list(range(2, 15)), f"Default should be full deck ranks 2-14, got {deck_ranks}"
    assert total_cards == 52, f"Default should have 52 cards, got {total_cards}"


def test_arbitrary_deck_size():
    """Test an arbitrary deck configuration (ranks 8-A = 28 cards)."""
    state = new_game(n_players=3, low_card_rank=8, high_card_rank=14, card_info_lut={})

    deck_ranks = state._get_deck_ranks()
    assert deck_ranks == [8, 9, 10, 11, 12, 13, 14], f"Expected [8, 9, 10, 11, 12, 13, 14], got {deck_ranks}"

    total_cards = len(state._table.dealer.deck)
    assert total_cards == 28, f"Expected 28 cards (7 ranks x 4 suits), got {total_cards}"

    # Verify each rank appears 4 times
    from collections import Counter
    all_cards = state._table.dealer.deck._cards_in_deck + state._table.dealer.deck._dealt_cards
    rank_counts = Counter([card.rank_int for card in all_cards])
    for rank in [8, 9, 10, 11, 12, 13, 14]:
        assert rank_counts[rank] == 4, f"Rank {rank} appears {rank_counts[rank]} times, expected 4"


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Deck Configuration Support")
    print("=" * 60)

    test_20_card_deck()
    print("  20-card deck: OK")
    test_36_card_deck()
    print("  36-card deck: OK")
    test_invalid_deck_size()
    print("  Invalid deck size: OK")
    test_default_deck_size()
    print("  Default deck size: OK")
    test_arbitrary_deck_size()
    print("  Arbitrary deck size: OK")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
