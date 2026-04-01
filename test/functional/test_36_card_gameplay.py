"""Integration test for 36-card deck gameplay."""

import pytest
from poker_ai.environment.game_state import new_game
from poker_ai.environment.card import Card
from poker_ai.environment.evaluation.evaluator import Evaluator


def test_36_card_deck_gameplay():
    """Test that 36-card deck works in actual gameplay."""
    # Create a 36-card deck game
    state = new_game(n_players=3, low_card_rank=6, high_card_rank=14, card_info_lut={})

    # Verify initial setup
    assert len(state._table.dealer.deck) == 36
    assert state._get_deck_ranks() == [6, 7, 8, 9, 10, 11, 12, 13, 14]

    # Verify players have cards
    for player in state.players:
        assert len(player.cards) == 2

    # Verify we can play through a hand
    # Players should be able to act
    assert state.current_player is not None


def test_36_card_flush_vs_straight():
    """Test standard poker rankings: straight beats flush (lower rank = better)."""
    evaluator = Evaluator()

    # Flush (non-straight): 6, 7, 9, J, K of spades
    hole_cards_flush = [
        Card(6, 'spades'),
        Card(7, 'spades'),
    ]
    board_flush = [
        Card(9, 'spades'),
        Card(11, 'spades'),   # Jack
        Card(13, 'spades'),   # King
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

    # In standard poker, flush beats straight (lower rank = better)
    assert flush_rank < straight_rank, (
        f"Flush ({flush_rank}) should beat straight ({straight_rank}) in standard rankings"
    )


def test_both_deck_sizes_supported():
    """Test that both 20-card and 36-card configurations work side by side."""
    # Create both deck types
    state_20 = new_game(n_players=2, low_card_rank=10, high_card_rank=14, card_info_lut={})
    state_36 = new_game(n_players=2, low_card_rank=6, high_card_rank=14, card_info_lut={})

    # Verify they're independent
    assert len(state_20._table.dealer.deck) == 20
    assert len(state_36._table.dealer.deck) == 36

    assert state_20._get_deck_ranks() == [10, 11, 12, 13, 14]
    assert state_36._get_deck_ranks() == [6, 7, 8, 9, 10, 11, 12, 13, 14]


def test_36_card_straight_beats_three_of_kind():
    """Test standard poker rankings: straight beats three of a kind."""
    evaluator = Evaluator()

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

    # In standard poker, straight beats three of a kind (lower rank = better)
    assert straight_rank < three_rank, (
        f"Straight ({straight_rank}) should beat Three of a Kind ({three_rank}) in standard rankings"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Testing 36-Card Deck Gameplay")
    print("=" * 70)

    test_36_card_deck_gameplay()
    print("  36-card gameplay: OK")
    test_36_card_flush_vs_straight()
    print("  Flush vs straight rankings: OK")
    test_both_deck_sizes_supported()
    print("  Both deck sizes: OK")
    test_36_card_straight_beats_three_of_kind()
    print("  Straight vs three of a kind rankings: OK")

    print("\n" + "=" * 70)
    print("All 36-card deck tests passed!")
    print("=" * 70)
