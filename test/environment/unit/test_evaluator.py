"""Unit tests for poker_ai/environment/lookup.py and evaluator.py.

Covers LookupTable construction, hand-rank constants, Gosper's hack generator,
and Evaluator hand ranking for known hand types.
"""

import pytest

from poker_ai.environment.lookup import LookupTable
from poker_ai.environment.evaluator import Evaluator
from poker_ai.environment.utils import new_card, make_card


# ---------------------------------------------------------------------------
# LookupTable
# ---------------------------------------------------------------------------

class TestLookupTableConstants:
    def test_max_straight_flush(self):
        assert LookupTable.MAX_STRAIGHT_FLUSH == 10

    def test_max_four_of_a_kind(self):
        assert LookupTable.MAX_FOUR_OF_A_KIND == 166

    def test_max_full_house(self):
        assert LookupTable.MAX_FULL_HOUSE == 322

    def test_max_flush(self):
        assert LookupTable.MAX_FLUSH == 1599

    def test_max_straight(self):
        assert LookupTable.MAX_STRAIGHT == 1609

    def test_max_three_of_a_kind(self):
        assert LookupTable.MAX_THREE_OF_A_KIND == 2467

    def test_max_two_pair(self):
        assert LookupTable.MAX_TWO_PAIR == 3325

    def test_max_pair(self):
        assert LookupTable.MAX_PAIR == 6185

    def test_max_high_card(self):
        assert LookupTable.MAX_HIGH_CARD == 7462

    def test_max_to_rank_class_has_nine_entries(self):
        assert len(LookupTable.MAX_TO_RANK_CLASS) == 9

    def test_max_to_rank_class_values_in_range(self):
        for v in LookupTable.MAX_TO_RANK_CLASS.values():
            assert 1 <= v <= 9

    def test_rank_class_to_string_has_nine_entries(self):
        assert len(LookupTable.RANK_CLASS_TO_STRING) == 9

    def test_rank_class_to_string_keys(self):
        assert set(LookupTable.RANK_CLASS_TO_STRING.keys()) == set(range(1, 10))


class TestLookupTableConstruction:
    def test_flush_lookup_non_empty(self, evaluator):
        assert len(evaluator.table.flush_lookup) > 0

    def test_unsuited_lookup_non_empty(self, evaluator):
        assert len(evaluator.table.unsuited_lookup) > 0

    def test_total_entries_equals_7462(self, evaluator):
        total = len(evaluator.table.flush_lookup) + len(evaluator.table.unsuited_lookup)
        assert total == 7462


class TestGospersHack:
    def test_same_popcount(self):
        table = LookupTable()
        gen = table.get_lexographically_next_bit_sequence(0b11111)
        for _ in range(5):
            val = next(gen)
            assert bin(val).count("1") == 5

    def test_ascending_order(self):
        table = LookupTable()
        gen = table.get_lexographically_next_bit_sequence(0b11111)
        prev = 0b11111
        for _ in range(5):
            val = next(gen)
            assert val > prev
            prev = val


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TestEvaluatorRoyalFlush:
    @pytest.mark.parametrize("suit", ["spades", "hearts", "diamonds", "clubs"])
    def test_royal_flush_is_rank_one(self, evaluator, suit):
        hand = [make_card(r, suit) for r in [14, 13, 12, 11, 10]]
        assert evaluator.evaluate(hand, []) == 1

    def test_royal_flush_rank_class(self, evaluator):
        hand = [make_card(r, "spades") for r in [14, 13, 12, 11, 10]]
        rank = evaluator.evaluate(hand, [])
        assert evaluator.get_rank_class(rank) == 1


class TestEvaluatorHandOrdering:
    def test_straight_flush_range(self, evaluator):
        # K-high straight flush (not royal)
        hand = [make_card(r, "hearts") for r in [13, 12, 11, 10, 9]]
        rank = evaluator.evaluate(hand, [])
        assert rank <= LookupTable.MAX_STRAIGHT_FLUSH

    def test_four_of_a_kind_range(self, evaluator):
        hand = [make_card(14, s) for s in ["spades", "hearts", "diamonds", "clubs"]]
        hand.append(make_card(2, "spades"))
        rank = evaluator.evaluate(hand, [])
        assert LookupTable.MAX_STRAIGHT_FLUSH < rank <= LookupTable.MAX_FOUR_OF_A_KIND

    def test_full_house_beats_flush(self, evaluator):
        full_house = [
            make_card(14, "spades"), make_card(14, "hearts"), make_card(14, "diamonds"),
            make_card(13, "spades"), make_card(13, "hearts"),
        ]
        flush = [make_card(r, "spades") for r in [14, 12, 10, 8, 6]]
        rank_fh = evaluator.evaluate(full_house, [])
        rank_fl = evaluator.evaluate(flush, [])
        assert rank_fh < rank_fl

    def test_flush_beats_straight(self, evaluator):
        flush = [make_card(r, "spades") for r in [14, 12, 10, 8, 6]]
        straight = [
            make_card(14, "spades"), make_card(13, "hearts"),
            make_card(12, "diamonds"), make_card(11, "clubs"),
            make_card(10, "spades"),
        ]
        rank_fl = evaluator.evaluate(flush, [])
        rank_st = evaluator.evaluate(straight, [])
        assert rank_fl < rank_st


class TestEvaluatorMultiCard:
    def test_seven_card_hand_selects_best_five(self, evaluator):
        # Royal flush in spades plus two irrelevant cards
        hand = [make_card(r, "spades") for r in [14, 13, 12, 11, 10]]
        board = [make_card(2, "hearts"), make_card(3, "clubs")]
        rank = evaluator.evaluate(hand, board)
        assert rank == 1

    def test_six_card_hand(self, evaluator):
        hand = [make_card(r, "spades") for r in [14, 13, 12, 11, 10]]
        board = [make_card(2, "hearts")]
        rank = evaluator.evaluate(hand, board)
        assert rank == 1


class TestGetRankClass:
    def test_rank_1_is_class_1(self, evaluator):
        assert evaluator.get_rank_class(1) == 1

    def test_rank_7462_is_class_9(self, evaluator):
        assert evaluator.get_rank_class(7462) == 9

    def test_invalid_rank_raises(self, evaluator):
        with pytest.raises(ValueError):
            evaluator.get_rank_class(7463)

    @pytest.mark.parametrize("rank,expected_class", [
        (10, 1),   # straight flush upper boundary
        (11, 2),   # start of four of a kind range
        (166, 2),  # four of a kind upper boundary
        (322, 3),  # full house upper boundary
        (323, 4),  # start of flush range
    ])
    def test_rank_class_boundaries(self, evaluator, rank, expected_class):
        assert evaluator.get_rank_class(rank) == expected_class


class TestClassToString:
    def test_class_1_straight_flush(self, evaluator):
        assert evaluator.class_to_string(1) == "Straight Flush"

    def test_class_9_high_card(self, evaluator):
        assert evaluator.class_to_string(9) == "High Card"

    @pytest.mark.parametrize("cls,name", [
        (1, "Straight Flush"), (2, "Four of a Kind"), (3, "Full House"),
        (4, "Flush"), (5, "Straight"), (6, "Three of a Kind"),
        (7, "Two Pair"), (8, "Pair"), (9, "High Card"),
    ])
    def test_all_classes(self, evaluator, cls, name):
        assert evaluator.class_to_string(cls) == name


class TestFiveCardRoutePaths:
    """Verify _five() dispatches to flush vs unsuited tables correctly."""

    def test_flush_hand_uses_flush_lookup(self, evaluator):
        # A♠ K♠ Q♠ J♠ 9♠ — flush (not straight flush)
        hand = [make_card(r, "spades") for r in [14, 13, 12, 11, 9]]
        rank = evaluator.evaluate(hand, [])
        assert LookupTable.MAX_FULL_HOUSE < rank <= LookupTable.MAX_FLUSH

    def test_unsuited_hand_uses_unsuited_lookup(self, evaluator):
        # A♠ A♥ A♦ K♠ K♥ — full house
        hand = [
            make_card(14, "spades"), make_card(14, "hearts"), make_card(14, "diamonds"),
            make_card(13, "spades"), make_card(13, "hearts"),
        ]
        rank = evaluator.evaluate(hand, [])
        assert LookupTable.MAX_FOUR_OF_A_KIND < rank <= LookupTable.MAX_FULL_HOUSE


class TestMultiCardBestHandSelection:
    def test_six_card_best_is_not_first_five(self, evaluator):
        # Add a weak card first; the best 5 excludes it
        weak = make_card(2, "hearts")
        royal = [make_card(r, "spades") for r in [14, 13, 12, 11, 10]]
        # hand = [weak] + first 4 of royal; board = [10s]
        rank = evaluator.evaluate([weak] + royal[:4], [royal[4]])
        assert rank == 1  # royal flush still found

    def test_seven_card_best_requires_selection(self, evaluator):
        # Board has 5 cards making a royal flush; player holds two junk cards
        board = [make_card(r, "spades") for r in [14, 13, 12, 11, 10]]
        hand = [make_card(2, "clubs"), make_card(3, "diamonds")]
        rank = evaluator.evaluate(hand, board)
        assert rank == 1  # royal flush found among all 7


class TestGetFiveCardRankPercentage:
    def test_rank_1_near_zero(self, evaluator):
        pct = evaluator.get_five_card_rank_percentage(1)
        assert pct < 0.001

    def test_rank_7462_is_one(self, evaluator):
        pct = evaluator.get_five_card_rank_percentage(7462)
        assert pct == pytest.approx(1.0)

    def test_range(self, evaluator):
        for rank in [1, 1000, 3731, 7462]:
            pct = evaluator.get_five_card_rank_percentage(rank)
            assert 0.0 <= pct <= 1.0
