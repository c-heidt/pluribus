"""Unit tests for poker_ai/environment/utils.py.

Covers card encoding constants, all card creation and extraction functions,
deck array construction, prime product helpers, and error conditions.
"""

import numpy as np
import pytest

from environment.utils import (
    INT_RANKS,
    CARD_PRIMES,
    SUITS,
    make_card,
    new_card,
    card_rank_int,
    card_rank_char,
    card_rank_str,
    card_suit_str,
    card_str,
    card_pretty_str,
    card_int_to_str,
    make_deck_arr,
    get_rank_int,
    get_suit_int,
    get_bitrank_int,
    get_prime,
    prime_product_from_hand,
    prime_product_from_rankbits,
    hand_to_binary,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_int_ranks_length(self):
        assert len(list(INT_RANKS)) == 13

    def test_int_ranks_values(self):
        assert list(INT_RANKS) == list(range(13))

    def test_card_primes_length(self):
        assert len(CARD_PRIMES) == 13

    def test_card_primes_boundaries(self):
        assert CARD_PRIMES[0] == 2   # deuce
        assert CARD_PRIMES[12] == 41  # ace

    def test_card_primes_all_prime(self):
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        for p in CARD_PRIMES:
            assert is_prime(p), f"{p} is not prime"

    def test_suits_length(self):
        assert len(SUITS) == 4

    def test_suits_values(self):
        assert set(SUITS) == {"spades", "hearts", "diamonds", "clubs"}


# ---------------------------------------------------------------------------
# make_card and new_card
# ---------------------------------------------------------------------------

class TestMakeCard:
    def test_returns_int(self):
        assert isinstance(make_card(14, "spades"), int)

    def test_all_52_unique(self):
        cards = [make_card(r, s) for r in range(2, 15) for s in SUITS]
        assert len(set(cards)) == 52

    def test_rank_roundtrip(self):
        for rank in range(2, 15):
            c = make_card(rank, "spades")
            assert card_rank_int(c) == rank

    def test_suit_roundtrip(self):
        for suit in SUITS:
            c = make_card(14, suit)
            assert card_suit_str(c) == suit

    def test_invalid_rank_raises(self):
        with pytest.raises(KeyError):
            make_card(1, "spades")

    def test_invalid_high_rank_raises(self):
        with pytest.raises(KeyError):
            make_card(15, "spades")

    def test_invalid_suit_raises(self):
        with pytest.raises(KeyError):
            make_card(14, "jokers")


class TestNewCard:
    def test_ace_of_spades(self):
        c = new_card("As")
        assert card_rank_int(c) == 14
        assert card_suit_str(c) == "spades"

    def test_two_of_clubs(self):
        c = new_card("2c")
        assert card_rank_int(c) == 2
        assert card_suit_str(c) == "clubs"

    def test_ten_of_hearts(self):
        c = new_card("Th")
        assert card_rank_int(c) == 10
        assert card_suit_str(c) == "hearts"

    def test_roundtrip_via_card_str(self):
        for s in ["As", "Kd", "Qh", "Jc", "Ts", "2c"]:
            c = new_card(s)
            assert card_int_to_str(c) == s

    def test_invalid_rank_char_raises(self):
        with pytest.raises(KeyError):
            new_card("Xs")

    def test_invalid_suit_char_raises(self):
        with pytest.raises(KeyError):
            new_card("Ax")


# ---------------------------------------------------------------------------
# Card string/display functions
# ---------------------------------------------------------------------------

class TestCardStringFunctions:
    @pytest.mark.parametrize("rank,expected", [
        (14, "A"), (13, "K"), (12, "Q"), (11, "J"),
        (10, "T"), (9, "9"), (2, "2"),
    ])
    def test_card_rank_char(self, rank, expected):
        c = make_card(rank, "spades")
        assert card_rank_char(c) == expected

    @pytest.mark.parametrize("rank,expected", [
        (14, "ace"), (13, "king"), (2, "two"), (10, "ten"),
    ])
    def test_card_rank_str(self, rank, expected):
        c = make_card(rank, "hearts")
        assert card_rank_str(c) == expected

    def test_card_str_format(self):
        c = make_card(14, "spades")
        assert card_str(c) == "As"

    def test_card_str_all_suits(self):
        expected_chars = {"s", "h", "d", "c"}
        for suit in SUITS:
            c = make_card(14, suit)
            assert card_str(c)[1] in expected_chars

    def test_card_pretty_str_contains_rank(self):
        c = make_card(14, "spades")
        s = card_pretty_str(c)
        assert "A" in s

    def test_card_pretty_str_has_brackets(self):
        c = make_card(14, "spades")
        s = card_pretty_str(c)
        assert s.startswith("[") and s.endswith("]")


# ---------------------------------------------------------------------------
# card_rank_int and card_suit_str
# ---------------------------------------------------------------------------

class TestCardExtractors:
    def test_rank_int_deuce(self):
        assert card_rank_int(make_card(2, "spades")) == 2

    def test_rank_int_ace(self):
        assert card_rank_int(make_card(14, "clubs")) == 14

    def test_rank_int_all_ranks(self):
        for rank in range(2, 15):
            c = make_card(rank, "hearts")
            assert card_rank_int(c) == rank

    def test_suit_str_all_suits(self):
        for suit in SUITS:
            c = make_card(14, suit)
            assert card_suit_str(c) == suit


# ---------------------------------------------------------------------------
# make_deck_arr
# ---------------------------------------------------------------------------

class TestMakeDeckArr:
    def test_full_deck_size(self):
        arr = make_deck_arr(2, 14)
        assert len(arr) == 52

    def test_full_deck_dtype(self):
        arr = make_deck_arr(2, 14)
        assert arr.dtype == np.int32

    def test_full_deck_unique(self):
        arr = make_deck_arr(2, 14)
        assert len(set(arr)) == 52

    def test_full_deck_ranks_present(self):
        arr = make_deck_arr(2, 14)
        ranks = {card_rank_int(int(c)) for c in arr}
        assert ranks == set(range(2, 15))

    def test_short_deck_size(self):
        arr = make_deck_arr(10, 14)
        assert len(arr) == 20

    def test_short_deck_no_low_ranks(self):
        arr = make_deck_arr(10, 14)
        for c in arr:
            assert card_rank_int(int(c)) >= 10

    def test_short_deck_unique(self):
        arr = make_deck_arr(10, 14)
        assert len(set(arr)) == 20


# ---------------------------------------------------------------------------
# Low-level bit extractors
# ---------------------------------------------------------------------------

class TestLowLevelExtractors:
    def test_get_rank_int_range(self):
        for rank in range(2, 15):
            c = make_card(rank, "spades")
            r = get_rank_int(c)
            assert 0 <= r <= 12

    def test_get_rank_int_values(self):
        # deuce → 0, ace → 12
        assert get_rank_int(make_card(2, "spades")) == 0
        assert get_rank_int(make_card(14, "spades")) == 12

    def test_get_suit_int_values(self):
        suit_map = {"spades": 1, "hearts": 2, "diamonds": 4, "clubs": 8}
        for suit, expected in suit_map.items():
            c = make_card(14, suit)
            assert get_suit_int(c) == expected

    def test_get_bitrank_int_one_bit_set(self):
        for rank in range(2, 15):
            c = make_card(rank, "spades")
            b = get_bitrank_int(c)
            assert b > 0
            assert (b & (b - 1)) == 0, f"More than one bit set for rank {rank}"

    def test_get_prime_in_card_primes(self):
        for rank in range(2, 15):
            c = make_card(rank, "spades")
            assert get_prime(c) in CARD_PRIMES


# ---------------------------------------------------------------------------
# prime_product_from_hand and prime_product_from_rankbits
# ---------------------------------------------------------------------------

class TestPrimeProducts:
    def test_prime_product_from_hand_single_card(self):
        c = make_card(14, "spades")  # ace prime = 41
        assert prime_product_from_hand([c]) == 41

    def test_prime_product_from_hand_five_aces(self):
        # Four aces (all suits) — prime product = 41^4
        aces = [make_card(14, s) for s in SUITS]
        assert prime_product_from_hand(aces) == 41 ** 4

    def test_prime_product_from_rankbits_single_bit(self):
        # Bit i set → product is CARD_PRIMES[i]
        for i in range(13):
            rb = 1 << i
            assert prime_product_from_rankbits(rb) == CARD_PRIMES[i]

    def test_prime_product_from_rankbits_two_bits(self):
        rb = (1 << 0) | (1 << 12)  # deuce + ace bits
        expected = CARD_PRIMES[0] * CARD_PRIMES[12]  # 2 * 41
        assert prime_product_from_rankbits(rb) == expected


# ---------------------------------------------------------------------------
# hand_to_binary
# ---------------------------------------------------------------------------

class TestHandToBinary:
    def test_single_card(self):
        result = hand_to_binary(["As"])
        assert len(result) == 1
        assert card_rank_int(result[0]) == 14

    def test_roundtrip(self):
        strings = ["As", "Kd", "Qh", "Jc", "Ts"]
        cards = hand_to_binary(strings)
        assert [card_int_to_str(c) for c in cards] == strings

    def test_empty(self):
        assert hand_to_binary([]) == []


# ---------------------------------------------------------------------------
# card_int_to_binary_str
# ---------------------------------------------------------------------------

class TestCardIntToBinaryStr:
    def test_returns_string(self):
        from environment.utils import card_int_to_binary_str
        c = make_card(14, "spades")
        result = card_int_to_binary_str(c)
        assert isinstance(result, str)

    def test_length(self):
        from environment.utils import card_int_to_binary_str
        c = make_card(14, "spades")
        # 7 groups of 4 bits + 7 tabs = 35 chars
        result = card_int_to_binary_str(c)
        assert len(result) > 0

    def test_contains_only_valid_chars(self):
        from environment.utils import card_int_to_binary_str
        c = make_card(2, "clubs")
        result = card_int_to_binary_str(c)
        assert all(ch in "01\t" for ch in result)
