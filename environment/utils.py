"""Card encoding and utility functions for the poker environment.

Cards are represented as 32-bit integers throughout the game layer.
The encoding follows the Cactus Kev scheme:

                  bitrank     suit rank   prime
            +--------+--------+--------+--------+
            |xxxbbbbb|bbbbbbbb|cdhsrrrr|xxpppppp|
            +--------+--------+--------+--------+

  1) p = prime number of rank (deuce=2, trey=3, four=5, ..., ace=41)
  2) r = rank of card (deuce=0, trey=1, ..., ace=12)
  3) cdhs = suit of card (one bit per suit: clubs=8, diamonds=4, hearts=2, spades=1)
  4) b = bit set for the rank position (used in flush/straight detection)

This encoding supports O(1) hand evaluation via prime-product hashing and
bitwise flush detection. It is used by the environment, clustering pipeline,
hand evaluator, and visualisation layer.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Encoding constants
# ---------------------------------------------------------------------------

_STR_RANKS = "23456789TJQKA"

INT_RANKS = range(13)
"""Zero-based rank indices (0=deuce, 1=trey, ..., 12=ace)."""

CARD_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
"""Prime number assigned to each rank (index 0=deuce, 12=ace).

Used to compute unique prime products for hand equivalence classes.
"""

_CHAR_RANK_TO_INT_RANK: dict = dict(zip(list(_STR_RANKS), INT_RANKS))
_CHAR_SUIT_TO_INT_SUIT: dict = {"s": 1, "h": 2, "d": 4, "c": 8}
_INT_SUIT_TO_CHAR_SUIT = "xshxdxxxc"
_PRETTY_SUITS: dict = {1: chr(9824), 2: chr(9829), 4: chr(9830), 8: chr(9827)}
_PRETTY_REDS = [2, 4]

SUITS = ("spades", "hearts", "diamonds", "clubs")
"""All four suit names in deck order."""

_SUIT_CHARS: dict = {"spades": "s", "hearts": "h", "diamonds": "d", "clubs": "c"}
_RANK_CHARS: dict = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
}
_RANK_NAMES: dict = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 11: "jack", 12: "queen",
    13: "king", 14: "ace",
}
_SUIT_INT_TO_STR: dict = {1: "spades", 2: "hearts", 4: "diamonds", 8: "clubs"}


# ---------------------------------------------------------------------------
# Low-level bit operations (used by Evaluator and LookupTable)
# ---------------------------------------------------------------------------

def new_card(string: str) -> int:
    """Convert a two-character card string to a 32-bit card integer.

    Parameters
    ----------
    string : str
        Two-character card string where the first character is the rank
        (``'2'``–``'9'``, ``'T'``, ``'J'``, ``'Q'``, ``'K'``, ``'A'``)
        and the second is the suit (``'s'``, ``'h'``, ``'d'``, ``'c'``).
        Example: ``'As'``, ``'Th'``, ``'2c'``.

    Returns
    -------
    int
        32-bit card integer encoding rank, suit, bitrank, and prime.

    Raises
    ------
    KeyError
        If the rank or suit character is not recognised.
    """
    rank_int = _CHAR_RANK_TO_INT_RANK[string[0]]
    suit_int = _CHAR_SUIT_TO_INT_SUIT[string[1]]
    return (1 << rank_int << 16) | (suit_int << 12) | (rank_int << 8) | CARD_PRIMES[rank_int]


def get_rank_int(card_int: int) -> int:
    """Extract the 0-based rank index from a card integer.

    Parameters
    ----------
    card_int : int
        32-bit card integer.

    Returns
    -------
    int
        Rank index in the range [0, 12] where 0=deuce and 12=ace.
    """
    return (card_int >> 8) & 0xF


def get_suit_int(card_int: int) -> int:
    """Extract the suit integer from a card integer.

    Parameters
    ----------
    card_int : int
        32-bit card integer.

    Returns
    -------
    int
        Suit as a power-of-two integer: 1=spades, 2=hearts, 4=diamonds,
        8=clubs.
    """
    return (card_int >> 12) & 0xF


def get_bitrank_int(card_int: int) -> int:
    """Extract the bitrank field from a card integer.

    The bitrank field has exactly one bit set, corresponding to the card's
    rank position. It is used for straight and flush detection.

    Parameters
    ----------
    card_int : int
        32-bit card integer.

    Returns
    -------
    int
        13-bit integer with one bit set at the rank position.
    """
    return (card_int >> 16) & 0x1FFF


def get_prime(card_int: int) -> int:
    """Extract the prime encoding of the rank from a card integer.

    Parameters
    ----------
    card_int : int
        32-bit card integer.

    Returns
    -------
    int
        Prime number for the card's rank (2 for deuce, 41 for ace).
    """
    return card_int & 0x3F


def prime_product_from_hand(card_ints) -> int:
    """Compute the prime product of all cards in a hand.

    Multiplying the lowest 8 bits (the prime) of each card produces a
    unique integer for each distinct set of ranks, which is used as a
    hash key into the hand evaluation lookup table.

    Parameters
    ----------
    card_ints : list[int]
        Card integers for the hand (typically 5 cards).

    Returns
    -------
    int
        Product of the prime encodings of all cards.
    """
    product = 1
    for c in card_ints:
        product *= c & 0xFF
    return product


def prime_product_from_rankbits(rankbits: int) -> int:
    """Compute the prime product from a rankbits integer.

    Used when all ranks in the hand are known to be distinct (flushes and
    straights), allowing the rank bits to be ORed together before hashing.

    Parameters
    ----------
    rankbits : int
        13-bit integer where each set bit represents a rank present in
        the hand.

    Returns
    -------
    int
        Product of the primes for every rank whose bit is set.
    """
    product = 1
    for i in INT_RANKS:
        if rankbits & (1 << i):
            product *= CARD_PRIMES[i]
    return product


def hand_to_binary(card_strs) -> list:
    """Convert a list of card strings to card integers.

    Parameters
    ----------
    card_strs : list[str]
        Two-character card strings (e.g. ``['As', 'Kh', 'Qd', 'Jc', 'Ts']``).

    Returns
    -------
    list[int]
        Corresponding 32-bit card integers.
    """
    return [new_card(c) for c in card_strs]


def card_int_to_str(card_int: int) -> str:
    """Return the two-character string representation of a card integer.

    Parameters
    ----------
    card_int : int
        32-bit card integer.

    Returns
    -------
    str
        Two-character string, e.g. ``'Ah'``, ``'Ks'``, ``'2c'``.
    """
    return _STR_RANKS[get_rank_int(card_int)] + _INT_SUIT_TO_CHAR_SUIT[get_suit_int(card_int)]


def card_int_to_pretty_str(card_int: int) -> str:
    """Return a formatted string with a unicode suit symbol.

    Uses terminal colour (red for hearts/diamonds) when the
    ``termcolor`` package is available.

    Parameters
    ----------
    card_int : int
        32-bit card integer.

    Returns
    -------
    str
        Formatted string such as ``'[A♠]'`` or ``'[K♥]'``.
    """
    color = False
    try:
        from termcolor import colored
        color = True
    except ImportError:
        pass
    suit_int = get_suit_int(card_int)
    s = _PRETTY_SUITS[suit_int]
    if color and suit_int in _PRETTY_REDS:
        s = colored(s, "red")
    return f"[{_STR_RANKS[get_rank_int(card_int)]}{s}]"


def card_int_to_binary_str(card_int: int) -> str:
    """Return the binary representation of a card integer for debugging.

    Parameters
    ----------
    card_int : int
        32-bit card integer.

    Returns
    -------
    str
        Human-readable binary string grouped into nibbles.
    """
    bstr = bin(card_int)[2:][::-1]
    output = list("".join(["0000" + "\t"] * 7) + "0000")
    for i in range(len(bstr)):
        output[i + int(i / 4)] = bstr[i]
    output.reverse()
    return "".join(output)


# ---------------------------------------------------------------------------
# Higher-level card API
# ---------------------------------------------------------------------------

def make_card(rank: int, suit: str) -> int:
    """Create a card integer from a rank number and a suit name.

    Parameters
    ----------
    rank : int
        Card rank as an integer from 2 (deuce) to 14 (ace).
    suit : str
        Suit name: one of ``'spades'``, ``'hearts'``, ``'diamonds'``,
        ``'clubs'``.

    Returns
    -------
    int
        32-bit card integer.

    Raises
    ------
    KeyError
        If ``rank`` is not in [2, 14] or ``suit`` is not recognised.
    """
    return new_card(_RANK_CHARS[rank] + _SUIT_CHARS[suit])


def make_deck_arr(low_rank: int = 2, high_rank: int = 14) -> np.ndarray:
    """Return an unshuffled deck as a numpy array of card integers.

    Parameters
    ----------
    low_rank : int
        Lowest rank to include, from 2 (deuce) to 14 (ace). Default 2.
    high_rank : int
        Highest rank to include, from 2 (deuce) to 14 (ace). Default 14.

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``((high_rank - low_rank + 1) * 4,)`` with
        dtype ``int32``, containing one card integer per card.
    """
    return np.array(
        [make_card(r, s) for r in range(low_rank, high_rank + 1) for s in SUITS],
        dtype=np.int32,
    )


def card_rank_int(c: int) -> int:
    """Extract the rank as an integer in [2, 14].

    Parameters
    ----------
    c : int
        32-bit card integer.

    Returns
    -------
    int
        Rank from 2 (deuce) to 14 (ace).
    """
    return get_rank_int(c) + 2


def card_rank_str(c: int) -> str:
    """Extract the rank as a lowercase English word.

    Parameters
    ----------
    c : int
        32-bit card integer.

    Returns
    -------
    str
        Rank name, e.g. ``'ace'``, ``'king'``, ``'two'``.
    """
    return _RANK_NAMES[card_rank_int(c)]


def card_rank_char(c: int) -> str:
    """Extract the rank as a single display character.

    Parameters
    ----------
    c : int
        32-bit card integer.

    Returns
    -------
    str
        Single character: ``'2'``–``'9'``, ``'T'``, ``'J'``, ``'Q'``,
        ``'K'``, or ``'A'``.
    """
    return _RANK_CHARS[card_rank_int(c)]


def card_suit_str(c: int) -> str:
    """Extract the suit as a lowercase string.

    Parameters
    ----------
    c : int
        32-bit card integer.

    Returns
    -------
    str
        Suit name: ``'spades'``, ``'hearts'``, ``'diamonds'``, or
        ``'clubs'``.
    """
    return _SUIT_INT_TO_STR[get_suit_int(c)]


def card_str(c: int) -> str:
    """Return the two-character string for a card integer.

    Parameters
    ----------
    c : int
        32-bit card integer.

    Returns
    -------
    str
        Two-character string, e.g. ``'Ah'``, ``'Ks'``, ``'2c'``.
    """
    return card_int_to_str(c)


def card_pretty_str(c: int) -> str:
    """Return a formatted string with a unicode suit symbol.

    Parameters
    ----------
    c : int
        32-bit card integer.

    Returns
    -------
    str
        Formatted string such as ``'[A♠]'`` or ``'[K♥]'``.
    """
    return card_int_to_pretty_str(c)
