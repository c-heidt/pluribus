"""Shared combinatorics helpers for information-set indexing.

Used by both the consumption side (``lookup.MemmapLookup``) and the build side
(``build.card_combos.CardCombos``).
"""
from typing import Tuple

try:
    from math import comb
except ImportError:
    from scipy.special import comb as _comb

    def comb(n, k):
        return int(_comb(n, k, exact=True))


def lex_rank(combo: Tuple[int, ...], n: int) -> int:
    """Lexicographic rank of a combination in O(k) time.

    Given a k-combination ``(c_0, c_1, ..., c_{k-1})`` with strictly
    ascending elements drawn from ``{0, 1, ..., n-1}``, returns its
    position (0-based) in lexicographic order among all ``C(n, k)``
    combinations.

    Uses the identity
    ``sum_{j=a}^{b-1} C(n-j-1, r) = C(n-a, r+1) - C(n-b, r+1)``
    to avoid an inner loop.
    """
    k = len(combo)
    rank = 0
    prev = -1
    for i in range(k):
        start = prev + 1
        remaining = k - i
        rank += comb(n - start, remaining) - comb(n - combo[i], remaining)
        prev = combo[i]
    return rank
