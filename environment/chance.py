"""Stochastic transitions for the poker environment.

The Deck class encapsulates all chance nodes in a poker hand:
shuffling, dealing private hole cards, and dealing community cards.
Cards are represented as 32-bit integers throughout (see utils.py).
"""

from __future__ import annotations

import numpy as np

from environment.utils import make_deck_arr


class Deck:
    """Shuffled deck stored as a numpy int32 array of card integers.

    All deal operations return 32-bit card integers (see ``utils.py``
    for the encoding).

    Parameters
    ----------
    low_rank : int
        Lowest card rank to include (2=Two, ..., 14=Ace). Default 2.
    high_rank : int
        Highest card rank to include. Default 14.

    Attributes
    ----------
    remaining : numpy.ndarray
        Undealt portion of the deck as a 1-D int32 array.
    """

    __slots__ = ("_cards", "_idx")

    def __init__(self, low_rank: int = 2, high_rank: int = 14):
        self._cards: np.ndarray = make_deck_arr(low_rank, high_rank)
        np.random.shuffle(self._cards)
        self._idx: int = 0

    def deal_private_cards(self, players) -> None:
        """Deal 2 hole cards to each player in standard 2-pass dealing order.

        Each player receives their first card before any player receives
        their second card, matching real dealing convention.

        Parameters
        ----------
        players : list[Player]
            Players to deal to, in betting order.
        """
        for _ in range(2):
            for player in players:
                player._cards += (int(self._cards[self._idx]),)
                self._idx += 1

    def deal_community(self, n: int) -> tuple:
        """Deal n community cards.

        Parameters
        ----------
        n : int
            Number of cards to deal (3 for flop, 1 for turn/river).

        Returns
        -------
        tuple
            Tuple of n eval_card integers.
        """
        cards = tuple(int(c) for c in self._cards[self._idx: self._idx + n])
        self._idx += n
        return cards

    @property
    def remaining(self) -> np.ndarray:
        """Return the undealt portion of the deck as a numpy array."""
        return self._cards[self._idx:]
