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

    def capture(self) -> int:
        """Snapshot the deal cursor for make/undo.

        Dealing only advances ``_idx``; the ``_cards`` array is never
        mutated by ``deal_community`` / ``deal_private_cards``, so the
        cursor alone fully captures the deck's dealing state.
        """
        return self._idx

    def restore(self, cursor: int) -> None:
        """Restore the deal cursor captured by :meth:`capture`."""
        self._idx = cursor

    @property
    def remaining(self) -> np.ndarray:
        """Return the undealt portion of the deck as a numpy array."""
        return self._cards[self._idx:]

    def replace_drawn(
        self,
        old_cards: tuple,
        new_cards: tuple,
    ) -> None:
        """Swap card values in ``_cards`` so ``new_cards`` occupy
        ``old_cards``' positions in the drawn segment as a multiset.

        Used by :meth:`PokerEnv.with_hole_cards` to keep the deck
        consistent after a hole-card replacement: the cards that
        appear in ``new`` but not ``old`` are moved into the
        positions held by cards in ``old`` but not ``new``; the
        displaced cards land where the new cards used to be.
        ``_idx`` is unchanged.

        The operation is set-based rather than position-by-position,
        so it is robust when ``new`` reuses one of the seat's own
        current cards in a different slot (a pairwise iteration
        would lose the second card to a transient ordering error).

        Caller's contract — each card in ``set(new) - set(old)``
        must currently occupy a position outside any other dealt
        slot (community, another seat's hole).  Otherwise the swap
        would corrupt that slot.  :meth:`PokerEnv.with_hole_cards`
        validates this before calling.

        Parameters
        ----------
        old_cards : tuple[int, ...]
            Card ints currently in the drawn segment that may be
            displaced (typically the seat's prior hole).
        new_cards : tuple[int, ...]
            Card ints that should end up in the seat's hole
            positions.  Cards present in both ``old`` and ``new``
            are no-ops; the difference is what actually swaps.
        """
        old_set = set(int(c) for c in old_cards)
        new_set = set(int(c) for c in new_cards)
        dropped = sorted(old_set - new_set)  # cards leaving the slot
        added = sorted(new_set - old_set)    # cards entering the slot
        # |dropped| == |added| holds because |old| == |new| as input.
        for d, a in zip(dropped, added):
            p_d = int(np.where(self._cards == d)[0][0])
            p_a = int(np.where(self._cards == a)[0][0])
            tmp = int(self._cards[p_d])
            self._cards[p_d] = self._cards[p_a]
            self._cards[p_a] = tmp

    def shuffle_undealt(self) -> None:
        """Shuffle the undealt segment in place (positions ``>= _idx``).

        Drawn segment (``< _idx``) is untouched.  Called by
        :meth:`PokerEnv.with_hole_cards` after :meth:`replace_drawn`
        so the next community deal samples uniformly over the
        remaining cards instead of preferring the specific positions
        that received displaced cards from the swap.  Uses the
        global numpy RNG, matching the construction-time shuffle.
        """
        np.random.shuffle(self._cards[self._idx:])
