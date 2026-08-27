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

    __slots__ = ("_cards", "_idx", "_board_start")

    def __init__(self, low_rank: int = 2, high_rank: int = 14):
        self._cards: np.ndarray = make_deck_arr(low_rank, high_rank)
        np.random.shuffle(self._cards)
        self._idx: int = 0
        # Position in ``_cards`` where the community (board) region begins —
        # set once when the private hole cards finish dealing (they occupy
        # ``[0:_board_start)``, so ``_board_start == 2 * n_players``).  The board
        # occupies ``[_board_start : _board_start + 5)``; :meth:`board_runout`
        # reads it without any caller re-deriving the ``2 * n`` offset.  Fixed for
        # the hand (private dealing happens once), so make/undo leaves it alone.
        self._board_start: int = 0

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
        # Private cards now fill ``[0:_idx)``; the board region starts here.
        self._board_start = self._idx

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

    def board_runout(self, board_len: int = 5) -> np.ndarray:
        """The ``board_len`` cards that form this hand's final board, in order.

        The community region sits at ``[_board_start : _board_start + board_len)``
        immediately after the private cards, so this returns the board cards
        already dealt **plus** the next undealt cards that will complete the
        runout (turn/river peek), given the current deck order.  This is the
        single accessor for the board layout: readers (the compiled ``FastState``
        and its reference twin) call it instead of slicing ``_cards`` with a
        hand-rolled ``2 * n`` offset, and any writer that rebuilds the deck must
        produce a layout consistent with it (private cards, then the board, then
        the rest).  ``_board_start`` is set by :meth:`deal_private_cards`.
        """
        return self._cards[self._board_start: self._board_start + board_len]

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

    def force_next(self, cards) -> list:
        """Make ``cards`` the next cards :meth:`deal_community` will hand out.

        Swaps each requested card into position ``_idx + k`` of ``_cards``, so a
        subsequent ``deal_community(len(cards))`` returns exactly ``cards`` in the
        given order.  Returns a **token** that :meth:`unforce` consumes to put the
        array back; the two are exact inverses (each swap is a transposition,
        replayed in reverse).

        This is the engine's *counterfactual board* primitive: it answers "what if
        this street had come ``c`` instead?" down the same betting line, which is
        what an offline chance-node control variate needs
        (:mod:`evaluation.aivat`).  Nothing in the played game calls it.

        .. warning::
           **Mutates ``_cards``, which no other deal operation does.**
           :meth:`capture` / :meth:`restore` snapshot only the cursor, precisely
           because dealing never disturbs the array — so
           :meth:`PokerEnv.undo` will *not* reverse a force.  Every caller must
           pair ``force_next`` with :meth:`unforce`, and should only ever force on
           a throwaway copy (:meth:`PokerEnv.with_hole_cards`), never on the env
           whose deal is the played hand.

        Parameters
        ----------
        cards : Sequence[int]
            Card integers to pin, in deal order.  Each must still be undealt (at a
            position ``>= _idx``) and distinct from the others.

        Returns
        -------
        list[tuple[int, int]]
            Swap token for :meth:`unforce`.  Empty when every card already sat in
            its target position (a no-op force).

        Raises
        ------
        ValueError
            If a card is already dealt, absent from the deck, or requested twice,
            or if ``cards`` runs past the end of the deck.
        """
        n = len(cards)
        if self._idx + n > self._cards.shape[0]:
            raise ValueError(
                f"force_next: {n} cards requested but only "
                f"{self._cards.shape[0] - self._idx} remain undealt."
            )
        swaps: list = []
        try:
            for k, card in enumerate(cards):
                target = self._idx + k
                card = int(card)
                # Search from ``target``, not from ``_idx``: cards pinned earlier in
                # this same call already occupy ``[_idx:target)`` and must not be
                # re-found.  A card that is dealt (or already pinned) is therefore
                # simply absent from the searched region — the error we want.
                hits = np.where(self._cards[target:] == card)[0]
                if hits.size == 0:
                    raise ValueError(
                        f"force_next: card {card} is not available to deal "
                        f"(already dealt, pinned twice, or not in this deck)."
                    )
                pos = target + int(hits[0])
                if pos != target:
                    self._cards[target], self._cards[pos] = (
                        int(self._cards[pos]),
                        int(self._cards[target]),
                    )
                    swaps.append((target, pos))
        except Exception:
            # Never leave the deck half-forced: a partial force would silently
            # corrupt the runout of whatever the caller does next.
            self.unforce(swaps)
            raise
        return swaps

    def unforce(self, token) -> None:
        """Undo a :meth:`force_next`, restoring ``_cards`` exactly.

        Replays the token's transpositions in reverse.  ``_idx`` is not touched —
        a caller that also stepped the env restores the cursor through
        :meth:`PokerEnv.undo` (or :meth:`restore`), in either order.
        """
        for target, pos in reversed(token):
            self._cards[target], self._cards[pos] = (
                int(self._cards[pos]),
                int(self._cards[target]),
            )

    def shuffle_undealt(self, rng=None) -> None:
        """Shuffle the undealt segment in place (positions ``>= _idx``).

        Drawn segment (``< _idx``) is untouched.  Called by
        :meth:`PokerEnv.with_hole_cards` after :meth:`replace_drawn`
        so the next community deal samples uniformly over the
        remaining cards instead of preferring the specific positions
        that received displaced cards from the swap.

        Parameters
        ----------
        rng : numpy.random.Generator, optional
            Stream to shuffle from.  **Every off-game caller should pass
            one.**  Only the played hand's own deal belongs on the global
            stream; a hypothetical re-deal (a search leaf rollout, an AIVAT
            value evaluation) that draws from the global RNG makes its own
            randomness depend on how much *other* work consumed that stream —
            see :mod:`poker_ai.search.rng`.  ``None`` keeps the global
            ``np.random`` draw, matching the construction-time shuffle.
        """
        segment = self._cards[self._idx:]
        if rng is None:
            np.random.shuffle(segment)
        else:
            rng.shuffle(segment)
