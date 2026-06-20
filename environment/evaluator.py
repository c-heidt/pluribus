"""Poker hand evaluator using bit arithmetic and prime-product lookup tables.

Evaluates 5-, 6-, and 7-card poker hands and maps them to a rank in the
range [1, 7462], where 1 is the best possible hand (royal flush) and 7462
is the worst (7-high).
"""

import itertools

import numpy as np

from environment.utils import prime_product_from_hand, prime_product_from_rankbits
from environment.hand_rank_table import HandRankTable


class Evaluator(object):
    """Poker hand strength evaluator backed by prime-product lookup tables.

    All evaluation is performed with bit arithmetic and dictionary lookups,
    making per-hand evaluation very fast. Supports 5-, 6-, and 7-card hands.

    Attributes
    ----------
    table : HandRankTable
        Precomputed flush and unsuited lookup dictionaries.
    hand_size_map : dict
        Mapping from total card count (5, 6, or 7) to the corresponding
        evaluation method.
    """

    def __init__(self):
        """Initialise the evaluator by building the lookup table.

        Besides the scalar ``HandRankTable`` dictionaries, this builds dense
        numpy lookup artifacts used by the vectorised :meth:`evaluate_batch`
        path.  They are **derived from** the same ``HandRankTable`` (the single
        source of truth), so the batch and scalar evaluators are guaranteed
        consistent — proven exhaustively in the tests.
        """
        self.table = HandRankTable()
        self.hand_size_map = {5: self._five, 6: self._six, 7: self._seven}
        self._build_vectorised_tables()

    # ------------------------------------------------------------------
    # Vectorised (batch) evaluation
    # ------------------------------------------------------------------

    def _build_vectorised_tables(self) -> None:
        """Derive the dense numpy lookup tables from :class:`HandRankTable`."""
        flush_lookup = self.table.flush_lookup
        unsuited_lookup = self.table.unsuited_lookup

        # Dense flush table: 13-bit rank-OR pattern -> flush/straight-flush rank.
        # Only patterns with exactly five set bits are real 5-card flushes; the
        # rest stay 0 (a sentinel never indexed for a valid flush).
        self._flush_rank = np.zeros(1 << 13, dtype=np.int16)
        for rankbits in range(1 << 13):
            if bin(rankbits).count("1") == 5:
                rank = flush_lookup.get(prime_product_from_rankbits(rankbits))
                if rank is not None:
                    self._flush_rank[rankbits] = rank

        # Non-flush hands: prime-product -> rank, sorted for vectorised
        # searchsorted.  dict key/value iteration order is consistent (Py3.7+).
        keys = np.fromiter(unsuited_lookup.keys(), dtype=np.int64)
        vals = np.fromiter(unsuited_lookup.values(), dtype=np.int16)
        order = np.argsort(keys)
        self._unsuited_keys = keys[order]
        self._unsuited_ranks = vals[order]

        # Fixed 5-card subset index tables for the best-of-K reduction.
        self._subsets = {
            k: np.array(list(itertools.combinations(range(k), 5)), dtype=np.intp)
            for k in (5, 6, 7)
        }

    def _eval5_vec(self, cards5: np.ndarray) -> np.ndarray:
        """Rank a batch of exactly-5-card hands.

        Parameters
        ----------
        cards5 : numpy.ndarray
            ``(M, 5)`` array of card integers.

        Returns
        -------
        numpy.ndarray
            ``(M,)`` int16 ranks in [1, 7462] (lower = stronger), matching
            :meth:`_five` element-for-element.
        """
        suits = (cards5 >> 12) & 0xF
        is_flush = np.bitwise_and.reduce(suits, axis=1) != 0
        out = np.empty(cards5.shape[0], dtype=np.int16)

        if is_flush.any():
            rankbits = np.bitwise_or.reduce(
                (cards5[is_flush] >> 16) & 0x1FFF, axis=1
            )
            out[is_flush] = self._flush_rank[rankbits]

        nf = ~is_flush
        if nf.any():
            products = np.prod((cards5[nf] & 0xFF).astype(np.int64), axis=1)
            idx = np.searchsorted(self._unsuited_keys, products)
            out[nf] = self._unsuited_ranks[idx]
        return out

    def evaluate_batch(self, cards: np.ndarray) -> np.ndarray:
        """Rank a batch of hands of a fixed card count.

        The vectorised counterpart of :meth:`evaluate` — same ranks, evaluated
        for many hands at once.  Use it on hot many-hands paths (range showdown
        ranking, runout completions); use the scalar :meth:`evaluate` for single
        hands.

        Parameters
        ----------
        cards : numpy.ndarray
            ``(N, K)`` array of card integers, ``K in {5, 6, 7}`` (every row has
            the same card count).

        Returns
        -------
        numpy.ndarray
            ``(N,)`` int64 ranks in [1, 7462]; entry ``i`` is the best 5-card
            hand reachable from row ``i``.

        Raises
        ------
        ValueError
            If ``cards`` is not 2-D or ``K`` is unsupported.
        """
        cards = np.asarray(cards)
        if cards.ndim != 2:
            raise ValueError(f"evaluate_batch expects a 2-D array, got {cards.ndim}-D")
        n, k = cards.shape
        subsets = self._subsets.get(k)
        if subsets is None:
            raise ValueError(f"evaluate_batch supports K in {{5, 6, 7}}, got K={k}")

        out = np.empty(n, dtype=np.int64)
        if n == 0:
            return out
        n_subsets = subsets.shape[0]
        # Chunk over rows so peak memory stays bounded for large callers.
        chunk = 1 << 15
        for start in range(0, n, chunk):
            block = cards[start : start + chunk]
            b = block.shape[0]
            sub = block[:, subsets]  # (b, n_subsets, 5)
            ranks5 = self._eval5_vec(sub.reshape(b * n_subsets, 5))
            out[start : start + b] = ranks5.reshape(b, n_subsets).min(axis=1)
        return out

    def evaluate(self, cards, board):
        """Return the rank of the best 5-card hand reachable from the given cards.

        Combines ``cards`` and ``board``, converts each to its integer
        representation, then dispatches to the appropriate evaluator based on
        the total number of cards.

        Parameters
        ----------
        cards : list
            Player's hole cards. Each element must be convertible to ``int``
            via the card integer encoding.
        board : list
            Community cards currently in play. May be empty.

        Returns
        -------
        int
            Hand rank in the range [1, 7462]. Lower values are stronger hands.
        """
        all_cards = [int(c) for c in cards + board]
        return self.hand_size_map[len(all_cards)](all_cards)

    def _five(self, cards):
        """Evaluate exactly 5 cards given as a list of card integers.

        Detects flushes via the suit bits and routes to the appropriate lookup
        table. For flushes, the rank-bit OR is used to compute the prime
        product; for non-flushes the product of the rank primes is used
        directly.

        Parameters
        ----------
        cards : list of int
            Exactly 5 card integers in the internal bit encoding.

        Returns
        -------
        int
            Hand rank in the range [1, 7462]. Lower values are stronger hands.
        """
        # if flush
        if cards[0] & cards[1] & cards[2] & cards[3] & cards[4] & 0xF000:
            handOR = (cards[0] | cards[1] | cards[2] | cards[3] | cards[4]) >> 16
            prime = prime_product_from_rankbits(handOR)
            return self.table.flush_lookup[prime]

        # otherwise
        else:
            prime = prime_product_from_hand(cards)
            return self.table.unsuited_lookup[prime]

    def _six(self, cards):
        """Evaluate a 6-card hand by finding the best 5-card subset.

        Iterates over all C(6, 5) = 6 combinations and returns the minimum
        (strongest) rank found.

        Parameters
        ----------
        cards : list of int
            Exactly 6 card integers in the internal bit encoding.

        Returns
        -------
        int
            Best hand rank in the range [1, 7462]. Lower values are stronger.
        """
        minimum = HandRankTable.MAX_HIGH_CARD

        all5cardcombobs = itertools.combinations(cards, 5)
        for combo in all5cardcombobs:

            score = self._five(combo)
            if score < minimum:
                minimum = score

        return minimum

    def _seven(self, cards):
        """Evaluate a 7-card hand by finding the best 5-card subset.

        Iterates over all C(7, 5) = 21 combinations and returns the minimum
        (strongest) rank found.

        Parameters
        ----------
        cards : list of int
            Exactly 7 card integers in the internal bit encoding.

        Returns
        -------
        int
            Best hand rank in the range [1, 7462]. Lower values are stronger.
        """
        minimum = HandRankTable.MAX_HIGH_CARD

        all5cardcombobs = itertools.combinations(cards, 5)
        for combo in all5cardcombobs:

            score = self._five(combo)
            if score < minimum:
                minimum = score

        return minimum

    def get_rank_class(self, hr):
        """Map a numeric hand rank to its hand-class integer.

        Parameters
        ----------
        hr : int
            Hand rank as returned by :meth:`evaluate`, in the range [1, 7462].

        Returns
        -------
        int
            Hand class in the range [1, 9], where 1 is straight flush and
            9 is high card. See ``HandRankTable.RANK_CLASS_TO_STRING`` for the
            full mapping.

        Raises
        ------
        ValueError
            If ``hr`` is outside the valid range [0, 7462].
        """
        if hr >= 0 and hr <= HandRankTable.MAX_STRAIGHT_FLUSH:
            c = HandRankTable.MAX_TO_RANK_CLASS[HandRankTable.MAX_STRAIGHT_FLUSH]
        elif hr <= HandRankTable.MAX_FOUR_OF_A_KIND:
            c = HandRankTable.MAX_TO_RANK_CLASS[HandRankTable.MAX_FOUR_OF_A_KIND]
        elif hr <= HandRankTable.MAX_FULL_HOUSE:
            c = HandRankTable.MAX_TO_RANK_CLASS[HandRankTable.MAX_FULL_HOUSE]
        elif hr <= HandRankTable.MAX_FLUSH:
            c = HandRankTable.MAX_TO_RANK_CLASS[HandRankTable.MAX_FLUSH]
        elif hr <= HandRankTable.MAX_STRAIGHT:
            c = HandRankTable.MAX_TO_RANK_CLASS[HandRankTable.MAX_STRAIGHT]
        elif hr <= HandRankTable.MAX_THREE_OF_A_KIND:
            c = HandRankTable.MAX_TO_RANK_CLASS[HandRankTable.MAX_THREE_OF_A_KIND]
        elif hr <= HandRankTable.MAX_TWO_PAIR:
            c = HandRankTable.MAX_TO_RANK_CLASS[HandRankTable.MAX_TWO_PAIR]
        elif hr <= HandRankTable.MAX_PAIR:
            c = HandRankTable.MAX_TO_RANK_CLASS[HandRankTable.MAX_PAIR]
        elif hr <= HandRankTable.MAX_HIGH_CARD:
            c = HandRankTable.MAX_TO_RANK_CLASS[HandRankTable.MAX_HIGH_CARD]
        else:
            raise ValueError("Invalid hand rank, cannot return rank class")
        return c

    def class_to_string(self, class_int):
        """Convert a hand-class integer to a human-readable hand name.

        Parameters
        ----------
        class_int : int
            Hand class in the range [1, 9] as returned by
            :meth:`get_rank_class`.

        Returns
        -------
        str
            Human-readable hand name, e.g. ``"Straight Flush"`` or
            ``"Two Pair"``.
        """
        return HandRankTable.RANK_CLASS_TO_STRING[class_int]

    def get_five_card_rank_percentage(self, hand_rank):
        """Normalise a hand rank to the [0.0, 1.0] range.

        A value of 0.0 corresponds to the best possible hand (rank 1) and
        1.0 corresponds to the worst (rank 7462).

        Parameters
        ----------
        hand_rank : int
            Hand rank in the range [1, 7462] as returned by :meth:`evaluate`.

        Returns
        -------
        float
            Normalised rank between 0.0 (best) and 1.0 (worst).
        """
        return float(hand_rank) / float(HandRankTable.MAX_HIGH_CARD)

    def hand_summary(self, board, hands):
        """Print a street-by-street summary of hand strengths and winners.

        Evaluates each player's hand at the flop, turn, and river, printing
        the hand class, percentile rank, and current leader at each stage.
        Declares the overall winner at the river.

        Parameters
        ----------
        board : list
            Exactly 5 community cards in chronological order (flop cards
            first, then turn, then river).
        hands : list of list
            Each element is a 2-card list representing one player's hole
            cards.

        Raises
        ------
        AssertionError
            If ``board`` does not contain exactly 5 cards, or if any element
            of ``hands`` does not contain exactly 2 cards.
        """

        assert len(board) == 5, "Invalid board length"
        for hand in hands:
            assert len(hand) == 2, "Invalid hand length"

        line_length = 10
        stages = ["FLOP", "TURN", "RIVER"]

        for i in range(len(stages)):
            line = "=" * line_length
            print(f"{line} {stages[i]} {line}")

            best_rank = 7463  # rank one worse than worst hand
            winners = []
            for player, hand in enumerate(hands):

                # evaluate current board position
                rank = self.evaluate(hand, board[: (i + 3)])
                rank_class = self.get_rank_class(rank)
                class_string = self.class_to_string(rank_class)
                percentage = 1.0 - self.get_five_card_rank_percentage(
                    rank
                )  # higher better here
                print(
                    f"Player {player + 1} hand = {class_string}, percentage rank among all hands = {percentage}"
                )

                # detect winner
                if rank == best_rank:
                    winners.append(player)
                    best_rank = rank
                elif rank < best_rank:
                    winners = [player]
                    best_rank = rank

            # if we're not on the river
            if i != stages.index("RIVER"):
                if len(winners) == 1:
                    print(f"Player {winners[0] + 1} hand is currently winning.\n")
                else:
                    print(
                        f"Players {[x + 1 for x in winners]} are tied for the lead.\n"
                    )

            # otherwise on all other streets
            else:
                hand_result = self.class_to_string(
                    self.get_rank_class(self.evaluate(hands[winners[0]], board))
                )
                print()
                print(f"{line} HAND OVER {line}")
                if len(winners) == 1:
                    print(
                        f"Player {winners[0] + 1} is the winner with a {hand_result}\n"
                    )
                else:
                    print(f"Players {winners} tied for the win with a {hand_result}\n")
