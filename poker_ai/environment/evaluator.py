"""Poker hand evaluator using bit arithmetic and prime-product lookup tables.

Evaluates 5-, 6-, and 7-card poker hands and maps them to a rank in the
range [1, 7462], where 1 is the best possible hand (royal flush) and 7462
is the worst (7-high).
"""

import itertools

from poker_ai.environment.utils import prime_product_from_hand, prime_product_from_rankbits
from poker_ai.environment.lookup import LookupTable


class Evaluator(object):
    """Poker hand strength evaluator backed by prime-product lookup tables.

    All evaluation is performed with bit arithmetic and dictionary lookups,
    making per-hand evaluation very fast. Supports 5-, 6-, and 7-card hands.

    Attributes
    ----------
    table : LookupTable
        Precomputed flush and unsuited lookup dictionaries.
    hand_size_map : dict
        Mapping from total card count (5, 6, or 7) to the corresponding
        evaluation method.
    """

    def __init__(self):
        """Initialise the evaluator by building the lookup table."""
        self.table = LookupTable()
        self.hand_size_map = {5: self._five, 6: self._six, 7: self._seven}

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
        minimum = LookupTable.MAX_HIGH_CARD

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
        minimum = LookupTable.MAX_HIGH_CARD

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
            9 is high card. See ``LookupTable.RANK_CLASS_TO_STRING`` for the
            full mapping.

        Raises
        ------
        ValueError
            If ``hr`` is outside the valid range [0, 7462].
        """
        if hr >= 0 and hr <= LookupTable.MAX_STRAIGHT_FLUSH:
            c = LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_STRAIGHT_FLUSH]
        elif hr <= LookupTable.MAX_FOUR_OF_A_KIND:
            c = LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FOUR_OF_A_KIND]
        elif hr <= LookupTable.MAX_FULL_HOUSE:
            c = LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FULL_HOUSE]
        elif hr <= LookupTable.MAX_FLUSH:
            c = LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FLUSH]
        elif hr <= LookupTable.MAX_STRAIGHT:
            c = LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_STRAIGHT]
        elif hr <= LookupTable.MAX_THREE_OF_A_KIND:
            c = LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_THREE_OF_A_KIND]
        elif hr <= LookupTable.MAX_TWO_PAIR:
            c = LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_TWO_PAIR]
        elif hr <= LookupTable.MAX_PAIR:
            c = LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_PAIR]
        elif hr <= LookupTable.MAX_HIGH_CARD:
            c = LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_HIGH_CARD]
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
        return LookupTable.RANK_CLASS_TO_STRING[class_int]

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
        return float(hand_rank) / float(LookupTable.MAX_HIGH_CARD)

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
