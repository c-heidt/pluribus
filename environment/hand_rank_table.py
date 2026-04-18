"""Prime-product lookup tables for 5-card poker hand ranking.

Maps every distinct 5-card hand, represented as the product of its card
prime values, to a rank in the range [1, 7462]. Rank 1 is the best
possible hand (royal flush) and rank 7462 is the worst (7-high).

Two separate tables are maintained: one for flush hands (looked up via the
OR of rank-bit patterns) and one for all unsuited hands (looked up via the
product of rank primes).

Distinct hand counts
--------------------
Straight Flush    10
Four of a Kind   156   = C(13,2) * C(2,1)
Full House       156   = C(13,2) * C(2,1)
Flush           1277   = C(13,5) - 10
Straight          10
Three of a Kind  858   = C(13,3) * C(3,1)
Two Pair         858   = C(13,3) * C(3,2)
One Pair        2860   = C(13,4) * C(4,1)
High Card       1277   = C(13,5) - 10
                ----
Total           7462
"""

import itertools

from environment.utils import INT_RANKS, CARD_PRIMES, prime_product_from_rankbits


class HandRankTable(object):
    """Lookup tables mapping prime products of 5-card hands to hand ranks.

    Ranks are integers in [1, 7462] where lower is stronger. Two
    dictionaries are populated at construction time:

    * ``flush_lookup`` — keys are prime products derived from rank-bit
      OR patterns; covers straight flushes and flushes.
    * ``unsuited_lookup`` — keys are prime products of card rank primes;
      covers everything else (straights, high cards, and all multiples).

    Class-level constants define the upper bound of each hand category
    within the ranking range, and two class-level dicts allow conversion
    between rank values, category integers, and human-readable strings.

    Attributes
    ----------
    flush_lookup : dict
        Maps prime product (int) to rank (int) for flush-type hands.
    unsuited_lookup : dict
        Maps prime product (int) to rank (int) for non-flush hands.

    Class Attributes
    ----------------
    MAX_STRAIGHT_FLUSH : int
        Highest rank assigned to a straight flush (10).
    MAX_FOUR_OF_A_KIND : int
        Highest rank assigned to four of a kind (166).
    MAX_FULL_HOUSE : int
        Highest rank assigned to a full house (322).
    MAX_FLUSH : int
        Highest rank assigned to a flush (1599).
    MAX_STRAIGHT : int
        Highest rank assigned to a straight (1609).
    MAX_THREE_OF_A_KIND : int
        Highest rank assigned to three of a kind (2467).
    MAX_TWO_PAIR : int
        Highest rank assigned to two pair (3325).
    MAX_PAIR : int
        Highest rank assigned to one pair (6185).
    MAX_HIGH_CARD : int
        Highest rank assigned to a high-card hand (7462).
    MAX_TO_RANK_CLASS : dict
        Maps each ``MAX_*`` constant to a hand-class integer in [1, 9].
    RANK_CLASS_TO_STRING : dict
        Maps hand-class integers in [1, 9] to human-readable hand names.
    """

    MAX_STRAIGHT_FLUSH = 10
    MAX_FOUR_OF_A_KIND = 166
    MAX_FULL_HOUSE = 322
    MAX_FLUSH = 1599
    MAX_STRAIGHT = 1609
    MAX_THREE_OF_A_KIND = 2467
    MAX_TWO_PAIR = 3325
    MAX_PAIR = 6185
    MAX_HIGH_CARD = 7462

    MAX_TO_RANK_CLASS = {
        MAX_STRAIGHT_FLUSH: 1,
        MAX_FOUR_OF_A_KIND: 2,
        MAX_FULL_HOUSE: 3,
        MAX_FLUSH: 4,
        MAX_STRAIGHT: 5,
        MAX_THREE_OF_A_KIND: 6,
        MAX_TWO_PAIR: 7,
        MAX_PAIR: 8,
        MAX_HIGH_CARD: 9,
    }

    RANK_CLASS_TO_STRING = {
        1: "Straight Flush",
        2: "Four of a Kind",
        3: "Full House",
        4: "Flush",
        5: "Straight",
        6: "Three of a Kind",
        7: "Two Pair",
        8: "Pair",
        9: "High Card",
    }

    def __init__(self):
        """Build ``flush_lookup`` and ``unsuited_lookup`` at construction time.

        Calls :meth:`flushes` (which also populates straights and high cards
        via :meth:`straight_and_highcards`) and then :meth:`multiples`.
        """
        # create dictionaries
        self.flush_lookup = {}
        self.unsuited_lookup = {}

        # create the lookup table in piecewise fashion
        # this will call straights and high cards method,
        # we reuse some of the bit sequences
        self.flushes()
        self.multiples()

    def flushes(self):
        """Populate ``flush_lookup`` for straight flushes and flushes.

        Generates all 13-bit rank patterns for flushes using
        :meth:`get_lexographically_next_bit_sequence`, filters out patterns
        that match a straight flush, reverses the remaining list so the
        strongest patterns rank lowest, then delegates to
        :meth:`_fill_in_lookup_table` for both categories. Also calls
        :meth:`straight_and_highcards` to reuse the same bit sequences for
        the unsuited straight and high-card entries.

        Lookup is performed on a 13-bit integer where each bit represents
        a rank::

            xxxbbbbb bbbbbbbb  =>  integer rank index
        """

        # straight flushes in rank order
        straight_flushes = [
            7936,  # int('0b1111100000000', 2), # royal flush
            3968,  # int('0b111110000000', 2),
            1984,  # int('0b11111000000', 2),
            992,  # int('0b1111100000', 2),
            496,  # int('0b111110000', 2),
            248,  # int('0b11111000', 2),
            124,  # int('0b1111100', 2),
            62,  # int('0b111110', 2),
            31,  # int('0b11111', 2),
            4111,  # int('0b1000000001111', 2) # 5 high
        ]

        # now we'll dynamically generate all the other
        # flushes (including straight flushes)
        flushes = []
        gen = self.get_lexographically_next_bit_sequence(int("0b11111", 2))

        # 1277 = number of high cards
        # 1277 + len(str_flushes) is number of hands with all cards unique rank
        for i in range(1277 + len(straight_flushes) - 1):
            # we also iterate over SFs
            # pull the next flush pattern from our generator
            f = next(gen)

            # if this flush matches perfectly any
            # straight flush, do not add it
            notSF = True
            for sf in straight_flushes:
                # if f XOR sf == 0, then bit pattern
                # is same, and we should not add
                if not f ^ sf:
                    notSF = False

            if notSF:
                flushes.append(f)

        # we started from the lowest straight pattern, now we want to start
        # ranking from the most powerful hands, so we reverse
        flushes.reverse()
        # now add to the lookup map:
        # start with straight flushes and the rank of 1
        # since it is the best hand in poker
        # rank 1 = Royal Flush!
        self._fill_in_lookup_table(
            rank_init=1,
            rankbits_list=straight_flushes,
            lookup_table=self.flush_lookup)
        # we start the counting for flushes on max full house, which
        # is the worst rank that a full house can have (2,2,2,3,3)
        self._fill_in_lookup_table(
            rank_init=HandRankTable.MAX_FULL_HOUSE + 1,
            rankbits_list=flushes,
            lookup_table=self.flush_lookup)
        # we can reuse these bit sequences for straights
        # and high cards since they are inherently related
        # and differ only by context
        self.straight_and_highcards(straight_flushes, flushes)

    def _fill_in_lookup_table(self, rank_init, rankbits_list, lookup_table):
        """Insert prime-product-to-rank entries into a lookup dictionary.

        For each rank-bit pattern in ``rankbits_list``, computes the
        corresponding prime product and writes the mapping
        ``prime_product -> rank`` into ``lookup_table``. Ranks are assigned
        consecutively starting from ``rank_init``.

        Parameters
        ----------
        rank_init : int
            The rank value assigned to the first entry in ``rankbits_list``.
        rankbits_list : list of int
            Ordered sequence of 13-bit rank patterns, from strongest to
            weakest hand.
        lookup_table : dict
            The dictionary to populate in place.
        """
        rank = rank_init
        for rb in rankbits_list:
            prime_product = prime_product_from_rankbits(rb)
            lookup_table[prime_product] = rank
            rank += 1

    def straight_and_highcards(self, straights, highcards):
        """Populate ``unsuited_lookup`` for straights and high-card hands.

        Reuses the 13-bit rank-bit patterns generated during flush
        construction to fill in the unsuited entries for straights
        (ranks ``MAX_FLUSH + 1`` through ``MAX_STRAIGHT``) and high cards
        (ranks ``MAX_PAIR + 1`` through ``MAX_HIGH_CARD``).

        Parameters
        ----------
        straights : list of int
            Ordered 13-bit rank patterns for the 10 straight hands,
            from strongest to weakest.
        highcards : list of int
            Ordered 13-bit rank patterns for the 1277 high-card hands,
            from strongest to weakest.
        """
        self._fill_in_lookup_table(
            rank_init=HandRankTable.MAX_FLUSH + 1,
            rankbits_list=straights,
            lookup_table=self.unsuited_lookup)
        self._fill_in_lookup_table(
            rank_init=HandRankTable.MAX_PAIR + 1,
            rankbits_list=highcards,
            lookup_table=self.unsuited_lookup)

    def multiples(self):
        """Populate ``unsuited_lookup`` for all multiple-card hand types.

        Fills in entries for four of a kind, full house, three of a kind,
        two pair, and one pair in that order, using products of card rank
        primes raised to the appropriate powers. Ranks are assigned within
        the boundaries defined by the ``MAX_*`` class constants.
        """
        backwards_ranks = list(range(len(INT_RANKS) - 1, -1, -1))

        # 1) Four of a Kind
        rank = HandRankTable.MAX_STRAIGHT_FLUSH + 1

        # for each choice of a set of four rank
        for i in backwards_ranks:

            # and for each possible kicker rank
            kickers = backwards_ranks[:]
            kickers.remove(i)
            for k in kickers:
                product = CARD_PRIMES[i] ** 4 * CARD_PRIMES[k]
                self.unsuited_lookup[product] = rank
                rank += 1

        # 2) Full House
        rank = HandRankTable.MAX_FOUR_OF_A_KIND + 1

        # for each three of a kind
        for i in backwards_ranks:

            # and for each choice of pair rank
            pairranks = backwards_ranks[:]
            pairranks.remove(i)
            for pr in pairranks:
                product = CARD_PRIMES[i] ** 3 * CARD_PRIMES[pr] ** 2
                self.unsuited_lookup[product] = rank
                rank += 1

        # 3) Three of a Kind
        rank = HandRankTable.MAX_STRAIGHT + 1

        # pick three of one rank
        for r in backwards_ranks:

            kickers = backwards_ranks[:]
            kickers.remove(r)
            gen = itertools.combinations(kickers, 2)

            for kickers in gen:

                c1, c2 = kickers
                product = CARD_PRIMES[r] ** 3 * CARD_PRIMES[c1] * CARD_PRIMES[c2]
                self.unsuited_lookup[product] = rank
                rank += 1

        # 4) Two Pair
        rank = HandRankTable.MAX_THREE_OF_A_KIND + 1

        tpgen = itertools.combinations(backwards_ranks, 2)
        for tp in tpgen:

            pair1, pair2 = tp
            kickers = backwards_ranks[:]
            kickers.remove(pair1)
            kickers.remove(pair2)
            for kicker in kickers:

                product = (
                    CARD_PRIMES[pair1] ** 2
                    * CARD_PRIMES[pair2] ** 2
                    * CARD_PRIMES[kicker]
                )
                self.unsuited_lookup[product] = rank
                rank += 1

        # 5) Pair
        rank = HandRankTable.MAX_TWO_PAIR + 1

        # choose a pair
        for pairrank in backwards_ranks:

            kickers = backwards_ranks[:]
            kickers.remove(pairrank)
            kgen = itertools.combinations(kickers, 3)

            for kickers in kgen:

                k1, k2, k3 = kickers
                product = (
                    CARD_PRIMES[pairrank] ** 2
                    * CARD_PRIMES[k1]
                    * CARD_PRIMES[k2]
                    * CARD_PRIMES[k3]
                )
                self.unsuited_lookup[product] = rank
                rank += 1

    def write_table_to_disk(self, table, filepath):
        """Write a lookup table to a CSV file on disk.

        Each line contains one ``prime_product,rank`` pair.

        Parameters
        ----------
        table : dict
            Lookup dictionary mapping prime products (int) to ranks (int).
        filepath : str
            Destination file path. The file is created or overwritten.
        """
        with open(filepath, "w") as f:
            for prime_prod, rank in table.iteritems():
                f.write(str(prime_prod) + "," + str(rank) + "\n")

    def get_lexographically_next_bit_sequence(self, bits):
        """Generate lexicographically successive integers with the same popcount.

        Uses the bit-manipulation identity known as Gosper's hack to yield
        each successive integer that has the same number of set bits as
        ``bits``, in ascending numeric order. Because 5-bit patterns are
        enumerated in ascending order they correspond to weakest-to-strongest
        flush/high-card hands, so no post-sort is required.

        Parameters
        ----------
        bits : int
            Starting bit pattern. Must have at least one set bit.

        Yields
        ------
        int
            Next integer with the same number of set bits as ``bits``,
            in lexicographically (numerically) ascending order.
        """
        t = int((bits | (bits - 1))) + 1
        next = t | ((int(((t & -t) / (bits & -bits))) >> 1) - 1)
        yield next
        while True:
            t = (next | (next - 1)) + 1
            next = t | ((((t & -t) // (next & -next)) >> 1) - 1)
            yield next
