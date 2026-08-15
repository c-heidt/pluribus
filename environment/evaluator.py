"""Poker hand evaluator using bit arithmetic and prime-product lookup tables.

Evaluates 5-, 6-, and 7-card poker hands and maps them to a rank in the
range [1, 7462], where 1 is the best possible hand (royal flush) and 7462
is the worst (7-high).
"""

import itertools

import numpy as np

from environment.utils import (
    CARD_PRIMES,
    prime_product_from_hand,
    prime_product_from_rankbits,
)
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
        self._build_multicard_tables()

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

    def _build_multicard_tables(self) -> None:
        """Build exact O(1) lookup tables for 6- and 7-card scalar evaluation.

        Derived entirely from :class:`HandRankTable` (the single source of
        truth), so results are byte-identical to the 21-subset enumeration the
        tables replace — proven exhaustively in
        ``test_evaluator_multicard.py``.  Two correctness theorems make the
        replacement exact:

        * **Flush precludes quads/full-house** (6 and 7 cards): a 5+-card flush
          leaves <=2 off-suit cards, too few to also form quads (needs >=3
          off-suit) or a full house (needs >=3 off-suit), and at most one suit
          can hold >=5 cards.  So when a flush exists the best hand is a
          straight flush or plain flush, fully determined by the flush suit's
          13-bit rank mask — hence ``_flush_best``.
        * **Non-flush prime product is a collision-free perfect hash**: the
          product of rank primes (distinct primes 2..41) is injective by unique
          factorisation, and in the non-flush branch no 5-card subset is a
          flush, so the best-5 rank is a pure function of the rank multiset —
          hence ``_nonflush7`` / ``_nonflush6`` keyed by that product.
        """
        flush_lookup = self.table.flush_lookup
        unsuited_lookup = self.table.unsuited_lookup
        primes = CARD_PRIMES
        max_sf = HandRankTable.MAX_STRAIGHT_FLUSH
        max_flush = HandRankTable.MAX_FLUSH
        max_high = HandRankTable.MAX_HIGH_CARD

        # --- _flush_best: 13-bit suit rank mask (popcount >= 5) -> best flush /
        # straight-flush rank = min over all 5-bit submasks.  Dense int16 array
        # (16 KB) for O(1) direct indexing; masks with <5 bits stay 0 (never
        # queried on the flush branch).
        self._flush_best = np.zeros(1 << 13, dtype=np.int16)
        for mask in range(1 << 13):
            bits = [i for i in range(13) if mask & (1 << i)]
            if len(bits) < 5:
                continue
            best = max_high
            for sub in itertools.combinations(bits, 5):
                sub_prime = prime_product_from_rankbits(
                    sum(1 << i for i in sub)
                )
                rank = flush_lookup[sub_prime]  # KeyError if encoding drifts
                if rank < best:
                    best = rank
            assert 1 <= best <= max_flush
            self._flush_best[mask] = best

        # --- _nonflush{7,6}: product of all rank primes -> best non-flush rank
        # = min over 5-sub-multisets of unsuited_lookup.  Built in rank space
        # (suit-agnostic), which is exactly what the runtime product encodes.
        def build_nonflush(n: int) -> "dict":
            table = {}
            for multiset in itertools.combinations_with_replacement(range(13), n):
                counts = [0] * 13
                for r in multiset:
                    counts[r] += 1
                if any(c > 4 for c in counts):  # impossible with a 52-card deck
                    continue
                best = max_high
                for sub in itertools.combinations(multiset, 5):
                    p5 = 1
                    for r in sub:
                        p5 *= primes[r]
                    rank = unsuited_lookup[p5]  # KeyError if encoding drifts
                    if rank < best:
                        best = rank
                product = 1
                for r in multiset:
                    product *= primes[r]
                table[product] = best
            return table

        self._nonflush7 = build_nonflush(7)
        self._nonflush6 = build_nonflush(6)

        # Perfect-hash key-space sizes: distinct products == distinct multisets
        # (empirical injectivity proof); a collision would shrink these.
        assert len(self._nonflush7) == 49205, len(self._nonflush7)
        assert len(self._nonflush6) == 18395, len(self._nonflush6)
        # Non-flush hands can never be a straight flush, and are bounded by the
        # worst high card.
        assert min(self._nonflush7.values()) > max_sf
        assert max(self._nonflush7.values()) <= max_high
        assert min(self._nonflush6.values()) > max_sf
        assert max(self._nonflush6.values()) <= max_high

        # Sorted (product -> rank) arrays mirroring the scalar dicts, so the
        # vectorised evaluator (:meth:`_multicard_vec`) can resolve the
        # non-flush branch with one ``searchsorted`` instead of per-row dict
        # lookups.  Same key space as the dicts (the size asserts above pin it).
        def sorted_arrays(table: "dict"):
            k = np.fromiter(table.keys(), dtype=np.int64)
            v = np.fromiter(table.values(), dtype=np.int16)
            order = np.argsort(k)
            return k[order], v[order]

        self._nonflush7_keys, self._nonflush7_ranks = sorted_arrays(self._nonflush7)
        self._nonflush6_keys, self._nonflush6_ranks = sorted_arrays(self._nonflush6)

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

    def _multicard_vec(self, cards: np.ndarray, k: int) -> np.ndarray:
        """Vectorised O(1)-per-hand rank of a block of exactly-``k``-card hands.

        The batch counterpart of :meth:`_seven` / :meth:`_six`: it uses the same
        exact lookup tables (:attr:`_flush_best`, :attr:`_nonflush7` /
        :attr:`_nonflush6`), so it is byte-identical to the 21-subset
        :meth:`_evaluate_batch_oracle` — proven exhaustively over all C(52,7)
        and C(52,6) hands in ``test_evaluator_multicard.py``.

        Per row: count cards per suit and OR their rank bits; a suit with >=5
        cards is a flush (at most one can be, and by the flush-precludes-
        quads/full-house theorem it decides the hand) resolved via
        ``_flush_best[mask]``; otherwise the rank-prime product keys the
        non-flush table via a single ``searchsorted``.  ``k == 5`` has no
        subsets to reduce, so it is exactly :meth:`_eval5_vec`.

        Parameters
        ----------
        cards : numpy.ndarray
            ``(b, k)`` array of card integers (one block; caller chunks).
        k : int
            Card count, one of ``{5, 6, 7}``.

        Returns
        -------
        numpy.ndarray
            ``(b,)`` int64 ranks in [1, 7462] (lower = stronger).
        """
        if k == 5:
            return self._eval5_vec(cards).astype(np.int64)

        suit = (cards >> 12) & 0xF
        rankbits = (cards >> 16) & 0x1FFF
        b = cards.shape[0]
        out = np.empty(b, dtype=np.int64)
        is_flush = np.zeros(b, dtype=bool)
        flush_mask = np.zeros(b, dtype=np.int64)
        # At most one suit can hold >=5 of 6/7 cards, so these branches are
        # mutually exclusive across rows — no row is written twice.
        for sv in (1, 2, 4, 8):
            sel = suit == sv
            take = sel.sum(axis=1) >= 5
            if take.any():
                m = np.bitwise_or.reduce(np.where(sel, rankbits, 0), axis=1)
                flush_mask[take] = m[take]
                is_flush |= take
        if is_flush.any():
            out[is_flush] = self._flush_best[flush_mask[is_flush]]

        nf = ~is_flush
        if nf.any():
            keys = self._nonflush7_keys if k == 7 else self._nonflush6_keys
            ranks = self._nonflush7_ranks if k == 7 else self._nonflush6_ranks
            products = np.prod((cards[nf] & 0xFF).astype(np.int64), axis=1)
            idx = np.searchsorted(keys, products)
            out[nf] = ranks[idx]
        return out

    def evaluate_batch(self, cards: np.ndarray) -> np.ndarray:
        """Rank a batch of hands of a fixed card count (fast exact table lookup).

        The vectorised counterpart of :meth:`evaluate` — same ranks, evaluated
        for many hands at once.  Use it on hot many-hands paths (range showdown
        ranking, runout completions); use the scalar :meth:`evaluate` for single
        hands.

        Backed by the exact multicard LUT (:meth:`_multicard_vec`), which is the
        batch form of the scalar :meth:`_seven` / :meth:`_six` and shares their
        tables — so concrete (scalar) and batch payouts stay aligned by
        construction.  :meth:`_evaluate_batch_oracle` retains the original
        21-subset enumeration as the independent cross-check the exhaustive
        tests validate this path against.

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
        if k not in (5, 6, 7):
            raise ValueError(f"evaluate_batch supports K in {{5, 6, 7}}, got K={k}")

        out = np.empty(n, dtype=np.int64)
        if n == 0:
            return out
        # Chunk over rows so peak memory stays bounded for large callers.
        chunk = 1 << 15
        for start in range(0, n, chunk):
            block = cards[start : start + chunk]
            out[start : start + block.shape[0]] = self._multicard_vec(block, k)
        return out

    def _evaluate_batch_oracle(self, cards: np.ndarray) -> np.ndarray:
        """Reference batch evaluator: min over all C(K,5) five-card subsets.

        The original :meth:`evaluate_batch` implementation, kept verbatim as the
        **independent oracle**.  It shares no code with the multicard LUT (it
        enumerates every 5-card subset and reduces via :meth:`_eval5_vec`), so
        the exhaustive tests validate the fast :meth:`evaluate_batch` /
        :meth:`_seven` / :meth:`_six` path against a genuinely different
        algorithm.  Not used on any hot path — tests and diagnostics only.
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
        """Evaluate a 6-card hand by exact table lookup (best 5-card hand).

        O(1) replacement for the former C(6, 5) = 6 subset enumeration: a
        single pass computes per-suit counts / rank masks and the rank-prime
        product, then one table lookup gives the best-5 rank.  Byte-identical
        to the enumeration (see :meth:`_build_multicard_tables`).

        Parameters
        ----------
        cards : list of int
            Exactly 6 card integers in the internal bit encoding.

        Returns
        -------
        int
            Best hand rank in the range [1, 7462]. Lower values are stronger.
        """
        counts = [0] * 16
        masks = [0] * 16
        product = 1
        for card in cards:
            s = (card >> 12) & 0xF
            counts[s] += 1
            masks[s] |= card >> 16
            product *= card & 0xFF
        for s in (1, 2, 4, 8):
            if counts[s] >= 5:
                return int(self._flush_best[masks[s] & 0x1FFF])
        return self._nonflush6[product]

    def _seven(self, cards):
        """Evaluate a 7-card hand by exact table lookup (best 5-card hand).

        O(1) replacement for the former C(7, 5) = 21 subset enumeration — the
        dominant per-node cost in CFR training.  A single pass computes
        per-suit counts / rank masks and the rank-prime product; a flush suit
        (>=5 cards) resolves via :attr:`_flush_best`, otherwise the product
        keys :attr:`_nonflush7`.  Byte-identical to the enumeration (see
        :meth:`_build_multicard_tables`).

        Parameters
        ----------
        cards : list of int
            Exactly 7 card integers in the internal bit encoding.

        Returns
        -------
        int
            Best hand rank in the range [1, 7462]. Lower values are stronger.
        """
        counts = [0] * 16
        masks = [0] * 16
        product = 1
        for card in cards:
            s = (card >> 12) & 0xF
            counts[s] += 1
            masks[s] |= card >> 16
            product *= card & 0xFF
        for s in (1, 2, 4, 8):
            if counts[s] >= 5:
                return int(self._flush_best[masks[s] & 0x1FFF])
        return self._nonflush7[product]

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


# The single shared evaluator instance for the whole environment.  Building the
# vectorised lookup tables in ``Evaluator.__init__`` is non-trivial, so every
# terminal-payout path (concrete settlement in :mod:`dynamics`, the range-vs-range
# showdown in :mod:`environment.range_showdown`, and ``PokerEnv``'s batch ranking)
# shares this one instance.  Hosting it here at the leaf evaluator layer — rather
# than in ``dynamics`` — keeps :mod:`environment.range_showdown` dependent on the
# evaluator only (no cycle through ``dynamics``/``poker_env``), and it is what makes
# the concrete and vectorised payouts *aligned by construction*: identical tables,
# identical ranks.
default_evaluator: Evaluator = Evaluator()


# ---------------------------------------------------------------------------
# Optional compiled-core evaluator (Phase 1d: _five/_six/_seven; batch: item 4)
# ---------------------------------------------------------------------------
# When the extension is built AND enabled, route the shared ``default_evaluator``
# through the Cython kernel(s).  Two independently gated pieces, both
# byte-identical drop-ins — they index the evaluator's own dumped tables
# (``_flush_best`` / ``_flush_rank`` and the sorted ``_unsuited`` /
# ``_nonflush{6,7}`` key/rank arrays) — so every consumer of
# ``default_evaluator`` (terminal settlement, range showdown, batch ranking)
# sees identical ranks either way:
#   * ``evaluator`` — scalar ``_five``/``_six``/``_seven``.  ``evaluate`` looks
#     the dispatch up in ``hand_size_map`` at call time, so swapping the map is
#     transparent; the scalar methods stay untouched as the byte-exact oracle.
#   * ``evaluator_batch`` — the numpy-vectorised ``_multicard_vec`` (the
#     ``evaluate_batch`` engine).  ``evaluate_batch`` only ever calls
#     ``self._multicard_vec(block, k)``, so swapping that ONE instance
#     attribute (exactly the ``hand_size_map`` pattern) transparently
#     accelerates every caller of ``evaluate_batch`` — no other call site
#     needs to change.
# Both flags normally move together (``kernel_enabled`` falls back to the
# master ``PLURIBUS_SEARCH_CORE``/``PLURIBUS_CFR_CORE`` switches when no
# ``PLURIBUS_CORE_KERNELS`` dev override is set), but ``configure()`` must run
# whenever EITHER is on — a dev enabling ``evaluator_batch`` alone via the
# fine-grained override must not skip table installation.
try:
    from poker_ai._core import CORE_AVAILABLE as _CORE_AVAILABLE
    from poker_ai._core.flags import kernel_enabled as _kernel_enabled

    _use_scalar = _CORE_AVAILABLE and _kernel_enabled("evaluator")
    _use_batch = _CORE_AVAILABLE and _kernel_enabled("evaluator_batch")

    if _use_scalar or _use_batch:
        from poker_ai._core._eval import configure as _core_eval_configure

        _core_eval_configure(
            default_evaluator._flush_best,
            default_evaluator._flush_rank,
            default_evaluator._unsuited_keys,
            default_evaluator._unsuited_ranks,
            default_evaluator._nonflush6_keys,
            default_evaluator._nonflush6_ranks,
            default_evaluator._nonflush7_keys,
            default_evaluator._nonflush7_ranks,
        )

    if _use_scalar:
        from poker_ai._core._eval import (
            five as _core_five,
            six as _core_six,
            seven as _core_seven,
        )

        default_evaluator.hand_size_map = {
            5: _core_five,
            6: _core_six,
            7: _core_seven,
        }

    if _use_batch:
        from poker_ai._core._eval import multicard_batch as _core_multicard_batch

        default_evaluator._multicard_vec = _core_multicard_batch
except ImportError:
    pass
