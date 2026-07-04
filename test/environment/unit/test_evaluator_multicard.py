"""Correctness of the O(1) multicard scalar evaluator (`_six`/`_seven`).

The rewritten scalar 6-/7-card evaluators return the best-5 rank via precomputed
tables (`_flush_best`, `_nonflush6`, `_nonflush7`) instead of enumerating all
C(K,5) subsets.  They MUST be byte-identical to the enumeration.  The oracle is
`Evaluator.evaluate_batch` — the untouched, independently-tested vectorised
min-over-C(K,5) path (see ``test_evaluator_batch.py``), which is a genuinely
different algorithm from the new tables.

Guarantees, strongest first:
* ``TestExhaustiveSevenCardTable`` / ``TestExhaustiveSixCardTable`` (slow):
  EVERY possible 7-card (C(52,7)=133,784,560) and 6-card (C(52,6)=20,358,520)
  hand, new tables vs ``evaluate_batch``, exact.
* ``TestScalarWiring``: large random sample through the ACTUAL scalar
  ``_seven``/``_six`` Python methods (catches wiring bugs the vectorised lookup
  would miss — int cast, list/tuple, product dtype, flush-suit routing).
* ``TestTableSizes`` / build-time asserts: pin the perfect-hash key-space sizes.
"""

import itertools

import numpy as np
import pytest

from environment.evaluator import Evaluator
from environment.utils import make_deck_arr


@pytest.fixture(scope="module")
def ev():
    return Evaluator()


def _full_deck():
    return make_deck_arr(2, 14).astype(np.int64)


def _nonflush_arrays(ev, k):
    """Sorted (keys, vals) arrays for the k-card non-flush product table."""
    table = ev._nonflush7 if k == 7 else ev._nonflush6
    keys = np.array(sorted(table), dtype=np.int64)
    vals = np.array([table[int(x)] for x in keys], dtype=np.int64)
    return keys, vals


def _new_table_vec(ev, cards, keys, vals):
    """Vectorised reimplementation of the new scalar table lookup.

    Mirrors ``_seven``/``_six`` exactly (per-suit counts+masks, flush suit ->
    ``_flush_best[mask]``, else product -> non-flush table) so the exhaustive
    tests can compare against ``evaluate_batch`` at scale.
    """
    suit = (cards >> 12) & 0xF
    rb = (cards >> 16) & 0x1FFF
    n = cards.shape[0]
    out = np.empty(n, dtype=np.int64)
    is_flush = np.zeros(n, dtype=bool)
    flush_mask = np.zeros(n, dtype=np.int64)
    for sv in (1, 2, 4, 8):
        sel = suit == sv
        cnt = sel.sum(axis=1)
        m = np.bitwise_or.reduce(np.where(sel, rb, 0), axis=1)
        take = cnt >= 5
        flush_mask[take] = m[take]
        is_flush |= take
    out[is_flush] = ev._flush_best[flush_mask[is_flush]]
    nf = ~is_flush
    if nf.any():
        prod = np.prod((cards[nf] & 0xFF).astype(np.int64), axis=1)
        idx = np.searchsorted(keys, prod)
        # Perfect hash: every real non-flush product must be present exactly.
        assert np.array_equal(keys[idx], prod)
        out[nf] = vals[idx]
    return out


def _ncr(n, r):
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    num = den = 1
    for i in range(r):
        num *= n - i
        den *= i + 1
    return num // den


def _exhaustive_vs_batch(ev, k, shard, nshards, chunk=1 << 18):
    """Compare new tables vs evaluate_batch over the ``shard`` slice of all
    C(52,k) hands.

    Sharded by the index of a combination's FIRST card: every combination has a
    unique first card ``deck[i0]``, so assigning ``i0`` to a shard via
    ``i0 % nshards`` partitions the whole space exactly once (no overlap, no
    gap) — the per-shard processed count is checked against C(51-i0, k-1).
    This lets the slow proof run in parallel (e.g. ``pytest -n 8``).
    """
    deck = _full_deck().tolist()
    keys, vals = _nonflush_arrays(ev, k)
    processed = 0
    expected = 0
    for i0 in range(len(deck)):
        if i0 % nshards != shard:
            continue
        first = deck[i0]
        rest = deck[i0 + 1:]
        expected += _ncr(len(rest), k - 1)
        it = itertools.combinations(rest, k - 1)
        while True:
            block = list(itertools.islice(it, chunk))
            if not block:
                break
            hands = np.empty((len(block), k), dtype=np.int64)
            hands[:, 0] = first
            hands[:, 1:] = np.array(block, dtype=np.int64)
            oracle = ev.evaluate_batch(hands)
            new = _new_table_vec(ev, hands, keys, vals)
            assert np.array_equal(oracle, new), (
                f"{k}-card mismatch, shard {shard}, first-card idx {i0}"
            )
            processed += hands.shape[0]
    assert processed == expected, (shard, processed, expected)


_NSHARDS = 8


@pytest.mark.slow
@pytest.mark.parametrize("shard", range(_NSHARDS))
class TestExhaustiveSevenCardTable:
    def test_all_c52_7_hands_match_oracle(self, ev, shard):
        _exhaustive_vs_batch(ev, 7, shard, _NSHARDS)


@pytest.mark.slow
@pytest.mark.parametrize("shard", range(_NSHARDS))
class TestExhaustiveSixCardTable:
    def test_all_c52_6_hands_match_oracle(self, ev, shard):
        _exhaustive_vs_batch(ev, 6, shard, _NSHARDS)


class TestScalarWiring:
    """Drive the ACTUAL scalar methods (not the vectorised test lookup)."""

    def _sample(self, k, n, seed):
        deck = _full_deck()
        rng = np.random.default_rng(seed)
        hands = np.empty((n, k), dtype=np.int64)
        for i in range(n):
            hands[i] = rng.choice(deck, size=k, replace=False)
        return hands

    def test_seven_card_scalar_matches_oracle(self, ev):
        hands = self._sample(7, 300_000, seed=0)
        oracle = ev.evaluate_batch(hands)
        scalar = np.fromiter(
            (ev._seven(list(h)) for h in hands.tolist()),
            dtype=np.int64,
            count=hands.shape[0],
        )
        assert np.array_equal(oracle, scalar)
        # Returned values are native python ints (not numpy scalars).
        assert type(ev._seven(list(hands[0]))) is int

    def test_six_card_scalar_matches_oracle(self, ev):
        hands = self._sample(6, 200_000, seed=1)
        oracle = ev.evaluate_batch(hands)
        scalar = np.fromiter(
            (ev._six(list(h)) for h in hands.tolist()),
            dtype=np.int64,
            count=hands.shape[0],
        )
        assert np.array_equal(oracle, scalar)
        assert type(ev._six(list(hands[0]))) is int

    def test_evaluate_dispatch_matches_scalar(self, ev):
        # evaluate(cards, board) must route through the new _seven/_six.
        hands = self._sample(7, 5_000, seed=2)
        for h in hands.tolist():
            assert ev.evaluate(h[:2], h[2:]) == ev._seven(h)


class TestTableSizes:
    def test_perfect_hash_key_space_sizes(self, ev):
        assert len(ev._nonflush7) == 49205
        assert len(ev._nonflush6) == 18395

    def test_flush_best_populated_for_ge5_masks(self, ev):
        # Every 13-bit mask with popcount >= 5 has a real flush rank; all others 0.
        popcounts = np.array(
            [bin(m).count("1") for m in range(1 << 13)], dtype=np.int64
        )
        ge5 = popcounts >= 5
        assert np.all(ev._flush_best[ge5] > 0)
        assert np.all(ev._flush_best[~ge5] == 0)
        # Filled entries are flush/straight-flush ranks only.
        assert ev._flush_best[ge5].max() <= 1599  # MAX_FLUSH
        assert ev._flush_best[ge5].min() == 1     # royal flush
