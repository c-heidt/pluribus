"""Correctness of the O(1) multicard evaluator — scalar (`_six`/`_seven`) AND
the vectorised batch path (`evaluate_batch` / `_multicard_vec`).

Both return the best-5 rank via precomputed tables (`_flush_best`,
`_nonflush6`, `_nonflush7`) instead of enumerating all C(K,5) subsets.  They
MUST be byte-identical to the enumeration.  The oracle is
`Evaluator._evaluate_batch_oracle` — the original vectorised min-over-C(K,5)
path (see ``test_evaluator_batch.py``), retained verbatim as an INDEPENDENT
cross-check: it reduces `_eval5_vec` over every 5-subset using a different
table family (`_flush_rank` / `_unsuited_*`), so it shares no lookup table
with the multicard LUT under test.

Guarantees, strongest first:
* ``TestExhaustiveSevenCardTable`` / ``TestExhaustiveSixCardTable`` (slow):
  EVERY possible 7-card (C(52,7)=133,784,560) and 6-card (C(52,6)=20,358,520)
  hand, production ``evaluate_batch`` (the LUT) vs ``_evaluate_batch_oracle``,
  exact.
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
    """Compare production ``evaluate_batch`` (LUT) vs ``_evaluate_batch_oracle``
    (21-subset) over the ``shard`` slice of all C(52,k) hands.

    Sharded by the index of a combination's FIRST card: every combination has a
    unique first card ``deck[i0]``, so assigning ``i0`` to a shard via
    ``i0 % nshards`` partitions the whole space exactly once (no overlap, no
    gap) — the per-shard processed count is checked against C(51-i0, k-1).
    This lets the slow proof run in parallel (e.g. ``pytest -n 8``).
    """
    deck = _full_deck().tolist()
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
            oracle = ev._evaluate_batch_oracle(hands)
            new = ev.evaluate_batch(hands)
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
        # Compare against the independent 21-subset oracle (evaluate_batch now
        # shares the scalar's tables, so it is not an independent reference).
        oracle = ev._evaluate_batch_oracle(hands)
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
        oracle = ev._evaluate_batch_oracle(hands)
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
