# cython: language_level=3
"""Hand-evaluator kernel (Phase 1d) — Cython port of
``environment.evaluator.Evaluator._five`` / ``_six`` / ``_seven``.

These three scalar evaluators map a 5-, 6- or 7-card hand (card ints in the
internal Cactus-Kev-style bit encoding) to a rank in ``[1, 7462]`` (1 = best).
``_seven`` is called once per active player at **every showdown leaf** of a CFR
traversal, so it is a training hot path; this kernel is also the terminal-payout
evaluator Phase 3's in-core traversal will call directly.

Byte-identical to the Python references **by construction** — it reuses the exact
same dense lookup tables (the single source of truth is ``HandRankTable``, from
which the evaluator derives them), and both branches match the vectorised
evaluator (``_eval5_vec`` / ``_multicard_vec``) which the exhaustive tests already
prove equals the scalar dicts and the 21-subset oracle:

* **Flush branch** — a suit with >=5 cards decides the hand (flush precludes
  quads/full-house); the rank is ``_flush_best[mask]`` for 6/7 cards, or
  ``_flush_rank[rankOR]`` for exactly 5.  Both dense ``int16[8192]`` tables are
  indexed by the 13-bit rank mask directly.
* **Non-flush branch** — the product of the cards' rank primes (``card & 0xFF``)
  is a collision-free perfect hash of the rank multiset; the rank is resolved by
  a binary search over the sorted ``(product -> rank)`` arrays, which is exactly
  the ``np.searchsorted`` the vectorised path uses.  A product absent from the
  table (only possible if the encoding drifts) raises ``KeyError``, preserving
  the Python dict's loud-on-drift behaviour.

The card encoding carries everything the kernel needs, so no external prime table
enters the core: suit bits = ``card >> 12 & 0xF`` (one-hot 1/2/4/8), 13-bit rank
mask = ``card >> 16 & 0x1FFF``, rank prime = ``card & 0xFF``.

The dense tables are DUMPED from the live ``Evaluator`` via :func:`configure`
(never rebuilt here) so the kernel and the Python oracle can never disagree on
the table contents; the kernel refuses to run until configured.
"""

from libc.stdint cimport int16_t, int64_t
cimport cython
import numpy as np


# Tables installed by configure() — memoryviews over the evaluator's own arrays
# (kept alive by the memoryview reference).  int16 ranks, int64 product keys.
cdef int16_t[::1] _flush_best      # 6/7-card flush: rank mask -> best flush rank
cdef int16_t[::1] _flush_rank      # 5-card flush:   rank mask -> flush rank
cdef int64_t[::1] _unsuited_keys   # 5-card non-flush: sorted prime products
cdef int16_t[::1] _unsuited_ranks
cdef int64_t[::1] _nf6_keys        # 6-card non-flush: sorted prime products
cdef int16_t[::1] _nf6_ranks
cdef int64_t[::1] _nf7_keys        # 7-card non-flush: sorted prime products
cdef int16_t[::1] _nf7_ranks
cdef bint _configured = False


def configure(flush_best, flush_rank, unsuited_keys, unsuited_ranks,
              nf6_keys, nf6_ranks, nf7_keys, nf7_ranks):
    """Install the evaluator's dense lookup tables (call once at wire time).

    All eight arrays are dumped from a live :class:`Evaluator` (``_flush_best``,
    ``_flush_rank``, and the sorted ``_unsuited`` / ``_nonflush{6,7}`` key/rank
    arrays) so the kernel indexes the *same* tables the Python oracle does.
    """
    global _flush_best, _flush_rank, _unsuited_keys, _unsuited_ranks
    global _nf6_keys, _nf6_ranks, _nf7_keys, _nf7_ranks, _configured
    _flush_best = np.ascontiguousarray(flush_best, dtype=np.int16)
    _flush_rank = np.ascontiguousarray(flush_rank, dtype=np.int16)
    _unsuited_keys = np.ascontiguousarray(unsuited_keys, dtype=np.int64)
    _unsuited_ranks = np.ascontiguousarray(unsuited_ranks, dtype=np.int16)
    _nf6_keys = np.ascontiguousarray(nf6_keys, dtype=np.int64)
    _nf6_ranks = np.ascontiguousarray(nf6_ranks, dtype=np.int16)
    _nf7_keys = np.ascontiguousarray(nf7_keys, dtype=np.int64)
    _nf7_ranks = np.ascontiguousarray(nf7_ranks, dtype=np.int16)
    _configured = True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef short _lookup(int64_t[::1] keys, int16_t[::1] ranks,
                   long long product) except? -1:
    """searchsorted-left over ``keys`` + exact-match check; ``ranks[idx]``.

    Matches the scalar dict's semantics: an exact hit for every valid hand
    (products are a perfect hash of the rank multiset), ``KeyError`` otherwise.
    """
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = keys.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = (lo + hi) >> 1
        if keys[mid] < product:
            lo = mid + 1
        else:
            hi = mid
    if lo >= keys.shape[0] or keys[lo] != product:
        raise KeyError(product)
    return ranks[lo]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _multi(cards, int64_t[::1] keys, int16_t[::1] ranks):
    """Shared 6/7-card evaluator: one pass for per-suit counts / rank masks and
    the rank-prime product, then flush table or non-flush binary search."""
    cdef int counts0 = 0, counts1 = 0, counts2 = 0, counts4 = 0, counts8 = 0
    cdef int mask1 = 0, mask2 = 0, mask4 = 0, mask8 = 0
    cdef long long product = 1
    cdef long long card
    cdef int s
    for c in cards:
        card = c
        s = <int>((card >> 12) & 0xF)
        # One-hot suit bits (1/2/4/8); counts0 collects any stray (never used).
        if s == 1:
            counts1 += 1
            mask1 |= <int>(card >> 16)
        elif s == 2:
            counts2 += 1
            mask2 |= <int>(card >> 16)
        elif s == 4:
            counts4 += 1
            mask4 |= <int>(card >> 16)
        elif s == 8:
            counts8 += 1
            mask8 |= <int>(card >> 16)
        else:
            counts0 += 1
        product *= (card & 0xFF)
    # Suit order (1, 2, 4, 8) matches the Python loop.
    if counts1 >= 5:
        return <int>_flush_best[mask1 & 0x1FFF]
    if counts2 >= 5:
        return <int>_flush_best[mask2 & 0x1FFF]
    if counts4 >= 5:
        return <int>_flush_best[mask4 & 0x1FFF]
    if counts8 >= 5:
        return <int>_flush_best[mask8 & 0x1FFF]
    return <int>_lookup(keys, ranks, product)


def five(cards):
    """Rank exactly 5 card ints; drop-in for ``Evaluator._five``."""
    if not _configured:
        raise RuntimeError("poker_ai._core._eval used before configure()")
    cdef long long c0 = cards[0]
    cdef long long c1 = cards[1]
    cdef long long c2 = cards[2]
    cdef long long c3 = cards[3]
    cdef long long c4 = cards[4]
    cdef int hand_or
    cdef long long product
    if (c0 & c1 & c2 & c3 & c4 & 0xF000) != 0:
        hand_or = <int>((c0 | c1 | c2 | c3 | c4) >> 16)
        return <int>_flush_rank[hand_or & 0x1FFF]
    product = (c0 & 0xFF) * (c1 & 0xFF) * (c2 & 0xFF) * (c3 & 0xFF) * (c4 & 0xFF)
    return <int>_lookup(_unsuited_keys, _unsuited_ranks, product)


def six(cards):
    """Rank exactly 6 card ints; drop-in for ``Evaluator._six``."""
    if not _configured:
        raise RuntimeError("poker_ai._core._eval used before configure()")
    return _multi(cards, _nf6_keys, _nf6_ranks)


def seven(cards):
    """Rank exactly 7 card ints; drop-in for ``Evaluator._seven``."""
    if not _configured:
        raise RuntimeError("poker_ai._core._eval used before configure()")
    return _multi(cards, _nf7_keys, _nf7_ranks)
