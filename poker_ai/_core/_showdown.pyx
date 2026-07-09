# cython: language_level=3
"""Range-showdown kernels (Phase 1) — Cython port of
``environment.range_showdown.showdown_cfv`` and ``reach_after_removal``.

These are the vector regime's per-terminal settlement (``vector_payout`` →
``showdown_cfv`` for a showdown, ``reach_after_removal`` for a fold), the single
biggest cost of a vector CFR iteration (~48%, dominated by the argsort + the
~10 numpy passes of ``showdown_cfv``).

**Byte-identical to the numpy reference by construction** — the frozen vector
golden digest must survive turning these on:

- ``showdown_cfv`` uses only ``bincount`` + ``cumsum`` (both *sequential* float64
  reductions), so the accumulation order is reproduced exactly with in-order
  loops.  It uses **no** ``np.sum`` over the combo axis, so no pairwise sum is
  needed here.
- The rank→group assignment (numpy: stable ``argsort`` + adjacent-diff ``cumsum``)
  is replaced by an O(n + R) **counting sort** over the bounded hand-rank domain
  ``[1, 7462]`` (the plan's key win: the argsort is ~10% of a vector iter by
  itself).  ``group_of[i]`` is *uniquely* the dense ascending rank of the value,
  independent of sort stability, so this is bit-identical to the argsort path.
  Board-incompatible combos carry the sentinel rank ``1<<30`` (> every real rank)
  and fall into one trailing group; they contribute zero reach and are zeroed at
  the end, exactly as numpy.  Any rank outside ``[1,7462] ∪ {sentinel}`` (only
  reachable from synthetic fuzz, never the evaluator) falls back to numpy
  ``unique(return_inverse)`` — correct, just not the fast path.
- ``reach_after_removal`` sums ``opp_reach`` over the combo axis, so it *does* use
  :func:`poker_ai._core._mathutil.pairwise_sum` to match numpy's non-sequential
  reduction bit-for-bit.
"""

cimport cython
import numpy as np

from poker_ai._core._mathutil cimport pairwise_sum

# Hand-rank domain for the counting sort.  ``_MAX_REAL_RANK`` is the standard
# Cactus-Kev distinct-rank count (best = 1 .. worst = 7462); ``_SENTINEL`` mirrors
# ``environment.range_showdown._SENTINEL_RANK`` (board-incompatible combos).  Only
# the *fast path* depends on these — any out-of-domain rank routes to the numpy
# fallback, so a drift in either constant costs speed, never correctness.
cdef long _MAX_REAL_RANK = 7462
cdef long _SENTINEL = 1 << 30


def _removal_index(combo_cards):
    """Local copy of ``range_showdown.removal_index`` for the ``removal=None`` path.

    Imported lazily from the Python module would risk an import cycle (that module
    imports this kernel for wiring); the hot path always passes ``removal``, so
    this is only the rare standalone fallback.
    """
    uniq = np.unique(combo_cards)
    s0 = np.searchsorted(uniq, combo_cards[:, 0])
    s1 = np.searchsorted(uniq, combo_cards[:, 1])
    return s0.astype(np.int64), s1.astype(np.int64), int(uniq.shape[0])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _group_of(const long[::1] ranks, Py_ssize_t n):
    """Return ``(group_of int64[n], n_groups)`` = dense ascending rank of each value.

    Fast counting-sort path for the ``[1,7462] ∪ {sentinel}`` domain; numpy
    ``unique(return_inverse)`` fallback otherwise (bit-identical either way).
    """
    cdef long r
    cdef Py_ssize_t i
    cdef bint has_sentinel = False

    present = np.zeros(_MAX_REAL_RANK + 1, dtype=np.uint8)
    cdef unsigned char[::1] pres = present
    for i in range(n):
        r = ranks[i]
        if 1 <= r <= _MAX_REAL_RANK:
            pres[r] = 1
        elif r == _SENTINEL:
            has_sentinel = True
        else:
            # Out-of-domain rank (synthetic only): exact numpy fallback.
            uniq, inv = np.unique(np.asarray(ranks), return_inverse=True)
            return np.ascontiguousarray(inv, dtype=np.int64), int(uniq.shape[0])

    # Dense id per present real rank (ascending), sentinel last.
    gmap = np.empty(_MAX_REAL_RANK + 1, dtype=np.int64)
    cdef long[::1] gm = gmap
    cdef long next_id = 0
    for r in range(1, _MAX_REAL_RANK + 1):
        if pres[r]:
            gm[r] = next_id
            next_id += 1
    cdef long sentinel_id = next_id
    cdef long n_groups = next_id + (1 if has_sentinel else 0)

    group_of = np.empty(n, dtype=np.int64)
    cdef long[::1] g = group_of
    for i in range(n):
        r = ranks[i]
        if r == _SENTINEL:
            g[i] = sentinel_id
        else:
            g[i] = gm[r]
    return group_of, int(n_groups)


@cython.boundscheck(False)
@cython.wraparound(False)
def showdown_cfv(ranks, valid, combo_cards, opp_reach, stake,
                 dead=0.0, removal=None):
    """Counterfactual value per acting combo vs a reach-weighted opponent range.

    Byte-identical drop-in for ``range_showdown.showdown_cfv``.  See that function
    for the full semantics (``v = stake*(beats-loses) + dead*(beats + ties/2)``
    with exact card removal).
    """
    if removal is None:
        removal = _removal_index(combo_cards)
    s0_arr, s1_arr, deck_size_py = removal

    cdef const long[::1] rk = np.ascontiguousarray(ranks, dtype=np.int64)
    cdef const unsigned char[::1] vld = np.ascontiguousarray(
        np.asarray(valid, dtype=bool).view(np.uint8))
    cdef const long[::1] s0 = np.ascontiguousarray(s0_arr, dtype=np.int64)
    cdef const long[::1] s1 = np.ascontiguousarray(s1_arr, dtype=np.int64)
    cdef const double[::1] reach = np.ascontiguousarray(opp_reach, dtype=np.float64)

    cdef Py_ssize_t n = rk.shape[0]
    cdef Py_ssize_t ds = deck_size_py
    cdef double stake_c = stake
    cdef double dead_c = dead

    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    cdef double[::1] o = out

    # Board-masked opponent reach (numpy: where(valid, opp_reach, 0.0)).
    w_arr = np.empty(n, dtype=np.float64)
    cdef double[::1] w = w_arr
    cdef Py_ssize_t i
    for i in range(n):
        w[i] = reach[i] if vld[i] else 0.0

    group_of, n_groups_py = _group_of(rk, n)
    cdef const long[::1] g = group_of
    cdef Py_ssize_t ng = n_groups_py

    # Per-group total reach (gt) and per-group per-card reach (M, flat ng*ds),
    # both accumulated in ascending combo index — exactly numpy's bincount order.
    gt_arr = np.zeros(ng, dtype=np.float64)
    M_arr = np.zeros(ng * ds, dtype=np.float64)
    cdef double[::1] gt = gt_arr
    cdef double[::1] M = M_arr
    cdef long gi, c0, c1
    for i in range(n):
        gt[g[i]] += w[i]
    for i in range(n):           # all s0 contributions first ...
        M[g[i] * ds + s0[i]] += w[i]
    for i in range(n):           # ... then all s1 (numpy concat order)
        M[g[i] * ds + s1[i]] += w[i]

    # Exclusive prefix (over groups) via cumsum — sequential, matches numpy.
    gt_cum_arr = np.empty(ng, dtype=np.float64)
    Mcum_arr = np.empty(ng * ds, dtype=np.float64)
    cdef double[::1] gt_cum = gt_cum_arr
    cdef double[::1] Mcum = Mcum_arr
    cdef Py_ssize_t gidx, c
    gt_cum[0] = gt[0]
    for gidx in range(1, ng):
        gt_cum[gidx] = gt_cum[gidx - 1] + gt[gidx]
    for c in range(ds):
        Mcum[c] = M[c]
    for gidx in range(1, ng):
        for c in range(ds):
            Mcum[gidx * ds + c] = Mcum[(gidx - 1) * ds + c] + M[gidx * ds + c]

    cdef double lower_total, upper_total, loses, beats, vi
    cdef double lc0, lc1, uc0, uc1, avail, ties, ct0, ct1
    cdef Py_ssize_t last = (ng - 1) * ds
    cdef double gt_total = gt_cum[ng - 1]
    for i in range(n):
        if not vld[i]:
            o[i] = 0.0
            continue
        gi = g[i]
        c0 = s0[i]
        c1 = s1[i]
        lower_total = gt_cum[gi] - gt[gi]
        lc0 = Mcum[gi * ds + c0] - M[gi * ds + c0]
        lc1 = Mcum[gi * ds + c1] - M[gi * ds + c1]
        loses = lower_total - lc0 - lc1
        upper_total = gt_total - gt_cum[gi]
        uc0 = Mcum[last + c0] - Mcum[gi * ds + c0]
        uc1 = Mcum[last + c1] - Mcum[gi * ds + c1]
        beats = upper_total - uc0 - uc1
        vi = stake_c * (beats - loses)
        if dead_c != 0.0:
            ct0 = Mcum[last + c0]
            ct1 = Mcum[last + c1]
            avail = gt_total - ct0 - ct1 + w[i]
            ties = avail - beats - loses
            vi = vi + dead_c * (beats + 0.5 * ties)
        o[i] = vi
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def reach_after_removal(combo_cards, opp_reach, removal=None):
    """Opponent reach on combos sharing no card with each acting combo.

    Byte-identical drop-in for ``range_showdown.reach_after_removal``
    (``total - on_card[c0] - on_card[c1] + opp_reach[i]``).  The ``total`` uses the
    pairwise sum to match numpy's ``w.sum()`` bit-for-bit.
    """
    if removal is None:
        removal = _removal_index(combo_cards)
    s0_arr, s1_arr, deck_size_py = removal

    cdef const long[::1] s0 = np.ascontiguousarray(s0_arr, dtype=np.int64)
    cdef const long[::1] s1 = np.ascontiguousarray(s1_arr, dtype=np.int64)
    cdef const double[::1] w = np.ascontiguousarray(opp_reach, dtype=np.float64)
    cdef Py_ssize_t n = w.shape[0]
    cdef Py_ssize_t ds = deck_size_py

    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    cdef double[::1] o = out

    cdef double total = pairwise_sum(&w[0], n)

    on_arr = np.zeros(ds, dtype=np.float64)
    cdef double[::1] on = on_arr
    cdef Py_ssize_t i
    for i in range(n):           # numpy: bincount over concat([s0, s1]) — s0 first
        on[s0[i]] += w[i]
    for i in range(n):
        on[s1[i]] += w[i]

    for i in range(n):
        o[i] = total - on[s0[i]] - on[s1[i]] + w[i]
    return out
