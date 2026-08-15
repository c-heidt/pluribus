# cython: language_level=3
"""Compiled kernel for the search core's leaf-rollout cluster refresh.

``FastState.refresh_clusters`` (``_state.pyx``) draws a fresh board per leaf-
rollout sample and needs each seat's LUT cluster at that board, for every
``MemmapLookup``-backed street. Doing that as ``n_players`` scalar Python
dict/``MemmapLookup.__getitem__`` calls (each paying a Python-level
``lex_rank()`` loop + ``comb()`` calls) is real cost at scale — see
``information_abstraction/lookup.py``'s ``lex_rank``/``_lex_rank_vec`` for the
combinadic identity this mirrors.

The existing vectorized alternative (``clusters_for_board``, built for
thousands of combo rows against one board) was tried here first and reverted:
its fixed per-call numpy overhead only pays off well above the player counts
this codebase runs. This kernel instead reimplements the row-index math
directly in ``nogil`` C — no Python object overhead per seat, and no numpy
per-call setup cost — matching ``refresh_clusters``'s actual shape (a
handful of seats, one board, called every rollout sample).
"""
import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t


cdef int64_t _lex_rank_c(int64_t* combo, int k, int64_t n, const int64_t[:, ::1] C) nogil:
    """Lexicographic rank of a k-combination — same identity as
    ``information_abstraction.lookup.lex_rank``/``_lex_rank_vec``, scalar-C."""
    cdef int64_t rank = 0
    cdef int64_t prev = -1
    cdef int64_t start
    cdef int64_t remaining
    cdef int i
    for i in range(k):
        start = prev + 1
        remaining = k - i
        rank += C[n - start, remaining] - C[n - combo[i], remaining]
        prev = combo[i]
    return rank


cdef int64_t _deck_idx_c(int64_t card, const int64_t[::1] keys_sorted,
                          const int64_t[::1] vals_sorted, int n_cards) nogil:
    """Binary search ``keys_sorted`` for ``card``; -1 if not found.

    Should never miss for a real dealt hole/board card, but must not be
    undefined behavior if it ever does — the caller treats -1 as "no
    cluster", the same convention ``clusters_for_board`` uses.
    """
    cdef int lo = 0
    cdef int hi = n_cards - 1
    cdef int mid
    while lo <= hi:
        mid = (lo + hi) // 2
        if keys_sorted[mid] == card:
            return vals_sorted[mid]
        elif keys_sorted[mid] < card:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def batch_row_lookup(holes, board, keys_sorted, vals_sorted, C, mm, int n_cards):
    """Cluster id for ``n`` hole pairs against ONE fixed board.

    This is ``refresh_clusters``'s call shape — a handful of seats' hole
    pairs against one sampled board — NOT ``clusters_for_board``'s shape
    (thousands of combo rows). Every row is independent; a row that fails to
    resolve (an unknown card, or an out-of-bounds row index — neither should
    happen for real dealt cards) returns ``-1`` rather than raising or
    reading out of bounds, matching the existing "no cluster" convention
    ``refresh_clusters`` already treats as ``has_cluster = 0``.

    Parameters
    ----------
    holes : (n, 2) int64 array-like
        Each seat's hole cards, eval-card integers.
    board : (k,) int64 array-like
        The public cards for this street.
    keys_sorted, vals_sorted : (n_cards,) int64 arrays
        ``MemmapLookup._indexer()``'s deck-index tables.
    C : (n_cards + 1, 6) int64 array
        ``MemmapLookup._indexer()``'s precomputed comb table.
    mm : (n_rows,) uint16 array
        The street's cluster-id memmap (or a plain array in tests).
    n_cards : int
        Deck size.
    """
    cdef const int64_t[:, ::1] h = np.ascontiguousarray(holes, dtype=np.int64)
    cdef const int64_t[::1] b = np.ascontiguousarray(board, dtype=np.int64)
    cdef Py_ssize_t n = h.shape[0]
    cdef int k = <int>b.shape[0]
    out = np.empty(n, dtype=np.int64)
    if n == 0:
        return out
    cdef int64_t[::1] out_view = out
    cdef const int64_t[::1] ks = np.ascontiguousarray(keys_sorted, dtype=np.int64)
    cdef const int64_t[::1] vs = np.ascontiguousarray(vals_sorted, dtype=np.int64)
    cdef const int64_t[:, ::1] Cv = np.ascontiguousarray(C, dtype=np.int64)
    cdef const unsigned short[::1] mm_v = np.ascontiguousarray(mm, dtype=np.uint16)
    cdef int64_t n_rows = mm_v.shape[0]
    cdef int64_t h_idx[2]
    cdef int64_t p_idx[5]          # max board len is 5 (river)
    cdef int64_t hole_rank, public_rank, row, h0, h1, c, tmp
    cdef Py_ssize_t i
    cdef int j, jj
    cdef bint bad
    with nogil:
        for i in range(n):
            bad = False
            h_idx[0] = _deck_idx_c(h[i, 0], ks, vs, n_cards)
            h_idx[1] = _deck_idx_c(h[i, 1], ks, vs, n_cards)
            if h_idx[0] < 0 or h_idx[1] < 0:
                bad = True
            elif h_idx[0] > h_idx[1]:
                tmp = h_idx[0]
                h_idx[0] = h_idx[1]
                h_idx[1] = tmp
            if not bad:
                for j in range(k):
                    c = _deck_idx_c(b[j], ks, vs, n_cards)
                    if c < 0:
                        bad = True
                        break
                    p_idx[j] = c
            if not bad:
                h0 = h_idx[0]
                h1 = h_idx[1]
                for j in range(k):
                    c = p_idx[j]
                    p_idx[j] = c - (1 if h0 < c else 0) - (1 if h1 < c else 0)
                # insertion sort, k <= 5
                for j in range(1, k):
                    tmp = p_idx[j]
                    jj = j - 1
                    while jj >= 0 and p_idx[jj] > tmp:
                        p_idx[jj + 1] = p_idx[jj]
                        jj -= 1
                    p_idx[jj + 1] = tmp
                hole_rank = _lex_rank_c(h_idx, 2, n_cards, Cv)
                public_rank = _lex_rank_c(p_idx, k, n_cards - 2, Cv)
                row = hole_rank * Cv[n_cards - 2, k] + public_rank
                if row < 0 or row >= n_rows:
                    bad = True
            out_view[i] = mm_v[row] if not bad else -1
    return out
