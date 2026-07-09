# cython: language_level=3
"""Shared numeric primitives for the compiled core (header-only, cimport-able).

:func:`pairwise_sum` reproduces numpy's ``float64`` reduction **bit-for-bit**.
numpy does not sum contiguous ``float64`` naively — ``ndarray.sum`` /
``np.add.reduce`` over a contiguous axis uses the *pairwise* summation in
``numpy/core/src/umath/loops_utils.h.src`` (an 8-accumulator unrolled base case
for ``n <= 128`` and a divide-and-conquer split above it).  Any kernel that must
be byte-identical to a numpy ``.sum()`` (e.g. ``reach_after_removal``'s
``opp_reach.sum()`` over ~1326 combos, or ``_regret_match_matrix``'s per-row sum
over the action width) has to fold in exactly the same order — a naive left-to-
right accumulation differs in the last ULP and would break the frozen golden
digest.  This is the search-core analogue of the "accumulate in index order"
gotcha the blueprint kernels carried, but for numpy's *non*-sequential order.

The algorithm mirrors numpy 1.17.4 (the pinned build) verbatim; XXH-style, it is
frozen upstream, so this stays correct as long as the numpy pin holds.
"""

cimport cython


@cython.cdivision(True)
cdef inline double pairwise_sum(const double* a, Py_ssize_t n) nogil:
    """Bit-identical replica of numpy's contiguous ``float64`` pairwise sum."""
    cdef Py_ssize_t i, n2
    cdef double res
    cdef double r0, r1, r2, r3, r4, r5, r6, r7

    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[i]
        return res
    elif n <= 128:
        # 8-accumulator unrolled base case (numpy PW_BLOCKSIZE == 128).
        r0 = a[0]; r1 = a[1]; r2 = a[2]; r3 = a[3]
        r4 = a[4]; r5 = a[5]; r6 = a[6]; r7 = a[7]
        i = 8
        while i < n - (n % 8):
            r0 += a[i + 0]; r1 += a[i + 1]; r2 += a[i + 2]; r3 += a[i + 3]
            r4 += a[i + 4]; r5 += a[i + 5]; r6 += a[i + 6]; r7 += a[i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[i]
            i += 1
        return res
    else:
        # Divide and conquer, keeping the left half a multiple of 8 (numpy).
        n2 = n // 2
        n2 -= n2 % 8
        return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2)
