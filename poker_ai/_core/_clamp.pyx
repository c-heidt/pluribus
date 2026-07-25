# cython: language_level=3
"""Opponent-model clamp kernel (A-path P2) — fused gather + blend.

The clamp computes a modeled seat's *realized* strategy at every decision node of
every iteration (opponent_modeling §5.2):

``σ̃ = c·σ̂ + (1 − c)·σ``

The pure-numpy form materialises four temporaries per node — two fancy-index
gathers (``m_rows[gof]``, ``c_rows[gof]``) plus the two products — and walks the
``(n_combos, width)`` block several times.  Since this runs at **every node on
every iteration** (unlike the model queries, which are paid once per row and then
cached), it is the clamp's steady-state cost.

This kernel fuses the gather into the arithmetic and writes **in place** into
``sigma``: one pass, zero temporaries, no gather allocations.  In-place is safe
because ``sigma`` is always freshly allocated by the caller —
``regret_match_matrix(...)`` at a root node, ``sigma_rows[gof]`` (fancy indexing,
which copies) at a clustered node.

It evaluates ``c·σ̂ + (1−c)·σ`` verbatim — **not** the cheaper ``σ + c·(σ̂ − σ)``,
which is *not* exact at ``c = 1`` (``σ + (σ̂ − σ) != σ̂`` in float64).  Condition
**B1 is ``c ≡ 1``** (naive best response), so an inexact endpoint would quietly
corrupt a headline eval arm.  Built with ``-ffp-contract=off`` so gcc cannot fuse
the multiply-add: FMA rounds differently from numpy's separate ops and would break
byte-identity with the oracle in :mod:`poker_ai.search.vform`.
"""

cimport cython
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def clamp_sigma(double[:, ::1] sigma,
                double[:, ::1] m_rows,
                double[:, ::1] c_rows,
                long long[::1] gof):
    """In-place ``σ ← σ + c·(σ̂ − σ)``; returns ``sigma``.

    ``gof`` maps combo → model row (the precomputed
    :meth:`~poker_ai.search.cluster_maps.ClusterMapper.gather_of`).  Pass ``None``
    for a root node, where row ``i`` *is* combo ``i``.
    """
    cdef Py_ssize_t n = sigma.shape[0]
    cdef Py_ssize_t w = sigma.shape[1]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t r
    cdef double c

    if gof is None:
        for i in range(n):
            c = c_rows[i, 0]
            if c == 0.0:
                continue                      # free strategy unchanged
            for j in range(w):
                sigma[i, j] = c * m_rows[i, j] + (1.0 - c) * sigma[i, j]
    else:
        for i in range(n):
            r = <Py_ssize_t>gof[i]
            c = c_rows[r, 0]
            if c == 0.0:
                continue
            for j in range(w):
                sigma[i, j] = c * m_rows[r, j] + (1.0 - c) * sigma[i, j]
    return np.asarray(sigma)
