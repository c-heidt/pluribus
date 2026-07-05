# cython: language_level=3
"""Regret-matching kernel (Phase 1a) — Cython port of
``poker_ai.blueprint.tree_utils.calculate_strategy_from_row``.

Byte-identical to the pure-Python reference **by construction**:

- the positive-regret total is accumulated as ``float64`` in ascending index
  order (same order as the reference's ``for i in range(n)`` scan),
- the strategy is ``pos * (1.0 / total)`` in ``float64`` (same ``1.0/total``),
- and the final ``float64 -> float32`` cast is IEEE round-to-nearest-even, which
  is exactly what ``numpy.array(python_float_list, dtype=float32)`` does.

The reference is already de-numpy'd (it works on ``regret_row.tolist()``); this
kernel removes the ``tolist()`` allocation and the Python per-element loop on the
CFR hot path — a C-contiguous ``int32`` regret row — and falls back to the same
list-based logic for any other dtype (tests / non-hot callers).  Rows are tiny
(<= MAX_ACTIONS_PER_STREET, ~5-6), so this removes per-call overhead rather than
vectorising a large loop.
"""

cimport cython
import numpy as np


def calculate_strategy_from_row(regret_row, valid_mask=None):
    """Positive-regret-proportional strategy with uniform fallback (float32).

    Drop-in for the pure-Python reference; see that function's docstring for the
    full semantics.  ``regret_row`` is a 1-D numeric ndarray (int32 on the hot
    path); ``valid_mask`` is ``None``, a list, or a bool ndarray over the same
    canonical action set.
    """
    if (isinstance(regret_row, np.ndarray)
            and regret_row.dtype == np.int32
            and regret_row.flags.c_contiguous):
        return _from_int32(regret_row, valid_mask)
    # Fallback for any other dtype: identical logic over the .tolist() values,
    # matching the reference exactly (which also goes through .tolist()).
    return _from_list(regret_row.tolist(), valid_mask)


cdef object _coerce_mask(object valid_mask):
    """Return ``valid_mask`` as ``None`` or a Python list — matches the reference.

    The reference accepts ``None`` (all legal), a list (used verbatim), or a bool
    ndarray (``.tolist()``-ed).
    """
    if valid_mask is None:
        return None
    if isinstance(valid_mask, list):
        return valid_mask
    return valid_mask.tolist()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _from_int32(const int[::1] regrets, object valid_mask):
    cdef Py_ssize_t n = regrets.shape[0]
    mask = _coerce_mask(valid_mask)
    cdef bint has_mask = mask is not None

    out = np.zeros(n, dtype=np.float32)
    cdef float[::1] o = out

    cdef double total = 0.0, inv, p
    cdef Py_ssize_t n_valid = 0, i
    cdef int r

    for i in range(n):
        if has_mask and not mask[i]:
            continue
        n_valid += 1
        r = regrets[i]
        if r > 0:
            total += r

    if total > 0.0:
        inv = 1.0 / total
        for i in range(n):
            if has_mask and not mask[i]:
                continue
            r = regrets[i]
            if r > 0:
                o[i] = <float>((<double>r) * inv)
        return out

    # Uniform fallback over the legal subset (or all actions when unmasked).
    if has_mask:
        if n_valid > 0:
            p = 1.0 / n_valid
            for i in range(n):
                if mask[i]:
                    o[i] = <float>p
    else:
        p = 1.0 / n
        for i in range(n):
            o[i] = <float>p
    return out


cdef object _from_list(list regrets, object valid_mask):
    cdef Py_ssize_t n = len(regrets)
    mask = _coerce_mask(valid_mask)
    cdef bint has_mask = mask is not None

    out = np.zeros(n, dtype=np.float32)
    cdef float[::1] o = out

    cdef double total = 0.0, inv, p, rv
    cdef Py_ssize_t n_valid = 0, i

    for i in range(n):
        if has_mask and not mask[i]:
            continue
        n_valid += 1
        rv = regrets[i]          # Python int/float -> C double
        if rv > 0.0:
            total += rv

    if total > 0.0:
        inv = 1.0 / total
        for i in range(n):
            if has_mask and not mask[i]:
                continue
            rv = regrets[i]
            if rv > 0.0:
                o[i] = <float>(rv * inv)
        return out

    if has_mask:
        if n_valid > 0:
            p = 1.0 / n_valid
            for i in range(n):
                if mask[i]:
                    o[i] = <float>p
    else:
        p = 1.0 / n
        for i in range(n):
            o[i] = <float>p
    return out
