# cython: language_level=3
"""Toolchain probe for the compiled MCCFR core (Phase 0 build skeleton).

This module carries **no training logic**.  Its sole purpose is to prove, on the
cluster's pinned Python 3.7 / numpy 1.17.4 toolchain, that:

1. the Cython extension builds, links, and imports, and
2. a typed memoryview binds **zero-copy** to the exact array layouts the real
   core reads on its hot path — a C-contiguous ``int32`` matrix (the ``/dev/shm``
   regret / strategy chunk tables; see :mod:`poker_ai.tables.chunk_store`) and a
   ``uint64`` vector (the open-addressing index cache; see
   :mod:`poker_ai.tables.shm_index_cache`).

If this compiles and :func:`self_check` passes, the architecture's load-bearing
integration seam — zero-copy access to mmap'd numpy arrays from compiled code —
is sound.  Real kernels arrive in Phase 1; the runtime never imports this on a
training path.
"""

import numpy as np


def core_build_ok() -> bool:
    """Return ``True`` — importing this proves the compiled extension loaded."""
    return True


def sum_int32_matrix(int[:, ::1] mat) -> int:
    """Sum a C-contiguous ``int32`` matrix through a typed memoryview.

    Mirrors the read the real core performs against a shm chunk view of shape
    ``(CHUNK_SIZE, n_actions)`` — a pure C loop with no Python per element and
    no copy of the underlying buffer.  Bound as ``int[:, ::1]`` because on this
    LP64 target C ``int`` is 32-bit, matching numpy ``int32`` byte-for-byte.
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = mat.shape[0]
    cdef Py_ssize_t m = mat.shape[1]
    cdef long long total = 0
    for i in range(n):
        for j in range(m):
            total += mat[i, j]
    return total


def read_uint64_slot(unsigned long long[::1] rows, Py_ssize_t slot) -> int:
    """Read one ``uint64`` slot through a typed memoryview (the cache commit word).

    Mirrors the lock-free probe's first load (``rows[slot]``) in
    :meth:`poker_ai.tables.shm_index_cache.ShmIndexCache.probe`.
    """
    return rows[slot]


def self_check() -> bool:
    """Round-trip both memoryview bindings against known values.

    Returns ``True`` iff the zero-copy int32-matrix sum and uint64-slot read
    both agree with numpy — i.e. the compiled core can read the two shm array
    layouts it depends on.  Raises ``AssertionError`` on any mismatch so a
    broken toolchain fails loud rather than silently degrading.
    """
    mat = np.arange(12, dtype=np.int32).reshape(3, 4)
    assert sum_int32_matrix(mat) == int(mat.sum()), "int32 memoryview seam broken"

    rows = np.array([10, 20, 30], dtype=np.uint64)
    assert read_uint64_slot(rows, 1) == 20, "uint64 memoryview seam broken"
    return True
