# cython: language_level=3
"""Info-set hashing + shm index probe kernel (Phase 1c).

Cython ports of two references that resolve ``info_set -> flat_row`` on the CFR
hot path, both byte/behaviour-identical to their Python originals:

* :func:`hash_info_set_128` — ``poker_ai.tables.index.hash_info_set_128``.
  Computes the 128-bit XXH3 digest with the **vendored** ``xxhash.h`` (v0.8.2,
  ``XXH_INLINE_ALL`` — compiled in, no external link) and returns
  ``(high64, low64)`` exactly as the Python reference does
  (``high64 = intdigest() >> 64``, ``low64 = intdigest() & 0xFFFF…``).  The pip
  ``xxhash`` package this replaces bundles the *same* upstream 0.8.2, and XXH3
  has been frozen since 0.8.0, so the digests are identical bit-for-bit — locked
  by :func:`self_check` (run at import) and a fuzz corpus in the tests.

* :func:`probe` — ``ShmIndexCache.probe``.  Walks the ``/dev/shm``
  open-addressing table over typed ``uint64`` memoryviews, reproducing the
  lock-free read exactly: load the commit word ``rows[slot]`` **first**
  (``EMPTY`` ⇒ miss), then compare all 128 key bits.

The inverted high/low word convention — the #1 silent-divergence trap
--------------------------------------------------------------------
``hash_info_set_128`` returns ``(high64, low64)``, but every cache call site
unpacks it as ``low, high = hash_info_set_128(...)`` and then calls
``probe(low, high)`` — so the probe's *first* argument (the word the slot is
indexed by, and ``keys[slot, 0]``) is the digest's **high64**, and its second
(``keys[slot, 1]``) is **low64**.  :func:`lookup` bakes that swap in
(``probe(high64, low64)``) so Phase 3's in-core traversal resolves rows through
the identical mapping the Python index wrote them under.  A naive port that
indexed by ``low64`` would miss *every* row — "uniform strategy everywhere",
with no crash to flag it.  The end-to-end test gates precisely this: it prewarms
a cache from a real LMDB (Python side, pip-``xxhash`` digests) and asserts
:func:`lookup` matches ``InfosetIndex.get`` for every key and rejects unseen
ones.

The vendored ``xxhash.h`` is BSD-2-Clause (Copyright (C) 2012-2021 Yann Collet).
"""

from libc.stdint cimport uint64_t
cimport cython
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_GET_SIZE


cdef extern from "_xxh3.h":
    ctypedef struct XXH128_hash_t:
        uint64_t low64
        uint64_t high64
    XXH128_hash_t XXH3_128bits(const void* data, size_t length) nogil


# All-ones free-slot sentinel — identical to ``ShmIndexCache._EMPTY``.  Real flat
# rows are dense and far below it (insert refuses ``row >= _EMPTY``), so the
# commit-word compare is unambiguous.
cdef uint64_t _EMPTY = <uint64_t>0xFFFFFFFFFFFFFFFF


cdef XXH128_hash_t _hash_bytes(bytes b):
    # PyBytes_AS_STRING / PyBytes_GET_SIZE are macros over a known ``bytes`` (no
    # error path, no refcount); ``b`` is kept alive by the caller for the call.
    cdef const char* p = PyBytes_AS_STRING(b)
    cdef Py_ssize_t n = PyBytes_GET_SIZE(b)
    return XXH3_128bits(<const void*>p, <size_t>n)


def hash_info_set_128(info_set):
    """Return the 128-bit XXH3 digest of ``info_set`` as ``(high64, low64)``.

    Drop-in for ``poker_ai.tables.index.hash_info_set_128``: ``bytes`` are hashed
    directly (the production path — compact keys from ``encode_info_set``); a
    ``str`` is utf-8 encoded first so string test doubles keep working.  The
    returned words are unsigned (``high64 = digest >> 64``,
    ``low64 = digest & 0xFFFF…``), matching the reference's ``(high, low)``.
    """
    if isinstance(info_set, str):
        info_set = (<str>info_set).encode("utf-8")
    if not isinstance(info_set, bytes):
        raise TypeError("info_set must be bytes or str")
    cdef XXH128_hash_t h = _hash_bytes(info_set)
    return (h.high64, h.low64)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Py_ssize_t _probe(uint64_t[:, ::1] keys, uint64_t[::1] rows,
                       uint64_t mask, uint64_t digest_low,
                       uint64_t digest_high) nogil:
    """Lock-free open-addressing lookup; flat row, or ``-1`` on miss."""
    cdef uint64_t slot = digest_low & mask
    cdef uint64_t r
    while True:
        r = rows[slot]            # commit word first (see module docstring)
        if r == _EMPTY:
            return -1
        if keys[slot, 0] == digest_low and keys[slot, 1] == digest_high:
            return <Py_ssize_t>r
        slot = (slot + 1) & mask


def probe(uint64_t[:, ::1] keys, uint64_t[::1] rows, mask,
          digest_low, digest_high):
    """C reproduction of ``ShmIndexCache.probe``; return the flat row or ``-1``.

    ``keys`` is the cache's ``(capacity, 2)`` uint64 array, ``rows`` its
    ``(capacity,)`` uint64 array, ``mask`` its ``capacity - 1``.  ``digest_low``
    / ``digest_high`` follow the cache's argument convention verbatim — see the
    module docstring: ``digest_low`` is the word the slot is indexed by and is
    matched against ``keys[slot, 0]``.  Returns ``-1`` for a miss (the Python
    reference returns ``None``).
    """
    return _probe(keys, rows, <uint64_t>mask,
                  <uint64_t>digest_low, <uint64_t>digest_high)


def lookup(uint64_t[:, ::1] keys, uint64_t[::1] rows, mask, info_set):
    """Full ``InfosetIndex.get`` shm path in C: hash ``info_set``, then probe
    with the inverted word convention (slot indexed by ``high64``).

    Returns the flat row or ``-1`` on miss — the Phase-3 in-core row resolver.
    Equivalent to ``InfosetIndex.get`` with a cache attached, for the identical
    ``(keys, rows, mask)`` of that cache.
    """
    if isinstance(info_set, str):
        info_set = (<str>info_set).encode("utf-8")
    if not isinstance(info_set, bytes):
        raise TypeError("info_set must be bytes or str")
    cdef XXH128_hash_t h = _hash_bytes(info_set)
    # index.get: ``low, high = hash_info_set_128()`` (returns (high64, low64)),
    # then ``probe(low, high)`` -> probe(digest_low=high64, digest_high=low64).
    return _probe(keys, rows, <uint64_t>mask, h.high64, h.low64)


# Known-answer vector locking the vendored XXH3 (0.8.2) against the pip
# ``xxhash`` the tables were written with.  ``b"hello world"`` -> these words
# (verified against ``xxhash.xxh3_128(...).intdigest()``); a version/ABI drift
# changes them and fails the import LOUDLY.  This module is only imported when
# the kernel is explicitly enabled, so the check fires exactly when it matters.
cdef uint64_t _KAT_HIGH = <uint64_t>0xdf8d09e93f874900
cdef uint64_t _KAT_LOW = <uint64_t>0xa99b8775cc15b6c7


def self_check():
    """Return ``True`` iff the vendored XXH3 matches the known digest of a fixed
    vector (``b"hello world"``); raise ``RuntimeError`` on mismatch."""
    cdef XXH128_hash_t h = _hash_bytes(b"hello world")
    cdef object got_high = h.high64
    cdef object got_low = h.low64
    if got_high != _KAT_HIGH or got_low != _KAT_LOW:
        raise RuntimeError(
            "vendored XXH3 digest mismatch for b'hello world': "
            "got high={:#018x} low={:#018x} — the vendored xxhash.h disagrees "
            "with the pip xxhash the index was written under (version/ABI "
            "drift).".format(got_high, got_low)
        )
    return True


# Fail loud at import when this kernel is enabled and the vendored hash is wrong.
self_check()
