"""Shared-memory read cache for the info-set → flat-row index.

Every decision node of every CFR traversal looks up ``info_set → flat_row``
(``get_node_strategy`` → ``ChunkedTable.get_row_if_exists`` →
``InfosetIndex.get``).  Backing that with a per-call LMDB read transaction is
one of the two dominant inner-loop costs.  :class:`ShmIndexCache` replaces the
read with a lock-free probe of a ``/dev/shm`` open-addressing hash table that
is shared across the forked worker pool exactly like the chunk mmaps
(:mod:`poker_ai.tables.chunk_store`).

**LMDB stays authoritative.**  This is a *cache*, never the source of truth:
the LMDB index remains the sole persisted artifact and the sole allocator, so
checkpoint / resume / warm-start / inference are unchanged.  The cache is
ephemeral — created in the parent before the workers fork, prewarmed from the
open LMDB (:meth:`prewarm_from_cursor`), and kept current by inserting each
row the moment LMDB allocates it (:meth:`insert`).  A cache miss therefore
either means the key genuinely does not exist yet, or was allocated in the
microscopic window between the LMDB commit and the following cache insert — in
both cases the caller falls back to the existing "unseen → uniform / zero
regret" contract, which is safe.

Slot layout and the memory-ordering argument (x86-64)
-----------------------------------------------------
Two parallel ``uint64`` arrays over ``capacity`` slots (``capacity`` a power
of two, so the bucket index is a mask):

- ``keys[slot, 0] = digest_low``, ``keys[slot, 1] = digest_high`` — the full
  128-bit xxh3 digest (the same digest LMDB keys on).
- ``rows[slot] = flat_row`` — the **commit word**; ``rows[slot] == EMPTY``
  (all-ones) marks a free slot.

*Read* (lock-free): load ``rows[slot]`` **first**; ``EMPTY`` ⇒ stop the probe
(miss); otherwise load the two key words and compare all 128 bits — match ⇒
hit, else probe the next slot.

*Write* (under :attr:`_lock`): store ``keys[slot, 0]``, then ``keys[slot, 1]``,
then **last** ``rows[slot] = row``.

Why a lock-free read is safe: on x86-64 (TSO) stores are not reordered with
stores and loads are not reordered with loads, and an aligned 8-byte
store/load is atomic.  The writer publishes both key words before the commit
word; the reader reads the commit word before the key words.  So any slot a
reader observes as non-``EMPTY`` is guaranteed to carry its fully-written key.
A reader that catches a slot mid-insert (keys written, row not yet) sees
``EMPTY`` → miss → safe fallback.  Because the read verifies the full 128 bits,
the only way to return a *wrong* row is a genuine 128-bit digest collision —
identically unlikely to LMDB's own risk.  A cache bug can therefore only ever
*drop* a slot (return a miss), never return a wrong row.  This argument is
x86-64-specific; a weakly-ordered arch would need explicit fences.

Inserts are serialised by :attr:`_lock` (an inherited ``mp.Lock``); they are
rare relative to reads (once per first-seen info-set), so the lock is nearly
uncontended.  The table never grows past :attr:`_max_load` — a forked mmap
cannot be resized — so overflow raises loudly rather than corrupting.
"""

import logging
import mmap
import multiprocessing as mp
import os
import struct
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("poker_ai.tables.shm_index_cache")

# All-ones sentinel marking a free slot.  Real flat rows are dense and far
# below 2**63, so they never collide with it.
_EMPTY: np.uint64 = np.uint64(0xFFFF_FFFF_FFFF_FFFF)
_EMPTY_INT: int = int(_EMPTY)

DEFAULT_LOAD_FACTOR: float = 0.5
"""Max occupancy before :meth:`insert` refuses.  Linear probing degrades
sharply past ~0.7; 0.5 keeps the average probe length ~1.5 with headroom."""


def next_pow2(n: int) -> int:
    """Smallest power of two ``>= max(n, 1)``."""
    n = max(int(n), 1)
    return 1 << (n - 1).bit_length()


def capacity_for(expected_rows: int, load_factor: float = DEFAULT_LOAD_FACTOR) -> int:
    """Power-of-two capacity holding ``expected_rows`` at ``load_factor``."""
    if not (0.0 < load_factor < 1.0):
        raise ValueError(f"load_factor must be in (0, 1), got {load_factor}")
    return next_pow2(int(np.ceil(max(expected_rows, 1) / load_factor)))


class ShmIndexCache:
    """A fork-shared open-addressing ``digest → flat_row`` hash table.

    Must be constructed in the **parent process before workers fork** so the
    two mmaps, the allocation lock, and the occupancy counter are inherited.
    Children need no per-fork action — the ``MAP_SHARED`` mmaps already point
    at the same physical pages and reads are lock-free.
    """

    def __init__(
        self,
        name: str,
        capacity: int,
        shm_dir: str = "/dev/shm",
        load_factor: float = DEFAULT_LOAD_FACTOR,
    ) -> None:
        """Create (or attach) the two shm arrays for this street's cache.

        Parameters
        ----------
        name : str
            Unique file-name stem; the arrays live at
            ``{shm_dir}/{name}__keys`` and ``{shm_dir}/{name}__rows``.
        capacity : int
            Number of slots.  **Forced to a power of two** (rounded up) so the
            bucket index is a bit-mask; must exceed ``expected_rows /
            load_factor`` — a forked mmap cannot grow, so pick it generously
            (see :func:`capacity_for`).
        shm_dir : str
            Shared-memory directory (tmpfs); defaults to ``/dev/shm``.
        load_factor : float
            Occupancy ceiling; inserts past ``capacity * load_factor`` raise.
        """
        self._name = name
        self._shm_dir = shm_dir
        self._capacity = next_pow2(capacity)
        self._mask = self._capacity - 1
        self._max_load = load_factor
        self._max_occupancy = int(self._capacity * load_factor)

        # The parent process is the sole creator; workers inherit the live
        # mmap via fork, so we always create fresh (unlinking any stale file
        # left by a prior crashed run — attaching to it could mean a
        # different capacity / stale data).
        self._keys_path = os.path.join(shm_dir, f"{name}__keys")
        self._rows_path = os.path.join(shm_dir, f"{name}__rows")
        self._keys_mm, keys_buf = self._create(self._keys_path, self._capacity * 2)
        self._rows_mm, rows_buf = self._create(self._rows_path, self._capacity)
        self._keys = keys_buf.reshape(self._capacity, 2)
        self._rows = rows_buf
        self._rows[:] = _EMPTY  # all slots free

        self._lock: mp.synchronize.Lock = mp.Lock()  # type: ignore
        self._occupancy: mp.Value = mp.Value("q", 0)  # type: ignore

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def probe(self, digest_low: int, digest_high: int) -> Optional[int]:
        """Lock-free lookup: flat row for the 128-bit digest, or ``None``.

        See the module docstring for why reading the commit word (``rows``)
        before the key words makes this safe without a lock.
        """
        rows = self._rows
        keys = self._keys
        mask = self._mask
        slot = digest_low & mask
        while True:
            r = int(rows[slot])            # commit word first
            if r == _EMPTY_INT:
                return None
            if int(keys[slot, 0]) == digest_low and int(keys[slot, 1]) == digest_high:
                return r
            slot = (slot + 1) & mask

    def insert(self, digest_low: int, digest_high: int, row: int) -> None:
        """Publish ``digest → row`` under the allocation lock (idempotent).

        A no-op if the digest is already present (concurrent allocators that
        both missed and both allocated the *same* LMDB row).  Raises
        :class:`IndexError` if the table has reached its load-factor ceiling —
        loud failure rather than silent wrap / unbounded probe.
        """
        with self._lock:
            self._insert_locked(digest_low, digest_high, row)

    def _insert_locked(self, digest_low: int, digest_high: int, row: int) -> None:
        if row >= _EMPTY_INT:
            raise ValueError(f"flat row {row} collides with the EMPTY sentinel")
        rows = self._rows
        keys = self._keys
        mask = self._mask
        slot = digest_low & mask
        while True:
            r = int(rows[slot])
            if r == _EMPTY_INT:
                break
            if int(keys[slot, 0]) == digest_low and int(keys[slot, 1]) == digest_high:
                return  # already present — idempotent
            slot = (slot + 1) & mask
        if self._occupancy.value >= self._max_occupancy:
            raise IndexError(
                f"ShmIndexCache '{self._name}' full: {self._occupancy.value} "
                f"entries at load factor {self._max_load} (capacity "
                f"{self._capacity}). Raise the per-street index capacity "
                f"(PLURIBUS_INDEX_CAPACITY) and restart."
            )
        # Publish: both key words first, commit word (row) strictly last.
        keys[slot, 0] = np.uint64(digest_low)
        keys[slot, 1] = np.uint64(digest_high)
        rows[slot] = np.uint64(row)
        self._occupancy.value += 1

    def get_or_claim(self, digest_low: int, digest_high: int) -> tuple:
        """Atomically resolve-or-**allocate** ``digest`` → ``(row, is_new)``.

        The deferred-durability allocator: the cache — not LMDB — assigns the
        row.  A lock-free :meth:`probe` handles the common case (already-seen
        info set) with no lock.  On a miss the allocation lock is taken and the
        digest re-probed (another worker may have claimed it in the gap); if
        still absent, the next dense row (``_occupancy.value`` — the occupancy
        counter already equals the dense row count) is claimed and published via
        :meth:`_insert_locked`, which bumps the counter.

        This replaces the LMDB write transaction (B-tree traversal + meta-page
        commit, ~µs) with a re-probe + three ``uint64`` stores (~tens of ns) —
        the whole point of deferred durability.  Row agreement across workers is
        preserved exactly as before: the in-lock re-probe is the same
        double-checked-lock guarantee the LMDB writer mutex used to give, on a
        far cheaper lock.  Raises :class:`IndexError` on overflow (via
        ``_insert_locked``), the same loud failure as :meth:`insert`.

        Returns
        -------
        tuple[int, bool]
            ``(flat_row, is_new)`` — ``is_new`` is ``True`` iff this call
            allocated the row.
        """
        row = self.probe(digest_low, digest_high)
        if row is not None:
            return row, False
        with self._lock:
            # Re-probe under the lock — a concurrent claimer may have won.
            row = self.probe(digest_low, digest_high)
            if row is not None:
                return row, False
            new_row = int(self._occupancy.value)
            self._insert_locked(digest_low, digest_high, new_row)
            return new_row, True

    def get_or_claim_many(self, digests) -> list:
        """Batched :meth:`get_or_claim`: one lock acquisition for all misses.

        Probes every ``(low, high)`` lock-free; if any miss, takes the
        allocation lock **once** and claims them all (re-probing each under the
        lock, so duplicate digests within the batch collapse to one row). Order-
        preserving; returns ``[(row, is_new), ...]``.
        """
        results: list = [None] * len(digests)
        misses = []
        for i, (low, high) in enumerate(digests):
            row = self.probe(low, high)
            if row is not None:
                results[i] = (row, False)
            else:
                misses.append(i)
        if misses:
            with self._lock:
                for i in misses:
                    low, high = digests[i]
                    row = self.probe(low, high)  # re-probe (dedups within batch)
                    if row is not None:
                        results[i] = (row, False)
                    else:
                        new_row = int(self._occupancy.value)
                        self._insert_locked(low, high, new_row)
                        results[i] = (new_row, True)
        return results

    def export_since(self, watermark: int) -> tuple:
        """Return ``(keys, rows)`` for every entry with ``row >= watermark``.

        The set of rows allocated since the last durability flush — vectorised
        scan of the ``rows`` array (``!= EMPTY`` and ``>= watermark``, which are
        exactly the dense rows ``[watermark, occupancy)``).  Consumed by
        :meth:`InfosetIndex.bulk_persist` to write those digests into LMDB in one
        transaction.  O(capacity) numpy, run once per checkpoint off the hot
        path — never per allocation.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``keys`` shape ``(m, 2)`` uint64 (``[:, 0]`` = low, ``[:, 1]`` =
            high) and ``rows`` shape ``(m,)`` uint64, for the ``m`` entries at or
            above ``watermark``.
        """
        rows = self._rows
        mask = (rows != _EMPTY) & (rows >= np.uint64(watermark))
        idx = np.nonzero(mask)[0]
        return self._keys[idx], rows[idx]

    # ------------------------------------------------------------------
    # Prewarm / audit (LMDB-backed)
    # ------------------------------------------------------------------

    def prewarm_from_cursor(self, env) -> int:
        """Populate the cache from an open LMDB env (parent, pre-fork).

        Cursor-scans every real digest key (16-byte keys; the reserved
        ``__next_row__`` watermark and any debug shadow keys have other
        lengths) and inserts ``digest → row`` directly.  Single-threaded here,
        so it bypasses the lock.  Returns the number of entries loaded.
        """
        loaded = 0
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                if len(key) != 16:
                    continue  # reserved key (watermark / shadow), not a digest
                low, high = struct.unpack("<QQ", key)
                row = struct.unpack("<Q", val)[0]
                self._insert_locked(low, high, row)
                loaded += 1
        log.info(
            "Prewarmed cache '%s' with %d entries (%.1f%% of capacity %d)",
            self._name, loaded, 100.0 * loaded / self._capacity, self._capacity,
        )
        return loaded

    def audit(self, env) -> int:
        """Assert every LMDB entry resolves in the cache to the same row.

        Returns the number of entries checked.  Raises ``AssertionError`` on
        the first mismatch.  Diagnostic — run once after a verification run,
        not in the hot path.
        """
        checked = 0
        with env.begin() as txn:
            for key, val in txn.cursor():
                if len(key) != 16:
                    continue
                low, high = struct.unpack("<QQ", key)
                row = struct.unpack("<Q", val)[0]
                got = self.probe(low, high)
                assert got == row, (
                    f"cache '{self._name}' audit mismatch for {key.hex()}: "
                    f"cache={got} lmdb={row}"
                )
                checked += 1
        return checked

    # ------------------------------------------------------------------
    # Introspection / lifecycle
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def n_bytes(self) -> int:
        """Resident bytes for this cache (keys + rows arrays)."""
        return self._capacity * 8 * 3  # keys=2*8, rows=8

    def occupancy(self) -> int:
        return int(self._occupancy.value)

    def close(self) -> None:
        """Close both mmap handles (does not unlink the files)."""
        for mm in (self._keys_mm, self._rows_mm):
            try:
                mm.close()
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Failed to close cache mmap: %s", exc)

    def unlink(self) -> None:
        """Remove the two shm files (idempotent)."""
        for path in (self._keys_path, self._rows_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Failed to unlink %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create(self, path: str, n_uint64: int):
        """Create a fresh ``MAP_SHARED`` mmap of ``n_uint64`` uint64s.

        Unlinks any pre-existing file first (a stale cache from a crashed run
        could carry a different capacity), then ``O_CREAT|O_EXCL``.  Returns
        ``(mmap, uint64_view)``.
        """
        size_bytes = n_uint64 * 8
        Path(self._shm_dir).mkdir(parents=True, exist_ok=True)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.ftruncate(fd, size_bytes)
        mm = mmap.mmap(fd, size_bytes, mmap.MAP_SHARED)
        os.close(fd)
        buf = np.frombuffer(mm, dtype=np.uint64)
        return mm, buf
