"""Chunked shared-memory regret table.

Phase 2 of the training pipeline refactor.

Architecture
------------
Storage backend
    Named files in ``shm_dir`` (default: ``/dev/shm``) mmap'd with
    ``MAP_SHARED``.  This provides:

    * Named blocks visible inside the directory — enables orphan detection
      (Section 2.6) by scanning ``/dev/shm`` for files starting with the
      session-specific prefix.
    * Shared memory semantics: all processes that map the same file share
      the same physical pages.  A write by any process is immediately visible
      to every other process with the file open.
    * Python 3.7 compatibility: ``multiprocessing.shared_memory`` was added
      in Python 3.8; named mmap files achieve identical semantics on POSIX.

Chunk layout
    Each chunk is a ``(CHUNK_SIZE, N_ACTIONS)`` ``int32`` array backed by a
    single mmap file.  Rows correspond to infosets in the order they were
    first encountered during training.

Dirty tracking
    ``_dirty_shared`` is a ``multiprocessing.Array('b', _MAX_DIRTY_CHUNKS)``
    — one byte per potential chunk, visible from any process.  A byte is set
    to ``1`` when data is written to the chunk (during sync flush).  The
    CheckpointManager reads this to decide which chunks to write to disk.

Alloc lock / stripe locks
    Both are ``multiprocessing.Lock()`` objects that MUST be created in the
    parent process before workers are spawned.  After fork, parent and
    children share the underlying POSIX semaphore, so locking is
    cross-process.

    * ``_alloc_lock`` — held only when a new chunk file needs to be created.
      The LMDB index serialises row allocation; the alloc lock serialises
      chunk file creation so that exactly one process calls ``os.open`` with
      ``O_CREAT | O_EXCL`` for each chunk.

    * ``_stripe_locks[stripe_id]`` where ``stripe_id = chunk_id % 256`` —
      acquired by workers only during the sync flush step when merging their
      local delta into the shared chunk.

Sync boundary
    ``_at_sync_boundary`` is a ``multiprocessing.Value('b', 0)``.  The server
    sets it to ``1`` before broadcasting a discount job and resets it to ``0``
    afterwards.  ``apply_discount`` asserts it is set to prevent accidental
    invocation outside controlled boundaries.
"""

import logging
import mmap
import multiprocessing as mp
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from poker_ai.ai.index import CHUNK_SIZE, InfosetIndex

log = logging.getLogger("poker_ai.ai.regret_table")

# ----------------------------------------------------------------------------
# Module-level constants
# ----------------------------------------------------------------------------

REGRET_FLOOR: np.int32 = np.int32(-310_000_000)
"""Per-action regret floor from Pluribus supplementary material (Section S2).
Prevents int32 underflow and allows pruned actions to recover."""

N_STRIPE_LOCKS: int = 256
"""Number of stripe locks.  With 64 workers the expected per-stripe contention
ratio is 64/256 = 0.25 — contention is low after the initial warm-up phase."""

_MAX_DIRTY_CHUNKS: int = 2048
"""Maximum number of chunks tracked for dirty-flag purposes.  At
CHUNK_SIZE=100_000 this covers up to 204.8M infosets — more than enough for
any realistic poker abstraction."""


# ----------------------------------------------------------------------------
# Helper — orphan detection (Section 2.6)
# ----------------------------------------------------------------------------

def list_orphaned_blocks(shm_dir: str = "/dev/shm") -> List[str]:
    """Return paths of any leftover ``pluribus_*`` shared memory files.

    Orphaned blocks are created when a training run is killed without a clean
    shutdown.  Call this on startup and optionally unlink them before creating
    a new run.

    Parameters
    ----------
    shm_dir:
        Directory to scan (typically ``/dev/shm``).

    Returns
    -------
    List of absolute file paths whose basename starts with ``pluribus_``.
    """
    try:
        names = os.listdir(shm_dir)
    except FileNotFoundError:
        return []
    return [
        os.path.join(shm_dir, name)
        for name in names
        if name.startswith("pluribus_")
    ]


# ----------------------------------------------------------------------------
# SparseRegretTable
# ----------------------------------------------------------------------------

class SparseRegretTable:
    """Chunked, shared-memory infoset regret table.

    Each infoset is mapped to a fixed ``(chunk_id, row)`` location via the
    ``InfosetIndex``.  All chunks are mmap'd so that any write is immediately
    visible to all processes that have the chunk open.

    This class must be instantiated in the **parent (server) process** before
    any workers are spawned.  Workers inherit the POSIX semaphores behind
    ``_alloc_lock`` and ``_stripe_locks`` and can then lazily attach to new
    chunks by calling ``_ensure_chunk``.

    Parameters
    ----------
    n_actions:
        Number of actions at every infoset.  Must match the action abstraction
        used during training.
    table_name:
        Unique name used to form shared memory file names
        (``{table_name}_{chunk_id:06d}``).  Conventionally
        ``pluribus_regret_{street}`` or ``pluribus_strategy_{street}``.
    index:
        ``InfosetIndex`` instance providing persistent infoset → row mapping.
        When ``None`` an internal index is created at a temporary directory
        (useful for standalone tests).
    shm_dir:
        Directory in which to create shared memory files.  Defaults to
        ``/dev/shm``.  Can be overridden to any writable directory (e.g. a
        ``tmp`` directory in tests).
    """

    CHUNK_SIZE: int = CHUNK_SIZE  # kept as class attribute for external use

    def __init__(
        self,
        n_actions: int,
        table_name: str,
        index: Optional[InfosetIndex] = None,
        shm_dir: str = "/dev/shm",
    ) -> None:
        if n_actions < 1:
            raise ValueError(f"n_actions must be >= 1, got {n_actions}")

        self._n_actions: int = n_actions
        self._table_name: str = table_name
        if index is None:
            import tempfile as _tempfile
            _tmp = _tempfile.mkdtemp()
            self._index: InfosetIndex = InfosetIndex(os.path.join(_tmp, "lmdb"))
            self._owns_index: bool = True
        else:
            self._index = index
            self._owns_index = False
        self._shm_dir: str = shm_dir

        # Per-table row counter (incremented when get_row allocates a new row).
        # Shared across all processes; used by CheckpointManager instead of
        # _index.n_entries which is global across all eight tables.
        self._n_allocated: mp.Value = mp.Value("l", 0)

        # Per-chunk local handles — each process maintains its own list.
        # Workers inherit the parent's list after fork; new chunks are lazily
        # opened by any process that needs them via _ensure_chunk().
        self._shm_mmaps: List[mmap.mmap] = []
        self._chunks: List[np.ndarray] = []
        self._shm_paths: List[str] = []

        # Dirty tracking — mp.Array so all processes can mark chunks dirty
        # during sync flush.
        self._dirty_shared: mp.Array = mp.Array(
            "b", _MAX_DIRTY_CHUNKS, lock=False
        )

        # Locks — MUST be created before workers fork.
        # mp.Lock() uses a POSIX semaphore shared across forked processes.
        self._alloc_lock: mp.Lock = mp.Lock()
        self._stripe_locks: List[mp.Lock] = [
            mp.Lock() for _ in range(N_STRIPE_LOCKS)
        ]

        # Sync boundary flag — server sets this before broadcasting discount.
        self._at_sync_boundary: mp.Value = mp.Value("b", 0)

        # Restore chunks for any infosets already in the index (resume path).
        n_existing = self._index.n_entries
        if n_existing > 0:
            n_chunks_needed = (n_existing + CHUNK_SIZE - 1) // CHUNK_SIZE
            log.info(
                "Resuming: restoring %d existing chunks from %d infoset entries",
                n_chunks_needed,
                n_existing,
            )
            for chunk_id in range(n_chunks_needed):
                self._ensure_chunk(chunk_id)

    # -----------------------------------------------------------------------
    # Section 2.1 — chunk file naming
    # -----------------------------------------------------------------------

    def _chunk_name(self, chunk_id: int) -> str:
        """Return the file name (not full path) for *chunk_id*."""
        return f"{self._table_name}_{chunk_id:06d}"

    def _chunk_path(self, chunk_id: int) -> str:
        """Return the full path of the shared memory file for *chunk_id*."""
        return os.path.join(self._shm_dir, self._chunk_name(chunk_id))

    # -----------------------------------------------------------------------
    # Internal — chunk lifecycle
    # -----------------------------------------------------------------------

    def _chunk_size_bytes(self) -> int:
        return CHUNK_SIZE * self._n_actions * np.dtype(np.int32).itemsize

    def _ensure_chunk(self, chunk_id: int) -> None:
        """Ensure this process has *chunk_id* open (create if missing).

        This method is the single point of entry for chunk creation and
        attachment.  It is called from ``get_row``, ``get_row_if_exists``,
        and ``get_row_by_location``.

        Thread and process safety
        -------------------------
        The fast path (``chunk_id < len(self._chunks)``) reads a Python
        ``list`` length without a lock.  On CPython, the GIL makes this safe
        for threads.  For processes, each process has its own ``_chunks``
        list; the fast path check is always safe because it only reads local
        state.

        The slow path acquires ``_alloc_lock`` (POSIX semaphore) and loops
        until ``len(self._chunks) > chunk_id``, handling concurrent
        allocations from multiple processes.
        """
        if chunk_id < len(self._chunks):
            return  # Fast path — already attached in this process.

        with self._alloc_lock:
            # Re-check under the lock: another process may have created the
            # chunk between our fast-path check and lock acquisition.
            while len(self._chunks) <= chunk_id:
                next_cid = len(self._chunks)
                self._open_or_create_chunk(next_cid)

    def _open_or_create_chunk(self, chunk_id: int) -> None:
        """Open or create the mmap file for *chunk_id* and append to locals.

        Must be called with ``_alloc_lock`` held.
        ``_chunks`` length must equal *chunk_id* on entry (sequential growth).
        """
        assert len(self._chunks) == chunk_id, (
            f"Non-sequential chunk allocation: expected {len(self._chunks)}, "
            f"got {chunk_id}"
        )

        path = self._chunk_path(chunk_id)
        size_bytes = self._chunk_size_bytes()

        # Attempt atomic create.  O_EXCL ensures only one process creates the
        # file; if FileExistsError is raised, another process already created
        # it and we fall through to plain O_RDWR.
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, size_bytes)
            # The OS guarantees the file is zero-initialised.
            log.debug("Created new chunk file: %s", path)
        except FileExistsError:
            fd = os.open(path, os.O_RDWR)
            log.debug("Attached to existing chunk file: %s", path)

        mm = mmap.mmap(fd, size_bytes, mmap.MAP_SHARED)
        # The fd can be closed after mmap; the mmap holds the inode reference.
        os.close(fd)

        flat = np.frombuffer(mm, dtype=np.int32)
        chunk = flat.reshape(CHUNK_SIZE, self._n_actions)

        self._shm_mmaps.append(mm)
        self._chunks.append(chunk)
        self._shm_paths.append(path)
        # Dirty flag initialised to False; only set when data is written.
        # No bounds check: we assert _MAX_DIRTY_CHUNKS is always enough.
        if chunk_id < _MAX_DIRTY_CHUNKS:
            self._dirty_shared[chunk_id] = 0

    # -----------------------------------------------------------------------
    # Section 2.2 — core access methods
    # -----------------------------------------------------------------------

    def get_row(self, info_set: str) -> np.ndarray:
        """Return the regret row for *info_set*, allocating on first visit.

        Returns a live view into shared memory.  Any write to the returned
        array is immediately visible in all processes that have this chunk
        attached.  Zero-initialisation is guaranteed for new rows (the OS
        zeroes the mmap backing file on creation).

        Parameters
        ----------
        info_set:
            String representation of the information set.

        Returns
        -------
        np.ndarray
            1-D ``int32`` view of shape ``(n_actions,)``.
        """
        location, is_new = self._index.get_or_create(info_set)
        if is_new:
            with self._n_allocated.get_lock():
                self._n_allocated.value += 1
        chunk_id, row = location
        self._ensure_chunk(chunk_id)
        return self._chunks[chunk_id][row]

    def merge_delta_row(self, info_set: str, delta: np.ndarray) -> None:
        """Add *delta* to the regret row for *info_set* under the stripe lock.

        Allocates a new row (and increments ``n_allocated``) on first visit.
        Acquires the stripe lock for the chunk so concurrent worker syncs do
        not race.

        Parameters
        ----------
        info_set:
            Information set string.
        delta:
            1-D array of regret increments of length ``n_actions``.
            Cast to ``int32`` before adding to the shared row.
        """
        location, is_new = self._index.get_or_create(info_set)
        if is_new:
            with self._n_allocated.get_lock():
                self._n_allocated.value += 1
        chunk_id, row_idx = location
        self._ensure_chunk(chunk_id)
        lock = self.get_stripe_lock(chunk_id)
        lock.acquire()
        try:
            np.add(
                self._chunks[chunk_id][row_idx],
                delta.astype(np.int32),
                out=self._chunks[chunk_id][row_idx],
            )
            self._mark_dirty(chunk_id)
        finally:
            lock.release()

    def get_row_if_exists(self, info_set: str) -> Optional[np.ndarray]:
        """Return the regret row for *info_set*, or ``None`` if never visited.

        Unlike ``get_row``, this never allocates a new row and is safe to
        call on preterminal or unvisited nodes without spurious allocation.

        Parameters
        ----------
        info_set:
            String representation of the information set.

        Returns
        -------
        np.ndarray or None
        """
        location = self._index.get(info_set)
        if location is None:
            return None
        chunk_id, row = location
        self._ensure_chunk(chunk_id)
        return self._chunks[chunk_id][row]

    def get_row_by_location(self, chunk_id: int, row: int) -> np.ndarray:
        """Return the regret row at a known ``(chunk_id, row)`` location.

        Callers (e.g. the sync flush loop) that already hold an index entry
        should use this instead of ``get_row_if_exists`` to avoid a redundant
        LMDB lookup.

        Parameters
        ----------
        chunk_id:
            Chunk index (0-based).
        row:
            Row within the chunk (0-based, must be < ``CHUNK_SIZE``).

        Returns
        -------
        np.ndarray
            1-D ``int32`` view of shape ``(n_actions,)``.
        """
        if row >= CHUNK_SIZE:
            raise IndexError(
                f"row {row} out of range for CHUNK_SIZE={CHUNK_SIZE}"
            )
        self._ensure_chunk(chunk_id)
        return self._chunks[chunk_id][row]

    # -----------------------------------------------------------------------
    # Section 2.3 — stripe locking
    # -----------------------------------------------------------------------

    def get_stripe_lock(self, chunk_id: int) -> mp.synchronize.Lock:
        """Return the stripe lock for *chunk_id*.

        Stripe locks serialise concurrent writes to the same chunk during sync
        flush.  Workers should acquire the stripe lock, do ``np.add(...)``,
        then release it — holding the lock for the shortest possible time.

        Parameters
        ----------
        chunk_id:
            Chunk index whose stripe lock is requested.

        Returns
        -------
        mp.Lock
        """
        return self._stripe_locks[chunk_id % N_STRIPE_LOCKS]

    # -----------------------------------------------------------------------
    # Section 2.4 — dirty tracking
    # -----------------------------------------------------------------------

    def _mark_dirty(self, chunk_id: int) -> None:
        """Mark *chunk_id* as containing data not yet written to disk."""
        if chunk_id < _MAX_DIRTY_CHUNKS:
            self._dirty_shared[chunk_id] = 1

    def get_dirty_chunks(self) -> List[int]:
        """Return chunk IDs that have been written since the last checkpoint.

        Returns
        -------
        List[int]
            Sorted list of chunk IDs with their dirty flag set.
        """
        n_chunks = len(self._chunks)
        limit = min(n_chunks, _MAX_DIRTY_CHUNKS)
        return [i for i in range(limit) if self._dirty_shared[i]]

    def clear_dirty(self, chunk_id: int) -> None:
        """Clear the dirty flag for *chunk_id* after a successful write."""
        if chunk_id < _MAX_DIRTY_CHUNKS:
            self._dirty_shared[chunk_id] = 0

    def clear_all_dirty(self) -> None:
        """Clear all dirty flags — called after a full checkpoint."""
        for i in range(_MAX_DIRTY_CHUNKS):
            self._dirty_shared[i] = 0

    # -----------------------------------------------------------------------
    # Section 2.5 — discount application
    # -----------------------------------------------------------------------

    @property
    def _at_sync_boundary_flag(self) -> bool:
        return bool(self._at_sync_boundary.value)

    def set_sync_boundary(self, value: bool) -> None:
        """Set or clear the sync-boundary flag.

        The server calls this immediately before broadcasting the discount job
        and clears it after the job completes.  This ensures ``apply_discount``
        can only be called from the correct control-flow context.

        Parameters
        ----------
        value:
            ``True`` to signal that a sync boundary is active.
        """
        self._at_sync_boundary.value = int(value)

    def apply_discount(self, factor: float) -> None:
        """Apply LCFR discount to all allocated regret rows.

        Must only be called at sync boundaries (all workers idle, all local
        deltas flushed into shared memory).  The server enforces this by only
        dispatching the discount job after a full ``job_queue.join()``.

        Algorithm
        ---------
        For each allocated chunk:

        1. Cast to ``float32``, multiply by ``factor``, cast back to ``int32``.
        2. Apply ``np.maximum(..., REGRET_FLOOR)`` to prevent underflow
           and allow pruned actions to recover (Pluribus Section S2).
        3. Write result back into the shared memory view in-place.

        Only valid rows in the last partial chunk are processed.

        Parameters
        ----------
        factor:
            Discount factor in (0, 1].  Typically computed as
            ``(t / discount_interval) / ((t / discount_interval) + 1)``
            per the Pluribus LCFR schedule.
        """
        assert self._at_sync_boundary_flag, (
            "apply_discount called outside sync boundary.  "
            "The server must call set_sync_boundary(True) before dispatching "
            "the discount job."
        )
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"Discount factor must be in (0, 1], got {factor}"
            )

        n_rows_total: int = self.n_allocated
        factor32 = np.float32(factor)

        for chunk_idx, chunk in enumerate(self._chunks):
            valid_rows = min(
                n_rows_total - chunk_idx * CHUNK_SIZE,
                CHUNK_SIZE,
            )
            if valid_rows <= 0:
                break
            view = chunk[:valid_rows]
            # Cast to float32 for multiply, then back to int32.  Never use
            # in-place float32 multiply on int32 arrays — numpy would silently
            # truncate to int32 before the floor check.
            result = (view.astype(np.float32) * factor32).astype(np.int32)
            np.maximum(result, REGRET_FLOOR, out=result)
            view[:] = result

    # -----------------------------------------------------------------------
    # Section 2.6 — orphan detection helpers
    # -----------------------------------------------------------------------

    def list_own_blocks(self) -> List[str]:
        """Return paths of all chunk files owned by this table."""
        return list(self._shm_paths)

    # -----------------------------------------------------------------------
    # Resume helper
    # -----------------------------------------------------------------------

    def _restore_chunk(self, chunk_id: int, arr: np.ndarray) -> None:
        """Write *arr* into the shared memory for *chunk_id*, creating if needed.

        Used during resume (Phase 6 ``CheckpointManager``) to reload a
        saved ``.npy`` array back into the shared memory backing store.

        Parameters
        ----------
        chunk_id:
            Chunk index to restore into.
        arr:
            2-D ``int32`` array of shape ``(n_rows, n_actions)``.
            ``n_rows`` must be ≤ ``CHUNK_SIZE``.
        """
        self._ensure_chunk(chunk_id)
        n_rows = arr.shape[0]
        if n_rows > CHUNK_SIZE:
            raise ValueError(
                f"arr has {n_rows} rows but CHUNK_SIZE={CHUNK_SIZE}"
            )
        self._chunks[chunk_id][:n_rows] = arr.astype(np.int32)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def n_chunks(self) -> int:
        """Number of chunks currently opened in this process."""
        return len(self._chunks)

    @property
    def n_actions(self) -> int:
        """Number of actions per infoset (fixed at construction)."""
        return self._n_actions

    @property
    def n_allocated(self) -> int:
        """Number of rows allocated in this table.

        Tracked per-table (not via ``_index.n_entries`` which is global
        across all eight tables sharing the same LMDB index).
        """
        return self._n_allocated.value

    @property
    def table_name(self) -> str:
        """Table name used in shared memory file names."""
        return self._table_name

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Close all mmap handles in this process.

        Does NOT unlink the underlying files — other processes may still
        need them.  Call ``unlink_all()`` when the training session is
        fully complete and no other process needs the shared memory.
        """
        for mm in self._shm_mmaps:
            try:
                mm.close()
            except Exception as exc:  # pragma: no cover
                log.warning("Failed to close mmap: %s", exc)
        self._shm_mmaps.clear()
        self._chunks.clear()
        if self._owns_index:
            self._index.close()
        log.info(
            "SparseRegretTable closed: table=%s (%d chunks)",
            self._table_name,
            len(self._shm_paths),
        )

    def unlink_all(self) -> None:
        """Unlink all chunk files from the filesystem.

        Call this only after all processes have closed the table.  After
        unlinking, the files are removed from the directory listing even
        though processes that still have them mmap'd can continue to use
        the pages until they close their handles.
        """
        for path in self._shm_paths:
            try:
                os.unlink(path)
                log.debug("Unlinked chunk file: %s", path)
            except FileNotFoundError:
                pass
            except Exception as exc:  # pragma: no cover
                log.warning("Failed to unlink %s: %s", path, exc)

    def __repr__(self) -> str:
        return (
            f"SparseRegretTable("
            f"table_name={self._table_name!r}, "
            f"n_chunks={self.n_chunks}, "
            f"n_actions={self._n_actions}, "
            f"n_allocated={self.n_allocated}"
            f")"
        )

    def __enter__(self) -> "SparseRegretTable":
        return self

    def __exit__(self, *_) -> None:
        self.close()
        self.unlink_all()
