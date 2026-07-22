"""Per-infoset array access layer for CFR regret and strategy tables.

A :class:`ChunkedTable` is the data-access facade that CFR code uses
to read and write per-infoset integer arrays.  It composes two lower
layers:

- :class:`~poker_ai.tables.index.InfosetIndex` — persistent string → row
  mapping (LMDB-backed).
- :class:`~poker_ai.tables.chunk_store.ChunkStore` — shared-memory chunk
  files holding the actual integer arrays.

Higher-level code (CFR traversals, strategy updates) never sees the
chunk IDs, the mmap, or the LMDB keys.  It just calls
:meth:`get_row`, :meth:`update_row`, or :meth:`merge_delta_row` on
the table.

Concurrency model
-----------------
The same table is accessed by many worker processes simultaneously.
Two race classes are handled separately:

- **Row allocation** — an information set seen for the first time
  must be assigned a unique flat row number.  This is serialised by
  the LMDB write transaction inside
  :meth:`InfosetIndex.get_or_create`; see its docstring for why the
  one-shot transaction is race-free.
- **Row updates** — concurrent writers to the same chunk would race
  on the shared-memory view.  :class:`ChunkedTable` mitigates this
  with a set of stripe locks (``N_STRIPE_LOCKS``); every
  :meth:`update_row` and :meth:`merge_delta_row` acquires the lock
  corresponding to ``chunk_id % N_STRIPE_LOCKS`` before touching the
  array.  Different chunks therefore rarely contend, while writes to
  the same chunk are fully serialised.

Must be instantiated in the **parent process before workers are
forked** so that the stripe locks (POSIX semaphores) and the per-table
row counter (a :class:`multiprocessing.Value`) are inherited.
"""

import logging
import multiprocessing as mp
import os
from typing import List, Optional

import numpy as np

from poker_ai.tables.chunk_store import CHUNK_SIZE, ChunkStore
from poker_ai.tables.index import InfosetIndex

log = logging.getLogger("poker_ai.tables.chunked_table")

N_STRIPE_LOCKS: int = 256
"""Number of stripe locks.

With 64 workers the expected per-stripe contention ratio is
``64 / 256 = 0.25`` — low enough that stripe locks rarely serialise
meaningful work, while still giving one-lock-per-chunk safety.
"""


def list_orphaned_blocks(shm_dir: str = "/dev/shm") -> List[str]:
    """Return paths of any leftover ``pluribus_*`` shared-memory files.

    Useful for shutdown cleanup and for diagnostic scripts that want
    to detect stray chunk files from a crashed run.

    Parameters
    ----------
    shm_dir : str, optional
        Shared-memory directory to inspect.  Defaults to ``/dev/shm``.

    Returns
    -------
    list[str]
        Absolute paths of files whose names start with ``pluribus_``
        inside ``shm_dir``.  Empty if the directory does not exist.
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


class ChunkedTable:
    """Infoset-keyed accessor over a chunked shared-memory table.

    One instance represents one logical CFR table (e.g. pre-flop
    regrets, turn strategy visits).  The instance is built on a
    shared :class:`~poker_ai.tables.index.InfosetIndex` plus a private
    :class:`~poker_ai.tables.chunk_store.ChunkStore`.

    Attributes
    ----------
    store : ChunkStore
        The underlying chunk storage.  Exposed so callers that need
        chunk-level operations (e.g. checkpoint save/restore, row
        location lookup) can go through a stable public interface
        instead of reaching into private members.
    n_chunks : int
        Number of chunks opened in the current process.
    n_actions : int
        Row width — number of action columns per row.
    n_allocated : int
        Number of rows allocated by this table (distinct from
        :attr:`InfosetIndex.n_allocated_rows`, which is per-street and
        counts every infoset on that street regardless of which
        table is consuming it).
    table_name : str
        Unique prefix for this table's chunk-file names.
    """

    def __init__(
        self,
        n_actions: int,
        table_name: str,
        index: Optional[InfosetIndex] = None,
        shm_dir: str = "/dev/shm",
    ) -> None:
        """Create a new :class:`ChunkedTable`.

        Parameters
        ----------
        n_actions : int
            Number of action columns per row.  Must be at least 1.
        table_name : str
            Unique prefix for shared-memory file names
            (``{table_name}_{chunk_id:06d}``).
        index : InfosetIndex, optional
            Shared infoset index.  When ``None`` a private index
            backed by a temporary directory is created — useful for
            standalone unit tests, not for production training where
            multiple tables share a single per-street index.
        shm_dir : str, optional
            Directory for shared-memory chunk files.  Defaults to
            ``/dev/shm``.
        """
        if n_actions < 1:
            raise ValueError(f"n_actions must be >= 1, got {n_actions}")

        self._n_actions = n_actions
        self._table_name = table_name

        if index is None:
            import tempfile as _tempfile
            _tmp = _tempfile.mkdtemp()
            self._index = InfosetIndex(os.path.join(_tmp, "lmdb"))
            self._owns_index = True
        else:
            self._index = index
            self._owns_index = False

        self._store = ChunkStore(
            table_name=table_name,
            n_actions=n_actions,
            shm_dir=shm_dir,
        )

        # Per-table row counter — tracks how many rows this table has
        # allocated.  Distinct from index.n_allocated_rows which counts
        # every infoset registered in the shared index.
        self._n_allocated: mp.Value = mp.Value("l", 0) # type: ignore

        self._stripe_locks: List[mp.Lock] = [ # type: ignore
            mp.Lock() for _ in range(N_STRIPE_LOCKS)
        ]

        # Pre-open chunks for rows already registered in the index
        # (resume path).  Workers inherit the open mmaps via fork.
        n_existing = self._index.n_allocated_rows
        if n_existing > 0:
            n_chunks = (n_existing + CHUNK_SIZE - 1) // CHUNK_SIZE
            log.info(
                "Resuming: opening %d existing chunks from %d infoset entries",
                n_chunks, n_existing,
            )
            for chunk_id in range(n_chunks):
                self._store.ensure_open(chunk_id)
            self._n_allocated.value = n_existing

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_row(self, info_set: str) -> np.ndarray:
        """Return the row for *info_set*, allocating one on first visit.

        The returned array is a view into shared memory; callers can
        read and write it directly, but any write MUST go through
        :meth:`update_row` or :meth:`merge_delta_row` to maintain
        stripe-lock correctness across processes.  ``get_row`` is
        intended for read-only access or for single-process tests.

        Parameters
        ----------
        info_set : str
            Information-set string.

        Returns
        -------
        np.ndarray
            1-D int32 view of length ``n_actions``.
        """
        chunk_id, local_row = self._locate_row(info_set)
        return self._store.view(chunk_id)[local_row]

    def get_row_if_exists(self, info_set: str) -> Optional[np.ndarray]:
        """Return the row for *info_set*, or ``None`` if never allocated.

        Unlike :meth:`get_row` this does not allocate a new row on a
        miss.  Used by CFR traversal to detect unseen infosets so it
        can fall back to a uniform-strategy zero vector without
        polluting the index with rows for nodes it never returned to.

        Parameters
        ----------
        info_set : str
            Information-set string.

        Returns
        -------
        np.ndarray or None
            1-D int32 view if the infoset was previously allocated,
            otherwise ``None``.
        """
        flat_row = self._index.get(info_set)
        if flat_row is None:
            return None
        chunk_id, local_row = divmod(flat_row, CHUNK_SIZE)
        return self._store.view(chunk_id)[local_row]

    def get_row_by_location(self, chunk_id: int, row: int) -> np.ndarray:
        """Return the row at a known ``(chunk_id, row)`` location.

        Used by iteration-style utilities (e.g. checkpoint restore)
        that already know the physical layout and do not need to
        consult the index.

        Parameters
        ----------
        chunk_id : int
            Zero-based chunk index.
        row : int
            Row index inside the chunk.  Must be ``< CHUNK_SIZE``.

        Returns
        -------
        np.ndarray
            1-D int32 view of length ``n_actions``.
        """
        if row >= CHUNK_SIZE:
            raise IndexError(f"row {row} out of range for CHUNK_SIZE={CHUNK_SIZE}")
        return self._store.view(chunk_id)[row]

    def update_row(self, info_set: str, action_idx: int, amount: int) -> None:
        """Add *amount* to a single action slot under the stripe lock.

        Used by the strategy-update traversal to increment one visit
        count at a time.  The stripe lock serialises concurrent
        updates to the same chunk so visit counts cannot be lost to
        race conditions.

        Parameters
        ----------
        info_set : str
            Information-set string.  Allocated if not yet present.
        action_idx : int
            Canonical action column to update.
        amount : int
            Integer increment (typically ``1``).
        """
        chunk_id, local_row = self._locate_row(info_set)
        lock = self.get_stripe_lock(chunk_id)
        lock.acquire()
        try:
            self._store.view(chunk_id)[local_row, action_idx] += amount
            self._store.mark_dirty(chunk_id)
        finally:
            lock.release()

    def merge_delta_row(self, info_set: str, delta: np.ndarray) -> None:
        """Add *delta* to the row under the stripe lock.

        Used by :func:`poker_ai.blueprint.cfr.merge_local_delta` to flush a
        per-infoset regret increment into the shared table.  The
        stripe lock guarantees that concurrent merges for the same
        chunk serialise correctly.

        Parameters
        ----------
        info_set : str
            Information-set string.  Allocated if not yet present.
        delta : np.ndarray
            1-D integer array of length ``n_actions`` to add to the
            row.  Cast to int32 before the add.
        """
        chunk_id, local_row = self._locate_row(info_set)
        lock = self.get_stripe_lock(chunk_id)
        lock.acquire()
        try:
            np.add(
                self._store.view(chunk_id)[local_row],
                delta.astype(np.int32),
                out=self._store.view(chunk_id)[local_row],
            )
            self._store.mark_dirty(chunk_id)
        finally:
            lock.release()

    def merge_delta_rows(self, items) -> None:
        """Batch-merge many ``(info_set, delta)`` pairs, one lock per chunk.

        Semantically identical to calling :meth:`merge_delta_row` for each
        pair, but groups the rows by chunk and acquires each chunk's stripe
        lock **once** — applying every row for that chunk inside a single
        critical section — instead of once per info set.  The regret flush
        (:func:`poker_ai.blueprint.cfr.merge_local_delta`) touches many info
        sets that map to only a handful of chunks, so this cuts the number of
        (futex) lock acquisitions from *O(touched info sets)* to *O(distinct
        chunks touched)* per flush, shrinking both syscall overhead and the
        window in which workers contend for a chunk.

        Rows are resolved (and allocated) up front, before any stripe lock is
        taken; each chunk's lock is acquired and released in turn, so no two
        stripe locks are ever held at once (no deadlock).

        Parameters
        ----------
        items : Iterable[Tuple[str, np.ndarray]]
            ``(info_set, delta)`` pairs.  Each ``delta`` is a 1-D integer
            array of length ``n_actions`` added to that info set's row (cast
            to int32 before the add), exactly as in :meth:`merge_delta_row`.
        """
        by_chunk = {}
        locations = self._locate_rows([info_set for info_set, _ in items])
        for (info_set, delta), (chunk_id, local_row) in zip(items, locations):
            by_chunk.setdefault(chunk_id, []).append((local_row, delta))
        for chunk_id, rows in by_chunk.items():
            lock = self.get_stripe_lock(chunk_id)
            lock.acquire()
            try:
                view = self._store.view(chunk_id)
                for local_row, delta in rows:
                    np.add(
                        view[local_row],
                        delta.astype(np.int32),
                        out=view[local_row],
                    )
                self._store.mark_dirty(chunk_id)
            finally:
                lock.release()

    # ------------------------------------------------------------------
    # Stripe locking
    # ------------------------------------------------------------------

    def get_stripe_lock(self, chunk_id: int) -> mp.synchronize.Lock:
        """Return the stripe lock guarding updates to *chunk_id*.

        The mapping is deterministic: every process computes the
        same lock for the same ``chunk_id``.  Exposed publicly so
        tests and low-level code can acquire the lock directly if
        needed.
        """
        return self._stripe_locks[chunk_id % N_STRIPE_LOCKS]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def store(self) -> ChunkStore:
        """The underlying :class:`ChunkStore`."""
        return self._store

    @property
    def n_chunks(self) -> int:
        """Number of chunks opened in the current process."""
        return self._store.n_open

    @property
    def n_actions(self) -> int:
        """Number of action columns in each row."""
        return self._n_actions

    @property
    def n_allocated(self) -> int:
        """Number of rows allocated by this table."""
        return self._n_allocated.value

    @property
    def table_name(self) -> str:
        """Unique file-name prefix for this table's chunks."""
        return self._table_name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying chunk store and, if owned, the index.

        Does not unlink files from disk; see :meth:`unlink_all`.
        """
        self._store.close()
        if self._owns_index:
            self._index.close()

    def unlink_all(self) -> None:
        """Remove all chunk files this table created or attached."""
        self._store.unlink_all()

    def __repr__(self) -> str:
        """Debug representation with table name, chunk and row counts."""
        return (
            f"ChunkedTable("
            f"table_name={self._table_name!r}, "
            f"n_chunks={self.n_chunks}, "
            f"n_actions={self._n_actions}, "
            f"n_allocated={self.n_allocated})"
        )

    def __enter__(self) -> "ChunkedTable":
        """Enter a context-manager scope; returns ``self``."""
        return self

    def __exit__(self, *_) -> None:
        """Exit the context manager, closing and unlinking the table."""
        self.close()
        self.unlink_all()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _locate_row(self, info_set: str) -> tuple:
        """Resolve *info_set* to a ``(chunk_id, local_row)`` pair.

        Looks up or allocates the flat row number via the
        :class:`InfosetIndex`, increments the per-table allocation
        counter if the row is freshly allocated, then decomposes the
        flat number into chunk coordinates and ensures the chunk is
        mapped in the current process.
        """
        flat_row, is_new = self._index.get_or_create(info_set)
        if is_new:
            with self._n_allocated.get_lock():
                self._n_allocated.value += 1
        chunk_id, local_row = divmod(flat_row, CHUNK_SIZE)
        self._store.ensure_open(chunk_id)
        return chunk_id, local_row

    def _locate_rows(self, info_sets) -> list:
        """Batched :meth:`_locate_row`: ``(chunk_id, local_row)`` per info set.

        Resolves/allocates every flat row through the index in **one** batched
        call (:meth:`InfosetIndex.get_or_create_many` — one LMDB write txn for
        the whole batch instead of one per new info set), then applies the same
        per-row bookkeeping :meth:`_locate_row` does: bump the per-table
        allocation counter once by the number of freshly-allocated rows, and
        ``ensure_open`` each row's chunk (idempotent; opens any chunks a batch of
        allocations crossed into).  Order-preserving.
        """
        results = self._index.get_or_create_many(info_sets)
        n_new = 0
        locations = []
        for flat_row, is_new in results:
            if is_new:
                n_new += 1
            chunk_id, local_row = divmod(flat_row, CHUNK_SIZE)
            self._store.ensure_open(chunk_id)
            locations.append((chunk_id, local_row))
        if n_new:
            with self._n_allocated.get_lock():
                self._n_allocated.value += n_new
        return locations
