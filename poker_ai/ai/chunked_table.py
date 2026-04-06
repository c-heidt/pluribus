"""Infoset data access layer.

Composes a ``ChunkStore`` (shared-memory file management) with an
``InfosetIndex`` (persistent string → row mapping) to provide
per-infoset array access for CFR training.

Concurrency is handled via stripe locks — workers acquire the stripe
lock for a chunk during delta merges to prevent races.
"""

import logging
import multiprocessing as mp
import os
from typing import List, Optional

import numpy as np

from poker_ai.ai.chunk_store import CHUNK_SIZE, ChunkStore
from poker_ai.ai.index import InfosetIndex

log = logging.getLogger("poker_ai.ai.chunked_table")

N_STRIPE_LOCKS: int = 256
"""Number of stripe locks.  With 64 workers the expected per-stripe
contention ratio is 64/256 = 0.25."""


def list_orphaned_blocks(shm_dir: str = "/dev/shm") -> List[str]:
    """Return paths of any leftover ``pluribus_*`` shared memory files."""
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
    """Infoset → shared-memory array accessor.

    Each infoset is mapped to a ``(chunk_id, local_row)`` location via
    the ``InfosetIndex`` and a ``ChunkStore``.  Writes are immediately
    visible to all processes sharing the same mmap files.

    Must be instantiated in the **parent process before workers are
    forked** so that locks are shared across processes.

    Parameters
    ----------
    n_actions:
        Number of actions per infoset row.
    table_name:
        Unique prefix for shared-memory file names.
    index:
        ``InfosetIndex`` for persistent row allocation.  When ``None``
        an internal index is created (useful for standalone tests).
    shm_dir:
        Directory for shared-memory chunk files.
    """

    def __init__(
        self,
        n_actions: int,
        table_name: str,
        index: Optional[InfosetIndex] = None,
        shm_dir: str = "/dev/shm",
    ) -> None:
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
        # allocated.  Distinct from index.n_allocated_rows which is global
        # across all tables sharing the same index.
        self._n_allocated: mp.Value = mp.Value("l", 0)

        self._stripe_locks: List[mp.Lock] = [
            mp.Lock() for _ in range(N_STRIPE_LOCKS)
        ]

        # Pre-open chunks for rows already in the index (resume path).
        n_existing = self._index.n_allocated_rows
        if n_existing > 0:
            n_chunks = (n_existing + CHUNK_SIZE - 1) // CHUNK_SIZE
            log.info(
                "Resuming: opening %d existing chunks from %d infoset entries",
                n_chunks, n_existing,
            )
            for chunk_id in range(n_chunks):
                self._store.ensure_open(chunk_id)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_row(self, info_set: str) -> np.ndarray:
        """Return the row for *info_set*, allocating on first visit."""
        chunk_id, local_row = self._locate_row(info_set)
        return self._store.view(chunk_id)[local_row]

    def get_row_if_exists(self, info_set: str) -> Optional[np.ndarray]:
        """Return the row for *info_set*, or ``None`` if never visited."""
        flat_row = self._index.get(info_set)
        if flat_row is None:
            return None
        chunk_id, local_row = divmod(flat_row, CHUNK_SIZE)
        return self._store.view(chunk_id)[local_row]

    def get_row_by_location(self, chunk_id: int, row: int) -> np.ndarray:
        """Return the row at a known (chunk_id, row) location."""
        if row >= CHUNK_SIZE:
            raise IndexError(f"row {row} out of range for CHUNK_SIZE={CHUNK_SIZE}")
        return self._store.view(chunk_id)[row]

    def update_row(self, info_set: str, action_idx: int, amount: int) -> None:
        """Add *amount* to a single action slot under the stripe lock."""
        chunk_id, local_row = self._locate_row(info_set)
        lock = self.get_stripe_lock(chunk_id)
        lock.acquire()
        try:
            self._store.view(chunk_id)[local_row, action_idx] += amount
            self._store.mark_dirty(chunk_id)
        finally:
            lock.release()

    def merge_delta_row(self, info_set: str, delta: np.ndarray) -> None:
        """Merge *delta* into the row under the stripe lock."""
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

    # ------------------------------------------------------------------
    # Stripe locking
    # ------------------------------------------------------------------

    def get_stripe_lock(self, chunk_id: int) -> mp.synchronize.Lock:
        """Return the stripe lock for *chunk_id*."""
        return self._stripe_locks[chunk_id % N_STRIPE_LOCKS]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def store(self) -> ChunkStore:
        """The underlying chunk storage."""
        return self._store

    @property
    def n_chunks(self) -> int:
        """Number of chunks opened in this process."""
        return self._store.n_open

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def n_allocated(self) -> int:
        """Number of rows allocated by this table."""
        return self._n_allocated.value

    @property
    def table_name(self) -> str:
        return self._table_name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all mmap handles."""
        self._store.close()
        if self._owns_index:
            self._index.close()

    def unlink_all(self) -> None:
        """Remove all chunk files from the filesystem."""
        self._store.unlink_all()

    def __repr__(self) -> str:
        return (
            f"ChunkedTable("
            f"table_name={self._table_name!r}, "
            f"n_chunks={self.n_chunks}, "
            f"n_actions={self._n_actions}, "
            f"n_allocated={self.n_allocated})"
        )

    def __enter__(self) -> "ChunkedTable":
        return self

    def __exit__(self, *_) -> None:
        self.close()
        self.unlink_all()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _locate_row(self, info_set: str) -> tuple:
        """Look up or allocate *info_set* and return (chunk_id, local_row)."""
        flat_row, is_new = self._index.get_or_create(info_set)
        if is_new:
            with self._n_allocated.get_lock():
                self._n_allocated.value += 1
        chunk_id, local_row = divmod(flat_row, CHUNK_SIZE)
        self._store.ensure_open(chunk_id)
        return chunk_id, local_row
