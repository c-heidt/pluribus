"""Shared-memory chunk storage for CFR regret and strategy tables.

A :class:`ChunkStore` owns a set of mmap-backed files in a
shared-memory directory (typically ``/dev/shm`` on Linux) and exposes
each file as a ``(CHUNK_SIZE, n_actions)`` int32 array.  Because the
files are mmapped with ``MAP_SHARED``, any write from any process
that has the same file mapped is immediately visible in every other
mapping — this is how CFR workers share regret and strategy state
without IPC.

The store additionally tracks which chunks have been modified since
the last checkpoint.  On HPC clusters the checkpoint target is a slow
network filesystem, so only the dirty chunks are written at each
checkpoint; the rest are carried forward as hardlinks in the
:class:`~poker_ai.ai.checkpoint.CheckpointManager`.

The store is agnostic to the meaning of the rows — it neither knows
about information sets nor about the CFR tables that compose it.
Higher layers (:class:`~poker_ai.ai.chunked_table.ChunkedTable`,
:class:`~poker_ai.ai.cfr_tables.CFRTables`) assign semantics.

Process-safety notes
--------------------
``ChunkStore`` must be instantiated in the **parent process before
workers are forked** so that its allocation lock (a POSIX semaphore
created via :class:`multiprocessing.Lock`) and its shared dirty-flag
array are inherited by every worker.  Chunks created after the fork
are visible in other processes as soon as they call
:meth:`ensure_open`.
"""

import logging
import math
import mmap
import multiprocessing as mp
import os
from pathlib import Path
from typing import List

import numpy as np

from poker_ai.utils.io import atomic_numpy_save

log = logging.getLogger("poker_ai.ai.chunk_store")

CHUNK_SIZE: int = 1_000_000
"""Number of rows in a single chunk.

All chunks except the last are full.  Sizing is a trade-off between
filesystem overhead on checkpoint writes (fewer larger files are
cheaper on Lustre/GPFS) and the amount of memory that has to be
re-mmapped when a new infoset row overflows into a new chunk.
"""

_MAX_DIRTY_CHUNKS: int = 1024
"""Upper bound on the number of tracked chunks for dirty flags.

Sized so that the shared dirty array is a cheap inheritance across
fork.  Chunks beyond this index are treated as "always dirty" when
saved (they simply fall out of the dirty-only optimisation path).
"""


class ChunkStore:
    """Mmap-backed chunk file manager with dirty tracking.

    A :class:`ChunkStore` owns the shared-memory file layout for one
    regret or strategy table.  It does not know about information sets
    — it maps chunk IDs to shared numpy arrays and nothing else.

    Each chunk file is named ``{table_name}_{chunk_id:06d}`` inside
    ``shm_dir``.  Chunks are created lazily: a chunk file only exists
    on disk once some process has called :meth:`ensure_open` with its
    index.  The first caller to open a given chunk creates the file
    with ``O_CREAT | O_EXCL``, truncates it to the expected size, and
    mmaps it.  Subsequent callers (including workers that hit the
    chunk post-fork) attach to the existing file via ``os.O_RDWR``.

    Attributes
    ----------
    n_open : int
        Number of chunks that have been opened in the current process.
    paths : list[str]
        Paths of all chunk files created or attached by this store in
        the current process.
    """

    def __init__(
        self,
        table_name: str,
        n_actions: int,
        shm_dir: str = "/dev/shm",
    ) -> None:
        """Initialise a chunk store but do not create any files yet.

        Parameters
        ----------
        table_name : str
            Unique file-name prefix for this store.  All chunks are
            created as ``{shm_dir}/{table_name}_{chunk_id:06d}``.
        n_actions : int
            Column width of each chunk row (number of abstract actions
            for the betting round this table represents).
        shm_dir : str, optional
            Directory in which to place chunk files.  Defaults to
            ``/dev/shm`` (tmpfs on Linux).
        """
        self._table_name = table_name
        self._n_actions = n_actions
        self._shm_dir = shm_dir

        self._shm_mmaps: List[mmap.mmap] = []
        self._chunks: List[np.ndarray] = []
        self._shm_paths: List[str] = []

        self._alloc_lock: mp.Lock = mp.Lock() # type: ignore

        # Dirty tracking — shared byte array visible from all processes.
        self._dirty: mp.Array = mp.Array("b", _MAX_DIRTY_CHUNKS, lock=False) # type: ignore

    # ------------------------------------------------------------------
    # Chunk lifecycle
    # ------------------------------------------------------------------

    def ensure_open(self, chunk_id: int) -> None:
        """Ensure the current process has *chunk_id* mapped.

        Fast path: the chunk is already open in this process, return
        immediately.  Slow path: acquire the allocation lock and
        create or attach every chunk up to and including *chunk_id*.
        The slow path is safe across processes because the lock is
        inherited via fork.

        Parameters
        ----------
        chunk_id : int
            Zero-based chunk index.
        """
        if chunk_id < len(self._chunks):
            return
        with self._alloc_lock:
            while len(self._chunks) <= chunk_id:
                self._open_or_create(len(self._chunks))

    def view(self, chunk_id: int) -> np.ndarray:
        """Return the full ``(CHUNK_SIZE, n_actions)`` view for *chunk_id*.

        Opens the chunk first if it is not yet attached in this
        process.  The returned array is a view over shared memory —
        writes are visible to all other processes that have the same
        file mapped.

        Parameters
        ----------
        chunk_id : int
            Zero-based chunk index.

        Returns
        -------
        np.ndarray
            Int32 array of shape ``(CHUNK_SIZE, n_actions)``.
        """
        self.ensure_open(chunk_id)
        return self._chunks[chunk_id]

    # ------------------------------------------------------------------
    # Dirty tracking
    # ------------------------------------------------------------------

    def mark_dirty(self, chunk_id: int) -> None:
        """Mark *chunk_id* as modified since the last checkpoint.

        Writes are absorbed silently for chunks beyond
        :data:`_MAX_DIRTY_CHUNKS`; those chunks fall out of the
        dirty-only optimisation and are instead saved unconditionally
        when :meth:`save_dirty` is called.

        Parameters
        ----------
        chunk_id : int
            Zero-based chunk index.
        """
        if chunk_id < _MAX_DIRTY_CHUNKS:
            self._dirty[chunk_id] = 1

    def clear_dirty(self) -> None:
        """Clear every dirty flag.

        Called by the checkpoint manager after a successful save so
        that the next checkpoint only has to write chunks touched
        since this point.
        """
        for i in range(_MAX_DIRTY_CHUNKS):
            self._dirty[i] = 0

    # ------------------------------------------------------------------
    # Checkpoint serialisation
    # ------------------------------------------------------------------

    def save_dirty(self, dir_path: Path, n_entries: int, prefix: str) -> int:
        """Save only the chunks marked dirty since the last clear.

        Parameters
        ----------
        dir_path : Path
            Target directory.  One ``{prefix}_chunk_{chunk_id:06d}.npy``
            file is written per dirty chunk.
        n_entries : int
            Total number of valid rows across all chunks.  Used to
            compute the number of chunks in flight and to trim the
            last chunk to its valid prefix.
        prefix : str
            File-name prefix for the emitted ``.npy`` files (e.g.
            ``regret_0`` for the pre-flop regret table).

        Returns
        -------
        int
            Number of chunks written.
        """
        return self._save(dir_path, n_entries, prefix, dirty_only=True)

    def save_all(self, dir_path: Path, n_entries: int, prefix: str) -> int:
        """Save every allocated chunk regardless of dirty state.

        Used for final or emergency checkpoints where we want a
        self-contained snapshot even if the dirty-flag bookkeeping is
        stale or truncated.

        Parameters
        ----------
        dir_path : Path
            Target directory.
        n_entries : int
            Total number of valid rows across all chunks.
        prefix : str
            File-name prefix for the emitted ``.npy`` files.

        Returns
        -------
        int
            Number of chunks written.
        """
        return self._save(dir_path, n_entries, prefix, dirty_only=False)

    def _save(self, dir_path: Path, n_entries: int, prefix: str, dirty_only: bool) -> int:
        """Save chunks to *dir_path*.

        Shared implementation of :meth:`save_dirty` and
        :meth:`save_all`.  Iterates over every in-flight chunk
        (``chunk_id < ceil(n_entries / CHUNK_SIZE)``), optionally
        skipping the ones whose dirty flag is clear.  Each chunk is
        trimmed to its valid prefix before being atomically written to
        disk via :func:`poker_ai.utils.io.atomic_numpy_save`.
        """
        n_chunks = math.ceil(n_entries / CHUNK_SIZE) if n_entries > 0 else 0
        written = 0
        for chunk_id in range(n_chunks):
            if dirty_only and (chunk_id >= _MAX_DIRTY_CHUNKS or not self._dirty[chunk_id]):
                continue
            self.ensure_open(chunk_id)
            valid_rows = min(n_entries - chunk_id * CHUNK_SIZE, CHUNK_SIZE)
            atomic_numpy_save(
                self._chunks[chunk_id][:valid_rows].copy(),
                dir_path / f"{prefix}_chunk_{chunk_id:06d}.npy",
            )
            written += 1
        return written

    def restore(self, chunk_id: int, data: np.ndarray) -> None:
        """Copy *data* into the shared memory for *chunk_id*.

        Used by the checkpoint manager during resume to populate
        shared memory from the saved ``.npy`` files.  The chunk is
        opened first if needed.

        Parameters
        ----------
        chunk_id : int
            Zero-based chunk index.
        data : np.ndarray
            Array of shape ``(n_rows, n_actions)`` with
            ``n_rows <= CHUNK_SIZE``.  Cast to int32 before being
            written into the mmap.
        """
        self.ensure_open(chunk_id)
        n_rows = data.shape[0]
        if n_rows > CHUNK_SIZE:
            raise ValueError(f"data has {n_rows} rows but CHUNK_SIZE={CHUNK_SIZE}")
        self._chunks[chunk_id][:n_rows] = data.astype(np.int32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_open(self) -> int:
        """Number of chunks currently open in this process."""
        return len(self._chunks)

    @property
    def paths(self) -> List[str]:
        """Copy of the list of chunk-file paths created by this store."""
        return list(self._shm_paths)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close every open mmap handle without unlinking the files.

        Safe to call multiple times.  Closing the mmap does not delete
        the backing file — call :meth:`unlink_all` to remove the
        ``/dev/shm`` entries.
        """
        for mm in self._shm_mmaps:
            try:
                mm.close()
            except Exception as exc:
                log.warning("Failed to close mmap: %s", exc)
        self._shm_mmaps.clear()
        self._chunks.clear()
        log.info(
            "ChunkStore closed: %s (%d chunks)", self._table_name, len(self._shm_paths)
        )

    def unlink_all(self) -> None:
        """Remove every chunk file this store has created or attached.

        Idempotent: missing files are silently ignored.  Call after
        :meth:`close` at program shutdown to release the shared-memory
        allocation.
        """
        for path in self._shm_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except Exception as exc:
                log.warning("Failed to unlink %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _chunk_path(self, chunk_id: int) -> str:
        """Build the absolute path of the file backing *chunk_id*."""
        return os.path.join(self._shm_dir, f"{self._table_name}_{chunk_id:06d}")

    def _chunk_size_bytes(self) -> int:
        """Size in bytes of one chunk file (fixed at ``CHUNK_SIZE * n_actions * 4``)."""
        return CHUNK_SIZE * self._n_actions * np.dtype(np.int32).itemsize

    def _open_or_create(self, chunk_id: int) -> None:
        """Create or attach the chunk file for *chunk_id*.

        Must be called while holding ``_alloc_lock``.  The first
        process to reach this code path creates the file with
        ``O_CREAT | O_EXCL``, truncates it to the expected size, and
        mmaps it.  Any process that arrives later finds the file
        already present and attaches via ``os.O_RDWR`` — both paths
        produce equivalent numpy views backed by the same physical
        pages.
        """
        assert len(self._chunks) == chunk_id

        path = self._chunk_path(chunk_id)
        size_bytes = self._chunk_size_bytes()

        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, size_bytes)
            log.debug("Created chunk file: %s", path)
        except FileExistsError:
            fd = os.open(path, os.O_RDWR)
            log.debug("Attached to existing chunk file: %s", path)

        mm = mmap.mmap(fd, size_bytes, mmap.MAP_SHARED)
        os.close(fd)

        flat = np.frombuffer(mm, dtype=np.int32)
        chunk = flat.reshape(CHUNK_SIZE, self._n_actions)

        self._shm_mmaps.append(mm)
        self._chunks.append(chunk)
        self._shm_paths.append(path)
