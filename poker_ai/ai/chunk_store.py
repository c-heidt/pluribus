"""Shared-memory chunk storage with incremental dirty tracking.

Manages mmap-backed chunk files in ``shm_dir`` (typically ``/dev/shm``).
Each chunk is a ``(CHUNK_SIZE, n_actions)`` int32 array.  All processes
that map the same file share the same physical pages — a write in one
process is immediately visible in every other.

Dirty tracking records which chunks have been modified since the last
checkpoint.  Only dirty chunks are written to the (slow) network
workspace on HPC clusters, making incremental saves fast.

Must be instantiated in the **parent process before workers are forked**
so that ``_alloc_lock`` (a POSIX semaphore) is shared across processes.
"""

import logging
import math
import mmap
import multiprocessing as mp
import os
from pathlib import Path
from typing import List, Set

import numpy as np

from poker_ai.utils.io import atomic_numpy_save

log = logging.getLogger("poker_ai.ai.chunk_store")

CHUNK_SIZE: int = 1_000_000
"""Rows per chunk.  All chunks except the last are full."""

_MAX_DIRTY_CHUNKS: int = 1024
"""Upper bound on dirty-flag tracking."""


class ChunkStore:
    """Mmap-backed chunk file manager with dirty tracking.

    Parameters
    ----------
    table_name:
        Unique prefix for shared-memory file names
        (``{table_name}_{chunk_id:06d}``).
    n_actions:
        Column width of each chunk (number of actions per infoset).
    shm_dir:
        Directory for chunk files (default ``/dev/shm``).
    """

    def __init__(
        self,
        table_name: str,
        n_actions: int,
        shm_dir: str = "/dev/shm",
    ) -> None:
        self._table_name = table_name
        self._n_actions = n_actions
        self._shm_dir = shm_dir

        self._shm_mmaps: List[mmap.mmap] = []
        self._chunks: List[np.ndarray] = []
        self._shm_paths: List[str] = []

        self._alloc_lock: mp.Lock = mp.Lock()

        # Dirty tracking — simple shared byte array visible from all processes.
        self._dirty: mp.Array = mp.Array("b", _MAX_DIRTY_CHUNKS, lock=False)

    # ------------------------------------------------------------------
    # Chunk lifecycle
    # ------------------------------------------------------------------

    def ensure_open(self, chunk_id: int) -> None:
        """Ensure this process has *chunk_id* open, creating the file if needed.

        Fast path (no lock): chunk already attached in this process.
        Slow path (alloc lock): create file with ``O_CREAT | O_EXCL``, mmap it.
        """
        if chunk_id < len(self._chunks):
            return
        with self._alloc_lock:
            while len(self._chunks) <= chunk_id:
                self._open_or_create(len(self._chunks))

    def view(self, chunk_id: int) -> np.ndarray:
        """Return the full ``(CHUNK_SIZE, n_actions)`` shared-memory view."""
        self.ensure_open(chunk_id)
        return self._chunks[chunk_id]

    # ------------------------------------------------------------------
    # Dirty tracking
    # ------------------------------------------------------------------

    def mark_dirty(self, chunk_id: int) -> None:
        """Mark *chunk_id* as modified since the last checkpoint."""
        if chunk_id < _MAX_DIRTY_CHUNKS:
            self._dirty[chunk_id] = 1

    def clear_dirty(self) -> None:
        """Clear all dirty flags after a successful checkpoint."""
        for i in range(_MAX_DIRTY_CHUNKS):
            self._dirty[i] = 0

    # ------------------------------------------------------------------
    # Checkpoint serialization
    # ------------------------------------------------------------------

    def save_dirty(self, dir_path: Path, n_entries: int, prefix: str) -> int:
        """Save only dirty chunks to *dir_path*.  Returns count written."""
        return self._save(dir_path, n_entries, prefix, dirty_only=True)

    def save_all(self, dir_path: Path, n_entries: int, prefix: str) -> int:
        """Save all allocated chunks to *dir_path*.  Returns count written."""
        return self._save(dir_path, n_entries, prefix, dirty_only=False)

    def _save(self, dir_path: Path, n_entries: int, prefix: str, dirty_only: bool) -> int:
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
        """Write *data* into the shared memory for *chunk_id*."""
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
        """Number of chunks opened in this process."""
        return len(self._chunks)

    @property
    def paths(self) -> List[str]:
        """Paths of all chunk files created by this store."""
        return list(self._shm_paths)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all mmap handles (does NOT unlink files)."""
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
        """Remove all chunk files from the filesystem."""
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
        return os.path.join(self._shm_dir, f"{self._table_name}_{chunk_id:06d}")

    def _chunk_size_bytes(self) -> int:
        return CHUNK_SIZE * self._n_actions * np.dtype(np.int32).itemsize

    def _open_or_create(self, chunk_id: int) -> None:
        """Create or attach a chunk file.  Must hold ``_alloc_lock``."""
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
