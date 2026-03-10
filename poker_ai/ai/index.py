"""Persistent infoset → (chunk_id, row) index backed by LMDB.

Phase 1.3 of the training pipeline refactor.

Design notes
------------
Keys
    16-byte raw digest produced by ``hash_info_set_bytes()``.  Using the raw
    bytes avoids variable-length string storage and keeps individual LMDB
    entries at a fixed size.

Values
    8 bytes: ``struct.pack("<II", chunk_id, row)``

Counter
    A special key ``b"__next_row__"`` stores the global next-row counter as a
    little-endian uint64.  This is the single source of truth for row
    allocation so that ``SparseRegretTable`` (Phase 2) does not need to
    maintain a separate counter.

Collision detection (debug mode)
    When the environment variable ``POKER_AI_DEBUG`` is set, every new
    insertion checks whether the hash key already maps to a *different*
    infoset string.  The original string is stored under a shadow key
    prefixed with ``b"__str__"`` for this purpose.  This roughly doubles
    LMDB storage usage, so it must only be enabled during development.

map_size
    Set to 50 GiB.  LMDB does not pre-allocate disk space; the actual file
    grows on demand.  50 GiB is a safe upper bound for a 6-player game with
    aggressive abstraction.
"""

import logging
import os
import struct
from pathlib import Path
from typing import Optional, Tuple, Union

import lmdb

from poker_ai.utils.io import hash_info_set_bytes

log = logging.getLogger("poker_ai.ai.index")

# Must match SparseRegretTable.CHUNK_SIZE (set consistently in Phase 2).
CHUNK_SIZE: int = 100_000

_MAP_SIZE: int = 50 * 1024 ** 3  # 50 GiB — does not pre-allocate

# Special LMDB keys (prefixed with null byte to avoid clash with hash keys).
_NEXT_ROW_KEY: bytes = b"\x00__next_row__"
_STR_PREFIX: bytes = b"\x00__str__"


class InfosetIndex:
    """Persistent mapping from infoset string to ``(chunk_id, row)`` location.

    The index survives process restarts: open an existing LMDB directory to
    resume from a previous training checkpoint.  Pass ``path`` to the same
    directory on resume and all previous allocations will be recovered.

    Parameters
    ----------
    path:
        Directory where LMDB stores its data files.  Created if it does not
        exist.
    debug:
        When ``True``, every new insertion validates that the hash key does not
        already map to a different infoset string.  Overrides the
        ``POKER_AI_DEBUG`` environment variable.  Significantly slower — only
        use during development.

    Attributes
    ----------
    n_entries : int
        Number of infoset entries currently stored (excludes metadata keys).
    """

    def __init__(self, path: Union[str, Path], debug: bool = False):
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._debug: bool = debug or bool(os.environ.get("POKER_AI_DEBUG", False))
        self._env: lmdb.Environment = lmdb.open(
            str(self._path),
            map_size=_MAP_SIZE,
            writemap=True,
            map_async=True,
            # Allow multiple readers from different processes so that workers
            # can look up existing entries without acquiring a write lock.
            max_readers=256,
        )
        log.info(
            "InfosetIndex opened at %s (debug=%s, n_entries=%d)",
            self._path,
            self._debug,
            self.n_entries,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_entries(self) -> int:
        """Return the total number of infoset entries (not counting metadata)."""
        with self._env.begin() as txn:
            raw = txn.get(_NEXT_ROW_KEY)
        if raw is None:
            return 0
        return struct.unpack("<Q", raw)[0]

    def get(self, info_set: str) -> Optional[Tuple[int, int]]:
        """Look up *info_set* and return its ``(chunk_id, row)`` or ``None``.

        This is a read-only operation and is safe to call from multiple
        concurrent readers.
        """
        key = hash_info_set_bytes(info_set)
        with self._env.begin() as txn:
            val = txn.get(key)
        if val is None:
            return None
        chunk_id, row = struct.unpack("<II", val)
        return chunk_id, row

    def get_or_create(self, info_set: str) -> Tuple[Tuple[int, int], bool]:
        """Return the ``(chunk_id, row)`` for *info_set*, allocating if new.

        LMDB provides single-writer semantics at the native level: this method
        uses an exclusive write transaction so that concurrent calls from
        different processes are serialised correctly without any additional
        Python-level lock.

        Returns
        -------
        location : Tuple[int, int]
            ``(chunk_id, row)`` where this infoset's data lives.
        is_new : bool
            ``True`` if a new entry was allocated, ``False`` if it already
            existed.
        """
        key = hash_info_set_bytes(info_set)

        # Single write transaction for check-and-insert atomicity.
        with self._env.begin(write=True) as txn:
            val = txn.get(key)
            if val is not None:
                chunk_id, row = struct.unpack("<II", val)
                return (chunk_id, row), False

            # Allocate a new row.
            meta = txn.get(_NEXT_ROW_KEY)
            next_global_row: int = struct.unpack("<Q", meta)[0] if meta else 0

            chunk_id = next_global_row // CHUNK_SIZE
            row = next_global_row % CHUNK_SIZE
            location = (chunk_id, row)

            packed_val = struct.pack("<II", chunk_id, row)
            txn.put(key, packed_val)
            txn.put(_NEXT_ROW_KEY, struct.pack("<Q", next_global_row + 1))

            if self._debug:
                # Store original string under shadow key to detect future
                # collisions — any hash that already maps to a value but
                # reaches this branch means a second hash collision was
                # inserted separately, which is fine.  What we guard against
                # is the same key byte sequence appearing from *two distinct
                # info_set strings*.
                shadow_key = _STR_PREFIX + key
                existing_str = txn.get(shadow_key)
                if existing_str is not None and existing_str != info_set.encode():
                    raise AssertionError(
                        f"128-bit hash collision detected!\n"
                        f"  info_set A (existing): {existing_str.decode()!r}\n"
                        f"  info_set B (new):      {info_set!r}\n"
                        f"  hash key: {key.hex()}"
                    )
                txn.put(shadow_key, info_set.encode())

        return location, True

    def put(self, info_set: str, chunk_id: int, row: int) -> None:
        """Explicitly store a known mapping for *info_set*.

        This is an escape hatch for callers that manage their own row counter
        (e.g. when rebuilding the index from a checkpoint).  Prefer
        ``get_or_create`` for normal use.

        Raises
        ------
        ValueError
            If *info_set* is already present in the index at a *different*
            location, indicating an inconsistent state.
        """
        key = hash_info_set_bytes(info_set)
        packed_val = struct.pack("<II", chunk_id, row)
        with self._env.begin(write=True) as txn:
            existing = txn.get(key)
            if existing is not None:
                ex_chunk, ex_row = struct.unpack("<II", existing)
                if (ex_chunk, ex_row) != (chunk_id, row):
                    raise ValueError(
                        f"InfosetIndex.put conflict: {info_set!r} already maps to "
                        f"({ex_chunk}, {ex_row}), attempted to overwrite with "
                        f"({chunk_id}, {row})."
                    )
                return  # idempotent
            txn.put(key, packed_val)
            # Advance counter if this row is beyond the current watermark.
            meta = txn.get(_NEXT_ROW_KEY)
            current = struct.unpack("<Q", meta)[0] if meta else 0
            global_row = chunk_id * CHUNK_SIZE + row
            if global_row >= current:
                txn.put(_NEXT_ROW_KEY, struct.pack("<Q", global_row + 1))

    def reopen_after_fork(self) -> None:
        """Reopen the LMDB environment in a forked child process.

        LMDB registers reader lock-table slots per-PID.  When the parent opens
        the environment and then forks workers, the children inherit the
        file descriptor with a slot tied to the *parent's* PID.  Any
        transaction attempt in the child raises ``MDB_BAD_RSLOT``.

        Call this method at the very beginning of ``Worker.run()`` (before any
        LMDB transaction) to close the inherited handle and open a fresh one
        for the child's PID.
        """
        try:
            self._env.close()
        except Exception:
            pass
        self._env = lmdb.open(
            str(self._path),
            map_size=_MAP_SIZE,
            writemap=True,
            map_async=True,
            max_readers=256,
        )
        log.debug("InfosetIndex reopened after fork (pid=%d)", os.getpid())

    def flush(self) -> None:
        """Force all pending writes to disk.

        Call this before writing a training checkpoint to ensure the index is
        consistent with the regret/strategy arrays that are being snapshotted.
        """
        # lmdb 0.9.x does not accept keyword arguments for sync(); pass True
        # positionally to request an fsync.  Older builds may not accept *any*
        # argument, so we fall back to a no-argument call on TypeError.
        try:
            self._env.sync(True)  # type: ignore[call-arg]
        except TypeError:
            self._env.sync()
        log.debug("InfosetIndex flushed to %s", self._path)

    def close(self) -> None:
        """Flush and close the LMDB environment.

        After calling ``close()`` no further operations on this instance are
        valid.
        """
        self.flush()
        self._env.close()
        log.info("InfosetIndex closed (%s)", self._path)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "InfosetIndex":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"InfosetIndex(path={self._path!r}, n_entries={self.n_entries}, "
            f"debug={self._debug})"
        )
