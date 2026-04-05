"""Persistent infoset → flat row number index backed by LMDB.

A pure string → int mapping.  The index has no knowledge of chunks or
shared-memory storage — callers decompose the flat row number into
(chunk_id, local_row) themselves.

Keys
    16-byte raw digest produced by ``hash_info_set_bytes()``.

Values
    8 bytes: ``struct.pack("<Q", row_number)`` (little-endian uint64).

Counter
    A special key ``b"\\x00__next_row__"`` stores the next-row watermark.

Collision detection (debug mode)
    When ``POKER_AI_DEBUG`` is set, every insertion stores the original
    string under a shadow key to detect hash collisions.  Roughly doubles
    storage — development only.
"""

import logging
import multiprocessing as mp
import os
import struct
from pathlib import Path
from typing import Optional, Union

import lmdb

from poker_ai.utils.io import hash_info_set_bytes

log = logging.getLogger("poker_ai.ai.index")

_DEFAULT_MAP_SIZE: int = 10 * 1024 ** 3  # 10 GiB fallback
_MAP_SIZE: int = int(os.environ.get("PLURIBUS_LMDB_MAP_SIZE", _DEFAULT_MAP_SIZE))

_NEXT_ROW_KEY: bytes = b"\x00__next_row__"
_STR_PREFIX: bytes = b"\x00__str__"


def lmdb_map_size_for_players(n_players: int) -> int:
    """Return an appropriate LMDB map_size for the given player count.

    Both values create a sparse file on Linux — real usage is much smaller
    than the reservation.
    """
    if n_players <= 2:
        return 1 * 1024 ** 3   # 1 GiB
    return 50 * 1024 ** 3      # 50 GiB


def _open_lmdb(path: str, **kwargs):
    """Open an LMDB environment, handling API differences across versions."""
    opener = getattr(lmdb, "open", None) or getattr(lmdb, "Environment", None)
    if opener is None:
        raise RuntimeError(
            f"lmdb installation is broken — neither 'open' nor 'Environment' found. "
            f"Version: {getattr(lmdb, '__version__', 'unknown')}, attrs: {dir(lmdb)}. "
            "Please reinstall: pip install 'lmdb>=1.0.0'"
        )
    return opener(path, **kwargs)


class InfosetIndex:
    """Persistent mapping from infoset string to flat row number.

    The index survives restarts: open an existing LMDB directory to resume
    and all previous allocations are recovered.

    Parameters
    ----------
    path:
        Directory for LMDB data files.  Created if absent.
    debug:
        Enable hash collision detection (slow, doubles storage).
    map_size:
        LMDB map_size reservation in bytes.  Sparse on Linux.
    """

    def __init__(
        self,
        path: Union[str, Path],
        debug: bool = False,
        map_size: Optional[int] = None,
    ) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._debug: bool = debug or bool(os.environ.get("POKER_AI_DEBUG", False))
        resolved_map_size = map_size if map_size is not None else _MAP_SIZE
        self._env: lmdb.Environment = _open_lmdb(
            str(self._path),
            map_size=resolved_map_size,
            writemap=True,
            map_async=True,
            max_readers=256,
        )
        self._map_size: int = resolved_map_size

        # Shared counter mirroring __next_row__ in LMDB.  Initialised here
        # (pre-fork, safe to read LMDB) so any process can query it without
        # an LMDB transaction (which triggers MDB_BAD_RSLOT post-fork).
        self._n_allocated_mp: mp.Value = mp.Value("Q", self._read_next_row())

        log.info(
            "InfosetIndex opened at %s (debug=%s, map_size=%d GiB, n_entries=%d). "
            "data.mdb apparent size = map_size (sparse file); check real usage with: "
            "du -sh %s/data.mdb",
            self._path,
            self._debug,
            resolved_map_size // 1024 ** 3,
            self.n_allocated_rows,
            self._path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_allocated_rows(self) -> int:
        """Total number of allocated rows (safe to read from any process)."""
        return self._n_allocated_mp.value

    def get(self, info_set: str) -> Optional[int]:
        """Look up *info_set* and return its flat row number, or ``None``."""
        key = hash_info_set_bytes(info_set)
        with self._env.begin() as txn:
            val = txn.get(key)
        if val is None:
            return None
        return struct.unpack("<Q", val)[0]

    def get_or_create(self, info_set: str) -> tuple:
        """Return ``(flat_row, is_new)`` for *info_set*, allocating if new.

        Retries automatically on ``MapFullError`` (doubles the map_size).
        """
        while True:
            try:
                return self._get_or_create_once(info_set)
            except lmdb.MapFullError:
                self._reopen()

    # ------------------------------------------------------------------
    # Post-fork
    # ------------------------------------------------------------------

    def reopen_after_fork(self) -> None:
        """Reopen the LMDB environment in a forked child process.

        Workers must call this at the start of ``run()`` before any LMDB
        transaction to avoid ``MDB_BAD_RSLOT``.
        """
        try:
            self._env.close()
        except Exception:
            pass
        self._env = _open_lmdb(
            str(self._path),
            map_size=_MAP_SIZE,
            writemap=True,
            map_async=True,
            max_readers=256,
        )
        log.debug("InfosetIndex reopened after fork (pid=%d)", os.getpid())

    # ------------------------------------------------------------------
    # Flush / close
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Force all pending writes to disk."""
        try:
            self._env.sync(True)
        except TypeError:
            self._env.sync()
        log.debug("InfosetIndex flushed to %s", self._path)

    def close(self) -> None:
        """Flush and close the LMDB environment."""
        self.flush()
        self._env.close()
        log.info("InfosetIndex closed (%s)", self._path)

    def __enter__(self) -> "InfosetIndex":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"InfosetIndex(path={self._path!r}, "
            f"n_allocated_rows={self.n_allocated_rows}, "
            f"debug={self._debug})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_next_row(self) -> int:
        """Read the __next_row__ counter from LMDB (pre-fork only)."""
        with self._env.begin() as txn:
            raw = txn.get(_NEXT_ROW_KEY)
        if raw is None:
            return 0
        return struct.unpack("<Q", raw)[0]

    def _reopen(self) -> None:
        """Double map_size and reopen (called on MapFullError)."""
        self._env.close()
        self._map_size *= 2
        log.warning(
            "LMDB map full — reopening %s with map_size=%d GiB",
            self._path,
            self._map_size // 1024 ** 3,
        )
        self._env = _open_lmdb(
            str(self._path),
            map_size=self._map_size,
            writemap=True,
            map_async=True,
            max_readers=256,
        )

    def _get_or_create_once(self, info_set: str) -> tuple:
        key = hash_info_set_bytes(info_set)

        with self._env.begin(write=True) as txn:
            val = txn.get(key)
            if val is not None:
                row = struct.unpack("<Q", val)[0]
                return row, False

            # Allocate a new row.
            meta = txn.get(_NEXT_ROW_KEY)
            next_row: int = struct.unpack("<Q", meta)[0] if meta else 0

            txn.put(key, struct.pack("<Q", next_row))
            txn.put(_NEXT_ROW_KEY, struct.pack("<Q", next_row + 1))
            with self._n_allocated_mp.get_lock():
                self._n_allocated_mp.value = next_row + 1

            if self._debug:
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

        return next_row, True
