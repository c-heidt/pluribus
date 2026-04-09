"""Persistent information-set index backed by LMDB.

A :class:`InfosetIndex` is a pure string-to-integer mapping: given an
information-set string it returns a stable 64-bit "flat row number"
that the caller uses to locate the infoset's regret or strategy row
inside the shared-memory chunk store.  The index has no knowledge of
chunks, streets, or CFR tables — it is the one place in the codebase
where the ``infoset → row`` relation lives.

The mapping is persistent: opening an existing LMDB directory
resumes training with every previously-allocated row intact.  Keys
are 16-byte raw digests produced by
:func:`poker_ai.utils.io.hash_info_set_bytes` so the on-disk format
is insensitive to the length of the original infoset string.  Values
are packed little-endian uint64s.  A reserved metadata key
``b"\\x00__next_row__"`` stores the next-row watermark that drives
row allocation.

Debug mode
----------
When the ``POKER_AI_DEBUG`` environment variable is set (or
``debug=True`` is passed to the constructor), each insertion also
writes the original infoset string under a shadow key
(``b"\\x00__str__" + digest``).  Subsequent insertions compare
against the shadow entry to detect 128-bit hash collisions.  This
roughly doubles LMDB storage and is intended for development only.

Process-fork safety
-------------------
LMDB reader slots are not safe to share across a fork — a child
process that re-enters an inherited environment will hit
``MDB_BAD_RSLOT``.  Workers must therefore call
:meth:`reopen_after_fork` at the top of their ``run()`` method before
starting any LMDB transaction.  The cached row count
``n_allocated_rows`` is served from a :class:`multiprocessing.Value`
that is initialised from LMDB once in the parent process so that
post-fork readers never need to open a transaction to learn the
current row count.
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
    """Return a sensible LMDB ``map_size`` for the given player count.

    LMDB allocates the full ``map_size`` as a sparse file on Linux, so
    oversizing is cheap — the file grows only when real data is
    written.  The returned value is a pragmatic default based on how
    large the infoset space actually gets; users can override it via
    the ``PLURIBUS_LMDB_MAP_SIZE`` environment variable.

    Parameters
    ----------
    n_players : int
        Number of players in the game being trained.

    Returns
    -------
    int
        Recommended LMDB ``map_size`` in bytes.
    """
    if n_players <= 2:
        return 1 * 1024 ** 3   # 1 GiB
    return 50 * 1024 ** 3      # 50 GiB


def _open_lmdb(path: str, **kwargs):
    """Open an LMDB environment, tolerating minor version differences.

    Different installed ``lmdb`` Python bindings expose either
    ``lmdb.open`` or ``lmdb.Environment``; this helper picks whichever
    is available and raises a friendly error when neither is.

    Parameters
    ----------
    path : str
        LMDB directory.
    **kwargs
        Forwarded to the environment constructor.
    """
    opener = getattr(lmdb, "open", None) or getattr(lmdb, "Environment", None)
    if opener is None:
        raise RuntimeError(
            f"lmdb installation is broken — neither 'open' nor 'Environment' found. "
            f"Version: {getattr(lmdb, '__version__', 'unknown')}, attrs: {dir(lmdb)}. "
            "Please reinstall: pip install 'lmdb>=1.0.0'"
        )
    return opener(path, **kwargs)


class InfosetIndex:
    """Persistent infoset → flat row number mapping.

    The index is opened once in the parent process and inherited by
    workers through fork.  Each worker must call
    :meth:`reopen_after_fork` before its first transaction.  Reads of
    :attr:`n_allocated_rows` go through a
    :class:`multiprocessing.Value` and therefore do not require an
    LMDB transaction, making them safe immediately after fork.

    Attributes
    ----------
    n_allocated_rows : int
        Total number of rows that have ever been allocated.  Equals
        the ``__next_row__`` watermark persisted inside LMDB and is
        mirrored in a shared :class:`multiprocessing.Value` for
        lock-free reads from any process.
    """

    def __init__(
        self,
        path: Union[str, Path],
        debug: bool = False,
        map_size: Optional[int] = None,
    ) -> None:
        """Open or create an LMDB-backed index at *path*.

        Parameters
        ----------
        path : str or Path
            Directory for the LMDB data and lock files.  Created if
            absent.
        debug : bool, optional
            Enable hash-collision detection via shadow keys.  Also
            triggered by setting the ``POKER_AI_DEBUG`` environment
            variable.  Development only — roughly doubles storage.
        map_size : int, optional
            LMDB ``map_size`` reservation in bytes.  Defaults to
            ``PLURIBUS_LMDB_MAP_SIZE`` or a 10 GiB fallback.  The file
            is sparse on Linux, so oversizing is cheap.
        """
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
        # (pre-fork, safe to read LMDB) so any process can query the row
        # count without an LMDB transaction (which triggers MDB_BAD_RSLOT
        # post-fork).
        self._n_allocated_mp: mp.Value = mp.Value("Q", self._read_next_row()) # type: ignore

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
        """Total number of rows allocated so far, lock-free safe to read."""
        return self._n_allocated_mp.value

    def get(self, info_set: str) -> Optional[int]:
        """Look up *info_set* and return its flat row number.

        Parameters
        ----------
        info_set : str
            Information-set string (typically a JSON-encoded
            cluster+history).

        Returns
        -------
        int or None
            Flat row number if *info_set* was previously allocated,
            otherwise ``None``.  Callers that need allocate-on-miss
            should use :meth:`get_or_create` instead.
        """
        key = hash_info_set_bytes(info_set)
        with self._env.begin() as txn:
            val = txn.get(key)
        if val is None:
            return None
        return struct.unpack("<Q", val)[0]

    def get_or_create(self, info_set: str) -> tuple:
        """Return ``(flat_row, is_new)`` for *info_set*, allocating on miss.

        The allocation is performed inside a single write transaction
        so concurrent callers observe a consistent mapping — LMDB
        serialises writers at the environment level.  On
        :class:`~lmdb.MapFullError`, the method automatically doubles
        the ``map_size`` and retries.

        Parameters
        ----------
        info_set : str
            Information-set string.

        Returns
        -------
        tuple[int, bool]
            ``(flat_row, is_new)`` where ``is_new`` is ``True`` iff
            the row was allocated by this call.
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
        """Reopen the LMDB environment in the current (forked) process.

        Must be the first LMDB call a worker makes after
        :meth:`multiprocessing.Process.run` starts.  Reusing an
        inherited environment across a fork triggers
        ``MDB_BAD_RSLOT``; this method closes the inherited handle
        and opens a fresh one bound to the current process's reader
        slot.
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
        """Force pending LMDB writes to disk.

        Called by the checkpoint manager before serialising the
        chunk data so that the on-disk LMDB is guaranteed to be
        consistent with the shared-memory table contents.
        """
        try:
            self._env.sync(True)
        except TypeError:
            self._env.sync()
        log.debug("InfosetIndex flushed to %s", self._path)

    def close(self) -> None:
        """Flush and close the LMDB environment.

        Safe to call multiple times — subsequent calls are no-ops
        because LMDB's own ``close`` is idempotent.
        """
        self.flush()
        self._env.close()
        log.info("InfosetIndex closed (%s)", self._path)

    def __enter__(self) -> "InfosetIndex":
        """Enter a context-manager scope; returns ``self``."""
        return self

    def __exit__(self, *_) -> None:
        """Exit the context manager, closing the LMDB environment."""
        self.close()

    def __repr__(self) -> str:
        """Debug representation showing path, row count, and debug flag."""
        return (
            f"InfosetIndex(path={self._path!r}, "
            f"n_allocated_rows={self.n_allocated_rows}, "
            f"debug={self._debug})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_next_row(self) -> int:
        """Read the ``__next_row__`` watermark from LMDB.

        Called only in the parent process at construction time so the
        row count can be cached in the shared
        :class:`multiprocessing.Value` before workers fork.
        """
        with self._env.begin() as txn:
            raw = txn.get(_NEXT_ROW_KEY)
        if raw is None:
            return 0
        return struct.unpack("<Q", raw)[0]

    def _reopen(self) -> None:
        """Double ``map_size`` and reopen the environment.

        Called from :meth:`get_or_create` when a write hits
        :class:`~lmdb.MapFullError`.  The new environment starts
        serving transactions immediately; any caller that races with
        the reopen simply retries.
        """
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
        """Single-shot get-or-create, wrapped by the retry loop.

        Runs the entire get-or-allocate logic inside one LMDB write
        transaction so concurrent allocators for the same infoset
        cannot race and create duplicate rows — LMDB blocks the
        second writer until the first commits, at which point the
        second sees the freshly-inserted row on ``txn.get``.
        """
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
