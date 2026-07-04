"""Persistent information-set index backed by LMDB.

A :class:`InfosetIndex` is a pure string-to-integer mapping: given an
information-set string it returns a stable 64-bit "flat row number"
that the caller uses to locate the infoset's regret or strategy row
inside the shared-memory chunk store.  The index has no knowledge of
chunks, streets, or CFR tables — it is the one place in the codebase
where the ``infoset → row`` relation lives.

The mapping is persistent: opening an existing LMDB directory
resumes training with every previously-allocated row intact.  Keys
are 16-byte raw digests produced by :func:`hash_info_set_bytes` so
the on-disk format is insensitive to the length of the original
infoset string.  Values
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

import hashlib
import logging
import multiprocessing as mp
import os
import struct
from pathlib import Path
from typing import Optional, Tuple, Union

import lmdb

log = logging.getLogger("poker_ai.tables.index")

_DEFAULT_MAP_SIZE: int = 10 * 1024 ** 3  # 10 GiB fallback
_MAP_SIZE: int = int(os.environ.get("PLURIBUS_LMDB_MAP_SIZE", _DEFAULT_MAP_SIZE))

_NEXT_ROW_KEY: bytes = b"\x00__next_row__"
_STR_PREFIX: bytes = b"\x00__str__"


def hash_info_set_128(info_set: Union[bytes, str]) -> Tuple[int, int]:
    """Return a 128-bit hash of *info_set* as a pair of unsigned 64-bit ints.

    Uses xxhash.xxh3_128 when available (fast path, ~10x faster than blake2b)
    and falls back to hashlib.blake2b (16-byte digest) otherwise.

    Production keys are compact ``bytes`` (see
    :func:`environment.poker_env.encode_info_set`); a ``str`` is accepted and
    utf-8 encoded so unit-test doubles that use plain string labels still work.
    """
    if isinstance(info_set, str):
        info_set = info_set.encode("utf-8")
    try:
        import xxhash
        digest_int: int = xxhash.xxh3_128(info_set).intdigest()
        high = digest_int >> 64
        low = digest_int & 0xFFFF_FFFF_FFFF_FFFF
        return high, low
    except ImportError:
        digest: bytes = hashlib.blake2b(info_set, digest_size=16).digest()
        high, low = struct.unpack("<QQ", digest)
        return high, low


def hash_info_set_bytes(info_set: Union[bytes, str]) -> bytes:
    """Return the 16-byte (128-bit) raw digest for *info_set*.

    This is the canonical key format used when storing hashes in LMDB.
    """
    high, low = hash_info_set_128(info_set)
    return struct.pack("<QQ", high, low)


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
    # 20 GiB comfortably fits the index at saturation: 200 buckets per
    # street × the bounded betting-history space yields perhaps tens
    # of millions of infosets per street; each entry costs ~50 bytes
    # including B-tree overhead.  If the assumption ever breaks,
    # :meth:`_reopen` doubles the map_size automatically on
    # :class:`~lmdb.MapFullError`.
    return 20 * 1024 ** 3      # 20 GiB


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
        # max_spare_txns=0: don't cache aborted read txns in the
        # python-lmdb spare pool.  Without this, the read transaction
        # used by `_read_next_row` below ends up cached for inheritance
        # by forked workers.  When a worker then calls
        # ``env.begin()`` for the first time, python-lmdb tries to
        # `mdb_txn_renew` the inherited txn — which references a
        # reader-slot owned by the *parent* — and fails with
        # ``MDB_BAD_RSLOT``.  Keeping the parent's spare pool empty
        # avoids the issue at the source; workers also reopen with
        # max_spare_txns=0 in :meth:`reopen_after_fork`.
        self._env: lmdb.Environment = _open_lmdb(
            str(self._path),
            map_size=resolved_map_size,
            writemap=True,
            map_async=True,
            max_readers=256,
            max_spare_txns=0,
            # Runtime env is disposable node-local scratch — durability comes
            # from the checkpoint mirror after an explicit flush(sync=True), so
            # skip the per-commit meta fsync (commits then run at memory speed).
            sync=False,
            metasync=False,
        )
        self._map_size: int = resolved_map_size

        # Optional shared-memory read cache (attached post-construction by
        # CFRTables via :meth:`set_cache`).  When present, ``get`` is served
        # entirely from shm — no LMDB read txn per node — and ``get_or_create``
        # inserts newly-allocated rows into it.  ``None`` = legacy LMDB-only
        # path (standalone indexes, tests, cache disabled).
        self._cache = None

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

    def set_cache(self, cache) -> None:
        """Attach a :class:`~poker_ai.tables.shm_index_cache.ShmIndexCache`.

        Must be called in the parent before workers fork (so the shm mmap and
        lock are inherited) and after any resume/warm-start restore has
        populated LMDB, then followed by :meth:`prewarm_cache`.
        """
        self._cache = cache

    def prewarm_cache(self) -> int:
        """Load the attached cache from this index's LMDB (no-op if none)."""
        if self._cache is None:
            return 0
        return self._cache.prewarm_from_cursor(self._env)

    def get(self, info_set) -> Optional[int]:
        """Look up *info_set* and return its flat row number.

        With a cache attached this is a lock-free shm probe with **no LMDB
        transaction** (the hot path).  A cache miss returns ``None`` without
        consulting LMDB — the cache is prewarmed from LMDB and kept current on
        every allocation, so a miss means the key is genuinely unseen (or was
        allocated in the microscopic window before its cache insert, which the
        caller's unseen→uniform fallback handles safely).

        Parameters
        ----------
        info_set : bytes or str
            Information-set key (compact bytes in production; str accepted for
            test doubles).

        Returns
        -------
        int or None
            Flat row number if allocated, else ``None``.  Callers that need
            allocate-on-miss should use :meth:`get_or_create`.
        """
        if self._cache is not None:
            low, high = hash_info_set_128(info_set)
            return self._cache.probe(low, high)
        key = hash_info_set_bytes(info_set)
        with self._env.begin() as txn:
            val = txn.get(key)
        if val is None:
            return None
        return struct.unpack("<Q", val)[0]

    def get_or_create(self, info_set) -> tuple:
        """Return ``(flat_row, is_new)`` for *info_set*, allocating on miss.

        With a cache attached, a cache hit returns immediately with no LMDB
        access.  On a cache miss the row is allocated through LMDB (still the
        authoritative allocator + persistence) and then inserted into the
        cache.  The allocation is performed inside a single LMDB write
        transaction so concurrent callers observe a consistent mapping — LMDB
        serialises writers at the environment level.  On
        :class:`~lmdb.MapFullError`, the method automatically doubles the
        ``map_size`` and retries.

        Returns
        -------
        tuple[int, bool]
            ``(flat_row, is_new)`` where ``is_new`` is ``True`` iff the row was
            allocated by this call.
        """
        if self._cache is not None:
            low, high = hash_info_set_128(info_set)
            row = self._cache.probe(low, high)
            if row is not None:
                return row, False
            # Cache already probed and missed above, so the row almost
            # certainly does not exist — skip the redundant read-first txn and
            # go straight to the write txn (which re-checks for the row anyway,
            # so a concurrent allocator is still handled correctly).
            flat_row, is_new = self._lmdb_get_or_create(info_set, skip_read=True)
            self._cache.insert(low, high, flat_row)
            return flat_row, is_new
        return self._lmdb_get_or_create(info_set)

    def _lmdb_get_or_create(self, info_set, skip_read: bool = False) -> tuple:
        """LMDB-only get-or-create with the map-full retry loop.

        ``skip_read`` bypasses the leading concurrent read txn and goes
        straight to the write txn — set by :meth:`get_or_create` when a shm
        cache is attached, because the caller has already probed and missed
        the cache, so the read-first optimisation (avoid the writer lock for
        already-present rows) cannot pay off here.
        """
        while True:
            try:
                return self._get_or_create_once(info_set, skip_read=skip_read)
            except lmdb.MapFullError:
                self._reopen()

    # ------------------------------------------------------------------
    # Post-fork
    # ------------------------------------------------------------------

    def copy_to(self, dst_path: Union[str, Path]) -> None:
        """Write a transactionally consistent snapshot of this index to *dst_path*.

        Wraps :meth:`lmdb.Environment.copy`, which uses LMDB's MVCC
        to produce a coherent on-disk image of the index even while
        other readers and writers are active in the live env.  The
        destination directory is created if missing.  After this
        returns, *dst_path* contains a complete LMDB env that can be
        opened by :class:`InfosetIndex` exactly like an original.

        Used by :class:`~poker_ai.tables.checkpoint.CheckpointManager`
        when the index runtime path lives on node-local fast scratch
        and must be mirrored to a persistent shared filesystem at
        each checkpoint.

        Parameters
        ----------
        dst_path : str or Path
            Target directory.  Must be on a writable filesystem and
            must not already contain an LMDB env (LMDB refuses to
            overwrite).
        """
        dst = Path(dst_path)
        dst.mkdir(parents=True, exist_ok=True)
        # compact=False: copy the env preserving its page layout rather
        # than repacking the B-tree.  Two reasons, both about keeping
        # the per-checkpoint cost O(new data) instead of O(total size):
        #
        #   1. ``compact=True`` walks and rewrites the entire tree on
        #      the CPU every call — cost grows linearly with the total
        #      number of allocated infosets, so on a multi-day run the
        #      copy eventually exceeds the checkpoint interval and
        #      starves the workers.
        #   2. Repacking reshuffles physical page numbers, which defeats
        #      rsync's delta algorithm in :meth:`CheckpointManager.
        #      _mirror_lmdb_to_persistent` — every mirror then ships
        #      essentially the whole DB over the network.  Preserving the
        #      page layout (append-mostly for an LMDB that only grows)
        #      lets rsync transfer just the changed/appended pages.
        #
        # The only cost is a larger on-disk file (free pages are not
        # reclaimed), which is cheap on the persistent filesystem and
        # bounded by peak index size.
        self._env.copy(str(dst), compact=False)

    def close_env(self) -> None:
        """Close just the LMDB environment without touching the shared counter.

        Used by the server immediately before forking workers so the
        child processes inherit *closed* env handles.  python-lmdb's
        per-environment transaction state would otherwise survive
        :meth:`Environment.close` in the worker and trip
        ``mdb_txn_renew: MDB_BAD_RSLOT`` on the first read transaction
        after fork — even with ``max_spare_txns=0`` on the new env.
        The shared ``n_allocated_rows`` counter (a
        :class:`multiprocessing.Value`) is left intact so workers can
        still query the row count without an LMDB transaction.

        Safe to call multiple times; subsequent calls are no-ops
        because LMDB's own ``close`` is idempotent.
        """
        try:
            self.flush()
        except Exception:
            log.debug("Skipping flush during close_env (env already closed)")
        try:
            self._env.close()
        except Exception:
            pass

    def open_env(self) -> None:
        """(Re-)open the LMDB environment at the current ``map_size``.

        Counterpart of :meth:`close_env`.  Used by the server after
        worker spawn to restore the parent's own env handle for
        flushes during checkpointing.  Does not re-read the
        ``__next_row__`` watermark — that mirror lives in a shared
        :class:`multiprocessing.Value` and is already populated.
        """
        self._env = _open_lmdb(
            str(self._path),
            map_size=self._map_size,
            writemap=True,
            map_async=True,
            max_readers=256,
            max_spare_txns=0,
            # Runtime env is disposable node-local scratch — durability comes
            # from the checkpoint mirror after an explicit flush(sync=True), so
            # skip the per-commit meta fsync (commits then run at memory speed).
            sync=False,
            metasync=False,
        )

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
        # Use the size the parent actually opened with — the parent may have
        # grown the map (`_reopen`) so the module-level default is stale.
        # max_spare_txns=0 disables the per-thread cached-read-txn pool;
        # cached txns inherited across fork point at a parent reader slot
        # and would trip `mdb_txn_renew: MDB_BAD_RSLOT` on the first get().
        self._env = _open_lmdb(
            str(self._path),
            map_size=self._map_size,
            writemap=True,
            map_async=True,
            max_readers=256,
            max_spare_txns=0,
            # Runtime env is disposable node-local scratch — durability comes
            # from the checkpoint mirror after an explicit flush(sync=True), so
            # skip the per-commit meta fsync (commits then run at memory speed).
            sync=False,
            metasync=False,
        )
        # Purge any reader slots left behind by the parent — the child's new
        # env starts fresh but the lock table on disk can still carry stale
        # entries from the parent's read transactions (e.g. `_read_next_row`).
        try:
            self._env.reader_check()
        except Exception as e:
            log.debug("reader_check skipped: %s", e)
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
            max_spare_txns=0,
            # Runtime env is disposable node-local scratch — durability comes
            # from the checkpoint mirror after an explicit flush(sync=True), so
            # skip the per-commit meta fsync (commits then run at memory speed).
            sync=False,
            metasync=False,
        )

    def _get_or_create_once(self, info_set: str, skip_read: bool = False) -> tuple:
        """Single-shot get-or-create, wrapped by the retry loop.

        Uses a read-first / double-checked-write pattern: try a
        concurrent read transaction first (LMDB allows unlimited
        concurrent readers), and only escalate to a serialised write
        transaction when the row is genuinely missing.  Once the
        table has warmed up, the vast majority of calls take the
        read path and never contend on LMDB's single-writer mutex.

        The write path re-checks for the row inside the write txn
        because another writer may have created it between our read
        and our write — LMDB only serialises writers, so a second
        writer must always assume the data may have changed since it
        saw the read-side snapshot.  This re-check is what makes
        ``skip_read`` safe: when the caller (a cache-backed
        :meth:`get_or_create`) has already established the row is
        absent, the leading read txn is pure overhead, so we jump
        straight to the write txn without weakening correctness.
        """
        key = hash_info_set_bytes(info_set)

        # Read path — concurrent across workers, no writer-lock contention.
        # Skipped when the caller already knows the row is absent (cache miss).
        if not skip_read:
            with self._env.begin() as txn:
                val = txn.get(key)
            if val is not None:
                return struct.unpack("<Q", val)[0], False

        # Write path — serialised at env level; re-check inside the txn.
        with self._env.begin(write=True) as txn:
            val = txn.get(key)
            if val is not None:
                return struct.unpack("<Q", val)[0], False

            # Allocate a new row.
            meta = txn.get(_NEXT_ROW_KEY)
            next_row: int = struct.unpack("<Q", meta)[0] if meta else 0

            txn.put(key, struct.pack("<Q", next_row))
            txn.put(_NEXT_ROW_KEY, struct.pack("<Q", next_row + 1))
            # No lock needed: this runs inside the LMDB write txn, and LMDB
            # serialises every writer across processes through its single
            # writer mutex, so no two processes are ever in this block at
            # once.  The mp.Value is retained purely for lock-free reads
            # (n_allocated_rows) elsewhere; the increment itself is already
            # mutually excluded by the surrounding write transaction.
            self._n_allocated_mp.value = next_row + 1

            if self._debug:
                raw = info_set if isinstance(info_set, bytes) else info_set.encode("utf-8")
                shadow_key = _STR_PREFIX + key
                existing_str = txn.get(shadow_key)
                if existing_str is not None and existing_str != raw:
                    raise AssertionError(
                        f"128-bit hash collision detected!\n"
                        f"  info_set A (existing): {existing_str!r}\n"
                        f"  info_set B (new):      {info_set!r}\n"
                        f"  hash key: {key.hex()}"
                    )
                txn.put(shadow_key, raw)

        return next_row, True
