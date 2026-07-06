"""Per-street container for CFR regret and strategy tables.

A single :class:`CFRTables` instance holds every table needed to
train a CFR blueprint:

- Four :class:`~poker_ai.tables.index.InfosetIndex` instances — one per
  betting street.  Splitting the index by street keeps each LMDB
  small and lets per-street row counts be read independently, which
  in turn simplifies discount bookkeeping and checkpoint layout.
- Four regret :class:`~poker_ai.tables.chunked_table.ChunkedTable`
  instances, one per street, exposed as ``tables.regret[r]``.
- Four strategy :class:`~poker_ai.tables.chunked_table.ChunkedTable`
  instances, one per street, exposed as ``tables.strategy[r]``.

External code never touches the indexes directly — they are an
implementation detail shared between each street's regret and
strategy tables so both refer to the same row numbers for the same
information set.

The class also owns the bulk operations that must span every table:
checkpoint I/O (:meth:`save_chunks`, :meth:`validate_chunks`,
:meth:`restore_chunks`), periodic LCFR discounting
(:meth:`apply_discount`), and lifecycle
(:meth:`reopen_after_fork`, :meth:`flush_indexes`, :meth:`close`).

Must be constructed in the **parent process before workers are
forked** so that every child inherits the underlying stripe locks,
shared-memory mmaps, and LMDB environments.
"""

import logging
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from poker_ai.tables.chunk_store import CHUNK_SIZE
from poker_ai.tables.index import InfosetIndex
from poker_ai.tables.chunked_table import ChunkedTable
from poker_ai.tables.shm_index_cache import (
    DEFAULT_LOAD_FACTOR,
    ShmIndexCache,
    capacity_for,
    next_pow2,
)
from utils.io import atomic_numpy_save

log = logging.getLogger("poker_ai.tables.cfr_tables")

# Fallback per-street cache capacities when the shm index cache is enabled
# without an explicit size and there are no existing rows to size from (e.g.
# a small fresh run).  The real large run sets PLURIBUS_INDEX_CAPACITY.
_DEFAULT_CACHE_CAPACITY: Dict[int, int] = {0: 2 ** 22, 1: 2 ** 23, 2: 2 ** 23, 3: 2 ** 23}


def _parse_capacities_env() -> Optional[Dict[int, int]]:
    """Parse ``PLURIBUS_INDEX_CAPACITY`` (``"pf,flop,turn,river"``) → dict."""
    raw = os.environ.get("PLURIBUS_INDEX_CAPACITY")
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(
            f"PLURIBUS_INDEX_CAPACITY must be 4 comma-separated ints "
            f"(pre_flop,flop,turn,river), got {raw!r}"
        )
    return {r: int(parts[r]) for r in range(4)}


def _cache_capacity(
    street: int,
    n_allocated: int,
    capacities: Optional[Dict[int, int]],
    load_factor: float,
    headroom: float,
) -> int:
    """Power-of-two cache capacity for a street.

    Two regimes, chosen so a **resume never shrinks the cache below what the
    original run used** (which would overflow as the resumed run keeps
    allocating):

    - **Explicit capacity** (``capacities[street]`` — the env
      ``PLURIBUS_INDEX_CAPACITY`` on a fresh run, or the capacity persisted in
      the checkpoint on resume): honour it as the saturation target.  Only
      bump it if it cannot even hold the rows already present (a safety floor,
      no headroom multiplier — existing rows ``<=`` saturation by definition).
    - **No explicit capacity**: auto-size from existing rows grown by
      ``headroom``, floored at a modest default so a small resumed run keeps at
      least the fresh run's default capacity.
    """
    if capacities and capacities.get(street):
        cap = next_pow2(int(capacities[street]))
        if n_allocated > 0:
            cap = max(cap, capacity_for(n_allocated, load_factor))
        return cap
    base = (
        capacity_for(int(math.ceil(n_allocated * headroom)), load_factor)
        if n_allocated > 0
        else 0
    )
    return max(base, _DEFAULT_CACHE_CAPACITY[street])

REGRET_FLOOR: np.int32 = np.int32(-310_000_000)
"""Per-action regret floor.

Taken from the Pluribus supplementary material (Section S2).  Prevents
int32 underflow during very long training runs and ensures actions
that have been pruned for a long time can still recover — if regret
could drop arbitrarily low, the regret-matching step would never
bring the action back above zero.
"""


class CFRTables:
    """Per-street regret and strategy tables with per-street indexes.

    The class is a thin orchestrator: it creates four index + eight
    table instances up front, exposes the tables as ``regret[r]`` and
    ``strategy[r]`` dicts, and forwards bulk operations to each
    component.  All CFR algorithms interact with it through these
    dicts and the bulk methods below.

    Attributes
    ----------
    regret : dict[int, ChunkedTable]
        Per-street regret tables, keyed by betting round (0 = pre-flop,
        3 = river).
    strategy : dict[int, ChunkedTable]
        Per-street strategy tables, keyed by betting round.
    """

    def __init__(
        self,
        index_path: Union[str, Path],
        shm_dir: str = "/dev/shm",
        lmdb_map_size: Optional[int] = None,
        actions_per_street: Optional[Dict[int, int]] = None,
        enable_index_cache: bool = False,
        index_capacities: Optional[Dict[int, int]] = None,
    ) -> None:
        """Open (or create) the four indexes and the eight tables.

        Parameters
        ----------
        index_path : str or Path
            Base directory for the per-street LMDB indexes.  One
            subdirectory ``street_{r}`` is created under this base for
            each betting round.
        shm_dir : str, optional
            Directory for the shared-memory chunk files.  Defaults to
            ``/dev/shm``.
        lmdb_map_size : int, optional
            LMDB map_size passed to each :class:`InfosetIndex`.  When
            ``None`` the index picks its own default from the
            ``PLURIBUS_LMDB_MAP_SIZE`` environment variable.
        actions_per_street : dict[int, int]
            Mandatory mapping from street index to number of abstract
            actions.  Supplied by
            :data:`environment.action_space.MAX_ACTIONS_PER_STREET`.
        enable_index_cache : bool, optional
            Build a per-street :class:`ShmIndexCache` in front of each LMDB
            index so hot-path ``get`` reads never open an LMDB transaction.
            Off by default (standalone indexes and unit tests keep the
            LMDB-only path).  Must be constructed in the parent before workers
            fork; call :meth:`prewarm_caches` before the fork.
        index_capacities : dict[int, int], optional
            Per-street cache slot counts (fresh-run sizing).  When ``None`` and
            the cache is enabled, ``PLURIBUS_INDEX_CAPACITY`` is consulted; on
            resume the size is derived from the existing row count regardless.
        """
        if actions_per_street is None:
            raise ValueError("actions_per_street is required")

        # Deferred-durability allocation (PLURIBUS_DEFERRED_ALLOC): the shm cache
        # assigns rows under its lightweight lock and LMDB is written in bulk at
        # checkpoints (:meth:`persist_indexes`), taking the LMDB writer mutex off
        # the allocation hot path.  Requires the cache — a no-op otherwise.
        self._deferred_alloc: bool = (
            enable_index_cache
            and os.environ.get("PLURIBUS_DEFERRED_ALLOC", "0") == "1"
        )

        base = Path(index_path)
        self._indexes: Dict[int, InfosetIndex] = {
            r: InfosetIndex(
                base / f"street_{r}",
                map_size=lmdb_map_size,
                deferred=self._deferred_alloc,
            )
            for r in range(4)
        }

        # Optional shm read caches (parent-side; inherited by forked workers).
        # Sized after the indexes exist so a resume/warm-start can auto-size
        # from each street's existing row count.
        self._index_caches: Optional[Dict[int, ShmIndexCache]] = None
        if enable_index_cache:
            self._build_index_caches(shm_dir, index_capacities)

        self.regret: Dict[int, ChunkedTable] = {
            r: ChunkedTable(
                n_actions=actions_per_street[r],
                table_name=f"pluribus_regret_{r}",
                index=self._indexes[r],
                shm_dir=shm_dir,
            )
            for r in range(4)
        }
        self.strategy: Dict[int, ChunkedTable] = {
            r: ChunkedTable(
                n_actions=actions_per_street[r],
                table_name=f"pluribus_strategy_{r}",
                index=self._indexes[r],
                shm_dir=shm_dir,
            )
            for r in range(4)
        }

    # ------------------------------------------------------------------
    # Shared-memory index cache
    # ------------------------------------------------------------------

    def _build_index_caches(
        self, shm_dir: str, index_capacities: Optional[Dict[int, int]]
    ) -> None:
        """Create and attach a per-street :class:`ShmIndexCache`."""
        capacities = index_capacities or _parse_capacities_env()
        load_factor = float(
            os.environ.get("PLURIBUS_INDEX_LOAD_FACTOR", DEFAULT_LOAD_FACTOR)
        )
        headroom = float(os.environ.get("PLURIBUS_INDEX_GROWTH_HEADROOM", 1.3))
        caches: Dict[int, ShmIndexCache] = {}
        total_bytes = 0
        for r in range(4):
            cap = _cache_capacity(
                r, self._indexes[r].n_allocated_rows, capacities, load_factor, headroom
            )
            cache = ShmIndexCache(
                name=f"pluribus_index_cache_{r}",
                capacity=cap,
                shm_dir=shm_dir,
                load_factor=load_factor,
            )
            self._indexes[r].set_cache(cache)
            caches[r] = cache
            total_bytes += cache.n_bytes
        self._index_caches = caches
        # The capacities actually used — persisted in the checkpoint so a
        # resume rebuilds the same-size cache instead of auto-shrinking.
        self._index_cache_capacities = {r: c.capacity for r, c in caches.items()}
        log.info(
            "Index caches enabled: capacities=%s, total %.2f GiB in %s",
            self._index_cache_capacities,
            total_bytes / 1024 ** 3,
            shm_dir,
        )

    def prewarm_caches(self) -> None:
        """Populate every index cache from its LMDB (parent, before fork).

        No-op when the cache is disabled.  Must run after any resume /
        warm-start restore (so LMDB holds the rows to load) and before the
        worker pool forks (so children inherit a warm, consistent cache).
        """
        if self._index_caches is None:
            return
        for r in range(4):
            n = self._indexes[r].prewarm_cache()
            if n:
                log.info("Street %d: prewarmed %d index-cache entries", r, n)

    def index_cache_total_bytes(self) -> int:
        """Total resident bytes across all index caches (0 if disabled)."""
        if self._index_caches is None:
            return 0
        return sum(c.n_bytes for c in self._index_caches.values())

    def index_cache_capacities(self) -> Optional[Dict[int, int]]:
        """Per-street cache capacities in use, or ``None`` when disabled.

        Persisted in ``server_state.pkl`` so a resume rebuilds the same-size
        cache (see :func:`_cache_capacity`).
        """
        if self._index_caches is None:
            return None
        return dict(self._index_cache_capacities)

    # ------------------------------------------------------------------
    # Process-fork safety
    # ------------------------------------------------------------------

    def reopen_after_fork(self) -> None:
        """Reopen every per-street LMDB index in the current (forked) process.

        The four :class:`InfosetIndex` LMDB environments are **not** safe to share
        across a ``fork`` — a child that reuses the parent's inherited reader-lock
        slot trips ``mdb_txn_renew: MDB_BAD_RSLOT`` on its first read
        (:meth:`InfosetIndex.reopen_after_fork`).  A forked worker that will *read*
        these tables (e.g. a parallel search replica querying a blueprint at a
        depth-limit leaf) must call this once, before its first lookup.  The chunk
        stores are read-only ``/dev/shm`` mmaps shared copy-on-write, so only the
        indexes need reopening; both table families reference the same index object
        per street, so reopening the index fixes their reads too.
        """
        for index in self._indexes.values():
            index.reopen_after_fork()

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def n_chunks_per_street(self) -> Dict[int, int]:
        """Return the number of allocated chunks per street.

        Returns
        -------
        dict[int, int]
            Street index → number of allocated chunks.  Computed from
            each index's row count and :data:`CHUNK_SIZE` — both the
            regret and strategy tables for a given street share the
            same row count because they share an index.
        """
        result = {}
        for r in range(4):
            n = self._indexes[r].n_allocated_rows
            result[r] = math.ceil(n / CHUNK_SIZE) if n > 0 else 0
        return result

    def save_chunks(self, dir_path: Path) -> int:
        """Save every dirty chunk across all eight tables to *dir_path*.

        Delegates to :meth:`ChunkStore.save_dirty
        <poker_ai.tables.chunk_store.ChunkStore.save_dirty>` for each
        table, then clears the dirty flags so the next checkpoint
        only has to write chunks touched after this point.  Chunk
        files are written atomically via
        :func:`poker_ai.tables.chunk_store._atomic_save`.

        Parameters
        ----------
        dir_path : Path
            Target directory for the ``.npy`` chunk files.

        Returns
        -------
        int
            Total number of chunk files written across all streets
            and both table families.
        """
        total = 0
        for r in range(4):
            n_entries = self._indexes[r].n_allocated_rows
            for table, prefix in [
                (self.regret[r], f"regret_{r}"),
                (self.strategy[r], f"strategy_{r}"),
            ]:
                total += table.store.save_dirty(dir_path, n_entries, prefix)
                table.store.clear_dirty()
        log.info("Saved %d dirty chunks to %s", total, dir_path)
        return total

    def snapshot_dirty_chunks(self) -> List[Tuple[str, np.ndarray]]:
        """Copy every dirty chunk into private buffers, clear dirty flags.

        Counterpart of :meth:`save_chunks` for async checkpointing.
        Must be called inside the sync barrier (all workers flushed
        and idle) so the snapshot is transactionally consistent with
        the per-street index watermarks.  After this returns, workers
        may resume and begin dirtying chunks again; those new dirty
        marks will be captured by the *next* snapshot.

        Returns
        -------
        list[tuple[str, np.ndarray]]
            One entry per dirty chunk across all eight tables.  The
            string is the filename (no directory); the ndarray is a
            freshly-allocated int32 buffer owned by the caller.
        """
        buffers: List[Tuple[str, np.ndarray]] = []
        for r in range(4):
            n_entries = self._indexes[r].n_allocated_rows
            for table, prefix in [
                (self.regret[r], f"regret_{r}"),
                (self.strategy[r], f"strategy_{r}"),
            ]:
                buffers.extend(table.store.snapshot_dirty(n_entries, prefix))
        log.info("Snapshotted %d dirty chunks", len(buffers))
        return buffers

    @staticmethod
    def write_buffers(
        buffers: List[Tuple[str, np.ndarray]], dir_path: Path
    ) -> int:
        """Serialise pre-copied chunk buffers to *dir_path*.

        Consumes the list produced by :meth:`snapshot_dirty_chunks`.
        Safe to call from a background thread while workers are
        running — the buffers are private copies, not shared memory.

        Parameters
        ----------
        buffers : list[tuple[str, np.ndarray]]
            ``(filename, array)`` pairs to write.
        dir_path : Path
            Target directory (must exist).

        Returns
        -------
        int
            Number of files written.
        """
        for filename, array in buffers:
            atomic_numpy_save(array, dir_path / filename)
        return len(buffers)

    def validate_chunks(
        self, dir_path: Path, n_chunks: Dict[int, int]
    ) -> bool:
        """Check that every expected chunk ``.npy`` file exists in *dir_path*.

        Used by :class:`~poker_ai.tables.checkpoint.CheckpointManager` to
        decide whether a candidate checkpoint directory is complete
        enough to restore from.

        Parameters
        ----------
        dir_path : Path
            Directory to inspect.
        n_chunks : dict[int, int]
            Expected number of chunks per street (taken from the
            saved state dict of the candidate checkpoint).

        Returns
        -------
        bool
            ``True`` iff every expected ``regret_{r}_chunk_*.npy`` and
            ``strategy_{r}_chunk_*.npy`` file is present.
        """
        for r in range(4):
            for chunk_id in range(n_chunks.get(r, 0)):
                for prefix in (f"regret_{r}", f"strategy_{r}"):
                    if not (dir_path / f"{prefix}_chunk_{chunk_id:06d}.npy").exists():
                        return False
        return True

    def restore_chunks(self, dir_path: Path, n_chunks: Dict[int, int]) -> None:
        """Load chunk ``.npy`` files from *dir_path* into shared memory.

        Walks every street and chunk id in ``n_chunks``, loads the
        corresponding ``regret_{r}_chunk_*.npy`` and
        ``strategy_{r}_chunk_*.npy`` files, and copies them into the
        mmapped chunks via :meth:`ChunkStore.restore
        <poker_ai.tables.chunk_store.ChunkStore.restore>`.  Missing
        strategy chunks are tolerated and zero-initialised (some very
        old checkpoints did not save strategy chunks).

        Parameters
        ----------
        dir_path : Path
            Source directory.
        n_chunks : dict[int, int]
            Expected number of chunks per street, typically taken from
            the checkpoint's saved state dict.
        """
        for r in range(4):
            for chunk_id in range(n_chunks.get(r, 0)):
                regret_path = dir_path / f"regret_{r}_chunk_{chunk_id:06d}.npy"
                self.regret[r].store.restore(chunk_id, np.load(regret_path))

                strategy_path = dir_path / f"strategy_{r}_chunk_{chunk_id:06d}.npy"
                if strategy_path.exists():
                    self.strategy[r].store.restore(
                        chunk_id, np.load(strategy_path)
                    )
                else:
                    log.warning(
                        "Strategy chunk missing: %s — zero-initialised",
                        strategy_path,
                    )

    # ------------------------------------------------------------------
    # Discount
    # ------------------------------------------------------------------

    def apply_discount(self, factor: float) -> None:
        """Multiply every regret and strategy entry by *factor* in place.

        This is the core operation driving Linear CFR weighting — the
        :class:`~poker_ai.blueprint.training.DiscountState` schedule calls
        it at sync boundaries with a factor of
        ``discount_step / (discount_step + 1)`` to produce an
        effective linear weighting of the accumulated regrets and
        strategies.

        Regret entries are clamped to :data:`REGRET_FLOOR` after the
        multiplication to prevent int32 underflow and to keep pruned
        actions recoverable.  Strategy entries are non-negative visit
        counts, so no floor is applied to them — clamping them would
        bias the distribution.

        The method reads the shared mmaps directly, which is safe
        **only when every worker is idle** (i.e. immediately after a
        sync barrier).  The server enforces this ordering before
        calling :meth:`apply_discount`.

        Parameters
        ----------
        factor : float
            Discount factor in the interval ``(0, 1]``.  A factor of
            ``1`` leaves the tables unchanged.
        """
        if not (0.0 < factor <= 1.0):
            raise ValueError(f"Discount factor must be in (0, 1], got {factor}")

        factor32 = np.float32(factor)

        for r in range(4):
            n_entries = self._indexes[r].n_allocated_rows
            n_chunks = math.ceil(n_entries / CHUNK_SIZE) if n_entries > 0 else 0

            for chunk_id in range(n_chunks):
                valid_rows = min(
                    n_entries - chunk_id * CHUNK_SIZE, CHUNK_SIZE
                )

                # Regret: discount + floor clamp.  ``rint`` (round half to
                # even) rather than a plain int cast: the cast truncates
                # toward zero, which systematically bleeds ~0.5 per entry
                # per application — negligible for chip-scale regrets but
                # fatal for the unit-scale strategy counts below, so both
                # use the same unbiased rounding.
                rview = self.regret[r].store.view(chunk_id)[:valid_rows]
                rresult = np.rint(rview.astype(np.float32) * factor32).astype(np.int32)
                np.maximum(rresult, REGRET_FLOOR, out=rresult)
                rview[:] = rresult
                self.regret[r].store.mark_dirty(chunk_id)

                # Strategy: discount only (non-negative visit counts).
                # Truncation here zeroed any count of 1 on every discount
                # application, erasing the strategy mass accumulated inside
                # the discount window; rounding keeps small counts alive
                # under the mild late-window factors while still applying
                # the intended linear down-weighting.
                sview = self.strategy[r].store.view(chunk_id)[:valid_rows]
                sview[:] = np.rint(sview.astype(np.float32) * factor32).astype(np.int32)
                self.strategy[r].store.mark_dirty(chunk_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def copy_indexes_to(self, dst_dir: Union[str, Path]) -> None:
        """Write a consistent snapshot of every per-street LMDB index to *dst_dir*.

        Used by :class:`~poker_ai.tables.checkpoint.CheckpointManager`
        to mirror a node-local runtime LMDB back to the persistent
        save directory at checkpoint time.  Each street is written
        as a sub-directory ``street_{r}/`` under *dst_dir*, matching
        the layout that :class:`CFRTables` consumes on construction
        and that :func:`_load_checkpoint_if_exists` looks for on
        resume.

        The destination is created if missing; existing per-street
        sub-directories are removed first because LMDB refuses to
        write into a non-empty env directory.

        Parameters
        ----------
        dst_dir : str or Path
            Target directory that will receive
            ``street_0/.../street_3/`` LMDB environments.
        """
        dst = Path(dst_dir)
        dst.mkdir(parents=True, exist_ok=True)
        for r, idx in self._indexes.items():
            street_dst = dst / f"street_{r}"
            if street_dst.exists():
                shutil.rmtree(street_dst)
            idx.copy_to(street_dst)

    def close_envs(self) -> None:
        """Close every per-street LMDB environment in this process.

        Used by the server immediately before forking workers so the
        child processes inherit *closed* env handles, sidestepping
        ``MDB_BAD_RSLOT`` errors that can otherwise trip on the first
        post-fork read transaction.  Workers will open their own
        envs in :meth:`reopen_after_fork`; the parent reopens its
        envs via :meth:`open_envs` immediately after spawning the
        pool so checkpoint-time index flushes keep working.
        """
        for idx in self._indexes.values():
            idx.close_env()

    def open_envs(self) -> None:
        """(Re-)open every per-street LMDB environment in this process."""
        for idx in self._indexes.values():
            idx.open_env()

    def reopen_after_fork(self) -> None:
        """Reopen every per-street LMDB index in the current process.

        Workers call this at the top of their
        :meth:`~multiprocessing.Process.run` method so inherited LMDB
        environments are replaced with freshly-opened handles; see
        :meth:`InfosetIndex.reopen_after_fork
        <poker_ai.tables.index.InfosetIndex.reopen_after_fork>` for
        details.
        """
        for idx in self._indexes.values():
            idx.reopen_after_fork()

    def persist_indexes(self) -> int:
        """Bulk-flush deferred-allocation rows into LMDB (no-op if not deferred).

        In deferred-allocation mode the shm cache is the live row authority and
        LMDB lags; this writes every row allocated since the last flush into LMDB
        in one transaction per street so the on-disk index is consistent with the
        chunk snapshot taken at the same checkpoint (both cover ``[0,
        occupancy)``).  Must run under the sync barrier (no worker allocating),
        before :meth:`flush_indexes` and ``snapshot_dirty_chunks``.  Returns the
        total number of rows persisted across all streets.
        """
        return sum(idx.bulk_persist() for idx in self._indexes.values())

    def flush_indexes(self) -> None:
        """Flush every LMDB index to disk.

        Called by the checkpoint manager before serialising chunk
        data so the on-disk index and the chunk files stay in sync.
        """
        for idx in self._indexes.values():
            idx.flush()

    def close(self) -> None:
        """Close and unlink every table and index.

        Called once at shutdown.  After this call the instance is
        unusable and all shared-memory files have been removed from
        the filesystem.
        """
        for r in range(4):
            self.regret[r].close()
            self.strategy[r].close()
            self.regret[r].unlink_all()
            self.strategy[r].unlink_all()
            self._indexes[r].close()
        if self._index_caches is not None:
            for cache in self._index_caches.values():
                cache.close()
                cache.unlink()
