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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from poker_ai.tables.chunk_store import CHUNK_SIZE
from poker_ai.tables.index import InfosetIndex
from poker_ai.tables.chunked_table import ChunkedTable
from utils.io import atomic_numpy_save

log = logging.getLogger("poker_ai.tables.cfr_tables")

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
        """
        if actions_per_street is None:
            raise ValueError("actions_per_street is required")

        base = Path(index_path)
        self._indexes: Dict[int, InfosetIndex] = {
            r: InfosetIndex(base / f"street_{r}", map_size=lmdb_map_size)
            for r in range(4)
        }
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

                # Regret: discount + floor clamp.
                rview = self.regret[r].store.view(chunk_id)[:valid_rows]
                rresult = (rview.astype(np.float32) * factor32).astype(np.int32)
                np.maximum(rresult, REGRET_FLOOR, out=rresult)
                rview[:] = rresult
                self.regret[r].store.mark_dirty(chunk_id)

                # Strategy: discount only (non-negative visit counts).
                sview = self.strategy[r].store.view(chunk_id)[:valid_rows]
                sview[:] = (sview.astype(np.float32) * factor32).astype(np.int32)
                self.strategy[r].store.mark_dirty(chunk_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
