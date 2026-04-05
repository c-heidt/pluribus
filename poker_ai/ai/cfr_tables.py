"""Per-street table container with checkpoint I/O and discount.

Owns four ``InfosetIndex`` instances (one per betting street) and eight
``ChunkedTable`` instances (regret + strategy per street).  External code
accesses individual tables via ``.regret[r]`` and ``.strategy[r]`` and
calls bulk methods for checkpoint, discount, and lifecycle operations.
"""

import logging
import math
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from poker_ai.ai.chunk_store import CHUNK_SIZE
from poker_ai.ai.index import InfosetIndex
from poker_ai.ai.chunked_table import ChunkedTable

log = logging.getLogger("poker_ai.ai.cfr_tables")

REGRET_FLOOR: np.int32 = np.int32(-310_000_000)
"""Per-action regret floor from Pluribus supplementary material (Section S2).
Prevents int32 underflow and allows pruned actions to recover."""


class CFRTables:
    """Per-street regret and strategy tables with per-street LMDB indexes.

    Parameters
    ----------
    index_path:
        Base directory for LMDB indexes.  Per-street subdirectories
        (``street_0`` ... ``street_3``) are created automatically.
    shm_dir:
        Shared-memory directory for chunk mmap files.
    lmdb_map_size:
        LMDB map_size passed to each per-street ``InfosetIndex``.
    actions_per_street:
        ``Dict[int, int]`` mapping street index (0-3) to the number of
        abstract actions on that street.
    """

    def __init__(
        self,
        index_path: Union[str, Path],
        shm_dir: str = "/dev/shm",
        lmdb_map_size: Optional[int] = None,
        actions_per_street: Optional[Dict[int, int]] = None,
    ) -> None:
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
        """Return the number of allocated chunks per street."""
        result = {}
        for r in range(4):
            n = self._indexes[r].n_allocated_rows
            result[r] = math.ceil(n / CHUNK_SIZE) if n > 0 else 0
        return result

    def save_chunks(self, dir_path: Path) -> int:
        """Save dirty chunks for all 8 tables to *dir_path*.

        Returns total number of chunks written.
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

    def validate_chunks(
        self, dir_path: Path, n_chunks: Dict[int, int]
    ) -> bool:
        """Check that all expected chunk .npy files exist in *dir_path*."""
        for r in range(4):
            for chunk_id in range(n_chunks.get(r, 0)):
                for prefix in (f"regret_{r}", f"strategy_{r}"):
                    if not (dir_path / f"{prefix}_chunk_{chunk_id:06d}.npy").exists():
                        return False
        return True

    def restore_chunks(self, dir_path: Path, n_chunks: Dict[int, int]) -> None:
        """Load chunk .npy files from *dir_path* back into shared memory."""
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
        """Apply LCFR discount to all regret and strategy tables.

        Must only be called when all workers are idle (sync boundary).
        """
        if not (0.0 < factor <= 1.0):
            raise ValueError(f"Discount factor must be in (0, 1], got {factor}")

        factor32 = np.float32(factor)

        for r in range(4):
            n_entries = self._indexes[r].n_allocated_rows
            n_chunks = math.ceil(n_entries / CHUNK_SIZE) if n_entries > 0 else 0

            for table in (self.regret[r], self.strategy[r]):
                for chunk_id in range(n_chunks):
                    chunk = table.store.view(chunk_id)
                    valid_rows = min(
                        n_entries - chunk_id * CHUNK_SIZE, CHUNK_SIZE
                    )
                    view = chunk[:valid_rows]
                    result = (view.astype(np.float32) * factor32).astype(np.int32)
                    np.maximum(result, REGRET_FLOOR, out=result)
                    view[:] = result
                    table.store.mark_dirty(chunk_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reopen_after_fork(self) -> None:
        """Reopen all per-street LMDB indexes in a forked child process."""
        for idx in self._indexes.values():
            idx.reopen_after_fork()

    def flush_indexes(self) -> None:
        """Flush all per-street LMDB indexes to disk."""
        for idx in self._indexes.values():
            idx.flush()

    def close(self) -> None:
        """Close and unlink all shared-memory tables and LMDB indexes."""
        for r in range(4):
            self.regret[r].close()
            self.strategy[r].close()
            self.regret[r].unlink_all()
            self.strategy[r].unlink_all()
            self._indexes[r].close()
