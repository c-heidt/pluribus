"""Checkpoint state for the chunked build pipeline.

A :class:`CheckpointManager` owns the on-disk ``checkpoint.json`` file that
tracks, per street, which chunks have completed, whether merging has run, and
whether clustering has run.  All writes are atomic (``rename`` after temp-file
write) and guarded by an ``fcntl`` file lock so multiple processes sharing
the same ``save_dir`` cannot race.

The class is deliberately ignorant of directory layout, chunk file formats,
or worker dispatch — those concerns belong to :mod:`.chunk_store`.
"""
import fcntl
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

log = logging.getLogger("information_abstraction.build.checkpoint")


STREETS = ("river", "turn", "flop")


def _empty_street_state() -> Dict[str, Any]:
    return {
        "completed_chunks": [],
        "total_chunks": 0,
        "merge_done": False,
        "clustering_done": False,
        "feature_dim": None,
    }


def _empty_checkpoint() -> Dict[str, Any]:
    return {
        "streets": {street: _empty_street_state() for street in STREETS},
        "config": {},
    }


class CheckpointManager:
    """Resume-friendly state for the chunked build pipeline.

    Parameters
    ----------
    checkpoint_path : Path
        Path to ``checkpoint.json`` inside the build save directory.
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                log.warning(
                    "Failed to load checkpoint, starting fresh: %s", e,
                )
        return _empty_checkpoint()

    def save(self, merge_with_disk: bool = True) -> None:
        """Persist the in-memory checkpoint atomically.

        Parameters
        ----------
        merge_with_disk : bool
            When ``True`` (default), union the in-memory
            ``completed_chunks`` lists with whatever is currently on disk
            before writing.  This lets a second process that wrote extra
            chunks not have its work clobbered.  Pass ``False`` when the
            caller explicitly wants to overwrite disk state (e.g. after
            :meth:`reset_street`).
        """
        lock_path = self.checkpoint_path.with_suffix(".lock")
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        try:
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    if merge_with_disk and self.checkpoint_path.exists():
                        try:
                            with open(self.checkpoint_path, "r") as f:
                                disk = json.load(f)
                            for street in self._state["streets"]:
                                disk_done = set(
                                    disk.get("streets", {})
                                    .get(street, {})
                                    .get("completed_chunks", [])
                                )
                                mem_done = set(
                                    self._state["streets"][street][
                                        "completed_chunks"
                                    ]
                                )
                                self._state["streets"][street][
                                    "completed_chunks"
                                ] = sorted(disk_done | mem_done)
                        except (json.JSONDecodeError, KeyError):
                            pass
                    with open(temp_path, "w") as f:
                        json.dump(self._state, f, indent=2)
                    shutil.move(str(temp_path), str(self.checkpoint_path))
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except IOError:
            if temp_path.exists():
                temp_path.unlink()
            raise

    # ------------------------------------------------------------------
    # Per-street queries
    # ------------------------------------------------------------------

    def get_completed_chunks(self, street: str) -> List[int]:
        return list(self._state["streets"][street]["completed_chunks"])

    def get_incomplete_chunks(self, street: str) -> List[int]:
        street_data = self._state["streets"][street]
        completed: Set[int] = set(street_data["completed_chunks"])
        total = street_data["total_chunks"]
        return [i for i in range(total) if i not in completed]

    def get_total_chunks(self, street: str) -> int:
        return int(self._state["streets"][street]["total_chunks"])

    def get_total_combos(self, street: str) -> Optional[int]:
        return self._state["streets"][street].get("total_combos")

    def get_feature_dim(self, street: str) -> Optional[int]:
        return self._state["streets"][street].get("feature_dim")

    def is_merge_done(self, street: str) -> bool:
        return bool(self._state["streets"][street].get("merge_done", False))

    def is_clustering_done(self, street: str) -> bool:
        return bool(
            self._state["streets"][street].get("clustering_done", False)
        )

    # ------------------------------------------------------------------
    # Per-street mutations
    # ------------------------------------------------------------------

    def mark_chunk_complete(self, street: str, chunk_idx: int) -> None:
        completed = self._state["streets"][street]["completed_chunks"]
        if chunk_idx not in completed:
            completed.append(chunk_idx)
            completed.sort()
        self.save()

    def update_completed_chunks(
        self, street: str, completed: Iterable[int], merge_with_disk: bool,
    ) -> None:
        """Replace ``completed_chunks`` for a street (bulk flush path)."""
        self._state["streets"][street]["completed_chunks"] = sorted(
            set(completed)
        )
        self.save(merge_with_disk=merge_with_disk)

    def mark_merge_done(self, street: str) -> None:
        self._state["streets"][street]["merge_done"] = True
        self.save()

    def unmark_merge_done(self, street: str) -> None:
        self._state["streets"][street]["merge_done"] = False
        self.save(merge_with_disk=False)

    def mark_clustering_done(self, street: str) -> None:
        self._state["streets"][street]["clustering_done"] = True
        self.save()

    def set_feature_dim(self, street: str, feature_dim: int) -> None:
        self._state["streets"][street]["feature_dim"] = int(feature_dim)
        self.save()

    def set_config(self, config: Dict[str, Any]) -> None:
        self._state["config"] = config

    def reset_street(
        self,
        street: str,
        total_chunks: int,
        total_combos: int,
    ) -> None:
        """Reset a street's checkpoint state (fresh run / config change)."""
        self._state["streets"][street] = {
            "completed_chunks": [],
            "total_chunks": total_chunks,
            "total_combos": total_combos,
            "merge_done": False,
            "clustering_done": False,
            "feature_dim": None,
        }

    def update_total_combos(self, street: str, total_combos: int) -> None:
        """Keep total_combos in sync (may be absent in old checkpoints)."""
        self._state["streets"][street]["total_combos"] = total_combos
