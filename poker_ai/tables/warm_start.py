"""Warm-start blueprint training from an existing base blueprint.

A biased run starts from a finished base blueprint's regret and
strategy tables rather than from zeros — the converged regrets are a
much better starting point than uniform random for the biased
variants.

The warm-start mechanism stages the base blueprint's on-disk state
into the new run's ``save_path`` before any :class:`CFRTables` is
constructed:

- The LMDB info-set index is copied directly so the new run sees
  exactly the same ``info_set → row_idx`` mapping as the base.
- The checkpoint chunk files (regret + strategy) are copied into a
  freshly-named ``checkpoint_<wall_time>`` subdirectory with a
  rewritten ``server_state.pkl`` that resets ``t = 0`` and
  ``discount_active = True`` so the LCFR window opens fresh and the
  iteration counter starts from zero.

For multi-process runs the staged checkpoint is auto-loaded by
:class:`~poker_ai.tables.checkpoint.CheckpointManager` during
construction, so the only call site is
:func:`apply_warm_start` invoked before
:class:`~poker_ai.blueprint.multiprocess.server.Server` constructs the
shared tables.

For single-process runs there is no checkpoint manager, so
:func:`stage_warm_start_lmdb` is called before
``CFRTables(...)`` and :func:`apply_warm_start_to_tables` is called
after, restoring chunks via
:meth:`~poker_ai.tables.cfr_tables.CFRTables.restore_chunks`.

Resume always wins: if the destination ``save_path`` already contains
a valid ``checkpoint_*`` directory, every helper here is a no-op.
The user can re-invoke a biased run with the same ``--nickname`` and
it will resume from its latest checkpoint, ignoring the warm-start
flag.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import joblib

from utils.io import atomic_joblib_dump

log = logging.getLogger("poker_ai.tables.warm_start")


def _latest_checkpoint(warm_start_path: Path) -> Path:
    """Return the most recent ``checkpoint_*`` directory under *warm_start_path*."""
    candidates = sorted(warm_start_path.glob("checkpoint_[0-9]*"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint_* directory found in warm-start path: {warm_start_path}"
        )
    return candidates[-1]


def _extract_n_chunks(state_dict: dict) -> Dict[int, int]:
    """Per-street chunk counts, supporting the legacy split format."""
    ncs = state_dict.get("n_chunks_per_street", {})
    old_strategy = state_dict.get("n_strategy_chunks_per_street", {})
    if old_strategy:
        return {
            r: max(int(ncs.get(r, 0)), int(old_strategy.get(r, 0)))
            for r in range(4)
        }
    if ncs:
        return {int(k): int(v) for k, v in ncs.items()}
    return {r: 0 for r in range(4)}


def _validate_n_players(state_dict: dict, expected_n_players: int) -> int:
    """Assert action-space compatibility via the ``n_players`` field."""
    saved = state_dict.get("n_players")
    if saved is None:
        raise ValueError(
            "Warm-start checkpoint is missing the 'n_players' field — "
            "cannot verify action-space compatibility."
        )
    if int(saved) != int(expected_n_players):
        raise ValueError(
            f"Warm-start n_players mismatch: warm_start={saved!r} "
            f"vs current={expected_n_players!r}. Re-train the base "
            f"blueprint with the matching player count or change the "
            f"current run's --n_players."
        )
    return int(saved)


def apply_warm_start(
    save_path: Path,
    warm_start_path: Path,
    expected_n_players: int,
) -> Optional[Path]:
    """Stage a base blueprint into *save_path* for multi-process resume.

    Copies the warm-start LMDB index and the latest checkpoint's chunk
    files into ``save_path``, rewriting ``server_state.pkl`` with
    ``t = 0`` and ``discount_active = True`` so the next
    :class:`CheckpointManager` finds a valid checkpoint and restores
    transparently.  Resume takes priority: when ``save_path`` already
    contains a ``checkpoint_*`` directory the function logs and
    returns ``None`` without copying anything.

    Parameters
    ----------
    save_path : Path
        Destination directory for the new biased run.
    warm_start_path : Path
        Directory holding the base blueprint's run output (LMDB index +
        ``checkpoint_*`` subdirectory).
    expected_n_players : int
        Player count of the new run.  Must match the saved checkpoint
        or the function raises.

    Returns
    -------
    Path or None
        Path to the staged checkpoint directory, or ``None`` if no-op.
    """
    save_path = Path(save_path)
    warm_start_path = Path(warm_start_path)

    if list(save_path.glob("checkpoint_[0-9]*")):
        log.info(
            f"Existing checkpoint in {save_path} — warm-start skipped "
            f"(resume takes priority)."
        )
        return None

    src_cp = _latest_checkpoint(warm_start_path)
    state_dict = joblib.load(src_cp / "server_state.pkl")
    _validate_n_players(state_dict, expected_n_players)

    src_lmdb = warm_start_path / "lmdb_index"
    if not src_lmdb.exists():
        raise FileNotFoundError(
            f"Warm-start path {warm_start_path} has no lmdb_index directory."
        )

    save_path.mkdir(parents=True, exist_ok=True)

    # LMDB index — full directory copy.  Small (info-set → row idx);
    # cost is negligible compared to chunk I/O.
    dst_lmdb = save_path / "lmdb_index"
    if dst_lmdb.exists():
        shutil.rmtree(dst_lmdb)
    shutil.copytree(src_lmdb, dst_lmdb)

    # Stage a fresh checkpoint dir.  Naming follows the convention
    # used by CheckpointManager so the auto-resume path picks it up.
    dst_cp = save_path / f"checkpoint_{int(time.time())}"
    dst_cp.mkdir(parents=True)
    n_chunks_copied = 0
    for npy in src_cp.glob("*.npy"):
        shutil.copy2(npy, dst_cp / npy.name)
        n_chunks_copied += 1

    # Minimal state dict.  Anything not in here is treated as "old
    # checkpoint format" by CheckpointManager._validate_config_compatibility
    # and skipped silently — that is how we let the user reconfigure
    # operational and bias hyperparameters without tripping the
    # structural-mismatch check.
    new_state = {
        "t": 0,
        "discount_active": True,
        "n_chunks_per_street": _extract_n_chunks(state_dict),
        "n_players": int(state_dict["n_players"]),
    }
    # Only forward chunk_size when the source actually had one — a
    # ``None`` value would trip the structural-config check in
    # CheckpointManager (None != current CHUNK_SIZE).
    if state_dict.get("chunk_size") is not None:
        new_state["chunk_size"] = state_dict["chunk_size"]
    atomic_joblib_dump(new_state, dst_cp / "server_state.pkl")

    log.info(
        f"Warm-started from {warm_start_path}: copied lmdb_index + "
        f"{n_chunks_copied} chunk files into {dst_cp.name}; iteration "
        f"counter reset to 0."
    )
    return dst_cp


def stage_warm_start_lmdb(
    save_path: Path,
    warm_start_path: Path,
    expected_n_players: int,
) -> bool:
    """Copy the warm-start LMDB index into *save_path* (single-process path).

    Must be called before :class:`CFRTables` is constructed so the
    new tables open the warm-start info-set mapping rather than an
    empty one.  Returns ``True`` if staging happened, ``False`` if it
    was skipped because *save_path* already has its own LMDB index.
    """
    save_path = Path(save_path)
    warm_start_path = Path(warm_start_path)

    dst_lmdb = save_path / "lmdb_index"
    if dst_lmdb.exists():
        log.info(
            f"Existing lmdb_index in {save_path} — warm-start LMDB stage "
            f"skipped."
        )
        return False

    src_cp = _latest_checkpoint(warm_start_path)
    state_dict = joblib.load(src_cp / "server_state.pkl")
    _validate_n_players(state_dict, expected_n_players)

    src_lmdb = warm_start_path / "lmdb_index"
    if not src_lmdb.exists():
        raise FileNotFoundError(
            f"Warm-start path {warm_start_path} has no lmdb_index directory."
        )
    save_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_lmdb, dst_lmdb)
    log.info(f"Warm-started LMDB index from {src_lmdb} → {dst_lmdb}")
    return True


def apply_warm_start_to_tables(
    tables,
    warm_start_path: Path,
    expected_n_players: int,
) -> None:
    """Restore warm-start chunks into already-constructed :class:`CFRTables`.

    Single-process counterpart to :func:`apply_warm_start`.  Reads the
    chunk count from the source ``server_state.pkl`` and forwards to
    :meth:`CFRTables.restore_chunks
    <poker_ai.tables.cfr_tables.CFRTables.restore_chunks>`.
    """
    warm_start_path = Path(warm_start_path)
    src_cp = _latest_checkpoint(warm_start_path)
    state_dict = joblib.load(src_cp / "server_state.pkl")
    _validate_n_players(state_dict, expected_n_players)

    ncs = _extract_n_chunks(state_dict)
    tables.restore_chunks(src_cp, ncs)
    log.info(
        f"Warm-started chunks from {src_cp} into in-memory tables "
        f"(per-street chunks: {ncs})."
    )
