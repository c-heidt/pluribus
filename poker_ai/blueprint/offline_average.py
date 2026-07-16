"""Offline post-flop average-strategy reconstruction (Pluribus-style).

Pluribus does not accumulate a post-flop average strategy online.  Instead it
snapshots the *current* strategy periodically after a warm-up and averages the
regret-matched strategies of those snapshots offline once training is done
(Science supplement p.15).  In this codebase every retained checkpoint is such
a snapshot (see :class:`poker_ai.tables.checkpoint.CheckpointManager` — the
delete-previous step was removed), so this module turns a training directory's
retained checkpoints into a single final blueprint whose post-flop strategy
tables hold the averaged strategy.

The output directory is byte-layout-identical to a normal training save dir
(``lmdb_index/`` + one ``checkpoint_<t>/`` with ``{regret,strategy}_{r}`` chunk
files and ``server_state.pkl``), so it loads through the existing evaluation /
search stack unchanged (``CFRTables(index_path=<out>/lmdb_index)`` +
:func:`poker_ai.tables.warm_start.apply_warm_start_to_tables`).

Row alignment is positional: every retained checkpoint shares the one
``lmdb_index`` written by the run, so ``(street, row)`` denotes the same
information set in every snapshot and the average is a plain per-row mean over
the snapshots that contain the row (rows allocated later in training are absent
from earlier, shorter chunk files — those contribute to a smaller divisor).

The post-flop average is stored as scaled integer pseudo-counts (the strategy
tables are ``int32`` visit counts), so a row that averaged the probability
vector ``p`` is written as ``round(SIGMA_SCALE * p)``.  At read time
:meth:`poker_ai.search.policy.BlueprintPolicy._average_strategy` masks the row
to the legal actions and renormalises, so the scale cancels and the
``min_strategy_mass`` gate (default 10) passes comfortably (row mass
``~SIGMA_SCALE`` = 1e6).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np

log = logging.getLogger("poker_ai.blueprint.offline_average")

#: Integer scale applied to the averaged probability vector before it is stored
#: in the ``int32`` strategy tables.  A row that averaged ``p`` (summing to 1
#: over the allocated columns) is written as ``round(SIGMA_SCALE * p)``, so the
#: stored row sums to ~``SIGMA_SCALE`` — far above ``min_strategy_mass`` and
#: well within ``int32`` range.
SIGMA_SCALE_DEFAULT = 1_000_000

#: Streets whose strategy is reconstructed offline from snapshots.  Pre-flop
#: (street 0) keeps its trained running average and is copied through verbatim.
_POSTFLOP_STREETS = (1, 2, 3)
_ALL_STREETS = (0, 1, 2, 3)


def sigma_from_regret_chunk(chunk: np.ndarray) -> np.ndarray:
    """Vectorised regret matching over an ``(n_rows, n_actions)`` regret chunk.

    Mirrors :func:`poker_ai.blueprint.tree_utils.calculate_strategy_from_row`
    with no legal-action mask: each row's probability is proportional to its
    positive cumulative regret, falling back to uniform over *all* columns when
    no regret is positive.  Running maskless is safe because illegal actions
    never accumulate positive regret (so they carry no mass on the positive
    path) and the readout masks + renormalises to the legal actions anyway.

    Parameters
    ----------
    chunk : np.ndarray
        2-D array of cumulative regrets, one row per information set.

    Returns
    -------
    np.ndarray
        Float64 probability matrix of the same shape; every row sums to 1.
    """
    pos = np.clip(chunk, 0, None).astype(np.float64)
    total = pos.sum(axis=1)
    sigma = np.empty_like(pos)
    nz = total > 0.0
    # Rows with positive regret: normalise by the positive-regret mass.
    sigma[nz] = pos[nz] / total[nz][:, None]
    # Rows with no positive regret: uniform over all columns (the readout mask
    # restricts this to the legal subset at query time).
    n_actions = chunk.shape[1]
    sigma[~nz] = 1.0 / n_actions
    return sigma


def _chunk_ids(checkpoint_dir: Path, prefix: str) -> List[int]:
    """Sorted chunk ids present for *prefix* (e.g. ``regret_3``) in a dir."""
    ids = []
    for p in checkpoint_dir.glob(f"{prefix}_chunk_*.npy"):
        stem = p.stem  # e.g. regret_3_chunk_000004
        ids.append(int(stem.rsplit("_", 1)[1]))
    return sorted(ids)


def average_street(
    snapshot_dirs: List[Path],
    final_dir: Path,
    r: int,
    out_dir: Path,
    scale: int,
) -> int:
    """Average the regret-matched strategy of street *r* across snapshots.

    The output shape (chunk count and per-chunk row count) is taken from
    *final_dir* — the latest checkpoint, which has the most allocated rows.
    Each output chunk accumulates the per-row strategy of every snapshot that
    contains that chunk/row and divides by the per-row presence count.

    Parameters
    ----------
    snapshot_dirs : list[Path]
        Checkpoint directories to average (already filtered by warm-up).
    final_dir : Path
        Latest checkpoint; defines the output row/chunk shape.
    r : int
        Street index (1, 2 or 3).
    out_dir : Path
        Destination checkpoint directory for the ``strategy_{r}`` chunks.
    scale : int
        Integer scale for the stored pseudo-counts (see ``SIGMA_SCALE``).

    Returns
    -------
    int
        Number of strategy chunks written for this street.
    """
    prefix = f"regret_{r}"
    written = 0
    for chunk_id in _chunk_ids(final_dir, prefix):
        # Shape only: mmap the header so we never pull the ~80 MB final chunk
        # into RAM here (it is read again, once, inside the snapshot loop).
        n_rows, n_actions = np.load(
            final_dir / f"{prefix}_chunk_{chunk_id:06d}.npy", mmap_mode="r"
        ).shape
        # One float64 accumulator + one int64 presence counter per chunk, plus
        # one snapshot chunk in flight at a time → peak RAM is ~a few hundred MB
        # per chunk regardless of how many snapshots or how large the run's total
        # on-disk footprint is (streaming, not load-everything).
        acc = np.zeros((n_rows, n_actions), dtype=np.float64)
        count = np.zeros(n_rows, dtype=np.int64)

        for snap in snapshot_dirs:
            path = snap / f"{prefix}_chunk_{chunk_id:06d}.npy"
            if not path.exists():
                # This snapshot had not allocated this chunk yet.
                continue
            arr = np.load(path)
            k = min(arr.shape[0], n_rows)  # earlier snapshots may hold fewer rows
            acc[:k] += sigma_from_regret_chunk(arr[:k])
            count[:k] += 1
            del arr  # release the ~80 MB snapshot chunk before the next load

        out = np.zeros((n_rows, n_actions), dtype=np.int32)
        present = count > 0
        if present.any():
            mean = acc[present] / count[present][:, None]
            out[present] = np.rint(scale * mean).astype(np.int32)
        # Rows present in no snapshot stay all-zero → the readout falls back to
        # regret matching for them (mass 0 < min_strategy_mass).

        np.save(out_dir / f"strategy_{r}_chunk_{chunk_id:06d}.npy", out)
        written += 1
    return written


def _load_state(checkpoint_dir: Path) -> dict:
    return joblib.load(checkpoint_dir / "server_state.pkl")


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink *src* to *dst*, copying if hardlinking is unavailable."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_lmdb_index(src_index: Path, dst_index: Path, n_players: int) -> None:
    """Copy the per-street LMDB index compactly (used pages only).

    ``shutil.copytree`` would copy each street's ``data.mdb`` byte-for-byte
    INCLUDING the sparse ``map_size`` reservation (tens of GiB of zero-holes,
    and larger at production scale), densifying it on the destination.  LMDB's
    own ``env.copy`` instead writes only the pages actually in use — the same
    mechanism the checkpoint mirror uses (:meth:`InfosetIndex.copy_to`) — so the
    copied index is sized to the real key count, not the reservation.  The
    logical infoset→row mapping is byte-identical (only physical page layout is
    repacked), so the averaged chunks stay row-aligned with it.
    """
    import lmdb
    from poker_ai.tables.index import lmdb_map_size_for_players

    # Read-only opens only need a map_size >= the source's used extent; the
    # canonical per-player reservation is what created it, so it always fits.
    map_size = lmdb_map_size_for_players(int(n_players))
    dst_index.mkdir(parents=True, exist_ok=True)
    for r in range(4):
        src_street = src_index / f"street_{r}"
        if not src_street.exists():
            raise FileNotFoundError(
                f"LMDB street index missing: {src_street} — is {src_index} a "
                f"training run's index directory?"
            )
        dst_street = dst_index / f"street_{r}"
        dst_street.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(
            str(src_street), map_size=map_size, subdir=True,
            readonly=True, lock=False,
        )
        try:
            # compact=True repacks the B-tree omitting free pages → smallest
            # long-lived blueprint index; cost is a one-shot tree walk, tiny
            # next to reading the run's regret snapshots.
            env.copy(str(dst_street), compact=True)
        finally:
            env.close()


def build_final_blueprint(
    train_dir: Path,
    out_dir: Path,
    scale: int = SIGMA_SCALE_DEFAULT,
    min_t: Optional[int] = None,
) -> Path:
    """Build a final blueprint by averaging a run's retained snapshots.

    Parameters
    ----------
    train_dir : Path
        Training directory containing ``lmdb_index/`` and the retained
        ``checkpoint_<t>/`` generations.
    out_dir : Path
        Destination directory (created; must not already contain a blueprint).
    scale : int, optional
        Integer scale for the stored strategy pseudo-counts.
    min_t : int, optional
        Exclude snapshots whose iteration ``t`` is below this from the
        post-flop average.  Defaults to the warm-up ``checkpoint_start_cycles *
        sync_interval`` recorded in the latest checkpoint (0 if absent), which
        drops any sub-warm-up end-of-run checkpoint from the average.  The
        latest checkpoint is always used for the regret / pre-flop copy-through
        regardless of this filter.

    Returns
    -------
    Path
        The created ``out_dir``.
    """
    train_dir = Path(train_dir)
    out_dir = Path(out_dir)

    checkpoints = sorted(train_dir.glob("checkpoint_[0-9]*"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint_* directories found under {train_dir} — nothing to "
            f"average (was the run configured to retain checkpoints?)."
        )

    # Order by the iteration counter recorded in each state dict.
    states = {cp: _load_state(cp) for cp in checkpoints}
    checkpoints.sort(key=lambda cp: states[cp]["t"])
    final_dir = checkpoints[-1]
    final_state = states[final_dir]

    # Consistency: every snapshot must share the run's structural layout, or
    # positional averaging across them is meaningless.
    for key in ("n_players", "chunk_size", "info_set_encoding"):
        final_val = final_state.get(key)
        for cp in checkpoints:
            val = states[cp].get(key)
            if val != final_val:
                raise ValueError(
                    f"Snapshot {cp.name} disagrees with the final checkpoint on "
                    f"{key!r} (saw {val!r}, expected {final_val!r}). These "
                    f"checkpoints are not from the same run/abstraction and "
                    f"cannot be averaged."
                )

    if min_t is None:
        start_cycles = int(final_state.get("checkpoint_start_cycles", 0) or 0)
        sync_interval = int(final_state.get("sync_interval", 1) or 1)
        min_t = start_cycles * sync_interval

    avg_snapshots = [cp for cp in checkpoints if states[cp]["t"] >= min_t]
    if not avg_snapshots:
        raise ValueError(
            f"No snapshot has t >= min_t={min_t:,}; every retained checkpoint is "
            f"below the warm-up. Lower --min_t (e.g. 0) to average the "
            f"available checkpoints anyway."
        )
    log.info(
        "Averaging %d/%d snapshots (t in [%s, %s], min_t=%s) into %s",
        len(avg_snapshots), len(checkpoints),
        f"{states[avg_snapshots[0]]['t']:,}", f"{states[avg_snapshots[-1]]['t']:,}",
        f"{min_t:,}", out_dir,
    )

    # ------------------------------------------------------------------
    # Assemble the output directory.
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    out_index = out_dir / "lmdb_index"
    if out_index.exists():
        raise FileExistsError(
            f"{out_index} already exists — refusing to overwrite an existing "
            f"blueprint. Choose a fresh --output_dir."
        )
    _copy_lmdb_index(
        train_dir / "lmdb_index", out_index, int(final_state["n_players"])
    )

    out_cp = out_dir / f"checkpoint_{final_state['t']}"
    out_cp.mkdir(parents=True, exist_ok=True)

    # Regret chunks (all streets) + pre-flop strategy carry through from the
    # latest checkpoint: the regret side keeps the readout's regret-matching
    # fallback live and satisfies validate_chunks; pre-flop strategy is the
    # trained running average and is used verbatim.
    for r in _ALL_STREETS:
        for chunk_id in _chunk_ids(final_dir, f"regret_{r}"):
            name = f"regret_{r}_chunk_{chunk_id:06d}.npy"
            _link_or_copy(final_dir / name, out_cp / name)
    for chunk_id in _chunk_ids(final_dir, "strategy_0"):
        name = f"strategy_0_chunk_{chunk_id:06d}.npy"
        _link_or_copy(final_dir / name, out_cp / name)

    # Post-flop strategy: the offline snapshot average.
    for r in _POSTFLOP_STREETS:
        n = average_street(avg_snapshots, final_dir, r, out_cp, scale)
        log.info("street %d: wrote %d averaged strategy chunk(s)", r, n)

    _link_or_copy(final_dir / "server_state.pkl", out_cp / "server_state.pkl")

    log.info("Final blueprint written to %s (checkpoint %s)", out_dir, out_cp.name)
    return out_dir


def _cli(train_dir: str, output_dir: str, scale: int, min_t: Optional[int]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_final_blueprint(Path(train_dir), Path(output_dir), scale=scale, min_t=min_t)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scale", type=int, default=SIGMA_SCALE_DEFAULT)
    parser.add_argument("--min_t", type=int, default=None)
    args = parser.parse_args()
    _cli(args.train_dir, args.output_dir, args.scale, args.min_t)
