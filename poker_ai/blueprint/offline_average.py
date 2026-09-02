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
the snapshots **that had a real regret-matching opinion for that row** — not
every snapshot the row is merely present in. A snapshot whose regret row for
an infoset never went positive contributes nothing (see :func:`average_chunk`):
its would-be maskless-uniform placeholder is excluded from both the sum and
the divisor, rather than diluting the mean of snapshots that *did* converge on
something. A row with NO retained snapshot ever recording positive regret for
it is written all-zero; at read time this correctly falls back to live regret
matching (:meth:`poker_ai.search.policy.BlueprintPolicy._average_strategy`,
mass 0 < ``min_strategy_mass``), which itself degrades to uniform-over-legal
for such a row (its regret is all-non-positive there too) — so the visible
behaviour for a genuinely never-converged infoset is unchanged, it just isn't
baked redundantly into the stored table, and the fallback masks to the node's
actual legal actions rather than spreading mass over every column.

The post-flop average is stored as scaled integer pseudo-counts (the strategy
tables are ``int32`` visit counts), so a row that averaged the probability
vector ``p`` is written as ``round(SIGMA_SCALE * p)``.  At read time
:meth:`poker_ai.search.policy.BlueprintPolicy._average_strategy` masks the row
to the legal actions and renormalises, so the scale cancels and the
``min_strategy_mass`` gate (default 10) passes comfortably (row mass
``~SIGMA_SCALE`` = 1e6).

A row also needs independent confirmation from enough retained snapshots
before it is published at all — one confirming snapshot alone is a single
categorical sample of an evolving strategy (regret matching's per-touch delta
is ``voa[action] - vo`` against that traversal's own mean, so the
best-performing action of a single touch is positive almost by construction;
it does not mean the row has converged). The requirement is
``max(MIN_CONFIRMING_SNAPSHOTS_DEFAULT, ceil(MIN_CONFIRMING_FRACTION_DEFAULT *
n_snapshots_averaged))`` — the absolute floor alone doesn't scale: on a run
with dozens of retained snapshots, "confirmed by any 2 of them" stays a very
low bar (a single-touch positive-regret blip only needs to land in 2 out of,
say, 81 snapshots, which is close to certain for anything touched at all
across a long run), so the fraction requirement — a majority by default —
usually dominates. Optionally, :data:`MIN_SNAPSHOT_REGRET_MAGNITUDE_DEFAULT`
raises the bar further by requiring each confirming snapshot's own positive
regret to clear a magnitude threshold too, not just be positive at all (unset
by default — no natural cutoff exists across differently-scaled runs, so this
needs calibrating per run rather than trusting a guess). Short of the
effective requirement, the row is written all-zero and — as above — correctly
deferred to the live regret-match fallback rather than publishing a
falsely-confident average. This keeps a published row's mass a genuine
confidence signal again (as it already was for pre-flop's real online visit
counts), rather than every row trivially clearing ``min_strategy_mass``
regardless of how much training it actually saw — so a plain mass check
(``evaluation/blueprint_metrics.py``, ``BlueprintPolicy``) is meaningful for
post-flop without any special-casing.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import joblib
import numpy as np
from tqdm import tqdm

from utils.io import atomic_numpy_save

log = logging.getLogger("poker_ai.blueprint.offline_average")

#: Approximate peak resident memory of one in-flight ``(street, chunk)`` task at
#: the default ``PLURIBUS_CHUNK_SIZE`` (4M rows): the float64 accumulator, the
#: int64 presence counter, one int32 snapshot chunk, and NumPy's temporaries.
#: Measured ~650-800 MB.  Total peak ≈ ``workers * _TASK_PEAK_MB``, which is how
#: ``--workers`` doubles as the memory dial.
_TASK_PEAK_MB = 800

#: Integer scale applied to the averaged probability vector before it is stored
#: in the ``int32`` strategy tables.  A row that averaged ``p`` (summing to 1
#: over the allocated columns) is written as ``round(SIGMA_SCALE * p)``, so the
#: stored row sums to ~``SIGMA_SCALE`` — far above ``min_strategy_mass`` and
#: well within ``int32`` range.
SIGMA_SCALE_DEFAULT = 1_000_000

#: How the retained snapshots are weighted when averaged.  ``"linear"`` — the
#: default — weights snapshot ``s`` by its iteration count ``t_s``, echoing Linear
#: CFR's ``t``-weighting of the running average: the true average strategy is
#: ``sum_t pi^t sigma^t / sum_t pi^t`` (``t``-weighted under Linear CFR), so the
#: ``"equal"`` mean lets an early, less-converged snapshot count as much as the
#: final one.  ``"linear"`` recovers the recency half of that; the reach (``pi``)
#: half is not available offline without a forward pass per snapshot.
#:
#: ``"equal"`` is the historical behaviour, kept for reproducing older blueprints.
#:
#: The default was flipped to ``"linear"`` on measured evidence, not theory — a
#: duplicate-paired head-to-head over 600k hands on the 20-card 2p blueprint:
#: linear beat equal by **+2.53 +/- 1.29 bb/100**, and a reverse-weighted control
#: (early snapshots heavy) *lost* to equal by **-3.70 +/- 1.61**, giving the
#: monotone ordering reverse < equal < linear.  The opposite-signed control is
#: what makes it recency rather than "any perturbation helps".
#:
#: ⚠The size of the effect is set by the spread of ``t`` across the retained
#: snapshots (13.3x in that run: 26 snapshots, t from 940k to 12.5M).  A run with
#: a different checkpoint schedule will not see the same magnitude.
#: ⚠Do NOT try to settle this kind of question with
#: ``evaluation/blueprint_metrics.py`` — its post-flop numbers cannot distinguish
#: the two artifacts at all (see that module's ``play_freq`` note).
SNAPSHOT_WEIGHTING_DEFAULT = "linear"
SNAPSHOT_WEIGHTINGS = ("equal", "linear")

#: Minimum number of RETAINED SNAPSHOTS that must independently record a
#: positive-regret action for a post-flop row before its averaged strategy is
#: published with full (trusted) mass.  A row confirmed by only one snapshot is
#: a single categorical sample of an evolving strategy — exactly the situation
#: :class:`poker_ai.search.policy.BlueprintPolicy` already distrusts for
#: pre-flop's real visit counts ("a row visited once ... is worse than regret
#: matching", see its docstring) — so the same standard is applied here rather
#: than inventing a payoff-scale-dependent regret-magnitude threshold (there is
#: no natural default for one; this constant needs none, since "reconfirmed
#: independently at least twice" is meaningful regardless of stakes or chunk
#: size). Below this, the row is written all-zero and deferred to the live
#: regret-match fallback (see :func:`average_chunk`).
#:
#: This is a FLOOR, not the only requirement — see
#: :data:`MIN_CONFIRMING_FRACTION_DEFAULT`, which usually dominates it on any
#: run with more than a handful of retained snapshots.
MIN_CONFIRMING_SNAPSHOTS_DEFAULT = 2

#: Minimum FRACTION of all averaged snapshots that must independently confirm
#: a row (in addition to, not instead of, the absolute floor above): the
#: effective requirement is ``max(MIN_CONFIRMING_SNAPSHOTS_DEFAULT,
#: ceil(MIN_CONFIRMING_FRACTION_DEFAULT * n_snapshots_averaged))``.  The
#: absolute floor alone doesn't scale: on a run with dozens of retained
#: snapshots, "confirmed by any 2 of them" is a very low bar — a row only
#: needs to get lucky (a single-touch positive blip, see
#: :data:`MIN_CONFIRMING_SNAPSHOTS_DEFAULT`'s docstring) in 2 out of, say, 81
#: snapshots, which is nearly certain for anything touched at all across a
#: long run. Expressing the bar as a fraction of the run's OWN snapshot count
#: keeps it meaningful regardless of how many checkpoints were retained.
#: 0.5 (a majority) is the natural, easily-justified default: it asks whether
#: the row's positive-regret evidence held up across most of training's span,
#: not just a couple of scattered points in it.
MIN_CONFIRMING_FRACTION_DEFAULT = 0.5

#: Optional minimum ``sum(positive regret)`` a SINGLE snapshot's row must show
#: before that snapshot counts toward confirming a row at all (on top of, not
#: instead of, the plain positivity test). ``None`` (default) keeps the plain
#: "any positive" per-snapshot test. Unlike the two constants above, there is
#: NO built-in default value here deliberately: what counts as "a meaningful
#: amount of regret" for one snapshot depends on this run's chip/payoff scale
#: (stakes, bet-sizing abstraction), the same problem that ruled out a
#: magnitude threshold as the sole fix earlier — see
#: [[project_blueprint_metrics_leaf_coverage_gap]]. Calibrate it for a specific
#: run (e.g. via a throwaway inspection of a few snapshots' regret chunks)
#: rather than trusting a guessed constant across configurations.
MIN_SNAPSHOT_REGRET_MAGNITUDE_DEFAULT: Optional[int] = None

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


def average_chunk(
    snapshot_dirs: List[Path],
    final_dir: Path,
    r: int,
    chunk_id: int,
    out_dir: Path,
    scale: int,
    resume: bool = False,
    min_confirming_snapshots: int = MIN_CONFIRMING_SNAPSHOTS_DEFAULT,
    min_confirming_fraction: float = MIN_CONFIRMING_FRACTION_DEFAULT,
    min_snapshot_regret_magnitude: Optional[int] = MIN_SNAPSHOT_REGRET_MAGNITUDE_DEFAULT,
    snapshot_weights: Optional[Sequence[float]] = None,
) -> bool:
    """Average one ``(street, chunk)`` across snapshots; the unit of work.

    Self-contained and independent of every other chunk — it reads only its own
    chunk file from each snapshot and writes only its own output — which is what
    makes the build both parallelisable (``workers``) and resumable (``resume``).

    Peak memory is one float64 accumulator + one int64 counter + one snapshot
    chunk (~``_TASK_PEAK_MB``), independent of the snapshot count or the run's
    total on-disk size.

    A row is published (its average written with real, ``min_strategy_mass``-
    clearing mass) only if it is independently confirmed by at least
    ``max(min_confirming_snapshots, ceil(min_confirming_fraction *
    len(snapshot_dirs)))`` snapshots — see :data:`MIN_CONFIRMING_SNAPSHOTS_DEFAULT`
    and :data:`MIN_CONFIRMING_FRACTION_DEFAULT`. A snapshot counts toward that
    tally if its row shows any positive regret, or — when
    ``min_snapshot_regret_magnitude`` is set — only if its positive-regret sum
    also clears that bar (see :data:`MIN_SNAPSHOT_REGRET_MAGNITUDE_DEFAULT`).
    Short of the confirmation requirement, the row is written all-zero,
    deferring to the live regret-match fallback.

    ``snapshot_weights`` (aligned with *snapshot_dirs*) makes the per-row mean a
    **weighted** mean.  ``None`` — the default — is an unweighted mean, byte-for-byte
    the historical behaviour.  The weights scale only the averaging; the
    confirmation tally stays an unweighted **count** of snapshots, so a heavy
    snapshot cannot publish a row on its own.

    Returns
    -------
    bool
        ``True`` if the chunk was computed and written, ``False`` if it was
        skipped because a complete output already existed (``resume``).
    """
    prefix = f"regret_{r}"
    out_path = out_dir / f"strategy_{r}_chunk_{chunk_id:06d}.npy"
    if resume and out_path.exists():
        # Outputs are written atomically (temp + rename), so a file that exists
        # is complete — safe to skip.
        return False

    # Shape only: mmap the header so we never pull the ~80 MB final chunk into
    # RAM here (it is read again, once, inside the snapshot loop).
    n_rows, n_actions = np.load(
        final_dir / f"{prefix}_chunk_{chunk_id:06d}.npy", mmap_mode="r"
    ).shape
    acc = np.zeros((n_rows, n_actions), dtype=np.float64)
    count = np.zeros(n_rows, dtype=np.int64)   # unweighted: the confirmation tally
    wsum = np.zeros(n_rows, dtype=np.float64)  # weighted: the mean's divisor

    if snapshot_weights is None:
        weights: Sequence[float] = [1.0] * len(snapshot_dirs)
    elif len(snapshot_weights) != len(snapshot_dirs):
        raise ValueError(
            f"snapshot_weights has {len(snapshot_weights)} entries but there are "
            f"{len(snapshot_dirs)} snapshots — they must align positionally."
        )
    else:
        weights = snapshot_weights

    for snap, w in zip(snapshot_dirs, weights):
        path = snap / f"{prefix}_chunk_{chunk_id:06d}.npy"
        if not path.exists():
            # This snapshot had not allocated this chunk yet.
            continue
        arr = np.load(path)
        k = min(arr.shape[0], n_rows)  # earlier snapshots may hold fewer rows
        chunk = arr[:k]
        # A row with no positive regret in THIS snapshot (or, if
        # min_snapshot_regret_magnitude is set, without ENOUGH positive
        # regret) has no real signal — sigma_from_regret_chunk's
        # maskless-uniform fallback for it is a pure placeholder, not a
        # genuine regret-matched opinion. Excluded from both the accumulator
        # and the divisor (not just zeroed, which would still dilute the mean
        # for rows with real signal in other snapshots): a row trained late
        # keeps its average over only the snapshots where it had something to
        # say, rather than being watered down by early snapshots' placeholder
        # uniform contributions.
        pos_sum = np.clip(chunk, 0, None).sum(axis=1)
        if min_snapshot_regret_magnitude is not None:
            has_signal = pos_sum >= min_snapshot_regret_magnitude
        else:
            has_signal = pos_sum > 0.0
        if has_signal.any():
            sigma = sigma_from_regret_chunk(chunk)
            acc[:k][has_signal] += float(w) * sigma[has_signal]
            count[:k][has_signal] += 1
            wsum[:k][has_signal] += float(w)
        del arr  # release the ~80 MB snapshot chunk before the next load

    # The fraction requirement scales with THIS run's own snapshot count, so
    # "confirmed by 2" doesn't become a trivial bar on a run with dozens of
    # retained snapshots (see MIN_CONFIRMING_FRACTION_DEFAULT's docstring for
    # why the absolute floor alone isn't enough at scale).
    required = max(
        min_confirming_snapshots,
        int(np.ceil(min_confirming_fraction * len(snapshot_dirs))),
    )
    out = np.zeros((n_rows, n_actions), dtype=np.int32)
    present = count >= required
    if present.any():
        mean = acc[present] / wsum[present][:, None]
        out[present] = np.rint(scale * mean).astype(np.int32)
    # Rows present in no snapshot, present but never showing (sufficient)
    # positive regret, or confirmed by fewer than `required` independent
    # snapshots, stay all-zero. At read time this correctly falls back to live
    # regret matching (mass 0 < min_strategy_mass, see
    # poker_ai.search.policy.BlueprintPolicy._average_strategy) — which
    # degrades to uniform-over-LEGAL-actions for a never-converged row too
    # (its regret is all-non-positive there as well), so the visible
    # behaviour for such a row is unchanged from the old maskless uniform
    # default, minus (a) wasted placeholder mass baked into every
    # under-explored row, and (b) that default's illegal-column leakage (it
    # spread mass over every action column regardless of the node's actual
    # legal set; the live fallback masks to state.valid_mask).

    # Atomic: a killed job never leaves a truncated .npy that a later --resume
    # would mistake for finished work.
    atomic_numpy_save(out, out_path)
    return True


def _average_chunk_task(args: Tuple) -> bool:
    """Picklable ``ProcessPoolExecutor`` entry point for :func:`average_chunk`."""
    return average_chunk(*args)


def _run_chunk_tasks(
    tasks: List[Tuple], workers: int, desc: str = "Averaging chunks"
) -> int:
    """Run ``(street, chunk)`` tasks serially or across a process pool.

    Returns the number of chunks actually written (skipped-by-resume excluded).
    Processes, not threads: each task's peak memory is then private and the
    total ceiling is exactly ``workers * _TASK_PEAK_MB``.

    Progress is reported per chunk as it *completes* (not submission order —
    chunks vary hugely in size, e.g. river dwarfs flop/turn, so completion
    order is the only one that reflects real progress).
    """
    if not tasks:
        return 0
    if workers <= 1:
        return sum(
            _average_chunk_task(t) for t in tqdm(tasks, desc=desc, unit="chunk")
        )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_average_chunk_task, t) for t in tasks]
        written = 0
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc=desc, unit="chunk"
        ):
            written += fut.result()
        return written


def average_street(
    snapshot_dirs: List[Path],
    final_dir: Path,
    r: int,
    out_dir: Path,
    scale: int,
    workers: int = 1,
    resume: bool = False,
    min_confirming_snapshots: int = MIN_CONFIRMING_SNAPSHOTS_DEFAULT,
    min_confirming_fraction: float = MIN_CONFIRMING_FRACTION_DEFAULT,
    min_snapshot_regret_magnitude: Optional[int] = MIN_SNAPSHOT_REGRET_MAGNITUDE_DEFAULT,
    snapshot_weights: Optional[Sequence[float]] = None,
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
    workers : int, optional
        Chunks to process concurrently (peak RAM ≈ ``workers * 800 MB``).
    resume : bool, optional
        Skip chunks whose output already exists.
    min_confirming_snapshots, min_confirming_fraction, min_snapshot_regret_magnitude
        See :func:`average_chunk` and the matching ``MIN_*_DEFAULT`` constants.

    Returns
    -------
    int
        Number of strategy chunks written for this street (skipped excluded).
    """
    tasks = [
        (snapshot_dirs, final_dir, r, chunk_id, out_dir, scale, resume,
         min_confirming_snapshots, min_confirming_fraction,
         min_snapshot_regret_magnitude, snapshot_weights)
        for chunk_id in _chunk_ids(final_dir, f"regret_{r}")
    ]
    return _run_chunk_tasks(tasks, workers, desc=f"street {r}")


def _load_state(checkpoint_dir: Path) -> dict:
    return joblib.load(checkpoint_dir / "server_state.pkl")


def _link_or_copy(src: Path, dst: Path, resume: bool = False) -> None:
    """Atomically materialise *src* at *dst* (hardlink, else copy).

    ``os.link`` is already atomic; the cross-filesystem ``copy2`` fallback is
    made atomic with a temp file + ``os.replace`` so an interrupted build never
    leaves a truncated file that a later ``--resume`` would treat as complete.
    """
    if resume and dst.exists():
        return
    try:
        os.link(src, dst)
        return
    except FileExistsError:
        return
    except OSError:
        pass
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


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
        if dst_street.exists():
            # Complete (the rename below is the last step) — resume past it.
            continue
        # Copy into a temp dir and rename: LMDB refuses to write into a
        # non-empty dir, and an interrupted copy must not leave a half-written
        # street_{r} that a later --resume would trust.
        tmp_street = dst_index / f"street_{r}.tmp"
        if tmp_street.exists():
            shutil.rmtree(tmp_street)
        tmp_street.mkdir(parents=True)
        # This runs single-threaded in the main process, ahead of the
        # --workers-parallelized chunk averaging below, and a large street's
        # B-tree walk can take a while — log around it so it doesn't read as
        # a hang.
        log.info("Copying LMDB index street_%d (compacting)...", r)
        t0 = time.time()
        env = lmdb.open(
            str(src_street), map_size=map_size, subdir=True,
            readonly=True, lock=False,
        )
        try:
            # compact=True repacks the B-tree omitting free pages → smallest
            # long-lived blueprint index; cost is a one-shot tree walk, tiny
            # next to reading the run's regret snapshots.
            env.copy(str(tmp_street), compact=True)
        finally:
            env.close()
        os.replace(tmp_street, dst_street)
        log.info("street_%d copied in %.1fs", r, time.time() - t0)


def build_final_blueprint(
    train_dir: Path,
    out_dir: Path,
    scale: int = SIGMA_SCALE_DEFAULT,
    min_t: Optional[int] = None,
    workers: int = 1,
    resume: bool = False,
    min_confirming_snapshots: int = MIN_CONFIRMING_SNAPSHOTS_DEFAULT,
    min_confirming_fraction: float = MIN_CONFIRMING_FRACTION_DEFAULT,
    min_snapshot_regret_magnitude: Optional[int] = MIN_SNAPSHOT_REGRET_MAGNITUDE_DEFAULT,
    snapshot_weighting: str = SNAPSHOT_WEIGHTING_DEFAULT,
) -> Path:
    """Build a final blueprint by averaging a run's retained snapshots.

    Parameters
    ----------
    train_dir : Path
        Training directory containing ``lmdb_index/`` and the retained
        ``checkpoint_<t>/`` generations.
    out_dir : Path
        Destination directory (created; must not already contain a blueprint
        unless ``resume``).
    scale : int, optional
        Integer scale for the stored strategy pseudo-counts.
    min_t : int, optional
        Exclude snapshots whose iteration ``t`` is below this from the
        post-flop average.  Defaults to the warm-up ``checkpoint_start_cycles *
        sync_interval`` recorded in the latest checkpoint (0 if absent), which
        drops any sub-warm-up end-of-run checkpoint from the average.  The
        latest checkpoint is always used for the regret / pre-flop copy-through
        regardless of this filter.
    workers : int, optional
        ``(street, chunk)`` tasks to process concurrently.  Each task's peak
        memory is private, so the total ceiling is ≈ ``workers * 800 MB`` —
        this is the memory dial.  Default 1 (serial, ~800 MB).
    resume : bool, optional
        Continue into an existing *out_dir*, skipping artefacts that are
        already complete.  Every output is written atomically, so anything
        present is finished and safe to skip.  Use after an interrupted build.
    min_confirming_snapshots, min_confirming_fraction, min_snapshot_regret_magnitude
        See :func:`average_chunk` and the matching ``MIN_*_DEFAULT`` constants.
        A post-flop row not independently confirmed by enough of the averaged
        snapshots is written all-zero instead of published, deferring to the
        live regret-match fallback at read time.

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
    if snapshot_weighting not in SNAPSHOT_WEIGHTINGS:
        raise ValueError(
            f"unknown snapshot_weighting {snapshot_weighting!r}; expected one of "
            f"{SNAPSHOT_WEIGHTINGS}"
        )
    if snapshot_weighting == "linear":
        # Normalised by the largest t purely to keep the accumulator O(1)-scaled;
        # any positive rescaling leaves the weighted mean unchanged.
        t_max = float(states[avg_snapshots[-1]]["t"])
        snapshot_weights: Optional[List[float]] = [
            float(states[cp]["t"]) / t_max for cp in avg_snapshots
        ]
        log.info(
            "Snapshot weighting 'linear': weights span %.3f (t=%s) to %.3f (t=%s) "
            "— a %.1fx spread the equal-weight mean would have flattened.",
            snapshot_weights[0], f"{states[avg_snapshots[0]]['t']:,}",
            snapshot_weights[-1], f"{states[avg_snapshots[-1]]['t']:,}",
            snapshot_weights[-1] / snapshot_weights[0] if snapshot_weights[0] else float("inf"),
        )
    else:
        snapshot_weights = None

    effective_required = max(
        min_confirming_snapshots,
        int(np.ceil(min_confirming_fraction * len(avg_snapshots))),
    )
    log.info(
        "Averaging %d/%d snapshots (t in [%s, %s], min_t=%s) into %s — a "
        "post-flop row must be independently confirmed by >= %d/%d of them "
        "(min_confirming_snapshots=%d, min_confirming_fraction=%s%s) to be "
        "published",
        len(avg_snapshots), len(checkpoints),
        f"{states[avg_snapshots[0]]['t']:,}", f"{states[avg_snapshots[-1]]['t']:,}",
        f"{min_t:,}", out_dir,
        effective_required, len(avg_snapshots),
        min_confirming_snapshots, min_confirming_fraction,
        f", min_snapshot_regret_magnitude={min_snapshot_regret_magnitude}"
        if min_snapshot_regret_magnitude is not None else "",
    )
    if len(avg_snapshots) < min_confirming_snapshots:
        log.warning(
            "Only %d snapshot(s) >= min_t are being averaged, but "
            "min_confirming_snapshots=%d — no post-flop row can ever be "
            "confirmed, so the entire post-flop average will be written "
            "empty (every row deferred to the live regret-match fallback). "
            "Lower --min_t to include more snapshots, or pass a lower "
            "min_confirming_snapshots, if that isn't intended.",
            len(avg_snapshots), min_confirming_snapshots,
        )

    # ------------------------------------------------------------------
    # Assemble the output directory.
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    out_index = out_dir / "lmdb_index"
    if out_index.exists() and not resume:
        raise FileExistsError(
            f"{out_index} already exists — refusing to overwrite an existing "
            f"blueprint. Choose a fresh --output_dir, or pass --resume to "
            f"continue an interrupted build."
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
            _link_or_copy(final_dir / name, out_cp / name, resume=resume)
    for chunk_id in _chunk_ids(final_dir, "strategy_0"):
        name = f"strategy_0_chunk_{chunk_id:06d}.npy"
        _link_or_copy(final_dir / name, out_cp / name, resume=resume)

    # Post-flop strategy: the offline snapshot average.  All post-flop chunks
    # go through ONE pool rather than a pool per street — the river alone is
    # ~75% of the work, so a flat task list keeps every worker busy to the end
    # instead of draining at each street boundary.
    tasks = [
        (avg_snapshots, final_dir, r, chunk_id, out_cp, scale, resume,
         min_confirming_snapshots, min_confirming_fraction,
         min_snapshot_regret_magnitude, snapshot_weights)
        for r in _POSTFLOP_STREETS
        for chunk_id in _chunk_ids(final_dir, f"regret_{r}")
    ]
    log.info(
        "Averaging %d post-flop chunk(s) with %d worker(s) "
        "(peak RAM ≈ %d x %d MB)%s",
        len(tasks), workers, workers, _TASK_PEAK_MB,
        " [resume: complete chunks skipped]" if resume else "",
    )
    written = _run_chunk_tasks(tasks, workers, desc="post-flop chunks")
    log.info(
        "Post-flop strategy: %d chunk(s) written, %d already complete",
        written, len(tasks) - written,
    )

    _link_or_copy(
        final_dir / "server_state.pkl", out_cp / "server_state.pkl", resume=resume
    )

    log.info("Final blueprint written to %s (checkpoint %s)", out_dir, out_cp.name)
    return out_dir


def _cli(
    train_dir: str,
    output_dir: str,
    scale: int,
    min_t: Optional[int],
    workers: int = 1,
    resume: bool = False,
    min_confirming_snapshots: int = MIN_CONFIRMING_SNAPSHOTS_DEFAULT,
    min_confirming_fraction: float = MIN_CONFIRMING_FRACTION_DEFAULT,
    min_snapshot_regret_magnitude: Optional[int] = MIN_SNAPSHOT_REGRET_MAGNITUDE_DEFAULT,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_final_blueprint(
        Path(train_dir), Path(output_dir), scale=scale, min_t=min_t,
        workers=workers, resume=resume,
        min_confirming_snapshots=min_confirming_snapshots,
        min_confirming_fraction=min_confirming_fraction,
        min_snapshot_regret_magnitude=min_snapshot_regret_magnitude,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scale", type=int, default=SIGMA_SCALE_DEFAULT)
    parser.add_argument("--min_t", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--min_confirming_snapshots", type=int,
        default=MIN_CONFIRMING_SNAPSHOTS_DEFAULT,
    )
    parser.add_argument(
        "--min_confirming_fraction", type=float,
        default=MIN_CONFIRMING_FRACTION_DEFAULT,
    )
    parser.add_argument(
        "--min_snapshot_regret_magnitude", type=int,
        default=MIN_SNAPSHOT_REGRET_MAGNITUDE_DEFAULT,
    )
    args = parser.parse_args()
    _cli(
        args.train_dir, args.output_dir, args.scale, args.min_t,
        workers=args.workers, resume=args.resume,
        min_confirming_snapshots=args.min_confirming_snapshots,
        min_confirming_fraction=args.min_confirming_fraction,
        min_snapshot_regret_magnitude=args.min_snapshot_regret_magnitude,
    )
