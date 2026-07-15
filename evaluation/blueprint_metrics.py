"""Standalone blueprint strategy sanity check (human-readable + JSON).

``python -m evaluation.blueprint_metrics <blueprint_path>`` reads the latest
``checkpoint_*`` directory of a trained blueprint and answers one question —
*did the blueprint actually learn a meaningful strategy, or is it still uniform
noise?* — printed as a human-readable block and written next to the blueprint as
``blueprint_metrics.md`` + ``blueprint_metrics.json`` for cross-run comparison.

Blueprints are large (200 GB+), so the default is a **sampled sanity check**:
only a few evenly-spaced chunks per street are read (``--sample-chunks``, default
4), which is enough to tell a trained strategy from an untrained one while
touching a few GB instead of the whole table.  ``--full`` reads every chunk for
exact frequencies.  Regret health is a training diagnostic, not a "did it learn"
signal, so it is **off by default** (``--regret`` to include it, doubling I/O).

Headline metrics, per street:

- **Play frequencies** (visit-weighted): the strategy tables store the average
  strategy as visit counts accumulated by the strategy-sampling traversal
  (:mod:`poker_ai.blueprint.strategy`), so the column-sum over all infosets,
  normalised by the total mass, is the frequency with which the blueprint
  plays each abstract action when it acts on that street under its own play.
  This is the "fold / call / raise probability across streets" headline and
  the primary sanity signal (uniform → untrained; sharply skewed → learned).
- **Per-infoset mean strategy** (unweighted): each visited row normalised to a
  distribution and averaged with equal weight, so rarely-reached infosets
  count as much as common ones.  Divergence from the visit-weighted numbers
  indicates the strategy differs sharply between hot and cold parts of the tree.
- **Determinism**: per-row normalised entropy and max-action-probability
  distributions — how mixed vs. pure the strategy is.  An all-uniform blueprint
  sits at entropy 1.0; a trained one sharpens, especially on later streets.
- **Coverage / training mass**: allocated infosets, visited fraction, and the
  distribution of per-row visit mass (log10 histogram) — how well trained the
  street is.
- **Regret health** (opt-in): fraction of regret entries at the
  :data:`~poker_ai.tables.cfr_tables.REGRET_FLOOR`, below the Pluribus prune
  threshold, and with positive regret.
- **Leaf coverage** (opt-in, ``--leaf-coverage``): the metric that matters when
  the blueprint is queried as a depth-limited-search *leaf* continuation.
  :class:`~poker_ai.search.policy.BlueprintPolicy` resolves each infoset to the
  **average** strategy when its visit mass is trustworthy (``>=
  min_strategy_mass``), else falls back to regret-matching the (far denser)
  **regret** row, else to uniform.  This mode row-joins the strategy and regret
  tables — valid because both share the per-street infoset index, so row *i* is
  the same infoset in both — and reports, per street, the fraction of infosets a
  leaf query would resolve to a trusted average (``avg_trusted``), to a regret
  fallback (``regret_fallback``), or to uniform (``uniform``).  ``effective`` =
  ``avg_trusted + regret_fallback`` is the fraction that yields a real (non-
  uniform) blueprint opinion — the true leaf coverage, which the strategy-table
  ``visited_frac`` alone understates.

Design points:

- **No LMDB, no LUT, no /dev/shm.**  Saved chunk files are trimmed to their
  valid row prefix (:meth:`ChunkStore._save
  <poker_ai.tables.chunk_store.ChunkStore._save>`), so the ``.npy`` files are
  self-describing: row counts come from file shapes and the analysis is a pure
  streaming pass with ``np.load(mmap_mode="r")``.  Memory is O(batch), so a
  full-size blueprint is analysed anywhere the checkpoint directory is
  mounted — no staging, no RAM footprint.
- **Sampling stays honest about the total.**  Evenly-spaced chunk sampling
  always includes chunk 0 (always a full chunk) and the final chunk (the only
  possibly-partial one), so the exact total infoset count is recovered from
  two file shapes even when only a handful of chunks are read.  Frequencies
  are labelled as estimates; the coverage total is exact.
- **Column semantics come from the file, labels from the action space.**  The
  canonical ordering is ``["fold", "call", "all_in", "raise:<f>", ...]``
  (:meth:`PokerEnv.get_canonical_actions
  <environment.poker_env.PokerEnv.get_canonical_actions>`).  When the chunk
  width matches the current :data:`~environment.action_space.CANONICAL_ACTIONS`
  the real labels are used; otherwise (checkpoint trained under a different
  action config) the first three columns are still fold/call/all_in by
  construction and the remaining raise columns get generic ``raise_#k`` labels.
- Percentiles are interpolated from fixed-bin histograms accumulated during
  the streaming pass, so no per-row values are ever materialised.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("evaluation.blueprint_metrics")

_STREET_NAME = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}

# Pluribus supplementary: actions whose regret sits below this threshold are
# skipped by negative-regret pruning during training.  Reported as the
# "prune-eligible" fraction; distinct from the hard REGRET_FLOOR clamp.
PRUNE_THRESHOLD: int = -300_000_000

_ENTROPY_BINS = 100          # normalised entropy histogram over [0, 1]
_MAXP_BINS = 100             # max-action-probability histogram over [0, 1]
_MASS_BINS = 120             # log10(row visit mass) histogram over [0, 12]
_MASS_LOG10_MAX = 12.0
_DEFAULT_BATCH_ROWS = 1_000_000
_DEFAULT_SAMPLE_CHUNKS = 4   # chunks/street read by the CLI sanity check

# Minimum strategy-row visit mass for BlueprintPolicy to trust the average
# strategy at a leaf; below it the policy falls back to regret matching.
# Mirrors the ``min_strategy_mass`` default in
# :class:`poker_ai.search.policy.BlueprintPolicy` — keep in sync.
_DEFAULT_MIN_STRATEGY_MASS = 10


# --------------------------------------------------------------------------- #
# Checkpoint discovery
# --------------------------------------------------------------------------- #


def _resolve_checkpoint(blueprint_path: Path, checkpoint: Optional[str]) -> Path:
    """Return the checkpoint directory to analyse.

    ``checkpoint`` selects a specific ``checkpoint_*`` subdirectory by name;
    otherwise the most recent one is used (same rule as warm-start,
    :func:`poker_ai.tables.warm_start._latest_checkpoint`).  A ``blueprint_path``
    that itself contains ``server_state.pkl`` is accepted directly, so the
    script also runs against a bare checkpoint directory.
    """
    blueprint_path = Path(blueprint_path)
    if checkpoint is not None:
        cp = blueprint_path / checkpoint
        if not (cp / "server_state.pkl").exists():
            raise FileNotFoundError(f"No server_state.pkl in {cp}")
        return cp
    if (blueprint_path / "server_state.pkl").exists():
        return blueprint_path
    candidates = sorted(blueprint_path.glob("checkpoint_[0-9]*"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint_* directory found in blueprint path: {blueprint_path}"
        )
    return candidates[-1]


def _chunk_files(cp_dir: Path, prefix: str, street: int) -> List[Path]:
    """Sorted ``{prefix}_{street}_chunk_*.npy`` files in *cp_dir*."""
    return sorted(cp_dir.glob(f"{prefix}_{street}_chunk_*.npy"))


def _select_chunks(
    files: Sequence[Path], sample_chunks: Optional[int]
) -> Tuple[List[Path], bool]:
    """Pick ``sample_chunks`` evenly-spaced files (incl. first + last); else all.

    Returns ``(selected, sampled)``.  The endpoints are always included so the
    caller can recover the exact total row count (chunk 0 is full, the last
    chunk is the only possibly-partial one).  ``sample_chunks`` of ``None``/``0``
    or a value ``>= len(files)`` returns every file with ``sampled=False``.
    """
    files = list(files)
    if not sample_chunks or sample_chunks <= 0 or len(files) <= sample_chunks:
        return files, False
    idx = sorted(
        {int(round(i)) for i in np.linspace(0, len(files) - 1, sample_chunks)}
    )
    return [files[i] for i in idx], True


def _select_indices(n: int, sample_chunks: Optional[int]) -> Tuple[List[int], bool]:
    """Chunk indices to read: ``sample_chunks`` evenly-spaced (incl. first+last).

    The index-based twin of :func:`_select_chunks`.  Returning indices (rather
    than files) lets a caller apply the *same* selection to two parallel file
    lists — used by :func:`compute_leaf_coverage` to keep the strategy and
    regret chunk streams row-aligned.
    """
    if not sample_chunks or sample_chunks <= 0 or n <= sample_chunks:
        return list(range(n)), False
    idx = sorted({int(round(i)) for i in np.linspace(0, n - 1, sample_chunks)})
    return idx, True


def _action_labels(width: int, street: int) -> Tuple[List[str], bool]:
    """Column labels for a strategy row of ``width`` actions on ``street``.

    Returns ``(labels, canonical)`` where ``canonical`` is True when the labels
    come from the live action space.  On width mismatch the fold/call/all_in
    prefix is still positional truth (see module docstring), so only the raise
    fractions degrade to generic names.
    """
    try:
        from environment.action_space import CANONICAL_ACTIONS

        canonical = CANONICAL_ACTIONS.get(street)
        if canonical is not None and len(canonical) == width:
            return list(canonical), True
    except Exception as exc:  # pragma: no cover - env import failure path
        log.warning("Could not import action space for labels: %s", exc)
    labels = ["fold", "call", "all_in"][:width]
    labels += [f"raise_#{k + 1}" for k in range(width - len(labels))]
    return labels, False


# --------------------------------------------------------------------------- #
# Streaming accumulator
# --------------------------------------------------------------------------- #


class _StreetAccumulator:
    """Single-pass accumulator over one street's strategy rows."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.n_rows = 0
        self.n_visited = 0
        self.total_mass = 0.0
        self.action_mass = np.zeros(width, dtype=np.float64)
        self.mean_strategy_sum = np.zeros(width, dtype=np.float64)
        self.entropy_hist = np.zeros(_ENTROPY_BINS, dtype=np.int64)
        self.maxp_hist = np.zeros(_MAXP_BINS, dtype=np.int64)
        self.mass_hist = np.zeros(_MASS_BINS, dtype=np.int64)
        self.n_maxp_gt = {0.5: 0, 0.9: 0, 0.99: 0}

    def update(self, batch: np.ndarray) -> None:
        """Fold a ``(rows, width)`` int32 batch into the running stats."""
        self.n_rows += batch.shape[0]
        row_mass = batch.sum(axis=1, dtype=np.int64)
        visited = row_mass > 0
        n_vis = int(visited.sum())
        if n_vis == 0:
            return
        self.n_visited += n_vis

        rows = batch[visited].astype(np.float64)
        mass = row_mass[visited].astype(np.float64)
        self.total_mass += float(mass.sum())
        self.action_mass += rows.sum(axis=0)

        probs = rows / mass[:, None]
        self.mean_strategy_sum += probs.sum(axis=0)

        # Normalised entropy in [0, 1] (0·log0 := 0; width-1 rows are pure).
        if self.width > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                plogp = np.where(probs > 0.0, probs * np.log(probs), 0.0)
            h = -plogp.sum(axis=1) / math.log(self.width)
        else:
            h = np.zeros(len(probs))
        idx = np.clip((h * _ENTROPY_BINS).astype(np.int64), 0, _ENTROPY_BINS - 1)
        np.add.at(self.entropy_hist, idx, 1)

        maxp = probs.max(axis=1)
        idx = np.clip((maxp * _MAXP_BINS).astype(np.int64), 0, _MAXP_BINS - 1)
        np.add.at(self.maxp_hist, idx, 1)
        for thr in self.n_maxp_gt:
            self.n_maxp_gt[thr] += int((maxp > thr).sum())

        logm = np.log10(mass)
        idx = np.clip(
            (logm / _MASS_LOG10_MAX * _MASS_BINS).astype(np.int64),
            0,
            _MASS_BINS - 1,
        )
        np.add.at(self.mass_hist, idx, 1)


def _hist_percentile(hist: np.ndarray, q: float, lo: float, hi: float) -> Optional[float]:
    """Linear-interpolated ``q``-percentile (0..100) from a fixed-bin histogram.

    Values are assumed uniformly distributed within each bin over ``[lo, hi]``;
    accurate to a bin width, which is plenty for a report.
    """
    total = int(hist.sum())
    if total == 0:
        return None
    target = (q / 100.0) * total
    cum = 0
    bin_width = (hi - lo) / len(hist)
    for i, count in enumerate(hist):
        if count > 0 and cum + count >= target:
            frac = (target - cum) / count
            return lo + (i + frac) * bin_width
        cum += count
    return hi


def _rows_in(path: Path) -> int:
    """Row count of a chunk ``.npy`` from its header only (no data read)."""
    return int(np.load(path, mmap_mode="r").shape[0])


def _infer_total_rows(all_files: Sequence[Path]) -> int:
    """Exact total infoset count across *all_files* from two chunk headers.

    Only the final chunk is ever partial (:meth:`ChunkStore._save`), so the total
    is ``(n - 1) * full_chunk_rows + last_chunk_rows`` where ``full_chunk_rows``
    is chunk 0's (always full) count.  Both are read from the ``.npy`` header
    alone, so this is exact even when the streaming pass only sampled a subset.
    """
    n = len(all_files)
    if n == 0:
        return 0
    last_rows = _rows_in(all_files[-1])
    if n == 1:
        return last_rows
    return (n - 1) * _rows_in(all_files[0]) + last_rows


def compute_street_metrics(
    files: Sequence[Path],
    street: int,
    *,
    sample_chunks: Optional[int] = None,
    batch_rows: int = _DEFAULT_BATCH_ROWS,
) -> Optional[dict]:
    """Stream one street's strategy chunk files into a metrics dict.

    Reads either every file or ``sample_chunks`` evenly-spaced ones (see
    :func:`_select_chunks`).  Frequencies are computed over the rows actually
    read (``n_infosets_scanned``); ``n_infosets_total`` is exact regardless of
    sampling.  Returns ``None`` when the street has no chunk files (never
    trained — normal for streets beyond a truncated test game).
    """
    files = list(files)
    if not files:
        return None
    selected, sampled = _select_chunks(files, sample_chunks)

    acc: Optional[_StreetAccumulator] = None
    for path in selected:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"{path} is not a 2-D chunk array (shape {arr.shape})")
        if acc is None:
            acc = _StreetAccumulator(arr.shape[1])
        elif arr.shape[1] != acc.width:
            raise ValueError(
                f"Inconsistent action width in {path}: {arr.shape[1]} != {acc.width}"
            )
        for start in range(0, arr.shape[0], batch_rows):
            acc.update(np.asarray(arr[start : start + batch_rows]))

    assert acc is not None
    labels, canonical = _action_labels(acc.width, street)

    play_freq: Dict[str, float] = {}
    mean_strategy: Dict[str, float] = {}
    if acc.total_mass > 0:
        freqs = acc.action_mass / acc.total_mass
        means = acc.mean_strategy_sum / acc.n_visited
        play_freq = {a: float(f) for a, f in zip(labels, freqs)}
        mean_strategy = {a: float(m) for a, m in zip(labels, means)}

    def _bucket(freqs: Dict[str, float]) -> Dict[str, float]:
        out = {"fold": 0.0, "call": 0.0, "all_in": 0.0, "raise": 0.0}
        for a, f in freqs.items():
            out[a if a in out else "raise"] += f
        return out

    nv = acc.n_visited
    return {
        "street": _STREET_NAME[street],
        "n_actions": acc.width,
        "action_labels_canonical": canonical,
        "n_chunks_total": len(files),
        "n_chunks_read": len(selected),
        "sampled": sampled,
        "n_infosets_total": _infer_total_rows(files),
        "n_infosets_scanned": acc.n_rows,
        "n_visited": nv,
        "visited_frac": (nv / acc.n_rows) if acc.n_rows else None,
        "total_visit_mass": acc.total_mass,
        "play_freq": play_freq,
        "play_freq_buckets": _bucket(play_freq),
        "mean_strategy": mean_strategy,
        "mean_strategy_buckets": _bucket(mean_strategy),
        "determinism": {
            "entropy_mean": (
                None if nv == 0 else float(
                    (acc.entropy_hist * (np.arange(_ENTROPY_BINS) + 0.5)
                     / _ENTROPY_BINS).sum() / nv
                )
            ),
            "entropy_p10": _hist_percentile(acc.entropy_hist, 10, 0.0, 1.0),
            "entropy_p50": _hist_percentile(acc.entropy_hist, 50, 0.0, 1.0),
            "entropy_p90": _hist_percentile(acc.entropy_hist, 90, 0.0, 1.0),
            "maxp_p50": _hist_percentile(acc.maxp_hist, 50, 0.0, 1.0),
            "frac_maxp_gt_50": None if nv == 0 else acc.n_maxp_gt[0.5] / nv,
            "frac_maxp_gt_90": None if nv == 0 else acc.n_maxp_gt[0.9] / nv,
            "frac_maxp_gt_99": None if nv == 0 else acc.n_maxp_gt[0.99] / nv,
        },
        "visit_mass_log10": {
            "p10": _hist_percentile(acc.mass_hist, 10, 0.0, _MASS_LOG10_MAX),
            "p50": _hist_percentile(acc.mass_hist, 50, 0.0, _MASS_LOG10_MAX),
            "p90": _hist_percentile(acc.mass_hist, 90, 0.0, _MASS_LOG10_MAX),
        },
    }


def compute_regret_metrics(
    files: Sequence[Path],
    *,
    sample_chunks: Optional[int] = None,
    batch_rows: int = _DEFAULT_BATCH_ROWS,
) -> Optional[dict]:
    """Stream one street's regret chunk files into a regret-health dict."""
    files = list(files)
    if not files:
        return None
    selected, sampled = _select_chunks(files, sample_chunks)
    try:
        from poker_ai.tables.cfr_tables import REGRET_FLOOR

        floor = int(REGRET_FLOOR)
    except Exception:  # pragma: no cover - keeps the module importable alone
        floor = -310_000_000

    n_entries = 0
    n_at_floor = 0
    n_prunable = 0
    n_positive = 0
    positive_sum = 0.0
    for path in selected:
        arr = np.load(path, mmap_mode="r")
        for start in range(0, arr.shape[0], batch_rows):
            batch = np.asarray(arr[start : start + batch_rows])
            n_entries += batch.size
            n_at_floor += int((batch <= floor).sum())
            n_prunable += int((batch < PRUNE_THRESHOLD).sum())
            pos = batch > 0
            n_positive += int(pos.sum())
            positive_sum += float(batch[pos].sum(dtype=np.float64))
    if n_entries == 0:
        return None
    return {
        "n_chunks_total": len(files),
        "n_chunks_read": len(selected),
        "sampled": sampled,
        "n_entries": n_entries,
        "frac_at_floor": n_at_floor / n_entries,
        "frac_prunable": n_prunable / n_entries,
        "frac_positive": n_positive / n_entries,
        "mean_positive_regret": (positive_sum / n_positive) if n_positive else None,
    }


def compute_leaf_coverage(
    strategy_files: Sequence[Path],
    regret_files: Sequence[Path],
    street: int,
    *,
    min_strategy_mass: int = _DEFAULT_MIN_STRATEGY_MASS,
    sample_chunks: Optional[int] = None,
    batch_rows: int = _DEFAULT_BATCH_ROWS,
) -> Tuple[Optional[dict], List[str]]:
    """Row-join strategy + regret chunks into a leaf-coverage dict.

    Classifies every allocated infoset by how
    :class:`~poker_ai.search.policy.BlueprintPolicy` would resolve it as a
    depth-limited-search leaf continuation:

    - ``avg_trusted`` — strategy visit mass ``>= min_strategy_mass``; the
      converged average strategy is used (the best case).
    - ``regret_fallback`` — not trusted, but the regret row has at least one
      positive entry, so regret matching yields a non-uniform distribution.
    - ``uniform`` — neither; the leaf gets no blueprint opinion.

    ``effective`` (``avg_trusted + regret_fallback``) is the fraction of leaf
    queries that resolve to a real strategy — the true leaf coverage.

    The join is positional: both tables share the per-street infoset index
    (:class:`poker_ai.tables.cfr_tables.CFRTables`), so row *i* is the same
    infoset in ``strategy_{street}_chunk_*`` and ``regret_{street}_chunk_*``.
    Alignment is verified per chunk (matching row counts); a mismatch aborts
    with a warning rather than silently reporting a bogus join.

    Returns ``(dict_or_None, warnings)``.  ``None`` when the street has no
    strategy or no regret chunks, or when the two tables cannot be aligned.

    Caveats (documented, mild, and in the safe direction for a diagnostic):

    - The policy masks the strategy row to the node's *legal* actions before the
      mass check; this pass uses the raw row sum, so ``avg_trusted`` is a slight
      over-count (a masked row could dip below the threshold at some node).
    - ``regret_fallback`` tests for any positive regret in the full row, not only
      among legal actions, for the same reason.
    """
    strategy_files = list(strategy_files)
    regret_files = list(regret_files)
    warnings: List[str] = []
    if not strategy_files or not regret_files:
        return None, warnings
    if len(strategy_files) != len(regret_files):
        warnings.append(
            f"street {street}: {len(strategy_files)} strategy chunks vs "
            f"{len(regret_files)} regret chunks — cannot row-join; leaf "
            f"coverage skipped"
        )
        return None, warnings

    idx, sampled = _select_indices(len(strategy_files), sample_chunks)

    n_scanned = 0
    n_avg_trusted = 0      # strat mass >= threshold
    n_regret_pos = 0       # any positive regret (informative fallback)
    n_effective = 0        # avg_trusted OR regret_pos (non-uniform leaf)
    width: Optional[int] = None
    for i in idx:
        s_arr = np.load(strategy_files[i], mmap_mode="r")
        r_arr = np.load(regret_files[i], mmap_mode="r")
        if s_arr.shape[0] != r_arr.shape[0]:
            warnings.append(
                f"street {street}: chunk {i} row mismatch "
                f"(strategy {s_arr.shape[0]} vs regret {r_arr.shape[0]}) — "
                f"leaf coverage skipped"
            )
            return None, warnings
        if width is None:
            width = int(s_arr.shape[1])
        for start in range(0, s_arr.shape[0], batch_rows):
            s_batch = np.asarray(s_arr[start : start + batch_rows])
            r_batch = np.asarray(r_arr[start : start + batch_rows])
            n_scanned += s_batch.shape[0]
            avg_trusted = s_batch.sum(axis=1, dtype=np.int64) >= min_strategy_mass
            regret_pos = (r_batch > 0).any(axis=1)
            n_avg_trusted += int(avg_trusted.sum())
            n_regret_pos += int(regret_pos.sum())
            n_effective += int((avg_trusted | regret_pos).sum())

    if n_scanned == 0:
        return None, warnings

    n_regret_fallback = n_effective - n_avg_trusted   # regret_pos & ~avg_trusted
    n_uniform = n_scanned - n_effective
    return {
        "min_strategy_mass": int(min_strategy_mass),
        "n_chunks_total": len(strategy_files),
        "n_chunks_read": len(idx),
        "sampled": sampled,
        "n_infosets_total": _infer_total_rows(strategy_files),
        "n_scanned": n_scanned,
        "avg_trusted_frac": n_avg_trusted / n_scanned,
        "regret_fallback_frac": n_regret_fallback / n_scanned,
        "effective_frac": n_effective / n_scanned,
        "uniform_frac": n_uniform / n_scanned,
        "counts": {
            "avg_trusted": n_avg_trusted,
            "regret_fallback": n_regret_fallback,
            "uniform": n_uniform,
            "scanned": n_scanned,
        },
    }, warnings


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #


def build_report(
    blueprint_path,
    *,
    checkpoint: Optional[str] = None,
    include_regret: bool = False,
    include_leaf_coverage: bool = False,
    min_strategy_mass: int = _DEFAULT_MIN_STRATEGY_MASS,
    sample_chunks: Optional[int] = None,
    batch_rows: int = _DEFAULT_BATCH_ROWS,
) -> dict:
    """Assemble the full metrics report dict for ``blueprint_path``.

    ``sample_chunks`` bounds how many chunks per street are read (``None`` =
    every chunk = exact frequencies).  ``include_regret`` adds the regret-health
    tables (off by default — it is a training diagnostic and doubles I/O).
    ``include_leaf_coverage`` adds the strategy/regret row-join that reports how
    a search leaf would resolve each infoset (average vs. regret fallback vs.
    uniform); it reads both tables, so it also roughly doubles I/O.
    """
    import joblib

    blueprint_path = Path(blueprint_path)
    cp_dir = _resolve_checkpoint(blueprint_path, checkpoint)
    state = joblib.load(cp_dir / "server_state.pkl")

    warnings: List[str] = []
    streets: Dict[str, dict] = {}
    regret: Dict[str, dict] = {}
    leaf_coverage: Dict[str, dict] = {}
    for r in range(4):
        s_files = _chunk_files(cp_dir, "strategy", r)
        n_expected = int(
            (state.get("n_chunks_per_street") or {}).get(r, len(s_files)) or 0
        )
        if s_files and len(s_files) < n_expected:
            warnings.append(
                f"street {r}: {len(s_files)} strategy chunk files on disk but "
                f"checkpoint expects {n_expected} — metrics cover the files present"
            )
        m = compute_street_metrics(
            s_files, r, sample_chunks=sample_chunks, batch_rows=batch_rows
        )
        if m is not None:
            streets[_STREET_NAME[r]] = m
        if include_regret:
            rm = compute_regret_metrics(
                _chunk_files(cp_dir, "regret", r),
                sample_chunks=sample_chunks,
                batch_rows=batch_rows,
            )
            if rm is not None:
                regret[_STREET_NAME[r]] = rm
        if include_leaf_coverage:
            lc, lc_warnings = compute_leaf_coverage(
                s_files,
                _chunk_files(cp_dir, "regret", r),
                r,
                min_strategy_mass=min_strategy_mass,
                sample_chunks=sample_chunks,
                batch_rows=batch_rows,
            )
            warnings.extend(lc_warnings)
            if lc is not None:
                leaf_coverage[_STREET_NAME[r]] = lc

    any_sampled = any(m["sampled"] for m in streets.values())
    report = {
        "meta": {
            "blueprint_path": str(blueprint_path),
            "checkpoint": cp_dir.name,
            "t": state.get("t"),
            "n_players": state.get("n_players"),
            "n_infosets_total": sum(m["n_infosets_total"] for m in streets.values()),
            "sampled": any_sampled,
            "sample_chunks": sample_chunks if any_sampled else None,
            "min_strategy_mass": min_strategy_mass if include_leaf_coverage else None,
        },
        "streets": streets,
        "regret": regret,
        "leaf_coverage": leaf_coverage,
        "warnings": warnings,
    }
    return report


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _fmt(v, spec: str = ".2f", none: str = "n/a") -> str:
    return none if v is None else format(v, spec)


def _print_human(report: dict) -> str:
    """Compact fixed-width block (summarize.py §8 style); also the log output."""
    m = report["meta"]
    L: List[str] = []
    L.append(
        f"blueprint  {m['blueprint_path']}   {m['checkpoint']}   "
        f"t={m['t']}   {m['n_players']}-max   "
        f"{m['n_infosets_total']:,} infosets"
    )
    if m["sampled"]:
        chunks = "  ".join(
            f"{name} {s['n_chunks_read']}/{s['n_chunks_total']}"
            for name, s in report["streets"].items()
        )
        L.append(f"SAMPLED (estimate — play freqs from a chunk subset): {chunks}")
    L.append("─" * 76)
    for name, s in report["streets"].items():
        b = s["play_freq_buckets"]
        L.append(
            f"{name.upper():<8} {s['n_infosets_total']:>12,} infosets   "
            f"visited {_fmt(s['visited_frac'], '.1%')}   "
            f"mass {s['total_visit_mass']:.3g}"
        )
        L.append(
            f"  play freq (weighted)   fold {_fmt(b['fold'], '.1%')}   "
            f"call {_fmt(b['call'], '.1%')}   raise {_fmt(b['raise'], '.1%')}   "
            f"all-in {_fmt(b['all_in'], '.1%')}"
        )
        raises = {
            a: f for a, f in s["play_freq"].items()
            if a not in ("fold", "call", "all_in") and f >= 0.0005
        }
        if raises:
            L.append(
                "    raise sizes          "
                + "   ".join(f"{a.split(':')[-1]}x {f:.1%}" for a, f in raises.items())
            )
        d = s["determinism"]
        L.append(
            f"  determinism            entropy p50 {_fmt(d['entropy_p50'], '.2f')}   "
            f"pure(>0.9) {_fmt(d['frac_maxp_gt_90'], '.1%')}   "
            f"pure(>0.99) {_fmt(d['frac_maxp_gt_99'], '.1%')}"
        )
        rg = report["regret"].get(name)
        if rg:
            L.append(
                f"  regret health          positive {_fmt(rg['frac_positive'], '.1%')}   "
                f"prunable {_fmt(rg['frac_prunable'], '.1%')}   "
                f"at floor {_fmt(rg['frac_at_floor'], '.2%')}"
            )
        lc = report.get("leaf_coverage", {}).get(name)
        if lc:
            L.append(
                f"  leaf coverage          effective {_fmt(lc['effective_frac'], '.1%')}   "
                f"(avg {_fmt(lc['avg_trusted_frac'], '.1%')} + "
                f"regret {_fmt(lc['regret_fallback_frac'], '.1%')})   "
                f"uniform {_fmt(lc['uniform_frac'], '.1%')}"
            )
    if report["warnings"]:
        L.append("")
        L.append("WARNINGS")
        for w in report["warnings"]:
            L.append(f"  ⚠ {w}")
    return "\n".join(L)


def render_markdown(report: dict) -> str:
    """Full markdown report with per-action tables (the stored artifact)."""
    m = report["meta"]
    L: List[str] = []
    L.append("# Blueprint metrics")
    L.append("")
    L.append(f"- **Blueprint**: `{m['blueprint_path']}` ({m['checkpoint']})")
    L.append(f"- **Training iteration**: {m['t']}")
    L.append(f"- **Players**: {m['n_players']}")
    L.append(f"- **Total infosets**: {m['n_infosets_total']:,}")
    if m["sampled"]:
        L.append(
            f"- **Sampled sanity check**: play frequencies estimated from "
            f"{m['sample_chunks']} evenly-spaced chunks per street (coverage "
            f"totals are exact). Run with `--full` for exact frequencies."
        )
    L.append("")
    L.append(
        "*Play freq* is visit-weighted (how often the blueprint plays the action "
        "when acting on the street under its own play); *mean strategy* weights "
        "every visited infoset equally."
    )

    for name, s in report["streets"].items():
        L.append("")
        L.append(f"## {name.capitalize()}")
        L.append("")
        scan = (
            f" — scanned {s['n_infosets_scanned']:,} rows in "
            f"{s['n_chunks_read']}/{s['n_chunks_total']} chunks"
            if s["sampled"] else ""
        )
        L.append(
            f"- Infosets: **{s['n_infosets_total']:,}** "
            f"({_fmt(s['visited_frac'], '.1%')} visited), "
            f"action columns: {s['n_actions']}{scan}"
            + ("" if s["action_labels_canonical"]
               else " *(non-canonical width — generic raise labels)*")
        )
        L.append(f"- Total visit mass (scanned): {s['total_visit_mass']:.4g}")
        mass = s["visit_mass_log10"]
        L.append(
            f"- Visit mass per infoset (log10): "
            f"p10 {_fmt(mass['p10'], '.1f')} / p50 {_fmt(mass['p50'], '.1f')} / "
            f"p90 {_fmt(mass['p90'], '.1f')}"
        )
        d = s["determinism"]
        L.append(
            f"- Normalised entropy: mean {_fmt(d['entropy_mean'], '.3f')}, "
            f"p10/p50/p90 {_fmt(d['entropy_p10'], '.2f')}/"
            f"{_fmt(d['entropy_p50'], '.2f')}/{_fmt(d['entropy_p90'], '.2f')}; "
            f"near-pure rows (max prob > 0.9): {_fmt(d['frac_maxp_gt_90'], '.1%')}"
        )
        rg = report["regret"].get(name)
        if rg:
            L.append(
                f"- Regret: {_fmt(rg['frac_positive'], '.1%')} of entries positive, "
                f"{_fmt(rg['frac_prunable'], '.1%')} prune-eligible "
                f"(< {PRUNE_THRESHOLD:,}), {_fmt(rg['frac_at_floor'], '.2%')} at floor"
            )
        lc = report.get("leaf_coverage", {}).get(name)
        if lc:
            L.append(
                f"- Leaf coverage (as search-leaf continuation, "
                f"min_strategy_mass={lc['min_strategy_mass']}): "
                f"**{_fmt(lc['effective_frac'], '.1%')} effective** "
                f"(non-uniform) = {_fmt(lc['avg_trusted_frac'], '.1%')} trusted "
                f"average + {_fmt(lc['regret_fallback_frac'], '.1%')} regret "
                f"fallback; {_fmt(lc['uniform_frac'], '.1%')} resolve to uniform"
            )
        L.append("")
        L.append("| action | play freq | mean strategy |")
        L.append("|---|---:|---:|")
        for a in s["play_freq"]:
            L.append(
                f"| `{a}` | {s['play_freq'][a]:.2%} | {s['mean_strategy'][a]:.2%} |"
            )
        bp, bm = s["play_freq_buckets"], s["mean_strategy_buckets"]
        L.append(
            f"| **any raise** | **{bp['raise']:.2%}** | **{bm['raise']:.2%}** |"
        )

    if report["warnings"]:
        L.append("")
        L.append("## Warnings")
        L.append("")
        for w in report["warnings"]:
            L.append(f"- ⚠ {w}")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Orchestrator + entry point
# --------------------------------------------------------------------------- #


def analyze(
    blueprint_path,
    *,
    out_dir=None,
    checkpoint: Optional[str] = None,
    include_regret: bool = False,
    include_leaf_coverage: bool = False,
    min_strategy_mass: int = _DEFAULT_MIN_STRATEGY_MASS,
    sample_chunks: Optional[int] = None,
    batch_rows: int = _DEFAULT_BATCH_ROWS,
    write_files: bool = True,
    echo: bool = True,
) -> dict:
    """Compute blueprint metrics; print the human block and store md + json.

    ``out_dir`` defaults to ``blueprint_path`` so the artifacts live next to
    the checkpoint they describe.  Returns the report dict.
    """
    report = build_report(
        blueprint_path,
        checkpoint=checkpoint,
        include_regret=include_regret,
        include_leaf_coverage=include_leaf_coverage,
        min_strategy_mass=min_strategy_mass,
        sample_chunks=sample_chunks,
        batch_rows=batch_rows,
    )
    if echo:
        print(_print_human(report))
    if write_files:
        out = Path(out_dir) if out_dir is not None else Path(blueprint_path)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "blueprint_metrics.json", "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        (out / "blueprint_metrics.md").write_text(render_markdown(report))
        log.info("Wrote blueprint_metrics.{json,md} to %s", out)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m evaluation.blueprint_metrics",
        description=(
            "Sanity-check a trained blueprint's learned strategy: per-street "
            "fold/call/raise frequencies, determinism, and coverage. Samples a "
            "few chunks per street by default (blueprints are 200 GB+)."
        ),
    )
    parser.add_argument(
        "blueprint",
        help="Trained-blueprint directory (contains checkpoint_*), or a "
        "checkpoint directory itself.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Analyse a specific checkpoint_* subdirectory (default: latest).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for blueprint_metrics.{md,json} (default: the blueprint dir).",
    )
    parser.add_argument(
        "--sample-chunks",
        type=int,
        default=_DEFAULT_SAMPLE_CHUNKS,
        help="Evenly-spaced chunks per street to read for the sanity check "
        f"(default: {_DEFAULT_SAMPLE_CHUNKS}). Frequencies are estimates; "
        "coverage totals stay exact.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Read every chunk for exact frequencies (reads the whole table).",
    )
    parser.add_argument(
        "--regret",
        action="store_true",
        help="Also scan the regret tables (training diagnostic; doubles I/O).",
    )
    parser.add_argument(
        "--leaf-coverage",
        action="store_true",
        help="Row-join strategy + regret tables and report how a search leaf "
        "would resolve each infoset (trusted average / regret fallback / "
        "uniform). The true leaf coverage for subgame-solving; reads both "
        "tables, so it roughly doubles I/O.",
    )
    parser.add_argument(
        "--min-strategy-mass",
        type=int,
        default=_DEFAULT_MIN_STRATEGY_MASS,
        help="Visit-mass threshold above which a leaf trusts the average "
        f"strategy over the regret fallback (default: {_DEFAULT_MIN_STRATEGY_MASS}, "
        "matching BlueprintPolicy). Only affects --leaf-coverage.",
    )
    parser.add_argument(
        "--no-files", action="store_true", help="Print only; do not write artifacts."
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=_DEFAULT_BATCH_ROWS,
        help="Rows per streaming batch (memory/speed knob).",
    )
    args = parser.parse_args(argv)
    analyze(
        args.blueprint,
        out_dir=args.out,
        checkpoint=args.checkpoint,
        include_regret=args.regret,
        include_leaf_coverage=args.leaf_coverage,
        min_strategy_mass=args.min_strategy_mass,
        sample_chunks=None if args.full else args.sample_chunks,
        batch_rows=args.batch_rows,
        write_files=not args.no_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
