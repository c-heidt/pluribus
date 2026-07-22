#!/usr/bin/env python3
"""Compute a data-driven CFR-P pruning threshold ``c`` from a checkpoint.

Background
----------
CFR-P (``poker_ai.blueprint.cfr.cfr_p``) explores an action iff its cumulative
regret is **strictly greater than** ``c`` (or the street is the river, which is
never pruned) — see ``cfr.py``'s ``_prune``.  ``c`` is a *negative* constant:

  * ``c`` closer to 0  -> more actions fall at/below it -> **more** pruning
  * ``c`` closer to the regret floor -> **less** pruning

The shipped default ``c = -300_000_000`` sits right against
``REGRET_FLOOR = -310_000_000``.  If the run's actual regret magnitudes never
approach 3e8 (they usually don't), essentially nothing is <= c and pruning does
nothing (``frac_prunable ~ 0`` in ``blueprint_metrics``).  Pruning that never
binds is wasted compute: CFR-P keeps traversing deep, clearly-losing subtrees.

This script measures the real regret distribution in a checkpoint and recommends
a ``c`` scaled to it, so pruning actually engages.  Because pruning skips a
whole subtree, the payoff is fewer nodes per traversal -> more iterations in the
same wall-clock.

What it reports
---------------
Per street (river shown for context but is never pruned):
  * counts of positive / negative / at-floor entries, mean positive regret
    (the "signal scale"), and the most-negative regret seen
  * the fraction of **negative** and of **non-zero** action entries that the
    *current* ``c`` would prune (expected ~0 with the shipped default)
  * a sweep of candidate ``c`` values with the prune fraction each would give

Then a single recommended ``c``, chosen so that ``--target-neg-frac`` (default
0.5) of the *negative* action-entries pooled over the prunable streets
(preflop+flop+turn) fall at/below it — i.e. c = -(median-magnitude of the
losing tail) by default.  The 5% unpruned fallback (``PRUNE_PROBABILITY``) plus
the river-always-explored rule make a moderately aggressive c safe: pruned
actions can still recover.

Usage
-----
    python scripts/compute_prune_c.py /path/to/models/<run> \
        [--checkpoint checkpoint_XXXX] [--sample-chunks 8] \
        [--target-neg-frac 0.5]

``--sample-chunks`` bounds chunks read per street (evenly spaced, endpoints
included); omit / 0 to read every chunk (exact, slower).  Default samples 8 for
a fast-but-representative estimate on a large checkpoint.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Repo root on sys.path so ``evaluation`` / ``poker_ai`` import when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.blueprint_metrics import (  # noqa: E402
    _STREET_NAME,
    _chunk_files,
    _resolve_checkpoint,
    _select_chunks,
)

try:
    from poker_ai.tables.cfr_tables import REGRET_FLOOR

    FLOOR = int(REGRET_FLOOR)
except Exception:  # keep runnable even if the package can't import
    FLOOR = -310_000_000

CURRENT_C = -300_000_000  # PRUNE_THRESHOLD shipped default

# Log-magnitude histogram of |negative regret|.  Negatives are int32 <= -1, so
# magnitude >= 1 and log10 >= 0; upper edge is the floor magnitude.
_NBINS = 340
_HI = math.log10(abs(FLOOR))
_BATCH_ROWS = 1_000_000

# Prunable streets (river is never pruned by cfr_p, so it is excluded from the
# pooled recommendation but still reported for context).
_PRUNABLE_STREETS = (0, 1, 2)


class _RegretDist:
    """Streaming stats + log-magnitude histogram of one street's regrets."""

    def __init__(self) -> None:
        self.n_entries = 0
        self.n_nonzero = 0
        self.n_negative = 0
        self.n_positive = 0
        self.n_at_floor = 0
        self.positive_sum = 0.0
        self.min_regret = 0  # most negative value seen
        self.hist = np.zeros(_NBINS, dtype=np.int64)  # over |neg regret|

    def update(self, batch: np.ndarray) -> None:
        flat = np.asarray(batch).reshape(-1)
        self.n_entries += flat.size
        nz = flat != 0
        self.n_nonzero += int(nz.sum())
        pos = flat > 0
        self.n_positive += int(pos.sum())
        if pos.any():
            self.positive_sum += float(flat[pos].sum(dtype=np.float64))
        neg = flat[flat < 0]
        if neg.size:
            self.n_negative += neg.size
            self.n_at_floor += int((flat <= FLOOR).sum())
            self.min_regret = min(self.min_regret, int(neg.min()))
            mag = (-neg).astype(np.float64)
            idx = np.clip(
                (np.log10(mag) / _HI * _NBINS).astype(np.int64), 0, _NBINS - 1
            )
            np.add.at(self.hist, idx, 1)

    def merge(self, other: "_RegretDist") -> None:
        self.n_entries += other.n_entries
        self.n_nonzero += other.n_nonzero
        self.n_negative += other.n_negative
        self.n_positive += other.n_positive
        self.n_at_floor += other.n_at_floor
        self.positive_sum += other.positive_sum
        self.min_regret = min(self.min_regret, other.min_regret)
        self.hist += other.hist

    @property
    def mean_positive_regret(self):
        return self.positive_sum / self.n_positive if self.n_positive else None

    def count_ge(self, mag: float) -> float:
        """Estimated number of |neg| entries with magnitude >= ``mag``.

        Interpolates within the bin containing ``mag`` (uniform-in-log), which
        is accurate to a bin width — plenty for choosing a threshold.
        """
        if mag <= 1.0:
            return float(self.n_negative)
        pos = math.log10(mag) / _HI * _NBINS
        b = int(math.floor(pos))
        if b >= _NBINS:
            return 0.0
        if b < 0:
            return float(self.n_negative)
        frac_above_in_b = 1.0 - (pos - b)  # portion of bin b at/above mag
        return float(self.hist[b]) * frac_above_in_b + float(self.hist[b + 1 :].sum())

    def prune_fracs_at(self, c: int):
        """(frac of negatives, frac of non-zeros) pruned at threshold ``c``.

        Pruned == regret <= c, i.e. |neg| >= |c| (c < 0).  Positives and zeros
        are never pruned, so the non-zero denominator is the realized per-node
        action-prune rate proxy.
        """
        if c >= 0:
            n_pruned = 0.0
        else:
            n_pruned = self.count_ge(float(abs(c)))
        frac_neg = n_pruned / self.n_negative if self.n_negative else 0.0
        frac_nz = n_pruned / self.n_nonzero if self.n_nonzero else 0.0
        return frac_neg, frac_nz

    def magnitude_at_upper_frac(self, f: float):
        """Magnitude M such that fraction ``f`` of negatives have |neg| >= M.

        Returns the recommended pruning magnitude for a target prune fraction
        of the losing tail.  Walks the histogram from the top.
        """
        if self.n_negative == 0:
            return None
        target = f * self.n_negative
        cum = 0.0
        for b in range(_NBINS - 1, -1, -1):
            h = float(self.hist[b])
            if cum + h >= target and h > 0:
                # interpolate inside bin b (uniform in log)
                need = target - cum
                frac_from_top = need / h  # portion of bin from its top edge
                lo = b / _NBINS * _HI
                hi = (b + 1) / _NBINS * _HI
                logM = hi - frac_from_top * (hi - lo)
                return 10.0 ** logM
            cum += h
        return 1.0


def _stream_street(files, sample_chunks, batch_rows) -> "_RegretDist | None":
    selected, sampled = _select_chunks(list(files), sample_chunks)
    if not selected:
        return None
    dist = _RegretDist()
    for path in selected:
        arr = np.load(path, mmap_mode="r")
        for start in range(0, arr.shape[0], batch_rows):
            dist.update(np.asarray(arr[start : start + batch_rows]))
    dist.sampled = sampled  # type: ignore[attr-defined]
    dist.n_chunks = (len(list(files)), len(selected))  # type: ignore[attr-defined]
    return dist


_SWEEP = [
    -1_000, -3_000, -10_000, -30_000, -100_000, -300_000,
    -1_000_000, -3_000_000, -10_000_000, -30_000_000,
    -100_000_000, -300_000_000,
]


def _fmt_int(v) -> str:
    return f"{v:,}" if v is not None else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("blueprint_path", type=Path, help="run dir or bare checkpoint dir")
    ap.add_argument("--checkpoint", default=None, help="checkpoint_* subdir (default: latest)")
    ap.add_argument("--sample-chunks", type=int, default=8,
                    help="chunks/street to read (0 = all; default 8)")
    ap.add_argument("--target-neg-frac", type=float, default=0.5,
                    help="fraction of negative entries to prune at the recommended c")
    ap.add_argument("--batch-rows", type=int, default=_BATCH_ROWS)
    args = ap.parse_args()

    cp_dir = _resolve_checkpoint(args.blueprint_path, args.checkpoint)
    print(f"checkpoint: {cp_dir}")
    print(f"REGRET_FLOOR={FLOOR:,}   current c (PRUNE_THRESHOLD)={CURRENT_C:,}\n")

    per_street = {}
    pooled = _RegretDist()
    for r in range(4):
        files = _chunk_files(cp_dir, "regret", r)
        dist = _stream_street(files, args.sample_chunks, args.batch_rows)
        if dist is None:
            print(f"[{_STREET_NAME[r]}] no regret chunks found")
            continue
        per_street[r] = dist
        if r in _PRUNABLE_STREETS:
            pooled.merge(dist)

        cur_neg, cur_nz = dist.prune_fracs_at(CURRENT_C)
        n_read, n_sel = dist.n_chunks  # type: ignore[attr-defined]
        note = "" if r in _PRUNABLE_STREETS else "  (river: never pruned)"
        print(f"[{_STREET_NAME[r]:7s}] chunks {n_sel}/{n_read}"
              f"{'  (sampled)' if dist.sampled else ''}{note}")  # type: ignore[attr-defined]
        print(f"    entries {_fmt_int(dist.n_entries)}   "
              f"nonzero {_fmt_int(dist.n_nonzero)}   "
              f"neg {_fmt_int(dist.n_negative)}   pos {_fmt_int(dist.n_positive)}   "
              f"at-floor {_fmt_int(dist.n_at_floor)}")
        print(f"    mean positive regret {_fmt_int(round(dist.mean_positive_regret)) if dist.mean_positive_regret else 'n/a'}"
              f"   most-negative {_fmt_int(dist.min_regret)}")
        print(f"    current c={CURRENT_C:,} prunes: "
              f"{cur_neg*100:5.1f}% of negatives, {cur_nz*100:5.1f}% of nonzero")
        print()

    if pooled.n_negative == 0:
        print("No negative regrets found on prunable streets — nothing to calibrate.")
        return 1

    # Candidate sweep over prunable streets (pooled) + per-street breakdown.
    print("candidate c   |  prunes (% of NEGATIVE entries) per prunable street  |  pooled")
    print("              |   preflop     flop     turn                          |  neg%  nonzero%")
    for c in _SWEEP:
        cells = []
        for r in _PRUNABLE_STREETS:
            if r in per_street:
                fn, _ = per_street[r].prune_fracs_at(c)
                cells.append(f"{fn*100:6.1f}%")
            else:
                cells.append("   n/a")
        pn, pnz = pooled.prune_fracs_at(c)
        marker = "  <- current" if c == CURRENT_C else ""
        print(f"  {c:>13,} |  " + "  ".join(cells)
              + f"     |  {pn*100:5.1f}%  {pnz*100:5.1f}%{marker}")
    print()

    M = pooled.magnitude_at_upper_frac(args.target_neg_frac)
    rec_c = -int(round(M))
    # Keep strictly inside (floor, 0): a c at/below the floor disables pruning.
    rec_c = max(rec_c, FLOOR + 1)
    rec_c = min(rec_c, -1)
    rn, rnz = pooled.prune_fracs_at(rec_c)
    print("=" * 68)
    print(f"RECOMMENDED c = {rec_c:,}")
    print(f"  targets {args.target_neg_frac*100:.0f}% of negative entries on "
          f"preflop+flop+turn (pooled)")
    print(f"  -> prunes {rn*100:.1f}% of negatives, {rnz*100:.1f}% of nonzero "
          f"action-entries at prunable nodes")
    print(f"  (current c={CURRENT_C:,} prunes {pooled.prune_fracs_at(CURRENT_C)[0]*100:.1f}% "
          f"of negatives — i.e. pruning is effectively off)")
    print()
    print(f"  run with:  --c {rec_c}")
    print("  tune aggressiveness with --target-neg-frac (higher = prune more).")
    if per_street.get(0) and getattr(per_street[0], "sampled", False):
        print("  NOTE: sampled estimate; re-run with --sample-chunks 0 for exact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
