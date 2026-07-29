"""Compare play-frequency composition of rows a stricter offline-average gate
kept vs. rows it filtered out.

Tests whether the confirmation gates in ``poker_ai/blueprint/offline_average.py``
(``min_confirming_fraction`` / ``min_snapshot_regret_magnitude``) prune infosets
uniformly across action types, or disproportionately drop close/marginal
decisions while keeping high-conviction ones (e.g. all-in nodes past
``MAX_RAISES_PER_ROUND``, which has nothing to split its regret with).

Usage::

    python scripts/compare_gate_filtering.py \\
        --loose /path/to/blueprint_v2 \\
        --strict /path/to/blueprint_v4 \\
        --streets flop turn river \\
        --workers 16

For each street, streams both blueprints' strategy chunk files in lockstep —
they share one ``lmdb_index`` (both built from the same ``train_dir``), so row
``i`` is the same infoset in both — and splits rows into:

- ``both``        — visited (mass > 0) in both blueprints.
- ``loose_only``  — visited in ``--loose`` but NOT in ``--strict``: what the
  stricter gate filtered out.
- ``strict_only`` — visited in ``--strict`` but not ``--loose``. Expected to be
  ~0: a strictly harder confirmation requirement (higher required count/
  fraction, plus an optional per-snapshot magnitude floor) can only shrink the
  confirmed set, never grow it, for two runs averaging the same snapshots. A
  non-trivial count here means that assumption doesn't hold for this pair of
  runs (e.g. different ``--min_t`` / retained snapshots) and the comparison
  below should be treated with caution.

Reports the ``--loose`` blueprint's own play-frequency composition separately
for each group, using ONE consistent strategy source (the loose run) for both
groups — this isolates the pure compositional question: of the rows that were
confirmed before, does the subset the stricter gate now rejects favor
different actions than the subset it still accepts?

Parallelized like ``poker_ai train average``: the unit of work is one
``(street, chunk)`` pair, read from both blueprints and reduced to small
per-group tallies (row count, mass, action-mass — not the raw rows) in the
worker process, so IPC stays cheap regardless of chunk size. ``--workers``
processes chunks concurrently across ALL requested streets in one pool
(river dominates the work, so a flat task list keeps every worker busy to the
end instead of draining at each street boundary).
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.blueprint_metrics import (  # noqa: E402
    _STREET_NAME,
    _action_labels,
    _chunk_files,
    _resolve_checkpoint,
)

_DEFAULT_BATCH_ROWS = 1_000_000
_STREET_BY_NAME = {v: k for k, v in _STREET_NAME.items()}
_GROUP_KEYS = ("both", "loose_only", "strict_only")


def _bucket(freqs: Dict[str, float]) -> Dict[str, float]:
    out = {"fold": 0.0, "call": 0.0, "all_in": 0.0, "raise": 0.0}
    for a, f in freqs.items():
        out[a if a in out else "raise"] += f
    return out


class _GroupAcc:
    """Weighted action-mass accumulator for one comparison group."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.n_rows = 0
        self.total_mass = 0.0
        self.action_mass = np.zeros(width, dtype=np.float64)

    def add_partial(self, n_rows: int, total_mass: float, action_mass: np.ndarray) -> None:
        self.n_rows += n_rows
        self.total_mass += total_mass
        self.action_mass += action_mass

    def play_freq(self, labels: Sequence[str]) -> Dict[str, float]:
        if self.total_mass <= 0:
            return {}
        freqs = self.action_mass / self.total_mass
        return {a: float(f) for a, f in zip(labels, freqs)}


def _chunk_task(args: Tuple[int, Path, Path, int]) -> dict:
    """Reduce one ``(street, chunk)`` pair to per-group ``(n_rows, mass, action_mass)``.

    Runs in a worker process; returns only small aggregates (arrays of size
    ``width``, not the raw chunk), so pickling the result back is cheap.
    """
    street, loose_path, strict_path, batch_rows = args
    loose_arr = np.load(loose_path, mmap_mode="r")
    strict_arr = np.load(strict_path, mmap_mode="r")
    if loose_arr.shape != strict_arr.shape:
        raise ValueError(
            f"street {street}: shape mismatch in {loose_path.name} "
            f"({loose_arr.shape} vs {strict_arr.shape}) — not row-aligned."
        )
    width = loose_arr.shape[1]
    groups = {k: _GroupAcc(width) for k in _GROUP_KEYS}

    for start in range(0, loose_arr.shape[0], batch_rows):
        end = start + batch_rows
        loose_batch = np.asarray(loose_arr[start:end])
        strict_batch = np.asarray(strict_arr[start:end])
        loose_visited = loose_batch.sum(axis=1) > 0
        strict_visited = strict_batch.sum(axis=1) > 0

        for key, mask in (
            ("both", loose_visited & strict_visited),
            ("loose_only", loose_visited & ~strict_visited),
            ("strict_only", ~loose_visited & strict_visited),
        ):
            rows = loose_batch[mask].astype(np.float64)
            if rows.shape[0] == 0:
                continue
            groups[key].add_partial(rows.shape[0], float(rows.sum()), rows.sum(axis=0))

    return {
        "street": street,
        "width": width,
        "groups": {
            k: (g.n_rows, g.total_mass, g.action_mass) for k, g in groups.items()
        },
    }


def _run_tasks(tasks: List[Tuple[int, Path, Path, int]], workers: int) -> List[dict]:
    if not tasks:
        return []
    if workers <= 1:
        return [_chunk_task(t) for t in tqdm(tasks, desc="Comparing chunks", unit="chunk")]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_chunk_task, t) for t in tasks]
        results = []
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Comparing chunks", unit="chunk"):
            results.append(fut.result())
        return results


def _build_tasks(
    loose_dir: Path, strict_dir: Path, streets: Sequence[str], batch_rows: int
) -> List[Tuple[int, Path, Path, int]]:
    tasks: List[Tuple[int, Path, Path, int]] = []
    for street_name in streets:
        street = _STREET_BY_NAME[street_name]
        loose_files = _chunk_files(loose_dir, "strategy", street)
        strict_files = _chunk_files(strict_dir, "strategy", street)
        if not loose_files or not strict_files:
            print(f"{street_name}: missing strategy chunks in one of the two blueprints — skipped")
            continue
        if len(loose_files) != len(strict_files):
            raise ValueError(
                f"{street_name}: chunk count mismatch ({len(loose_files)} vs "
                f"{len(strict_files)}) — these blueprints don't share the same "
                f"lmdb_index/final checkpoint, row alignment isn't guaranteed."
            )
        tasks.extend(
            (street, lp, sp, batch_rows) for lp, sp in zip(loose_files, strict_files)
        )
    return tasks


def _report(results: List[dict], streets: Sequence[str]) -> None:
    by_street: Dict[int, dict] = {}
    for r in results:
        street = r["street"]
        entry = by_street.setdefault(
            street, {"width": r["width"], "groups": {k: _GroupAcc(r["width"]) for k in _GROUP_KEYS}}
        )
        for key, (n_rows, total_mass, action_mass) in r["groups"].items():
            entry["groups"][key].add_partial(n_rows, total_mass, action_mass)

    for street_name in streets:
        street = _STREET_BY_NAME[street_name]
        entry = by_street.get(street)
        if entry is None:
            continue
        labels, _ = _action_labels(entry["width"], street)
        groups = entry["groups"]
        n_loose_visited = groups["both"].n_rows + groups["loose_only"].n_rows

        print(f"\n{street_name.upper()}")
        print(f"  loose-visited rows: {n_loose_visited:,}")
        if groups["strict_only"].n_rows:
            print(
                f"  ⚠ {groups['strict_only'].n_rows:,} rows visited in --strict but "
                f"NOT in --loose — the strict gate is not a strict subset of the loose "
                f"one for this pair of runs (different --min_t / retained snapshots?)."
            )
        for key, label in (("both", "still confirmed (both)"), ("loose_only", "filtered out by --strict")):
            g = groups[key]
            frac = g.n_rows / n_loose_visited if n_loose_visited else float("nan")
            print(f"  {label}: {g.n_rows:,} rows ({frac:.1%} of loose-visited)")
            b = _bucket(g.play_freq(labels))
            if b:
                print(
                    "    play freq (weighted, --loose values)   "
                    f"fold {b['fold']:.1%}   call {b['call']:.1%}   "
                    f"raise {b['raise']:.1%}   all-in {b['all_in']:.1%}"
                )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--loose", required=True, help="Blueprint built with looser confirmation gates")
    parser.add_argument("--strict", required=True, help="Blueprint built with stricter confirmation gates")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Specific checkpoint_* subdir name to use in BOTH blueprints; default latest",
    )
    parser.add_argument(
        "--streets", nargs="+", default=["flop", "turn", "river"],
        choices=list(_STREET_BY_NAME),
    )
    parser.add_argument("--batch-rows", type=int, default=_DEFAULT_BATCH_ROWS)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="(street, chunk) pairs to compare concurrently, across all requested "
             "streets in one pool. Each worker holds one loose + one strict chunk "
             "in memory at a time (same footprint as poker_ai train average).",
    )
    args = parser.parse_args(argv)

    loose_dir = _resolve_checkpoint(Path(args.loose), args.checkpoint)
    strict_dir = _resolve_checkpoint(Path(args.strict), args.checkpoint)
    print(f"loose:  {loose_dir}")
    print(f"strict: {strict_dir}")

    tasks = _build_tasks(loose_dir, strict_dir, args.streets, args.batch_rows)
    print(f"Comparing {len(tasks)} chunk(s) with {args.workers} worker(s)")
    results = _run_tasks(tasks, args.workers)
    _report(results, args.streets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
