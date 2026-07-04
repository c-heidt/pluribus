#!/usr/bin/env python3
"""Rank compute vs lookup/sync across one or more py-spy speedscope captures.

The training profiler (``PROFILE=1`` in ``scripts/training.sh``) writes py-spy
``--format speedscope`` files — one per capture window — under
``$NICKNAME/profiles/``.  Each file aggregates every thread of every worker
(``--subprocesses``) and, because the capture uses ``--idle``, includes
OFF-cpu waits (alloc lock, IPC ``_recv``, LMDB txns) as well as on-cpu compute.

This script reduces each capture to a single question: **where did the fleet
spend its time — running poker logic (compute) or coordinating (lookup/sync)?**
It does that two ways:

* **Mutually-exclusive attribution** (sums to ~100%): every sample is charged to
  the *innermost recognised* frame in its stack (walk leaf→root, first match
  wins), so low-level leaves (numpy, builtins) are attributed to the poker
  function that called them.  Buckets roll up into COMPUTE vs LOOKUP/SYNC.
* **Inclusive per-frame** table: fraction of samples whose stack *contains* a
  given frame — the raw "total time" ranking, easy to sanity-check.

With several captures it prints a trend table (chronological columns) so the
maturation curve of a fresh run — allocation/lookup decaying, compute holding
flat as the index saturates — is a table instead of four browser sessions.

Usage
-----
    python scripts/summarize_speedscope.py $NICKNAME/profiles/*.speedscope.json
    python scripts/summarize_speedscope.py            # defaults to ./*.speedscope.json
    python scripts/summarize_speedscope.py a.json b.json --top 25

Speedscope "sampled" format (what py-spy emits): ``shared.frames`` is a flat
list of ``{name, file, line}``; each ``profiles[i]`` is one thread with
``samples`` (each a list of frame indices, root-first) and parallel ``weights``
(per-sample time).  Fractions are weight-normalised, so the weight unit is
irrelevant.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Bucket classification.  Each rule: (top_bucket, subsystem, predicate).  A
# frame is classified by the FIRST matching rule, so order most-specific first.
# Predicates take (name_lower, file_lower).  Keep the names in sync with the
# hot-path map in memory/project_training_speedups.md.
# --------------------------------------------------------------------------
COMPUTE = "COMPUTE"
SYNC = "LOOKUP/SYNC"
OTHER = "OTHER"


def _name_in(*needles):
    return lambda n, f: any(x in n for x in needles)


def _file_has(*needles):
    return lambda n, f: any(x in f for x in needles)


def _both(name_needles, file_needles):
    return lambda n, f: any(x in n for x in name_needles) and any(
        y in f for y in file_needles
    )


# Ordered rules.  First match wins per frame.
_RULES: List[Tuple[str, str, object]] = [
    # ---- COMPUTE -------------------------------------------------------
    (COMPUTE, "env-step", _name_in(
        "_apply_action_in_place", "_capture_undo_token", "_compute_raise_chip",
        "_get_available_raise_sizes", "_current_public_state", "encode_info_set",
        "legal_actions", "_available_actions", "_state_version",
    )),
    # `undo` / `current_player` / `player_i` are generic names — pin by file.
    (COMPUTE, "env-step", _both(
        ("undo", "current_player", "player_i", "betting_round", "payout"),
        ("poker_env", "environment/"),
    )),
    (COMPUTE, "evaluator", _name_in(
        "_seven", "_six", "_five", "evaluate_batch", "prime_product",
        "_build_multicard", "_build_vectorised", "_eval5_vec",
    )),
    (COMPUTE, "evaluator", _both(("evaluate",), ("evaluator",))),
    (COMPUTE, "showdown", _name_in(
        "rank_players_by_best_hand", "side_pots", "get_rank_class",
        "_settle", "showdown",
    )),
    (COMPUTE, "regret/strategy", _name_in(
        "calculate_strategy_from_row", "get_node_strategy", "sample_action",
        "accumulate_regrets", "regret_match",
    )),
    # ---- LOOKUP/SYNC ---------------------------------------------------
    (SYNC, "index", _name_in(
        "get_or_create", "_get_or_create_once", "_lmdb_get_or_create",
        "hash_info_set", "get_row_if_exists", "_locate_row", "merge_delta_rows",
        "merge_local_delta",
    )),
    (SYNC, "index", _both(("probe", "insert", "get"), ("shm_index_cache", "index.py"))),
    (SYNC, "lmdb", _file_has("lmdb")),
    (SYNC, "lmdb", _name_in("mdb_", "_txn", "cursor")),
    (SYNC, "alloc/lock", _both(
        ("__enter__", "__exit__", "acquire", "release", "wait", "get_lock"),
        ("synchronize", "threading", "multiprocessing"),
    )),
    (SYNC, "alloc/lock", _name_in("_locate_row", "_n_allocated")),
    (SYNC, "ipc", _both(
        ("recv", "send", "poll", "_recv", "_send"),
        ("connection", "queues", "multiprocessing"),
    )),
    (SYNC, "memmap/io", _both(("__getitem__", "flush", "sync"), ("memmap", "chunk"))),
    # ---- OTHER (known but neither) ------------------------------------
    (OTHER, "traversal", lambda n, f: n in ("cfr", "cfrp", "update_strategy")),
]


def classify_frame(name: str, file: str) -> Optional[Tuple[str, str]]:
    n, f = name.lower(), file.lower()
    for top, sub, pred in _RULES:
        if pred(n, f):
            return top, sub
    return None


# --------------------------------------------------------------------------
# Per-capture reduction
# --------------------------------------------------------------------------
class Capture:
    def __init__(self, path: Path):
        self.path = path
        self.stamp = self._parse_stamp(path.name)
        data = json.loads(path.read_text())
        frames = data.get("shared", {}).get("frames", [])
        # Precompute (top, sub) per frame index — frames are shared across all
        # samples, so classify once.
        self._fclass: List[Optional[Tuple[str, str]]] = [
            classify_frame(fr.get("name", ""), fr.get("file", "") or "")
            for fr in frames
        ]
        self._fname = [fr.get("name", "?") for fr in frames]

        self.total_w = 0.0
        # Mutually-exclusive: (top, sub) -> weight (innermost recognised frame).
        self.excl: Dict[Tuple[str, str], float] = defaultdict(float)
        # Inclusive: frame index -> weight of samples whose stack contains it.
        self.incl: Dict[int, float] = defaultdict(float)
        self.unrecognised_w = 0.0

        for prof in data.get("profiles", []):
            samples = prof.get("samples", [])
            weights = prof.get("weights") or [1.0] * len(samples)
            for stack, w in zip(samples, weights):
                if not stack:
                    continue
                w = float(w)
                self.total_w += w
                # Inclusive: each distinct frame in the stack once.
                for idx in set(stack):
                    self.incl[idx] += w
                # Exclusive: innermost recognised frame (leaf -> root).
                hit = None
                for idx in reversed(stack):
                    c = self._fclass[idx] if idx < len(self._fclass) else None
                    if c is not None:
                        hit = c
                        break
                if hit is None:
                    self.unrecognised_w += w
                    self.excl[(OTHER, "unrecognised")] += w
                else:
                    self.excl[hit] += w

    @staticmethod
    def _parse_stamp(fname: str) -> int:
        m = re.search(r"spy_(\d+)", fname)
        return int(m.group(1)) if m else 0

    def pct(self, w: float) -> float:
        return 100.0 * w / self.total_w if self.total_w else 0.0

    def top_pct(self) -> Dict[str, float]:
        out = defaultdict(float)
        for (top, _sub), w in self.excl.items():
            out[top] += self.pct(w)
        return out

    def sub_rows(self) -> List[Tuple[str, str, float]]:
        rows = [(top, sub, self.pct(w)) for (top, sub), w in self.excl.items()]
        rows.sort(key=lambda r: -r[2])
        return rows

    def inclusive_top(self, n: int) -> List[Tuple[str, float]]:
        rows = [(self._fname[i], self.pct(w)) for i, w in self.incl.items()]
        rows.sort(key=lambda r: -r[1])
        return rows[:n]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _bar(pct: float, width: int = 24) -> str:
    filled = int(round(pct / 100.0 * width))
    return "█" * filled + "·" * (width - filled)


def render(captures: List[Capture], top_n: int) -> None:
    labels = [f"t{i+1}" for i in range(len(captures))]

    # ---- headline trend: COMPUTE vs LOOKUP/SYNC vs OTHER ----------------
    print("=" * 72)
    print("HEADLINE — mutually-exclusive time share (innermost recognised frame)")
    print("=" * 72)
    hdr = "  ".join(f"{l:>7}" for l in labels)
    print(f"{'bucket':<16}{hdr}")
    for top in (COMPUTE, SYNC, OTHER):
        cells = "  ".join(f"{c.top_pct().get(top, 0.0):6.1f}%" for c in captures)
        print(f"{top:<16}{cells}")
    print()
    for i, c in enumerate(captures):
        tp = c.top_pct()
        print(f"  {labels[i]}: {c.path.name}")
        print(f"       COMPUTE      {tp.get(COMPUTE,0):5.1f}% {_bar(tp.get(COMPUTE,0))}")
        print(f"       LOOKUP/SYNC  {tp.get(SYNC,0):5.1f}% {_bar(tp.get(SYNC,0))}")
    print()

    # ---- subsystem trend ----------------------------------------------
    print("=" * 72)
    print("SUBSYSTEM BREAKDOWN — % of time per capture (chronological)")
    print("=" * 72)
    subs: List[Tuple[str, str]] = []
    seen = set()
    for c in captures:
        for top, sub, _ in c.sub_rows():
            if (top, sub) not in seen:
                seen.add((top, sub))
                subs.append((top, sub))
    order = {COMPUTE: 0, SYNC: 1, OTHER: 2}
    subs.sort(key=lambda ts: (order.get(ts[0], 9), ts[1]))
    print(f"{'bucket / subsystem':<28}{hdr}")
    cur_top = None
    for top, sub in subs:
        if top != cur_top:
            print(f"{top}")
            cur_top = top
        pcts = []
        for c in captures:
            w = c.excl.get((top, sub), 0.0)
            pcts.append(f"{c.pct(w):6.1f}%")
        print(f"  {sub:<26}{'  '.join(pcts)}")
    print()

    # ---- inclusive top frames for the LAST (most mature) capture -------
    last = captures[-1]
    print("=" * 72)
    print(f"INCLUSIVE TOP {top_n} FRAMES — {labels[-1]} ({last.path.name})")
    print("  (fraction of samples whose stack CONTAINS the frame; includes callees)")
    print("=" * 72)
    for name, pct in last.inclusive_top(top_n):
        cls = classify_frame(name, "")  # name-only hint
        tag = ""
        for top, sub, pred in _RULES:
            if pred(name.lower(), ""):
                tag = f"  [{top}:{sub}]"
                break
        print(f"  {pct:6.1f}%  {name}{tag}")
    print()
    print("Read: COMPUTE flat while LOOKUP/SYNC decays t1->tN => mature run is")
    print("compute-bound (lever: env-step Cython).  LOOKUP/SYNC stays high =>")
    print("index/IPC path is the real bottleneck at scale.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="*", help="speedscope JSON files (or globs)")
    ap.add_argument("--top", type=int, default=20, help="inclusive frames to list")
    args = ap.parse_args()

    raw = args.paths or ["*.speedscope.json"]
    files: List[Path] = []
    for p in raw:
        expanded = glob.glob(p)
        files.extend(Path(x) for x in (expanded or ([p] if Path(p).exists() else [])))
    files = sorted(set(files), key=lambda p: (Capture._parse_stamp(p.name), p.name))
    if not files:
        print(f"No speedscope files matched: {raw}", file=sys.stderr)
        return 1

    print(f"Loading {len(files)} capture(s):")
    captures: List[Capture] = []
    for f in files:
        try:
            c = Capture(f)
        except Exception as e:  # noqa: BLE001 - report and skip bad files
            print(f"  SKIP {f}: {e}", file=sys.stderr)
            continue
        captures.append(c)
        print(f"  {f.name}: {c.total_w:,.0f} sample-weight, "
              f"{100.0 - c.pct(c.unrecognised_w):.0f}% recognised")
    print()
    if not captures:
        return 1
    render(captures, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
