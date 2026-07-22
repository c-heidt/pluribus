#!/usr/bin/env python
"""Live progress monitor for a running blueprint training job.

Since the average strategy is now computed **offline** (pre-flop-only phi
online, post-flop reconstructed by ``poker_ai train average`` from retained
snapshots), ``evaluation.blueprint_metrics`` can no longer be used to *watch a
run progress* — on a live training checkpoint its post-flop strategy numbers
read as empty by design.  This tool fills that gap: it reports what a live run
actually exposes, without touching or slowing the trainer.

Two sources, both read-only:

* **The trainer's progress log** (``logs/training-<jobid>.out`` on the cluster,
  or wherever you redirected stdout).  The server prints
  ``[t=... sync_step=...] elapsed=...h remaining≈...h`` every ~minute; parsing
  the last two gives live throughput (traversals-per-player / h), an ETA, and
  the current training *phase* (warm-up / LCFR discount / CFR-P pruning /
  snapshotting) derived from the run's own ``config.yaml``.
* **The run directory** (the ``--nickname`` save dir): the retained
  ``checkpoint_*`` snapshots that feed the offline average, and — with
  ``--regret`` — the per-street regret health + coverage from the newest
  snapshot, which is the live "is it actually learning?" signal (regret matching
  over these rows is exactly the last-iterate policy search/eval uses).

Usage
-----
    # one-shot
    python scripts/monitor_training.py /path/to/run_dir
    # auto-find the newest ./logs/training-*.out, else pass --log
    python scripts/monitor_training.py /path/to/run_dir --log logs/training-12345.out
    # refresh every 20s, include the regret learning panel (sampled I/O)
    python scripts/monitor_training.py /path/to/run_dir --watch --regret

Nothing here writes to the run directory or opens the trainer's shm; it is safe
to run against a live job (including from a second SSH session on the cluster).
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

# Make the repo importable when run as ``scripts/monitor_training.py`` from
# anywhere (reuses the regret streamer instead of duplicating it).
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #

# ``rich`` logging wraps a long record across visual lines and right-aligns a
# ``server.py:545`` source tag; strip both so a record's fields stay contiguous.
_TAG_RE = re.compile(r"\s+[A-Za-z_][\w.]*\.py:\d+")
_TS_RE = re.compile(r"\[\d\d:\d\d:\d\d\]")
_LEVEL_RE = re.compile(r"\b(INFO|WARNING|ERROR|DEBUG|CRITICAL)\b")

# A progress record, after flattening: [t=NNN  sync_step=NNN]  elapsed=X.XXh  remaining≈Y.YYh
_PROGRESS_RE = re.compile(
    r"t=(\d+)\s+sync_step=(\d+)\]\s*elapsed=([\d.]+)h\s*remaining\D*?([\d.]+)h"
)
_DONE_RE = re.compile(
    r"(?:Time limit reached after [\d.]+h —|Training complete —)\s*([\d,]+)"
)
_CKPT_RE = re.compile(r"t=(\d+)\]\s*Checkpoint\s*\(([^)]*)\)\s*starting")
# Two matchers: loose (case-insensitive, only tokens that can't collide with
# benign path/text substrings) and strict (case-sensitive, for tokens that
# WOULD collide — e.g. ``MDB_`` matching ``lmdb_index``, ``OOM`` matching
# ``room``, ``Killed`` matching ``skilled``).
_ERROR_SIGNS = re.compile(
    r"fatal error|WorkerError|Traceback|MemoryError|requeued|"
    r"terminating all workers|Segmentation fault|core dumped",
    re.IGNORECASE,
)
_ERROR_STRICT = re.compile(r"MDB_[A-Z]{3,}|\bOOM\b|\bKilled\b|\bException\b")


def _flatten(text: str) -> str:
    """Strip rich gutter (timestamps, level words, source tags) and unwrap."""
    text = _TAG_RE.sub(" ", text)
    text = _TS_RE.sub(" ", text)
    text = _LEVEL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text)


def _tail(path: Path, max_bytes: int = 400_000) -> str:
    """Read the last ``max_bytes`` of a (possibly huge) log file."""
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        return fh.read().decode("utf-8", "replace")


def _autodetect_log(run_dir: Path) -> Optional[Path]:
    """Newest ``logs/training-*.out`` under CWD or the run dir's parent."""
    candidates: List[Path] = []
    for base in (Path.cwd(), run_dir, run_dir.parent, _REPO):
        candidates += list((base / "logs").glob("training-*.out"))
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


class LogView:
    """Parsed view of the trainer's progress log."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        self.samples: List[Tuple[int, int, float, float]] = []  # t, sync, elapsed_h, remain_h
        self.done_t: Optional[int] = None
        self.ckpts: List[Tuple[int, str]] = []
        self.errors: List[str] = []
        self.mtime: Optional[float] = None
        if path and path.is_file():
            self._parse(_tail(path))
            self.mtime = path.stat().st_mtime

    def _parse(self, raw: str) -> None:
        flat = _flatten(raw)
        for m in _PROGRESS_RE.finditer(flat):
            self.samples.append(
                (int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)))
            )
        d = list(_DONE_RE.finditer(flat))
        if d:
            self.done_t = int(d[-1].group(1).replace(",", ""))
        self.ckpts = [(int(m.group(1)), m.group(2)) for m in _CKPT_RE.finditer(flat)]
        # Keep raw (unflattened) lines for error context — one per record.
        for line in raw.splitlines():
            if _ERROR_SIGNS.search(line) or _ERROR_STRICT.search(line):
                s = line.strip()
                if s:
                    self.errors.append(s[:200])

    @property
    def last_t(self) -> Optional[int]:
        if self.done_t is not None:
            return self.done_t
        return self.samples[-1][0] if self.samples else None


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #

def _human(n: Optional[float]) -> str:
    if n is None:
        return "—"
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def _dur(hours: Optional[float]) -> str:
    if hours is None:
        return "—"
    secs = int(hours * 3600)
    d, secs = divmod(secs, 86400)
    h, secs = divmod(secs, 3600)
    m, _ = divmod(secs, 60)
    if d:
        return f"{d}d{h:02d}h{m:02d}m"
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m"


def _age(mtime: Optional[float]) -> str:
    if mtime is None:
        return "—"
    return _dur((time.time() - mtime) / 3600.0) + " ago"


BAR_W = 32


def _bar(frac: float) -> str:
    frac = max(0.0, min(1.0, frac))
    fill = int(round(frac * BAR_W))
    return "[" + "█" * fill + "·" * (BAR_W - fill) + f"] {frac * 100:5.1f}%"


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #

def _load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.yaml"
    if cfg_path.is_file():
        with open(cfg_path) as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _snapshots(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("checkpoint_[0-9]*"))


def _server_state(cp: Path) -> dict:
    import pickle

    p = cp / "server_state.pkl"
    if p.is_file():
        try:
            with open(p, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return {}
    return {}


def render_progress(log: LogView, cfg: dict) -> str:
    out = ["── PROGRESS " + "─" * 45]
    sync_interval = int(cfg.get("sync_interval", 1000))
    max_h = cfg.get("max_runtime_hours")
    t = log.last_t

    if t is None:
        out.append("  no [t=…] progress record in the log yet "
                   "(workers still starting, or wrong --log).")
        return "\n".join(out)

    # Throughput: overall from the last sample; recent from the last two.
    overall = recent = None
    if log.samples:
        lt, lsync, lelapsed, lremain = log.samples[-1]
        if lelapsed > 0:
            overall = lt / lelapsed
        if len(log.samples) >= 2:
            pt, _, pe, _ = log.samples[-2]
            if lelapsed > pe:
                recent = (lt - pt) / (lelapsed - pe)
        elapsed_h, remain_h = lelapsed, lremain
        sync_step = lsync
    else:  # only a "complete" line
        elapsed_h = remain_h = None
        sync_step = t // sync_interval

    out.append(f"  t (traversals/player) : {_human(t)}   ({t:,})")
    out.append(f"  sync cycle            : {_human(sync_step)}   ({sync_step:,})")
    out.append(f"  throughput  overall   : {_human(overall)}/h"
               + (f"   recent : {_human(recent)}/h" if recent is not None else ""))
    if max_h:
        frac = min(1.0, elapsed_h / max_h) if elapsed_h else 0.0
        out.append(f"  wall clock            : {_dur(elapsed_h)} / {_dur(max_h)}  "
                   f"(ETA {_dur(remain_h)})")
        out.append("  budget                : " + _bar(frac))
    if log.done_t is not None:
        out.append("  ►► run FINISHED (time limit reached / training complete)")
    if log.mtime is not None:
        stale = (time.time() - log.mtime) > 300
        out.append(f"  log last updated      : {_age(log.mtime)}"
                   + ("   ⚠ >5m — job may be stalled/ended" if stale else ""))
    return "\n".join(out)


def render_phase(log: LogView, cfg: dict) -> str:
    out = ["── SCHEDULE PHASE " + "─" * 39]
    si = int(cfg.get("sync_interval", 1000))
    t = log.last_t
    if t is None:
        out.append("  (waiting for first progress record)")
        return "\n".join(out)
    sync_step = log.samples[-1][1] if log.samples else t // si

    def phase_line(label, done, detail):
        mark = "✓" if done else "…"
        return f"  {mark} {label:<26}{detail}"

    upd = int(cfg.get("update_threshold", 0))
    ddc = int(cfg.get("discount_duration_cycles", 0))
    csc = int(cfg.get("checkpoint_start_cycles", 0))
    prune = int(cfg.get("prune_threshold", 0))

    # LCFR discount window (cycles)
    out.append(phase_line("LCFR discounting", sync_step >= ddc,
                          f"window {ddc:,} cyc — "
                          + ("closed" if sync_step >= ddc
                             else f"{ddc - sync_step:,} cyc left")))
    # Pre-flop φ + snapshot warm-up (cycles) — shared gate
    warm = max(upd, csc)
    out.append(phase_line("warm-up (φ + snapshots)", sync_step >= warm,
                          f"gate {warm:,} cyc — "
                          + ("open: φ accumulating, snapshots retained"
                             if sync_step >= warm
                             else f"{warm - sync_step:,} cyc left (near-random era excluded)")))
    # CFR-P pruning (raw per-player traversals)
    out.append(phase_line("CFR-P pruning", t >= prune,
                          f"start {_human(prune)} t — "
                          + ("active" if t >= prune else f"{_human(prune - t)} t left")))
    return "\n".join(out)


def render_snapshots(run_dir: Path, cfg: dict, log: LogView) -> str:
    out = ["── SNAPSHOTS (offline-average inputs) " + "─" * 19]
    snaps = _snapshots(run_dir)
    si = int(cfg.get("sync_interval", 1000))
    warm_t = int(cfg.get("checkpoint_start_cycles", 0)) * si
    if not snaps:
        out.append("  none retained yet (before the checkpoint-start gate, "
                   "or run just began).")
    else:
        newest = snaps[-1]
        st = _server_state(newest)
        newest_t = st.get("t")
        valid = sum(1 for c in snaps if (_server_state(c).get("t", 0) > warm_t))
        out.append(f"  retained checkpoints  : {len(snaps)}  "
                   f"(≥warm-up, avg-eligible: {valid})")
        out.append(f"  newest                : {newest.name}  "
                   f"t={_human(newest_t)}  ({_age(newest.stat().st_mtime)})")
    if log.ckpts:
        lt, label = log.ckpts[-1]
        out.append(f"  last ckpt event (log) : t={_human(lt)} ({label}), "
                   f"{len(log.ckpts)} seen in log tail")
    out.append("  → build the post-flop blueprint with:  poker_ai train average "
               f"--train_dir {run_dir} --output_dir <out>")
    return "\n".join(out)


def render_regret(run_dir: Path, sample_chunks: int) -> str:
    """Per-street coverage + regret health from the newest snapshot."""
    out = ["── LEARNING SIGNAL — newest snapshot (regret) " + "─" * 11]
    try:
        from evaluation.blueprint_metrics import (
            _resolve_checkpoint,
            _chunk_files,
            compute_regret_metrics,
        )
        import numpy as np
    except Exception as exc:  # pragma: no cover
        out.append(f"  (could not import regret helpers: {exc})")
        return "\n".join(out)

    try:
        cp = _resolve_checkpoint(run_dir, None)
    except FileNotFoundError:
        out.append("  no checkpoint to read yet (regret health appears once the "
                   "first snapshot lands).")
        return "\n".join(out)

    out.append(f"  snapshot: {cp.name}  ({_age(cp.stat().st_mtime)})")
    out.append(f"  {'street':<8}{'rows':>12}{'pos%':>8}{'mean+reg':>12}{'floor%':>8}")
    streets = ["pre", "flop", "turn", "river"]
    any_row = False
    for r, name in enumerate(streets):
        files = _chunk_files(cp, "regret", r)
        rows = 0
        for f in files:  # header-only read → cheap exact coverage
            rows += int(np.load(f, mmap_mode="r").shape[0])
        rm = compute_regret_metrics(files, sample_chunks=sample_chunks)
        if rm is None:
            out.append(f"  {name:<8}{rows:>12,}{'—':>8}{'—':>12}{'—':>8}")
            continue
        any_row = True
        mp = rm["mean_positive_regret"]
        out.append(
            f"  {name:<8}{rows:>12,}"
            f"{rm['frac_positive'] * 100:>7.1f}%"
            f"{(_human(mp) if mp is not None else '—'):>12}"
            f"{rm['frac_at_floor'] * 100:>7.1f}%"
        )
    if any_row:
        out.append("  pos% ↑ and mean+reg ↑ over time ⇒ regret mass accumulating "
                   "(non-uniform last-iterate policy).")
    if sample_chunks:
        out.append(f"  (regret health sampled over ≤{sample_chunks} chunks/street; "
                   "rows/coverage are exact)")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def render(run_dir: Path, log_path: Optional[Path], regret: bool,
           sample_chunks: int) -> str:
    cfg = _load_config(run_dir)
    log = LogView(log_path)
    banner = f"TRAINING MONITOR  ·  {run_dir}"
    if cfg.get("n_players"):
        banner += f"  ·  {cfg['n_players']}p"
    if log.path:
        banner += f"  ·  log: {log.path.name}"
    else:
        banner += "  ·  log: (none — pass --log for live throughput)"
    parts = [banner, ""]
    parts.append(render_progress(log, cfg))
    parts.append("")
    parts.append(render_phase(log, cfg))
    parts.append("")
    parts.append(render_snapshots(run_dir, cfg, log))
    if regret:
        parts.append("")
        parts.append(render_regret(run_dir, sample_chunks))
    if log.errors:
        parts.append("")
        parts.append("── ⚠ ERROR/WARN LINES (log tail) " + "─" * 24)
        for e in log.errors[-6:]:
            parts.append("  " + e)
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path,
                    help="Training save directory (the --nickname dir): holds "
                         "config.yaml + checkpoint_* + lmdb_index.")
    ap.add_argument("--log", type=Path, default=None,
                    help="Trainer stdout/SLURM .out. Auto-detects the newest "
                         "./logs/training-*.out if omitted.")
    ap.add_argument("--watch", nargs="?", const=20, type=int, default=None,
                    metavar="SECS",
                    help="Refresh every SECS seconds (default 20). Omit for a "
                         "one-shot report.")
    ap.add_argument("--regret", action="store_true",
                    help="Also read the newest snapshot for per-street coverage "
                         "+ regret health (the live 'is it learning' signal). "
                         "Sampled I/O; safe on a live run.")
    ap.add_argument("--sample-chunks", type=int, default=4,
                    help="Chunks/street to sample for regret health "
                         "(0 = all; default 4).")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        ap.error(f"run_dir not found: {run_dir}")
    log_path = args.log or _autodetect_log(run_dir)

    if args.watch is None:
        print(render(run_dir, log_path, args.regret, args.sample_chunks))
        return

    interval = max(2, args.watch)
    try:
        while True:
            # Re-detect the log each tick (a new job writes a new .out).
            lp = args.log or _autodetect_log(run_dir)
            body = render(run_dir, lp, args.regret, args.sample_chunks)
            sys.stdout.write("\033[2J\033[H")  # clear + home
            sys.stdout.write(body + f"\n\n(refresh {interval}s · Ctrl-C to quit · "
                             + time.strftime("%H:%M:%S") + ")\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
