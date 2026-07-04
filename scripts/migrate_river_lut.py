#!/usr/bin/env python3
"""Seed a new abstraction ``save_dir`` that reuses cached river equities.

Why
---
The card abstraction is hierarchical / potential-aware (see
``information_abstraction/build/ehs.py``):

    river features = [win, loss, tie] equity   (the EXPENSIVE showdown MC)
    turn  features = histogram over river clusters
    flop  features = histogram over turn clusters

Changing any bucket count forces everything *downstream* to be rebuilt, because
the downstream feature vectors are histograms over the upstream cluster ids.
The one thing that never depends on a bucket count is the **river equity
computation** — it is cached in ``river/merged_data.dat``.

This script copies that cached river feature matrix into a fresh ``save_dir``
and writes a *correct* ``checkpoint.json`` so a subsequent abstraction build:

  * SKIPS the river equity Monte-Carlo (reuses ``merged_data.dat``),
  * RE-CLUSTERS river with a new ``--n_river_clusters`` (clustering_done=False),
  * REBUILDS turn then flop from scratch against the new river/turn labels.

How the reuse is made safe
--------------------------
``ChunkStore.initialize_street`` only *resets* a street (deleting its ``.dat`` /
``.npy`` and its checkpoint) when the incoming ``n_chunks`` differs from the
checkpoint's ``total_chunks`` (chunk_store.py). We therefore copy river's
``total_chunks`` / ``total_combos`` / ``feature_dim`` verbatim from the source
checkpoint so the rebuild takes the *no-reset* branch and keeps the migrated
merged data. ``completed_chunks`` is set to the full range so
``_process_incomplete_chunks`` finds nothing to do (the source's chunk files were
cleaned up after its own clustering — only ``merged_data.dat`` survives).

IMPORTANT: the rebuild MUST use the same ``chunk_size`` as the source run.
``n_chunks = ceil(total_combos / chunk_size)``; a different chunk_size changes
``n_chunks``, trips the reset branch, and deletes the migrated river data.

Usage
-----
    python scripts/migrate_river_lut.py --src <old_save_dir> --dst <new_save_dir> \
        [--link hardlink|symlink|copy] [--force]

Then build the new LUT (same chunk_size as the source run!):

    python -m information_abstraction.build.runner \
        --n_river_clusters <NEW> --n_turn_clusters <NEW> --n_flop_clusters <NEW> \
        --save_dir <new_save_dir> ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Mirror the on-disk schema from information_abstraction/build/checkpoint.py
# without importing it (keeps the script runnable standalone on the cluster).
_STREETS = ("river", "turn", "flop")
_MERGED_DTYPE_BYTES = 4  # float32, matches ChunkStore.get_or_merge_data default
_RIVER_ARTIFACTS = ("merged_data.dat", "all_combos.npy")


def _empty_street() -> dict:
    return {
        "completed_chunks": [],
        "total_chunks": 0,
        "total_combos": 0,
        "merge_done": False,
        "clustering_done": False,
        "feature_dim": None,
    }


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _place(src: Path, dst: Path, mode: str) -> str:
    """Put ``src`` at ``dst`` via hardlink/symlink/copy; return what happened.

    hardlink falls back to copy across filesystems (EXDEV).  merged_data.dat is
    opened read-only by the build, so a shared link is safe.
    """
    if dst.exists():
        dst.unlink()
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
        return "symlinked"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlinked"
        except OSError:
            shutil.copy2(src, dst)
            return "copied (cross-device; hardlink not possible)"
    shutil.copy2(src, dst)
    return "copied"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--src", type=Path, required=True,
                    help="existing abstraction save_dir with a completed river street")
    ap.add_argument("--dst", type=Path, required=True,
                    help="new save_dir to create (rebuild target)")
    ap.add_argument("--link", choices=("hardlink", "symlink", "copy"),
                    default="hardlink",
                    help="how to place river/merged_data.dat + all_combos.npy "
                         "(default: hardlink, falls back to copy across filesystems)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite dst/river and dst/checkpoint.json if present")
    args = ap.parse_args()

    src: Path = args.src
    dst: Path = args.dst

    # ---- validate source ------------------------------------------------
    src_ckpt_path = src / "checkpoint.json"
    if not src_ckpt_path.exists():
        print(f"ERROR: no checkpoint.json in src: {src}", file=sys.stderr)
        return 1
    src_ckpt = _load_json(src_ckpt_path)
    river_state = (src_ckpt.get("streets") or {}).get("river")
    if not river_state:
        print("ERROR: src checkpoint has no 'river' street state.", file=sys.stderr)
        return 1

    src_river_dir = src / "river"
    for name in _RIVER_ARTIFACTS:
        if not (src_river_dir / name).exists():
            print(f"ERROR: missing src river artifact: {src_river_dir / name}\n"
                  f"       (river equities not cached here — nothing to migrate)",
                  file=sys.stderr)
            return 1

    total_chunks = int(river_state.get("total_chunks", 0))
    total_combos = river_state.get("total_combos")
    feature_dim = river_state.get("feature_dim")
    if not river_state.get("merge_done"):
        print("WARNING: src river 'merge_done' is False — merged_data.dat may be "
              "incomplete. Proceeding, but verify the source build finished river.",
              file=sys.stderr)

    # Derive total_combos / feature_dim from all_combos.npy + file size if the
    # source checkpoint predates those fields (legacy) or is missing them.
    import numpy as np  # local import: only needed for validation

    all_combos = np.load(src_river_dir / "all_combos.npy", allow_pickle=True)
    n_rows = len(all_combos)
    if total_combos is None:
        total_combos = n_rows
    elif int(total_combos) != n_rows:
        print(f"ERROR: src river total_combos ({total_combos}) != len(all_combos) "
              f"({n_rows}).", file=sys.stderr)
        return 1

    merged_bytes = (src_river_dir / "merged_data.dat").stat().st_size
    if merged_bytes % (n_rows * _MERGED_DTYPE_BYTES) != 0:
        print(f"ERROR: merged_data.dat size ({merged_bytes} B) is not a multiple of "
              f"n_rows*{_MERGED_DTYPE_BYTES}; dtype/shape mismatch?", file=sys.stderr)
        return 1
    derived_dim = merged_bytes // (n_rows * _MERGED_DTYPE_BYTES)
    if feature_dim is None:
        feature_dim = int(derived_dim)
        print(f"note: feature_dim absent in src checkpoint; derived {feature_dim} "
              f"from merged_data.dat size.")
    elif int(feature_dim) != derived_dim:
        print(f"ERROR: src feature_dim ({feature_dim}) disagrees with size-derived "
              f"dim ({derived_dim}).", file=sys.stderr)
        return 1
    feature_dim = int(feature_dim)

    if total_chunks <= 0:
        print(f"ERROR: src river total_chunks is {total_chunks}; cannot build a "
              f"consistent completed_chunks range. Was river actually built here?",
              file=sys.stderr)
        return 1

    # ---- guard the destination -----------------------------------------
    dst_lut = dst / "card_info_lut.joblib"
    if dst_lut.exists() and not args.force:
        print(f"ERROR: {dst_lut} exists. The builder skips any street already "
              f"present in the LUT, so a river-containing LUT would defeat the "
              f"re-cluster. Remove it or pass --force.", file=sys.stderr)
        return 1

    dst_river_dir = dst / "river"
    if dst_river_dir.exists() and not args.force:
        print(f"ERROR: {dst_river_dir} already exists. Pass --force to overwrite.",
              file=sys.stderr)
        return 1

    # ---- perform the migration -----------------------------------------
    dst_river_dir.mkdir(parents=True, exist_ok=True)
    placements = []
    for name in _RIVER_ARTIFACTS:
        how = _place(src_river_dir / name, dst_river_dir / name, args.link)
        placements.append(f"    river/{name}: {how}")

    # Remove any stale clustering outputs so nothing pre-empts the re-cluster.
    for stale in ("cluster_ids.dat", "centroids.npy", "clusters.npy"):
        p = dst_river_dir / stale
        if p.exists():
            p.unlink()

    # Build the destination checkpoint: river reused (merge done, clustering
    # pending), turn/flop fresh so they rebuild against the new labels.
    dst_ckpt = {"streets": {}, "config": {}}
    dst_ckpt["streets"]["river"] = {
        "completed_chunks": list(range(total_chunks)),  # -> zero incomplete
        "total_chunks": total_chunks,                   # matches rebuild n_chunks
        "total_combos": int(total_combos),
        "merge_done": True,          # load merged_data.dat, don't re-merge
        "clustering_done": False,    # DO re-cluster with the new K
        "feature_dim": feature_dim,  # needed to size the read-only memmap
    }
    dst_ckpt["streets"]["turn"] = _empty_street()
    dst_ckpt["streets"]["flop"] = _empty_street()

    dst_ckpt_path = dst / "checkpoint.json"
    tmp = dst_ckpt_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(dst_ckpt, f, indent=2)
    os.replace(tmp, dst_ckpt_path)

    # ---- report ---------------------------------------------------------
    src_chunk_size_hint = (src_ckpt.get("config") or {}).get("chunk_size")
    print("Migration complete.")
    print(f"  src: {src}")
    print(f"  dst: {dst}")
    print("\n".join(placements))
    print(f"  river: total_chunks={total_chunks}  total_combos={total_combos:,}  "
          f"feature_dim={feature_dim}  -> merge_done=True, clustering_done=False")
    print("  turn/flop: fresh (will be fully rebuilt)")
    print()
    print("Next — build the new LUT (RE-USES river equities, rebuilds turn+flop):")
    print("  python -m information_abstraction.build.runner \\")
    print("      --n_river_clusters <NEW> --n_turn_clusters <NEW> "
          "--n_flop_clusters <NEW> \\")
    print(f"      --save_dir {dst} ...")
    cs = (f"{src_chunk_size_hint}" if src_chunk_size_hint is not None
          else "the SAME value the source run used")
    print(f"\n  ** Use chunk_size = {cs} ** — a different chunk_size changes")
    print("     n_chunks, trips initialize_street's reset, and deletes the")
    print("     migrated river data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
