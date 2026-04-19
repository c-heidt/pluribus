"""
Rebind a card_info_lut.joblib after moving its save_dir to a new location.

Usage:
    python rebind_lut.py <new_save_dir>

Example:
    python rebind_lut.py /new/path/to/monte_carlo

The script updates the internal cluster_ids.dat paths stored inside each
MemmapLookup object in card_info_lut.joblib, then saves the file in place.
centroids.joblib does not need rebinding (it contains only numpy arrays).
"""
import sys
from pathlib import Path

import joblib


def rebind(save_dir: str):
    save_dir = Path(save_dir).resolve()
    lut_path = save_dir / "card_info_lut.joblib"

    if not lut_path.exists():
        print(f"ERROR: {lut_path} not found.")
        sys.exit(1)

    print(f"Loading {lut_path} ...")
    lut = joblib.load(lut_path)

    rebound = []
    for street in ("river", "turn", "flop"):
        if street not in lut:
            continue
        new_ids_path = save_dir / street / "cluster_ids.dat"
        if not new_ids_path.exists():
            print(f"WARNING: {new_ids_path} not found — skipping {street}.")
            continue
        lut[street].rebind(str(new_ids_path))
        rebound.append(street)
        print(f"  {street}: -> {new_ids_path}")

    if not rebound:
        print("No streets rebound. Nothing saved.")
        return

    print(f"Saving updated {lut_path} ...")
    joblib.dump(lut, lut_path)
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    rebind(sys.argv[1])
