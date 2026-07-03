"""Audit the card-abstraction LUT: bucket counts and id ranges per street.

Run this **on the machine where the LUT lives** (the cluster), from the repo
checkout, with the training conda env active::

    conda activate pluribus
    python scripts/check_abstraction.py "$WORKSPACE/exact"

The argument is the LUT *directory* — the one containing
``card_info_lut.joblib`` (the same value ``training.sh`` uses as ``LUT_PATH``,
default ``$WORKSPACE/exact``).

Answers one question — *is the LUT actually a bucketed abstraction, or is it
(accidentally) near-lossless?* — by counting distinct cluster ids per street.
Expected for the standard build: pre_flop = 169 (lossless by design),
flop / turn / river = the ``--n_*_clusters`` values passed to the abstraction
builder (e.g. 200 each).  A distinct-count near the number of combos, or a
max id far above the configured cluster count, means the clustering step did
not do its job.

Memmap-backed streets (the river on 52-card decks, possibly flop/turn too)
are scanned in chunks with ``bincount`` — on a network filesystem this reads
the whole multi-GiB ``cluster_ids.dat`` once, so expect it to take minutes.
"""

import sys
from pathlib import Path

# Make the script runnable without an installed package: the repo root is
# the parent of scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from information_abstraction import load_info_set_lut  # noqa: E402
from information_abstraction.lookup import MemmapLookup  # noqa: E402

_CHUNK = 50_000_000  # ids per bincount batch (~400 MB as int64)


def _report(street: str, n_entries: int, n_distinct: int, max_id: int) -> None:
    print(
        f"{street:<9} {n_entries:>15,} combos   "
        f"{n_distinct:>7,} distinct clusters   max id {max_id}"
    )


def _scan_memmap(street: str, entry: MemmapLookup) -> None:
    entry._load()
    ids = entry._mm
    n = int(ids.shape[0])
    counts = np.zeros(1, dtype=np.int64)
    for start in range(0, n, _CHUNK):
        part = np.asarray(ids[start : start + _CHUNK]).astype(np.int64, copy=False)
        hi = int(part.max()) + 1
        if hi > counts.shape[0]:
            counts = np.pad(counts, (0, hi - counts.shape[0]))
        counts += np.bincount(part, minlength=counts.shape[0])
        done = min(start + _CHUNK, n)
        print(f"  ... {street}: scanned {done:,}/{n:,} rows", file=sys.stderr)
    nonzero = np.nonzero(counts)[0]
    _report(street, n, int(nonzero.size), int(nonzero[-1]))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1

    lut_dir = Path(sys.argv[1])
    joblib_file = lut_dir / "card_info_lut.joblib"
    if not joblib_file.is_file():
        print(f"ERROR: {joblib_file} not found.", file=sys.stderr)
        print(
            "Pass the LUT directory (training.sh's LUT_PATH, default "
            "$WORKSPACE/exact) — the directory that contains "
            "card_info_lut.joblib. This script must run on the machine "
            "where that file lives.",
            file=sys.stderr,
        )
        return 1

    lut = load_info_set_lut(str(lut_dir), pickle_dir=False)

    for street in ("pre_flop", "flop", "turn", "river"):
        entry = lut.get(street)
        if entry is None:
            print(f"{street:<9} (not present in LUT)")
        elif isinstance(entry, MemmapLookup):
            ids_path = Path(entry._ids_path)
            if not ids_path.is_file():
                print(
                    f"ERROR: {street} memmap points at missing file "
                    f"{ids_path}. If the LUT directory was moved, run "
                    f"scripts/rebind_lut.py {lut_dir} (or ensure "
                    f"{lut_dir / street / 'cluster_ids.dat'} exists so the "
                    "loader can auto-rebind).",
                    file=sys.stderr,
                )
                return 1
            _scan_memmap(street, entry)
        else:
            vals = np.fromiter(entry.values(), dtype=np.int64, count=len(entry))
            _report(street, len(vals), len(np.unique(vals)), int(vals.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
