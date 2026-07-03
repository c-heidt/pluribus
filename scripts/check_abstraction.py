"""Audit the card-abstraction LUT: bucket counts and id ranges per street.

Usage (on the machine where the LUT lives)::

    python scripts/check_abstraction.py /path/to/lut_dir

Answers one question — *is the LUT actually a bucketed abstraction, or is it
(accidentally) near-lossless?* — by counting distinct cluster ids per street.
Expected for the standard build: pre_flop = 169 (lossless by design),
flop / turn / river = the ``--n_*_clusters`` values passed to the abstraction
builder (e.g. 200 each).  A distinct-count near the number of combos, or a
max id far above the configured cluster count, means the clustering step did
not do its job.

The river street is a :class:`MemmapLookup` over a flat ``cluster_ids.dat``
(billions of rows on a 52-card deck), so it is scanned in chunks with
``bincount`` instead of materialising a unique() over the whole array.
"""

import sys

import numpy as np

from information_abstraction import load_info_set_lut


def _report(street: str, n_entries: int, n_distinct: int, max_id: int) -> None:
    print(
        f"{street:<9} {n_entries:>13,} combos   "
        f"{n_distinct:>6,} distinct clusters   max id {max_id}"
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    lut = load_info_set_lut(sys.argv[1], pickle_dir=False)

    for street in ("pre_flop", "flop", "turn"):
        table = lut[street]
        vals = np.fromiter(table.values(), dtype=np.int64, count=len(table))
        _report(street, len(vals), len(np.unique(vals)), int(vals.max()))

    river = lut["river"]
    river._load()
    ids = river._mm
    counts = np.zeros(0, dtype=np.int64)
    chunk = 100_000_000
    for start in range(0, ids.shape[0], chunk):
        part = np.asarray(ids[start : start + chunk]).astype(np.int64, copy=False)
        hi = int(part.max()) + 1
        if hi > counts.shape[0]:
            counts = np.pad(counts, (0, hi - counts.shape[0]))
        counts += np.bincount(part, minlength=counts.shape[0])
    _report(
        "river",
        int(ids.shape[0]),
        int((counts > 0).sum()),
        int(np.nonzero(counts)[0][-1]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
