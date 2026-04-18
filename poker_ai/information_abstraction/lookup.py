"""Card-information lookup table: consumption-side API.

The build pipeline in :mod:`poker_ai.information_abstraction.build` writes
``card_info_lut.joblib`` to disk; this module loads it and exposes the
``MemmapLookup`` class that replaces a multi-billion-entry Python dict for the
river street.

Nothing in this module depends on the build subpackage, so importing it pulls
no sklearn / multiprocessing machinery into the environment.
"""
import logging
import mmap as _mmap
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np

from poker_ai.information_abstraction._combinatorics import comb, lex_rank

log = logging.getLogger("poker_ai.information_abstraction.lookup")


InfoSetLut = Dict[str, Any]
"""Shape of a loaded LUT: ``{street: {combo_tuple: cluster_id}}``.

For pre-flop / flop / turn the inner value is a plain ``dict``.  For the river
it is a :class:`MemmapLookup` instance for memory efficiency on large decks.
"""


# ---------------------------------------------------------------------------
# MemmapLookup — memory-efficient river street lookup
# ---------------------------------------------------------------------------

class MemmapLookup:
    """Drop-in replacement for the ``{tuple: cluster_id}`` dict on large decks.

    For a 52-card deck the river has ~2.8 billion combos.  A Python dict for
    that many entries requires ~840 GB of RAM and makes ``joblib.dump`` hang
    for hours.  This class reads cluster IDs directly from the compact
    ``uint16`` ``cluster_ids.dat`` memmap written by the build pipeline, using
    O(1) combinadic indexing — consuming only a few KB when pickled.

    Call :meth:`rebind` after moving ``cluster_ids.dat`` to a new location.
    """

    def __init__(
        self,
        ids_path: Union[str, Path],
        card_to_idx: Dict[int, int],
        n_cards: int,
        n_rows: int,
    ):
        self._ids_path = str(ids_path)
        self._card_to_idx = card_to_idx
        self._n_cards = n_cards
        self._n_rows = n_rows
        self._mm: Optional[np.memmap] = None

    def _load(self):
        if self._mm is None:
            self._mm = np.memmap(
                self._ids_path, dtype=np.uint16, mode="r",
                shape=(self._n_rows,),
            )

    def __getitem__(self, combo) -> int:
        self._load()
        ints = [int(c) for c in combo]
        row = self._get_row_index(ints[:2], ints[2:])
        return int(self._mm[row])

    def _get_row_index(self, hole_ints, public_ints) -> int:
        card_to_idx = self._card_to_idx
        n = self._n_cards
        h_idx = sorted(card_to_idx[int(c)] for c in hole_ints)
        p_idx = sorted(card_to_idx[int(c)] for c in public_ints)
        hole_rank = lex_rank(tuple(h_idx), n)
        h0, h1 = h_idx[0], h_idx[1]
        p_reindexed = tuple(p - (h0 < p) - (h1 < p) for p in p_idx)
        n_remaining = n - 2
        k_public = len(p_idx)
        public_rank = lex_rank(p_reindexed, n_remaining)
        return hole_rank * comb(n_remaining, k_public) + public_rank

    def rebind(self, new_ids_path: Union[str, Path]) -> None:
        """Point this lookup at a new ``cluster_ids.dat`` path."""
        self._ids_path = str(new_ids_path)
        self._mm = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# ---------------------------------------------------------------------------
# Loader — single canonical entry point for every consumer
# ---------------------------------------------------------------------------

_LEGACY_PICKLE_FILES: Dict[str, str] = {
    "pre_flop": "preflop_lossless.pkl",
    "flop": "flop_lossy_2.pkl",
    "turn": "turn_lossy_2.pkl",
    "river": "river_lossy_2.pkl",
}


def load_info_set_lut(
    lut_path: Union[str, Path, None],
    pickle_dir: bool = False,
) -> InfoSetLut:
    """Load the card-information LUT from disk.

    Two on-disk layouts are supported:

    - **Joblib + mmap** (``pickle_dir=False``, default): a single
      ``card_info_lut.joblib`` file inside ``lut_path``.  The file is
      mmapped before ``joblib.load`` so multiple worker processes that
      open the same file share its physical pages via the kernel page
      cache; memory usage stays roughly constant in the number of
      workers.
    - **Legacy pickle directory** (``pickle_dir=True``): four per-street
      pickle files inside ``lut_path`` (``preflop_lossless.pkl``,
      ``flop_lossy_2.pkl``, ``turn_lossy_2.pkl``, ``river_lossy_2.pkl``).
      Kept for compatibility with very old datasets.

    When ``lut_path`` is empty or ``None`` and ``pickle_dir=False``,
    an empty dict is returned — this lets callers construct an
    abstraction-free environment without a LUT on disk.
    """
    if pickle_dir:
        if not lut_path:
            raise ValueError("pickle_dir=True requires a non-empty lut_path")
        base = Path(lut_path)
        log.info("Loading card LUT (legacy pickle-dir) from %s", base)
        out: InfoSetLut = {}
        for stage, file_name in _LEGACY_PICKLE_FILES.items():
            fp = base / file_name
            if not fp.is_file():
                raise ValueError(
                    f"File not found: {fp}. "
                    "Ensure lut_path contains the legacy pickle files."
                )
            with open(fp, "rb") as f:
                out[stage] = joblib.load(f)
        return out

    if not lut_path:
        return {}

    lut_file = Path(lut_path) / "card_info_lut.joblib"
    log.info("Loading card LUT from %s", lut_file)
    with open(lut_file, "rb") as f:
        with _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ) as mm:
            return joblib.load(mm)
