"""Card-information lookup table: consumption-side API.

The build pipeline in :mod:`information_abstraction.build` writes
``card_info_lut.joblib`` to disk; this module loads it and exposes the
``MemmapLookup`` class that replaces a multi-billion-entry Python dict for the
river street.

Nothing in this module depends on the build subpackage, so importing it pulls
no sklearn / multiprocessing machinery into the environment.
"""
import logging
import mmap as _mmap
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np

try:
    from math import comb
except ImportError:
    from scipy.special import comb as _comb

    def comb(n, k):
        return int(_comb(n, k, exact=True))


def lex_rank(combo: Tuple[int, ...], n: int) -> int:
    """Lexicographic rank of a combination in O(k) time.

    Given a k-combination ``(c_0, c_1, ..., c_{k-1})`` with strictly
    ascending elements drawn from ``{0, 1, ..., n-1}``, returns its
    position (0-based) in lexicographic order among all ``C(n, k)``
    combinations.

    Uses the identity
    ``sum_{j=a}^{b-1} C(n-j-1, r) = C(n-a, r+1) - C(n-b, r+1)``
    to avoid an inner loop.

    Shared with :class:`information_abstraction.build.card_combos.CardCombos`
    and the per-street EHS extractors in
    :mod:`information_abstraction.build.ehs` so the combinadic row
    layout is defined in exactly one place.

    Parameters
    ----------
    combo : Tuple[int, ...]
        Strictly ascending k-combination of elements drawn from
        ``{0, ..., n-1}``.
    n : int
        Size of the universe the combination was drawn from.

    Returns
    -------
    int
        0-based lexicographic rank in the range ``[0, C(n, k))``.
    """
    k = len(combo)
    rank = 0
    prev = -1
    for i in range(k):
        start = prev + 1
        remaining = k - i
        rank += comb(n - start, remaining) - comb(n - combo[i], remaining)
        prev = combo[i]
    return rank


log = logging.getLogger("information_abstraction.lookup")


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

    The memmap itself is opened lazily on the first lookup and re-opened
    transparently after pickling so that worker processes which receive the
    object across a ``multiprocessing`` boundary each set up their own file
    handle without the parent needing to reconnect first.

    Call :meth:`rebind` after moving ``cluster_ids.dat`` to a new location.

    Parameters
    ----------
    ids_path : str or Path
        Path to ``cluster_ids.dat`` (a ``uint16`` memmap of length ``n_rows``).
    card_to_idx : Dict[int, int]
        Mapping from eval-card integer to the 0-based index of that card in
        the ascending deck.  Shared with the combo generator so that row
        indices computed here match row layout on disk exactly.
    n_cards : int
        Deck size.
    n_rows : int
        Number of rows in the memmap — ``C(n_cards, 2) * C(n_cards - 2, k)``
        where ``k`` is the number of public cards for the street.
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
        """Lazily open the memmap on first access."""
        if self._mm is None:
            self._mm = np.memmap(
                self._ids_path, dtype=np.uint16, mode="r",
                shape=(self._n_rows,),
            )

    def __getitem__(self, combo) -> int:
        """Return the cluster id for ``combo = (hole_0, hole_1, *publics)``.

        Parameters
        ----------
        combo : Sequence[int]
            Flat sequence of eval-card integers whose first two entries
            are the hole cards and the remainder are the public cards.

        Returns
        -------
        int
            Cluster id as ``uint16`` widened to a Python ``int``.
        """
        self._load()
        ints = [int(c) for c in combo]
        row = self._get_row_index(ints[:2], ints[2:])
        return int(self._mm[row])

    def _get_row_index(self, hole_ints, public_ints) -> int:
        """Combinadic row index mirroring
        :meth:`information_abstraction.build.card_combos.CardCombos.get_row_index`.

        The rank of the hole pair in lexicographic order is multiplied by
        the number of public combinations drawn from the remaining
        ``n - 2`` cards; the public cards are re-indexed into that reduced
        deck before their own lex rank is computed.  Keeping the public
        layout hole-dependent removes any need to store mapping tables on
        disk.

        Parameters
        ----------
        hole_ints : Sequence[int]
            Two hole cards as eval-card integers, in any order.
        public_ints : Sequence[int]
            Public cards as eval-card integers, in any order.

        Returns
        -------
        int
            Row index into ``cluster_ids.dat``.
        """
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

    def prewarm(self) -> int:
        """Pull ``cluster_ids.dat`` into the OS page cache via a sequential read.

        Random-access lookups during CFR training would otherwise
        fault one 4 KB page at a time — devastatingly slow when the
        backing file lives on a network filesystem.  Reading the
        whole file once at startup amortises the I/O upfront so every
        subsequent :meth:`__getitem__` is a RAM-speed indexing
        operation.

        Uses raw ``open()`` / ``read()`` rather than a numpy operation
        on the memmap because numpy reductions allocate intermediate
        buffers, which would double peak RAM for a multi-GiB file.

        Returns
        -------
        int
            Number of bytes read (equal to the file size).
        """
        n_bytes = 0
        chunk = 1 << 26  # 64 MiB
        with open(self._ids_path, "rb") as f:
            while True:
                buf = f.read(chunk)
                if not buf:
                    break
                n_bytes += len(buf)
        # Open the memmap so subsequent __getitem__ calls don't pay
        # the first-access setup cost in a worker.
        self._load()
        return n_bytes

    def rebind(self, new_ids_path: Union[str, Path]) -> None:
        """Point this lookup at a new ``cluster_ids.dat`` path.

        Clears the cached memmap so the next access opens the new file.
        Useful when the build directory has been moved between training
        runs.

        Parameters
        ----------
        new_ids_path : str or Path
            New location of ``cluster_ids.dat``.
        """
        self._ids_path = str(new_ids_path)
        self._mm = None

    def __getstate__(self):
        """Drop the memmap handle before pickling.

        Memmaps are not portable across processes, so each recipient of the
        pickled object opens its own handle on first use via :meth:`_load`.
        """
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

    Parameters
    ----------
    lut_path : str, Path, or None
        Directory containing either ``card_info_lut.joblib`` or the
        four legacy per-street pickle files.  Empty or ``None`` is
        treated as "no LUT on disk" when ``pickle_dir=False``.
    pickle_dir : bool, optional
        Selects the legacy four-file layout when ``True``.  Defaults
        to ``False``.

    Returns
    -------
    InfoSetLut
        Dictionary keyed by street (``"pre_flop"``, ``"flop"``,
        ``"turn"``, ``"river"``).  The river entry is a
        :class:`MemmapLookup` instance; the others are plain dicts.

    Raises
    ------
    ValueError
        If ``pickle_dir=True`` and ``lut_path`` is empty, or if any of
        the expected legacy pickle files is missing.
    """
    if pickle_dir:
        if not lut_path:
            raise ValueError("pickle_dir=True requires a non-empty lut_path")
        base = Path(lut_path)
        log.info("Loading card LUT (legacy pickle-dir) from %s", base)
        t0 = time.monotonic()
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
        log.info("Card LUT loaded in %.1fs", time.monotonic() - t0)
        return out

    if not lut_path:
        return {}

    lut_file = Path(lut_path) / "card_info_lut.joblib"
    log.info("Loading card LUT from %s", lut_file)
    t0 = time.monotonic()
    with open(lut_file, "rb") as f:
        with _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ) as mm:
            out = joblib.load(mm)
    log.info("Card LUT loaded in %.1fs", time.monotonic() - t0)

    # MemmapLookup._ids_path is baked into the joblib at build time as
    # an absolute path.  When the LUT directory is moved (e.g. rsync'd
    # to node-local fast scratch), the deserialised lookup still points
    # at the original location.  Rebind any MemmapLookup whose sibling
    # cluster_ids.dat exists under the current lut_path so all reads
    # hit the local copy.
    lut_root = Path(lut_path)
    for stage, entry in out.items():
        if not isinstance(entry, MemmapLookup):
            continue
        candidate = lut_root / stage / "cluster_ids.dat"
        if candidate.is_file() and Path(entry._ids_path) != candidate:
            log.info(
                "Rebinding %s LUT memmap: %s → %s",
                stage, entry._ids_path, candidate,
            )
            entry.rebind(candidate)
    return out


def prewarm_lut(lut: InfoSetLut) -> int:
    """Pre-warm every memmap-backed street in *lut* into the page cache.

    For 52-card decks the river uses :class:`MemmapLookup`, which
    page-faults each lookup into the kernel page cache on demand.
    On a network filesystem that becomes a per-lookup network round
    trip and dominates worker wall time.  Pre-warming reads the
    whole file once upfront so every subsequent CFR lookup is
    RAM-speed (assuming enough RAM to keep the pages resident).

    Pre/flop/turn streets are plain Python dicts produced eagerly by
    ``joblib.load`` and need no further warming — they are already
    resident in heap memory.

    Parameters
    ----------
    lut : InfoSetLut
        LUT as returned by :func:`load_info_set_lut`.

    Returns
    -------
    int
        Total bytes paged into the OS cache across all
        :class:`MemmapLookup` entries.  Useful for the caller to log
        throughput.
    """
    total = 0
    for stage, entry in lut.items():
        if isinstance(entry, MemmapLookup):
            log.info("Pre-warming %s LUT (%s)", stage, entry._ids_path)
            t0 = time.monotonic()
            n_bytes = entry.prewarm()
            elapsed = time.monotonic() - t0
            log.info(
                "Pre-warmed %s LUT: %.2f GiB in %.1fs (%.0f MiB/s)",
                stage,
                n_bytes / 1024 ** 3,
                elapsed,
                n_bytes / max(elapsed, 1e-9) / 1024 ** 2,
            )
            total += n_bytes
    return total
