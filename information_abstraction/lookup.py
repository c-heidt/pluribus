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


def _comb_table(n: int, k_max: int) -> np.ndarray:
    """``C[a, r] == comb(a, r)`` for ``a in [0, n]``, ``r in [0, k_max]``.

    The vectorised counterpart of the ``comb`` calls inside :func:`lex_rank`.
    Depends only on the deck size, so callers hoist it (see
    :meth:`MemmapLookup._indexer`) rather than rebuilding it per lookup — that
    hoist is most of the speedup over the scalar path.
    """
    C = np.zeros((n + 1, k_max + 1), dtype=np.int64)
    for a in range(n + 1):
        for r in range(k_max + 1):
            C[a, r] = comb(a, r)
    return C


def _lex_rank_vec(combos: np.ndarray, n: int, C: np.ndarray) -> np.ndarray:
    """Lexicographic ranks of many k-combinations at once.

    Vectorised :func:`lex_rank`: ``combos`` is ``(m, k)`` int64 with strictly
    ascending rows drawn from ``{0, ..., n-1}``; returns ``(m,)`` 0-based ranks.
    Loops over the ``k`` axis (k <= 5 here), not over ``m``, and reads the
    binomials out of the precomputed ``C`` table.  Bit-exact with the scalar
    function by construction: same identity, same integer arithmetic, no floats.
    """
    m, k = combos.shape
    rank = np.zeros(m, dtype=np.int64)
    prev = np.full(m, -1, dtype=np.int64)
    for i in range(k):
        start = prev + 1
        remaining = k - i
        rank += C[n - start, remaining] - C[n - combos[:, i], remaining]
        prev = combos[:, i]
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

    def _indexer(self):
        """Deck-derived tables for the vectorised row index, built once.

        All three depend only on ``card_to_idx`` / ``n_cards``, so they are
        hoisted out of the per-lookup path and memoised on the instance.
        Rebuilt after unpickling (``__getstate__`` drops them), like ``_mm``.
        """
        idx = getattr(self, "_idx_cache", None)
        if idx is None:
            keys = np.fromiter(self._card_to_idx.keys(), dtype=np.int64)
            vals = np.fromiter(self._card_to_idx.values(), dtype=np.int64)
            order = np.argsort(keys)
            idx = (keys[order], vals[order], _comb_table(self._n_cards, 5))
            self._idx_cache = idx
        return idx

    def _to_deck_idx(self, cards: np.ndarray) -> np.ndarray:
        """Map eval-card integers to 0-based deck indices (vectorised)."""
        keys_sorted, vals_sorted, _ = self._indexer()
        return vals_sorted[np.searchsorted(keys_sorted, cards)]

    def clusters_for_board(
        self,
        combo_cards: np.ndarray,
        board: "np.ndarray",
        valid_mask: "Optional[np.ndarray]" = None,
    ) -> np.ndarray:
        """Cluster id for **every** combo against one fixed ``board``.

        The batched counterpart of ``__getitem__``: one call replaces
        ``n_combos`` scalar lookups when the board is fixed — which it is for a
        whole CFR iteration, so the vector search regime hoists this per
        ``(street, sampled completion)`` instead of paying it per node visit.

        Bit-exact with the scalar path (same combinadic identity, integer-only),
        but the per-combo Python work — two ``sorted`` calls, a tuple build and
        two ``lex_rank`` loops over ``math.comb`` — collapses into a handful of
        numpy ops plus one gather out of the ``uint16`` memmap.

        Parameters
        ----------
        combo_cards : numpy.ndarray
            ``(n_combos, 2)`` hole-card pairs (``PokerEnv.combo_cards``).
        board : numpy.ndarray
            The public cards for this street, as eval-card integers.  Its
            length selects the street's row layout, so it must match the street
            this lookup was built for.
        valid_mask : numpy.ndarray, optional
            ``(n_combos,)`` bool; ``False`` marks combos that cannot be held
            (they share a card with ``board``).  Those rows have **no** entry on
            disk — the scalar path raises ``KeyError`` — so they are excluded
            from the gather and receive ``-1``.  When omitted the conflict mask
            is derived here.

        Returns
        -------
        numpy.ndarray
            ``(n_combos,)`` int64 cluster ids, ``-1`` on board-conflicting combos.
        """
        self._load()
        _, _, C = self._indexer()
        cc = np.asarray(combo_cards, dtype=np.int64)
        board = np.asarray(board, dtype=np.int64)
        n_combos = cc.shape[0]

        if valid_mask is None:
            valid_mask = ~(np.isin(cc[:, 0], board) | np.isin(cc[:, 1], board))
        valid_mask = np.asarray(valid_mask, dtype=bool)

        out = np.full(n_combos, -1, dtype=np.int64)
        live = cc[valid_mask]
        if live.shape[0] == 0:
            return out

        n = self._n_cards
        k = board.shape[0]
        # Hole pair -> deck indices, ascending (combo_cards rows are already
        # sorted by card int, but deck index order is what lex_rank needs).
        h = np.sort(self._to_deck_idx(live), axis=1)
        hole_rank = _lex_rank_vec(h, n, C)
        h0, h1 = h[:, 0:1], h[:, 1:2]

        # Public cards re-indexed into the deck with the hole pair removed —
        # the hole-dependent layout the on-disk rows use (see _get_row_index).
        p = np.sort(self._to_deck_idx(board[None, :]).reshape(1, k), axis=1)
        pr = p - (h0 < p).astype(np.int64) - (h1 < p).astype(np.int64)
        public_rank = _lex_rank_vec(pr, n - 2, C)

        rows = hole_rank * C[n - 2, k] + public_rank
        out[valid_mask] = self._mm[rows].astype(np.int64)
        return out

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
        state.pop("_idx_cache", None)  # deck-derived; rebuilt lazily (see _indexer)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# ---------------------------------------------------------------------------
# Batched street lookup — works on any LUT layout
# ---------------------------------------------------------------------------

def clusters_for_board(
    lut_entry: Any,
    combo_cards: np.ndarray,
    board: "np.ndarray",
    valid_mask: "Optional[np.ndarray]" = None,
) -> np.ndarray:
    """Cluster id for every combo against one fixed ``board``, for any LUT entry.

    Dispatches on the street's storage: :class:`MemmapLookup` takes the
    vectorised combinadic path; a plain ``dict`` street (pre-flop, and any LUT
    built small enough to stay a dict) falls back to a scalar sweep.  Callers
    therefore need not know how a given LUT was built — which matters because
    the shipped 20-card test LUT and a production 52-card LUT differ in both
    layout and bucket count.

    **No cluster/bucket count is consulted anywhere.**  Cluster ids are returned
    as the LUT produced them; sizing a table over them is the caller's job (the
    search regime remaps to a dense local index over the ids actually reachable
    in its subgame, so it works identically on a 50- or 200-bucket LUT).

    Parameters
    ----------
    lut_entry : MemmapLookup or dict
        One street of a loaded :data:`InfoSetLut` (e.g. ``lut["turn"]``).
    combo_cards : numpy.ndarray
        ``(n_combos, 2)`` hole-card pairs (``PokerEnv.combo_cards``).
    board : numpy.ndarray
        Public cards for this street as eval-card integers; length must match
        the street ``lut_entry`` was built for.
    valid_mask : numpy.ndarray, optional
        ``(n_combos,)`` bool; ``False`` marks board-conflicting combos, which
        have no entry on disk and receive ``-1``.  Derived here when omitted.

    Returns
    -------
    numpy.ndarray
        ``(n_combos,)`` int64 cluster ids, ``-1`` on board-conflicting combos.
    """
    cc = np.asarray(combo_cards, dtype=np.int64)
    board = np.asarray(board, dtype=np.int64)
    if valid_mask is None:
        valid_mask = ~(np.isin(cc[:, 0], board) | np.isin(cc[:, 1], board))
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if isinstance(lut_entry, MemmapLookup):
        return lut_entry.clusters_for_board(cc, board, valid_mask)

    # Plain-dict street: the key is sorted(hole) + sorted(board), matching
    # PokerEnv.cluster_for.  Cheap — a dict street is small by construction.
    key_board = sorted(int(c) for c in board)
    out = np.full(cc.shape[0], -1, dtype=np.int64)
    for i in np.flatnonzero(valid_mask):
        hole = sorted((int(cc[i, 0]), int(cc[i, 1])))
        out[i] = int(lut_entry[tuple(hole + key_board)])
    return out


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
