"""Card-combination generator for the build pipeline.

Produces the per-street (hole, board) combo arrays consumed by the feature
extractors and the chunked dispatcher.  Combos are laid out in the exact
lexicographic order required by the O(1) combinadic index, so downstream
code can look up any combo's row with
:func:`information_abstraction.lookup.lex_rank`.
"""
import logging
import multiprocessing as mp
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from environment.utils import make_deck_arr
from information_abstraction.lookup import comb, lex_rank

log = logging.getLogger(
    "information_abstraction.build.card_combos",
)


_PUBLIC_CARDS_PER_STREET: Dict[str, int] = {
    "flop": 3, "turn": 4, "river": 5,
}


def _combine_hole_with_publics(
    hole_combo: np.ndarray,
    sorted_publics: np.ndarray,
) -> np.ndarray:
    """Tile one hole combo against every non-overlapping public combo.

    Shared inner kernel for both the sequential
    (:meth:`CardCombos._run_sequential`) and parallel
    (:meth:`CardCombos._run_parallel`) paths.

    Parameters
    ----------
    hole_combo : np.ndarray
        One hole pair, shape ``(2,)``.
    sorted_publics : np.ndarray
        All public combos of a given street, shape
        ``(n_publics, k_public)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_valid, 2 + k_public)`` — the Cartesian join minus
        any public combo that collides with ``hole_combo``.  Empty when
        every public combo collides.
    """
    overlap = np.any(np.isin(sorted_publics, hole_combo), axis=1)
    valid_publics = sorted_publics[~overlap]
    if valid_publics.size == 0:
        return np.empty((0, hole_combo.size + sorted_publics.shape[1]),
                        dtype=np.int32)
    hole_repeated = np.tile(hole_combo, (len(valid_publics), 1))
    return np.concatenate([hole_repeated, valid_publics], axis=1)


def _process_hole_combo_batch(args):
    """Worker entry point for parallel combo generation.

    Kept at module scope so ``multiprocessing.Pool`` can pickle it.

    Parameters
    ----------
    args : tuple
        ``(batch_holes, sorted_publics, num_hole, num_public)``.  The
        tuple shape exists because ``Pool.imap`` passes a single
        argument per call.

    Returns
    -------
    np.ndarray
        Stacked output of :func:`_combine_hole_with_publics` across the
        batch, shape ``(n_valid, num_hole + num_public)``.  Empty when
        every hole in the batch collides with every public combo.
    """
    batch_holes, sorted_publics, num_hole, num_public = args
    pieces = []
    for hole_combo in batch_holes:
        piece = _combine_hole_with_publics(hole_combo, sorted_publics)
        if piece.size:
            pieces.append(piece)
    if not pieces:
        return np.empty((0, num_hole + num_public), dtype=np.int32)
    return np.vstack(pieces).astype(np.int32)


class CardCombos:
    """Per-street card-combination arrays.

    Deterministic ascending order matches the combinadic index used by
    :class:`information_abstraction.lookup.MemmapLookup`.

    Attributes
    ----------
    _card_ints : np.ndarray
        Deck as ascending ``int32`` eval-card values.
    starting_hands : np.ndarray
        Shape ``(n_hole_pairs, 2)``.
    flop, turn, river : np.ndarray
        Lazy per-street arrays of shape ``(n_combos, 2 + public_cards)``.
        Each can be freed with ``self.river = None`` to reclaim memory
        between streets.
    """

    def __init__(
        self,
        low_card_rank: int,
        high_card_rank: int,
        parallel: bool = True,
        n_workers: Optional[int] = None,
    ):
        super().__init__()
        self.parallel = parallel
        self.n_workers = n_workers or mp.cpu_count()

        self._card_ints = np.sort(make_deck_arr(low_card_rank, high_card_rank))
        self._card_to_idx: Dict[int, int] = {
            int(c): i for i, c in enumerate(self._card_ints)
        }
        self._n_cards: int = len(self._card_ints)

        log.info(f"Generating starting hands for {self._n_cards} cards...")
        self.starting_hands = self._get_int_combos(2)
        log.info(f"Starting hands: {len(self.starting_hands):,}")

        # Lazy per-street storage — filled on first property access,
        # freed by caller via ``self.river = None`` once a street is done.
        self._flop: Optional[np.ndarray] = None
        self._turn: Optional[np.ndarray] = None
        self._river: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lazy per-street combo arrays
    # ------------------------------------------------------------------

    @property
    def river(self) -> np.ndarray:
        if self._river is None:
            self._river = self._build_street("river")
        return self._river

    @river.setter
    def river(self, value: Optional[np.ndarray]):
        self._river = value

    @property
    def turn(self) -> np.ndarray:
        if self._turn is None:
            self._turn = self._build_street("turn")
        return self._turn

    @turn.setter
    def turn(self, value: Optional[np.ndarray]):
        self._turn = value

    @property
    def flop(self) -> np.ndarray:
        if self._flop is None:
            self._flop = self._build_street("flop")
        return self._flop

    @flop.setter
    def flop(self, value: Optional[np.ndarray]):
        self._flop = value

    def _build_street(self, street: str) -> np.ndarray:
        """Materialise the combo array for ``street`` on demand.

        Called by the ``river`` / ``turn`` / ``flop`` properties the first
        time they are accessed.  The number of public cards is looked up
        from ``_PUBLIC_CARDS_PER_STREET`` so adding a new street only
        requires extending that table.

        Parameters
        ----------
        street : str
            One of ``"flop"``, ``"turn"``, ``"river"``.

        Returns
        -------
        np.ndarray
            Shape ``(n_combos, 2 + k_public)``.
        """
        n_public = _PUBLIC_CARDS_PER_STREET[street]
        combos = self._create_int_info_combos(
            self.starting_hands, self._get_int_combos(n_public), street,
        )
        log.info(f"Built {street}: {len(combos):,} combos")
        return combos

    # ------------------------------------------------------------------
    # Combo generation
    # ------------------------------------------------------------------

    def _get_int_combos(self, num_cards: int) -> np.ndarray:
        """Enumerate all ``num_cards``-sized combinations of the deck.

        Parameters
        ----------
        num_cards : int
            Size of each combination.

        Returns
        -------
        np.ndarray
            Shape ``(C(n_cards, num_cards), num_cards)``, ``int32``.
        """
        combos = list(combinations(self._card_ints, num_cards))
        return np.array(combos, dtype=np.int32)

    def _create_int_info_combos(
        self,
        start_combos: np.ndarray,
        publics: np.ndarray,
        betting_stage: str = "unknown",
    ) -> np.ndarray:
        """Join every hole combo with every non-overlapping public combo.

        Hole combos form the outer loop so the result matches the
        combinadic row index; public order within each hole matches the
        lexicographic order of ``publics``.

        Parameters
        ----------
        start_combos : np.ndarray
            Hole pairs, shape ``(n_holes, 2)``.
        publics : np.ndarray
            Public combos, shape ``(n_publics, k_public)``.
        betting_stage : str, optional
            Used only for the ``tqdm`` progress-bar label.

        Returns
        -------
        np.ndarray
            Stacked ``(hole, public)`` rows, shape
            ``(n_valid, 2 + k_public)``.
        """
        num_hole = start_combos.shape[1]
        num_public = publics.shape[1]
        sorted_holes = np.sort(start_combos, axis=1)
        sorted_publics = np.sort(publics, axis=1)

        use_parallel = self.parallel and len(sorted_holes) >= 200
        if use_parallel:
            pieces = self._run_parallel(
                sorted_holes, sorted_publics, num_hole, num_public,
                betting_stage,
            )
        else:
            pieces = self._run_sequential(
                sorted_holes, sorted_publics, betting_stage,
            )

        valid = [p for p in pieces if p.size]
        if not valid:
            return np.empty((0, num_hole + num_public), dtype=np.int32)
        return np.vstack(valid).astype(np.int32)

    def _run_sequential(
        self,
        sorted_holes: np.ndarray,
        sorted_publics: np.ndarray,
        betting_stage: str,
    ) -> List[np.ndarray]:
        """In-process combo generation for small decks.

        Parameters
        ----------
        sorted_holes : np.ndarray
            Pre-sorted hole pairs.
        sorted_publics : np.ndarray
            Pre-sorted public combos.
        betting_stage : str
            Progress-bar label.

        Returns
        -------
        List[np.ndarray]
            One array per hole pair — concatenated by the caller.
        """
        pieces: List[np.ndarray] = []
        for hole_combo in tqdm(
            sorted_holes,
            dynamic_ncols=True,
            desc=f"Creating {betting_stage} info combos",
        ):
            pieces.append(
                _combine_hole_with_publics(hole_combo, sorted_publics)
            )
        return pieces

    def _run_parallel(
        self,
        sorted_holes: np.ndarray,
        sorted_publics: np.ndarray,
        num_hole: int,
        num_public: int,
        betting_stage: str,
    ) -> List[np.ndarray]:
        """Multiprocess combo generation for large decks.

        Hole pairs are split into batches of roughly
        ``n_holes // (n_workers * 4)`` each so every worker sees several
        batches — ``imap`` then interleaves completion order with
        dispatch and keeps the pool fully loaded.

        Parameters
        ----------
        sorted_holes : np.ndarray
            Pre-sorted hole pairs.
        sorted_publics : np.ndarray
            Pre-sorted public combos.
        num_hole : int
            Width of a hole combo (always ``2``, propagated for
            empty-batch shape consistency).
        num_public : int
            Width of a public combo.
        betting_stage : str
            Progress-bar label.

        Returns
        -------
        List[np.ndarray]
            One array per batch — concatenated by the caller.
        """
        n_holes = len(sorted_holes)
        batch_size = max(10, n_holes // (self.n_workers * 4))
        batches = [
            (sorted_holes[i:i + batch_size],
             sorted_publics, num_hole, num_public)
            for i in range(0, n_holes, batch_size)
        ]
        log.info(
            f"Creating {betting_stage} info combos in parallel: "
            f"{len(batches)} batches, {self.n_workers} workers",
        )
        with mp.Pool(processes=self.n_workers) as pool:
            return list(tqdm(
                pool.imap(_process_hole_combo_batch, batches),
                total=len(batches),
                dynamic_ncols=True,
                desc=f"Creating {betting_stage} info combos (parallel)",
            ))

    # ------------------------------------------------------------------
    # O(1) row index — mirrors MemmapLookup._get_row_index
    # ------------------------------------------------------------------

    def get_row_index(self, hole_ints, public_ints) -> int:
        """Row at which this combo's features live in the merged memmap.

        Duplicated verbatim on
        :meth:`information_abstraction.lookup.MemmapLookup._get_row_index`
        so runtime consumers do not need to import the build subpackage.
        Any change to the layout must be applied in both places.

        Parameters
        ----------
        hole_ints : Sequence[int]
            Two hole cards as eval-card integers, in any order.
        public_ints : Sequence[int]
            Public cards as eval-card integers, in any order.

        Returns
        -------
        int
            Row index into the street's ``merged_data.dat`` /
            ``cluster_ids.dat``.
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
