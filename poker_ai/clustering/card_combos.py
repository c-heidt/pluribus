import logging
import multiprocessing as mp
try:
    from math import comb
except ImportError:
    from scipy.special import comb as _comb
    def comb(n, k):
        return int(_comb(n, k, exact=True))
from typing import Dict, List, Optional, Tuple
from itertools import combinations
import operator

import numpy as np
from tqdm import tqdm

from poker_ai.poker.card import Card
from poker_ai.poker.deck import get_all_suits


log = logging.getLogger("poker_ai.clustering.runner")


def _lex_rank(combo: Tuple[int, ...], n: int) -> int:
    """
    Compute the lexicographic rank of a combination in O(k) time.
    
    Given a k-combination (c_0, c_1, ..., c_{k-1}) with c_0 < c_1 < ... < c_{k-1}
    chosen from {0, 1, ..., n-1}, compute its position in lexicographic order
    among all C(n, k) combinations.
    
    Uses the identity: sum_{j=a}^{b-1} C(n-j-1, r) = C(n-a, r+1) - C(n-b, r+1)
    to avoid inner loops.
    
    Parameters
    ----------
    combo : Tuple[int, ...]
        Combination of indices in ascending order.
    n : int
        Size of the universe {0, ..., n-1}.
        
    Returns
    -------
    int
        Lexicographic rank (0-based).
    """
    k = len(combo)
    rank = 0
    prev = -1
    for i in range(k):
        start = prev + 1
        remaining_positions = k - i
        rank += comb(n - start, remaining_positions) - comb(n - combo[i], remaining_positions)
        prev = combo[i]
    return rank


def _process_hole_combo_batch(args):
    """
    Worker function for parallel combo generation.
    
    Processes batches of HOLE COMBOS to maintain ordering
    for direct combinadic indexing.
    
    Parameters
    ----------
    args : tuple
        (batch_holes, sorted_publics, num_hole, num_public)
        
    Returns
    -------
    np.ndarray
        Combined hole + public card combos for this batch.
    """
    batch_holes, sorted_publics, num_hole, num_public = args
    batch_result = []
    
    # Iterate over hole combos in this batch
    for hole_combo in batch_holes:
        # Vectorized overlap check against all public combos
        overlap_per_public = np.any(np.isin(sorted_publics, hole_combo), axis=1)
        valid_publics = sorted_publics[~overlap_per_public]
        
        n_valid = len(valid_publics)
        if n_valid > 0:
            hole_repeated = np.tile(hole_combo, (n_valid, 1))
            combined = np.concatenate([hole_repeated, valid_publics], axis=1)
            batch_result.append(combined)
    
    if batch_result:
        return np.vstack(batch_result).astype(np.int32)
    else:
        return np.empty((0, num_hole + num_public), dtype=np.int32)


class CardCombos:
    """
    This class stores combinations of cards (histories) per street.
    
    Uses integer card representation (eval_card values) for memory efficiency.
    The integer representation reduces memory usage by ~18x compared to Card objects.
    
    Attributes
    ----------
    _cards : np.ndarray
        Array of Card objects (kept for preflop bucket calculation).
    _card_ints : np.ndarray
        Array of integer card values (eval_card). Used for all combo arrays.
    _int_to_card : Dict[int, Card]
        Mapping from integer to Card object for final lookup table creation.
    starting_hands : np.ndarray
        2D array of starting hand combos as integers, shape (n_hands, 2).
    flop : np.ndarray
        2D array of flop combos as integers, shape (n_combos, 5).
    turn : np.ndarray
        2D array of turn combos as integers, shape (n_combos, 6).
    river : np.ndarray
        2D array of river combos as integers, shape (n_combos, 7).
    """

    def __init__(
        self,
        low_card_rank: int,
        high_card_rank: int,
        parallel: bool = True,
        n_workers: Optional[int] = None,
    ):
        """
        Initialize CardCombos.
        
        Parameters
        ----------
        low_card_rank : int
            Lowest card rank (2-14).
        high_card_rank : int
            Highest card rank (2-14).
        parallel : bool
            Whether to use parallel processing for large combo generation.
            Recommended for decks with >15 cards. Default False.
        n_workers : Optional[int]
            Number of worker processes for parallel processing.
            If None, uses cpu_count(). Only used if parallel=True.
        """
        super().__init__()
        self.parallel = parallel
        self.n_workers = n_workers or mp.cpu_count()
        
        # Sort for caching.
        suits: List[str] = sorted(list(get_all_suits()))
        ranks: List[int] = sorted(list(range(low_card_rank, high_card_rank + 1)))
        
        # Create Card objects (needed for preflop bucket calculation)
        self._cards = np.array(
            [Card(rank, suit) for suit in suits for rank in ranks]
        )
        
        # Create integer representation (eval_card values) - 18x more memory efficient
        # SORTED ASCENDING: required for deterministic combinadic indexing (O(1) lookup)
        self._card_ints = np.sort(np.array(
            [c.eval_card for c in self._cards], dtype=np.int32
        ))
        
        # Card-to-index mapping for O(1) combinadic rank computation
        self._card_to_idx: Dict[int, int] = {
            int(c): i for i, c in enumerate(self._card_ints)
        }
        self._n_cards: int = len(self._card_ints)
        
        # Mapping from integer back to Card object
        self._int_to_card: Dict[int, Card] = {
            c.eval_card: c for c in self._cards
        }
        
        # Generate only starting_hands upfront (small, needed for preflop
        # and as the hole-card input when building larger combo arrays).
        # river, turn, and flop are generated on demand in compute() so that
        # each large array can be freed immediately after its street is done.
        log.info(f"Generating starting hands for {len(self._cards)} cards...")
        self.starting_hands = self._get_int_combos(2)
        log.info(f"Starting hands: {len(self.starting_hands):,}")

        # Deferred: built lazily on first access via properties.
        self._flop: Optional[np.ndarray] = None
        self._turn: Optional[np.ndarray] = None
        self._river: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lazy combo properties — build on first access, free by setting None
    # ------------------------------------------------------------------

    @property
    def river(self) -> np.ndarray:
        if self._river is None:
            self._river = self.build_street_combos("river")
        return self._river

    @river.setter
    def river(self, value: Optional[np.ndarray]):
        self._river = value

    @property
    def turn(self) -> np.ndarray:
        if self._turn is None:
            self._turn = self.build_street_combos("turn")
        return self._turn

    @turn.setter
    def turn(self, value: Optional[np.ndarray]):
        self._turn = value

    @property
    def flop(self) -> np.ndarray:
        if self._flop is None:
            self._flop = self.build_street_combos("flop")
        return self._flop

    @flop.setter
    def flop(self, value: Optional[np.ndarray]):
        self._flop = value

    def build_street_combos(self, street: str) -> np.ndarray:
        """
        Generate (or return cached) combo array for *street* on demand.

        Parameters
        ----------
        street : str
            One of ``"flop"``, ``"turn"``, or ``"river"``.

        Returns
        -------
        np.ndarray
            2D array of (hole1, hole2, board...) integer combos.
        """
        n_public = {"flop": 3, "turn": 4, "river": 5}[street]
        combos = self._create_int_info_combos(
            self.starting_hands, self._get_int_combos(n_public), street
        )
        log.info(f"Built {street}: {len(combos):,} combos")
        return combos

    def _get_int_combos(self, num_cards: int) -> np.ndarray:
        """
        Get card combinations as integer arrays.

        Parameters
        ----------
        num_cards : int
            Number of cards per combination.

        Returns
        -------
        np.ndarray
            2D array of shape (n_combos, num_cards) with integer card values.
        """
        combos = list(combinations(self._card_ints, num_cards))
        return np.array(combos, dtype=np.int32)

    def _create_int_info_combos(
        self,
        start_combos: np.ndarray,
        publics: np.ndarray,
        betting_stage: str = "unknown",
    ) -> np.ndarray:
        """
        Combinations of private info (hole cards) and public info (board).
        
        Uses integer card representation for memory efficiency.
        Cards are sorted ascending for deterministic combinadic indexing.
        
        OPTIMIZED: Uses vectorized operations over NumPy arrays for ~10-100x speedup.

        Parameters
        ----------
        start_combos : np.ndarray
            Starting combinations (hole cards) as integers, shape (n, 2).
        publics : np.ndarray
            Public card combinations as integers, shape (m, num_public).
        betting_stage : str
            Name of the betting stage for progress logging.
            
        Returns
        -------
        np.ndarray
            2D array of combined hole + board cards as integers.
        """
        num_hole = start_combos.shape[1]
        num_public = publics.shape[1]
        total_cards = num_hole + num_public
        
        # Sort ascending for deterministic combinadic indexing (O(1) lookup)
        sorted_holes = np.sort(start_combos, axis=1)
        sorted_publics = np.sort(publics, axis=1)
        
        # Choose between parallel and sequential processing
        n_holes = len(sorted_holes)
        
        # Use parallel processing for large datasets (threshold: 200+ starting hands)
        if self.parallel and n_holes >= 200:
            return self._create_int_info_combos_parallel(
                sorted_holes, sorted_publics, num_hole, num_public, betting_stage
            )
        else:
            return self._create_int_info_combos_sequential(
                sorted_holes, sorted_publics, num_hole, num_public, betting_stage
            )
    
    def _create_int_info_combos_sequential(
        self,
        sorted_holes: np.ndarray,
        sorted_publics: np.ndarray,
        num_hole: int,
        num_public: int,
        betting_stage: str,
    ) -> np.ndarray:
        """
        Sequential vectorized combo generation.
        
        CRITICAL: Maintains original ordering (iterate hole combos first)
        to preserve binary search compatibility.
        """
        result = []
        
        # IMPORTANT: Iterate over HOLE COMBOS first (outer loop) to maintain order
        # Then vectorize the check against ALL public combos (inner operation)
        for hole_combo in tqdm(
            sorted_holes,
            dynamic_ncols=True,
            desc=f"Creating {betting_stage} info combos",
        ):
            # Vectorized overlap check: does this hole overlap with ANY card in each public combo?
            # For each public combo, check if any of its cards appear in hole_combo
            # Shape: (n_publics, num_public) -> check each public combo
            overlap_per_public = np.any(np.isin(sorted_publics, hole_combo), axis=1)
            
            # Get valid public combos (no overlap with this hole)
            valid_publics = sorted_publics[~overlap_per_public]
            
            n_valid = len(valid_publics)
            if n_valid > 0:
                # Replicate this hole combo for all valid public combos
                hole_repeated = np.tile(hole_combo, (n_valid, 1))
                
                # Combine: hole cards first, then public cards
                combined = np.concatenate([hole_repeated, valid_publics], axis=1)
                result.append(combined)
        
        if result:
            return np.vstack(result).astype(np.int32)
        else:
            return np.empty((0, num_hole + num_public), dtype=np.int32)
    
    def _create_int_info_combos_parallel(
        self,
        sorted_holes: np.ndarray,
        sorted_publics: np.ndarray,
        num_hole: int,
        num_public: int,
        betting_stage: str,
    ) -> np.ndarray:
        """
        Parallel combo generation for large datasets.
        
        CRITICAL: Batches HOLE COMBOS (not public combos) to maintain
        ordering for binary search compatibility.
        """
        n_holes = len(sorted_holes)
        
        # Split hole combos into batches for parallel processing
        # Each batch maintains sequential order of hole combos
        batch_size = max(10, n_holes // (self.n_workers * 4))
        batches = []
        for i in range(0, n_holes, batch_size):
            batch = sorted_holes[i:i+batch_size]
            batches.append((batch, sorted_publics, num_hole, num_public))
        
        log.info(
            f"Creating {betting_stage} info combos in parallel: "
            f"{len(batches)} batches, {self.n_workers} workers"
        )
        
        # Process batches in parallel
        # Results are combined in order, preserving the hole combo sequence
        with mp.Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_hole_combo_batch, batches),
                total=len(batches),
                dynamic_ncols=True,
                desc=f"Creating {betting_stage} info combos (parallel)",
            ))
        
        # Combine results IN ORDER (each batch maintains sequential hole combo order)
        valid_results = [r for r in results if len(r) > 0]
        if valid_results:
            return np.vstack(valid_results).astype(np.int32)
        else:
            return np.empty((0, num_hole + num_public), dtype=np.int32)

    # Legacy methods for backward compatibility
    def get_card_combos(self, num_cards: int) -> np.ndarray:
        """
        Get the card combinations for a given street.
        
        DEPRECATED: Use _get_int_combos for new code.
        Kept for backward compatibility with preflop calculation.

        Parameters
        ----------
        num_cards : int
            Number of cards you want returned

        Returns
        -------
            Combos of cards (Card) -> np.ndarray
        """
        return np.array([c for c in combinations(self._cards, num_cards)])

    def create_info_combos(
        self, start_combos: np.ndarray, publics: np.ndarray
    ) -> np.ndarray:
        """
        DEPRECATED: Use _create_int_info_combos for new code.
        Kept for backward compatibility.
        """
        if publics.shape[1] == 3:
            betting_stage = "flop"
        elif publics.shape[1] == 4:
            betting_stage = "turn"
        elif publics.shape[1] == 5:
            betting_stage = "river"
        else:
            betting_stage = "unknown"
        our_cards: List[int] = []
        for combos in tqdm(
            start_combos,
            dynamic_ncols=True,
            desc=f"Creating {betting_stage} info combos",
        ):
            # Descending sort combos.
            sorted_combos: List[Card] = sorted(
                list(combos),
                key=operator.attrgetter("eval_card"),
                reverse=True,
            )
            for public_combo in publics:
                # Descending sort public_combo.
                sorted_public_combo: List[Card] = sorted(
                    list(public_combo),
                    key=operator.attrgetter("eval_card"),
                    reverse=True,
                )
                if not np.any(np.isin(sorted_combos, sorted_public_combo)):
                    # Combine hand and public cards.
                    hand: np.array = np.array(
                        sorted_combos + sorted_public_combo
                    )
                    our_cards.append(hand)
        return np.array(our_cards)

    def int_to_cards(self, int_combo: np.ndarray) -> Tuple[Card, ...]:
        """
        Convert an integer combo array back to Card objects.
        
        Parameters
        ----------
        int_combo : np.ndarray
            Array of integer card values.
            
        Returns
        -------
        Tuple[Card, ...]
            Tuple of Card objects.
        """
        return tuple(self._int_to_card[int(c)] for c in int_combo)

    def get_row_index(self, hole_ints, public_ints) -> int:
        """
        Compute the exact row index for a card combo in O(1) time.
        
        Uses the combinatorial number system (combinadic) to compute the
        position of a combo in the array, eliminating any need for search.
        
        The combo array is structured as:
            for each hole_combo (in lex order):
                for each valid_public_combo (in lex order of remaining cards):
                    row_index += 1
        
        Since every hole combo has exactly C(N-2, k_public) valid public
        combos, the row index is:
            row = hole_rank * C(N-2, k_public) + public_rank_in_remaining
            
        Complexity: O(k) where k = number of public cards (≤5), effectively O(1).
        Memory: Zero extra memory.
        
        Parameters
        ----------
        hole_ints : array-like
            Two hole card eval_card integers.
        public_ints : array-like
            Public card eval_card integers (3 for flop, 4 for turn, 5 for river).
            
        Returns
        -------
        int
            Row index in the combo/data array.
        """
        card_to_idx = self._card_to_idx
        n = self._n_cards
        
        # Convert eval_card ints to sorted dense indices
        h_idx = sorted(card_to_idx[int(c)] for c in hole_ints)
        p_idx = sorted(card_to_idx[int(c)] for c in public_ints)
        
        # Hole rank in C(N, 2) combinations
        hole_rank = _lex_rank(tuple(h_idx), n)
        
        # Re-index public cards into {0..N-3} by excluding hole card indices
        h0, h1 = h_idx[0], h_idx[1]
        p_reindexed = tuple(
            p - (h0 < p) - (h1 < p) for p in p_idx
        )
        
        # Public rank among C(N-2, k_public) combinations
        n_remaining = n - 2
        k_public = len(p_idx)
        public_rank = _lex_rank(p_reindexed, n_remaining)
        
        return hole_rank * comb(n_remaining, k_public) + public_rank
