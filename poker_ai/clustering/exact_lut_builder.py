"""
Exact hand strength computation using exhaustive enumeration.

This module provides utilities for computing exact hand strength (EHS) by
enumerating all possible opponent hands and future cards, rather than using
Monte Carlo sampling. The computation works backwards from the river:

1. River: Compare hand against ALL possible opponent pairs
2. Turn: Enumerate ALL river cards, look up pre-computed river EHS from chunks
3. Flop: Enumerate ALL turn cards, look up pre-computed turn distributions from chunks

Inherits from CardInfoLutBuilder and overrides the EHS computation methods.
Optimized to reuse computed results: turn uses river EHS data, flop uses turn data.

Lookups use O(1) combinadic indexing (no binary search) via get_row_index().
"""
import logging
from itertools import combinations
try:
    from math import comb
except ImportError:
    from scipy.special import comb as _comb
    def comb(n, k):
        return int(_comb(n, k, exact=True))
from typing import Dict, Optional, Tuple

import numpy as np
import os
import threading

from poker_ai.clustering.card_info_lut_builder import CardInfoLutBuilder

log = logging.getLogger("poker_ai.clustering.exact_hand_strength")


# ============================================================================
# Module-level cache for multiprocessing support
# ============================================================================
# When using ProcessPoolExecutor, instance variables don't serialize to workers.
# Each worker process maintains its own module-level cache (lazy loaded on first use).
# With O(1) combinadic indexing, workers only need the memmap and centroids —
# no combos array or hole index needed.

_PROCESS_CACHE = {
    "river_memmap": None,
    "turn_memmap": None,
    "river_centroids": None,
    "turn_centroids": None,
    "save_dir": None,
}

_CACHE_LOAD_LOCK = threading.Lock()


def _get_process_cache(street: str, save_dir: str, n_rows: int, feature_dim: int):
    """
    Get or load lookup data for a street in the current process.
    
    With O(1) combinadic indexing, workers no longer need to load the combos
    array or build a hole card index. Only the memmap data and centroids
    are needed, saving significant memory and startup time.
    
    Parameters
    ----------
    street : str
        The street name ("river" or "turn").
    save_dir : str
        Path to the save directory.
    n_rows : int
        Total number of combos (rows in the memmap). Computed from the
        combinatorial formula: C(N,2) * C(N-2, k_public).
    feature_dim : int
        Number of features per row (3 for river EHS, n_clusters for distributions).
    """
    from pathlib import Path
    
    if _PROCESS_CACHE["save_dir"] != save_dir:
        _PROCESS_CACHE["save_dir"] = save_dir
        _PROCESS_CACHE["river_memmap"] = None
        _PROCESS_CACHE["turn_memmap"] = None
        _PROCESS_CACHE["river_centroids"] = None
        _PROCESS_CACHE["turn_centroids"] = None
    
    memmap_key = f"{street}_memmap"
    centroids_key = f"{street}_centroids"
    
    if _PROCESS_CACHE[memmap_key] is not None:
        return _PROCESS_CACHE[memmap_key], _PROCESS_CACHE[centroids_key]
    
    with _CACHE_LOAD_LOCK:
        if _PROCESS_CACHE[memmap_key] is not None:
            return _PROCESS_CACHE[memmap_key], _PROCESS_CACHE[centroids_key]
        
        log.debug(f"Worker {os.getpid()} loading {street} memmap + centroids")
        
        # Load merged_data.dat as memmap
        merged_path = Path(save_dir) / street / "merged_data.dat"
        if not merged_path.exists():
            log.warning(f"Merged data not found for {street} at {merged_path}")
            return None, None
        
        memmap = np.memmap(
            merged_path,
            dtype=np.float32,
            mode='r',
            shape=(n_rows, feature_dim),
        )
        _PROCESS_CACHE[memmap_key] = memmap
        
        # Load centroids
        centroids_path = Path(save_dir) / street / "centroids.npy"
        if not centroids_path.exists():
            log.warning(f"Centroids not found for {street}")
            return memmap, None
        
        centroids = np.load(centroids_path, allow_pickle=True)
        _PROCESS_CACHE[centroids_key] = centroids
        
        log.debug(
            f"Loaded {street}: memmap ({n_rows:,} x {feature_dim}), "
            f"{len(centroids)} centroids"
        )
        return memmap, centroids


def _clear_process_cache_for_street(street: str):
    """Clear cached data for a specific street to free memory."""
    _PROCESS_CACHE[f"{street}_memmap"] = None
    _PROCESS_CACHE[f"{street}_centroids"] = None
    log.debug(f"Cleared process cache for {street}")


class ExactHandStrengthBuilder(CardInfoLutBuilder):
    """
    Computes exact hand strength using exhaustive enumeration.
    
    Unlike the parent CardInfoLutBuilder which uses Monte Carlo methods,
    this class computes exact hand strength by:
    - Comparing against ALL possible opponent hands (river)
    - Enumerating ALL possible future cards (turn/flop)
    
    Uses O(1) combinadic indexing (via get_row_index) for lookups into
    pre-computed data — no binary search or combo arrays needed in workers.
    Each worker only loads the memmap data file and centroids.
    
    Inherits all chunked processing and clustering infrastructure from
    CardInfoLutBuilder.
    """

    def __init__(
        self,
        low_card_rank: int,
        high_card_rank: int,
        save_dir: str,
        workers: Optional[int] = None,
        chunk_size: int = 10000,
        use_mini_batch: bool = True,
        parallel_combos: bool = True,
    ):
        """
        Initialize the ExactHandStrengthBuilder.
        
        Parameters
        ----------
        low_card_rank : int
            Lowest card rank (2-14).
        high_card_rank : int
            Highest card rank (2-14).
        save_dir : str
            Directory to save results and checkpoints.
        workers : Optional[int]
            Number of worker processes. Defaults to CPU count.
        chunk_size : int
            Number of combinations per chunk. Default 10000.
        use_mini_batch : bool
            Whether to use MiniBatchKMeans for large datasets. Default True.
        parallel_combos : bool
            Whether to use parallel processing for combo generation.
            Recommended for decks with >15 cards. Default False.
        """
        # Initialize parent with dummy simulation counts (not used for exact)
        super().__init__(
            n_simulations_river=0,
            n_simulations_turn=0,
            n_simulations_flop=0,
            low_card_rank=low_card_rank,
            high_card_rank=high_card_rank,
            save_dir=save_dir,
            workers=workers,
            chunk_size=chunk_size,
            use_mini_batch=use_mini_batch,
            parallel_combos=parallel_combos,
        )
        
        # Override config to indicate exact computation method
        self._config["method"] = "exact"
        # Remove simulation counts from config (not applicable)
        self._config.pop("n_simulations_river", None)
        self._config.pop("n_simulations_turn", None)
        self._config.pop("n_simulations_flop", None)

    def compute(
        self,
        n_river_clusters: int,
        n_turn_clusters: int,
        n_flop_clusters: int,
    ):
        """
        Compute all clusters using exact hand strength.
        
        Overrides parent to log that exact computation is being used.
        """
        log.info("Starting computation of clusters using EXACT hand strength.")
        super().compute(n_river_clusters, n_turn_clusters, n_flop_clusters)

    def _on_street_clustering_complete(
        self,
        street: str,
        merged_data: np.ndarray,
        all_combos: np.ndarray,
    ):
        """
        Pre-populate the module-level process cache after clustering completes.
        
        With O(1) combinadic indexing, workers only need the memmap data
        and centroids — no combo arrays needed for lookups.
        """
        _PROCESS_CACHE["save_dir"] = self._config["save_dir"]
        _PROCESS_CACHE[f"{street}_memmap"] = merged_data
        _PROCESS_CACHE[f"{street}_centroids"] = self.centroids.get(street)

        log.info(
            f"Pre-populated process cache for {street} "
            f"(memmap rows: {len(merged_data):,}, "
            f"centroids: {len(self.centroids.get(street, []))})"
        )

    # =========================================================================
    # River: Override to use exact EHS computation
    # =========================================================================
    
    def compute_exact_river_ehs(self, our_hand: np.ndarray, board: np.ndarray) -> np.ndarray:
        """
        Compute exact expected hand strength on the river.
        
        Enumerates ALL possible opponent hole card pairs and computes exact
        win/loss/tie rates.
        
        Optimised with decomposed evaluation: a 7-card hand (5 board + 2 hole)
        has C(7,5)=21 five-card subsets, falling into three categories:
        
        - Category 0: 5 board cards alone (1 subset) — constant for all opponents
        - Category 1: 4 board + 1 hole card (5 subsets per card) — pre-computable
        - Category 2: 3 board + 2 hole cards (10 subsets per pair) — per-opponent
        
        Pre-computing categories 0 and 1 cuts per-opponent _five() calls from
        21 down to at most 10, and early termination often skips them entirely.
        
        Parameters
        ----------
        our_hand : np.ndarray
            Our two hole cards.
        board : np.ndarray
            The five community cards on the river.
            
        Returns
        -------
        np.ndarray
            [win_rate, loss_rate, tie_rate] - exact probabilities
        """
        # Cards not available for opponent
        unavailable_cards = set(our_hand.tolist() + board.tolist())
        available_cards = [c for c in self._card_ints if c not in unavailable_cards]
        
        # Pre-convert to plain int lists once (avoid repeated numpy conversions)
        board_ints = [int(c) for c in board]
        our_ints = [int(c) for c in our_hand]
        
        # Local reference to _five for speed (avoids attribute lookup in loop)
        _five = self._evaluator._five
        
        # Evaluate our 7-card hand once (bypass evaluate() overhead)
        our_rank = self._evaluator._seven(our_ints + board_ints)
        
        # === Decompose opponent evaluation ===
        # Category 0: the 5 board cards as a hand (same for all opponents)
        board_only_rank = _five(board_ints)
        
        # Pre-compute board subsets
        board_4_subsets = list(combinations(board_ints, 4))  # 5 subsets
        board_3_subsets = list(combinations(board_ints, 3))  # 10 subsets
        
        # Category 1: for each available card, best rank using board-only
        # or any (4-board + that card) subset
        single_best = {}
        for c in available_cards:
            ci = int(c)
            best = board_only_rank
            for b4 in board_4_subsets:
                score = _five(b4 + (ci,))
                if score < best:
                    best = score
            single_best[c] = best
        
        # Enumerate opponent pairs with decomposed evaluation
        wins = 0
        losses = 0
        ties = 0
        
        for o1, o2 in combinations(available_cards, 2):
            # Start with best from categories 0 & 1 (pre-computed)
            opp_rank = min(single_best[o1], single_best[o2])
            
            # Category 2: 3-board + both opponent cards (10 subsets).
            # Only needed if pair subsets could change the outcome.
            # Since _five can only return equal or lower (better) ranks,
            # skip when outcome is already determined.
            if opp_rank >= our_rank:
                o1i, o2i = int(o1), int(o2)
                pair = (o1i, o2i)
                for b3 in board_3_subsets:
                    score = _five(b3 + pair)
                    if score < opp_rank:
                        opp_rank = score
                        if opp_rank < our_rank:
                            break  # Outcome determined, stop checking
            
            if our_rank > opp_rank:
                wins += 1
            elif our_rank < opp_rank:
                losses += 1
            else:
                ties += 1
        
        total = wins + losses + ties
        if total == 0:
            return np.array([0.0, 0.0, 0.0])
        
        return np.array([wins / total, losses / total, ties / total])

    def process_river_ehs(self, public: np.ndarray) -> np.ndarray:
        """
        Override parent method to use exact EHS computation.
        
        Parameters
        ----------
        public : np.ndarray
            Cards array: [hole1, hole2, flop1, flop2, flop3, turn, river]
            
        Returns
        -------
        np.ndarray
            [win_rate, loss_rate, tie_rate]
        """
        our_hand = public[:2]
        board = public[2:7]
        return self.compute_exact_river_ehs(our_hand, board)

    # =========================================================================
    # Turn: Override to use exact EHS distribution computation
    # =========================================================================
    
    def compute_exact_turn_ehs_distribution(
        self,
        our_hand: np.ndarray,
        board: np.ndarray,
    ) -> np.ndarray:
        """
        Compute exact turn EHS distribution by enumerating all river cards.
        
        For each possible river card, looks up the pre-computed river EHS values
        using O(1) combinadic indexing via get_row_index().
        
        Parameters
        ----------
        our_hand : np.ndarray
            Our two hole cards.
        board : np.ndarray
            The four community cards on the turn.
            
        Returns
        -------
        np.ndarray
            Distribution over river clusters (normalized frequencies).
        """
        # Compute expected data dimensions for the memmap
        n = self._n_cards
        n_river_rows = comb(n, 2) * comb(n - 2, 5)
        
        # Get river lookup data from process cache
        river_memmap, river_centroids = _get_process_cache(
            "river", self._config["save_dir"],
            n_rows=n_river_rows,
            feature_dim=3,  # River EHS: [win_rate, loss_rate, tie_rate]
        )
        
        # Fallback to instance centroids if not in cache
        if river_centroids is None:
            if "river" not in self.centroids or self.centroids["river"] is None:
                self.centroids["river"] = self.chunked_processor.load_centroids("river")
            river_centroids = self.centroids["river"]
        
        # Cards not available for river
        unavailable_cards = set(our_hand.tolist() + board.tolist())
        available_cards = [c for c in self._card_ints if c not in unavailable_cards]
        
        n_river_centroids = len(river_centroids)
        distribution = np.zeros(n_river_centroids)
        
        if len(available_cards) == 0:
            return distribution
        
        # Enumerate all possible river cards
        for river_card in available_cards:
            # O(1) direct index computation — no search needed!
            river_board = np.append(board, river_card)
            row_idx = self.get_row_index(our_hand, river_board)
            
            ehs = None
            if river_memmap is not None:
                ehs = river_memmap[row_idx]
            
            if ehs is not None:
                min_idx = self._find_closest_centroid(ehs, river_centroids)
            else:
                # Fallback: compute exact river EHS
                ehs = self.compute_exact_river_ehs(our_hand, river_board)
                min_idx = self._find_closest_centroid(ehs, river_centroids)
            
            distribution[min_idx] += 1
        
        # Normalize
        total = distribution.sum()
        if total > 0:
            distribution /= total
        
        return distribution

    def process_turn_ehs_distributions(self, public: np.ndarray) -> np.ndarray:
        """
        Override parent method to use exact turn EHS distribution.
        
        Parameters
        ----------
        public : np.ndarray
            Cards array: [hole1, hole2, flop1, flop2, flop3, turn]
            
        Returns
        -------
        np.ndarray
            Distribution over river clusters.
        """
        our_hand = public[:2]
        board = public[2:6]
        return self.compute_exact_turn_ehs_distribution(our_hand, board)

    # =========================================================================
    # Flop: Override to use exact potential-aware distribution computation
    # =========================================================================
    
    def compute_exact_flop_distribution(
        self,
        our_hand: np.ndarray,
        board: np.ndarray,
    ) -> np.ndarray:
        """
        Compute exact flop potential-aware distribution.
        
        For each possible turn card, looks up the pre-computed turn EHS distribution
        using O(1) combinadic indexing via get_row_index().
        
        River cache is cleared on first call to free memory (flop doesn't need river data).
        
        Parameters
        ----------
        our_hand : np.ndarray
            Our two hole cards.
        board : np.ndarray
            The three community cards on the flop.
            
        Returns
        -------
        np.ndarray
            Distribution over turn clusters (normalized frequencies).
        """
        # Clear river cache on first flop call in this process to save memory
        if _PROCESS_CACHE["river_memmap"] is not None:
            _clear_process_cache_for_street("river")
        
        # Compute expected data dimensions for the memmap
        n = self._n_cards
        n_turn_rows = comb(n, 2) * comb(n - 2, 4)
        
        # We need to know feature_dim = n_turn_centroids, but we need centroids
        # first. Load centroids to determine feature_dim.
        if "turn" not in self.centroids or self.centroids["turn"] is None:
            self.centroids["turn"] = self.chunked_processor.load_centroids("turn")
        turn_centroids = self.centroids["turn"]
        n_turn_centroids = len(turn_centroids)
        
        # Get turn lookup data from process cache
        turn_memmap, cached_turn_centroids = _get_process_cache(
            "turn", self._config["save_dir"],
            n_rows=n_turn_rows,
            feature_dim=n_turn_centroids,
        )
        if cached_turn_centroids is not None:
            turn_centroids = cached_turn_centroids
        
        # Cards not available for turn
        unavailable_cards = set(our_hand.tolist() + board.tolist())
        available_cards = [c for c in self._card_ints if c not in unavailable_cards]
        
        distribution = np.zeros(n_turn_centroids)
        
        if len(available_cards) == 0:
            return distribution
        
        # Enumerate all possible turn cards
        for turn_card in available_cards:
            # O(1) direct index computation — no search needed!
            turn_board = np.append(board, turn_card)
            row_idx = self.get_row_index(our_hand, turn_board)
            
            turn_ehs_dist = None
            if turn_memmap is not None:
                turn_ehs_dist = turn_memmap[row_idx]
            
            if turn_ehs_dist is not None:
                min_idx = self._find_closest_centroid(turn_ehs_dist, turn_centroids)
            else:
                # Fallback: compute if not found in chunks
                turn_ehs_dist = self.compute_exact_turn_ehs_distribution(our_hand, turn_board)
                min_idx = self._find_closest_centroid(turn_ehs_dist, turn_centroids)
            
            distribution[min_idx] += 1
        
        # Normalize
        total = distribution.sum()
        if total > 0:
            distribution /= total
        
        return distribution

    def process_flop_potential_aware_distributions(self, public: np.ndarray) -> np.ndarray:
        """
        Override parent method to use exact flop distribution.
        
        Parameters
        ----------
        public : np.ndarray
            Cards array: [hole1, hole2, flop1, flop2, flop3]
            
        Returns
        -------
        np.ndarray
            Distribution over turn clusters.
        """
        our_hand = public[:2]
        board = public[2:5]
        return self.compute_exact_flop_distribution(our_hand, board)
