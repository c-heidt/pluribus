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
"""
import logging
from itertools import combinations
from typing import Dict,  Optional, Tuple

import numpy as np

from poker_ai.clustering.card_info_lut_builder import CardInfoLutBuilder

log = logging.getLogger("poker_ai.clustering.exact_hand_strength")


# ============================================================================
# Module-level cache for multiprocessing support
# ============================================================================
# When using ProcessPoolExecutor, instance variables don't serialize to workers.
# Each worker process maintains its own module-level cache (lazy loaded on first use).
# This allows efficient lookups across all workers without passing large data structures.

_PROCESS_CACHE = {
    "river_index": None,
    "turn_index": None,
    "river_memmap": None,
    "turn_memmap": None,
    "river_centroids": None,
    "turn_centroids": None,
    "save_dir": None,
}


def _get_process_cache(street: str, save_dir: str):
    """
    Get or load lookup data for a street in the current process.
    
    This function is called by worker processes to lazily load the index
    and memmap data they need. Each process loads once and caches.
    
    Parameters
    ----------
    street : str
        The street name ("river" or "turn").
    save_dir : str
        The save directory containing merged data.
        
    Returns
    -------
    Tuple[Dict[Tuple, int], np.memmap, np.ndarray]
        The index, memmap, and centroids for the street.
        Returns (None, None, None) if not available.
    """
    from pathlib import Path
    
    # Update save_dir if changed (for different calls to the builder)
    if _PROCESS_CACHE["save_dir"] != save_dir:
        _PROCESS_CACHE["save_dir"] = save_dir
        # Clear cache if save_dir changed
        _PROCESS_CACHE["river_index"] = None
        _PROCESS_CACHE["turn_index"] = None
        _PROCESS_CACHE["river_memmap"] = None
        _PROCESS_CACHE["turn_memmap"] = None
        _PROCESS_CACHE["river_centroids"] = None
        _PROCESS_CACHE["turn_centroids"] = None
    
    index_key = f"{street}_index"
    memmap_key = f"{street}_memmap"
    centroids_key = f"{street}_centroids"
    
    # Return cached if available
    if _PROCESS_CACHE[index_key] is not None:
        return _PROCESS_CACHE[index_key], _PROCESS_CACHE[memmap_key], _PROCESS_CACHE[centroids_key]
    
    # Load index (saved as .npy by chunked_processor)
    index_path = Path(save_dir) / street / "lookup_index.npy"
    if not index_path.exists():
        log.warning(f"Index not found for {street} at {index_path}")
        return None, None, None
    
    # Load index saved with np.save (dictionary stored with allow_pickle=True)
    index = np.load(index_path, allow_pickle=True).item()
    _PROCESS_CACHE[index_key] = index
    
    # Load memmap
    merged_path = Path(save_dir) / street / "merged_data.dat"
    if not merged_path.exists():
        log.warning(f"Merged data not found for {street} at {merged_path}")
        return index, None, None
    
    memmap = np.load(merged_path, mmap_mode='r', allow_pickle=True)
    _PROCESS_CACHE[memmap_key] = memmap
    
    # Load centroids
    centroids_path = Path(save_dir) / street / "centroids.npy"
    if not centroids_path.exists():
        log.warning(f"Centroids not found for {street} at {centroids_path}")
        # Return what we have so far
        return index, memmap, None
    
    centroids = np.load(centroids_path, allow_pickle=True)
    _PROCESS_CACHE[centroids_key] = centroids
    
    log.debug(f"Loaded {street} lookup data in process (index: {len(index)}, centroids: {len(centroids)})")
    return index, memmap, centroids


def _clear_process_cache_for_street(street: str):
    """
    Clear cached data for a specific street to free memory.
    
    Call this when a street's data is no longer needed:
    - Clear river cache when turn completes (flop doesn't need river)
    - Could clear turn cache when flop completes (nothing after flop)
    
    Parameters
    ----------
    street : str
        The street name ("river" or "turn").
    """
    index_key = f"{street}_index"
    memmap_key = f"{street}_memmap"
    centroids_key = f"{street}_centroids"
    
    # Clear the cache entries
    _PROCESS_CACHE[index_key] = None
    _PROCESS_CACHE[memmap_key] = None
    _PROCESS_CACHE[centroids_key] = None
    
    log.debug(f"Cleared process cache for {street}")


class ExactHandStrengthBuilder(CardInfoLutBuilder):
    """
    Computes exact hand strength using exhaustive enumeration.
    
    Unlike the parent CardInfoLutBuilder which uses Monte Carlo methods,
    this class computes exact hand strength by:
    - Comparing against ALL possible opponent hands (river)
    - Enumerating ALL possible future cards (turn/flop)
    
    Optimized to reuse pre-computed EHS values from chunk files using
    memory-efficient index-based lookups rather than loading all data
    into memory.
    
    Inherits all chunked processing and clustering infrastructure from
    CardInfoLutBuilder.
    
    Uses module-level caching for multiprocessing support - each worker process
    loads indices, memmaps, and centroids once and reuses them.
    """

    def __init__(
        self,
        low_card_rank: int,
        high_card_rank: int,
        save_dir: str,
        workers: Optional[int] = None,
        chunk_size: int = 10000,
        use_mini_batch: bool = True,
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

    # =========================================================================
    # River: Override to use exact EHS computation
    # =========================================================================
    
    def compute_exact_river_ehs(self, our_hand: np.ndarray, board: np.ndarray) -> np.ndarray:
        """
        Compute exact expected hand strength on the river.
        
        Enumerates ALL possible opponent hole card pairs and computes exact
        win/loss/tie rates.
        
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
        
        # Evaluate our hand once
        our_rank = self._evaluator.evaluate(
            board=board.astype(np.int64).tolist(),
            cards=our_hand.astype(np.int64).tolist(),
        )
        
        # Count outcomes
        wins = 0
        losses = 0
        ties = 0
        
        # Enumerate all possible opponent hands (2-card combinations)
        for opp_hand in combinations(available_cards, 2):
            opp_rank = self._evaluator.evaluate(
                board=board.astype(np.int64).tolist(),
                cards=[int(c) for c in opp_hand],
            )
            
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
        from chunk files instead of recomputing hand evaluations.
        
        Uses module-level cache for multiprocessing compatibility.
        
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
        # Get river lookup data from process cache (multiprocessing-safe)
        # This loads index, memmap, AND centroids to avoid repeated disk I/O
        river_index, river_memmap, river_centroids = _get_process_cache("river", self._config["save_dir"])
        
        # Fallback to instance centroids if not in cache (shouldn't happen normally)
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
        
        # Pre-compute hand and board for key generation (avoid repeated conversions)
        sorted_hand = sorted(our_hand.tolist(), reverse=True)
        sorted_board_prefix = sorted(board.tolist(), reverse=True)
        
        # Enumerate all possible river cards
        for river_card in available_cards:
            # Create key efficiently: hand + board + river_card (all sorted)
            sorted_board = sorted_board_prefix + [river_card]
            sorted_board.sort(reverse=True)
            key = tuple(sorted_hand + sorted_board)
            
            # Lookup using process cache
            ehs = None
            if river_index is not None and river_memmap is not None:
                row_idx = river_index.get(key)
                if row_idx is not None:
                    ehs = river_memmap[row_idx]
            
            if ehs is not None:
                # Find closest centroid using the pre-computed EHS
                min_idx = self._find_closest_centroid(ehs, river_centroids)
            else:
                # Fallback: compute exact river EHS (shouldn't happen if all chunks complete)
                # Need to construct river_board for fallback
                river_board = np.append(board, river_card)
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
        from the chunks instead of recomputing.
        
        Uses module-level cache for multiprocessing compatibility.
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
        # Flop only needs turn data, not river data
        if _PROCESS_CACHE["river_index"] is not None:
            _clear_process_cache_for_street("river")
        
        # Get turn lookup data from process cache (multiprocessing-safe)
        # This loads index, memmap, AND centroids to avoid repeated disk I/O
        turn_index, turn_memmap, turn_centroids = _get_process_cache("turn", self._config["save_dir"])
        
        # Fallback to instance centroids if not in cache (shouldn't happen normally)
        if turn_centroids is None:
            if "turn" not in self.centroids or self.centroids["turn"] is None:
                self.centroids["turn"] = self.chunked_processor.load_centroids("turn")
            turn_centroids = self.centroids["turn"]
        
        # Cards not available for turn
        unavailable_cards = set(our_hand.tolist() + board.tolist())
        available_cards = [c for c in self._card_ints if c not in unavailable_cards]
        
        n_turn_centroids = len(turn_centroids)
        distribution = np.zeros(n_turn_centroids)
        
        if len(available_cards) == 0:
            return distribution
        
        # Pre-compute hand and board for key generation (avoid repeated conversions)
        sorted_hand = sorted(our_hand.tolist(), reverse=True)
        sorted_board_prefix = sorted(board.tolist(), reverse=True)
        
        # Enumerate all possible turn cards
        for turn_card in available_cards:
            # Create key efficiently: hand + board + turn_card (all sorted)
            sorted_board = sorted_board_prefix + [turn_card]
            sorted_board.sort(reverse=True)
            key = tuple(sorted_hand + sorted_board)
            
            # Lookup using process cache
            turn_ehs_dist = None
            if turn_index is not None and turn_memmap is not None:
                row_idx = turn_index.get(key)
                if row_idx is not None:
                    turn_ehs_dist = turn_memmap[row_idx]
            
            if turn_ehs_dist is not None:
                # Find closest centroid for the distribution
                min_idx = self._find_closest_centroid(turn_ehs_dist, turn_centroids)
                distribution[min_idx] += 1
            else:
                # Fallback: compute if not found in chunks (shouldn't happen normally)
                # Need to construct turn_board for fallback
                turn_board = np.append(board, turn_card)
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
