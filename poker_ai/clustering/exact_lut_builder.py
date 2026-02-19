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
    
    Attributes
    ----------
    _river_ehs_index : Dict[Tuple, int]
        Lightweight index mapping river card combos to row index in merged_data.dat.
        Used for memory-mapped lookups into the merged data file.
    _turn_dist_index : Dict[Tuple, int]
        Lightweight index mapping turn card combos to row index in merged_data.dat.
        Used for memory-mapped lookups into the merged data file.
    _merged_data_cache : Dict[str, np.memmap]
        Cache for memory-mapped merged data arrays per street to reduce file access overhead.
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
        
        # Lightweight indices mapping card combos to row indices in merged data (lazy loaded)
        # These only store row indices (single integer), not actual data
        self._river_ehs_index: Optional[Dict[Tuple, int]] = None
        self._turn_dist_index: Optional[Dict[Tuple, int]] = None
        
        # Cache for memory-mapped merged data arrays (one per street)
        self._merged_data_cache: Dict[str, np.memmap] = {}

    def _get_or_build_index(self, street: str) -> Dict[Tuple, int]:
        """
        Get or build a lightweight lookup index for a street.
        
        Uses ChunkedProcessor's index-building functionality which only
        stores row indices (single integer), not the actual EHS data.
        Thread-safe: uses file locking and read-only memory mapping.
        
        Parameters
        ----------
        street : str
            The street name (river, turn, flop).
            
        Returns
        -------
        Dict[Tuple, int]
            Lightweight index mapping card combos to row indices in merged_data.dat.
        """
        # Load index if it exists, otherwise need to build it from merged data
        index = self.chunked_processor.load_lookup_index(street)
        if index is not None:
            return index
        
        # Need to build index - requires all_combos from merge
        log.info(f"Index not found for {street}, building from merged data...")
        _, all_combos = self.chunked_processor.merge_chunks_to_memmap(street)
        return self.chunked_processor.get_or_build_index(street, all_combos)
    
    def _lookup_ehs_data(
        self,
        street: str,
        combo_key: Tuple[int, ...],
        index: Dict[Tuple, int],
    ) -> Optional[np.ndarray]:
        """
        Look up EHS data for a card combo using memory-mapped merged data access.
        
        Thread-safe: uses read-only memory mapping.
        
        Parameters
        ----------
        street : str
            The street name.
        combo_key : Tuple[int, ...]
            The card combo key to look up.
        index : Dict[Tuple, int]
            The lookup index.
            
        Returns
        -------
        Optional[np.ndarray]
            The EHS data for the combo, or None if not found.
        """
        # Load merged data into cache if not already present
        if street not in self._merged_data_cache:
            merged_path = self.chunked_processor.get_merged_path(street)
            if merged_path.exists():
                self._merged_data_cache[street] = np.load(merged_path, mmap_mode='r')
            else:
                log.warning(f"Merged data not found for {street}")
                return None
        
        return self.chunked_processor.lookup_data_by_index(
            street, combo_key, index, self._merged_data_cache.get(street)
        )

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
    
    def _make_lut_key(
        self,
        our_hand: np.ndarray,
        board: np.ndarray,
    ) -> Tuple[int, ...]:
        """
        Create a properly sorted LUT key for lookup.
        
        Keys in card_info_lut are stored as tuples with:
        - Hole cards sorted by eval_card descending
        - Board cards sorted by eval_card descending
        
        Parameters
        ----------
        our_hand : np.ndarray
            Our two hole cards (eval_card values).
        board : np.ndarray
            The community cards (eval_card values).
            
        Returns
        -------
        Tuple[int, ...]
            Sorted key for LUT lookup.
        """
        sorted_hand = sorted(our_hand.tolist(), reverse=True)
        sorted_board = sorted(board.tolist(), reverse=True)
        return tuple(sorted_hand + sorted_board)
    
    def compute_exact_turn_ehs_distribution(
        self,
        our_hand: np.ndarray,
        board: np.ndarray,
    ) -> np.ndarray:
        """
        Compute exact turn EHS distribution by enumerating all river cards.
        
        For each possible river card, looks up the pre-computed river EHS values
        from chunk files instead of recomputing hand evaluations.
        
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
        # Ensure river centroids are available
        if "river" not in self.centroids or self.centroids["river"] is None:
            self.centroids["river"] = self.chunked_processor.load_centroids("river")
        
        # Build river EHS index if not already loaded (lightweight, only stores row indices)
        if self._river_ehs_index is None:
            self._river_ehs_index = self._get_or_build_index("river")
        
        # Cards not available for river
        unavailable_cards = set(our_hand.tolist() + board.tolist())
        available_cards = [c for c in self._card_ints if c not in unavailable_cards]
        
        n_river_centroids = len(self.centroids["river"])
        distribution = np.zeros(n_river_centroids)
        
        if len(available_cards) == 0:
            return distribution
        
        # Enumerate all possible river cards
        for river_card in available_cards:
            river_board = np.append(board, river_card)
            
            # Look up pre-computed EHS from river chunks via memory-mapped access
            key = self._make_lut_key(our_hand, river_board)
            ehs = self._lookup_ehs_data("river", key, self._river_ehs_index)
            
            if ehs is not None:
                # Find closest centroid using the pre-computed EHS
                min_idx = self._find_closest_centroid(ehs, self.centroids["river"])
            else:
                # Fallback: compute exact river EHS (shouldn't happen if all chunks complete)
                ehs = self.compute_exact_river_ehs(our_hand, river_board)
                min_idx = self._find_closest_centroid(ehs, self.centroids["river"])
            
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
        # Ensure centroids are available (needed for determining n_clusters)
        if "turn" not in self.centroids or self.centroids["turn"] is None:
            self.centroids["turn"] = self.chunked_processor.load_centroids("turn")
        
        # Build turn distribution index if not already loaded (lightweight, only stores row indices)
        if self._turn_dist_index is None:
            self._turn_dist_index = self._get_or_build_index("turn")
        
        # Cards not available for turn
        unavailable_cards = set(our_hand.tolist() + board.tolist())
        available_cards = [c for c in self._card_ints if c not in unavailable_cards]
        
        n_turn_centroids = len(self.centroids["turn"])
        distribution = np.zeros(n_turn_centroids)
        
        if len(available_cards) == 0:
            return distribution
        
        # Enumerate all possible turn cards
        for turn_card in available_cards:
            turn_board = np.append(board, turn_card)
            key = self._make_lut_key(our_hand, turn_board)
            
            # Look up pre-computed turn EHS distribution via memory-mapped access
            turn_ehs_dist = self._lookup_ehs_data("turn", key, self._turn_dist_index)
            
            if turn_ehs_dist is not None:
                # Find closest centroid for the distribution
                min_idx = self._find_closest_centroid(turn_ehs_dist, self.centroids["turn"])
                distribution[min_idx] += 1
            else:
                # Fallback: compute if not found in chunks (shouldn't happen normally)
                turn_ehs_dist = self.compute_exact_turn_ehs_distribution(our_hand, turn_board)
                min_idx = self._find_closest_centroid(turn_ehs_dist, self.centroids["turn"])
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
