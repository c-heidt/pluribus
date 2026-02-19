"""
Memory-efficient card information lookup table builder with checkpointing.

This module provides the CardInfoLutBuilder class which computes card
clustering for poker AI. It supports:
- Chunked processing to minimize RAM usage
- Checkpointing for resumable computations
- Memory-mapped arrays for large-scale clustering
- MiniBatchKMeans for efficient clustering of large datasets
"""
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures
import os

import joblib
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from poker_ai.clustering.card_combos import CardCombos
from poker_ai.clustering.chunked_processor import ChunkedProcessor
from poker_ai.clustering.game_utility import GameUtility
from poker_ai.clustering.preflop import compute_preflop_lossless_abstraction

log = logging.getLogger("poker_ai.clustering.card_info_lut_builder")


def atomic_joblib_dump(obj: Any, path: Path):
    """Save an object with joblib atomically using temp file."""
    temp_path = path.with_suffix(".tmp.joblib")
    try:
        joblib.dump(obj, temp_path)
        shutil.move(str(temp_path), str(path))
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to save {path}: {e}")


class CardInfoLutBuilder(CardCombos):
    """
    Stores info buckets for each street when called.
    
    This class builds card information lookup tables using a memory-efficient
    chunked processing approach with checkpointing support.

    Attributes
    ----------
    card_info_lut : Dict[str, Any]
        Lookup table of card combinations per betting round to a cluster id.
    centroids : Dict[str, Any]
        Centroids per betting round for use in clustering previous rounds by
        earth movers distance.
    chunked_processor : ChunkedProcessor
        Handles chunked file I/O and checkpointing.
    """

    def __init__(
        self,
        n_simulations_river: int,
        n_simulations_turn: int,
        n_simulations_flop: int,
        low_card_rank: int,
        high_card_rank: int,
        save_dir: str,
        workers: Optional[int] = None,
        chunk_size: int = 10000,
        use_mini_batch: bool = True,
    ):
        """
        Initialize the CardInfoLutBuilder.
        
        Parameters
        ----------
        n_simulations_river : int
            Number of opponent hand simulations on the river.
        n_simulations_turn : int
            Number of river card simulations on the turn.
        n_simulations_flop : int
            Number of turn card simulations on the flop.
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
        self.n_simulations_river = n_simulations_river
        self.n_simulations_turn = n_simulations_turn
        self.n_simulations_flop = n_simulations_flop
        self.workers = workers
        self.chunk_size = chunk_size
        self.use_mini_batch = use_mini_batch
        
        super().__init__(low_card_rank, high_card_rank)
        
        # Select appropriate evaluator based on deck size
        n_ranks = high_card_rank - low_card_rank + 1
        if n_ranks in [5, 9]:  # 20-card or 36-card deck
            from poker_ai.poker.evaluation.short_deck_evaluator import ShortDeckEvaluator
            self._evaluator = ShortDeckEvaluator()
        else:  # 52-card deck
            from poker_ai.poker.evaluation import Evaluator
            self._evaluator = Evaluator()
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.card_info_lut_path: Path = self.save_dir / "card_info_lut.joblib"
        self.centroid_path: Path = self.save_dir / "centroids.joblib"
        
        # Initialize chunked processor
        self.chunked_processor = ChunkedProcessor(
            save_dir=self.save_dir,
            chunk_size=chunk_size,
        )
        
        # Load existing results if available
        try:
            self.card_info_lut: Dict[str, Any] = joblib.load(self.card_info_lut_path)
            self.centroids: Dict[str, Any] = joblib.load(self.centroid_path)
        except FileNotFoundError:
            self.centroids: Dict[str, Any] = {}
            self.card_info_lut: Dict[str, Any] = {}
        
        # Store configuration for validation
        self._config = {
            "n_simulations_river": n_simulations_river,
            "n_simulations_turn": n_simulations_turn,
            "n_simulations_flop": n_simulations_flop,
            "low_card_rank": low_card_rank,
            "high_card_rank": high_card_rank,
            "chunk_size": chunk_size,
        }

    def _get_worker_count(self) -> int:
        """Get the number of worker processes to use."""
        if self.workers and int(self.workers) > 0:
            return int(self.workers)
        return os.cpu_count() or 1

    def compute(
        self,
        n_river_clusters: int,
        n_turn_clusters: int,
        n_flop_clusters: int,
    ):
        """
        Compute all clusters and save to card_info_lut dictionary.

        Will attempt to load previous progress and will save after each cluster
        is computed. Uses chunked processing for memory efficiency.
        
        Parameters
        ----------
        n_river_clusters : int
            Number of clusters for river.
        n_turn_clusters : int
            Number of clusters for turn.
        n_flop_clusters : int
            Number of clusters for flop.
        """
        log.info("Starting computation of clusters.")
        start = time.time()
        
        # Update config with cluster counts
        self._config.update({
            "n_river_clusters": n_river_clusters,
            "n_turn_clusters": n_turn_clusters,
            "n_flop_clusters": n_flop_clusters,
        })
        
        if "pre_flop" not in self.card_info_lut:
            log.info("Computing pre-flop abstraction...")
            self.card_info_lut["pre_flop"] = compute_preflop_lossless_abstraction(
                builder=self
            )
            atomic_joblib_dump(self.card_info_lut, self.card_info_lut_path)
        
        if "river" not in self.card_info_lut:
            self.card_info_lut["river"] = self._compute_river_clusters(
                n_river_clusters,
            )
            atomic_joblib_dump(self.card_info_lut, self.card_info_lut_path)
            atomic_joblib_dump(self.centroids, self.centroid_path)
        
        if "turn" not in self.card_info_lut:
            self.card_info_lut["turn"] = self._compute_turn_clusters(n_turn_clusters)
            atomic_joblib_dump(self.card_info_lut, self.card_info_lut_path)
            atomic_joblib_dump(self.centroids, self.centroid_path)
        
        if "flop" not in self.card_info_lut:
            self.card_info_lut["flop"] = self._compute_flop_clusters(n_flop_clusters)
            atomic_joblib_dump(self.card_info_lut, self.card_info_lut_path)
            atomic_joblib_dump(self.centroids, self.centroid_path)
        
        end = time.time()
        log.info(f"Finished computation of clusters - took {end - start:.2f} seconds.")
        
        # Final cleanup: remove any remaining intermediate files
        log.info("Performing final cleanup of intermediate files...")
        self.chunked_processor.cleanup_all_intermediate_files()

    def _compute_river_clusters(self, n_river_clusters: int) -> Dict:
        """
        Compute river clusters using chunked processing.
        
        Parameters
        ----------
        n_river_clusters : int
            Number of clusters.
            
        Returns
        -------
        Dict
            Lookup table mapping card combos to cluster IDs.
        """
        street = "river"
        log.info(f"\n{'='*80}")
        log.info(f"STAGE 1/3: RIVER CLUSTERING")
        log.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"Clusters: {n_river_clusters} | Total combos: {len(self.river):,}")
        log.info(f"{'='*80}\n")
        start = time.time()
        
        total_combos = len(self.river)
        self.chunked_processor.initialize_street(street, total_combos, self._config)
        
        # Process incomplete chunks
        incomplete_chunks = self.chunked_processor.get_incomplete_chunks(street)
        
        if incomplete_chunks:
            log.info(f"Processing {len(incomplete_chunks)} incomplete chunks for {street}")
            self._process_river_chunks(incomplete_chunks)
        
        # Check if clustering already done
        if self.chunked_processor.is_clustering_done(street):
            log.info(f"Loading existing clustering results for {street}")
            self.centroids["river"] = self.chunked_processor.load_centroids(street)
            clusters = self.chunked_processor.load_clusters(street)
            _, all_combos = self.chunked_processor.get_or_merge_data(street)
        else:
            # Merge (or load if already merged) and cluster
            merged_data, all_combos = self.chunked_processor.get_or_merge_data(street)
            
            # Build lookup index for later use (e.g., by turn stage)
            log.info(f"Building lookup index for {street}...")
            self.chunked_processor.get_or_build_index(street, all_combos)
            
            self.centroids["river"], clusters = self._cluster(
                num_clusters=n_river_clusters,
                X=merged_data,
                street=street,
            )
            
            # Save clustering results
            self.chunked_processor.save_centroids(street, self.centroids["river"])
            self.chunked_processor.save_clusters(street, clusters)
            self.chunked_processor.mark_clustering_done(street)
            
            # Cleanup intermediate files (chunks no longer needed)
            log.info(f"Cleaning up intermediate chunk files for {street}...")
            self.chunked_processor.cleanup_chunks(street)
            self.chunked_processor.cleanup_partial_clustering(street)
        
        end = time.time()
        log.info(f"Finished computation of {street} clusters - took {end - start:.2f} seconds.")
        
        return self.create_card_lookup(clusters, all_combos)

    def _process_river_chunks(self, chunk_indices: List[int]):
        """Process river chunks in parallel (2 chunks simultaneously)."""
        workers = self._get_worker_count()
        chunk_specs = self.chunked_processor.get_chunk_indices(len(self.river))
        parallel_chunks = 2  # Process 2 chunks at a time
        total_chunks = len(self.chunked_processor.get_chunk_indices(len(self.river)))
        
        log.info(f"Processing {len(chunk_indices)} chunks with {workers} workers...")
        start_time = time.time()
        
        # Reuse single executor for all chunks
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            # Process parallel_chunks at a time
            for batch_idx, batch_start in enumerate(range(0, len(chunk_indices), parallel_chunks)):
                chunk_batch = chunk_indices[batch_start:batch_start + parallel_chunks]
                futures = {}
                
                # Submit all chunks in this batch
                for chunk_idx in chunk_batch:
                    _, start_idx, end_idx = chunk_specs[chunk_idx]
                    chunk_combos = self.river[start_idx:end_idx]
                    chunksize = max(1, len(chunk_combos) // (workers * 4))
                    
                    future = executor.map(
                        self.process_river_ehs,
                        chunk_combos,
                        chunksize=chunksize,
                    )
                    futures[chunk_idx] = (future, chunk_combos)
                
                # Collect results and save
                for chunk_idx, (future, chunk_combos) in futures.items():
                    chunk_results = list(future)
                    self.chunked_processor.save_chunk(
                        "river",
                        chunk_idx,
                        np.array(chunk_results, dtype=np.float32),
                        chunk_combos,
                    )
                    # Defer checkpoint save until batch is complete
                    is_last_in_batch = chunk_idx == chunk_batch[-1]
                    self.chunked_processor.mark_chunk_complete("river", chunk_idx, defer_save=not is_last_in_batch)
                
                # Progress update every few batches
                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    completed = len(chunk_indices) - len(self.chunked_processor.get_incomplete_chunks("river"))
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total_chunks - completed) / rate if rate > 0 else 0
                    log.info(f"[RIVER] Chunk {completed}/{total_chunks} | "
                            f"Elapsed: {elapsed/3600:.1f}h | "
                            f"ETA: {remaining/3600:.1f}h | "
                            f"Rate: {rate*3600:.1f} chunks/hr")

    def _compute_turn_clusters(self, n_turn_clusters: int) -> Dict:
        """
        Compute turn clusters using chunked processing.
        
        Parameters
        ----------
        n_turn_clusters : int
            Number of clusters.
            
        Returns
        -------
        Dict
            Lookup table mapping card combos to cluster IDs.
        """
        street = "turn"
        log.info(f"\n{'='*80}")
        log.info(f"STAGE 2/3: TURN CLUSTERING")
        log.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"Clusters: {n_turn_clusters} | Total combos: {len(self.turn):,}")
        log.info(f"{'='*80}\n")
        start = time.time()
        
        total_combos = len(self.turn)
        self.chunked_processor.initialize_street(street, total_combos, self._config)
        
        # Process incomplete chunks
        incomplete_chunks = self.chunked_processor.get_incomplete_chunks(street)
        
        if incomplete_chunks:
            log.info(f"Processing {len(incomplete_chunks)} incomplete chunks for {street}")
            self._process_turn_chunks(incomplete_chunks)
        
        # Check if clustering already done
        if self.chunked_processor.is_clustering_done(street):
            log.info(f"Loading existing clustering results for {street}")
            self.centroids["turn"] = self.chunked_processor.load_centroids(street)
            clusters = self.chunked_processor.load_clusters(street)
            _, all_combos = self.chunked_processor.get_or_merge_data(street)
        else:
            # Merge (or load if already merged) and cluster
            merged_data, all_combos = self.chunked_processor.get_or_merge_data(street)
            
            # Build lookup index for later use (e.g., by flop stage)
            log.info(f"Building lookup index for {street}...")
            self.chunked_processor.get_or_build_index(street, all_combos)
            
            self.centroids["turn"], clusters = self._cluster(
                num_clusters=n_turn_clusters,
                X=merged_data,
                street=street,
            )
            
            # Save clustering results
            self.chunked_processor.save_centroids(street, self.centroids["turn"])
            self.chunked_processor.save_clusters(street, clusters)
            self.chunked_processor.mark_clustering_done(street)
            
            # Cleanup intermediate files (chunks no longer needed)
            log.info(f"Cleaning up intermediate chunk files for {street}...")
            self.chunked_processor.cleanup_chunks(street)
            self.chunked_processor.cleanup_partial_clustering(street)
        
        end = time.time()
        log.info(f"Finished computation of {street} clusters - took {end - start:.2f} seconds.")
        
        return self.create_card_lookup(clusters, all_combos)

    def _process_turn_chunks(self, chunk_indices: List[int]):
        """Process turn chunks in parallel (2 chunks simultaneously)."""
        workers = self._get_worker_count()
        chunk_specs = self.chunked_processor.get_chunk_indices(len(self.turn))
        parallel_chunks = 2  # Process 2 chunks at a time
        total_chunks = len(self.chunked_processor.get_chunk_indices(len(self.turn)))
        
        log.info(f"Processing {len(chunk_indices)} chunks with {workers} workers...")
        start_time = time.time()
        
        # Reuse single executor for all chunks
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            # Process parallel_chunks at a time
            for batch_idx, batch_start in enumerate(range(0, len(chunk_indices), parallel_chunks)):
                chunk_batch = chunk_indices[batch_start:batch_start + parallel_chunks]
                futures = {}
                
                # Submit all chunks in this batch
                for chunk_idx in chunk_batch:
                    _, start_idx, end_idx = chunk_specs[chunk_idx]
                    chunk_combos = self.turn[start_idx:end_idx]
                    chunksize = max(1, len(chunk_combos) // (workers * 4))
                    
                    future = executor.map(
                        self.process_turn_ehs_distributions,
                        chunk_combos,
                        chunksize=chunksize,
                    )
                    futures[chunk_idx] = (future, chunk_combos)
                
                # Collect results and save
                for chunk_idx, (future, chunk_combos) in futures.items():
                    chunk_results = list(future)
                    self.chunked_processor.save_chunk(
                        "turn",
                        chunk_idx,
                        np.array(chunk_results, dtype=np.float32),
                        chunk_combos,
                    )
                    # Defer checkpoint save until batch is complete
                    is_last_in_batch = chunk_idx == chunk_batch[-1]
                    self.chunked_processor.mark_chunk_complete("turn", chunk_idx, defer_save=not is_last_in_batch)
                
                # Progress update every few batches
                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    completed = len(chunk_indices) - len(self.chunked_processor.get_incomplete_chunks("turn"))
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total_chunks - completed) / rate if rate > 0 else 0
                    log.info(f"[TURN] Chunk {completed}/{total_chunks} | "
                            f"Elapsed: {elapsed/3600:.1f}h | "
                            f"ETA: {remaining/3600:.1f}h | "
                            f"Rate: {rate*3600:.1f} chunks/hr")

    def _compute_flop_clusters(self, n_flop_clusters: int) -> Dict:
        """
        Compute flop clusters using chunked processing.
        
        Parameters
        ----------
        n_flop_clusters : int
            Number of clusters.
            
        Returns
        -------
        Dict
            Lookup table mapping card combos to cluster IDs.
        """
        street = "flop"
        log.info(f"\n{'='*80}")
        log.info(f"STAGE 3/3: FLOP CLUSTERING")
        log.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"Clusters: {n_flop_clusters} | Total combos: {len(self.flop):,}")
        log.info(f"{'='*80}\n")
        start = time.time()
        
        total_combos = len(self.flop)
        self.chunked_processor.initialize_street(street, total_combos, self._config)
        
        # Process incomplete chunks
        incomplete_chunks = self.chunked_processor.get_incomplete_chunks(street)
        
        if incomplete_chunks:
            log.info(f"Processing {len(incomplete_chunks)} incomplete chunks for {street}")
            self._process_flop_chunks(incomplete_chunks)
        
        # Check if clustering already done
        if self.chunked_processor.is_clustering_done(street):
            log.info(f"Loading existing clustering results for {street}")
            self.centroids["flop"] = self.chunked_processor.load_centroids(street)
            clusters = self.chunked_processor.load_clusters(street)
            _, all_combos = self.chunked_processor.get_or_merge_data(street)
        else:
            # Merge (or load if already merged) and cluster
            merged_data, all_combos = self.chunked_processor.get_or_merge_data(street)
            
            # Build lookup index (though flop is final stage)
            log.info(f"Building lookup index for {street}...")
            self.chunked_processor.get_or_build_index(street, all_combos)
            
            self.centroids["flop"], clusters = self._cluster(
                num_clusters=n_flop_clusters,
                X=merged_data,
                street=street,
            )
            
            # Save clustering results
            self.chunked_processor.save_centroids(street, self.centroids["flop"])
            self.chunked_processor.save_clusters(street, clusters)
            self.chunked_processor.mark_clustering_done(street)
            
            # Cleanup intermediate files (chunks no longer needed)
            log.info(f"Cleaning up intermediate chunk files for {street}...")
            self.chunked_processor.cleanup_chunks(street)
            self.chunked_processor.cleanup_partial_clustering(street)
        
        end = time.time()
        log.info(f"Finished computation of {street} clusters - took {end - start:.2f} seconds.")
        
        return self.create_card_lookup(clusters, all_combos)

    def _process_flop_chunks(self, chunk_indices: List[int]):
        """Process flop chunks in parallel (2 chunks simultaneously)."""
        workers = self._get_worker_count()
        chunk_specs = self.chunked_processor.get_chunk_indices(len(self.flop))
        parallel_chunks = 2  # Process 2 chunks at a time
        total_chunks = len(self.chunked_processor.get_chunk_indices(len(self.flop)))
        
        log.info(f"Processing {len(chunk_indices)} chunks with {workers} workers...")
        start_time = time.time()
        
        # Reuse single executor for all chunks
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            # Process parallel_chunks at a time
            for batch_idx, batch_start in enumerate(range(0, len(chunk_indices), parallel_chunks)):
                chunk_batch = chunk_indices[batch_start:batch_start + parallel_chunks]
                futures = {}
                
                # Submit all chunks in this batch
                for chunk_idx in chunk_batch:
                    _, start_idx, end_idx = chunk_specs[chunk_idx]
                    chunk_combos = self.flop[start_idx:end_idx]
                    chunksize = max(1, len(chunk_combos) // (workers * 4))
                    
                    future = executor.map(
                        self.process_flop_potential_aware_distributions,
                        chunk_combos,
                        chunksize=chunksize,
                    )
                    futures[chunk_idx] = (future, chunk_combos)
                
                # Collect results and save
                for chunk_idx, (future, chunk_combos) in futures.items():
                    chunk_results = list(future)
                    self.chunked_processor.save_chunk(
                        "flop",
                        chunk_idx,
                        np.array(chunk_results, dtype=np.float32),
                        chunk_combos,
                    )
                    # Defer checkpoint save until batch is complete
                    is_last_in_batch = chunk_idx == chunk_batch[-1]
                    self.chunked_processor.mark_chunk_complete("flop", chunk_idx, defer_save=not is_last_in_batch)
                
                # Progress update every few batches
                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    completed = len(chunk_indices) - len(self.chunked_processor.get_incomplete_chunks("flop"))
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total_chunks - completed) / rate if rate > 0 else 0
                    log.info(f"[FLOP] Chunk {completed}/{total_chunks} | "
                            f"Elapsed: {elapsed/3600:.1f}h | "
                            f"ETA: {remaining/3600:.1f}h | "
                            f"Rate: {rate*3600:.1f} chunks/hr")

    def _cluster(
        self,
        num_clusters: int,
        X: np.ndarray,
        street: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform clustering using KMeans or MiniBatchKMeans.
        
        Uses MiniBatchKMeans for large datasets to reduce memory usage.
        Saves intermediate checkpoints during MiniBatchKMeans for resumability.
        
        Parameters
        ----------
        num_clusters : int
            Number of clusters.
        X : np.ndarray
            Data to cluster (can be memory-mapped).
        street : str
            Street name for logging.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (centroids, cluster_assignments).
        """
        n_samples = X.shape[0]
        log.info(f"Clustering {n_samples} samples into {num_clusters} clusters for {street}")
        
        # Handle edge case: fewer samples than clusters
        if n_samples < num_clusters:
            log.warning(
                f"Number of samples ({n_samples}) is less than number of clusters "
                f"({num_clusters}). Reducing clusters to {n_samples}."
            )
            num_clusters = n_samples
        
        # Handle edge case: single sample
        if n_samples == 1:
            return X.copy(), np.array([0])
        
        # Use MiniBatchKMeans for large datasets
        use_minibatch = self.use_mini_batch and n_samples > 50000
        
        if use_minibatch:
            log.info(f"Using MiniBatchKMeans for {street} (large dataset)")
            batch_size = min(10000, n_samples)
            
            # Check for partial clustering checkpoint
            partial_km_path = self.chunked_processor.get_street_dir(street) / "partial_kmeans.joblib"
            progress_path = self.chunked_processor.get_street_dir(street) / "kmeans_progress.json"
            
            start_batch = 0
            if partial_km_path.exists() and progress_path.exists():
                try:
                    log.info(f"Resuming partial clustering for {street}")
                    km = joblib.load(partial_km_path)
                    with open(progress_path, "r") as f:
                        progress = json.load(f)
                    start_batch = progress.get("completed_batches", 0)
                    log.info(f"Resuming from batch {start_batch}")
                except Exception as e:
                    log.warning(f"Failed to load clustering checkpoint, starting fresh: {e}")
                    start_batch = 0
                    km = MiniBatchKMeans(
                        n_clusters=num_clusters,
                        init="k-means++",
                        n_init=10,
                        max_iter=300,
                        batch_size=batch_size,
                        random_state=0,
                        verbose=0,
                    )
            else:
                km = MiniBatchKMeans(
                    n_clusters=num_clusters,
                    init="k-means++",
                    n_init=10,
                    max_iter=300,
                    batch_size=batch_size,
                    random_state=0,
                    verbose=0,
                )
            
            # Manual incremental fitting with checkpoints
            n_batches = (n_samples + batch_size - 1) // batch_size
            checkpoint_interval = max(1, n_batches // 10)  # Save every ~10%
            
            for i in tqdm(range(start_batch, n_batches), desc=f"Clustering {street}", initial=start_batch, total=n_batches):
                batch_start = i * batch_size
                batch_end = min(batch_start + batch_size, n_samples)
                batch_data = np.array(X[batch_start:batch_end])  # Load from memmap
                
                km.partial_fit(batch_data)
                
                # Checkpoint periodically
                if (i + 1) % checkpoint_interval == 0 or (i + 1) == n_batches:
                    log.info(f"Clustering checkpoint: {i + 1}/{n_batches} batches ({100*(i+1)//n_batches}%)")
                    # Save atomically
                    temp_km_path = partial_km_path.with_suffix(".tmp.joblib")
                    temp_progress_path = progress_path.with_suffix(".tmp.json")
                    try:
                        joblib.dump(km, temp_km_path)
                        with open(temp_progress_path, "w") as f:
                            json.dump({"completed_batches": i + 1, "total_batches": n_batches}, f)
                        shutil.move(str(temp_km_path), str(partial_km_path))
                        shutil.move(str(temp_progress_path), str(progress_path))
                    except Exception as e:
                        log.warning(f"Failed to save clustering checkpoint: {e}")
                        for p in [temp_km_path, temp_progress_path]:
                            if p.exists():
                                p.unlink()
            
            log.info(f"Predicting cluster assignments for {n_samples} samples...")
            y_km = km.predict(X)
            centroids = km.cluster_centers_
            
            # Remove partial checkpoint after successful completion
            if partial_km_path.exists():
                partial_km_path.unlink()
            if progress_path.exists():
                progress_path.unlink()
            
            log.info(f"Cleaned up partial clustering checkpoints for {street}")
        else:
            log.info(f"Using standard KMeans for {street}")
            km = KMeans(
                n_clusters=num_clusters,
                init="random",
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0,
            )
            y_km = km.fit_predict(X)
            centroids = km.cluster_centers_
        
        return centroids, y_km

    # Keep the old cluster method for backward compatibility
    @staticmethod
    def cluster(num_clusters: int, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy clustering method for backward compatibility.
        
        Parameters
        ----------
        num_clusters : int
            Number of clusters.
        X : np.ndarray
            Data to cluster.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (centroids, cluster_assignments).
        """
        km = KMeans(
            n_clusters=num_clusters,
            init="random",
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0,
        )
        y_km = km.fit_predict(X)
        centroids = km.cluster_centers_
        return centroids, y_km

    def _find_closest_centroid(
        self,
        point: np.ndarray,
        centroids: np.ndarray,
    ) -> int:
        """
        Find the closest centroid to a point using Wasserstein distance.
        
        Parameters
        ----------
        point : np.ndarray
            The point to compare.
        centroids : np.ndarray
            Array of centroids.
            
        Returns
        -------
        int
            Index of the closest centroid.
        """
        min_idx = 0
        min_emd = float('inf')
        
        for idx, centroid in enumerate(centroids):
            emd = wasserstein_distance(point, centroid)
            if emd < min_emd:
                min_idx = idx
                min_emd = emd
        
        return min_idx

    def simulate_get_ehs(self, game: GameUtility) -> np.ndarray:
        """
        Get expected hand strength object.

        Parameters
        ----------
        game : GameUtility
            GameState for help with determining winner and sampling opponent hand

        Returns
        -------
        ehs : np.ndarray
            [win_rate, loss_rate, tie_rate]
        """
        ehs: np.ndarray = np.zeros(3)
        for _ in range(self.n_simulations_river):
            idx: int = game.get_winner()
            # increment win rate for winner/tie
            ehs[idx] += 1 / self.n_simulations_river
        return ehs

    def simulate_get_turn_ehs_distributions(
        self,
        available_cards: np.ndarray,
        the_board: np.ndarray,
        our_hand: np.ndarray,
    ) -> np.ndarray:
        """
        Get histogram of frequencies that a given turn situation resulted in a
        certain cluster id after a river simulation.

        Parameters
        ----------
        available_cards : np.ndarray
            Array of available cards on the turn
        the_board : np.ndarray
            The board as of the turn
        our_hand : np.ndarray
            Cards in our hand

        Returns
        -------
        turn_ehs_distribution : np.ndarray
            Array of counts for each cluster the turn fell into by the river
            after simulations
        """
        n_river_centroids = len(self.centroids["river"])
        turn_ehs_distribution = np.zeros(n_river_centroids)
        
        # Handle edge case: no available cards
        if len(available_cards) == 0:
            return turn_ehs_distribution
        
        for _ in range(self.n_simulations_turn):
            river_card = np.random.choice(available_cards, 1, replace=False)
            board = np.append(the_board, river_card)
            game = GameUtility(
                our_hand=our_hand, 
                board=board, 
                cards=self._card_ints,
                evaluator=self._evaluator
            )
            ehs = self.simulate_get_ehs(game)
            
            # Find closest centroid using helper method
            min_idx = self._find_closest_centroid(ehs, self.centroids["river"])
            turn_ehs_distribution[min_idx] += 1 / self.n_simulations_turn
        
        return turn_ehs_distribution

    def process_river_ehs(self, public: np.ndarray) -> np.ndarray:
        """
        Get the expected hand strength for a particular card combo.

        Parameters
        ----------
        public : np.ndarray
            Cards to process

        Returns
        -------
            Expected hand strength
        """
        our_hand = public[:2]
        board = public[2:7]
        # Get expected hand strength
        game = GameUtility(
            our_hand=our_hand, 
            board=board, 
            cards=self._card_ints,
            evaluator=self._evaluator
        )
        return self.simulate_get_ehs(game)

    @staticmethod
    def get_available_cards(
        cards: np.ndarray, unavailable_cards: np.ndarray
    ) -> np.ndarray:
        """
        Get all cards that are available.

        Parameters
        ----------
        cards : np.ndarray
        unavailable_cards : np.array
            Cards that are not available.

        Returns
        -------
            Available cards
        """
        # Turn into set for O(1) lookup speed.
        unavailable_cards = set(unavailable_cards.tolist())
        return np.array([c for c in cards if c not in unavailable_cards])

    def process_turn_ehs_distributions(self, public: np.ndarray) -> np.ndarray:
        """
        Get the potential aware turn distribution for a particular card combo.

        Parameters
        ----------
        public : np.ndarray
            Cards to process

        Returns
        -------
            Potential aware turn distributions
        """
        available_cards: np.ndarray = self.get_available_cards(
            cards=self._card_ints, unavailable_cards=public
        )
        # sample river cards and run a simulation
        turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(
            available_cards, the_board=public[2:6], our_hand=public[:2],
        )
        return turn_ehs_distribution

    def process_flop_potential_aware_distributions(
        self, public: np.ndarray,
    ) -> np.ndarray:
        """
        Get the potential aware flop distribution for a particular card combo.

        Parameters
        ----------
        public : np.ndarray
            Cards to process

        Returns
        -------
            Potential aware flop distributions
        """
        available_cards: np.ndarray = self.get_available_cards(
            cards=self._card_ints, unavailable_cards=public
        )
        
        n_turn_centroids = len(self.centroids["turn"])
        potential_aware_distribution_flop = np.zeros(n_turn_centroids)
        
        # Handle edge case: not enough available cards
        if len(available_cards) == 0:
            return potential_aware_distribution_flop
        
        for _ in range(self.n_simulations_flop):
            turn_card = np.random.choice(available_cards, 1, replace=False)
            our_hand = public[:2]
            board = public[2:5]
            the_board = np.append(board, turn_card).tolist()
            
            available_cards_turn = np.array(
                [x for x in available_cards if x != turn_card[0]]
            )
            
            turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(
                available_cards_turn, the_board=the_board, our_hand=our_hand,
            )
            
            min_idx = self._find_closest_centroid(
                turn_ehs_distribution, self.centroids["turn"]
            )
            potential_aware_distribution_flop[min_idx] += 1 / self.n_simulations_flop
        
        return potential_aware_distribution_flop

    @staticmethod
    def create_card_lookup(clusters: np.ndarray, card_combos: np.ndarray) -> Dict:
        """
        Create lookup table.

        Parameters
        ----------
        clusters : np.ndarray
            Array of cluster ids.
        card_combos : np.ndarray
            The card combos to which the cluster ids belong.

        Returns
        -------
        lossy_lookup : Dict
            Lookup table for finding cluster ids.
        """
        log.info("Creating lookup table.")
        lossy_lookup = {}
        for i, card_combo in enumerate(tqdm(card_combos)):
            lossy_lookup[tuple(card_combo)] = clusters[i]
        return lossy_lookup
