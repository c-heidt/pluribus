"""
Chunked processing with checkpointing for memory-efficient clustering.

This module provides utilities for processing large datasets in chunks,
saving intermediate results to disk, and resuming from checkpoints.
"""
import concurrent.futures
import fcntl
import gzip
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger("poker_ai.clustering.chunked_processor")


def _process_single_chunk_worker(
    chunk_idx: int,
    chunk_combos: np.ndarray,
    item_processor: Callable[[np.ndarray], np.ndarray],
) -> Tuple[int, np.ndarray]:
    """
    Worker function to process a single chunk.
    
    This is a module-level function so it can be pickled for multiprocessing.
    Each worker processes ALL items in its assigned chunk sequentially.
    
    Parameters
    ----------
    chunk_idx : int
        The chunk index being processed.
    chunk_combos : np.ndarray
        Array of card combinations for this chunk.
    item_processor : Callable
        Function to process a single card combo, returns feature array.
        
    Returns
    -------
    Tuple[int, np.ndarray]
        Tuple of (chunk_idx, results_array).  chunk_combos is NOT returned;
        the main process reconstructs it from all_combos to avoid sending
        ~700 KB of redundant data back through IPC on every chunk.
    """
    # Process first combo to determine output dimension, then pre-allocate.
    first = item_processor(chunk_combos[0])
    results = np.empty((len(chunk_combos), len(first)), dtype=np.float32)
    results[0] = first
    for i in range(1, len(chunk_combos)):
        results[i] = item_processor(chunk_combos[i])
    return chunk_idx, results


class ChunkedProcessor:
    """
    Handles chunked processing with checkpointing for memory-efficient clustering.
    
    This class manages:
    - Splitting data into chunks for processing
    - Saving chunk results to disk immediately after processing
    - Tracking progress via a JSON checkpoint file
    - Resuming from incomplete runs
    - Merging chunks into memory-mapped arrays for clustering
    
    Attributes
    ----------
    save_dir : Path
        Directory where all chunked data and checkpoints are saved.
    chunk_size : int
        Number of card combinations to process per chunk.
    checkpoint_path : Path
        Path to the JSON checkpoint file.
    """
    
    def __init__(
        self,
        save_dir: Path,
        chunk_size: int = 10000,
        use_compression: bool = True,
        storage_dtype: np.dtype = np.float16,
    ):
        """
        Initialize the ChunkedProcessor.
        
        Parameters
        ----------
        save_dir : Path
            Directory to save all intermediate files.
        chunk_size : int
            Number of combinations to process per chunk. Default 10000.
        use_compression : bool
            Whether to use gzip compression for chunk files. Reduces disk
            usage by 60-80% with ~10-20% slower I/O. Default True.
        storage_dtype : np.dtype
            Data type for storing feature arrays. float16 saves 50% disk
            space vs float32 with negligible precision loss for equity
            histograms. Default np.float16.
        """
        self.save_dir = Path(save_dir)
        self.chunk_size = chunk_size
        self.use_compression = use_compression
        self.storage_dtype = storage_dtype
        self.checkpoint_path = self.save_dir / "checkpoint.json"
        
        # Ensure directories exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize checkpoint
        self._checkpoint = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint from disk or return empty checkpoint."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                log.warning(f"Failed to load checkpoint, starting fresh: {e}")
                return self._empty_checkpoint()
        return self._empty_checkpoint()
    
    @staticmethod
    def _empty_checkpoint() -> Dict[str, Any]:
        """Return an empty checkpoint structure."""
        return {
            "streets": {
                "river": {
                    "completed_chunks": [],
                    "total_chunks": 0,
                    "merge_done": False,
                    "clustering_done": False,
                    "feature_dim": None,
                },
                "turn": {
                    "completed_chunks": [],
                    "total_chunks": 0,
                    "merge_done": False,
                    "clustering_done": False,
                    "feature_dim": None,
                },
                "flop": {
                    "completed_chunks": [],
                    "total_chunks": 0,
                    "merge_done": False,
                    "clustering_done": False,
                    "feature_dim": None,
                },
            },
            "config": {},
        }
    
    def _save_checkpoint(self, merge_with_disk: bool = True):
        """Save checkpoint to disk atomically with file locking.
        
        Parameters
        ----------
        merge_with_disk : bool
            If True, merge completed_chunks from disk with in-memory state.
            Set to False when you want to overwrite disk state completely.
        """
        # Use a lock file to prevent concurrent checkpoint writes
        lock_path = self.checkpoint_path.with_suffix(".lock")
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        
        try:
            # Acquire exclusive lock
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    # Re-read checkpoint to get latest state from other processes
                    if merge_with_disk and self.checkpoint_path.exists():
                        try:
                            with open(self.checkpoint_path, "r") as f:
                                disk_checkpoint = json.load(f)
                            # Merge completed_chunks from disk (union of both)
                            for street in self._checkpoint["streets"]:
                                disk_completed = set(disk_checkpoint.get("streets", {}).get(street, {}).get("completed_chunks", []))
                                memory_completed = set(self._checkpoint["streets"][street]["completed_chunks"])
                                merged = sorted(disk_completed | memory_completed)
                                self._checkpoint["streets"][street]["completed_chunks"] = merged
                        except (json.JSONDecodeError, KeyError):
                            pass  # Use in-memory checkpoint if disk version is corrupted
                    
                    # Write checkpoint
                    with open(temp_path, "w") as f:
                        json.dump(self._checkpoint, f, indent=2)
                    # Atomic rename
                    shutil.move(str(temp_path), str(self.checkpoint_path))
                finally:
                    # Release lock (happens automatically, but explicit is better)
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except IOError as e:
            log.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def get_street_dir(self, street: str) -> Path:
        """Get the directory for a specific street."""
        street_dir = self.save_dir / street
        street_dir.mkdir(parents=True, exist_ok=True)
        return street_dir
    
    def get_chunks_dir(self, street: str) -> Path:
        """Get the chunks directory for a specific street."""
        chunks_dir = self.get_street_dir(street) / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        return chunks_dir
    
    def get_combos_dir(self, street: str) -> Path:
        """Get the combos directory for a specific street."""
        combos_dir = self.get_street_dir(street) / "combos"
        combos_dir.mkdir(parents=True, exist_ok=True)
        return combos_dir
    
    def initialize_street(self, street: str, total_combos: int, config: Optional[Dict] = None):
        """
        Initialize processing for a street.
        
        Parameters
        ----------
        street : str
            The street name (river, turn, flop).
        total_combos : int
            Total number of card combinations for this street.
        config : Optional[Dict]
            Configuration to store in checkpoint.
        """
        n_chunks = (total_combos + self.chunk_size - 1) // self.chunk_size
        
        # Only reset if total_chunks changed (different configuration)
        if self._checkpoint["streets"][street]["total_chunks"] != n_chunks:
            log.info(f"Initializing {street} with {n_chunks} chunks ({total_combos} combos)")
            self._checkpoint["streets"][street] = {
                "completed_chunks": [],
                "total_chunks": n_chunks,
                "total_combos": total_combos,
                "merge_done": False,
                "clustering_done": False,
                "feature_dim": None,
            }
            # Clear old chunk files if configuration changed
            self._clear_street_data(street)
        else:
            # Always keep total_combos up to date (may be absent in old checkpoints)
            self._checkpoint["streets"][street]["total_combos"] = total_combos
        
        if config:
            self._checkpoint["config"] = config
        
        # Save without merging when clearing (to ensure old state is overwritten)
        self._save_checkpoint(merge_with_disk=False)
    
    def _clear_street_data(self, street: str):
        """Clear all data files for a street."""
        street_dir = self.get_street_dir(street)
        if street_dir.exists():
            for subdir in ["chunks", "combos"]:
                subdir_path = street_dir / subdir
                if subdir_path.exists():
                    shutil.rmtree(subdir_path)
            # Also remove any merged files
            for f in street_dir.glob("*.npy"):
                f.unlink()
            for f in street_dir.glob("*.dat"):
                f.unlink()
    
    def get_incomplete_chunks(self, street: str) -> List[int]:
        """
        Get list of chunk indices that still need processing.
        
        Parameters
        ----------
        street : str
            The street name.
            
        Returns
        -------
        List[int]
            List of chunk indices that need processing.
        """
        street_data = self._checkpoint["streets"][street]
        completed: Set[int] = set(street_data["completed_chunks"])
        total = street_data["total_chunks"]
        return [i for i in range(total) if i not in completed]
    
    def get_completed_chunks(self, street: str) -> List[int]:
        """Get list of completed chunk indices."""
        return list(self._checkpoint["streets"][street]["completed_chunks"])
    
    def is_merge_done(self, street: str) -> bool:
        """Check if merge is complete for a street."""
        return self._checkpoint["streets"][street].get("merge_done", False)
    
    def mark_merge_done(self, street: str):
        """Mark merge as complete for a street."""
        self._checkpoint["streets"][street]["merge_done"] = True
        self._save_checkpoint()
    
    def is_clustering_done(self, street: str) -> bool:
        """Check if clustering is complete for a street."""
        return self._checkpoint["streets"][street].get("clustering_done", False)
    
    def get_chunk_path(self, street: str, chunk_idx: int) -> Path:
        """Get the file path for a chunk's data."""
        ext = ".npy.gz" if self.use_compression else ".npy"
        return self.get_chunks_dir(street) / f"chunk_{chunk_idx:06d}{ext}"
    
    def get_combos_path(self, street: str, chunk_idx: int) -> Path:
        """Get the file path for a chunk's card combos."""
        return self.get_combos_dir(street) / f"combos_{chunk_idx:06d}.npy"
    
    def save_chunk(
        self,
        street: str,
        chunk_idx: int,
        data: np.ndarray,
        combos: np.ndarray,
    ):
        """
        Save a processed chunk to disk.
        
        Parameters
        ----------
        street : str
            The street name.
        chunk_idx : int
            The chunk index.
        data : np.ndarray
            The computed EHS/distribution data for this chunk.
        combos : np.ndarray
            The card combinations for this chunk.
        """
        # Save data atomically using temp files
        chunk_path = self.get_chunk_path(street, chunk_idx)
        combos_path = self.get_combos_path(street, chunk_idx)
        
        # Determine temp file extensions
        temp_ext = ".tmp.npy.gz" if self.use_compression else ".tmp.npy"
        temp_chunk = chunk_path.parent / f"chunk_{chunk_idx:06d}{temp_ext}"
        temp_combos = combos_path.with_suffix(".tmp.npy")
        
        try:
            # Convert to storage dtype (float16 saves 50% disk space)
            data_to_save = data.astype(self.storage_dtype)
            
            if self.use_compression:
                # Save with gzip compression (saves additional 60-80%)
                with gzip.open(temp_chunk, 'wb', compresslevel=4) as f:
                    np.save(f, data_to_save)
            else:
                np.save(temp_chunk, data_to_save)
            
            # Combos are small integers, no need to compress
            np.save(temp_combos, combos)
            
            # Atomic rename
            shutil.move(str(temp_chunk), str(chunk_path))
            shutil.move(str(temp_combos), str(combos_path))
        except Exception as e:
            # Clean up temp files on failure
            for f in [temp_chunk, temp_combos]:
                if f.exists():
                    f.unlink()
            raise RuntimeError(f"Failed to save chunk {chunk_idx}: {e}")
    
    def mark_chunk_complete(self, street: str, chunk_idx: int):
        """
        Mark a chunk as completed in the checkpoint.
        
        Parameters
        ----------
        street : str
            The street name.
        chunk_idx : int
            The chunk index.
        """
        completed = self._checkpoint["streets"][street]["completed_chunks"]
        if chunk_idx not in completed:
            completed.append(chunk_idx)
            completed.sort()
        self._save_checkpoint()
    
    def mark_clustering_done(self, street: str):
        """Mark clustering as complete for a street."""
        self._checkpoint["streets"][street]["clustering_done"] = True
        self._save_checkpoint()
    
    def process_chunks_parallel(
        self,
        street: str,
        chunk_indices: List[int],
        all_combos: np.ndarray,
        item_processor: Callable[[np.ndarray], np.ndarray],
        workers: int,
    ):
        """
        Process multiple chunks in parallel, one chunk per worker.
        
        Each worker processes all items in its assigned chunk sequentially.
        This approach minimizes pickling overhead (each chunk's combos are sent
        once per worker) and allows workers to maintain local caches efficiently.
        
        Results are saved to disk as each chunk completes, enabling resumability.
        
        Parameters
        ----------
        street : str
            The street name (river, turn, flop).
        chunk_indices : List[int]
            List of chunk indices to process.
        all_combos : np.ndarray
            Full array of all card combinations for this street.
        item_processor : Callable
            Function to process a single card combo. Must be picklable.
            Signature: (combo: np.ndarray) -> np.ndarray
        workers : int
            Number of worker processes to use.
        """
        if not chunk_indices:
            log.info(f"No chunks to process for {street}")
            return
        
        chunk_specs = self.get_chunk_indices(len(all_combos))
        total_chunks = len(chunk_specs)
        
        log.info(f"Processing {len(chunk_indices)} chunks with {workers} workers (1 chunk per worker)...")
        start_time = time.time()
        
        # Track completed count for progress updates
        completed_count = len(self.get_completed_chunks(street))
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all chunks - each worker gets one chunk
            futures = {}
            for chunk_idx in chunk_indices:
                _, start_idx, end_idx = chunk_specs[chunk_idx]
                chunk_combos = all_combos[start_idx:end_idx]
                
                future = executor.submit(
                    _process_single_chunk_worker,
                    chunk_idx,
                    chunk_combos,
                    item_processor,
                )
                futures[future] = chunk_idx
            
            # Process results as they complete
            failed_chunks = []
            for future in concurrent.futures.as_completed(futures):
                chunk_idx = futures[future]
                try:
                    result_idx, results = future.result()
                    
                    # Reconstruct combos from all_combos — avoids shipping
                    # ~700 KB back from the worker on every chunk.
                    _, start_idx, end_idx = chunk_specs[result_idx]
                    combos = all_combos[start_idx:end_idx]

                    # Save chunk to disk
                    self.save_chunk(street, result_idx, results, combos)
                    self.mark_chunk_complete(street, result_idx)
                    completed_count += 1
                    
                    # Progress update every 60 chunks (or on first and last)
                    is_last = completed_count == total_chunks
                    if completed_count % 60 == 0 or completed_count == 1 or is_last:
                        elapsed = time.time() - start_time
                        rate = completed_count / elapsed if elapsed > 0 else 0
                        remaining = (total_chunks - completed_count) / rate if rate > 0 else 0
                        log.info(
                            f"[{street.upper()}] Chunk {completed_count}/{total_chunks} | "
                            f"Elapsed: {elapsed/60:.1f}m | "
                            f"ETA: {remaining/60:.1f}m | "
                            f"Rate: {rate*60:.1f} chunks/min"
                        )
                    
                except Exception as e:
                    log.error(f"Chunk {chunk_idx} failed: {e}")
                    failed_chunks.append(chunk_idx)
                    # Continue processing remaining futures — do NOT raise here.
                    # Failed chunks stay absent from the checkpoint so they are
                    # automatically retried on the next run.

        if failed_chunks:
            log.warning(
                f"{len(failed_chunks)} chunk(s) failed for {street} and will be "
                f"retried on the next run: {failed_chunks[:20]}"
                + (" ..." if len(failed_chunks) > 20 else "")
            )
    
    def load_chunk(self, street: str, chunk_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a chunk's data and combos from disk.
        
        Parameters
        ----------
        street : str
            The street name.
        chunk_idx : int
            The chunk index.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (data, combos). Data is converted to float32 for computation.
        """
        chunk_path = self.get_chunk_path(street, chunk_idx)
        combos_path = self.get_combos_path(street, chunk_idx)
        
        # Load data (handle gzip if compressed)
        if self.use_compression:
            with gzip.open(chunk_path, 'rb') as f:
                data = np.load(f, allow_pickle=True)
        else:
            data = np.load(chunk_path, allow_pickle=True)
        
        # Convert back to float32 for computation accuracy
        data = data.astype(np.float32)
        
        combos = np.load(combos_path, allow_pickle=True)
        return data, combos
    
    def merge_chunks_to_memmap(
        self,
        street: str,
        dtype: np.dtype = np.float32,
    ) -> Tuple[np.memmap, np.ndarray]:
        """
        Merge all chunk files into a single memory-mapped array.
        
        Parameters
        ----------
        street : str
            The street name.
        dtype : np.dtype
            Data type for the memory-mapped file.
            
        Returns
        -------
        Tuple[np.memmap, np.ndarray]
            Tuple of (data_memmap, all_combos).
        """
        log.info(f"Merging chunks for {street}...")
        
        completed_chunks = sorted(self.get_completed_chunks(street))
        if not completed_chunks:
            raise ValueError(f"No completed chunks found for {street}")

        # Verify all chunks are present before merging — merging partial data
        # would silently produce a card_info_lut with missing entries.
        total_chunks = self._checkpoint["streets"][street]["total_chunks"]
        if len(completed_chunks) != total_chunks:
            missing = sorted(set(range(total_chunks)) - set(completed_chunks))
            raise ValueError(
                f"Cannot merge {street}: {len(missing)} chunk(s) still incomplete "
                f"(indices {missing[:20]}{' ...' if len(missing) > 20 else ''}). "
                f"Re-run to process missing chunks."
            )

        # Compute total_rows and feature_dim without loading any files.
        # total_combos is stored in the checkpoint during initialize_street.
        total_combos = self._checkpoint["streets"][street].get("total_combos")
        if total_combos is not None:
            # Each chunk i covers rows [i*chunk_size, min((i+1)*chunk_size, total_combos))
            total_rows = sum(
                min((idx + 1) * self.chunk_size, total_combos) - idx * self.chunk_size
                for idx in completed_chunks
            )
            # Load only the first chunk to determine feature_dim
            first_chunk_data, _ = self.load_chunk(street, completed_chunks[0])
            feature_dim = first_chunk_data.shape[1] if first_chunk_data.ndim > 1 else 1
        else:
            # Fallback for old checkpoints: load every chunk once to measure sizes
            log.warning(
                f"total_combos not in checkpoint for {street}; "
                "falling back to full-scan for row count (slower)."
            )
            first_chunk_data, _ = self.load_chunk(street, completed_chunks[0])
            feature_dim = first_chunk_data.shape[1] if first_chunk_data.ndim > 1 else 1
            total_rows = first_chunk_data.shape[0]
            for chunk_idx in completed_chunks[1:]:
                chunk_data, _ = self.load_chunk(street, chunk_idx)
                total_rows += chunk_data.shape[0]
        
        log.info(f"Total rows for {street}: {total_rows}, feature_dim: {feature_dim}")
        
        # Store feature_dim in checkpoint for later use (after cleanup)
        self._checkpoint["streets"][street]["feature_dim"] = int(feature_dim)
        self._save_checkpoint()
        
        # Create memory-mapped file for merged data
        merged_path = self.get_street_dir(street) / "merged_data.dat"
        merged_data = np.memmap(
            merged_path,
            dtype=dtype,
            mode='w+',
            shape=(total_rows, feature_dim),
        )
        
        # Collect all combos (these are smaller, can fit in memory)
        all_combos = []
        
        # Second pass: copy data to memory-mapped file
        current_row = 0
        flush_interval = 10  # Flush every 10 chunks for safety
        for i, chunk_idx in enumerate(completed_chunks):
            chunk_data, chunk_combos = self.load_chunk(street, chunk_idx)
            n_rows = chunk_data.shape[0]
            
            if chunk_data.ndim == 1:
                chunk_data = chunk_data.reshape(-1, 1)
            
            merged_data[current_row:current_row + n_rows] = chunk_data
            all_combos.append(chunk_combos)
            current_row += n_rows
            
            # Periodic flush to ensure progress is saved to disk
            if (i + 1) % flush_interval == 0:
                merged_data.flush()
        
        # Final flush to disk
        merged_data.flush()
        
        # Concatenate combos and save atomically
        all_combos = np.concatenate(all_combos, axis=0)
        combos_path = self.get_street_dir(street) / "all_combos.npy"
        temp_combos = combos_path.with_suffix(".tmp.npy")
        try:
            np.save(temp_combos, all_combos)
            shutil.move(str(temp_combos), str(combos_path))
        except Exception as e:
            if temp_combos.exists():
                temp_combos.unlink()
            raise RuntimeError(f"Failed to save combos: {e}")
        
        log.info(f"Merged {len(completed_chunks)} chunks into memory-mapped file")
        
        return merged_data, all_combos
    
    def load_merged_data(
        self,
        street: str,
        dtype: np.dtype = np.float32,
    ) -> Tuple[np.memmap, np.ndarray]:
        """
        Load previously merged data from disk.
        
        Parameters
        ----------
        street : str
            The street name.
        dtype : np.dtype
            Data type for the memory-mapped file.
            
        Returns
        -------
        Tuple[np.memmap, np.ndarray]
            Tuple of (data_memmap, all_combos).
        """
        merged_path = self.get_street_dir(street) / "merged_data.dat"
        combos_path = self.get_street_dir(street) / "all_combos.npy"
        
        if not merged_path.exists() or not combos_path.exists():
            raise FileNotFoundError(f"Merged data not found for {street}")
        
        # Load combos to get shape
        all_combos = np.load(combos_path, allow_pickle=True)
        
        # Get feature dimension from checkpoint (stored during merge)
        feature_dim = self._checkpoint["streets"].get(street, {}).get("feature_dim")
        
        if feature_dim is None:
            # Fallback: try to determine from first chunk if still available
            completed_chunks = sorted(self.get_completed_chunks(street))
            if not completed_chunks:
                raise ValueError(f"No completed chunks found for {street}")
            try:
                first_chunk_data, _ = self.load_chunk(street, completed_chunks[0])
                feature_dim = first_chunk_data.shape[1] if first_chunk_data.ndim > 1 else 1
                # Store for next time
                self._checkpoint["streets"][street]["feature_dim"] = int(feature_dim)
                self._save_checkpoint()
            except FileNotFoundError:
                raise ValueError(
                    f"Cannot determine feature dimension for {street}: "
                    f"chunks cleaned up and feature_dim not in checkpoint"
                )
        
        # Load memory-mapped file
        total_rows = len(all_combos)
        merged_data = np.memmap(
            merged_path,
            dtype=dtype,
            mode='r',
            shape=(total_rows, feature_dim),
        )
        
        log.info(f"Loaded merged data for {street}: {total_rows} rows")
        return merged_data, all_combos
    
    def get_or_merge_data(
        self,
        street: str,
        dtype: np.dtype = np.float32,
    ) -> Tuple[np.memmap, np.ndarray]:
        """
        Get merged data, either by loading from disk or merging chunks.
        
        This method checks if merge is complete. If yes, loads existing data.
        If no, merges chunks and marks merge as complete.
        
        Parameters
        ----------
        street : str
            The street name.
        dtype : np.dtype
            Data type for the memory-mapped file.
            
        Returns
        -------
        Tuple[np.memmap, np.ndarray]
            Tuple of (data_memmap, all_combos).
        """
        if self.is_merge_done(street):
            try:
                log.info(f"Merge already complete for {street}, loading from disk...")
                return self.load_merged_data(street, dtype)
            except (FileNotFoundError, ValueError) as e:
                log.warning(f"Failed to load merged data: {e}. Re-merging from chunks...")
                # Fall through to merge
        
        # Merge from chunks
        merged_data, all_combos = self.merge_chunks_to_memmap(street, dtype)
        self.mark_merge_done(street)
        return merged_data, all_combos
    
    def get_chunk_indices(self, total_combos: int) -> List[Tuple[int, int, int]]:
        """
        Get list of (chunk_idx, start_idx, end_idx) tuples for all chunks.
        
        Parameters
        ----------
        total_combos : int
            Total number of combinations.
            
        Returns
        -------
        List[Tuple[int, int, int]]
            List of (chunk_idx, start_idx, end_idx) tuples.
        """
        n_chunks = (total_combos + self.chunk_size - 1) // self.chunk_size
        indices = []
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_combos)
            indices.append((chunk_idx, start_idx, end_idx))
        return indices
    
    def save_centroids(self, street: str, centroids: np.ndarray):
        """Save centroids for a street atomically."""
        centroids_path = self.get_street_dir(street) / "centroids.npy"
        temp_path = centroids_path.with_suffix(".tmp.npy")
        try:
            np.save(temp_path, centroids)
            shutil.move(str(temp_path), str(centroids_path))
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to save centroids: {e}")
    
    def load_centroids(self, street: str) -> np.ndarray:
        """Load centroids for a street."""
        centroids_path = self.get_street_dir(street) / "centroids.npy"
        return np.load(centroids_path, allow_pickle=True)
    
    def save_clusters(self, street: str, clusters: np.ndarray):
        """Save cluster assignments for a street atomically."""
        clusters_path = self.get_street_dir(street) / "clusters.npy"
        temp_path = clusters_path.with_suffix(".tmp.npy")
        try:
            np.save(temp_path, clusters)
            shutil.move(str(temp_path), str(clusters_path))
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to save clusters: {e}")
    
    def load_clusters(self, street: str) -> np.ndarray:
        """Load cluster assignments for a street."""
        clusters_path = self.get_street_dir(street) / "clusters.npy"
        return np.load(clusters_path, allow_pickle=True)
    
    def cleanup_chunks(self, street: str):
        """
        Remove chunk files after successful merge and clustering.
        
        Only call this after clustering is complete to free up disk space.
        Keeps merged_data.dat, lookup_index.npy, centroids.npy, clusters.npy.
        """
        log.info(f"Cleaning up chunk files for {street}...")
        chunks_dir = self.get_chunks_dir(street)
        combos_dir = self.get_combos_dir(street)
        
        deleted_size = 0
        for d in [chunks_dir, combos_dir]:
            if d.exists():
                # Calculate size before deletion for logging
                for file in d.rglob('*'):
                    if file.is_file():
                        deleted_size += file.stat().st_size
                shutil.rmtree(d)
        
        if deleted_size > 0:
            log.info(f"Freed {deleted_size / (1024**2):.1f} MB by removing {street} chunk files")
    
    def cleanup_partial_clustering(self, street: str):
        """
        Remove partial clustering checkpoint files.
        
        Called after successful clustering completion.
        """
        street_dir = self.get_street_dir(street)
        partial_km_path = street_dir / "partial_kmeans.joblib"
        progress_path = street_dir / "kmeans_progress.json"
        
        deleted = []
        for p in [partial_km_path, progress_path]:
            if p.exists():
                p.unlink()
                deleted.append(p.name)
        
        if deleted:
            log.info(f"Cleaned up partial clustering files for {street}: {', '.join(deleted)}")
    
    def cleanup_all_intermediate_files(self):
        """
        Remove all intermediate files for all streets.
        
        Keeps only the final results:
        - merged_data.dat (for lookups)
        - lookup_index.npy (for lookups) 
        - centroids.npy (final results)
        - clusters.npy (final results)
        - checkpoint.json (state tracking)
        
        Call this after all processing is complete to minimize disk usage.
        """
        log.info("Cleaning up all intermediate files...")
        total_freed = 0
        
        for street in ["river", "turn", "flop"]:
            if not self.is_clustering_done(street):
                log.warning(f"Skipping cleanup for {street} - clustering not complete")
                continue
            
            # Track size before cleanup
            street_dir = self.get_street_dir(street)
            if street_dir.exists():
                initial_size = sum(f.stat().st_size for f in street_dir.rglob('*') if f.is_file())
                
                self.cleanup_chunks(street)
                self.cleanup_partial_clustering(street)
                
                final_size = sum(f.stat().st_size for f in street_dir.rglob('*') if f.is_file())
                freed = initial_size - final_size
                total_freed += freed
        
        if total_freed > 0:
            log.info(f"Total disk space freed: {total_freed / (1024**2):.1f} MB")

   