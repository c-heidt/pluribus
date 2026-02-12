"""
Chunked processing with checkpointing for memory-efficient clustering.

This module provides utilities for processing large datasets in chunks,
saving intermediate results to disk, and resuming from checkpoints.
"""
import fcntl
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger("poker_ai.clustering.chunked_processor")


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
    
    def __init__(self, save_dir: Path, chunk_size: int = 10000):
        """
        Initialize the ChunkedProcessor.
        
        Parameters
        ----------
        save_dir : Path
            Directory to save all intermediate files.
        chunk_size : int
            Number of combinations to process per chunk. Default 10000.
        """
        self.save_dir = Path(save_dir)
        self.chunk_size = chunk_size
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
                "river": {"completed_chunks": [], "total_chunks": 0, "clustering_done": False},
                "turn": {"completed_chunks": [], "total_chunks": 0, "clustering_done": False},
                "flop": {"completed_chunks": [], "total_chunks": 0, "clustering_done": False},
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
        finally:
            # Clean up lock file
            if lock_path.exists():
                try:
                    lock_path.unlink()
                except OSError:
                    pass
    
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
                "clustering_done": False,
            }
            # Clear old chunk files if configuration changed
            self._clear_street_data(street)
        
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
    
    def is_clustering_done(self, street: str) -> bool:
        """Check if clustering is complete for a street."""
        return self._checkpoint["streets"][street]["clustering_done"]
    
    def get_chunk_path(self, street: str, chunk_idx: int) -> Path:
        """Get the file path for a chunk's data."""
        return self.get_chunks_dir(street) / f"chunk_{chunk_idx:06d}.npy"
    
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
        
        temp_chunk = chunk_path.with_suffix(".tmp.npy")
        temp_combos = combos_path.with_suffix(".tmp.npy")
        
        try:
            np.save(temp_chunk, data.astype(np.float32))
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
            Tuple of (data, combos).
        """
        chunk_path = self.get_chunk_path(street, chunk_idx)
        combos_path = self.get_combos_path(street, chunk_idx)
        
        return np.load(chunk_path), np.load(combos_path, allow_pickle=True)
    
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
        
        # First pass: determine total size and feature dimension
        first_chunk_data, first_chunk_combos = self.load_chunk(street, completed_chunks[0])
        feature_dim = first_chunk_data.shape[1] if first_chunk_data.ndim > 1 else 1
        
        total_rows = 0
        for chunk_idx in completed_chunks:
            chunk_path = self.get_chunk_path(street, chunk_idx)
            # Use memory-mapping to just get the shape without loading
            chunk_data = np.load(chunk_path, mmap_mode='r')
            total_rows += chunk_data.shape[0]
        
        log.info(f"Total rows for {street}: {total_rows}, feature_dim: {feature_dim}")
        
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
        for chunk_idx in completed_chunks:
            chunk_data, chunk_combos = self.load_chunk(street, chunk_idx)
            n_rows = chunk_data.shape[0]
            
            if chunk_data.ndim == 1:
                chunk_data = chunk_data.reshape(-1, 1)
            
            merged_data[current_row:current_row + n_rows] = chunk_data
            all_combos.append(chunk_combos)
            current_row += n_rows
        
        # Flush to disk
        merged_data.flush()
        
        # Concatenate combos
        all_combos = np.concatenate(all_combos, axis=0)
        
        log.info(f"Merged {len(completed_chunks)} chunks into memory-mapped file")
        
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
        """Save centroids for a street."""
        centroids_path = self.get_street_dir(street) / "centroids.npy"
        np.save(centroids_path, centroids)
    
    def load_centroids(self, street: str) -> np.ndarray:
        """Load centroids for a street."""
        centroids_path = self.get_street_dir(street) / "centroids.npy"
        return np.load(centroids_path)
    
    def save_clusters(self, street: str, clusters: np.ndarray):
        """Save cluster assignments for a street."""
        clusters_path = self.get_street_dir(street) / "clusters.npy"
        np.save(clusters_path, clusters)
    
    def load_clusters(self, street: str) -> np.ndarray:
        """Load cluster assignments for a street."""
        clusters_path = self.get_street_dir(street) / "clusters.npy"
        return np.load(clusters_path)
    
    def cleanup_chunks(self, street: str):
        """
        Remove chunk files after successful merge and clustering.
        
        Only call this after clustering is complete to free up disk space.
        """
        log.info(f"Cleaning up chunk files for {street}")
        chunks_dir = self.get_chunks_dir(street)
        combos_dir = self.get_combos_dir(street)
        
        for d in [chunks_dir, combos_dir]:
            if d.exists():
                shutil.rmtree(d)
