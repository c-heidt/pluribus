"""
Unit tests for the ChunkedProcessor class.

Tests memory-efficient chunked processing with checkpointing functionality
for the card clustering module.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from poker_ai.clustering.chunked_processor import ChunkedProcessor


class TestChunkedProcessorBasic:
    """Basic ChunkedProcessor tests."""
    
    def test_initialization(self):
        """Test ChunkedProcessor initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            
            assert processor.save_dir.exists()
            assert processor.chunk_size == 100
    
    def test_initialize_street(self):
        """Test street initialization creates checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            processor.initialize_street("river", 500)
            
            assert processor.checkpoint_path.exists()
            with open(processor.checkpoint_path) as f:
                checkpoint = json.load(f)
            assert checkpoint["streets"]["river"]["total_chunks"] == 5
            assert checkpoint["streets"]["river"]["clustering_done"] is False


class TestChunkedProcessorChunks:
    """Tests for chunk operations."""
    
    def test_chunk_indices(self):
        """Test chunk index calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            
            indices = processor.get_chunk_indices(350)
            assert len(indices) == 4
            assert indices[0] == (0, 0, 100)
            assert indices[1] == (1, 100, 200)
            assert indices[2] == (2, 200, 300)
            assert indices[3] == (3, 300, 350)
    
    def test_incomplete_chunks(self):
        """Test incomplete chunk tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            processor.initialize_street("river", 300)
            
            incomplete = processor.get_incomplete_chunks("river")
            assert len(incomplete) == 3
            assert set(incomplete) == {0, 1, 2}
    
    def test_save_and_load_chunk(self):
        """Test saving and loading chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            processor.initialize_street("river", 200)
            
            data = np.random.rand(100, 3).astype(np.float32)
            combos = np.array([[i, i+1, i+2] for i in range(100)])
            
            processor.save_chunk("river", 0, data, combos)
            processor.mark_chunk_complete("river", 0)
            
            loaded_data, loaded_combos = processor.load_chunk("river", 0)
            # Use atol=1e-3 because default storage_dtype is float16
            assert np.allclose(loaded_data, data, atol=1e-3)
            assert np.array_equal(loaded_combos, combos)
    
    def test_mark_chunk_complete(self):
        """Test marking chunks as complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            processor.initialize_street("river", 200)
            
            data = np.random.rand(100, 3).astype(np.float32)
            combos = np.array([[i, i+1] for i in range(100)])
            
            processor.save_chunk("river", 0, data, combos)
            processor.mark_chunk_complete("river", 0)
            
            incomplete = processor.get_incomplete_chunks("river")
            assert 0 not in incomplete
            assert len(incomplete) == 1


class TestChunkedProcessorMerge:
    """Tests for merging chunks."""
    
    def test_merge_chunks(self):
        """Test merging multiple chunks to memory-mapped file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=50)
            processor.initialize_street("river", 150)
            
            all_data = []
            for chunk_idx in range(3):
                data = np.random.rand(50, 3).astype(np.float32)
                combos = np.array([[chunk_idx * 50 + i, i+1] for i in range(50)])
                all_data.append(data)
                
                processor.save_chunk("river", chunk_idx, data, combos)
                processor.mark_chunk_complete("river", chunk_idx)
            
            merged_data, merged_combos = processor.merge_chunks_to_memmap("river")
            
            expected_data = np.concatenate(all_data, axis=0)
            assert merged_data.shape == (150, 3)
            assert len(merged_combos) == 150
            # Use atol=1e-3 because default storage_dtype is float16
            assert np.allclose(merged_data, expected_data, atol=1e-3)


class TestChunkedProcessorResume:
    """Tests for checkpoint resume functionality."""
    
    def test_resume_from_checkpoint(self):
        """Test resuming from an existing checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run - partial completion
            processor1 = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            processor1.initialize_street("river", 300)
            
            data = np.random.rand(100, 3).astype(np.float32)
            combos = np.array([[i, i+1] for i in range(100)])
            processor1.save_chunk("river", 0, data, combos)
            processor1.mark_chunk_complete("river", 0)
            
            # Second run - resume
            processor2 = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            
            incomplete = processor2.get_incomplete_chunks("river")
            assert len(incomplete) == 2
            assert 0 not in incomplete
    
    def test_clustering_done_marker(self):
        """Test clustering completion marker persists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            processor.initialize_street("river", 100)
            
            assert not processor.is_clustering_done("river")
            processor.mark_clustering_done("river")
            assert processor.is_clustering_done("river")
            
            # Reload and verify
            processor2 = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            assert processor2.is_clustering_done("river")


class TestChunkedProcessorCentroids:
    """Tests for centroid handling."""
    
    def test_save_and_load_centroids(self):
        """Test saving and loading centroids."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            processor.initialize_street("river", 100)
            
            centroids = np.random.rand(5, 3).astype(np.float32)
            processor.save_centroids("river", centroids)
            
            loaded = processor.load_centroids("river")
            assert np.allclose(loaded, centroids)
    
    def test_save_and_load_clusters(self):
        """Test saving and loading cluster assignments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            processor.initialize_street("river", 100)
            
            clusters = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
            processor.save_clusters("river", clusters)
            
            loaded = processor.load_clusters("river")
            assert np.array_equal(loaded, clusters)


class TestChunkedProcessorEdgeCases:
    """Tests for edge cases."""
    
    def test_single_item(self):
        """Test with single item."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            processor.initialize_street("river", 1)
            
            indices = processor.get_chunk_indices(1)
            assert len(indices) == 1
            assert indices[0] == (0, 0, 1)
    
    def test_empty(self):
        """Test with empty data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            
            indices = processor.get_chunk_indices(0)
            assert len(indices) == 0
    
    def test_exact_chunk_size(self):
        """Test with data exactly matching chunk size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            processor.initialize_street("river", 100)
            
            indices = processor.get_chunk_indices(100)
            assert len(indices) == 1
            assert indices[0] == (0, 0, 100)
    
    def test_config_change_clears_data(self):
        """Test that changing configuration clears existing data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run with chunk_size=100
            processor1 = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            processor1.initialize_street("river", 300)
            
            # Save a chunk
            data = np.random.rand(100, 3).astype(np.float32)
            combos = np.array([[i, i+1] for i in range(100)])
            processor1.save_chunk("river", 0, data, combos)
            processor1.mark_chunk_complete("river", 0)
            
            # Second run with different total_combos (simulating config change)
            processor2 = ChunkedProcessor(save_dir=Path(tmpdir), chunk_size=100)
            processor2.initialize_street("river", 500)  # Different total
            
            # Should have all chunks incomplete (data cleared)
            incomplete = processor2.get_incomplete_chunks("river")
            assert len(incomplete) == 5  # 500 / 100 = 5 chunks
