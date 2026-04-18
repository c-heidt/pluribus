"""Unit tests for ChunkStore: chunk I/O, merge, cleanup."""
import numpy as np
import pytest

from information_abstraction.build.chunk_store import (
    ChunkStore,
    CorruptChunkError,
)


def _store(tmp_path, **kwargs):
    return ChunkStore(
        save_dir=tmp_path,
        chunk_size=kwargs.pop("chunk_size", 4),
        use_compression=kwargs.pop("use_compression", False),
    )


class TestChunkRoundTrip:
    def test_save_then_load_preserves_values(self, tmp_path):
        store = _store(tmp_path)
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        store.initialize_street("river", total_combos=12)
        store.save_chunk("river", 1, data)
        loaded, combos = store.load_chunk("river", 1)
        assert combos is None  # no per-chunk combo file written any more
        np.testing.assert_array_almost_equal(loaded, data, decimal=3)
        assert loaded.dtype == np.float32

    def test_truncated_chunk_raises_on_save(self, tmp_path, monkeypatch):
        """save_chunk verifies round-trip and surfaces silent truncation."""
        store = _store(tmp_path)
        store.initialize_street("river", total_combos=4)

        real_np_save = np.save

        def bogus_save(file, arr, *a, **kw):
            # Write half the bytes.
            real_np_save(file, arr[: len(arr) // 2], *a, **kw)

        monkeypatch.setattr(np, "save", bogus_save)
        data = np.ones((4, 2), dtype=np.float32)
        with pytest.raises(RuntimeError, match="Verification failed"):
            store.save_chunk("river", 0, data)


class TestMergeAndCleanup:
    def test_merge_produces_concatenated_memmap(self, tmp_path):
        chunk_size = 4
        n_combos = 10
        n_features = 3
        store = ChunkStore(
            save_dir=tmp_path, chunk_size=chunk_size, use_compression=False,
        )
        store.initialize_street("river", total_combos=n_combos)

        all_combos = np.arange(n_combos * 7, dtype=np.int32).reshape(-1, 7)
        chunk_specs = store.get_chunk_indices(n_combos)
        for idx, start, end in chunk_specs:
            # Deterministic feature per row so we can assert on merge order.
            rows = np.tile(
                np.array([idx, start, end], dtype=np.float32),
                (end - start, 1),
            )
            store.save_chunk("river", idx, rows)
            store.checkpoint.mark_chunk_complete("river", idx)

        merged, combos = store.get_or_merge_data(
            "river", all_combos_full=all_combos,
        )
        assert merged.shape == (n_combos, n_features)
        assert combos.shape == (n_combos, 7)
        # Combos recover the original rows in the original order.
        np.testing.assert_array_equal(combos, all_combos)
        # Merge marks the flag and persists feature_dim.
        assert store.checkpoint.is_merge_done("river")
        assert store.checkpoint.get_feature_dim("river") == n_features

    def test_merge_refuses_partial_data(self, tmp_path):
        store = _store(tmp_path, chunk_size=4)
        store.initialize_street("river", total_combos=10)
        # Only write chunk 0 of 3 — refuse to merge.
        store.save_chunk("river", 0, np.ones((4, 2), dtype=np.float32))
        store.checkpoint.mark_chunk_complete("river", 0)
        with pytest.raises(ValueError, match="still incomplete"):
            store.merge_chunks_to_memmap(
                "river", all_combos_full=np.zeros((10, 7), dtype=np.int32),
            )

    def test_corrupt_chunk_detected_and_unmarked(self, tmp_path):
        chunk_size = 4
        n_combos = 8
        store = ChunkStore(
            save_dir=tmp_path, chunk_size=chunk_size, use_compression=False,
        )
        store.initialize_street("river", total_combos=n_combos)

        # Write two valid chunks.
        for idx in (0, 1):
            rows = np.ones((chunk_size, 2), dtype=np.float32) * idx
            store.save_chunk("river", idx, rows)
            store.checkpoint.mark_chunk_complete("river", idx)

        # Corrupt chunk 1: replace file with garbage bytes.
        chunk_path = store.get_chunk_path("river", 1)
        chunk_path.write_bytes(b"not a valid npy file")

        with pytest.raises(CorruptChunkError) as excinfo:
            store.merge_chunks_to_memmap(
                "river",
                all_combos_full=np.zeros((n_combos, 7), dtype=np.int32),
            )
        assert 1 in excinfo.value.corrupt_indices
        # Chunk 1 was unmarked and deleted so a retry only reprocesses it.
        assert 1 in store.checkpoint.get_incomplete_chunks("river")
        assert not store.get_chunk_path("river", 1).exists()

    def test_cleanup_chunks_removes_chunk_dirs(self, tmp_path):
        store = _store(tmp_path, chunk_size=4)
        store.initialize_street("river", total_combos=4)
        store.save_chunk(
            "river", 0, np.ones((4, 2), dtype=np.float32),
        )
        chunks_dir = store.get_chunks_dir("river")
        assert list(chunks_dir.iterdir())
        store.cleanup_chunks("river")
        assert not chunks_dir.exists()


class TestCentroidsAndClusters:
    def test_centroids_round_trip(self, tmp_path):
        store = _store(tmp_path)
        store.initialize_street("river", total_combos=4)
        centroids = np.random.rand(3, 5).astype(np.float32)
        store.save_centroids("river", centroids)
        loaded = store.load_centroids("river")
        np.testing.assert_array_equal(loaded, centroids)

    def test_clusters_round_trip(self, tmp_path):
        store = _store(tmp_path)
        store.initialize_street("river", total_combos=4)
        clusters = np.array([0, 1, 2, 1], dtype=np.int32)
        store.save_clusters("river", clusters)
        loaded = store.load_clusters("river")
        np.testing.assert_array_equal(loaded, clusters)
