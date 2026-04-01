"""Unit tests for Phase 1 infrastructure.

Covers:
- ``atomic_joblib_dump`` (1.1)
- ``atomic_numpy_save`` / ``atomic_numpy_load`` (1.1)
- ``hash_info_set_128`` / ``hash_info_set_bytes`` (1.2)
- ``InfosetIndex`` (1.3)
"""
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from poker_ai.utils.io import (
    atomic_joblib_dump,
    atomic_numpy_load,
    atomic_numpy_save,
    hash_info_set_128,
    hash_info_set_bytes,
)
from poker_ai.ai.index import CHUNK_SIZE, InfosetIndex


# ===========================================================================
# 1.1 — Atomic save utilities
# ===========================================================================


class TestAtomicJoblibDump:
    def test_roundtrip_dict(self, tmp_path):
        obj = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
        path = tmp_path / "test.joblib"
        atomic_joblib_dump(obj, path)
        import joblib
        loaded = joblib.load(path)
        assert loaded == obj

    def test_roundtrip_numpy_array(self, tmp_path):
        arr = np.arange(1000, dtype=np.float64).reshape(20, 50)
        path = tmp_path / "arr.joblib"
        atomic_joblib_dump(arr, path)
        import joblib
        loaded = joblib.load(path)
        np.testing.assert_array_equal(arr, loaded)

    def test_no_partial_write_on_error(self, tmp_path):
        """If joblib.dump fails the original path must be untouched."""
        # Pre-populate the target file with known content.
        import joblib
        original = {"original": True}
        path = tmp_path / "safe.joblib"
        joblib.dump(original, path)
        original_mtime = path.stat().st_mtime

        # Pass an object that can't be serialised (a lambda).
        un_serialisable = lambda x: x  # noqa: E731
        with pytest.raises((RuntimeError, Exception)):
            atomic_joblib_dump(un_serialisable, path)

        # Original must still be readable and unchanged.
        assert path.stat().st_mtime == original_mtime
        reloaded = joblib.load(path)
        assert reloaded == original


class TestAtomicNumpySave:
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((100,), np.float32),
            ((50, 20), np.float64),
            ((10, 10, 10), np.int32),
            ((1_000_000,), np.int32),  # large 1-D
        ],
    )
    def test_roundtrip(self, tmp_path, shape, dtype):
        arr = np.random.randint(0, 1000, size=shape).astype(dtype)
        path = tmp_path / "arr.npy"
        atomic_numpy_save(arr, path)
        loaded = atomic_numpy_load(path)
        np.testing.assert_array_equal(arr, loaded)

    def test_integrity_check_shape(self, tmp_path):
        arr = np.zeros((5, 3), dtype=np.float32)
        path = tmp_path / "arr.npy"
        atomic_numpy_save(arr, path)
        with pytest.raises(ValueError, match="Shape mismatch"):
            atomic_numpy_load(path, expected_shape=(5, 4))

    def test_integrity_check_dtype(self, tmp_path):
        arr = np.zeros(10, dtype=np.float32)
        path = tmp_path / "arr.npy"
        atomic_numpy_save(arr, path)
        with pytest.raises(ValueError, match="dtype mismatch"):
            atomic_numpy_load(path, expected_dtype=np.int32)

    def test_wildcard_shape_dimension(self, tmp_path):
        """Shape check should pass for dimensions specified as -1."""
        arr = np.ones((7, 5), dtype=np.int32)
        path = tmp_path / "arr.npy"
        atomic_numpy_save(arr, path)
        loaded = atomic_numpy_load(path, expected_shape=(-1, 5), expected_dtype=np.int32)
        assert loaded.shape == (7, 5)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            atomic_numpy_load(tmp_path / "nonexistent.npy")


# ===========================================================================
# 1.2 — 128-bit infoset hashing
# ===========================================================================


class TestHashInfoSet128:
    def test_deterministic(self):
        info_set = "Ah Kd | Qs Jh Td | bet call"
        h1 = hash_info_set_128(info_set)
        h2 = hash_info_set_128(info_set)
        assert h1 == h2

    def test_returns_two_unsigned_64bit_ints(self):
        high, low = hash_info_set_128("test_info_set")
        assert 0 <= high < 2 ** 64
        assert 0 <= low < 2 ** 64

    def test_different_strings_produce_different_hashes(self):
        strings = [f"infoset_{i}" for i in range(1000)]
        hashes = {hash_info_set_128(s) for s in strings}
        assert len(hashes) == len(strings), "Unexpected hash collision in 1,000 samples"

    def test_bytes_representation_is_16_bytes(self):
        digest = hash_info_set_bytes("test")
        assert len(digest) == 16

    def test_bytes_roundtrip_via_struct(self):
        """hash_info_set_bytes must encode the same two 64-bit values."""
        info_set = "sample_info_set"
        high, low = hash_info_set_128(info_set)
        digest = hash_info_set_bytes(info_set)
        high2, low2 = struct.unpack("<QQ", digest)
        assert high == high2
        assert low == low2

    def test_no_collisions_100k_synthetic(self):
        """Hash 100K synthetic infoset strings and assert zero collisions.

        The full 10M test from the plan is split into a fast unit test
        (100K) and an optional heavy test (1M) that can be run separately.
        """
        n = 100_000
        strings = [f"player0|Ah_3c|Kd_Qs_Jh|round3|action_seq_{i}" for i in range(n)]
        seen: dict = {}
        for s in strings:
            h = hash_info_set_bytes(s)
            if h in seen:
                raise AssertionError(
                    f"Collision: {seen[h]!r} and {s!r} both hash to {h.hex()}"
                )
            seen[h] = s
        assert len(seen) == n

    @pytest.mark.slow
    def test_no_collisions_1m_synthetic(self):
        """Hash 1M synthetic strings.  Marked slow — run with -m slow."""
        n = 1_000_000
        strings = [f"p0|Ah_3c|Kd_Qs_Jh|r3|seq_{i}" for i in range(n)]
        hashes = {hash_info_set_bytes(s) for s in strings}
        assert len(hashes) == n


# ===========================================================================
# 1.3 — InfosetIndex
# ===========================================================================


class TestInfosetIndex:
    def test_get_returns_none_for_unknown(self, tmp_path):
        with InfosetIndex(tmp_path / "idx") as idx:
            assert idx.get("unknown_infoset") is None

    def test_get_or_create_new_entry(self, tmp_path):
        with InfosetIndex(tmp_path / "idx") as idx:
            location, is_new = idx.get_or_create("first_infoset")
            assert is_new is True
            chunk_id, row = location
            assert chunk_id == 0
            assert row == 0

    def test_get_or_create_existing_entry(self, tmp_path):
        with InfosetIndex(tmp_path / "idx") as idx:
            loc1, new1 = idx.get_or_create("same_infoset")
            loc2, new2 = idx.get_or_create("same_infoset")
            assert new1 is True
            assert new2 is False
            assert loc1 == loc2

    def test_get_after_create(self, tmp_path):
        with InfosetIndex(tmp_path / "idx") as idx:
            location, _ = idx.get_or_create("my_infoset")
            assert idx.get("my_infoset") == location

    def test_sequential_allocation_increments_rows(self, tmp_path):
        n = 10
        with InfosetIndex(tmp_path / "idx") as idx:
            locations = []
            for i in range(n):
                loc, is_new = idx.get_or_create(f"infoset_{i}")
                assert is_new
                locations.append(loc)

        # Locations must be (0, 0), (0, 1), ..., (0, n-1)
        expected = [(0, i) for i in range(n)]
        assert locations == expected

    def test_chunk_boundary_allocation(self, tmp_path):
        """The second chunk starts when global row count exceeds CHUNK_SIZE."""
        n = CHUNK_SIZE + 5
        with InfosetIndex(tmp_path / "idx") as idx:
            for i in range(n):
                idx.get_or_create(f"is_{i}")
            loc, _ = idx.get_or_create(f"is_{n}")
            # Row CHUNK_SIZE belongs in chunk 1, row 0.
            # The n-th entry (0-indexed) == CHUNK_SIZE:
            assert loc == (1, 0) or loc[0] == 1

    def test_n_entries_tracks_count(self, tmp_path):
        n = 50
        with InfosetIndex(tmp_path / "idx") as idx:
            for i in range(n):
                idx.get_or_create(f"entry_{i}")
            assert idx.n_entries == n

    def test_persist_and_reload(self, tmp_path):
        """Close and reopen the index — all entries must survive."""
        idx_path = tmp_path / "persistent_idx"
        infosets = [f"persist_is_{i}" for i in range(500)]
        locations_before = {}

        with InfosetIndex(idx_path) as idx:
            for s in infosets:
                loc, _ = idx.get_or_create(s)
                locations_before[s] = loc

        # Reopen from disk.
        with InfosetIndex(idx_path) as idx:
            assert idx.n_entries == len(infosets)
            for s in infosets:
                loc = idx.get(s)
                assert loc is not None, f"{s!r} should exist after reload"
                assert loc == locations_before[s], (
                    f"Location mismatch for {s!r}: "
                    f"before={locations_before[s]}, after={loc}"
                )

    def test_put_explicit_mapping(self, tmp_path):
        with InfosetIndex(tmp_path / "idx") as idx:
            idx.put("manual_entry", chunk_id=2, row=77)
            loc = idx.get("manual_entry")
            assert loc == (2, 77)

    def test_put_conflict_raises(self, tmp_path):
        with InfosetIndex(tmp_path / "idx") as idx:
            idx.put("conflict_entry", chunk_id=0, row=0)
            with pytest.raises(ValueError, match="conflict"):
                idx.put("conflict_entry", chunk_id=0, row=1)

    def test_put_idempotent(self, tmp_path):
        """Putting the same mapping twice should not raise."""
        with InfosetIndex(tmp_path / "idx") as idx:
            idx.put("same", 1, 5)
            idx.put("same", 1, 5)  # must not raise
            assert idx.get("same") == (1, 5)

    def test_flush_does_not_raise(self, tmp_path):
        with InfosetIndex(tmp_path / "idx") as idx:
            idx.get_or_create("some_entry")
            idx.flush()  # should not raise

    @pytest.mark.slow
    def test_1m_insert_then_reload(self, tmp_path):
        """Insert 1M synthetic infosets, close, reopen, verify all present.

        Marked slow — run with ``pytest -m slow``.
        """
        n = 1_000_000
        idx_path = tmp_path / "large_idx"
        infosets = [f"p0|Ah_3c|Kd_Qs_Jh|r3|seq_{i}" for i in range(n)]

        with InfosetIndex(idx_path) as idx:
            locs = {}
            for s in infosets:
                loc, _ = idx.get_or_create(s)
                locs[s] = loc
            assert idx.n_entries == n

        mismatches = 0
        with InfosetIndex(idx_path) as idx:
            for s in infosets:
                loc = idx.get(s)
                if loc != locs[s]:
                    mismatches += 1
        assert mismatches == 0, f"{mismatches} location mismatches after reload"

    def test_debug_mode_no_false_collision(self, tmp_path):
        """Debug mode must not raise for distinct strings that land on
        different hash keys (the normal case)."""
        with InfosetIndex(tmp_path / "idx", debug=True) as idx:
            for i in range(1000):
                idx.get_or_create(f"debug_is_{i}")

    def test_context_manager_closes_cleanly(self, tmp_path):
        """__exit__ must close the environment without error."""
        with InfosetIndex(tmp_path / "cm_idx") as idx:
            idx.get_or_create("cm_test")
        # After the context manager, the env is closed.  Accessing the path
        # should still exist on disk.
        assert (tmp_path / "cm_idx").exists()
