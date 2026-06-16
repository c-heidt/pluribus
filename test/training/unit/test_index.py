"""Unit tests for ``poker_ai/ai/index.py``.

Covers:
- SipHash-128 used internally by :class:`~poker_ai.tables.index.InfosetIndex`:
  determinism, collision resistance, byte encoding.
- :class:`~poker_ai.tables.index.InfosetIndex` LMDB-backed string→row mapping:
  allocation, persistence, fork safety.
"""

import struct

import pytest

from poker_ai.tables.chunk_store import CHUNK_SIZE
from poker_ai.tables.index import InfosetIndex, hash_info_set_128, hash_info_set_bytes


# ---------------------------------------------------------------------------
# SipHash helpers
# ---------------------------------------------------------------------------


class TestSipHash:
    def test_deterministic(self):
        info_set = "Ah Kd | Qs Jh Td | bet call"
        assert hash_info_set_128(info_set) == hash_info_set_128(info_set)

    def test_returns_two_unsigned_64bit_ints(self):
        high, low = hash_info_set_128("test_info_set")
        assert 0 <= high < 2 ** 64
        assert 0 <= low < 2 ** 64

    def test_different_strings_produce_different_hashes(self):
        strings = [f"infoset_{i}" for i in range(1000)]
        hashes = {hash_info_set_128(s) for s in strings}
        assert len(hashes) == len(strings)

    def test_bytes_representation_is_16_bytes(self):
        assert len(hash_info_set_bytes("test")) == 16

    def test_bytes_roundtrip_via_struct(self):
        info_set = "sample_info_set"
        high, low = hash_info_set_128(info_set)
        high2, low2 = struct.unpack("<QQ", hash_info_set_bytes(info_set))
        assert high == high2
        assert low == low2

    def test_no_collisions_100k_synthetic(self):
        n = 100_000
        strings = [f"player0|Ah_3c|Kd_Qs_Jh|round3|action_seq_{i}" for i in range(n)]
        seen: dict = {}
        for s in strings:
            h = hash_info_set_bytes(s)
            assert h not in seen, f"Collision: {seen[h]!r} and {s!r}"
            seen[h] = s

    @pytest.mark.slow
    def test_no_collisions_1m_synthetic(self):
        n = 1_000_000
        strings = [f"p0|Ah_3c|Kd_Qs_Jh|r3|seq_{i}" for i in range(n)]
        hashes = {hash_info_set_bytes(s) for s in strings}
        assert len(hashes) == n


# ---------------------------------------------------------------------------
# InfosetIndex
# ---------------------------------------------------------------------------


class TestInfosetIndex:
    def test_get_returns_none_for_unknown(self, tmp_path):
        with InfosetIndex(tmp_path / "idx") as idx:
            assert idx.get("unknown_infoset") is None

    def test_get_or_create_new_entry(self, tmp_path):
        with InfosetIndex(tmp_path / "idx") as idx:
            flat_row, is_new = idx.get_or_create("first_infoset")
            assert is_new is True
            assert flat_row == 0

    def test_get_or_create_existing_entry(self, tmp_path):
        with InfosetIndex(tmp_path / "idx") as idx:
            row1, new1 = idx.get_or_create("same_infoset")
            row2, new2 = idx.get_or_create("same_infoset")
            assert new1 is True
            assert new2 is False
            assert row1 == row2

    def test_get_after_create(self, tmp_path):
        with InfosetIndex(tmp_path / "idx") as idx:
            flat_row, _ = idx.get_or_create("my_infoset")
            assert idx.get("my_infoset") == flat_row

    def test_sequential_allocation_increments_rows(self, tmp_path):
        n = 10
        with InfosetIndex(tmp_path / "idx") as idx:
            rows = [idx.get_or_create(f"infoset_{i}")[0] for i in range(n)]
        assert rows == list(range(n))

    def test_chunk_boundary_allocation(self, tmp_path):
        n = CHUNK_SIZE + 5
        with InfosetIndex(tmp_path / "idx") as idx:
            for i in range(n):
                idx.get_or_create(f"is_{i}")
            row, _ = idx.get_or_create(f"is_{n}")
            assert row == n

    def test_n_allocated_rows_tracks_count(self, tmp_path):
        n = 50
        with InfosetIndex(tmp_path / "idx") as idx:
            for i in range(n):
                idx.get_or_create(f"entry_{i}")
            assert idx.n_allocated_rows == n

    def test_persist_and_reload(self, tmp_path):
        idx_path = tmp_path / "persistent_idx"
        infosets = [f"persist_is_{i}" for i in range(500)]
        rows_before = {}
        with InfosetIndex(idx_path) as idx:
            for s in infosets:
                row, _ = idx.get_or_create(s)
                rows_before[s] = row
        with InfosetIndex(idx_path) as idx:
            assert idx.n_allocated_rows == len(infosets)
            for s in infosets:
                assert idx.get(s) == rows_before[s]

    def test_flush_does_not_raise(self, tmp_path):
        with InfosetIndex(tmp_path / "idx") as idx:
            idx.get_or_create("some_entry")
            idx.flush()

    def test_debug_mode_no_false_collision(self, tmp_path):
        with InfosetIndex(tmp_path / "idx", debug=True) as idx:
            for i in range(1000):
                idx.get_or_create(f"debug_is_{i}")

    def test_context_manager_closes_cleanly(self, tmp_path):
        with InfosetIndex(tmp_path / "cm_idx") as idx:
            idx.get_or_create("cm_test")
        assert (tmp_path / "cm_idx").exists()

    def test_copy_to_produces_loadable_snapshot(self, tmp_path):
        """``InfosetIndex.copy_to`` should write a consistent on-disk
        snapshot that can be opened by a fresh :class:`InfosetIndex`
        and serves the same row mappings as the original.
        """
        src_path = tmp_path / "src_idx"
        dst_path = tmp_path / "dst_idx"
        infosets = [f"copy_is_{i}" for i in range(250)]
        rows_before = {}
        with InfosetIndex(src_path) as idx:
            for s in infosets:
                row, _ = idx.get_or_create(s)
                rows_before[s] = row
            idx.copy_to(dst_path)
        # The original is closed; reopen the *destination* and confirm
        # it serves the exact same rows.
        with InfosetIndex(dst_path) as idx:
            assert idx.n_allocated_rows == len(infosets)
            for s in infosets:
                assert idx.get(s) == rows_before[s]

    def test_copy_to_works_with_open_env(self, tmp_path):
        """``env.copy`` is MVCC-consistent — calling it on a still-open
        env should yield a snapshot that reflects state at copy time
        and does not require closing the live env first.
        """
        src_path = tmp_path / "src_idx_live"
        dst_path = tmp_path / "dst_idx_live"
        with InfosetIndex(src_path) as src:
            for i in range(100):
                src.get_or_create(f"live_is_{i}")
            src.copy_to(dst_path)
            # Live env keeps working afterwards.
            row, is_new = src.get_or_create("post_copy")
            assert is_new
        with InfosetIndex(dst_path) as dst:
            # Snapshot has the 100 pre-copy entries; the post-copy
            # insert is NOT in the snapshot.
            assert dst.n_allocated_rows == 100
            assert dst.get("post_copy") is None

    @pytest.mark.slow
    def test_1m_insert_then_reload(self, tmp_path):
        n = 1_000_000
        idx_path = tmp_path / "large_idx"
        infosets = [f"p0|Ah_3c|Kd_Qs_Jh|r3|seq_{i}" for i in range(n)]
        rows = {}
        with InfosetIndex(idx_path) as idx:
            for s in infosets:
                row, _ = idx.get_or_create(s)
                rows[s] = row
            assert idx.n_allocated_rows == n
        mismatches = 0
        with InfosetIndex(idx_path) as idx:
            for s in infosets:
                if idx.get(s) != rows[s]:
                    mismatches += 1
        assert mismatches == 0
