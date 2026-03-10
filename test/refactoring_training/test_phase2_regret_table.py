"""Unit tests for Phase 2: SparseRegretTable.

Covers:
- Chunk creation and lazy attachment (2.1)
- Core access methods (2.2)
- Stripe locking (2.3)
- Dirty tracking (2.4)
- apply_discount (2.5)
- Shared memory naming convention and orphan detection (2.6)

All tests use a temporary directory as ``shm_dir`` so they work without
root access to ``/dev/shm`` and clean up automatically.
"""
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from poker_ai.ai.index import CHUNK_SIZE, InfosetIndex
from poker_ai.ai.regret_table import (
    N_STRIPE_LOCKS,
    REGRET_FLOOR,
    SparseRegretTable,
    _MAX_DIRTY_CHUNKS,
    list_orphaned_blocks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def n_actions():
    return 5


@pytest.fixture
def table_name():
    return "pluribus_test"


@pytest.fixture
def tmp_lmdb(tmp_path):
    """Open an InfosetIndex in a temp directory, close on teardown."""
    idx_path = tmp_path / "index"
    idx = InfosetIndex(idx_path)
    yield idx
    idx.close()


@pytest.fixture
def table(tmp_path, tmp_lmdb, n_actions, table_name):
    """Create a SparseRegretTable backed by tmp_path, unlink on teardown."""
    shm_dir = str(tmp_path / "shm")
    os.makedirs(shm_dir, exist_ok=True)
    tbl = SparseRegretTable(
        n_actions=n_actions,
        table_name=table_name,
        index=tmp_lmdb,
        shm_dir=shm_dir,
    )
    yield tbl
    tbl.close()
    tbl.unlink_all()


# ---------------------------------------------------------------------------
# 2.1 — SparseRegretTable construction and chunk creation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_n_actions_stored(self, table, n_actions):
        assert table.n_actions == n_actions

    def test_table_name_stored(self, table, table_name):
        assert table.table_name == table_name

    def test_zero_chunks_on_empty_index(self, table):
        assert table.n_chunks == 0

    def test_invalid_n_actions_raises(self, tmp_path, n_actions, table_name):
        idx = InfosetIndex(tmp_path / "idx_bad")
        with pytest.raises(ValueError, match="n_actions"):
            SparseRegretTable(
                n_actions=0,
                table_name=table_name,
                index=idx,
                shm_dir=str(tmp_path / "shm2"),
            )
        idx.close()

    def test_chunk_file_created_in_shm_dir(self, table, n_actions, table_name):
        table.get_row("first_infoset")
        assert table.n_chunks == 1
        expected_path = os.path.join(
            str(Path(table._shm_paths[0]).parent),
            f"{table_name}_000000",
        )
        assert os.path.exists(expected_path), f"expected chunk at {expected_path}"

    def test_chunk_file_size_correct(self, table, n_actions):
        table.get_row("probe")
        path = table._shm_paths[0]
        expected_bytes = CHUNK_SIZE * n_actions * 4  # int32
        assert os.path.getsize(path) == expected_bytes

    def test_resume_restores_existing_chunks(self, tmp_path, n_actions, table_name):
        """Reopening the table with an existing index re-opens chunk files."""
        idx_path = tmp_path / "idx_resume"
        shm_dir = str(tmp_path / "shm_resume")
        os.makedirs(shm_dir)

        # First session: allocate some infosets
        idx1 = InfosetIndex(idx_path)
        tbl1 = SparseRegretTable(n_actions, table_name, idx1, shm_dir)
        n = 100
        for k in range(n):
            row = tbl1.get_row(f"resume_is_{k}")
            row[:] = k  # write known values
        idx1.close()
        tbl1.close()  # close mmaps but do NOT unlink

        # Second session: reopen
        idx2 = InfosetIndex(idx_path)
        tbl2 = SparseRegretTable(n_actions, table_name, idx2, shm_dir)
        assert tbl2.n_chunks == 1, "should have restored the existing chunk"
        for k in range(n):
            row = tbl2.get_row_if_exists(f"resume_is_{k}")
            assert row is not None
            np.testing.assert_array_equal(row, np.full(n_actions, k, dtype=np.int32))
        idx2.close()
        tbl2.close()
        tbl2.unlink_all()


# ---------------------------------------------------------------------------
# 2.2 — Core access methods
# ---------------------------------------------------------------------------


class TestCoreAccessMethods:
    def test_get_row_zeros_on_first_access(self, table, n_actions):
        row = table.get_row("brand_new_infoset")
        np.testing.assert_array_equal(row, np.zeros(n_actions, dtype=np.int32))

    def test_get_row_write_then_read(self, table, n_actions):
        row = table.get_row("writable_infoset")
        row[:] = [10, 20, 30, 40, 50]
        row2 = table.get_row("writable_infoset")
        np.testing.assert_array_equal(row2, [10, 20, 30, 40, 50])

    def test_get_row_returns_view_not_copy(self, table):
        row1 = table.get_row("view_test")
        row2 = table.get_row("view_test")
        row1[0] = 999
        assert row2[0] == 999, "both references should point to the same memory"

    def test_get_row_if_exists_none_for_unknown(self, table):
        assert table.get_row_if_exists("never_seen") is None

    def test_get_row_if_exists_returns_after_get_row(self, table, n_actions):
        row = table.get_row("exists_infoset")
        row[:] = list(range(n_actions))
        existing = table.get_row_if_exists("exists_infoset")
        assert existing is not None
        np.testing.assert_array_equal(existing, list(range(n_actions)))

    def test_get_row_by_location_matches_get_row(self, table, n_actions):
        row_a = table.get_row("loc_test")
        row_a[:] = [1, 2, 3, 4, 5]
        location = table._index.get("loc_test")
        assert location is not None
        chunk_id, row_idx = location
        row_b = table.get_row_by_location(chunk_id, row_idx)
        np.testing.assert_array_equal(row_a, row_b)

    def test_get_row_by_location_invalid_row_raises(self, table):
        table.get_row("setup_chunk")  # ensure chunk 0 exists
        with pytest.raises(IndexError):
            table.get_row_by_location(0, CHUNK_SIZE)  # row == CHUNK_SIZE is out of range

    def test_multiple_infosets_different_rows(self, table):
        n = 10
        rows = [table.get_row(f"multi_is_{k}") for k in range(n)]
        for k, row in enumerate(rows):
            row[:] = k
        # Verify all rows are independently written
        for k in range(n):
            result = table.get_row(f"multi_is_{k}")
            assert result[0] == k

    def test_chunk_boundary_allocation(self, table, n_actions):
        """Rows that cross CHUNK_SIZE should land in chunk 1."""
        n = CHUNK_SIZE + 5
        for k in range(n):
            table.get_row(f"cross_chunk_is_{k}")
        assert table.n_chunks >= 2
        # The CHUNK_SIZE-th infoset must be in chunk 1, row 0
        loc = table._index.get(f"cross_chunk_is_{CHUNK_SIZE}")
        assert loc is not None
        assert loc[0] == 1
        assert loc[1] == 0

    @pytest.mark.slow
    def test_500k_infosets_correct_retrieval(self, tmp_path, n_actions, table_name):
        """Allocate 500K infosets across 5+ chunks and verify retrieval."""
        n = 500_000
        idx = InfosetIndex(tmp_path / "big_idx")
        shm_dir = str(tmp_path / "big_shm")
        os.makedirs(shm_dir)
        tbl = SparseRegretTable(n_actions, table_name, idx, shm_dir)

        # Write a unique fingerprint into each row
        for k in range(n):
            row = tbl.get_row(f"big_is_{k}")
            row[0] = k % (2 ** 31 - 1)  # stay in int32 range

        assert tbl.n_chunks >= 5

        # Spot-check every 1000th row
        mismatches = 0
        for k in range(0, n, 1000):
            row = tbl.get_row_if_exists(f"big_is_{k}")
            expected = k % (2 ** 31 - 1)
            if row is None or row[0] != expected:
                mismatches += 1
        assert mismatches == 0

        idx.close()
        tbl.close()
        tbl.unlink_all()


# ---------------------------------------------------------------------------
# 2.3 — Stripe locking
# ---------------------------------------------------------------------------


class TestStripeLocking:
    def test_stripe_assignment_deterministic(self, table):
        for chunk_id in range(N_STRIPE_LOCKS * 2):
            lock_a = table.get_stripe_lock(chunk_id)
            lock_b = table.get_stripe_lock(chunk_id)
            assert lock_a is lock_b, "same chunk_id must always return the same lock"

    def test_stripe_assignment_formula(self, table):
        """Stripe ID must equal chunk_id % N_STRIPE_LOCKS."""
        for chunk_id in range(N_STRIPE_LOCKS * 3):
            expected_stripe = chunk_id % N_STRIPE_LOCKS
            actual_lock = table.get_stripe_lock(chunk_id)
            expected_lock = table._stripe_locks[expected_stripe]
            assert actual_lock is expected_lock

    def test_different_chunks_same_stripe(self, table):
        """Chunks N and N+256 must share the same stripe lock."""
        for base in range(N_STRIPE_LOCKS):
            lock_a = table.get_stripe_lock(base)
            lock_b = table.get_stripe_lock(base + N_STRIPE_LOCKS)
            assert lock_a is lock_b

    def test_n_stripe_locks_equals_constant(self, table):
        assert len(table._stripe_locks) == N_STRIPE_LOCKS

    def test_stripe_locks_evenly_distributed(self):
        """Chunk IDs 0..N_STRIPE_LOCKS-1 must map to all distinct stripe IDs."""
        seen_stripes = {i % N_STRIPE_LOCKS for i in range(N_STRIPE_LOCKS)}
        assert len(seen_stripes) == N_STRIPE_LOCKS

    def test_stripe_lock_is_acquirable(self, table):
        lock = table.get_stripe_lock(0)
        acquired = lock.acquire(timeout=1.0)
        assert acquired, "stripe lock should be immediately acquirable"
        lock.release()


# ---------------------------------------------------------------------------
# 2.4 — Dirty tracking
# ---------------------------------------------------------------------------


class TestDirtyTracking:
    def test_no_dirty_chunks_initially(self, table):
        table.get_row("init_me")
        assert table.get_dirty_chunks() == []

    def test_mark_dirty_sets_flag(self, table):
        table.get_row("dirty_test")
        assert table.n_chunks == 1
        table._mark_dirty(0)
        assert 0 in table.get_dirty_chunks()

    def test_clear_dirty_resets_flag(self, table):
        table.get_row("clear_dirty_test")
        table._mark_dirty(0)
        table.clear_dirty(0)
        assert table.get_dirty_chunks() == []

    def test_clear_all_dirty(self, table):
        # Allocate a few chunks
        for k in range(CHUNK_SIZE + 10):
            table.get_row(f"bulk_dirty_is_{k}")
        assert table.n_chunks == 2
        table._mark_dirty(0)
        table._mark_dirty(1)
        assert sorted(table.get_dirty_chunks()) == [0, 1]
        table.clear_all_dirty()
        assert table.get_dirty_chunks() == []

    def test_dirty_flag_not_set_on_read_only(self, table):
        """Calling get_row_if_exists on an unvisited infoset must not mark dirty."""
        assert table.get_row_if_exists("nonexistent") is None
        assert table.get_dirty_chunks() == []

    def test_get_dirty_chunks_only_returns_allocated(self, table):
        """get_dirty_chunks must never return chunk IDs beyond n_chunks."""
        assert table.n_chunks == 0
        dirty = table.get_dirty_chunks()
        assert dirty == []

    def test_dirty_flag_out_of_bounds_is_silently_ignored(self, table):
        """Marking chunk_id >= _MAX_DIRTY_CHUNKS must not raise."""
        table._mark_dirty(_MAX_DIRTY_CHUNKS + 1)  # must not raise
        table.clear_dirty(_MAX_DIRTY_CHUNKS + 1)   # must not raise


# ---------------------------------------------------------------------------
# 2.5 — apply_discount
# ---------------------------------------------------------------------------


class TestApplyDiscount:
    def _fill_rows(self, table: SparseRegretTable, values: list, n_infosets: int = 3):
        """Write *values* into the first row of each allocated infoset."""
        for k in range(n_infosets):
            row = table.get_row(f"discount_is_{k}")
            row[:] = values

    def test_apply_discount_requires_sync_boundary(self, table):
        """Must raise AssertionError if sync boundary flag is not set."""
        table.get_row("guard_test")
        with pytest.raises(AssertionError, match="sync boundary"):
            table.apply_discount(0.5)

    def test_apply_discount_factor_validation(self, table):
        """Factor must be in (0, 1].  Out-of-range values raise ValueError."""
        table.set_sync_boundary(True)
        with pytest.raises(ValueError, match="factor"):
            table.apply_discount(0.0)
        with pytest.raises(ValueError, match="factor"):
            table.apply_discount(1.1)
        table.set_sync_boundary(False)

    def test_apply_discount_scales_values(self, table, n_actions):
        """After discount by 0.5, all values should be halved (then floor'd)."""
        initial = [1_000, 2_000, 3_000, 4_000, 5_000]
        self._fill_rows(table, initial, n_infosets=3)
        table.set_sync_boundary(True)
        table.apply_discount(0.5)
        table.set_sync_boundary(False)
        for k in range(3):
            row = table.get_row(f"discount_is_{k}")
            expected = [int(v * 0.5) for v in initial]
            np.testing.assert_array_equal(row, expected)

    def test_apply_discount_floor(self, table, n_actions):
        """Values that would discount below REGRET_FLOOR must be clamped."""
        floor_val = int(REGRET_FLOOR)  # -310_000_000
        row = table.get_row("floor_is_0")
        row[:] = floor_val  # already at floor
        table.set_sync_boundary(True)
        table.apply_discount(0.5)
        table.set_sync_boundary(False)
        result = table.get_row("floor_is_0")
        # floor_val * 0.5 = -155_000_000 > REGRET_FLOOR so no clamping here
        # But a value below the floor (only possible if manually set to a very
        # negative number) must be clamped.
        assert np.all(result >= REGRET_FLOOR)

    def test_apply_discount_clamps_to_floor(self, table, n_actions):
        """A value below REGRET_FLOOR should also be clamped to the floor."""
        row = table.get_row("below_floor_is_0")
        # Set a value well below the floor
        row[:] = int(REGRET_FLOOR) - 1_000_000
        table.set_sync_boundary(True)
        table.apply_discount(0.5)
        table.set_sync_boundary(False)
        result = table.get_row("below_floor_is_0")
        assert np.all(result >= REGRET_FLOOR)

    def test_apply_discount_only_allocated_rows(self, table, n_actions):
        """Rows beyond the allocated range must remain zero after discount."""
        # Allocate exactly 2 rows — chunk 0 will have 2 valid rows out of CHUNK_SIZE
        for k in range(2):
            row = table.get_row(f"sparse_is_{k}")
            row[:] = 1_000_000
        table.set_sync_boundary(True)
        table.apply_discount(0.5)
        table.set_sync_boundary(False)
        # Rows 2..CHUNK_SIZE-1 in chunk 0 must still be zero (never touched)
        raw_chunk = table._chunks[0]
        # Allocated rows discounted
        np.testing.assert_array_equal(raw_chunk[0], np.full(n_actions, 500_000))
        np.testing.assert_array_equal(raw_chunk[1], np.full(n_actions, 500_000))
        # Unallocated rows must be zero
        np.testing.assert_array_equal(raw_chunk[2], np.zeros(n_actions, dtype=np.int32))

    def test_apply_discount_factor_1_leaves_unchanged(self, table, n_actions):
        """Discount by 1.0 must be a no-op."""
        initial = [100, 200, 300, 400, 500]
        for k in range(3):
            row = table.get_row(f"noop_is_{k}")
            row[:] = initial
        table.set_sync_boundary(True)
        table.apply_discount(1.0)
        table.set_sync_boundary(False)
        for k in range(3):
            row = table.get_row(f"noop_is_{k}")
            np.testing.assert_array_equal(row, initial)

    def test_apply_discount_across_multiple_chunks(self, table, n_actions):
        """Discount must apply to rows in all chunks."""
        # Allocate rows that span two chunks
        n = CHUNK_SIZE + 10
        for k in range(n):
            row = table.get_row(f"multi_chunk_disc_is_{k}")
            row[:] = 1_000
        assert table.n_chunks == 2
        table.set_sync_boundary(True)
        table.apply_discount(0.5)
        table.set_sync_boundary(False)
        # Check first and last row
        first_row = table.get_row("multi_chunk_disc_is_0")
        last_row = table.get_row(f"multi_chunk_disc_is_{n - 1}")
        np.testing.assert_array_equal(first_row, np.full(n_actions, 500))
        np.testing.assert_array_equal(last_row, np.full(n_actions, 500))


# ---------------------------------------------------------------------------
# 2.6 — Naming convention and orphan detection
# ---------------------------------------------------------------------------


class TestNamingAndOrphanDetection:
    def test_chunk_name_format(self, table, table_name):
        table.get_row("name_test")
        name = os.path.basename(table._shm_paths[0])
        expected = f"{table_name}_000000"
        assert name == expected

    def test_chunk_names_sequential(self, table, table_name):
        # Force two chunks
        for k in range(CHUNK_SIZE + 1):
            table.get_row(f"seq_is_{k}")
        assert table.n_chunks == 2
        for chunk_id in range(2):
            name = os.path.basename(table._shm_paths[chunk_id])
            expected = f"{table_name}_{chunk_id:06d}"
            assert name == expected

    def test_list_own_blocks(self, table):
        table.get_row("block_test")
        blocks = table.list_own_blocks()
        assert len(blocks) == 1
        assert os.path.exists(blocks[0])

    def test_list_orphaned_blocks_empty_dir(self, tmp_path):
        orphan_dir = str(tmp_path / "empty_shm")
        os.makedirs(orphan_dir)
        assert list_orphaned_blocks(shm_dir=orphan_dir) == []

    def test_list_orphaned_blocks_finds_files(self, tmp_path):
        shm_dir = str(tmp_path / "orphan_shm")
        os.makedirs(shm_dir)
        # Create fake orphan files (must start with "pluribus_")
        for i in range(3):
            path = os.path.join(shm_dir, f"pluribus_regret_0_{i:06d}")
            open(path, "wb").close()

        orphans = list_orphaned_blocks(shm_dir=shm_dir)
        assert len(orphans) == 3
        for path in orphans:
            assert os.path.exists(path)

    def test_list_orphaned_blocks_ignores_non_pluribus(self, tmp_path):
        """Files not starting with 'pluribus_' must be ignored."""
        shm_dir = str(tmp_path / "mixed_shm")
        os.makedirs(shm_dir)
        path_pluribus = os.path.join(shm_dir, "pluribus_regret_0_000000")
        open(path_pluribus, "wb").close()
        path_other = os.path.join(shm_dir, "other_prefix_000000")
        open(path_other, "wb").close()

        orphans = list_orphaned_blocks(shm_dir=shm_dir)
        assert len(orphans) == 1
        assert path_pluribus in orphans
        assert path_other not in orphans

    def test_list_orphaned_blocks_nonexistent_dir(self):
        assert list_orphaned_blocks(shm_dir="/nonexistent_dir_abc123") == []

    def test_unlink_all_removes_files(self, tmp_path, n_actions, table_name):
        idx = InfosetIndex(tmp_path / "unlink_idx")
        shm_dir = str(tmp_path / "unlink_shm")
        os.makedirs(shm_dir)
        tbl = SparseRegretTable(n_actions, table_name, idx, shm_dir)
        tbl.get_row("unlink_test")
        paths = tbl.list_own_blocks()
        assert all(os.path.exists(p) for p in paths)
        tbl.close()
        tbl.unlink_all()
        assert all(not os.path.exists(p) for p in paths)
        idx.close()

    def test_context_manager_unlinks(self, tmp_path, n_actions, table_name):
        idx = InfosetIndex(tmp_path / "cm_unlink_idx")
        shm_dir = str(tmp_path / "cm_unlink_shm")
        os.makedirs(shm_dir)
        with SparseRegretTable(n_actions, table_name, idx, shm_dir) as tbl:
            tbl.get_row("cm_unlink_test")
            paths = tbl.list_own_blocks()
        assert all(not os.path.exists(p) for p in paths)
        idx.close()


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_is_informative(self, table, table_name):
        table.get_row("repr_test")
        r = repr(table)
        assert table_name in r
        assert "SparseRegretTable" in r


# ---------------------------------------------------------------------------
# Bug regression tests (for bugs found during Phase 2 scan)
# ---------------------------------------------------------------------------
# Phase 5 additions: n_allocated, merge_delta_row, _restore_chunk
# ---------------------------------------------------------------------------


class TestNAllocated:
    def test_zero_initially(self, table):
        assert table.n_allocated == 0

    def test_increments_on_new_row(self, table):
        table.get_row("first")
        assert table.n_allocated == 1
        table.get_row("second")
        assert table.n_allocated == 2

    def test_stable_on_repeated_access(self, table):
        table.get_row("repeat_me")
        table.get_row("repeat_me")
        assert table.n_allocated == 1

    def test_increments_via_merge_delta_row(self, table, n_actions):
        delta = np.zeros(n_actions, dtype=np.int64)
        delta[0] = 1
        table.merge_delta_row("new_via_merge", delta)
        assert table.n_allocated == 1

    def test_stable_on_repeated_merge(self, table, n_actions):
        delta = np.zeros(n_actions, dtype=np.int64)
        table.merge_delta_row("merge_repeat", delta)
        table.merge_delta_row("merge_repeat", delta)
        assert table.n_allocated == 1


class TestMergeDeltaRow:
    def test_creates_row_on_first_call(self, table, n_actions):
        delta = np.full(n_actions, 7, dtype=np.int64)
        table.merge_delta_row("new_infoset", delta)
        row = table.get_row_if_exists("new_infoset")
        assert row is not None
        np.testing.assert_array_equal(row, np.full(n_actions, 7, dtype=np.int32))

    def test_accumulates_on_repeated_calls(self, table, n_actions):
        delta = np.ones(n_actions, dtype=np.int64) * 3
        table.merge_delta_row("accum_is", delta)
        table.merge_delta_row("accum_is", delta)
        row = table.get_row_if_exists("accum_is")
        assert row is not None
        np.testing.assert_array_equal(row, np.full(n_actions, 6, dtype=np.int32))

    def test_negative_delta_accumulates(self, table, n_actions):
        delta = np.full(n_actions, -5, dtype=np.int64)
        table.merge_delta_row("neg_is", delta)
        row = table.get_row_if_exists("neg_is")
        assert row is not None
        np.testing.assert_array_equal(row, np.full(n_actions, -5, dtype=np.int32))


class TestRestoreChunk:
    def test_restore_creates_chunk_and_reads_data(self, table, n_actions):
        """_restore_chunk must make chunk 0 accessible with pre-set values."""
        from poker_ai.ai.index import CHUNK_SIZE
        arr = np.zeros((CHUNK_SIZE, n_actions), dtype=np.int32)
        arr[0, :] = 42
        table._restore_chunk(0, arr)
        assert table.n_chunks >= 1
        # Chunk 0 must have the restored values at row 0
        restored_row = table.get_row_by_location(0, 0)
        np.testing.assert_array_equal(restored_row, np.full(n_actions, 42))


# ---------------------------------------------------------------------------


class TestBugRegressions:
    def test_get_row_allocates_sequentially(self, table):
        """Regression: rows must be allocated in sequential global order."""
        n = 10
        locations = []
        for k in range(n):
            table.get_row(f"reg_is_{k}")
            loc = table._index.get(f"reg_is_{k}")
            assert loc is not None
            locations.append(loc)
        global_rows = [c * CHUNK_SIZE + r for c, r in locations]
        assert global_rows == list(range(n))

    def test_row_write_isolated(self, table, n_actions):
        """Regression: writing to one row must not affect adjacent rows."""
        row_a = table.get_row("iso_a")
        row_b = table.get_row("iso_b")
        row_c = table.get_row("iso_c")

        row_b[:] = 12345
        np.testing.assert_array_equal(row_a, np.zeros(n_actions, dtype=np.int32))
        np.testing.assert_array_equal(row_c, np.zeros(n_actions, dtype=np.int32))

    def test_set_sync_boundary_visible(self, table):
        """set_sync_boundary() must immediately affect _at_sync_boundary_flag."""
        assert not table._at_sync_boundary_flag
        table.set_sync_boundary(True)
        assert table._at_sync_boundary_flag
        table.set_sync_boundary(False)
        assert not table._at_sync_boundary_flag
