"""Unit tests for Phase 2: ChunkedTable + ChunkStore.

Covers:
- ChunkStore: mmap lifecycle, dirty tracking, save/restore
- ChunkedTable: data access, stripe locking, naming/orphan detection
- CFRTables.apply_discount (moved from ChunkedTable)

All tests use a temporary directory as ``shm_dir`` so they work without
root access to ``/dev/shm`` and clean up automatically.
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from poker_ai.ai.chunk_store import CHUNK_SIZE, ChunkStore, _MAX_DIRTY_CHUNKS
from poker_ai.ai.cfr_tables import REGRET_FLOOR, CFRTables
from poker_ai.ai.index import InfosetIndex
from poker_ai.ai.chunked_table import (
    N_STRIPE_LOCKS,
    ChunkedTable,
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
    """Create a ChunkedTable backed by tmp_path, unlink on teardown."""
    shm_dir = str(tmp_path / "shm")
    os.makedirs(shm_dir, exist_ok=True)
    tbl = ChunkedTable(
        n_actions=n_actions,
        table_name=table_name,
        index=tmp_lmdb,
        shm_dir=shm_dir,
    )
    yield tbl
    tbl.close()
    tbl.unlink_all()


# ---------------------------------------------------------------------------
# ChunkStore: mmap lifecycle and dirty tracking
# ---------------------------------------------------------------------------


class TestChunkStore:
    def test_ensure_open_creates_file(self, tmp_path, n_actions, table_name):
        shm = str(tmp_path / "shm")
        os.makedirs(shm)
        store = ChunkStore(table_name, n_actions, shm)
        store.ensure_open(0)
        assert store.n_open == 1
        assert len(store.paths) == 1
        assert os.path.exists(store.paths[0])
        store.close()
        store.unlink_all()

    def test_view_returns_correct_shape(self, tmp_path, n_actions, table_name):
        shm = str(tmp_path / "shm")
        os.makedirs(shm)
        store = ChunkStore(table_name, n_actions, shm)
        view = store.view(0)
        assert view.shape == (CHUNK_SIZE, n_actions)
        assert view.dtype == np.int32
        store.close()
        store.unlink_all()

    def test_dirty_tracking(self, tmp_path, n_actions, table_name):
        shm = str(tmp_path / "shm")
        os.makedirs(shm)
        store = ChunkStore(table_name, n_actions, shm)
        store.ensure_open(0)
        store.mark_dirty(0)
        # save_dirty should write the dirty chunk
        written = store.save_dirty(tmp_path, n_entries=10, prefix="test")
        assert written == 1
        store.clear_dirty()
        # After clearing, nothing to save
        written = store.save_dirty(tmp_path, n_entries=10, prefix="test")
        assert written == 0
        store.close()
        store.unlink_all()

    def test_save_all_ignores_dirty_flags(self, tmp_path, n_actions, table_name):
        shm = str(tmp_path / "shm")
        os.makedirs(shm)
        store = ChunkStore(table_name, n_actions, shm)
        store.ensure_open(0)
        # Don't mark dirty — save_all should still write
        written = store.save_all(tmp_path, n_entries=10, prefix="test")
        assert written == 1
        store.close()
        store.unlink_all()

    def test_restore_roundtrip(self, tmp_path, n_actions, table_name):
        shm = str(tmp_path / "shm")
        os.makedirs(shm)
        store = ChunkStore(table_name, n_actions, shm)
        data = np.full((5, n_actions), 42, dtype=np.int32)
        store.restore(0, data)
        view = store.view(0)
        np.testing.assert_array_equal(view[:5], data)
        np.testing.assert_array_equal(view[5], np.zeros(n_actions, dtype=np.int32))
        store.close()
        store.unlink_all()

    def test_dirty_out_of_bounds_ignored(self, tmp_path, n_actions, table_name):
        shm = str(tmp_path / "shm")
        os.makedirs(shm)
        store = ChunkStore(table_name, n_actions, shm)
        # Must not raise
        store.mark_dirty(_MAX_DIRTY_CHUNKS + 1)
        store.close()


# ---------------------------------------------------------------------------
# ChunkedTable construction and chunk creation
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
            ChunkedTable(
                n_actions=0,
                table_name=table_name,
                index=idx,
                shm_dir=str(tmp_path / "shm2"),
            )
        idx.close()

    def test_chunk_file_created_in_shm_dir(self, table, n_actions, table_name):
        table.get_row("first_infoset")
        assert table.n_chunks == 1
        paths = table.store.paths
        assert len(paths) == 1
        assert os.path.exists(paths[0])

    def test_chunk_file_size_correct(self, table, n_actions):
        table.get_row("probe")
        path = table.store.paths[0]
        expected_bytes = CHUNK_SIZE * n_actions * 4  # int32
        assert os.path.getsize(path) == expected_bytes

    def test_resume_restores_existing_chunks(self, tmp_path, n_actions, table_name):
        """Reopening the table with an existing index re-opens chunk files."""
        idx_path = tmp_path / "idx_resume"
        shm_dir = str(tmp_path / "shm_resume")
        os.makedirs(shm_dir)

        # First session: allocate some infosets
        idx1 = InfosetIndex(idx_path)
        tbl1 = ChunkedTable(n_actions, table_name, idx1, shm_dir)
        n = 100
        for k in range(n):
            row = tbl1.get_row(f"resume_is_{k}")
            row[:] = k
        idx1.close()
        tbl1.close()  # close mmaps but do NOT unlink

        # Second session: reopen
        idx2 = InfosetIndex(idx_path)
        tbl2 = ChunkedTable(n_actions, table_name, idx2, shm_dir)
        assert tbl2.n_chunks == 1, "should have restored the existing chunk"
        for k in range(n):
            row = tbl2.get_row_if_exists(f"resume_is_{k}")
            assert row is not None
            np.testing.assert_array_equal(row, np.full(n_actions, k, dtype=np.int32))
        idx2.close()
        tbl2.close()
        tbl2.unlink_all()


# ---------------------------------------------------------------------------
# Core access methods
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
        flat_row = table._index.get("loc_test")
        assert flat_row is not None
        chunk_id, row_idx = divmod(flat_row, CHUNK_SIZE)
        row_b = table.get_row_by_location(chunk_id, row_idx)
        np.testing.assert_array_equal(row_a, row_b)

    def test_get_row_by_location_invalid_row_raises(self, table):
        table.get_row("setup_chunk")
        with pytest.raises(IndexError):
            table.get_row_by_location(0, CHUNK_SIZE)

    def test_multiple_infosets_different_rows(self, table):
        n = 10
        rows = [table.get_row(f"multi_is_{k}") for k in range(n)]
        for k, row in enumerate(rows):
            row[:] = k
        for k in range(n):
            result = table.get_row(f"multi_is_{k}")
            assert result[0] == k

    def test_chunk_boundary_allocation(self, table, n_actions):
        """Rows that cross CHUNK_SIZE should land in chunk 1."""
        n = CHUNK_SIZE + 5
        for k in range(n):
            table.get_row(f"cross_chunk_is_{k}")
        assert table.n_chunks >= 2
        flat_row = table._index.get(f"cross_chunk_is_{CHUNK_SIZE}")
        assert flat_row is not None
        chunk_id, local_row = divmod(flat_row, CHUNK_SIZE)
        assert chunk_id == 1
        assert local_row == 0

    @pytest.mark.slow
    def test_500k_infosets_correct_retrieval(self, tmp_path, n_actions, table_name):
        """Allocate 500K infosets across 5+ chunks and verify retrieval."""
        n = 500_000
        idx = InfosetIndex(tmp_path / "big_idx")
        shm_dir = str(tmp_path / "big_shm")
        os.makedirs(shm_dir)
        tbl = ChunkedTable(n_actions, table_name, idx, shm_dir)

        for k in range(n):
            row = tbl.get_row(f"big_is_{k}")
            row[0] = k % (2 ** 31 - 1)

        assert tbl.n_chunks >= 5

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
# Stripe locking
# ---------------------------------------------------------------------------


class TestStripeLocking:
    def test_stripe_assignment_deterministic(self, table):
        for chunk_id in range(N_STRIPE_LOCKS * 2):
            lock_a = table.get_stripe_lock(chunk_id)
            lock_b = table.get_stripe_lock(chunk_id)
            assert lock_a is lock_b

    def test_stripe_assignment_formula(self, table):
        for chunk_id in range(N_STRIPE_LOCKS * 3):
            expected_stripe = chunk_id % N_STRIPE_LOCKS
            actual_lock = table.get_stripe_lock(chunk_id)
            expected_lock = table._stripe_locks[expected_stripe]
            assert actual_lock is expected_lock

    def test_different_chunks_same_stripe(self, table):
        for base in range(N_STRIPE_LOCKS):
            lock_a = table.get_stripe_lock(base)
            lock_b = table.get_stripe_lock(base + N_STRIPE_LOCKS)
            assert lock_a is lock_b

    def test_n_stripe_locks_equals_constant(self, table):
        assert len(table._stripe_locks) == N_STRIPE_LOCKS

    def test_stripe_locks_evenly_distributed(self):
        seen_stripes = {i % N_STRIPE_LOCKS for i in range(N_STRIPE_LOCKS)}
        assert len(seen_stripes) == N_STRIPE_LOCKS

    def test_stripe_lock_is_acquirable(self, table):
        lock = table.get_stripe_lock(0)
        acquired = lock.acquire(timeout=1.0)
        assert acquired
        lock.release()


# ---------------------------------------------------------------------------
# apply_discount (via CFRTables)
# ---------------------------------------------------------------------------


class TestApplyDiscount:
    """Discount is now a CFRTables-level operation.  We test it through
    a minimal CFRTables instance with one street."""

    @pytest.fixture
    def cfr(self, tmp_path):
        from poker_ai.ai.ai import MAX_ACTIONS_PER_STREET
        shm_dir = str(tmp_path / "shm_disc")
        os.makedirs(shm_dir)
        tables = CFRTables(
            index_path=tmp_path / "lmdb_disc",
            shm_dir=shm_dir,
            actions_per_street=MAX_ACTIONS_PER_STREET,
        )
        yield tables
        tables.close()

    def test_apply_discount_factor_validation(self, cfr):
        with pytest.raises(ValueError, match="factor"):
            cfr.apply_discount(0.0)
        with pytest.raises(ValueError, match="factor"):
            cfr.apply_discount(1.1)

    def test_apply_discount_scales_values(self, cfr):
        n_actions = cfr.regret[0].n_actions
        initial = list(range(1000, 1000 + n_actions))
        for k in range(3):
            row = cfr.regret[0].get_row(f"disc_is_{k}")
            row[:] = initial
        cfr.apply_discount(0.5)
        for k in range(3):
            row = cfr.regret[0].get_row(f"disc_is_{k}")
            expected = [int(v * 0.5) for v in initial]
            np.testing.assert_array_equal(row, expected)

    def test_apply_discount_floor(self, cfr):
        row = cfr.regret[0].get_row("floor_is_0")
        row[:] = int(REGRET_FLOOR)
        cfr.apply_discount(0.5)
        result = cfr.regret[0].get_row("floor_is_0")
        assert np.all(result >= REGRET_FLOOR)

    def test_apply_discount_factor_1_leaves_unchanged(self, cfr):
        n_actions = cfr.regret[0].n_actions
        initial = list(range(100, 100 + n_actions))
        for k in range(3):
            row = cfr.regret[0].get_row(f"noop_is_{k}")
            row[:] = initial
        cfr.apply_discount(1.0)
        for k in range(3):
            row = cfr.regret[0].get_row(f"noop_is_{k}")
            np.testing.assert_array_equal(row, initial)


# ---------------------------------------------------------------------------
# Naming convention and orphan detection
# ---------------------------------------------------------------------------


class TestNamingAndOrphanDetection:
    def test_chunk_name_format(self, table, table_name):
        table.get_row("name_test")
        name = os.path.basename(table.store.paths[0])
        expected = f"{table_name}_000000"
        assert name == expected

    def test_chunk_names_sequential(self, table, table_name):
        for k in range(CHUNK_SIZE + 1):
            table.get_row(f"seq_is_{k}")
        assert table.n_chunks == 2
        for chunk_id in range(2):
            name = os.path.basename(table.store.paths[chunk_id])
            expected = f"{table_name}_{chunk_id:06d}"
            assert name == expected

    def test_list_own_blocks(self, table):
        table.get_row("block_test")
        paths = table.store.paths
        assert len(paths) == 1
        assert os.path.exists(paths[0])

    def test_list_orphaned_blocks_empty_dir(self, tmp_path):
        orphan_dir = str(tmp_path / "empty_shm")
        os.makedirs(orphan_dir)
        assert list_orphaned_blocks(shm_dir=orphan_dir) == []

    def test_list_orphaned_blocks_finds_files(self, tmp_path):
        shm_dir = str(tmp_path / "orphan_shm")
        os.makedirs(shm_dir)
        for i in range(3):
            path = os.path.join(shm_dir, f"pluribus_regret_0_{i:06d}")
            open(path, "wb").close()
        orphans = list_orphaned_blocks(shm_dir=shm_dir)
        assert len(orphans) == 3

    def test_list_orphaned_blocks_ignores_non_pluribus(self, tmp_path):
        shm_dir = str(tmp_path / "mixed_shm")
        os.makedirs(shm_dir)
        path_pluribus = os.path.join(shm_dir, "pluribus_regret_0_000000")
        open(path_pluribus, "wb").close()
        path_other = os.path.join(shm_dir, "other_prefix_000000")
        open(path_other, "wb").close()
        orphans = list_orphaned_blocks(shm_dir=shm_dir)
        assert len(orphans) == 1
        assert path_pluribus in orphans

    def test_list_orphaned_blocks_nonexistent_dir(self):
        assert list_orphaned_blocks(shm_dir="/nonexistent_dir_abc123") == []

    def test_unlink_all_removes_files(self, tmp_path, n_actions, table_name):
        idx = InfosetIndex(tmp_path / "unlink_idx")
        shm_dir = str(tmp_path / "unlink_shm")
        os.makedirs(shm_dir)
        tbl = ChunkedTable(n_actions, table_name, idx, shm_dir)
        tbl.get_row("unlink_test")
        paths = tbl.store.paths
        assert all(os.path.exists(p) for p in paths)
        tbl.close()
        tbl.unlink_all()
        assert all(not os.path.exists(p) for p in paths)
        idx.close()

    def test_context_manager_unlinks(self, tmp_path, n_actions, table_name):
        idx = InfosetIndex(tmp_path / "cm_unlink_idx")
        shm_dir = str(tmp_path / "cm_unlink_shm")
        os.makedirs(shm_dir)
        with ChunkedTable(n_actions, table_name, idx, shm_dir) as tbl:
            tbl.get_row("cm_unlink_test")
            paths = tbl.store.paths
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
        assert "ChunkedTable" in r


# ---------------------------------------------------------------------------
# n_allocated
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


# ---------------------------------------------------------------------------
# merge_delta_row
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ChunkStore restore
# ---------------------------------------------------------------------------


class TestRestoreChunk:
    def test_restore_creates_chunk_and_reads_data(self, table, n_actions):
        arr = np.zeros((CHUNK_SIZE, n_actions), dtype=np.int32)
        arr[0, :] = 42
        table.store.restore(0, arr)
        assert table.n_chunks >= 1
        restored_row = table.get_row_by_location(0, 0)
        np.testing.assert_array_equal(restored_row, np.full(n_actions, 42))


# ---------------------------------------------------------------------------
# Bug regressions
# ---------------------------------------------------------------------------


class TestBugRegressions:
    def test_get_row_allocates_sequentially(self, table):
        n = 10
        flat_rows = []
        for k in range(n):
            table.get_row(f"reg_is_{k}")
            flat_row = table._index.get(f"reg_is_{k}")
            assert flat_row is not None
            flat_rows.append(flat_row)
        assert flat_rows == list(range(n))

    def test_row_write_isolated(self, table, n_actions):
        row_a = table.get_row("iso_a")
        row_b = table.get_row("iso_b")
        row_c = table.get_row("iso_c")
        row_b[:] = 12345
        np.testing.assert_array_equal(row_a, np.zeros(n_actions, dtype=np.int32))
        np.testing.assert_array_equal(row_c, np.zeros(n_actions, dtype=np.int32))
