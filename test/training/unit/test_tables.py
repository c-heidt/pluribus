"""Unit tests for the shared-memory table stack.

Covers:
- :class:`~poker_ai.tables.chunk_store.ChunkStore` — mmap lifecycle, dirty tracking,
  save and restore.
- :class:`~poker_ai.tables.chunked_table.ChunkedTable` — row allocation, data access,
  stripe locking, discount, naming, and orphan detection.
- :class:`~poker_ai.tables.cfr_tables.CFRTables` — per-street discount with correct
  REGRET_FLOOR clamping on regret tables and no clamping on strategy tables.
- Atomic I/O helpers from ``poker_ai/utils/io.py`` used by the checkpointing path.
"""

import os

import numpy as np
import pytest

from poker_ai.tables.chunk_store import CHUNK_SIZE, ChunkStore, _MAX_DIRTY_CHUNKS
from poker_ai.tables.cfr_tables import CFRTables, REGRET_FLOOR
from poker_ai.tables.index import InfosetIndex
from environment.action_space import MAX_ACTIONS_PER_STREET
from poker_ai.tables.chunked_table import (
    N_STRIPE_LOCKS,
    ChunkedTable,
    list_orphaned_blocks,
)


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def n_actions():
    return 5


@pytest.fixture
def table_name():
    return "pluribus_test"


@pytest.fixture
def tmp_lmdb(tmp_path):
    idx = InfosetIndex(tmp_path / "index")
    yield idx
    idx.close()


@pytest.fixture
def table(tmp_path, tmp_lmdb, n_actions, table_name):
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


@pytest.fixture
def cfr_tables(tmp_path):
    shm_dir = str(tmp_path / "shm_cfr")
    os.makedirs(shm_dir, exist_ok=True)
    tables = CFRTables(
        index_path=tmp_path / "lmdb_cfr",
        shm_dir=shm_dir,
        actions_per_street=MAX_ACTIONS_PER_STREET,
    )
    yield tables
    tables.close()


# ---------------------------------------------------------------------------
# Atomic I/O helpers
# ---------------------------------------------------------------------------


class TestAtomicIO:
    def test_atomic_joblib_roundtrip_dict(self, tmp_path):
        import joblib
        from poker_ai.tables.checkpoint import atomic_joblib_dump
        obj = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
        path = tmp_path / "test.joblib"
        atomic_joblib_dump(obj, path)
        assert joblib.load(path) == obj

    def test_atomic_joblib_roundtrip_numpy(self, tmp_path):
        import joblib
        from poker_ai.tables.checkpoint import atomic_joblib_dump
        arr = np.arange(1000, dtype=np.float64).reshape(20, 50)
        path = tmp_path / "arr.joblib"
        atomic_joblib_dump(arr, path)
        np.testing.assert_array_equal(arr, joblib.load(path))

    def test_atomic_joblib_no_partial_write_on_error(self, tmp_path):
        import joblib
        from poker_ai.tables.checkpoint import atomic_joblib_dump
        original = {"original": True}
        path = tmp_path / "safe.joblib"
        joblib.dump(original, path)
        original_mtime = path.stat().st_mtime
        with pytest.raises((RuntimeError, Exception)):
            atomic_joblib_dump(lambda x: x, path)
        assert path.stat().st_mtime == original_mtime
        assert joblib.load(path) == original

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((100,), np.float32),
            ((50, 20), np.float64),
            ((10, 10, 10), np.int32),
            ((1_000_000,), np.int32),
        ],
    )
    def test_atomic_numpy_roundtrip(self, tmp_path, shape, dtype):
        from utils.io import atomic_numpy_save
        arr = np.random.randint(0, 1000, size=shape).astype(dtype)
        path = tmp_path / "arr.npy"
        atomic_numpy_save(arr, path)
        np.testing.assert_array_equal(arr, np.load(path))


# ---------------------------------------------------------------------------
# ChunkStore
# ---------------------------------------------------------------------------


class TestChunkStore:
    def test_ensure_open_creates_file(self, tmp_path, n_actions, table_name):
        shm = str(tmp_path / "shm")
        os.makedirs(shm)
        store = ChunkStore(table_name, n_actions, shm)
        store.ensure_open(0)
        assert store.n_open == 1
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
        assert store.save_dirty(tmp_path, n_entries=10, prefix="test") == 1
        store.clear_dirty()
        assert store.save_dirty(tmp_path, n_entries=10, prefix="test") == 0
        store.close()
        store.unlink_all()

    def test_save_all_ignores_dirty_flags(self, tmp_path, n_actions, table_name):
        shm = str(tmp_path / "shm")
        os.makedirs(shm)
        store = ChunkStore(table_name, n_actions, shm)
        store.ensure_open(0)
        assert store.save_all(tmp_path, n_entries=10, prefix="test") == 1
        store.close()
        store.unlink_all()

    def test_restore_roundtrip(self, tmp_path, n_actions, table_name):
        shm = str(tmp_path / "shm")
        os.makedirs(shm)
        store = ChunkStore(table_name, n_actions, shm)
        data = np.full((5, n_actions), 42, dtype=np.int32)
        store.restore(0, data)
        np.testing.assert_array_equal(store.view(0)[:5], data)
        np.testing.assert_array_equal(store.view(0)[5], np.zeros(n_actions, dtype=np.int32))
        store.close()
        store.unlink_all()

    def test_dirty_out_of_bounds_ignored(self, tmp_path, n_actions, table_name):
        shm = str(tmp_path / "shm")
        os.makedirs(shm)
        store = ChunkStore(table_name, n_actions, shm)
        store.mark_dirty(_MAX_DIRTY_CHUNKS + 1)  # must not raise
        store.close()


# ---------------------------------------------------------------------------
# ChunkedTable construction
# ---------------------------------------------------------------------------


class TestChunkedTableConstruction:
    def test_n_actions_stored(self, table, n_actions):
        assert table.n_actions == n_actions

    def test_table_name_stored(self, table, table_name):
        assert table.table_name == table_name

    def test_zero_chunks_on_empty_index(self, table):
        assert table.n_chunks == 0

    def test_invalid_n_actions_raises(self, tmp_path, table_name):
        idx = InfosetIndex(tmp_path / "idx_bad")
        with pytest.raises(ValueError, match="n_actions"):
            ChunkedTable(n_actions=0, table_name=table_name, index=idx, shm_dir=str(tmp_path / "shm2"))
        idx.close()

    def test_chunk_file_created_in_shm_dir(self, table):
        table.get_row("first_infoset")
        assert table.n_chunks == 1
        assert os.path.exists(table.store.paths[0])

    def test_chunk_file_size_correct(self, table, n_actions):
        table.get_row("probe")
        assert os.path.getsize(table.store.paths[0]) == CHUNK_SIZE * n_actions * 4

    def test_resume_restores_existing_chunks(self, tmp_path, n_actions, table_name):
        idx_path = tmp_path / "idx_resume"
        shm_dir = str(tmp_path / "shm_resume")
        os.makedirs(shm_dir)
        idx1 = InfosetIndex(idx_path)
        tbl1 = ChunkedTable(n_actions, table_name, idx1, shm_dir)
        for k in range(100):
            tbl1.get_row(f"resume_is_{k}")[:] = k
        idx1.close()
        tbl1.close()

        idx2 = InfosetIndex(idx_path)
        tbl2 = ChunkedTable(n_actions, table_name, idx2, shm_dir)
        assert tbl2.n_chunks == 1
        for k in range(100):
            row = tbl2.get_row_if_exists(f"resume_is_{k}")
            assert row is not None
            np.testing.assert_array_equal(row, np.full(n_actions, k, dtype=np.int32))
        idx2.close()
        tbl2.close()
        tbl2.unlink_all()


# ---------------------------------------------------------------------------
# ChunkedTable access methods
# ---------------------------------------------------------------------------


class TestChunkedTableAccess:
    def test_get_row_zeros_on_first_access(self, table, n_actions):
        np.testing.assert_array_equal(
            table.get_row("brand_new"), np.zeros(n_actions, dtype=np.int32)
        )

    def test_get_row_write_then_read(self, table):
        row = table.get_row("writable")
        row[:] = [10, 20, 30, 40, 50]
        np.testing.assert_array_equal(table.get_row("writable"), [10, 20, 30, 40, 50])

    def test_get_row_returns_view_not_copy(self, table):
        r1 = table.get_row("view_test")
        r2 = table.get_row("view_test")
        r1[0] = 999
        assert r2[0] == 999

    def test_get_row_if_exists_none_for_unknown(self, table):
        assert table.get_row_if_exists("never_seen") is None

    def test_get_row_if_exists_returns_after_get_row(self, table, n_actions):
        row = table.get_row("exists_is")
        row[:] = list(range(n_actions))
        existing = table.get_row_if_exists("exists_is")
        assert existing is not None
        np.testing.assert_array_equal(existing, list(range(n_actions)))

    def test_get_row_by_location_matches_get_row(self, table):
        row_a = table.get_row("loc_test")
        row_a[:] = [1, 2, 3, 4, 5]
        flat_row = table._index.get("loc_test")
        chunk_id, row_idx = divmod(flat_row, CHUNK_SIZE)
        np.testing.assert_array_equal(table.get_row_by_location(chunk_id, row_idx), row_a)

    def test_get_row_by_location_invalid_row_raises(self, table):
        table.get_row("setup_chunk")
        with pytest.raises(IndexError):
            table.get_row_by_location(0, CHUNK_SIZE)

    def test_multiple_infosets_different_rows(self, table):
        rows = [table.get_row(f"multi_is_{k}") for k in range(10)]
        for k, row in enumerate(rows):
            row[:] = k
        for k in range(10):
            assert table.get_row(f"multi_is_{k}")[0] == k

    def test_chunk_boundary_allocation(self, table):
        n = CHUNK_SIZE + 5
        for k in range(n):
            table.get_row(f"cross_chunk_is_{k}")
        assert table.n_chunks >= 2
        flat_row = table._index.get(f"cross_chunk_is_{CHUNK_SIZE}")
        chunk_id, local_row = divmod(flat_row, CHUNK_SIZE)
        assert chunk_id == 1
        assert local_row == 0

    @pytest.mark.slow
    def test_multi_chunk_allocation_correct_retrieval(self, tmp_path, n_actions, table_name):
        n = 2 * CHUNK_SIZE + 5
        idx = InfosetIndex(tmp_path / "big_idx")
        shm_dir = str(tmp_path / "big_shm")
        os.makedirs(shm_dir)
        tbl = ChunkedTable(n_actions, table_name, idx, shm_dir)
        for k in range(n):
            tbl.get_row(f"big_is_{k}")[0] = k % (2 ** 31 - 1)
        assert tbl.n_chunks >= 3
        step = max(1, n // 1000)
        mismatches = 0
        for k in range(0, n, step):
            row = tbl.get_row_if_exists(f"big_is_{k}")
            if row is None or row[0] != k % (2 ** 31 - 1):
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
            assert table.get_stripe_lock(chunk_id) is table.get_stripe_lock(chunk_id)

    def test_stripe_assignment_formula(self, table):
        for chunk_id in range(N_STRIPE_LOCKS * 3):
            expected = table._stripe_locks[chunk_id % N_STRIPE_LOCKS]
            assert table.get_stripe_lock(chunk_id) is expected

    def test_different_chunks_same_stripe(self, table):
        for base in range(N_STRIPE_LOCKS):
            assert table.get_stripe_lock(base) is table.get_stripe_lock(base + N_STRIPE_LOCKS)

    def test_n_stripe_locks_equals_constant(self, table):
        assert len(table._stripe_locks) == N_STRIPE_LOCKS

    def test_stripe_locks_evenly_distributed(self):
        seen = {i % N_STRIPE_LOCKS for i in range(N_STRIPE_LOCKS)}
        assert len(seen) == N_STRIPE_LOCKS

    def test_stripe_lock_is_acquirable(self, table):
        lock = table.get_stripe_lock(0)
        assert lock.acquire(timeout=1.0)
        lock.release()


# ---------------------------------------------------------------------------
# ChunkedTable discount
# ---------------------------------------------------------------------------


class TestChunkedTableDiscount:
    def test_factor_validation(self, cfr_tables):
        with pytest.raises(ValueError, match="factor"):
            cfr_tables.apply_discount(0.0)
        with pytest.raises(ValueError, match="factor"):
            cfr_tables.apply_discount(1.1)

    def test_scales_values(self, cfr_tables):
        n_actions = cfr_tables.regret[0].n_actions
        initial = list(range(1000, 1000 + n_actions))
        for k in range(3):
            cfr_tables.regret[0].get_row(f"disc_is_{k}")[:] = initial
        cfr_tables.apply_discount(0.5)
        for k in range(3):
            # Discounting rounds to nearest (rint) rather than
            # truncating toward zero — truncation would bleed ~0.5 per
            # entry per application, which zeroes unit-scale strategy
            # counts.
            expected = np.rint(np.array(initial, dtype=np.float32) * 0.5)
            np.testing.assert_array_equal(
                cfr_tables.regret[0].get_row(f"disc_is_{k}"), expected
            )

    def test_small_strategy_counts_survive_mild_discount(self, cfr_tables):
        """A visit count of 1 must survive a late-window discount factor.

        With truncation, ``int(1 * 0.9) == 0`` erased every count
        written since the previous discount application; rounding keeps
        it alive (regression test for the strategy-mass wipe-out).
        """
        cfr_tables.strategy[0].get_row("tiny_mass")[:] = 1
        cfr_tables.apply_discount(0.9)
        assert np.all(cfr_tables.strategy[0].get_row("tiny_mass") == 1)

    def test_regret_floor_applied(self, cfr_tables):
        row = cfr_tables.regret[0].get_row("floor_is_0")
        row[:] = int(REGRET_FLOOR)
        cfr_tables.apply_discount(0.5)
        assert np.all(cfr_tables.regret[0].get_row("floor_is_0") >= REGRET_FLOOR)

    def test_factor_1_leaves_unchanged(self, cfr_tables):
        n_actions = cfr_tables.regret[0].n_actions
        initial = list(range(100, 100 + n_actions))
        for k in range(3):
            cfr_tables.regret[0].get_row(f"noop_is_{k}")[:] = initial
        cfr_tables.apply_discount(1.0)
        for k in range(3):
            np.testing.assert_array_equal(
                cfr_tables.regret[0].get_row(f"noop_is_{k}"), initial
            )


# ---------------------------------------------------------------------------
# CFRTables-level discount (regret vs. strategy split)
# ---------------------------------------------------------------------------


class TestCFRTablesDiscount:
    def test_scales_both_table_types(self, cfr_tables):
        """After apply_discount, both regret and strategy tables are scaled."""
        n = cfr_tables.regret[0].n_actions
        for street in range(4):
            cfr_tables.regret[street].get_row("both_test")[:] = 1000
            cfr_tables.strategy[street].get_row("both_test")[:] = 1000
        cfr_tables.apply_discount(0.5)
        for street in range(4):
            r_row = cfr_tables.regret[street].get_row("both_test")
            s_row = cfr_tables.strategy[street].get_row("both_test")
            assert np.all(r_row <= 500)
            assert np.all(s_row <= 500)

    def test_regret_floor_not_breached(self, cfr_tables):
        """After discounting a value at REGRET_FLOOR, it stays >= REGRET_FLOOR."""
        row = cfr_tables.regret[0].get_row("floor_test")
        row[:] = int(REGRET_FLOOR)
        cfr_tables.apply_discount(0.5)
        assert np.all(cfr_tables.regret[0].get_row("floor_test") >= REGRET_FLOOR)

    def test_strategy_no_floor_clamp(self, cfr_tables):
        """Strategy table values are scaled but not clamped to REGRET_FLOOR.

        Strategy values represent non-negative visit counts, so the meaningful
        lower bound is 0.  Applying REGRET_FLOOR (-310M) to strategy tables
        would be incorrect — a small positive strategy count discounted to 0
        is correct, not to REGRET_FLOOR.
        """
        row = cfr_tables.strategy[0].get_row("strat_test")
        row[:] = 1000
        cfr_tables.apply_discount(0.5)
        result = cfr_tables.strategy[0].get_row("strat_test")
        # Values should be around 500, not clamped to REGRET_FLOOR
        assert np.all(result >= 0)
        assert np.all(result < 1000)


class TestCFRTablesCopyIndexes:
    """``copy_indexes_to`` is the building block for the
    CheckpointManager's LMDB writeback path: it has to produce a
    snapshot per street under ``dst/street_{r}/`` that another
    :class:`CFRTables` can be opened against and that exposes the
    same infoset → row mapping as the source.
    """

    def test_round_trip_preserves_rows(self, cfr_tables, tmp_path):
        # Allocate a few infosets on every street through the live
        # CFRTables so each street's index has work to mirror.
        expected = {}
        for street in range(4):
            for i in range(15):
                key = f"r{street}_is{i}"
                cfr_tables.regret[street].update_row(key, 0, 1)
                expected[(street, key)] = cfr_tables._indexes[street].get(key)

        # Mirror to a fresh "persistent" directory.
        persistent = tmp_path / "persistent_indexes"
        cfr_tables.copy_indexes_to(persistent)

        # Per-street layout must match what CFRTables expects on init.
        for street in range(4):
            sd = persistent / f"street_{street}"
            assert sd.exists()
            assert (sd / "data.mdb").exists()

        # Open a fresh CFRTables pointing at the mirror; every key
        # should resolve to the same row id as in the original.
        shm_dir = str(tmp_path / "shm_clone")
        os.makedirs(shm_dir, exist_ok=True)
        clone = CFRTables(
            index_path=persistent,
            shm_dir=shm_dir,
            actions_per_street=MAX_ACTIONS_PER_STREET,
        )
        try:
            for (street, key), row in expected.items():
                assert clone._indexes[street].get(key) == row
        finally:
            clone.close()

    def test_overwrites_existing_destination(self, cfr_tables, tmp_path):
        """``copy_indexes_to`` must be idempotent so the CheckpointManager
        can call it repeatedly into the same temp directory across
        successive checkpoint writes.  LMDB's ``env.copy`` refuses
        non-empty target directories, so the aggregator has to clear
        each per-street directory first.
        """
        cfr_tables.regret[0].update_row("repeat_test", 0, 1)
        dst = tmp_path / "dst_dir"
        # First mirror.
        cfr_tables.copy_indexes_to(dst)
        # Allocate another row, then re-mirror to the same dst path.
        cfr_tables.regret[0].update_row("repeat_test_two", 0, 1)
        cfr_tables.copy_indexes_to(dst)
        # The second mirror should reflect both rows.
        shm_dir = str(tmp_path / "shm_clone2")
        os.makedirs(shm_dir, exist_ok=True)
        clone = CFRTables(
            index_path=dst,
            shm_dir=shm_dir,
            actions_per_street=MAX_ACTIONS_PER_STREET,
        )
        try:
            assert clone._indexes[0].get("repeat_test") is not None
            assert clone._indexes[0].get("repeat_test_two") is not None
        finally:
            clone.close()


# ---------------------------------------------------------------------------
# Naming and orphan detection
# ---------------------------------------------------------------------------


class TestChunkedTableOrphans:
    def test_chunk_name_format(self, table, table_name):
        table.get_row("name_test")
        assert os.path.basename(table.store.paths[0]) == f"{table_name}_000000"

    def test_chunk_names_sequential(self, table, table_name):
        for k in range(CHUNK_SIZE + 1):
            table.get_row(f"seq_is_{k}")
        assert table.n_chunks == 2
        for chunk_id in range(2):
            assert os.path.basename(table.store.paths[chunk_id]) == f"{table_name}_{chunk_id:06d}"

    def test_list_orphaned_blocks_empty_dir(self, tmp_path):
        orphan_dir = str(tmp_path / "empty_shm")
        os.makedirs(orphan_dir)
        assert list_orphaned_blocks(shm_dir=orphan_dir) == []

    def test_list_orphaned_blocks_finds_files(self, tmp_path):
        shm_dir = str(tmp_path / "orphan_shm")
        os.makedirs(shm_dir)
        for i in range(3):
            open(os.path.join(shm_dir, f"pluribus_regret_0_{i:06d}"), "wb").close()
        assert len(list_orphaned_blocks(shm_dir=shm_dir)) == 3

    def test_list_orphaned_blocks_ignores_non_pluribus(self, tmp_path):
        shm_dir = str(tmp_path / "mixed_shm")
        os.makedirs(shm_dir)
        p = os.path.join(shm_dir, "pluribus_regret_0_000000")
        open(p, "wb").close()
        open(os.path.join(shm_dir, "other_prefix_000000"), "wb").close()
        orphans = list_orphaned_blocks(shm_dir=shm_dir)
        assert len(orphans) == 1
        assert p in orphans

    def test_unlink_all_removes_files(self, tmp_path, n_actions, table_name):
        idx = InfosetIndex(tmp_path / "unlink_idx")
        shm_dir = str(tmp_path / "unlink_shm")
        os.makedirs(shm_dir)
        tbl = ChunkedTable(n_actions, table_name, idx, shm_dir)
        tbl.get_row("unlink_test")
        paths = list(tbl.store.paths)
        tbl.close()
        tbl.unlink_all()
        assert all(not os.path.exists(p) for p in paths)
        idx.close()

    def test_context_manager_unlinks(self, tmp_path, n_actions, table_name):
        idx = InfosetIndex(tmp_path / "cm_idx")
        shm_dir = str(tmp_path / "cm_shm")
        os.makedirs(shm_dir)
        with ChunkedTable(n_actions, table_name, idx, shm_dir) as tbl:
            tbl.get_row("cm_test")
            paths = list(tbl.store.paths)
        assert all(not os.path.exists(p) for p in paths)
        idx.close()


# ---------------------------------------------------------------------------
# Row allocation counters
# ---------------------------------------------------------------------------


class TestChunkedTableAllocation:
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

    def test_merge_delta_creates_row(self, table, n_actions):
        delta = np.full(n_actions, 7, dtype=np.int64)
        table.merge_delta_row("new_infoset", delta)
        row = table.get_row_if_exists("new_infoset")
        assert row is not None
        np.testing.assert_array_equal(row, np.full(n_actions, 7, dtype=np.int32))

    def test_merge_delta_accumulates(self, table, n_actions):
        delta = np.ones(n_actions, dtype=np.int64) * 3
        table.merge_delta_row("accum_is", delta)
        table.merge_delta_row("accum_is", delta)
        np.testing.assert_array_equal(
            table.get_row_if_exists("accum_is"), np.full(n_actions, 6, dtype=np.int32)
        )

    def test_merge_delta_negative(self, table, n_actions):
        delta = np.full(n_actions, -5, dtype=np.int64)
        table.merge_delta_row("neg_is", delta)
        np.testing.assert_array_equal(
            table.get_row_if_exists("neg_is"), np.full(n_actions, -5, dtype=np.int32)
        )
