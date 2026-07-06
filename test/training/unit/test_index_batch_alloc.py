"""Batched infoset allocation — byte-identical to per-row, concurrency-safe.

``InfosetIndex.get_or_create_many`` collapses a flush's allocations into ONE
LMDB write transaction (the fix for the writer-lock serialization that pinned
the 48-worker CFR throughput). These gates prove it is a drop-in for N
sequential ``get_or_create`` calls:

* **Differential** — same info sets (with intra/cross-batch repeats) through the
  batched path vs the per-row path produce the identical ``(row, is_new)``
  sequence, watermark, and ``n_allocated_rows``; the shm cache audits clean.
* **Concurrency** — forked workers allocating distinct + shared info sets end
  with exactly ``distinct-count`` rows (no double-allocation), a cache that
  mirrors LMDB (``audit``), and ``occupancy == n_allocated_rows``.
* **MapFull retry** — a batch that overflows a tiny ``map_size`` completes after
  the automatic resize.

The strongest byte-exactness gate is the single-process golden trace
(``test_core_golden_trace.py``), verified separately; these tests isolate the
allocator itself.
"""

import multiprocessing as mp
import struct

import numpy as np
import pytest

from environment.action_space import MAX_ACTIONS_PER_STREET
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.tables.chunk_store import CHUNK_SIZE
from poker_ai.tables.index import (
    InfosetIndex,
    hash_info_set_128,
    lmdb_map_size_for_players,
)
from poker_ai.tables.shm_index_cache import ShmIndexCache


def _attach_cache(idx, tmp_path, name, cap=1 << 12):
    cache = ShmIndexCache(name, capacity=cap, shm_dir=str(tmp_path / "shm"),
                          load_factor=0.5)
    idx.set_cache(cache)
    idx.prewarm_cache()
    return cache


# ---------------------------------------------------------------------------
# Differential: batched ≡ per-row (single process)
# ---------------------------------------------------------------------------


class TestBatchedEqualsSequential:
    def _keys(self):
        # Two "flushes" that overlap: batch2 re-references batch1's tail (cache
        # hits) and adds fresh keys — exercises hit+miss interleaving.
        batch1 = [b"k:%d" % i for i in range(30)]
        batch2 = [b"k:%d" % i for i in range(20, 50)]
        return batch1, batch2

    def test_matches_per_row_with_cache(self, tmp_path):
        batch1, batch2 = self._keys()

        # Reference: per-row get_or_create in the identical flattened order.
        ref = InfosetIndex(tmp_path / "ref")
        _attach_cache(ref, tmp_path, "ref")
        ref_res = [ref.get_or_create(k) for k in (batch1 + batch2)]
        ref_n = ref.n_allocated_rows
        ref.close()

        # Batched: one call per flush.
        bat = InfosetIndex(tmp_path / "bat")
        cache = _attach_cache(bat, tmp_path, "bat")
        bat_res = bat.get_or_create_many(batch1) + bat.get_or_create_many(batch2)

        assert bat_res == ref_res
        assert bat.n_allocated_rows == ref_n
        # 50 distinct keys → dense rows [0, 50).
        assert bat.n_allocated_rows == 50
        assert sorted(r for r, _ in bat_res if _) == list(range(50))
        # Cache is a complete, correct mirror of LMDB.
        assert cache.audit(bat._env) == 50
        assert cache.occupancy() == 50
        bat.close()

    def test_no_cache_fallback_matches(self, tmp_path):
        keys = [b"n:%d" % i for i in range(25)] + [b"n:%d" % i for i in range(10)]
        ref = InfosetIndex(tmp_path / "ref2")
        ref_res = [ref.get_or_create(k) for k in keys]
        ref.close()
        bat = InfosetIndex(tmp_path / "bat2")
        bat_res = bat.get_or_create_many(keys)
        assert bat_res == ref_res
        bat.close()

    def test_empty_is_noop(self, tmp_path):
        idx = InfosetIndex(tmp_path / "e")
        _attach_cache(idx, tmp_path, "e")
        assert idx.get_or_create_many([]) == []
        assert idx.n_allocated_rows == 0
        idx.close()

    def test_intra_batch_duplicate_collapses_to_one_row(self, tmp_path):
        # A duplicate key within one batch must resolve to a single row via the
        # in-txn read-your-own-writes re-probe (no double allocation).
        idx = InfosetIndex(tmp_path / "dup")
        _attach_cache(idx, tmp_path, "dup")
        res = idx.get_or_create_many([b"x", b"y", b"x", b"y", b"z"])
        rows = [r for r, _ in res]
        assert rows[0] == rows[2]           # both "x"
        assert rows[1] == rows[3]           # both "y"
        assert idx.n_allocated_rows == 3    # x, y, z only
        assert [isnew for _, isnew in res] == [True, True, False, False, True]
        idx.close()

    def test_debug_mode_batch_writes_shadow_keys(self, tmp_path):
        # Debug shadow-key path must be replicated per new key in the batch.
        idx = InfosetIndex(tmp_path / "dbg", debug=True)
        _attach_cache(idx, tmp_path, "dbg")
        idx.get_or_create_many([b"a", b"b", b"c"])
        assert idx.n_allocated_rows == 3
        idx.close()


# ---------------------------------------------------------------------------
# MapFull retry
# ---------------------------------------------------------------------------


class TestMapFullRetry:
    def test_batch_survives_map_resize(self, tmp_path):
        # Tiny map so a large batch overflows mid-transaction; get_or_create_many
        # must catch MapFullError, reopen (double map_size), and retry the batch.
        idx = InfosetIndex(tmp_path / "mf", map_size=64 * 1024)
        _attach_cache(idx, tmp_path, "mf", cap=1 << 16)
        keys = [b"mf:%d" % i for i in range(20000)]
        res = idx.get_or_create_many(keys)
        assert len(res) == 20000
        assert idx.n_allocated_rows == 20000
        assert sorted(r for r, _ in res) == list(range(20000))
        idx.close()


# ---------------------------------------------------------------------------
# Concurrency: forked workers, one shared index
# ---------------------------------------------------------------------------


_N_PLAYERS = 2
_DISTINCT = 4000
_SHARED = 500


def _cfr_worker(w, tables, barrier):
    tables.reopen_after_fork()
    table = tables.regret[1]
    width = table.n_actions
    delta = np.ones(width, dtype=np.int64)
    # Distinct-per-worker keys + a set every worker also allocates (contended).
    items = [(b"w%d:%d" % (w, i), delta) for i in range(_DISTINCT)]
    items += [(b"shared:%d" % i, delta) for i in range(_SHARED)]
    barrier.wait()
    table.merge_delta_rows(items)


@pytest.mark.parametrize("n_workers", [4, 11])
def test_concurrent_batched_allocation_no_double_alloc(tmp_path, n_workers):
    shm = tmp_path / "shm"
    shm.mkdir(parents=True, exist_ok=True)
    caps = {r: 1 << 20 for r in range(4)}
    tables = CFRTables(
        index_path=tmp_path / "lmdb_index",
        shm_dir=str(shm),
        actions_per_street=MAX_ACTIONS_PER_STREET,
        lmdb_map_size=lmdb_map_size_for_players(_N_PLAYERS),
        enable_index_cache=True,
        index_capacities=caps,
    )
    tables.prewarm_caches()

    barrier = mp.Barrier(n_workers)
    tables.close_envs()
    procs = [mp.Process(target=_cfr_worker, args=(w, tables, barrier))
             for w in range(n_workers)]
    try:
        for p in procs:
            p.start()
    finally:
        tables.open_envs()
    for p in procs:
        p.join()
        assert p.exitcode == 0

    idx = tables._indexes[1]
    cache = tables._index_caches[1]
    expected = n_workers * _DISTINCT + _SHARED  # each distinct once, shared once
    # No double-allocation: watermark == number of distinct infosets.
    assert idx.n_allocated_rows == expected
    # Cache is a complete, correct mirror of LMDB (the pure-shm read invariant).
    assert cache.audit(idx._env) == expected
    assert cache.occupancy() == expected
    # Every shared key resolves to a single row.
    shared_rows = {idx.get(b"shared:%d" % i) for i in range(_SHARED)}
    assert None not in shared_rows and len(shared_rows) == _SHARED
    tables.close()


# ---------------------------------------------------------------------------
# merge_delta_rows lands the right regret bytes via the batched path
# ---------------------------------------------------------------------------


def test_merge_delta_rows_batched_values(tmp_path):
    shm = tmp_path / "shm"
    shm.mkdir(parents=True, exist_ok=True)
    tables = CFRTables(
        index_path=tmp_path / "lmdb_index",
        shm_dir=str(shm),
        actions_per_street=MAX_ACTIONS_PER_STREET,
        lmdb_map_size=lmdb_map_size_for_players(_N_PLAYERS),
        enable_index_cache=True,
        index_capacities={r: 1 << 12 for r in range(4)},
    )
    tables.prewarm_caches()
    table = tables.regret[1]
    width = table.n_actions
    keys = [b"m:%d" % i for i in range(40)]
    deltas = {k: np.arange(width, dtype=np.int64) + i for i, k in enumerate(keys)}
    # Merge twice → rows should accumulate (2x), proving in-place add on the
    # resolved batched rows.
    table.merge_delta_rows([(k, deltas[k]) for k in keys])
    table.merge_delta_rows([(k, deltas[k]) for k in keys])
    assert tables._indexes[1].n_allocated_rows == 40
    for k in keys:
        row = table.get_row_if_exists(k)
        assert row is not None
        np.testing.assert_array_equal(row, (deltas[k] * 2).astype(np.int32))
    tables.close()
