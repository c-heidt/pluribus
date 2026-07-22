"""Deferred-durability allocator — cache allocates, LMDB persists in bulk.

With ``PLURIBUS_DEFERRED_ALLOC=1`` the shm index cache assigns row numbers under
its own lightweight lock (no LMDB write txn on the hot path); LMDB is brought
current in bulk at checkpoints / run-end via ``bulk_persist``. These gates prove
it is a drop-in for the LMDB-synchronous allocator:

* **Differential** — the same info-set stream through deferred vs synchronous
  allocation gives the identical ``(row, is_new)`` sequence and row count
  (single-process determinism ⇒ the golden trace stays byte-identical, verified
  separately).
* **Durability/resume** — LMDB lags during a run, is exact after a flush, and a
  fresh reopen + ``prewarm_from_cursor`` rebuilds a cache that ``audit``s clean
  with ``occupancy == n_allocated``.
* **Concurrency** — forked workers allocating distinct + shared info sets end
  with exactly ``distinct-count`` rows (no double claim), then persist + audit
  clean.
"""

import multiprocessing as mp

import numpy as np
import pytest

from environment.action_space import MAX_ACTIONS_PER_STREET
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.tables.index import InfosetIndex, hash_info_set_128
from poker_ai.tables.index import lmdb_map_size_for_players
from poker_ai.tables.shm_index_cache import ShmIndexCache

_N_PLAYERS = 2
_CAPS = {r: 1 << 16 for r in range(4)}


def _cache_index(tmp_path, name, deferred, cap=1 << 14):
    idx = InfosetIndex(tmp_path / name, deferred=deferred)
    cache = ShmIndexCache(name, capacity=cap, shm_dir=str(tmp_path / "shm"),
                          load_factor=0.5)
    idx.set_cache(cache)
    idx.prewarm_cache()
    return idx, cache


# ---------------------------------------------------------------------------
# Differential: deferred ≡ synchronous (single process)
# ---------------------------------------------------------------------------


class TestDeferredEqualsSync:
    def _stream(self):
        b1 = [b"k:%d" % i for i in range(30)]
        b2 = [b"k:%d" % i for i in range(20, 50)]
        return b1, b2

    def test_row_sequence_matches_sync(self, tmp_path):
        b1, b2 = self._stream()
        sync, _ = _cache_index(tmp_path, "sync", deferred=False)
        sync_res = sync.get_or_create_many(b1) + sync.get_or_create_many(b2)
        sync_n = sync.n_allocated_rows
        sync.close()

        dfr, _ = _cache_index(tmp_path, "dfr", deferred=True)
        dfr_res = dfr.get_or_create_many(b1) + dfr.get_or_create_many(b2)

        assert dfr_res == sync_res
        assert dfr.n_allocated_rows == sync_n == 50
        assert sorted(r for r, isnew in dfr_res if isnew) == list(range(50))
        dfr.close()

    def test_single_get_or_create_deferred(self, tmp_path):
        dfr, _ = _cache_index(tmp_path, "single", deferred=True)
        r0, n0 = dfr.get_or_create(b"a")
        r1, n1 = dfr.get_or_create(b"b")
        r0b, n0b = dfr.get_or_create(b"a")   # hit
        assert (r0, n0, r1, n1) == (0, True, 1, True)
        assert (r0b, n0b) == (0, False)
        assert dfr.n_allocated_rows == 2
        dfr.close()

    def test_intra_batch_duplicate_collapses(self, tmp_path):
        dfr, _ = _cache_index(tmp_path, "dup", deferred=True)
        res = dfr.get_or_create_many([b"x", b"y", b"x", b"y", b"z"])
        rows = [r for r, _ in res]
        assert rows[0] == rows[2] and rows[1] == rows[3]
        assert dfr.n_allocated_rows == 3
        assert [isnew for _, isnew in res] == [True, True, False, False, True]
        dfr.close()


# ---------------------------------------------------------------------------
# Durability / resume
# ---------------------------------------------------------------------------


class TestDurabilityResume:
    def test_lmdb_lags_then_exact_after_flush(self, tmp_path):
        dfr, cache = _cache_index(tmp_path, "dur", deferred=True)
        dfr.get_or_create_many([b"k:%d" % i for i in range(40)])
        assert dfr.n_allocated_rows == 40              # cache authority
        # LMDB watermark still 0 (nothing persisted yet).
        assert dfr._read_next_row() == 0
        assert dfr.bulk_persist() == 40
        assert dfr._read_next_row() == 40              # now exact
        # A second flush with no new rows is a no-op.
        assert dfr.bulk_persist() == 0
        dfr.close()

    def test_reopen_prewarm_audits_clean(self, tmp_path):
        dfr, _ = _cache_index(tmp_path, "res", deferred=True)
        keys = [b"k:%d" % i for i in range(40)]
        dfr.get_or_create_many(keys)
        dfr.bulk_persist()
        dfr.close()

        # Fresh reopen: LMDB is the source, prewarm rebuilds the cache.
        idx2, cache2 = _cache_index(tmp_path, "res", deferred=True)
        assert idx2.n_allocated_rows == 40
        assert cache2.occupancy() == 40
        assert cache2.audit(idx2._env) == 40
        # persist watermark seeded to occupancy → new allocs only flush the delta.
        assert idx2._persist_watermark == 40
        idx2.get_or_create(b"new")
        assert idx2.bulk_persist() == 1                # only the 1 new row
        idx2.close()


# ---------------------------------------------------------------------------
# Concurrency (forked workers via CFRTables) + persist + audit
# ---------------------------------------------------------------------------


_DISTINCT = 4000
_SHARED = 500


def _cfr_worker(w, tables, barrier):
    tables.reopen_after_fork()
    table = tables.regret[1]
    delta = np.ones(table.n_actions, dtype=np.int64)
    items = [(b"w%d:%d" % (w, i), delta) for i in range(_DISTINCT)]
    items += [(b"shared:%d" % i, delta) for i in range(_SHARED)]
    barrier.wait()
    table.merge_delta_rows(items)


@pytest.mark.parametrize("n_workers", [4, 11])
def test_concurrent_deferred_no_double_alloc(tmp_path, n_workers, monkeypatch):
    monkeypatch.setenv("PLURIBUS_DEFERRED_ALLOC", "1")
    shm = tmp_path / "shm"
    shm.mkdir(parents=True, exist_ok=True)
    tables = CFRTables(
        index_path=tmp_path / "lmdb_index",
        shm_dir=str(shm),
        actions_per_street=MAX_ACTIONS_PER_STREET,
        lmdb_map_size=lmdb_map_size_for_players(_N_PLAYERS),
        enable_index_cache=True,
        index_capacities={r: 1 << 20 for r in range(4)},
    )
    assert tables._deferred_alloc is True
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
    expected = n_workers * _DISTINCT + _SHARED
    # No double-claim: cache occupancy == number of distinct info sets.
    assert idx.n_allocated_rows == expected
    assert cache.occupancy() == expected
    # LMDB lags until we persist; then it is exact and audits clean.
    n = tables.persist_indexes()
    assert n == expected
    assert cache.audit(idx._env) == expected
    # Every shared key resolves to a single row.
    shared_rows = {cache.probe(*hash_info_set_128(b"shared:%d" % i))
                   for i in range(_SHARED)}
    assert None not in shared_rows and len(shared_rows) == _SHARED
    tables.close()


def test_deferred_flag_off_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("PLURIBUS_DEFERRED_ALLOC", raising=False)
    shm = tmp_path / "shm"
    shm.mkdir(parents=True, exist_ok=True)
    tables = CFRTables(
        index_path=tmp_path / "lmdb_index",
        shm_dir=str(shm),
        actions_per_street=MAX_ACTIONS_PER_STREET,
        lmdb_map_size=lmdb_map_size_for_players(_N_PLAYERS),
        enable_index_cache=True,
        index_capacities=_CAPS,
    )
    assert tables._deferred_alloc is False
    assert tables.persist_indexes() == 0   # no-op on the synchronous path
    tables.close()
