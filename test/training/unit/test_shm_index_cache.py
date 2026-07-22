"""Tests for :class:`poker_ai.tables.shm_index_cache.ShmIndexCache`.

The cache is a fork-shared open-addressing ``digest -> flat_row`` hash table in
``/dev/shm``.  These tests pin the correctness properties the design depends
on: round-trip, probe wraparound, full-128-bit verification (a wrong key is a
miss, never a wrong row), loud overflow, fork-shared visibility, concurrent
insert without loss, and prewarm/audit against a real LMDB index.
"""

import multiprocessing as mp

import numpy as np
import pytest

from poker_ai.tables.shm_index_cache import (
    ShmIndexCache,
    capacity_for,
    next_pow2,
)


@pytest.fixture
def cache(tmp_path):
    c = ShmIndexCache("t", capacity=64, shm_dir=str(tmp_path), load_factor=0.5)
    yield c
    c.close()
    c.unlink()


# ---------------------------------------------------------------------------
# Sizing helpers
# ---------------------------------------------------------------------------


class TestSizing:
    def test_next_pow2(self):
        assert next_pow2(1) == 1
        assert next_pow2(3) == 4
        assert next_pow2(16) == 16
        assert next_pow2(17) == 32

    def test_capacity_for_respects_load_factor(self):
        # 100 rows at load 0.5 -> >=200 slots -> next pow2 = 256.
        assert capacity_for(100, 0.5) == 256

    def test_capacity_is_power_of_two(self, cache):
        assert cache.capacity == 64  # already pow2


# ---------------------------------------------------------------------------
# Core probe / insert
# ---------------------------------------------------------------------------


class TestProbeInsert:
    def test_insert_then_get_returns_row(self, cache):
        cache.insert(111, 222, 9)
        assert cache.probe(111, 222) == 9

    def test_get_missing_returns_none(self, cache):
        assert cache.probe(1, 2) is None

    def test_insert_is_idempotent(self, cache):
        cache.insert(5, 6, 3)
        cache.insert(5, 6, 3)
        assert cache.occupancy() == 1
        assert cache.probe(5, 6) == 3

    def test_probe_wraparound(self, cache):
        # All keys hash to the last bucket (low & (cap-1) == cap-1), forcing
        # the probe to wrap past index 0.
        cap = cache.capacity
        base = cap - 1
        for i in range(5):
            cache.insert(base + i * cap, 0, 100 + i)
        for i in range(5):
            assert cache.probe(base + i * cap, 0) == 100 + i

    def test_wrong_key_same_bucket_is_miss_not_wrong_row(self, cache):
        # Two distinct 128-bit keys sharing a bucket: the second must miss,
        # never return the first's row (full-key verification).
        cache.insert(7, 1, 42)
        assert cache.probe(7, 2) is None          # same low, different high
        assert cache.probe(7 + cache.capacity, 1) is None  # same bucket, diff low

    def test_overflow_fails_loud(self, tmp_path):
        c = ShmIndexCache("of", capacity=8, shm_dir=str(tmp_path), load_factor=0.5)
        try:
            # max_occupancy = 8 * 0.5 = 4.
            for i in range(4):
                c.insert(i, 0, i)
            with pytest.raises(IndexError, match="full"):
                c.insert(99, 0, 99)
        finally:
            c.close()
            c.unlink()

    def test_row_equal_to_sentinel_rejected(self, cache):
        with pytest.raises(ValueError, match="sentinel"):
            cache.insert(1, 2, (1 << 64) - 1)


# ---------------------------------------------------------------------------
# Fork-shared behaviour
# ---------------------------------------------------------------------------


def _child_read_and_write(cache, ready, done, result):
    # Child sees the parent's pre-fork insert via the inherited MAP_SHARED mmap.
    result.value = cache.probe(11, 22) or -1
    cache.insert(33, 44, 77)  # visible back in the parent
    done.set()


class TestForkShared:
    def test_fork_shared_visibility(self, tmp_path):
        ctx = mp.get_context("fork")
        c = ShmIndexCache("fk", capacity=64, shm_dir=str(tmp_path), load_factor=0.5)
        try:
            c.insert(11, 22, 55)  # before fork
            ready = ctx.Event()
            done = ctx.Event()
            result = ctx.Value("q", 0)
            p = ctx.Process(
                target=_child_read_and_write, args=(c, ready, done, result)
            )
            p.start()
            p.join(timeout=30)
            assert p.exitcode == 0
            assert result.value == 55            # child saw parent's insert
            assert c.probe(33, 44) == 77         # parent sees child's insert
        finally:
            c.close()
            c.unlink()


def _insert_disjoint(cache, start, count, done):
    for i in range(start, start + count):
        cache.insert(i, i * 7 + 1, i)
    done.set()


class TestConcurrentInsert:
    def test_concurrent_insert_no_loss(self, tmp_path):
        ctx = mp.get_context("fork")
        # 4 procs * 50 keys = 200 entries; capacity 1024 keeps load < 0.5.
        c = ShmIndexCache("cc", capacity=1024, shm_dir=str(tmp_path), load_factor=0.5)
        try:
            procs = []
            dones = []
            for w in range(4):
                done = ctx.Event()
                dones.append(done)
                p = ctx.Process(
                    target=_insert_disjoint, args=(c, w * 50, 50, done)
                )
                p.start()
                procs.append(p)
            for p in procs:
                p.join(timeout=30)
                assert p.exitcode == 0
            # Every key retrievable and nothing double-counted.
            for i in range(200):
                assert c.probe(i, i * 7 + 1) == i, i
            assert c.occupancy() == 200
        finally:
            c.close()
            c.unlink()


# ---------------------------------------------------------------------------
# Prewarm / audit against a real LMDB index
# ---------------------------------------------------------------------------


class TestPrewarmAudit:
    def test_prewarm_matches_lmdb(self, tmp_path):
        from poker_ai.tables.index import InfosetIndex, hash_info_set_128

        idx = InfosetIndex(tmp_path / "lmdb")
        keys = [f"info_{i}".encode() for i in range(50)]
        expected = {}
        for k in keys:
            row, _ = idx.get_or_create(k)
            expected[k] = row

        cap = capacity_for(len(keys) * 2)
        c = ShmIndexCache("pw", capacity=cap, shm_dir=str(tmp_path / "shm"))
        try:
            loaded = c.prewarm_from_cursor(idx._env)
            assert loaded == len(keys)
            # Every key resolves to the same row LMDB assigned.
            for k, row in expected.items():
                low, high = hash_info_set_128(k)
                assert c.probe(low, high) == row
            # And the built-in audit agrees.
            assert c.audit(idx._env) == len(keys)
        finally:
            c.close()
            c.unlink()
            idx.close()

    def test_prewarm_skips_reserved_keys(self, tmp_path):
        # The __next_row__ watermark (len != 16) must not be loaded as a digest.
        from poker_ai.tables.index import InfosetIndex

        idx = InfosetIndex(tmp_path / "lmdb")
        for i in range(5):
            idx.get_or_create(f"k{i}".encode())
        c = ShmIndexCache("rk", capacity=64, shm_dir=str(tmp_path / "shm"))
        try:
            loaded = c.prewarm_from_cursor(idx._env)
            assert loaded == 5  # exactly the 5 digests, not the watermark
        finally:
            c.close()
            c.unlink()
            idx.close()
