"""Unit tests for :class:`poker_ai.tables.checkpoint.CheckpointManager`.

Focused on the non-blocking skip-coalesce guard in
:meth:`CheckpointManager.checkpoint`.  The guard's correctness rests
entirely on bailing *before* ``snapshot_dirty_chunks`` runs: that call
clears the chunk dirty flags, so a skip that happened after it would
silently drop the chunks dirtied since the last successful checkpoint.
These tests pin that ordering down with a fake server so they stay fast
and free of any real training machinery.
"""

from poker_ai.tables.checkpoint import CheckpointManager

import pytest


class _FakeTables:
    def __init__(self):
        self.snapshot_calls = 0
        self.flush_index_calls = 0
        self.persist_index_calls = 0

    def persist_indexes(self):
        # Deferred-allocation bulk flush; a no-op on the synchronous path.
        self.persist_index_calls += 1
        return 0

    def snapshot_dirty_chunks(self):
        # Returning [] keeps the (now-stopped) writer's work trivial if
        # the snapshot path is exercised; the call counter is what the
        # tests assert on.
        self.snapshot_calls += 1
        return []

    def flush_indexes(self):
        self.flush_index_calls += 1


class _FakeServer:
    def __init__(self):
        self._tables = _FakeTables()
        self.flush_worker_calls = 0

    def flush_all_workers(self):
        self.flush_worker_calls += 1

    def to_dict(self, t=None):
        return {"t": t}


@pytest.fixture
def manager(tmp_path):
    """A CheckpointManager whose background writer is stopped.

    Stopping the writer makes the ``maxsize=1`` write queue
    deterministic: nothing we enqueue is consumed behind the test's
    back, so ``full()`` reflects exactly what the test put there.  LMDB
    writeback is disabled (no ``lmdb_runtime_dir``) so the checkpoint
    path needs nothing beyond the fake server.
    """
    server = _FakeServer()
    mgr = CheckpointManager(server, tmp_path)
    mgr.shutdown()  # join the daemon writer thread
    assert not mgr._writer_thread.is_alive()
    yield mgr, server


class TestCheckpointSkipCoalesce:
    def test_skips_when_writer_busy_without_snapshotting(self, manager):
        """A scheduled checkpoint must bail when a prior snapshot is
        still queued — and must do so before touching the tables, so the
        dirty flags survive for the next checkpoint."""
        mgr, server = manager
        # Simulate a snapshot the writer has not yet drained.
        mgr._write_queue.put(object())
        assert mgr._write_queue.full()

        mgr.checkpoint(t=42)  # scheduled (emergency=False, wait=False)

        # Guard returned early: no snapshot, no worker flush, no index
        # flush — nothing that would clear or mutate training state.
        assert server._tables.snapshot_calls == 0
        assert server._tables.flush_index_calls == 0
        assert server.flush_worker_calls == 0
        # The previously-queued snapshot is left untouched.
        assert mgr._write_queue.full()

    def test_enqueues_when_writer_idle(self, manager):
        """When the queue has a free slot the scheduled checkpoint
        snapshots and hands the result to the writer without blocking."""
        mgr, server = manager
        assert not mgr._write_queue.full()

        mgr.checkpoint(t=7)  # scheduled

        assert server._tables.snapshot_calls == 1
        assert server._tables.flush_index_calls == 1
        assert server.flush_worker_calls == 1
        # Snapshot was handed off (writer is stopped, so it stays queued).
        assert mgr._write_queue.full()

    def test_emergency_is_never_skipped(self, manager):
        """Emergency/final checkpoints must never hit the skip guard —
        they drain the queue and run the write inline so the process
        cannot exit with unsaved state.  With the writer stopped and the
        queue empty, the inline write completes against the save dir."""
        mgr, server = manager
        mgr.checkpoint(t=99, emergency=True)
        assert server._tables.snapshot_calls == 1
        # The inline write produced a real checkpoint directory.
        assert list(mgr._save_path.glob("checkpoint_[0-9]*"))


class TestCheckpointRetention:
    """Every checkpoint is retained as an average-strategy snapshot — the
    previous generation is no longer deleted.  These tests pin the
    non-deleting invariant and the unique, monotonic naming that keeps
    retained generations from colliding when written in the same second.
    """

    def test_previous_checkpoint_is_not_deleted(self, manager):
        """A second checkpoint must leave the first on disk (it is a
        retained snapshot), not roll it over as the old behaviour did."""
        mgr, server = manager

        mgr.checkpoint(t=1, emergency=True)
        first = mgr._last_checkpoint_path
        assert first is not None and first.exists()

        mgr.checkpoint(t=2, emergency=True)
        second = mgr._last_checkpoint_path

        # Both generations survive; they are distinct directories.
        assert first != second
        assert first.exists()
        assert second.exists()
        assert len(list(mgr._save_path.glob("checkpoint_[0-9]*"))) == 2

    def test_same_second_names_are_unique_and_monotonic(self, manager, monkeypatch):
        """Checkpoints written within the same wall-clock second must get
        distinct, strictly increasing names so the retained dirs neither
        collide on rename nor confuse the lexical 'latest' resume pick.

        The clock is pinned to a constant so every write reports the same
        second, deterministically exercising the collision-bump path (the
        exact scenario that would crash a same-second rename onto a
        non-empty retained dir)."""
        import poker_ai.tables.checkpoint as checkpoint_mod

        mgr, server = manager
        monkeypatch.setattr(checkpoint_mod.time, "time", lambda: 1_700_000_000.0)

        names = []
        for t in range(3):
            mgr.checkpoint(t=t, emergency=True)
            names.append(mgr._last_checkpoint_path.name)

        # Same second for all three, yet all names unique and strictly
        # increasing (the bump path fired), and creation order == sort order.
        assert names == ["checkpoint_1700000000",
                         "checkpoint_1700000001",
                         "checkpoint_1700000002"]
        assert len(set(names)) == 3
        assert names == sorted(names)
        # Every generation is still on disk (nothing deleted, nothing clobbered).
        assert len(list(mgr._save_path.glob("checkpoint_[0-9]*"))) == 3

    def test_hardlink_carry_forward_sources_from_latest(self, manager, tmp_path):
        """Unchanged chunk files carry forward via hardlink from the most
        recent retained checkpoint, so a new generation is self-contained
        while sharing inodes with the one before it."""
        import numpy as np

        mgr, server = manager

        # First generation carries a chunk file.
        def _one_chunk():
            server._tables.snapshot_calls += 1
            return [("regret_0_chunk_000000.npy", np.zeros((2, 3), dtype=np.int32))]

        server._tables.snapshot_dirty_chunks = _one_chunk
        mgr.checkpoint(t=1, emergency=True)
        first = mgr._last_checkpoint_path
        first_chunk = first / "regret_0_chunk_000000.npy"
        assert first_chunk.exists()

        # Second generation writes no new chunks — the file must appear via
        # a hardlink to the first (same inode), and both dirs still exist.
        server._tables.snapshot_dirty_chunks = lambda: []
        mgr.checkpoint(t=2, emergency=True)
        second = mgr._last_checkpoint_path
        second_chunk = second / "regret_0_chunk_000000.npy"
        assert second_chunk.exists()
        assert first_chunk.stat().st_ino == second_chunk.stat().st_ino
        assert first.exists() and second.exists()
