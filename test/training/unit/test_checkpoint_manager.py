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
