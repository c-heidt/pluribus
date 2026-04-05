"""Unit tests for Phase 4 — Worker local delta sync protocol.

Covers:
- 4.3  ``_sync_to_master()`` flushes local delta into agent.regret
- 4.4  ``sync`` and ``terminate`` job handlers
- 4.5  NUMA-pinning helpers
- 4.6  ``_local_delta`` empty after sync; LUT not loaded before fork
"""
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pytest

from poker_ai.ai.cfr_tables import CFRTables
from poker_ai.ai.ai import ACTION_TO_IDX, MAX_ACTIONS_PER_STREET
from poker_ai.ai.multiprocess.worker import Worker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_worker(tmp_path: Path) -> tuple:
    """Instantiate a Worker suitable for unit testing (not started)."""
    shm_dir = str(tmp_path / "shm")
    os.makedirs(shm_dir, exist_ok=True)
    tables = CFRTables(
        index_path=tmp_path / "lmdb_index",
        shm_dir=shm_dir,
        actions_per_street=MAX_ACTIONS_PER_STREET,
    )
    locks = {
        "strategy_update_lock": mp.Lock(),
    }
    job_queue = mp.JoinableQueue()
    logging_queue = mp.Queue()
    worker = Worker(
        job_queue=job_queue,
        logging_queue=logging_queue,
        locks=locks,
        tables=tables,
        lut_path=tmp_path,
        pickle_dir=False,
        n_players=2,
        prune_threshold=200,
        c=-20000,
        save_path=tmp_path,
    )
    return worker, tables


# ---------------------------------------------------------------------------
# 4.3 — _sync_to_master()
# ---------------------------------------------------------------------------

class TestSyncToMaster:
    def _make_delta(self, r: int, **action_values) -> np.ndarray:
        arr = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int64)
        for action, val in action_values.items():
            arr[ACTION_TO_IDX[r][action]] = int(val)
        return arr

    def test_flushes_single_infoset(self, tmp_path):
        w, tables = _make_worker(tmp_path)
        w._local_delta = {(0, "hand_A"): self._make_delta(0, fold=2, call=-1)}
        w._sync_to_master()
        row = tables.regret[0].get_row_if_exists("hand_A")
        assert row is not None
        assert row[ACTION_TO_IDX[0]["fold"]] == 2
        assert row[ACTION_TO_IDX[0]["call"]] == -1

    def test_flushes_multiple_infosets(self, tmp_path):
        w, tables = _make_worker(tmp_path)
        w._local_delta = {
            (0, "hand_A"): self._make_delta(0, fold=1, call=0),
            (0, "hand_B"): self._make_delta(0, fold=0, call=3),
        }
        w._sync_to_master()
        assert tables.regret[0].get_row_if_exists("hand_A") is not None
        assert tables.regret[0].get_row_if_exists("hand_B") is not None

    def test_accumulates_onto_existing_regrets(self, tmp_path):
        """Sync adds to existing entries rather than replacing them."""
        w, tables = _make_worker(tmp_path)
        tables.regret[0].merge_delta_row(
            "hand_A", self._make_delta(0, fold=5, call=3)
        )
        w._local_delta = {(0, "hand_A"): self._make_delta(0, fold=1, call=-2)}
        w._sync_to_master()
        row = tables.regret[0].get_row_if_exists("hand_A")
        assert row is not None
        assert row[ACTION_TO_IDX[0]["fold"]] == 6
        assert row[ACTION_TO_IDX[0]["call"]] == 1

    def test_local_delta_cleared_after_sync(self, tmp_path):
        w, tables = _make_worker(tmp_path)
        w._local_delta = {(0, "hand_A"): self._make_delta(0, fold=1)}
        w._sync_to_master()
        assert w._local_delta == {}

    def test_no_op_when_delta_is_empty(self, tmp_path):
        """_sync_to_master() must return immediately when delta is empty."""
        w, tables = _make_worker(tmp_path)
        w._sync_to_master()  # must not raise or block
        assert tables.regret[0].n_allocated == 0

    def test_logs_infoset_count(self, tmp_path):
        w, tables = _make_worker(tmp_path)
        w._local_delta = {
            (0, f"hand_{k}"): self._make_delta(0, fold=1)
            for k in range(5)
        }
        w._sync_to_master()
        msg = w._logging_queue.get(block=True, timeout=2)
        assert "5" in msg


# ---------------------------------------------------------------------------
# 4.3 — Local delta empty after sync
# ---------------------------------------------------------------------------

class TestLocalDeltaState:
    def test_initial_delta_is_empty(self, tmp_path):
        w, _ = _make_worker(tmp_path)
        assert w._local_delta == {}

    def test_initial_iteration_count_is_zero(self, tmp_path):
        w, _ = _make_worker(tmp_path)
        assert w._local_iteration_count == 0

    def test_delta_empty_after_repeated_syncs(self, tmp_path):
        w, tables = _make_worker(tmp_path)
        for i in range(3):
            w._local_delta = {
                (0, f"iter_{i}"): np.zeros(MAX_ACTIONS_PER_STREET[0], dtype=np.int64)
            }
            w._sync_to_master()
            assert w._local_delta == {}


# ---------------------------------------------------------------------------
# 4.4 — LUT not loaded before fork
# ---------------------------------------------------------------------------

class TestLutLoadedAfterFork:
    def test_no_info_set_lut_before_run(self, tmp_path):
        """_info_set_lut must not exist on the Worker before run() is called.

        The LUT is loaded in run() — after fork — so that the parent process
        never holds the large object in memory.
        """
        w, _ = _make_worker(tmp_path)
        assert not hasattr(w, "_info_set_lut")


# ---------------------------------------------------------------------------
# 4.5 — NUMA pinning helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 4.4 — terminate job flushes before exiting (integration, slow)
# ---------------------------------------------------------------------------

def _worker_target(job_queue, logging_queue, locks, tables, lut_path):
    """Run a worker that processes a few cfr jobs then terminates."""
    worker = Worker(
        job_queue=job_queue,
        logging_queue=logging_queue,
        locks=locks,
        tables=tables,
        lut_path=str(lut_path),
        pickle_dir=False,
        n_players=2,
        prune_threshold=9999999,
        c=-20000,
        save_path=lut_path,
    )
    worker.run()


@pytest.mark.slow
def test_terminate_flushes_before_exit(tmp_path):
    """Terminate job must call _sync_to_master() before the worker exits.

    Requires ``data/clustering/20cards_exact`` to be present.
    """
    lut_path = Path("data/clustering/20cards_exact")
    if not lut_path.exists():
        pytest.skip("20cards_exact LUT not available")

    shm_dir = str(tmp_path / "shm")
    os.makedirs(shm_dir, exist_ok=True)
    tables = CFRTables(
        index_path=tmp_path / "lmdb_index",
        shm_dir=shm_dir,
        actions_per_street=MAX_ACTIONS_PER_STREET,
    )
    locks = {
        "strategy_update_lock": mp.Lock(),
    }
    job_queue = mp.JoinableQueue()
    logging_queue = mp.Queue()

    # Dispatch a couple of cfr jobs followed by terminate.
    job_queue.put(("cfr", {"t": 10, "i": 0}))
    job_queue.put(("cfr", {"t": 11, "i": 1}))
    job_queue.put(("terminate", {}))

    proc = mp.Process(
        target=_worker_target,
        args=(job_queue, logging_queue, locks, tables, lut_path),
    )
    proc.start()
    proc.join(timeout=60)
    assert proc.exitcode == 0, "Worker did not exit cleanly"

    # After termination the agent's regret tables must be non-empty — the
    # cfr traversals accumulated regrets that terminate must have flushed.
    total_allocated = sum(
        tables.regret[r].n_allocated for r in range(4)
    )
    assert total_allocated > 0, "Regret tables are empty — terminate did not flush"
