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

from poker_ai.ai.agent import Agent
from poker_ai.ai.multiprocess.worker import Worker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_worker(tmp_path: Path) -> tuple:
    """Instantiate a Worker suitable for unit testing (not started)."""
    os.environ["TESTING_SUITE"] = "1"
    agent = Agent()
    locks = {
        "regret": mp.Lock(),
        "strategy": mp.Lock(),
        "pre_flop_strategy": mp.Lock(),
    }
    job_queue = mp.JoinableQueue()
    status_queue = mp.Queue()
    logging_queue = mp.Queue()
    worker = Worker(
        job_queue=job_queue,
        status_queue=status_queue,
        logging_queue=logging_queue,
        locks=locks,
        agent=agent,
        lut_path=tmp_path,
        pickle_dir=False,
        n_players=2,
        prune_threshold=200,
        c=-20000,
        discount_interval=10,
        save_path=tmp_path,
    )
    return worker, agent


# ---------------------------------------------------------------------------
# 4.3 — _sync_to_master()
# ---------------------------------------------------------------------------

class TestSyncToMaster:
    def test_flushes_single_infoset(self, tmp_path):
        w, agent = _make_worker(tmp_path)
        w._local_delta = {"hand_A": {"fold": 2.0, "call": -1.0}}
        w._sync_to_master()
        result = dict(agent.regret["hand_A"])
        assert result == pytest.approx({"fold": 2.0, "call": -1.0})

    def test_flushes_multiple_infosets(self, tmp_path):
        w, agent = _make_worker(tmp_path)
        w._local_delta = {
            "hand_A": {"fold": 1.5, "call": -0.5},
            "hand_B": {"raise": 3.0, "fold": -1.0},
        }
        w._sync_to_master()
        assert set(agent.regret.keys()) == {"hand_A", "hand_B"}

    def test_accumulates_onto_existing_regrets(self, tmp_path):
        """Sync adds to existing entries rather than replacing them."""
        w, agent = _make_worker(tmp_path)
        agent.regret["hand_A"] = {"fold": 5.0, "call": 3.0}
        w._local_delta = {"hand_A": {"fold": 1.0, "call": -2.0, "raise": 4.0}}
        w._sync_to_master()
        result = dict(agent.regret["hand_A"])
        assert result["fold"] == pytest.approx(6.0)
        assert result["call"] == pytest.approx(1.0)
        assert result["raise"] == pytest.approx(4.0)

    def test_local_delta_cleared_after_sync(self, tmp_path):
        w, agent = _make_worker(tmp_path)
        w._local_delta = {"hand_A": {"fold": 1.0}}
        w._sync_to_master()
        assert w._local_delta == {}

    def test_no_op_when_delta_is_empty(self, tmp_path):
        """Does not acquire lock or touch agent when delta is empty."""
        w, agent = _make_worker(tmp_path)
        # Lock the regret lock so that acquiring it would deadlock — if
        # _sync_to_master() tries to acquire on an empty delta it will hang.
        locked = w._locks["regret"].acquire(block=False)
        assert locked, "Could not acquire lock for test setup"
        try:
            w._sync_to_master()  # Must return immediately without blocking
        finally:
            w._locks["regret"].release()
        assert len(agent.regret) == 0

    def test_logs_infoset_count(self, tmp_path):
        w, agent = _make_worker(tmp_path)
        w._local_delta = {f"hand_{k}": {"fold": 1.0} for k in range(5)}
        w._sync_to_master()
        # mp.Queue.empty() is unreliable; use get() with a timeout instead.
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
        w, agent = _make_worker(tmp_path)
        for i in range(3):
            w._local_delta = {f"iter_{i}": {"fold": float(i)}}
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

class TestParseCpulist:
    def test_single_core(self):
        assert Worker._parse_cpulist("4") == [4]

    def test_simple_range(self):
        assert Worker._parse_cpulist("0-3") == [0, 1, 2, 3]

    def test_multiple_ranges(self):
        assert Worker._parse_cpulist("0-3,8-11") == [0, 1, 2, 3, 8, 9, 10, 11]

    def test_mixed_range_and_single(self):
        assert Worker._parse_cpulist("0,2-4,7") == [0, 2, 3, 4, 7]

    def test_single_range_one_element(self):
        assert Worker._parse_cpulist("5-5") == [5]

    def test_empty_string(self):
        assert Worker._parse_cpulist("") == []


class TestGetNumaNodes:
    def test_returns_list(self, tmp_path):
        w, _ = _make_worker(tmp_path)
        nodes = w._get_numa_nodes()
        assert isinstance(nodes, list)

    def test_all_non_negative_integers(self, tmp_path):
        w, _ = _make_worker(tmp_path)
        nodes = w._get_numa_nodes()
        assert all(isinstance(n, int) and n >= 0 for n in nodes)

    def test_sorted(self, tmp_path):
        w, _ = _make_worker(tmp_path)
        nodes = w._get_numa_nodes()
        assert nodes == sorted(nodes)


class TestTryNumaPin:
    def test_does_not_raise(self, tmp_path):
        """_try_numa_pin() must never raise regardless of environment."""
        w, _ = _make_worker(tmp_path)
        w._try_numa_pin()  # No assertion needed — must not raise

    def test_get_cores_returns_list(self, tmp_path):
        w, _ = _make_worker(tmp_path)
        nodes = w._get_numa_nodes()
        if nodes:
            cores = w._get_cores_for_numa_node(nodes[0])
            assert isinstance(cores, list)
            assert all(isinstance(c, int) and c >= 0 for c in cores)

    def test_get_cores_unknown_node_returns_empty(self, tmp_path):
        """A node ID that doesn't exist on disk returns an empty list."""
        w, _ = _make_worker(tmp_path)
        cores = w._get_cores_for_numa_node(99999)
        assert cores == []


# ---------------------------------------------------------------------------
# 4.4 — terminate job flushes before exiting (integration, slow)
# ---------------------------------------------------------------------------

def _worker_target(job_queue, status_queue, logging_queue, locks, agent,
                   lut_path):
    """Run a worker that processes a few cfr jobs then terminates."""
    import os
    os.environ["TESTING_SUITE"] = "1"
    worker = Worker(
        job_queue=job_queue,
        status_queue=status_queue,
        logging_queue=logging_queue,
        locks=locks,
        agent=agent,
        lut_path=str(lut_path),
        pickle_dir=False,
        n_players=2,
        prune_threshold=9999999,
        c=-20000,
        discount_interval=10,
        save_path=lut_path,
    )
    worker.run()


@pytest.mark.slow
def test_terminate_flushes_before_exit():
    """Terminate job must call _sync_to_master() before the worker exits.

    Requires ``data/clustering/20cards_exact`` to be present.
    """
    lut_path = Path("data/clustering/20cards_exact")
    if not lut_path.exists():
        pytest.skip("20cards_exact LUT not available")

    os.environ["TESTING_SUITE"] = "1"
    agent = Agent()
    locks = {
        "regret": mp.Lock(),
        "strategy": mp.Lock(),
        "pre_flop_strategy": mp.Lock(),
    }
    job_queue = mp.JoinableQueue()
    status_queue = mp.Queue()
    logging_queue = mp.Queue()

    # Dispatch a couple of cfr jobs followed by terminate.
    job_queue.put(("cfr", {"t": 10, "i": 0}))
    job_queue.put(("cfr", {"t": 11, "i": 1}))
    job_queue.put(("terminate", {}))

    proc = mp.Process(
        target=_worker_target,
        args=(job_queue, status_queue, logging_queue, locks, agent, lut_path),
    )
    proc.start()
    proc.join(timeout=60)
    assert proc.exitcode == 0, "Worker did not exit cleanly"

    # After termination the agent's regret table must be non-empty — the
    # cfr traversals accumulated regrets that terminate must have flushed.
    assert len(agent.regret) > 0, "Regret table is empty — terminate did not flush"
