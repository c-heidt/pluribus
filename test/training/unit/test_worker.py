"""Unit tests for ``poker_ai/ai/multiprocess/worker.py``.

Covers:
- :meth:`~poker_ai.blueprint.multiprocess.worker.Worker._flush_delta` — delta merge,
  clear, and logging level.
- Local delta lifecycle: starts empty, cleared after each flush.
- LUT is not loaded until :meth:`~poker_ai.blueprint.multiprocess.worker.Worker.run`
  is called (post-fork).
- Integration: worker processes cfr jobs and flushes tables on terminate (slow).
"""

import logging
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pytest

from environment.action_space import ACTION_TO_IDX, MAX_ACTIONS_PER_STREET
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.blueprint.multiprocess.worker import Worker


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_worker(tmp_path: Path):
    """Instantiate a :class:`Worker` for unit testing (not started)."""
    shm_dir = str(tmp_path / "shm")
    os.makedirs(shm_dir, exist_ok=True)
    tables = CFRTables(
        index_path=tmp_path / "lmdb_index",
        shm_dir=shm_dir,
        actions_per_street=MAX_ACTIONS_PER_STREET,
    )
    worker = Worker(
        job_queue=mp.JoinableQueue(),
        logging_queue=mp.Queue(),
        locks={"strategy_update_lock": mp.Lock()},
        tables=tables,
        lut_path=tmp_path,
        pickle_dir=False,
        n_players=2,
        prune_threshold=200,
        c=-20_000,
        save_path=tmp_path,
    )
    return worker, tables


def _make_delta(r: int, **action_values) -> np.ndarray:
    arr = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int64)
    for action, val in action_values.items():
        arr[ACTION_TO_IDX[r][action]] = int(val)
    return arr


# ---------------------------------------------------------------------------
# _flush_delta
# ---------------------------------------------------------------------------


class TestWorkerDeltaFlush:
    def test_flushes_single_infoset(self, tmp_path):
        w, tables = _make_worker(tmp_path)
        w._local_delta = {(0, "hand_A"): _make_delta(0, fold=2, call=-1)}
        w._flush_delta()
        row = tables.regret[0].get_row_if_exists("hand_A")
        assert row is not None
        assert row[ACTION_TO_IDX[0]["fold"]] == 2
        assert row[ACTION_TO_IDX[0]["call"]] == -1

    def test_flushes_multiple_infosets(self, tmp_path):
        w, tables = _make_worker(tmp_path)
        w._local_delta = {
            (0, "hand_A"): _make_delta(0, fold=1, call=0),
            (0, "hand_B"): _make_delta(0, fold=0, call=3),
        }
        w._flush_delta()
        assert tables.regret[0].get_row_if_exists("hand_A") is not None
        assert tables.regret[0].get_row_if_exists("hand_B") is not None

    def test_accumulates_onto_existing_regrets(self, tmp_path):
        w, tables = _make_worker(tmp_path)
        tables.regret[0].merge_delta_row("hand_A", _make_delta(0, fold=5, call=3))
        w._local_delta = {(0, "hand_A"): _make_delta(0, fold=1, call=-2)}
        w._flush_delta()
        row = tables.regret[0].get_row_if_exists("hand_A")
        assert row[ACTION_TO_IDX[0]["fold"]] == 6
        assert row[ACTION_TO_IDX[0]["call"]] == 1

    def test_local_delta_cleared_after_sync(self, tmp_path):
        w, _ = _make_worker(tmp_path)
        w._local_delta = {(0, "hand_A"): _make_delta(0, fold=1)}
        w._flush_delta()
        assert w._local_delta == {}

    def test_no_op_when_delta_is_empty(self, tmp_path):
        w, tables = _make_worker(tmp_path)
        w._flush_delta()
        assert tables.regret[0].n_allocated == 0

    def test_flush_logs_at_debug_not_info(self, tmp_path, caplog):
        """_flush_delta must emit a DEBUG record and NOT push to the logging queue."""
        w, _ = _make_worker(tmp_path)
        w._local_delta = {(0, f"hand_{k}"): _make_delta(0, fold=1) for k in range(5)}
        with caplog.at_level(logging.DEBUG):
            w._flush_delta()
        # Nothing pushed to the logging queue
        assert w._logging_queue.empty()
        # A DEBUG log record was emitted (not INFO)
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(debug_records) > 0


# ---------------------------------------------------------------------------
# Local delta lifecycle
# ---------------------------------------------------------------------------


class TestLocalDeltaState:
    def test_initial_delta_is_empty(self, tmp_path):
        w, _ = _make_worker(tmp_path)
        assert w._local_delta == {}

    def test_delta_empty_after_repeated_syncs(self, tmp_path):
        w, _ = _make_worker(tmp_path)
        for i in range(3):
            w._local_delta = {
                (0, f"iter_{i}"): np.zeros(MAX_ACTIONS_PER_STREET[0], dtype=np.int64)
            }
            w._flush_delta()
            assert w._local_delta == {}


# ---------------------------------------------------------------------------
# LUT loading deferred until after fork
# ---------------------------------------------------------------------------


class TestLutLoadedAfterFork:
    def test_no_info_set_lut_before_run(self, tmp_path):
        """_info_set_lut must not be loaded in __init__ — only after fork."""
        w, _ = _make_worker(tmp_path)
        assert not hasattr(w, "_info_set_lut")


# ---------------------------------------------------------------------------
# Integration: terminate job flushes before exit (slow)
# ---------------------------------------------------------------------------


def _worker_target(job_queue, logging_queue, locks, tables, lut_path):
    worker = Worker(
        job_queue=job_queue,
        logging_queue=logging_queue,
        locks=locks,
        tables=tables,
        lut_path=str(lut_path),
        pickle_dir=False,
        n_players=2,
        prune_threshold=9_999_999,
        c=-20_000,
        save_path=lut_path,
    )
    worker.run()


@pytest.mark.slow
def test_terminate_flushes_before_exit(tmp_path):
    """The terminate job must call _flush_delta() so tables are non-empty."""
    lut_path = Path("data/20cards_exact")
    if not lut_path.exists():
        pytest.skip("20cards_exact LUT not available")

    shm_dir = str(tmp_path / "shm")
    os.makedirs(shm_dir, exist_ok=True)
    tables = CFRTables(
        index_path=tmp_path / "lmdb_index",
        shm_dir=shm_dir,
        actions_per_street=MAX_ACTIONS_PER_STREET,
    )
    job_queue = mp.JoinableQueue()
    logging_queue = mp.Queue()
    locks = {"strategy_update_lock": mp.Lock()}

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
    assert sum(tables.regret[r].n_allocated for r in range(4)) > 0
