"""What the parallel runner keeps when it does NOT finish cleanly.

The per-hand pool writes into per-worker node-local DBs and used to read them exactly
once, at the end, so every abnormal exit threw the whole run away.  Three things are
pinned here, each against the real ``run_evaluation_parallel`` under a real forked pool:

- **checkpoint** — the permanent snapshot is populated *during* the run, on the
  ``sync_interval_*`` cadence, not only after the final merge;
- **rescue** — a worker killed outright (the cgroup OOM killer's SIGKILL) still leaves
  the hands its siblings finished merged into the target;
- **circuit breaker** — a systematic per-hand failure aborts instead of spending the
  whole wall budget writing ``hand_failures`` rows.

These are slow-ish (they fork), but each one guards hours of cluster time.
"""

import dataclasses
import os
import sqlite3

import pytest

from evaluation import runner as R
from evaluation.hand_pool import WorkerDiedError
from evaluation.runner import run_evaluation_parallel
from evaluation.sqlite_logging import ExperimentLog
from test.evaluation.test_parallel_reproducibility import _session


def _n_games(path) -> int:
    if not os.path.exists(os.fspath(path)):
        return 0
    con = sqlite3.connect(os.fspath(path))
    try:
        return con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    finally:
        con.close()


def _fresh(tmp_path, name):
    db = tmp_path / f"{name}.sqlite"
    ExperimentLog.open(os.fspath(db)).close()          # create the merge target
    return db, tmp_path / f"{name}.perm.sqlite"


def _syncer(db, perm):
    def _sync():
        log = ExperimentLog.open(os.fspath(db))
        try:
            log.sync_to(perm)
        finally:
            log.close()
    return _sync


def test_permanent_snapshot_is_populated_mid_run(tmp_path):
    """A checkpoint lands while the pool is still working, not only at the end."""
    db, perm = _fresh(tmp_path, "ckpt")
    session = _session("ckpt", condition="vanilla")
    # Cadence low enough that several checkpoints fall inside a short run.
    session.config = dataclasses.replace(
        session.config, sync_interval_hands=5, sync_interval_minutes=0.0
    )
    seen = []
    real = R._checkpoint_parallel

    def _spy(target, worker_dir, sync_path):
        real(target, worker_dir, sync_path)
        seen.append(_n_games(sync_path))       # rows visible at THIS checkpoint

    R._checkpoint_parallel = _spy
    try:
        n = run_evaluation_parallel(
            session, db, n_workers=3, max_hands=40, progress_interval=0,
            sync_fn=_syncer(db, perm), sync_path=perm,
        )
    finally:
        R._checkpoint_parallel = real

    assert n == 40
    assert seen, "no checkpoint ran during the pool — the cadence is inert again"
    # The point of the fix: hands were durable BEFORE the run ended.
    assert seen[0] > 0
    assert _n_games(perm) == 40


def test_a_killed_worker_does_not_discard_its_siblings_work(tmp_path):
    """SIGKILL one worker; the hands the others committed must still be merged."""
    db, perm = _fresh(tmp_path, "oom")
    real = R._play_and_log_one

    def _killer(session, log, cfg, fingerprint, hand_index, *a, **kw):
        ok = real(session, log, cfg, fingerprint, hand_index, *a, **kw)
        if hand_index == 17:               # exactly how the OOM killer arrives: no unwind
            os._exit(-9 & 0xFF)
        return ok

    R._play_and_log_one = _killer
    try:
        with pytest.raises(WorkerDiedError):
            run_evaluation_parallel(
                session=_session("oom", condition="vanilla"), target_db_path=db,
                n_workers=3, max_hands=60, progress_interval=0,
                sync_fn=_syncer(db, perm), sync_path=perm,
            )
    finally:
        R._play_and_log_one = real

    rescued = _n_games(db)
    assert rescued > 0, "the surviving workers' completed hands were thrown away"
    assert _n_games(perm) == rescued, "rescued hands never reached the permanent FS"


def test_systematic_failure_trips_the_breaker(tmp_path):
    """Every hand failing must abort, not consume the whole budget."""
    db, _ = _fresh(tmp_path, "breaker")
    real = R._play_and_log_one
    R._play_and_log_one = lambda *a, **kw: False        # always fails, logs nothing
    try:
        # max_hands is enormous: only the breaker can end this.
        with pytest.raises(RuntimeError, match="consecutive job failures"):
            run_evaluation_parallel(
                session=_session("breaker", condition="vanilla"), target_db_path=db,
                n_workers=2, max_hands=1_000_000, progress_interval=0,
                max_consecutive_failures=20,
            )
    finally:
        R._play_and_log_one = real


def test_breaker_ignores_processes_that_report_no_outcome(tmp_path):
    """A ``process`` returning None (the calibration sweep) can never trip the breaker."""
    from evaluation.hand_pool import run_index_pool

    def _process(idx, state, shared):
        state["n"] += 1                    # returns None, like _sweep_process

    payloads = run_index_pool(
        n_workers=2, setup=lambda wid, sh: {"n": 0}, process=_process,
        teardown=lambda st: st["n"], target=50, max_consecutive_failures=5,
    )
    assert sum(payloads) == 50
