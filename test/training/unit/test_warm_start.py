"""Unit tests for ``poker_ai/tables/warm_start.py``.

Covers the warm-start staging helpers used to seed a biased blueprint
run from a finished base blueprint:

- LMDB index + checkpoint files are copied into the destination.
- ``server_state.pkl`` is rewritten with ``t = 0`` and
  ``discount_active = True``.
- ``n_players`` mismatch raises a clear error.
- Resume takes priority: a destination that already has a checkpoint
  is left untouched.

Chunk files themselves are not exercised here (real chunks are tens
of MB and would dominate the test budget); per-street chunk count is
set to zero so :meth:`CFRTables.restore_chunks` is a no-op.  The
chunk-copy code path is still tested by writing dummy ``.npy`` files
into the source checkpoint and asserting they appear in the staged
copy.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import joblib
import numpy as np
import pytest

from environment.action_space import MAX_ACTIONS_PER_STREET
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.tables.warm_start import (
    apply_warm_start,
    apply_warm_start_to_tables,
    stage_warm_start_lmdb,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source_blueprint(
    path: Path,
    n_players: int = 6,
    saved_t: int = 1234,
    n_chunks_per_street: dict | None = None,
    write_dummy_npys: bool = True,
) -> Path:
    """Lay out a minimal warm-start source under *path*.

    Creates a real (empty) LMDB index by constructing a transient
    :class:`CFRTables`, plus a ``checkpoint_<wall>`` directory holding
    ``server_state.pkl`` and a few placeholder ``.npy`` files.

    Parameters
    ----------
    path : Path
        Root of the source blueprint.
    n_players : int
        Player count to embed in the saved ``server_state.pkl``.
    saved_t : int
        Iteration counter to embed.  The warm-start helper must
        rewrite this to 0 in the staged copy.
    n_chunks_per_street : dict, optional
        Per-street chunk counts for the saved state.  Defaults to all
        zeros so :meth:`CFRTables.restore_chunks` is a no-op.
    write_dummy_npys : bool
        Whether to create a placeholder ``regret_0_chunk_000000.npy``
        and ``strategy_0_chunk_000000.npy`` in the source checkpoint
        so we can verify the chunk-copy code path runs.
    """
    if n_chunks_per_street is None:
        n_chunks_per_street = {r: 0 for r in range(4)}

    path.mkdir(parents=True, exist_ok=True)
    shm = path / "shm"
    shm.mkdir(parents=True, exist_ok=True)
    tables = CFRTables(
        index_path=path / "lmdb_index",
        shm_dir=str(shm),
        actions_per_street=MAX_ACTIONS_PER_STREET,
    )
    tables.close()

    cp = path / f"checkpoint_{int(time.time())}"
    cp.mkdir(parents=True)
    if write_dummy_npys:
        np.save(cp / "regret_0_chunk_000000.npy", np.zeros(4, dtype=np.int32))
        np.save(cp / "strategy_0_chunk_000000.npy", np.zeros(4, dtype=np.int32))

    from environment.poker_env import INFO_SET_ENCODING, action_grid_fingerprint

    state = {
        "n_players": n_players,
        "t": saved_t,
        "n_chunks_per_street": n_chunks_per_street,
        "chunk_size": 4_000_000,
        "discount_active": False,
        "info_set_encoding": INFO_SET_ENCODING,
        "action_grid_fingerprint": action_grid_fingerprint(n_players),
    }
    joblib.dump(state, cp / "server_state.pkl")
    return cp


# ---------------------------------------------------------------------------
# apply_warm_start (multi-process staging)
# ---------------------------------------------------------------------------


class TestApplyWarmStart:
    def test_lmdb_index_and_checkpoint_copied(self, tmp_path):
        src = tmp_path / "base"
        dst = tmp_path / "fold_biased"
        _make_source_blueprint(src, n_players=6)

        staged = apply_warm_start(dst, src, expected_n_players=6)
        assert staged is not None
        assert (dst / "lmdb_index").exists()
        assert staged.exists()
        # The two dummy npy files were carried over.
        assert (staged / "regret_0_chunk_000000.npy").exists()
        assert (staged / "strategy_0_chunk_000000.npy").exists()

    def test_state_dict_resets_t_and_reactivates_discount(self, tmp_path):
        src = tmp_path / "base"
        dst = tmp_path / "biased"
        _make_source_blueprint(src, saved_t=999_999)

        staged = apply_warm_start(dst, src, expected_n_players=6)
        new_state = joblib.load(staged / "server_state.pkl")
        assert new_state["t"] == 0
        assert new_state["discount_active"] is True
        # n_chunks_per_street and n_players preserved so the
        # CheckpointManager can validate and restore.
        assert new_state["n_players"] == 6
        assert "n_chunks_per_street" in new_state

    def test_resume_takes_priority_when_dst_has_checkpoint(self, tmp_path):
        src = tmp_path / "base"
        dst = tmp_path / "biased"
        _make_source_blueprint(src)
        dst.mkdir()
        existing_cp = dst / "checkpoint_1700000000"
        existing_cp.mkdir()

        result = apply_warm_start(dst, src, expected_n_players=6)
        assert result is None
        # Nothing else was added: no lmdb_index, no second checkpoint dir.
        assert not (dst / "lmdb_index").exists()
        assert sorted(p.name for p in dst.glob("checkpoint_*")) == [
            "checkpoint_1700000000"
        ]

    def test_n_players_mismatch_raises(self, tmp_path):
        src = tmp_path / "base"
        dst = tmp_path / "biased"
        _make_source_blueprint(src, n_players=6)

        with pytest.raises(ValueError, match="n_players mismatch"):
            apply_warm_start(dst, src, expected_n_players=2)

    def test_missing_warm_start_checkpoint_raises(self, tmp_path):
        src = tmp_path / "no_checkpoint"
        src.mkdir()
        (src / "lmdb_index").mkdir()
        dst = tmp_path / "biased"

        with pytest.raises(FileNotFoundError):
            apply_warm_start(dst, src, expected_n_players=6)

    def test_missing_lmdb_raises(self, tmp_path):
        src = tmp_path / "base"
        src.mkdir()
        cp = src / f"checkpoint_{int(time.time())}"
        cp.mkdir()
        from environment.poker_env import INFO_SET_ENCODING, action_grid_fingerprint
        joblib.dump(
            {"n_players": 6, "t": 1, "n_chunks_per_street": {r: 0 for r in range(4)},
             "chunk_size": 4_000_000, "discount_active": False,
             "info_set_encoding": INFO_SET_ENCODING,
             "action_grid_fingerprint": action_grid_fingerprint(6)},
            cp / "server_state.pkl",
        )
        dst = tmp_path / "biased"

        with pytest.raises(FileNotFoundError, match="lmdb_index"):
            apply_warm_start(dst, src, expected_n_players=6)

    def test_legacy_encoding_raises(self, tmp_path):
        """A base blueprint without the info_set_encoding marker (or with a
        stale one) is rejected — its keys were written under a different
        encoding and would silently 100%-miss."""
        src = tmp_path / "base"
        _make_source_blueprint(src, n_players=6)
        # Overwrite server_state.pkl to drop the encoding marker.
        cp = sorted(src.glob("checkpoint_[0-9]*"))[-1]
        state = joblib.load(cp / "server_state.pkl")
        state.pop("info_set_encoding", None)
        joblib.dump(state, cp / "server_state.pkl")
        dst = tmp_path / "biased"
        with pytest.raises(ValueError, match="info_set_encoding"):
            apply_warm_start(dst, src, expected_n_players=6)


# ---------------------------------------------------------------------------
# Single-process helpers
# ---------------------------------------------------------------------------


class TestSingleProcessHelpers:
    def test_stage_warm_start_lmdb_copies_index(self, tmp_path):
        src = tmp_path / "base"
        dst = tmp_path / "biased"
        _make_source_blueprint(src)

        staged = stage_warm_start_lmdb(dst, src, expected_n_players=6)
        assert staged is True
        assert (dst / "lmdb_index").exists()

    def test_stage_warm_start_lmdb_skips_when_dst_has_lmdb(self, tmp_path):
        src = tmp_path / "base"
        dst = tmp_path / "biased"
        _make_source_blueprint(src)
        # Pre-existing lmdb_index in dst.
        (dst / "lmdb_index").mkdir(parents=True)

        result = stage_warm_start_lmdb(dst, src, expected_n_players=6)
        assert result is False

    def test_apply_warm_start_to_tables_runs(self, tmp_path):
        """With zero chunks-per-street, restore_chunks is a no-op but the
        helper must still validate n_players and accept the call."""
        src = tmp_path / "base"
        _make_source_blueprint(src, n_chunks_per_street={r: 0 for r in range(4)})

        # Build a destination tables instance.
        dst = tmp_path / "biased"
        shm = dst / "shm"
        shm.mkdir(parents=True)
        tables = CFRTables(
            index_path=dst / "lmdb_index",
            shm_dir=str(shm),
            actions_per_street=MAX_ACTIONS_PER_STREET,
        )
        try:
            apply_warm_start_to_tables(tables, src, expected_n_players=6)
        finally:
            tables.close()

    def test_apply_warm_start_to_tables_n_players_mismatch_raises(self, tmp_path):
        src = tmp_path / "base"
        _make_source_blueprint(src, n_players=6)
        dst = tmp_path / "biased"
        shm = dst / "shm"
        shm.mkdir(parents=True)
        tables = CFRTables(
            index_path=dst / "lmdb_index",
            shm_dir=str(shm),
            actions_per_street=MAX_ACTIONS_PER_STREET,
        )
        try:
            with pytest.raises(ValueError, match="n_players mismatch"):
                apply_warm_start_to_tables(tables, src, expected_n_players=2)
        finally:
            tables.close()
