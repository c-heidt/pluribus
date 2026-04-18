"""Functional tests for ``poker_ai/ai/singleprocess/train.py``.

Exercises :func:`~poker_ai.blueprint.singleprocess.train.simple_search` end-to-end
in a single process.  These tests require the 20-card LUT to be present at
``data/20cards_exact``; they are automatically skipped via the
``requires_lut`` marker when the LUT is absent.

Tests that only inspect module-level behaviour (imports, logging helpers) do
not require the LUT and run unconditionally.
"""

import logging
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from poker_ai.blueprint.singleprocess.train import simple_search
from poker_ai.blueprint.training import DiscountState

_LUT_PATH = Path("data/20cards_exact")


def _lut_available():
    return (_LUT_PATH / "card_info_lut.joblib").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_search(save_path: Path, **overrides):
    """Call simple_search with safe minimal defaults and optional overrides."""
    defaults = dict(
        config={},
        save_path=save_path,
        lut_path=str(_LUT_PATH),
        pickle_dir=False,
        strategy_interval=1,
        n_iterations=20,
        discount_duration_cycles=100,
        prune_threshold=9_999_999,
        c=-300_000_000,
        n_players=2,
        update_threshold=0,
        sync_interval=5,
        discount_interval=1,
    )
    defaults.update(overrides)
    simple_search(**defaults)


# ---------------------------------------------------------------------------
# TestSimpleSearch
# ---------------------------------------------------------------------------


@pytest.mark.requires_lut
class TestSimpleSearch:
    def test_runs_without_crash(self, tmp_path):
        _minimal_search(tmp_path, n_iterations=10)

    def test_tables_populated_after_iterations(self, tmp_path):
        """After a short run, at least one regret infoset must be allocated."""
        from environment.action_space import MAX_ACTIONS_PER_STREET
        from poker_ai.tables.cfr_tables import CFRTables
        from poker_ai.tables.index import lmdb_map_size_for_players

        _minimal_search(tmp_path, n_iterations=20)
        # simple_search writes chunks to save_path/shm — reopen from there
        t = CFRTables(
            index_path=tmp_path / "lmdb_index",
            shm_dir=str(tmp_path / "shm"),
            actions_per_street=MAX_ACTIONS_PER_STREET,
            lmdb_map_size=lmdb_map_size_for_players(2),
        )
        total = sum(t.regret[r].n_allocated for r in range(4))
        t.close()
        assert total > 0

    def test_sync_schedule_respected(self, tmp_path):
        """With update_threshold=0, strategy tables must be written."""
        from poker_ai.tables.cfr_tables import CFRTables
        from environment.action_space import MAX_ACTIONS_PER_STREET
        from poker_ai.tables.index import lmdb_map_size_for_players

        _minimal_search(
            tmp_path,
            n_iterations=10,
            sync_interval=5,
            strategy_interval=1,
            update_threshold=0,
        )
        # simple_search writes chunks to save_path/shm — reopen from there
        t = CFRTables(
            index_path=tmp_path / "lmdb_index",
            shm_dir=str(tmp_path / "shm"),
            actions_per_street=MAX_ACTIONS_PER_STREET,
            lmdb_map_size=lmdb_map_size_for_players(2),
        )
        total_strategy = sum(t.strategy[r].n_allocated for r in range(4))
        t.close()
        assert total_strategy > 0

    def test_discount_applied_within_run(self, tmp_path, monkeypatch):
        """DiscountState.apply must be called at each discount sync barrier."""
        calls = []
        original_apply = DiscountState.apply

        def tracking_apply(self, tables, sync_step):
            calls.append(sync_step)
            original_apply(self, tables, sync_step)

        monkeypatch.setattr(DiscountState, "apply", tracking_apply)
        _minimal_search(
            tmp_path,
            n_iterations=20,
            sync_interval=5,
            discount_interval=1,
            discount_duration_cycles=100,
        )
        assert len(calls) > 0


# ---------------------------------------------------------------------------
# TestProgressLogging
# ---------------------------------------------------------------------------


@pytest.mark.requires_lut
class TestProgressLogging:
    def test_progress_log_emitted_after_interval(self, tmp_path, monkeypatch, caplog):
        """A progress log containing 'elapsed' must appear when time advances."""
        import poker_ai.blueprint.singleprocess.train as train_module

        _call_count = [0]
        _base = time.monotonic()

        def fake_monotonic():
            _call_count[0] += 1
            return _base + _call_count[0] * 65.0

        monkeypatch.setattr(train_module.time, "monotonic", fake_monotonic)

        with caplog.at_level(logging.INFO, logger="poker_ai.blueprint.singleprocess"):
            _minimal_search(tmp_path, n_iterations=10, sync_interval=5)

        progress_records = [r for r in caplog.records if "elapsed" in r.message]
        assert len(progress_records) >= 1


class TestModuleInspection:
    def test_no_trange_import(self):
        """tqdm must not be imported by the singleprocess training module."""
        import poker_ai.blueprint.singleprocess.train as train_module
        assert "trange" not in dir(train_module)
        assert "tqdm" not in dir(train_module)
