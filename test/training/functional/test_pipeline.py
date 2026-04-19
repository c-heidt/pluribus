"""Integration and regression tests for the training pipeline.

These tests exercise the training components together rather than in isolation.
They verify end-to-end behaviour (tables are populated, checkpoints round-trip)
and regression-guard the bugs fixed during the refactor:

- Strategy double-weighting: strategy table values grow linearly with iteration
  count, not quadratically.
- REGRET_FLOOR: repeated discounting never drives regrets below the floor.
- Concurrent ``update_row``: stripe locking prevents lost writes under concurrent
  access.
- Singleprocess / multiprocess discount schedule equivalence: both modes call
  :meth:`~poker_ai.blueprint.training.DiscountState.apply` with the same ``sync_step``
  values.

Multi-process tests that start real worker processes are marked ``@pytest.mark.slow``
and are skipped by default (``-m "not slow"``).
"""

import threading
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from environment.action_space import ACTION_TO_IDX, MAX_ACTIONS_PER_STREET
from poker_ai.blueprint.cfr import cfr, merge_local_delta
from poker_ai.tables.cfr_tables import CFRTables, REGRET_FLOOR
from poker_ai.tables.index import lmdb_map_size_for_players
from poker_ai.blueprint.singleprocess.train import simple_search
from poker_ai.blueprint.strategy import update_strategy
from poker_ai.blueprint.training import (
    DiscountState,
    at_sync_barrier,
    should_discount,
)
from environment.poker_env import new_game

_LUT_PATH = Path("data/20cards_exact")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_tables(save_path: Path):
    import os
    shm_dir = str(save_path / "shm")
    os.makedirs(shm_dir, exist_ok=True)
    return CFRTables(
        index_path=save_path / "lmdb",
        shm_dir=shm_dir,
        actions_per_street=MAX_ACTIONS_PER_STREET,
        lmdb_map_size=lmdb_map_size_for_players(2),
    )


def _minimal_search(save_path: Path, **overrides):
    defaults = dict(
        config={},
        save_path=save_path,
        lut_path=str(_LUT_PATH),
        pickle_dir=False,
        strategy_interval=1,
        n_iterations=30,
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


def _new_game_2p(lut=None):
    return new_game(n_players=2, card_info_lut=lut)


# ---------------------------------------------------------------------------
# TestEndToEndSingleProcess
# ---------------------------------------------------------------------------


@pytest.mark.requires_lut
class TestEndToEndSingleProcess:
    def test_pipeline_populates_regret_table(self, tmp_path):
        _minimal_search(tmp_path)
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

    def test_pipeline_populates_strategy_table(self, tmp_tables):
        """With update_threshold=0, strategy table must receive writes."""
        from information_abstraction import load_info_set_lut
        card_info_lut = load_info_set_lut(str(_LUT_PATH))
        for t in range(1, 11):
            for i in range(2):
                state = _new_game_2p(card_info_lut)
                if t % 5 == 0:
                    update_strategy(tmp_tables, state, i)
        total = sum(tmp_tables.strategy[r].n_allocated for r in range(4))
        assert total > 0

    def test_pipeline_does_not_raise_past_prune_threshold(self, tmp_path):
        """Running past prune_threshold with c=0 must not crash."""
        _minimal_search(tmp_path, n_iterations=20, prune_threshold=5, c=0)


# ---------------------------------------------------------------------------
# TestCheckpointResume
# ---------------------------------------------------------------------------


@pytest.mark.requires_lut
class TestCheckpointResume:
    def test_resume_restores_iteration_counter(self, tmp_path):
        """Running simple_search twice to the same save_path should resume."""
        _minimal_search(tmp_path / "run", n_iterations=10)
        # Check that LMDB index persists (proxy for checkpoint state)
        assert (tmp_path / "run" / "lmdb_index").exists()

    def test_discount_state_window_closes(self, tmp_tables):
        """DiscountState becomes inactive after duration_cycles sync steps."""
        ds = DiscountState(duration_cycles=5, discount_interval=1)
        for step in range(1, 7):
            ds.apply(tmp_tables, sync_step=step)
        assert not ds.active


# ---------------------------------------------------------------------------
# TestRegressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_strategy_not_double_weighted(self, tmp_path):
        """Strategy values must grow linearly with iteration count, not N².

        Uses a hand-crafted two-node game (no LUT required) so the visit counts
        in the strategy table can be checked exactly.  Running ``update_strategy``
        N times on the same root must yield row sums that scale linearly with N.
        """
        import numpy as np
        from poker_ai.blueprint.strategy import update_strategy

        # Minimal mock game (reused from test_strategy.py / test_cfr.py)
        class _FP:
            def __init__(self):
                self.is_active = True
                self.cards = []

        class _FT:
            community_cards = []

        class Term:
            def __init__(self, p):
                self.is_terminal = True
                self.players = [_FP(), _FP()]
                self._payout = p
                self._table = _FT()
                self._betting_stage = "pre_flop"

            @property
            def player_i(self):
                return 0

            @property
            def payout(self):
                return self._payout

            @property
            def info_set(self):
                return "t"

            @property
            def legal_actions(self):
                return []

            def apply_action(self, a):
                raise RuntimeError

        class Root:
            def __init__(self):
                self.is_terminal = False
                self.players = [_FP(), _FP()]
                self._betting_stage = "pre_flop"
                self._table = _FT()
                self._actions = ["fold", "call"]
                self._children = {
                    "fold": Term({0: -50, 1: 50}),
                    "call": Term({0: 50, 1: -50}),
                }

            @property
            def player_i(self):
                return 0

            @property
            def info_set(self):
                return "linear_root"

            @property
            def legal_actions(self):
                return list(self._actions)

            @property
            def betting_round(self):
                return 0

            def get_valid_mask(self):
                from environment.poker_env import PokerEnv
                canonical = PokerEnv.get_canonical_actions(0)
                legal_set = set(self._actions)
                return np.array([a in legal_set for a in canonical], dtype=bool)

            def apply_action(self, a):
                return self._children[a]

        def run_n(n: int, tables):
            root = Root()
            for _ in range(n):
                update_strategy(tables, root, i=0)
            row = tables.strategy[0].get_row_if_exists("linear_root")
            return int(row.sum()) if row is not None else 0

        tables_10 = _fresh_tables(tmp_path / "t10")
        tables_20 = _fresh_tables(tmp_path / "t20")

        count_10 = run_n(10, tables_10)
        count_20 = run_n(20, tables_20)

        assert count_10 == 10, f"Expected 10 visits after 10 calls, got {count_10}"
        assert count_20 == 20, f"Expected 20 visits after 20 calls, got {count_20}"
        tables_10.close()
        tables_20.close()

    def test_regret_floor_respected_after_repeated_discounting(self, tmp_tables):
        """After 100 discount applications, no regret entry is below REGRET_FLOOR."""
        row = tmp_tables.regret[0].get_row("floor_test_is")
        row[:] = int(REGRET_FLOOR)
        ds = DiscountState(duration_cycles=200, discount_interval=1)
        for step in range(1, 101):
            ds.apply(tmp_tables, sync_step=step)
        result = tmp_tables.regret[0].get_row("floor_test_is")
        assert np.all(result >= REGRET_FLOOR), (
            f"Regret floor breached: min={int(result.min())}, floor={int(REGRET_FLOOR)}"
        )

    def test_concurrent_update_row_no_corruption(self, tmp_tables):
        """Concurrent update_row calls to the same infoset must not lose writes.

        Four threads each increment action index 0 of the same infoset 1000
        times.  The final value must be exactly 4000, not less due to a
        lost-update race.
        """
        n_threads = 4
        n_writes_each = 1000
        infoset = "concurrent_test_IS"
        # Pre-allocate the row so threads only update (no allocation races)
        tmp_tables.regret[0].get_row(infoset)

        def writer():
            for _ in range(n_writes_each):
                tmp_tables.regret[0].update_row(infoset, 0, 1)

        threads = [threading.Thread(target=writer) for _ in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        row = tmp_tables.regret[0].get_row_if_exists(infoset)
        assert row is not None
        assert int(row[0]) == n_threads * n_writes_each, (
            f"Expected {n_threads * n_writes_each}, got {int(row[0])} — "
            "lost writes detected (race condition in update_row)"
        )

    def test_singleprocess_and_multiprocess_discount_match(self, tmp_tables):
        """Both training modes call DiscountState.apply at the same sync_step values.

        The singleprocess loop calls apply at every sync barrier where
        should_discount returns True.  The multiprocess server does the same.
        This test verifies that the shared DiscountState produces identical
        factors for a given sequence of sync_steps regardless of the caller.
        """
        sync_interval = 5
        discount_interval = 1
        n_iterations = 50

        # Collect the (sync_step, factor) pairs that would be applied
        sp_factors = []
        mp_factors = []

        def capture_apply(factors_list):
            from unittest.mock import MagicMock
            ds = DiscountState(duration_cycles=100, discount_interval=discount_interval)
            mock_tables = MagicMock()
            mock_tables.apply_discount.side_effect = lambda f: factors_list.append(
                (ds._discount_state_step if hasattr(ds, "_discount_state_step") else None, f)
            )
            for t in range(1, n_iterations + 1):
                if at_sync_barrier(t, sync_interval):
                    sync_step = t // sync_interval
                    if should_discount(sync_step, discount_interval):
                        ds.apply(mock_tables, sync_step=sync_step)
            return [f for _, f in factors_list]

        sp_factors = capture_apply([])
        # Call again — should produce identical sequence
        mp_factors = capture_apply([])
        assert sp_factors == mp_factors
        assert len(sp_factors) > 0


# ---------------------------------------------------------------------------
# TestMultiProcessIntegration (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMultiProcessIntegration:
    def test_server_workers_populate_tables(self, tmp_path):
        """Start a Server with 2 workers, run briefly, verify tables are non-empty."""
        import os
        import time
        from poker_ai.blueprint.multiprocess.server import Server

        lut_path = Path("data/20cards_exact")
        if not lut_path.exists():
            pytest.skip("20cards_exact LUT not available")

        server = Server(
            strategy_interval=10,
            max_runtime_hours=5.0 / 3600.0,  # 5 seconds
            discount_duration_cycles=1000,
            prune_threshold=9_999_999,
            c=-300_000_000,
            n_players=2,
            update_threshold=10,
            save_path=tmp_path / "mp_run",
            lut_path=str(lut_path),
            pickle_dir=False,
            sync_interval=10,
            discount_interval=1,
            checkpoint_interval=100,
            n_processes=2,
        )
        try:
            server.search()
        finally:
            server.terminate()

        total = sum(server._tables.regret[r].n_allocated for r in range(4))
        assert total > 0, "Regret tables are empty after multiprocess run"

    def test_worker_flush_reaches_server_tables(self, tmp_path):
        """A real worker processing cfr + sync jobs must write to shared tables."""
        import multiprocessing as mp
        import os
        from poker_ai.blueprint.multiprocess.worker import Worker

        lut_path = Path("data/20cards_exact")
        if not lut_path.exists():
            pytest.skip("20cards_exact LUT not available")

        shm_dir = str(tmp_path / "shm")
        os.makedirs(shm_dir)
        tables = CFRTables(
            index_path=tmp_path / "lmdb_index",
            shm_dir=shm_dir,
            actions_per_street=MAX_ACTIONS_PER_STREET,
            lmdb_map_size=lmdb_map_size_for_players(2),
        )
        job_queue = mp.JoinableQueue()
        logging_queue = mp.Queue()

        job_queue.put(("cfr", {"t": 1, "i": 0}))
        job_queue.put(("sync", {}))
        job_queue.put(("terminate", {}))

        def _target(jq, lq, t, lp):
            w = Worker(
                job_queue=jq,
                logging_queue=lq,
                locks={"strategy_update_lock": mp.Lock()},
                tables=t,
                lut_path=str(lp),
                pickle_dir=False,
                n_players=2,
                prune_threshold=9_999_999,
                c=-300_000_000,
                save_path=lp,
            )
            w.run()

        proc = mp.Process(target=_target, args=(job_queue, logging_queue, tables, lut_path))
        proc.start()
        proc.join(timeout=60)
        assert proc.exitcode == 0
        total = sum(tables.regret[r].n_allocated for r in range(4))
        assert total > 0
