"""Phase-4 wiring — the compiled-core driver behind ``PLURIBUS_CFR_CORE``.

Phase 3 proved the in-core ``_traverse`` byte-identical to Python under a
recorded opponent sequence.  Phase 4 is the *plumbing*: the ``core_enabled``
gate, the :class:`CoreDriver` that builds a ``CoreTables`` and runs a traversal
into a caller-owned ``local_delta``, the pure-shm cache guardrail, and the
end-to-end single-process loop with the flag set.  RNG byte-parity with the
Python path is not pursued (per the plan), so these gates check *structure* and
*invariants*, not byte-equality of a live run:

* ``core_enabled`` gates on the flag and refuses biased runs (not ported).
* :class:`CoreDriver` requires the shm cache and verifies it mirrors the index.
* ``run_cfr`` accumulates well-formed ``(round, bytes) -> int64`` rows in place.
* a short ``simple_search`` with ``PLURIBUS_CFR_CORE=1`` runs and allocates rows.
"""

import numpy as np
import pytest

import poker_ai.blueprint.cfr as cfr_mod
from environment.action_space import MAX_ACTIONS_PER_STREET
from environment.poker_env import new_game
from poker_ai.blueprint.core_runner import CoreDriver, core_enabled
from poker_ai.blueprint.training import cfr_step
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.tables.index import lmdb_map_size_for_players

N_PLAYERS = 2
_CAPS = {r: 1 << 16 for r in range(4)}


def _cached_tables(tmp_path, lut, n_players=N_PLAYERS, n_iters=30, seed=123):
    """Cache-enabled, lightly pre-trained tables (mirrors test_core_traverse)."""
    shm = tmp_path / "shm"
    shm.mkdir(parents=True, exist_ok=True)
    tables = CFRTables(
        index_path=tmp_path / "lmdb_index",
        shm_dir=str(shm),
        actions_per_street=MAX_ACTIONS_PER_STREET,
        lmdb_map_size=lmdb_map_size_for_players(n_players),
        enable_index_cache=True,
        index_capacities=_CAPS,
    )
    np.random.seed(seed)
    for t in range(1, n_iters + 1):
        for i in range(n_players):
            cfr_mod.cfr(tables, new_game(n_players, lut), i, t)
    tables.prewarm_caches()
    return tables


# ---------------------------------------------------------------------------
# core_enabled gate
# ---------------------------------------------------------------------------


class TestCoreEnabled:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("PLURIBUS_CFR_CORE", raising=False)
        assert core_enabled() is False

    def test_on_when_flag_set(self, monkeypatch):
        monkeypatch.setenv("PLURIBUS_CFR_CORE", "1")
        assert core_enabled("none") is True

    def test_off_for_any_non_one_value(self, monkeypatch):
        monkeypatch.setenv("PLURIBUS_CFR_CORE", "0")
        assert core_enabled() is False

    def test_biased_run_falls_back_to_python(self, monkeypatch):
        # The biased traversal is not ported into the core (deferred); a biased
        # run must silently stay on the Python path, not drop the bias.
        monkeypatch.setenv("PLURIBUS_CFR_CORE", "1")
        assert core_enabled(bias="fold") is False


# ---------------------------------------------------------------------------
# CoreDriver construction guardrails
# ---------------------------------------------------------------------------


@pytest.mark.requires_lut
class TestCoreDriverGuardrails:
    def test_requires_index_cache(self, tmp_path, lut):
        shm = tmp_path / "shm"
        shm.mkdir(parents=True, exist_ok=True)
        tables = CFRTables(
            index_path=tmp_path / "lmdb_index",
            shm_dir=str(shm),
            actions_per_street=MAX_ACTIONS_PER_STREET,
            lmdb_map_size=lmdb_map_size_for_players(N_PLAYERS),
            enable_index_cache=False,
        )
        try:
            with pytest.raises(RuntimeError, match="pure-shm"):
                CoreDriver(tables)
        finally:
            tables.close()

    def test_rejects_incomplete_cache_mirror(self, tmp_path, lut):
        # A cache that is not a complete mirror of the index would make the
        # pure-shm read path silently return "uniform" on the missing rows.
        # The driver must refuse to start, loudly.
        tables = _cached_tables(tmp_path, lut)
        try:
            # Desync one street's occupancy counter from its allocated rows.
            tables._index_caches[0]._occupancy.value -= 1
            with pytest.raises(RuntimeError, match="complete mirror"):
                CoreDriver(tables)
        finally:
            tables._index_caches[0]._occupancy.value += 1
            tables.close()

    def test_builds_on_prewarmed_cache(self, tmp_path, lut):
        tables = _cached_tables(tmp_path, lut)
        try:
            driver = CoreDriver(tables)  # must not raise
            assert driver is not None
        finally:
            tables.close()


# ---------------------------------------------------------------------------
# run_cfr — well-formed in-place accumulation
# ---------------------------------------------------------------------------


@pytest.mark.requires_lut
class TestRunCfr:
    def test_accumulates_wellformed_local_delta(self, tmp_path, lut):
        tables = _cached_tables(tmp_path, lut)
        try:
            driver = CoreDriver(tables)
            local_delta = {}
            np.random.seed(7)
            state = new_game(N_PLAYERS, lut)
            driver.run_cfr(state, 0, 100, c=0, prune_on=False, local_delta=local_delta)
            assert local_delta, "core produced no regret rows"
            widths = MAX_ACTIONS_PER_STREET
            for (r, iset), row in local_delta.items():
                assert isinstance(r, int) and 0 <= r < 4
                assert isinstance(iset, bytes)
                assert row.dtype == np.int64
                assert row.shape == (widths[r],)
        finally:
            tables.close()

    def test_dispatch_through_cfr_step(self, tmp_path, lut):
        # cfr_step(core=driver) must route to the core and leave local_delta
        # populated in the same shape the Python path leaves it.
        tables = _cached_tables(tmp_path, lut)
        try:
            driver = CoreDriver(tables)
            local_delta = {}
            np.random.seed(11)
            state = new_game(N_PLAYERS, lut)
            cfr_step(
                tables, state, 0, 100, prune_threshold=0, c=0,
                local_delta=local_delta, core=driver,
            )
            assert local_delta
        finally:
            tables.close()

    def test_batch_accumulates_in_place(self, tmp_path, lut):
        # Two traversals into one buffer: shared keys must sum, not overwrite —
        # the worker's persistent-batch semantics.
        tables = _cached_tables(tmp_path, lut)
        try:
            driver = CoreDriver(tables)
            np.random.seed(3)
            buf = {}
            for _ in range(5):
                driver.run_cfr(
                    new_game(N_PLAYERS, lut), 0, 100, c=0, prune_on=False,
                    local_delta=buf,
                )
            assert buf
            # Every row is int64 and width-correct after repeated += merges.
            for (r, _iset), row in buf.items():
                assert row.dtype == np.int64
                assert row.shape == (MAX_ACTIONS_PER_STREET[r],)
        finally:
            tables.close()


# ---------------------------------------------------------------------------
# End-to-end single-process loop under the flag
# ---------------------------------------------------------------------------


@pytest.mark.requires_lut
class TestSingleProcessWithCore:
    def test_short_run_allocates_rows(self, tmp_path, lut, monkeypatch):
        from test.conftest import _LUT_PATH
        from poker_ai.blueprint.singleprocess.train import simple_search

        monkeypatch.setenv("PLURIBUS_CFR_CORE", "1")
        monkeypatch.setenv("PLURIBUS_INDEX_CACHE", "1")
        monkeypatch.setenv("PLURIBUS_INDEX_CAPACITY", "65536,65536,65536,65536")
        save_path = tmp_path / "run"
        save_path.mkdir()
        simple_search(
            config={},
            save_path=save_path,
            lut_path=str(_LUT_PATH),
            pickle_dir=False,
            strategy_interval=2,
            n_iterations=25,
            discount_duration_cycles=100,
            prune_threshold=1_000_000,  # never prune in this short run
            c=-300_000_000,
            n_players=N_PLAYERS,
            update_threshold=0,
            sync_interval=5,
            discount_interval=10,
        )
        # The core path must actually populate the shared index (proves the
        # merge landed rows, not just that the loop returned).
        from poker_ai.tables.index import InfosetIndex
        total = 0
        for r in range(4):
            idx = InfosetIndex(save_path / "lmdb_index" / f"street_{r}")
            total += idx.n_allocated_rows
            idx.close()
        assert total > 0, "core-driven run allocated no infosets"
