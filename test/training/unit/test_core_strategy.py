"""Strategy-sampling walk in the compiled core — byte-exact vs the Python reference.

The average-strategy counterpart of the Phase-3 traverse gate
(:mod:`test.training.unit.test_core_traverse`).  ``strategy_step`` /
``update_strategy`` used to run only on the Python path; it is now ported into the
core (``strategy_replay`` / ``strategy_rng``) so the strategy pass can be
core-accelerated and decoupled from the sync barrier.  These gates certify the
port the same way the cfr traversal is certified:

* **byte-exact replay differential** (the strongest gate) — record a Python
  ``update_strategy`` pre-flop pass's sampled actions, replay the *same* sequence
  into both the Python reference and the core, and assert the two visit-count
  ``local_delta`` dicts are identical.  The strategy walk now samples at
  **traverser** nodes only and branches over every legal action at opponent nodes
  (the mirror of the cfr walk), so the recorded sequence spans player nodes only —
  and if the core's opponent branching diverged, its accumulated counts would
  differ from the branching Python reference and this gate would fail.
* **truncation non-vacuity** — a short replay must raise, proving the gate is not
  trivially satisfied.
* **rng path structure** + in-place accumulation — the production sampler is not
  certified by RNG byte-parity (deliberately), so it is checked for well-formed
  output and the caller-owned-buffer contract ``strategy_rng`` relies on.
* **merge** — ``merge_local_strategy_delta`` writes the accumulated counts into
  ``tables.strategy`` intact (the batched +Δ that replaces the old per-visit
  ``update_row``).

Uses cache-enabled tables (``requires_lut``): the in-core read path is pure-shm.
"""

import numpy as np
import pytest

import poker_ai.blueprint.cfr as cfr_mod
from environment.action_space import (
    ACTION_TO_IDX,
    CANONICAL_ACTIONS,
    MAX_ACTIONS_PER_STREET,
)
from environment.poker_env import (
    MAX_RAISES_PER_ROUND,
    RAISE_SIZES_BY_STAGE,
    _ACTION_BYTE,
    _STAGE_ID,
    new_game,
)
from poker_ai.blueprint.cfr import merge_local_strategy_delta
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.tables.index import lmdb_map_size_for_players
from poker_ai._core import _state as cy
from poker_ai._core import _traverse as cyt
from test.training.core_diff import (
    assert_local_delta_equal,
    run_recording_strategy,
    run_replay_strategy,
)

# Dump the encoding alphabet + raise grid into the compiled state engine once.
cy.configure(_STAGE_ID, _ACTION_BYTE, RAISE_SIZES_BY_STAGE, MAX_RAISES_PER_ROUND)

N_PLAYERS = 2
_CAPS = {r: 1 << 16 for r in range(4)}


def _build_cached_tables(tmp_path, lut, n_players, n_iters=40, seed=123):
    """Cache-enabled, lightly pre-trained tables (non-uniform regrets → real σ)."""
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


def _core_tables(tables):
    return cyt.CoreTables(
        tables, CANONICAL_ACTIONS, ACTION_TO_IDX, MAX_ACTIONS_PER_STREET
    )


@pytest.mark.requires_lut
class TestCoreStrategyByteIdentical:
    def test_visit_counts_byte_identical(self, tmp_path, lut):
        """Core strategy replay reproduces the Python visit counts exactly."""
        tables = _build_cached_tables(tmp_path, lut, N_PLAYERS)
        try:
            ct = _core_tables(tables)
            total_choices = 0
            total_counts = 0
            compared = 0
            for deal_seed in range(16):
                np.random.seed(3000 + deal_seed)
                state = new_game(N_PLAYERS, lut)
                for i in range(N_PLAYERS):
                    delta_py, choices = run_recording_strategy(
                        tables, state, i, seed=deal_seed
                    )
                    # Cross-check the Python replay reproduces the recording too
                    # (guards the harness itself), then gate the core against it.
                    delta_py_replay = run_replay_strategy(tables, state, i, choices)
                    assert_local_delta_equal(delta_py, delta_py_replay)

                    fast_state = cy.FastState.from_poker_env(state)
                    delta_core = cyt.strategy_replay(ct, fast_state, i, choices)
                    assert_local_delta_equal(delta_py, delta_core)

                    total_choices += len(choices)
                    total_counts += sum(int(v.sum()) for v in delta_core.values())
                    compared += 1
            assert compared > 0
            # Traverser nodes sampled (choices recorded)...
            assert total_choices > 0
            # ...and player-i nodes actually recorded visit counts.
            assert total_counts > 0
        finally:
            tables.close()

    def test_truncated_replay_raises(self, tmp_path, lut):
        """A short replay into the core raises (non-vacuity)."""
        tables = _build_cached_tables(tmp_path, lut, N_PLAYERS)
        try:
            ct = _core_tables(tables)
            found = False
            for deal_seed in range(20):
                np.random.seed(6000 + deal_seed)
                state = new_game(N_PLAYERS, lut)
                for i in range(N_PLAYERS):
                    _, choices = run_recording_strategy(
                        tables, state, i, seed=deal_seed
                    )
                    if len(choices) >= 1:
                        found = True
                        fast_state = cy.FastState.from_poker_env(state)
                        with pytest.raises(AssertionError):
                            cyt.strategy_replay(ct, fast_state, i, choices[:-1])
                        break
                if found:
                    break
            assert found, "no strategy walk with a sampled node found to truncate"
        finally:
            tables.close()

    def test_rng_path_runs_and_shapes(self, tmp_path, lut):
        """The production rng path returns a well-formed visit-count local_delta."""
        tables = _build_cached_tables(tmp_path, lut, N_PLAYERS)
        try:
            ct = _core_tables(tables)
            rng = np.random.RandomState(7)
            np.random.seed(4242)
            produced = 0
            for _ in range(20):
                state = new_game(N_PLAYERS, lut)
                for i in range(N_PLAYERS):
                    fast_state = cy.FastState.from_poker_env(state)
                    delta = cyt.strategy_rng(ct, fast_state, i, rng)
                    for (r, iset), row in delta.items():
                        assert isinstance(r, int) and isinstance(iset, bytes)
                        assert row.dtype == np.int64
                        assert row.shape == (MAX_ACTIONS_PER_STREET[r],)
                        assert (row >= 0).all()  # visit counts are non-negative
                        produced += 1
            assert produced > 0
        finally:
            tables.close()

    def test_rng_accumulates_in_place(self, tmp_path, lut):
        """strategy_rng accumulates into a caller-owned dict across playthroughs."""
        tables = _build_cached_tables(tmp_path, lut, N_PLAYERS)
        try:
            ct = _core_tables(tables)
            rng = np.random.RandomState(11)
            np.random.seed(555)
            lsd = {}
            for _ in range(10):
                state = new_game(N_PLAYERS, lut)
                for i in range(N_PLAYERS):
                    fast_state = cy.FastState.from_poker_env(state)
                    ret = cyt.strategy_rng(
                        ct, fast_state, i, rng, local_strategy_delta=lsd
                    )
                    assert ret is lsd  # the same buffer is threaded, not replaced
            assert sum(int(v.sum()) for v in lsd.values()) > 0
        finally:
            tables.close()

    def test_merge_writes_accumulated_counts(self, tmp_path, lut):
        """merge_local_strategy_delta lands the accumulated counts in tables.strategy.

        The shared strategy tables start empty (only ``cfr`` pre-trained the
        regrets), so after a merge every touched row must equal the accumulated
        delta exactly — the batched +Δ that replaces the barriered per-visit
        ``update_row(+1)`` sequence.
        """
        tables = _build_cached_tables(tmp_path, lut, N_PLAYERS)
        try:
            ct = _core_tables(tables)
            rng = np.random.RandomState(13)
            np.random.seed(999)
            lsd = {}
            for _ in range(12):
                state = new_game(N_PLAYERS, lut)
                for i in range(N_PLAYERS):
                    fast_state = cy.FastState.from_poker_env(state)
                    cyt.strategy_rng(
                        ct, fast_state, i, rng, local_strategy_delta=lsd
                    )
            assert lsd, "no strategy counts accumulated"
            expected = {k: v.copy() for k, v in lsd.items()}

            merge_local_strategy_delta(tables, lsd)

            for (r, info_set), row in expected.items():
                stored = tables.strategy[r].get_row_if_exists(info_set)
                assert stored is not None, f"row for {(r, info_set)!r} not written"
                assert np.array_equal(stored, row), (
                    f"strategy row for {(r, info_set)!r} mismatch: "
                    f"{stored.tolist()} vs {row.tolist()}"
                )
        finally:
            tables.close()


@pytest.mark.requires_lut
class TestCoreStrategyMultiway:
    """3-player gate: exercises the folded-``i``-non-terminal descent — the
    ``not is_seat_active(i)`` early return at a node where other seats keep acting.
    Heads-up never distinguishes it (a fold there ends the hand)."""

    def test_visit_counts_byte_identical_3player(self, tmp_path, lut):
        try:
            new_game(3, lut)
        except Exception as exc:  # pragma: no cover - LUT may be HU-only
            pytest.skip(f"3-player new_game unsupported on this LUT: {exc}")
        tables = _build_cached_tables(tmp_path, lut, n_players=3, n_iters=30)
        try:
            ct = _core_tables(tables)
            compared = 0
            for deal_seed in range(16):
                np.random.seed(7000 + deal_seed)
                state = new_game(3, lut)
                for i in range(3):
                    delta_py, choices = run_recording_strategy(
                        tables, state, i, seed=deal_seed
                    )
                    fast_state = cy.FastState.from_poker_env(state)
                    delta_core = cyt.strategy_replay(ct, fast_state, i, choices)
                    assert_local_delta_equal(delta_py, delta_core)
                    compared += 1
            assert compared > 0
        finally:
            tables.close()
