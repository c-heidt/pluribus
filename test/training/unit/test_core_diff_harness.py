"""Validate the record/replay differential harness (Phase 0 safety net).

The harness (:mod:`test.training.core_diff`) is the byte-exact, RNG-free oracle
the compiled-core rewrite is certified against.  Before trusting it in Phase 3
it must be shown to (a) reproduce a traversal's ``local_delta`` exactly when the
same opponent sequence is replayed, (b) be deterministic, and — crucially — (c)
actually *detect* divergence, so a passing comparison is meaningful rather than
vacuous.

Uses the 20-card LUT (``requires_lut``); the tables are lightly pre-trained so
regret matching runs on realistic non-uniform regrets rather than the degenerate
all-uniform fresh state.
"""

import numpy as np
import pytest

import poker_ai.blueprint.cfr as cfr_mod
from environment.action_space import MAX_ACTIONS_PER_STREET
from environment.poker_env import new_game
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.tables.index import lmdb_map_size_for_players
from test.training.core_diff import (
    assert_local_delta_equal,
    run_recording,
    run_replay,
)

N_PLAYERS = 2
_TRAVERSAL_T = 100  # arbitrary iteration index passed through to the traversal


@pytest.fixture
def trained_tables(tmp_path, lut):
    """Fresh CFR tables, lightly pre-trained for non-uniform regrets.

    Pre-training merges into the tables (no explicit ``local_delta``); it is
    seeded so the resulting snapshot is identical every run, which keeps the
    recorded opponent sequences comparable across runs.
    """
    shm_dir = tmp_path / "shm"
    shm_dir.mkdir(parents=True, exist_ok=True)
    tables = CFRTables(
        index_path=tmp_path / "lmdb_index",
        shm_dir=str(shm_dir),
        actions_per_street=MAX_ACTIONS_PER_STREET,
        lmdb_map_size=lmdb_map_size_for_players(N_PLAYERS),
    )
    np.random.seed(123)
    for t in range(1, 40):
        for i in range(N_PLAYERS):
            cfr_mod.cfr(tables, new_game(N_PLAYERS, lut), i, t)
    yield tables
    tables.close()


@pytest.mark.requires_lut
class TestRecordReplay:
    def test_record_replay_byte_identical(self, trained_tables, lut):
        """Replaying a recorded opponent sequence reproduces local_delta exactly."""
        total_opponent_choices = 0
        for deal_seed in range(10):
            np.random.seed(2000 + deal_seed)
            state = new_game(N_PLAYERS, lut)
            for i in range(N_PLAYERS):
                delta_rec, choices = run_recording(
                    trained_tables, state, i, _TRAVERSAL_T, seed=deal_seed
                )
                delta_rep = run_replay(
                    trained_tables, state, i, _TRAVERSAL_T, choices
                )
                assert_local_delta_equal(delta_rec, delta_rep)
                total_opponent_choices += len(choices)
        # The suite must actually exercise opponent (external-sampling) nodes,
        # otherwise the replay path is never tested.
        assert total_opponent_choices > 0

    def test_recording_is_deterministic(self, trained_tables, lut):
        """Same (hand, player, seed) records the same choices and the same delta."""
        np.random.seed(4242)
        state = new_game(N_PLAYERS, lut)
        d1, c1 = run_recording(trained_tables, state, 0, _TRAVERSAL_T, seed=7)
        d2, c2 = run_recording(trained_tables, state, 0, _TRAVERSAL_T, seed=7)
        assert c1 == c2
        assert_local_delta_equal(d1, d2)

    def test_harness_detects_truncated_replay(self, trained_tables, lut):
        """A short replay must raise, not silently pass — proves non-vacuity.

        Find a (hand, player) with at least one opponent node, drop the last
        recorded choice, and confirm the driven walk out-runs the sequence
        (``replay exhausted``) instead of quietly agreeing.
        """
        found = False
        for deal_seed in range(20):
            np.random.seed(5000 + deal_seed)
            state = new_game(N_PLAYERS, lut)
            for i in range(N_PLAYERS):
                _, choices = run_recording(
                    trained_tables, state, i, _TRAVERSAL_T, seed=deal_seed
                )
                if len(choices) >= 1:
                    found = True
                    with pytest.raises(AssertionError):
                        run_replay(
                            trained_tables, state, i, _TRAVERSAL_T, choices[:-1]
                        )
                    break
            if found:
                break
        assert found, "no hand with an opponent node found to test truncation"

    def test_harness_detects_perturbed_delta(self, trained_tables, lut):
        """assert_local_delta_equal must reject a one-chip perturbation."""
        np.random.seed(9999)
        state = new_game(N_PLAYERS, lut)
        delta, _ = run_recording(trained_tables, state, 0, _TRAVERSAL_T, seed=1)
        if not delta:
            pytest.skip("degenerate hand produced no regret updates")
        perturbed = {k: v.copy() for k, v in delta.items()}
        key = next(iter(perturbed))
        perturbed[key][0] += 1  # a single-chip drift, the subtlest real failure
        with pytest.raises(AssertionError):
            assert_local_delta_equal(delta, perturbed)
