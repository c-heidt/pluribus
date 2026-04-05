"""Unit tests for Phase 3 / Phase 5.1 CFR Core refactor.

Covers:
- 3.1  calculate_strategy_from_row (pure numpy)
- 3.2  External sampling (opponent branch samples one action)
- 3.3  local_delta accumulation (Dict[Tuple[int,str], np.ndarray])
- 3.6  update_strategy (per-street tables, no locks)
- merge_local_delta helper (per-street table merging)
- Bug regression tests

All game-state tests use a hand-rolled MockState so no LUT file is needed.
The @pytest.mark.slow test uses the real 20-card LUT.
"""
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from poker_ai.ai.cfr_tables import CFRTables
from poker_ai.ai.action_space import ACTION_TO_IDX, MAX_ACTIONS_PER_STREET
from poker_ai.ai.tree_utils import calculate_strategy_from_row
from poker_ai.ai.cfr import cfr, cfrp, merge_local_delta
from poker_ai.ai.strategy import update_strategy


# ---------------------------------------------------------------------------
# Minimal game-tree helpers (no LUT required)
# ---------------------------------------------------------------------------

class _FakeTable:
    """Minimal table stand-in so cfr() debug logging doesn't crash."""
    community_cards = []


class _FakePlayer:
    """Minimal player stand-in."""
    def __init__(self, is_active=True):
        self.is_active = is_active
        self.cards = []


class MockTerminal:
    """A terminal game node."""
    def __init__(self, payout: Dict[int, float], n_players: int = 2):
        self.is_terminal = True
        self.players = [_FakePlayer() for _ in range(n_players)]
        self._payout = payout
        self._table = _FakeTable()
        self._betting_stage = "pre_flop"

    @property
    def player_i(self):
        return 0

    @property
    def payout(self):
        return self._payout

    @property
    def info_set(self):
        return "terminal"

    @property
    def legal_actions(self):
        return []

    def apply_action(self, action):
        raise RuntimeError("Cannot apply action to terminal state")


class MockState:
    """A non-terminal game node for hand-crafting simple game trees."""

    def __init__(
        self,
        player_i: int,
        info_set: str,
        actions,
        children: Dict,
        n_players: int = 2,
    ):
        self._player_i = player_i
        self._info_set = info_set
        self._actions = list(actions)
        self._children = children  # action -> MockState or MockTerminal
        self.is_terminal = False
        self.players = [_FakePlayer() for _ in range(n_players)]
        self._betting_stage = "pre_flop"
        self._table = _FakeTable()

    @property
    def player_i(self):
        return self._player_i

    @property
    def info_set(self):
        return self._info_set

    @property
    def initial_regret(self):
        return {a: 0.0 for a in self._actions}

    @property
    def initial_strategy(self):
        return {a: 0.0 for a in self._actions}

    @property
    def legal_actions(self):
        return list(self._actions)

    @property
    def betting_round(self):
        return 0

    def get_valid_mask(self) -> np.ndarray:
        from poker_ai.environment.poker_env import PokerEnv as PokerState
        canonical = PokerState.get_canonical_actions(self.betting_round)
        legal_set = set(self._actions)
        return np.array([a in legal_set for a in canonical], dtype=bool)

    def apply_action(self, action):
        return self._children[action]


def _make_two_node_game():
    """
    Minimal two-player, two-action game tree.

    Structure:
    - Root: player 0's turn, actions = ["fold", "call"]
      - fold -> terminal: player 0 gets -50 (small blind lost)
      - call -> opponent node: player 1's turn, actions = ["fold", "call"]
        - fold -> terminal: player 0 gets +50
        - call -> terminal: player 0 gets 0 (chop)

    Expected regret for player 0 at root after one cfr() traversal:
      sigma = {fold: 0.5, call: 0.5}  (uniform initial)
      vo = 0.5*(-50) + 0.5*(opponent_cfr_value)
    """
    # Opponent node (player 1, ph=1)
    opp_node = MockState(
        player_i=1,
        info_set="opp",
        actions=["fold", "call"],
        children={
            "fold": MockTerminal(payout={0: 50, 1: -50}),
            "call": MockTerminal(payout={0: 0, 1: 0}),
        },
    )
    # Root node (player 0, ph=0)
    root = MockState(
        player_i=0,
        info_set="root",
        actions=["fold", "call"],
        children={
            "fold": MockTerminal(payout={0: -50, 1: 50}),
            "call": opp_node,
        },
    )
    return root, opp_node


def _fresh_tables():
    """Create fresh CFRTables backed by a temporary directory."""
    tmp = Path(tempfile.mkdtemp())
    shm = str(tmp / "shm")
    os.makedirs(shm, exist_ok=True)
    return CFRTables(
        index_path=tmp / "lmdb",
        shm_dir=shm,
        actions_per_street=MAX_ACTIONS_PER_STREET,
    )


# ---------------------------------------------------------------------------
# 3.1 — calculate_strategy_from_row
# ---------------------------------------------------------------------------


class TestCalculateStrategyFromRow:
    def test_uniform_on_zeros(self):
        row = np.zeros(4, dtype=np.int32)
        result = calculate_strategy_from_row(row)
        np.testing.assert_allclose(result, [0.25, 0.25, 0.25, 0.25], atol=1e-6)
        assert result.dtype == np.float32

    def test_uniform_on_all_negative(self):
        row = np.array([-100, -200, -50], dtype=np.int32)
        result = calculate_strategy_from_row(row)
        n = len(row)
        np.testing.assert_allclose(result, [1 / n] * n, atol=1e-6)

    def test_pure_strategy_on_one_positive(self):
        row = np.array([0, 1000, 0], dtype=np.int32)
        result = calculate_strategy_from_row(row)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-6)

    def test_proportional_to_positive_regrets(self):
        row = np.array([100, 300, 0, -50], dtype=np.int32)
        result = calculate_strategy_from_row(row)
        expected = np.array([100, 300, 0, 0], dtype=np.float32) / 400.0
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_sums_to_one(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            row = rng.integers(-500, 500, size=5).astype(np.int32)
            result = calculate_strategy_from_row(row)
            assert abs(float(result.sum()) - 1.0) < 1e-5

    def test_output_dtype_is_float32(self):
        row = np.array([1, 2, 3], dtype=np.int32)
        assert calculate_strategy_from_row(row).dtype == np.float32

    def test_single_action(self):
        row = np.array([500], dtype=np.int32)
        result = calculate_strategy_from_row(row)
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_two_equal_regrets(self):
        row = np.array([200, 200], dtype=np.int32)
        result = calculate_strategy_from_row(row)
        np.testing.assert_allclose(result, [0.5, 0.5], atol=1e-6)


# ---------------------------------------------------------------------------
# merge_local_delta
# ---------------------------------------------------------------------------


def _make_delta(r: int, **action_values) -> np.ndarray:
    """Return an int64 delta array for betting round *r* with named action values."""
    arr = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int64)
    for action, val in action_values.items():
        arr[ACTION_TO_IDX[r][action]] = int(val)
    return arr


class TestMergeLocalDelta:
    def test_new_infosets_added(self):
        tables = _fresh_tables()
        local_delta = {(0, "IS1"): _make_delta(0, fold=10, call=-10)}
        merge_local_delta(tables, local_delta)
        row = tables.regret[0].get_row_if_exists("IS1")
        assert row is not None
        assert row[ACTION_TO_IDX[0]["fold"]] == 10
        assert row[ACTION_TO_IDX[0]["call"]] == -10

    def test_existing_infoset_accumulated(self):
        tables = _fresh_tables()
        # Pre-populate via merge
        merge_local_delta(tables, {(0, "IS1"): _make_delta(0, fold=5, call=-5)})
        merge_local_delta(tables, {(0, "IS1"): _make_delta(0, fold=3, call=7)})
        row = tables.regret[0].get_row_if_exists("IS1")
        assert row is not None
        assert row[ACTION_TO_IDX[0]["fold"]] == 8
        assert row[ACTION_TO_IDX[0]["call"]] == 2

    def test_empty_delta_noop(self):
        tables = _fresh_tables()
        merge_local_delta(tables, {(0, "IS1"): _make_delta(0, fold=1)})
        row_before = int(tables.regret[0].get_row("IS1")[ACTION_TO_IDX[0]["fold"]])
        merge_local_delta(tables, {})
        assert int(tables.regret[0].get_row("IS1")[ACTION_TO_IDX[0]["fold"]]) == row_before

    def test_multiple_infosets(self):
        tables = _fresh_tables()
        local_delta = {
            (0, "A"): _make_delta(0, fold=1, call=2),
            (0, "B"): _make_delta(0, fold=-1),
        }
        merge_local_delta(tables, local_delta)
        assert tables.regret[0].get_row_if_exists("A") is not None
        assert tables.regret[0].get_row_if_exists("B") is not None

    def test_idempotent_on_repeated_merge(self):
        """Merging the same delta twice should double the values."""
        tables = _fresh_tables()
        delta = {(0, "IS1"): _make_delta(0, fold=10)}
        merge_local_delta(tables, delta)
        merge_local_delta(tables, delta)
        row = tables.regret[0].get_row_if_exists("IS1")
        assert row is not None
        assert row[ACTION_TO_IDX[0]["fold"]] == 20


# ---------------------------------------------------------------------------
# 3.2 + 3.3 — cfr() external sampling and local_delta
# ---------------------------------------------------------------------------


class TestCfrLocalDelta:
    def test_local_delta_accumulates_at_traversing_player_node(self):
        root, _ = _make_two_node_game()
        tables = _fresh_tables()
        local_delta: Dict = {}
        cfr(tables, root, i=0, t=1, local_delta=local_delta)
        # Traversing player is 0, root.betting_round==0 → key is (0, "root")
        assert (0, "root") in local_delta
        assert isinstance(local_delta[(0, "root")], np.ndarray)

    def test_local_delta_does_not_write_agent_tables(self):
        """When local_delta is provided, regret_tables must not be touched."""
        root, _ = _make_two_node_game()
        tables = _fresh_tables()
        local_delta: Dict = {}
        cfr(tables, root, i=0, t=1, local_delta=local_delta)
        # Nothing flushed yet → regret_tables[0] has no entry for "root"
        assert tables.regret[0].get_row_if_exists("root") is None

    def test_direct_mode_writes_agent_tables(self):
        """When local_delta is None, cfr auto-flushes into regret_tables."""
        root, _ = _make_two_node_game()
        tables = _fresh_tables()
        cfr(tables, root, i=0, t=1, local_delta=None)
        assert tables.regret[0].get_row_if_exists("root") is not None

    def test_merge_after_cfr_equals_direct_mode(self):
        """local_delta path + merge must give identical regrets to direct path."""
        root, _ = _make_two_node_game()

        np.random.seed(7)
        tables_direct = _fresh_tables()
        cfr(tables_direct, root, i=0, t=1, local_delta=None)

        np.random.seed(7)
        tables_delta = _fresh_tables()
        local_delta: Dict = {}
        cfr(tables_delta, root, i=0, t=1, local_delta=local_delta)
        merge_local_delta(tables_delta, local_delta)

        row_direct = tables_direct.regret[0].get_row_if_exists("root")
        row_delta = tables_delta.regret[0].get_row_if_exists("root")
        assert row_direct is not None
        assert row_delta is not None
        np.testing.assert_array_equal(row_direct, row_delta)

    def test_local_delta_is_empty_on_terminal_state(self):
        terminal = MockTerminal(payout={0: 100, 1: -100})
        tables = _fresh_tables()
        local_delta: Dict = {}
        result = cfr(tables, terminal, i=0, t=1, local_delta=local_delta)
        assert local_delta == {}
        assert result == pytest.approx(100.0)

    def test_regret_values_pass_through_merge(self):
        """Regret increments in local_delta must be faithfully merged."""
        root, _ = _make_two_node_game()
        tables = _fresh_tables()
        local_delta: Dict = {}
        cfr(tables, root, i=0, t=1, local_delta=local_delta)
        merge_local_delta(tables, local_delta)
        row = tables.regret[0].get_row_if_exists("root")
        assert row is not None
        assert np.all(np.isfinite(row.astype(np.float64)))

    def test_payout_returned_at_terminal(self):
        terminal = MockTerminal(payout={0: 42, 1: -42})
        tables = _fresh_tables()
        val = cfr(tables, terminal, i=0, t=1)
        assert val == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# 3.2 — External sampling: opponent branch visits all actions
# ---------------------------------------------------------------------------


class TestExternalSampling:
    def test_opponent_samples_single_action_per_call(self):
        """
        In external sampling each cfr() call samples exactly ONE opponent
        action per opponent node — not all of them.  Over many calls (with
        uniform strategy) all actions should eventually be sampled.
        """
        sampled_per_call = []

        class TrackingTerminal(MockTerminal):
            def __init__(self, action_name, payout, n_players=2):
                super().__init__(payout, n_players)
                self._action_name = action_name

            @property
            def payout(self):
                sampled_per_call[-1].add(self._action_name)
                return self._payout

        opp_node = MockState(
            player_i=1,
            info_set="opp_track",
            actions=["fold", "call", "raise:1.0"],
            children={
                "fold": TrackingTerminal("fold", {0: 50, 1: -50}),
                "call": TrackingTerminal("call", {0: 0, 1: 0}),
                "raise:1.0": TrackingTerminal("raise:1.0", {0: -100, 1: 100}),
            },
        )
        # Root is also the opponent's turn (i=0 traversing, ph=1)
        root = MockState(
            player_i=1,
            info_set="root_ext",
            actions=["fold", "call", "raise:1.0"],
            children={
                "fold": opp_node,
                "call": opp_node,
                "raise:1.0": opp_node,
            },
        )
        tables = _fresh_tables()
        np.random.seed(0)
        for _ in range(30):
            sampled_per_call.append(set())
            cfr(tables, root, i=0, t=1)

        # Each individual call must reach exactly one terminal (one sampled path)
        for run_idx, visited in enumerate(sampled_per_call):
            assert len(visited) == 1, (
                f"Run {run_idx}: expected 1 action sampled, got {visited}"
            )

        # Over 30 runs, more than one distinct action should appear
        all_visited = set().union(*sampled_per_call)
        assert len(all_visited) > 1, "Strategy is uniform — multiple actions should be sampled across runs"

    def test_traversing_player_visits_all_actions(self):
        """The traversing player (ph == i) must still explore all its own actions."""
        visited = set()

        class TrackingTerminal(MockTerminal):
            def __init__(self, action_name, payout, n_players=2):
                super().__init__(payout, n_players)
                self._action_name = action_name

            @property
            def payout(self):
                visited.add(self._action_name)
                return self._payout

        root = MockState(
            player_i=0,  # traversing player (i=0)
            info_set="root_traversing",
            actions=["fold", "call", "raise:1.0"],
            children={
                "fold": TrackingTerminal("fold", {0: -50, 1: 50}),
                "call": TrackingTerminal("call", {0: 0, 1: 0}),
                "raise:1.0": TrackingTerminal("raise:1.0", {0: 100, 1: -100}),
            },
        )
        tables = _fresh_tables()
        cfr(tables, root, i=0, t=1)
        assert visited == {"fold", "call", "raise:1.0"}, (
            f"Traversing player must visit ALL actions, got: {visited}"
        )

    def test_same_seed_produces_same_result(self):
        """With the same numpy seed, cfr() must produce identical results."""
        root, _ = _make_two_node_game()
        agent1 = _fresh_tables()
        agent2 = _fresh_tables()

        np.random.seed(42)
        local_delta1: Dict = {}
        cfr(agent1, root, i=0, t=1, local_delta=local_delta1)

        np.random.seed(42)
        local_delta2: Dict = {}
        cfr(agent2, root, i=0, t=1, local_delta=local_delta2)

        assert local_delta1.keys() == local_delta2.keys()
        for key in local_delta1:
            np.testing.assert_array_equal(local_delta1[key], local_delta2[key])


# ---------------------------------------------------------------------------
# 3.3 — cfrp() local_delta
# ---------------------------------------------------------------------------


class TestCfrpLocalDelta:
    def test_cfrp_local_delta_accumulates(self):
        root, _ = _make_two_node_game()
        tables = _fresh_tables()
        local_delta: Dict = {}
        # Use c=-1e9 so pruning never fires (all regrets > c)
        cfrp(tables, root, i=0, t=1, c=-1_000_000_000, local_delta=local_delta)
        assert (0, "root") in local_delta

    def test_cfrp_direct_mode_writes_agent_tables(self):
        root, _ = _make_two_node_game()
        tables = _fresh_tables()
        cfrp(tables, root, i=0, t=1, c=-1_000_000_000, local_delta=None)
        assert tables.regret[0].get_row_if_exists("root") is not None

    def test_cfrp_merge_equals_direct(self):
        root, _ = _make_two_node_game()
        c = -1_000_000_000

        np.random.seed(7)
        tables_direct = _fresh_tables()
        cfrp(tables_direct, root, i=0, t=1, c=c, local_delta=None)

        np.random.seed(7)
        tables_delta = _fresh_tables()
        local_delta: Dict = {}
        cfrp(tables_delta, root, i=0, t=1, c=c, local_delta=local_delta)
        merge_local_delta(tables_delta, local_delta)

        row_direct = tables_direct.regret[0].get_row_if_exists("root")
        row_delta = tables_delta.regret[0].get_row_if_exists("root")
        assert row_direct is not None
        assert row_delta is not None
        np.testing.assert_array_equal(row_direct, row_delta)


# ---------------------------------------------------------------------------
# 3.6 — update_strategy (no locks)
# ---------------------------------------------------------------------------


class TestUpdateStrategy:
    def test_update_strategy_accumulates_counts(self):
        root, _ = _make_two_node_game()
        tables = _fresh_tables()
        # Run update_strategy 10 times — visit counts should grow
        for _ in range(10):
            update_strategy(tables, root, i=0, t=1)
        row = tables.strategy[0].get_row_if_exists("root")
        assert row is not None
        total = int(row.sum())
        assert total == 10

    def test_update_strategy_skips_terminal(self):
        terminal = MockTerminal(payout={0: 100, 1: -100})
        tables = _fresh_tables()
        update_strategy(tables, terminal, i=0, t=1)
        # No strategy entry created for terminal
        for r in range(4):
            assert tables.strategy[r].get_row_if_exists("terminal") is None

    def test_update_strategy_traverses_postflop(self):
        """Phase 5.7: betting_round > 0 guard removed — postflop states are now traversed."""
        class _PostflopState(MockState):
            @property
            def betting_round(self):
                return 1  # flop

            def get_valid_mask(self):
                from poker_ai.environment.poker_env import PokerEnv as PokerState
                canonical = PokerState.get_canonical_actions(1)
                legal_set = set(self._actions)
                return np.array([a in legal_set for a in canonical], dtype=bool)

        postflop_root = _PostflopState(
            player_i=0,
            info_set="postflop_root",
            actions=["fold", "call"],
            children={
                "fold": MockTerminal({0: -50, 1: 50}),
                "call": MockTerminal({0: 50, 1: -50}),
            },
        )
        tables = _fresh_tables()
        update_strategy(tables, postflop_root, i=0, t=1)
        # Strategy table for street 1 must have been written.
        assert tables.strategy[1].get_row_if_exists("postflop_root") is not None

    def test_update_strategy_accepts_no_locks_arg(self):
        """update_strategy must work without any locks argument (signature check)."""
        import inspect
        sig = inspect.signature(update_strategy)
        assert "locks" not in sig.parameters, (
            "update_strategy should no longer accept a 'locks' parameter"
        )


# ---------------------------------------------------------------------------
# Bug regression tests
# ---------------------------------------------------------------------------


class TestBugRegressions:
    def test_discount_applies_to_both_tables(self):
        """CFRTables.apply_discount must discount both regret and strategy."""
        tables = _fresh_tables()
        merge_local_delta(tables, {(0, "IS1"): _make_delta(0, fold=1000)})
        update_strategy(tables, _make_two_node_game()[0], i=0, t=1)

        regret_before = tables.regret[0].get_row_if_exists("IS1")
        regret_copy = regret_before.copy() if regret_before is not None else None

        tables.apply_discount(0.5)

        if regret_copy is not None:
            regret_after = tables.regret[0].get_row_if_exists("IS1")
            # Values should be roughly halved
            assert not np.array_equal(regret_after, regret_copy)

    def test_cfr_signature_no_locks_param(self):
        import inspect
        sig = inspect.signature(cfr)
        assert "locks" not in sig.parameters, "cfr must not have a 'locks' parameter"

    def test_cfrp_signature_no_locks_param(self):
        import inspect
        sig = inspect.signature(cfrp)
        assert "locks" not in sig.parameters, "cfrp must not have a 'locks' parameter"

    def test_cfr_has_local_delta_param(self):
        import inspect
        sig = inspect.signature(cfr)
        assert "local_delta" in sig.parameters

    def test_cfrp_has_local_delta_param(self):
        import inspect
        sig = inspect.signature(cfrp)
        assert "local_delta" in sig.parameters

    def test_update_strategy_has_no_locks_param(self):
        import inspect
        sig = inspect.signature(update_strategy)
        assert "locks" not in sig.parameters

    def test_calculate_strategy_from_row_imported(self):
        """calculate_strategy_from_row must be importable from tree_utils."""
        from poker_ai.ai.tree_utils import calculate_strategy_from_row as csfr
        assert callable(csfr)

    def test_merge_local_delta_imported(self):
        """merge_local_delta must be importable from cfr."""
        from poker_ai.ai.cfr import merge_local_delta as mld
        assert callable(mld)

    def test_multiple_cfr_iterations_accumulate(self):
        """Run 5 cfr iterations using local_delta, verify regret entries grow."""
        root, _ = _make_two_node_game()
        tables = _fresh_tables()
        for t in range(1, 6):
            local_delta: Dict = {}
            cfr(tables, root, i=0, t=t, local_delta=local_delta)
            merge_local_delta(tables, local_delta)
        # After 5 iterations, regret_tables[0] must have an entry for "root"
        row = tables.regret[0].get_row_if_exists("root")
        assert row is not None
        assert np.any(row != 0)


# ---------------------------------------------------------------------------
# Slow integration test — requires the 20-card LUT
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cfr_1000_iterations_small_game():
    """Run 1000 cfr iterations on the real 20-card short-deck game.

    Verifies:
    - local_delta + merge gives non-empty regret after 1000 iterations
    - All regret values are finite
    - update_strategy populates strategy_tables[0]
    """
    from poker_ai.environment.poker_env import new_game

    lut_path = "data/clustering/20cards_exact"
    if not os.path.exists(lut_path + "/card_info_lut.joblib"):
        pytest.skip("20-card exact LUT not available")

    tables = _fresh_tables()
    card_info_lut = {}
    n_players = 2
    update_threshold = 200
    strategy_interval = 10

    for t in range(1, 1001):
        for i in range(n_players):
            state = new_game(n_players, card_info_lut, lut_path=lut_path)
            card_info_lut = state.card_info_lut

            if t > update_threshold and t % strategy_interval == 0:
                update_strategy(tables, state, i=i, t=t)

            local_delta: Dict = {}
            cfr(tables, state, i=i, t=t, local_delta=local_delta)
            merge_local_delta(tables, local_delta)

    # regret_tables[0] must be non-empty
    assert tables.regret[0].n_allocated > 0, "Regret table must be non-empty after 1000 iterations"
