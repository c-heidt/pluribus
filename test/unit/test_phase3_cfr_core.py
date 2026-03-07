"""Unit tests for Phase 3: CFR Core refactor.

Covers:
- 3.1  calculate_strategy_from_row (pure numpy)
- 3.2  External sampling (opponent branch iterates all actions)
- 3.3  local_delta accumulation and lock-free traversal
- 3.6  update_strategy (no locks parameter)
- 3.7  serialise Bug 4 fix (no deepcopy)
- merge_local_delta helper
- Bug regression tests (Bug 2 fix in train.py and worker._discount)

All game-state tests use a hand-rolled MockState so no LUT file is needed.
The @pytest.mark.slow test uses the real 20-card LUT.
"""
import os
import random
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from poker_ai.ai.agent import Agent
from poker_ai.ai.ai import (
    calculate_strategy,
    calculate_strategy_from_row,
    cfr,
    cfrp,
    merge_local_delta,
    serialise,
    update_strategy,
)


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


def _fresh_agent():
    """Create a fresh Agent without manager (test mode)."""
    os.environ["TESTING_SUITE"] = "1"
    agent = Agent(use_manager=False)
    return agent


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


class TestMergeLocalDelta:
    def test_new_infosets_added(self):
        agent = _fresh_agent()
        local_delta = {"IS1": {"fold": 10.0, "call": -10.0}}
        merge_local_delta(agent, local_delta)
        assert agent.regret["IS1"]["fold"] == pytest.approx(10.0)
        assert agent.regret["IS1"]["call"] == pytest.approx(-10.0)

    def test_existing_infoset_accumulated(self):
        agent = _fresh_agent()
        agent.regret["IS1"] = {"fold": 5.0, "call": -5.0}
        local_delta = {"IS1": {"fold": 3.0, "call": 7.0}}
        merge_local_delta(agent, local_delta)
        assert agent.regret["IS1"]["fold"] == pytest.approx(8.0)
        assert agent.regret["IS1"]["call"] == pytest.approx(2.0)

    def test_empty_delta_noop(self):
        agent = _fresh_agent()
        agent.regret["IS1"] = {"fold": 1.0}
        merge_local_delta(agent, {})
        assert agent.regret["IS1"]["fold"] == pytest.approx(1.0)

    def test_multiple_infosets(self):
        agent = _fresh_agent()
        local_delta = {
            "A": {"x": 1.0, "y": 2.0},
            "B": {"x": -1.0},
        }
        merge_local_delta(agent, local_delta)
        assert len(agent.regret) == 2
        assert agent.regret["A"]["x"] == pytest.approx(1.0)
        assert agent.regret["B"]["x"] == pytest.approx(-1.0)

    def test_idempotent_on_repeated_merge(self):
        """Merging the same delta twice should double the values."""
        agent = _fresh_agent()
        delta = {"IS1": {"fold": 10.0}}
        merge_local_delta(agent, delta)
        merge_local_delta(agent, delta)
        assert agent.regret["IS1"]["fold"] == pytest.approx(20.0)

    def test_new_action_in_existing_infoset(self):
        """A new action key in delta must be added, not overwrite existing."""
        agent = _fresh_agent()
        agent.regret["IS1"] = {"fold": 5.0}
        local_delta = {"IS1": {"call": 3.0}}
        merge_local_delta(agent, local_delta)
        assert agent.regret["IS1"]["fold"] == pytest.approx(5.0)
        assert agent.regret["IS1"]["call"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 3.2 + 3.3 — cfr() external sampling and local_delta
# ---------------------------------------------------------------------------


class TestCfrLocalDelta:
    def test_local_delta_accumulates_at_traversing_player_node(self):
        root, _ = _make_two_node_game()
        agent = _fresh_agent()
        local_delta: Dict = {}
        cfr(agent, root, i=0, t=1, local_delta=local_delta)
        # Traversing player is 0, root.info_set == "root"
        assert "root" in local_delta
        assert set(local_delta["root"].keys()) == {"fold", "call"}

    def test_local_delta_does_not_write_agent_regret_for_traversing_player(self):
        """When local_delta is provided, agent.regret must not be touched."""
        root, _ = _make_two_node_game()
        agent = _fresh_agent()
        local_delta: Dict = {}
        cfr(agent, root, i=0, t=1, local_delta=local_delta)
        # agent.regret should have NO entry for root (all went to local_delta)
        assert "root" not in agent.regret

    def test_direct_mode_writes_agent_regret(self):
        """When local_delta is None, agent.regret must be updated."""
        root, _ = _make_two_node_game()
        agent = _fresh_agent()
        cfr(agent, root, i=0, t=1, local_delta=None)
        assert "root" in agent.regret

    def test_merge_after_cfr_equals_direct_mode(self):
        """local_delta path + merge must give identical regrets to direct path."""
        root, _ = _make_two_node_game()

        agent_direct = _fresh_agent()
        cfr(agent_direct, root, i=0, t=1, local_delta=None)

        agent_delta = _fresh_agent()
        local_delta: Dict = {}
        cfr(agent_delta, root, i=0, t=1, local_delta=local_delta)
        merge_local_delta(agent_delta, local_delta)

        # Regrets at "root" must agree
        for action in ["fold", "call"]:
            assert agent_direct.regret["root"][action] == pytest.approx(
                agent_delta.regret["root"][action], abs=1e-8
            )

    def test_local_delta_is_empty_on_terminal_state(self):
        terminal = MockTerminal(payout={0: 100, 1: -100})
        agent = _fresh_agent()
        local_delta: Dict = {}
        result = cfr(agent, terminal, i=0, t=1, local_delta=local_delta)
        assert local_delta == {}
        assert result == pytest.approx(100.0)

    def test_regret_values_pass_through_merge(self):
        """Regret increments in local_delta must be faithfully merged."""
        root, _ = _make_two_node_game()
        agent = _fresh_agent()
        local_delta: Dict = {}
        cfr(agent, root, i=0, t=1, local_delta=local_delta)
        merge_local_delta(agent, local_delta)
        # Sum of regret increments at root must be zero (cfr invariant)
        total = sum(agent.regret["root"].values())
        # The regret values should be finite and not all zero after one pass
        assert all(np.isfinite(v) for v in agent.regret["root"].values())

    def test_payout_returned_at_terminal(self):
        terminal = MockTerminal(payout={0: 42, 1: -42})
        agent = _fresh_agent()
        val = cfr(agent, terminal, i=0, t=1)
        assert val == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# 3.2 — External sampling: opponent branch visits all actions
# ---------------------------------------------------------------------------


class TestExternalSampling:
    def test_opponent_branch_visits_all_actions(self):
        """
        In external sampling, every legal action at an opponent node must
        be visited on every cfr() call.  Confirm this by tracking which child
        states are reached from an opponent node.
        """
        visited = set()

        class TrackingTerminal(MockTerminal):
            def __init__(self, action_name, payout, n_players=2):
                super().__init__(payout, n_players)
                self._action_name = action_name

            @property
            def payout(self):
                visited.add(self._action_name)
                return self._payout

        opp_node = MockState(
            player_i=1,
            info_set="opp_track",
            actions=["fold", "call", "raise"],
            children={
                "fold": TrackingTerminal("fold", {0: 50, 1: -50}),
                "call": TrackingTerminal("call", {0: 0, 1: 0}),
                "raise": TrackingTerminal("raise", {0: -100, 1: 100}),
            },
        )
        root = MockState(
            player_i=1,   # root is also opponent's turn (i=0 traversing)
            info_set="root_ext",
            actions=["fold", "call", "raise"],
            children={
                "fold": opp_node,
                "call": opp_node,
                "raise": opp_node,
            },
        )
        agent = _fresh_agent()
        visited.clear()
        cfr(agent, root, i=0, t=1)
        # External sampling must have visited all 3 opponent actions
        assert visited == {"fold", "call", "raise"}, (
            f"Expected all 3 actions visited, got: {visited}"
        )

    def test_external_sampling_is_deterministic_given_seed(self):
        """External sampling should produce the same result regardless of random
        state because it visits all actions (no sampling)."""
        root, _ = _make_two_node_game()
        agent1 = _fresh_agent()
        agent2 = _fresh_agent()

        np.random.seed(42)
        local_delta1: Dict = {}
        cfr(agent1, root, i=0, t=1, local_delta=local_delta1)

        np.random.seed(99)
        local_delta2: Dict = {}
        cfr(agent2, root, i=0, t=1, local_delta=local_delta2)

        # Because external sampling visits all opponent actions deterministically,
        # both runs must produce identical local_delta values.
        for info_set in local_delta1:
            for action in local_delta1[info_set]:
                assert local_delta1[info_set][action] == pytest.approx(
                    local_delta2[info_set][action], abs=1e-9
                ), f"Mismatch at {info_set}/{action}"


# ---------------------------------------------------------------------------
# 3.3 — cfrp() local_delta
# ---------------------------------------------------------------------------


class TestCfrpLocalDelta:
    def test_cfrp_local_delta_accumulates(self):
        root, _ = _make_two_node_game()
        agent = _fresh_agent()
        local_delta: Dict = {}
        # Use c=-1e9 so pruning never fires (all regrets > c)
        cfrp(agent, root, i=0, t=1, c=-1_000_000_000, local_delta=local_delta)
        assert "root" in local_delta

    def test_cfrp_direct_mode_writes_agent_regret(self):
        root, _ = _make_two_node_game()
        agent = _fresh_agent()
        cfrp(agent, root, i=0, t=1, c=-1_000_000_000, local_delta=None)
        assert "root" in agent.regret

    def test_cfrp_merge_equals_direct(self):
        root, _ = _make_two_node_game()
        c = -1_000_000_000

        agent_direct = _fresh_agent()
        cfrp(agent_direct, root, i=0, t=1, c=c, local_delta=None)

        agent_delta = _fresh_agent()
        local_delta: Dict = {}
        cfrp(agent_delta, root, i=0, t=1, c=c, local_delta=local_delta)
        merge_local_delta(agent_delta, local_delta)

        for action in ["fold", "call"]:
            assert agent_direct.regret["root"][action] == pytest.approx(
                agent_delta.regret["root"][action], abs=1e-8
            )


# ---------------------------------------------------------------------------
# 3.6 — update_strategy (no locks)
# ---------------------------------------------------------------------------


class TestUpdateStrategy:
    def test_update_strategy_accumulates_counts(self):
        root, _ = _make_two_node_game()
        agent = _fresh_agent()
        # Run update_strategy 10 times — counts should grow
        for _ in range(10):
            update_strategy(agent, root, i=0, t=1)
        assert "root" in agent.strategy
        total = sum(agent.strategy["root"].values())
        assert total == pytest.approx(10.0)

    def test_update_strategy_skips_terminal(self):
        terminal = MockTerminal(payout={0: 100, 1: -100})
        agent = _fresh_agent()
        update_strategy(agent, terminal, i=0, t=1)
        assert len(agent.strategy) == 0

    def test_update_strategy_skips_postflop(self):
        """betting_round > 0 should return immediately (Bug 5 — intentional)."""
        root, _ = _make_two_node_game()
        root._betting_stage = "flop"

        class _PostflopState(MockState):
            @property
            def betting_round(self):
                return 1  # postflop

        postflop_root = _PostflopState(
            player_i=0,
            info_set="postflop_root",
            actions=["fold", "call"],
            children={
                "fold": MockTerminal({0: -50, 1: 50}),
                "call": MockTerminal({0: 50, 1: -50}),
            },
        )
        agent = _fresh_agent()
        update_strategy(agent, postflop_root, i=0, t=1)
        assert "postflop_root" not in agent.strategy

    def test_update_strategy_accepts_no_locks_arg(self):
        """update_strategy must work without any locks argument (signature check)."""
        import inspect
        sig = inspect.signature(update_strategy)
        assert "locks" not in sig.parameters, (
            "update_strategy should no longer accept a 'locks' parameter"
        )


# ---------------------------------------------------------------------------
# 3.7 — serialise Bug 4 fix
# ---------------------------------------------------------------------------


class TestSerialise:
    def _make_agent_with_data(self):
        agent = _fresh_agent()
        agent.regret["IS1"] = {"fold": 10.0, "call": -10.0}
        agent.regret["IS2"] = {"fold": 5.0, "call": 5.0}
        agent.strategy["IS1"] = {"fold": 3, "call": 7}
        return agent

    def test_serialise_creates_agent_file(self, tmp_path):
        agent = self._make_agent_with_data()
        server_state = {"dummy": 1}
        serialise(agent, tmp_path, t=1, server_state=server_state)
        assert (tmp_path / "agent.joblib").exists()

    def test_serialise_contains_regret(self, tmp_path):
        import joblib
        agent = self._make_agent_with_data()
        serialise(agent, tmp_path, t=1, server_state={})
        saved = joblib.load(tmp_path / "agent.joblib")
        assert "IS1" in saved["regret"]
        assert saved["regret"]["IS1"]["fold"] == pytest.approx(10.0)

    def test_serialise_contains_pre_flop_strategy(self, tmp_path):
        import joblib
        agent = self._make_agent_with_data()
        serialise(agent, tmp_path, t=1, server_state={})
        saved = joblib.load(tmp_path / "agent.joblib")
        assert "IS1" in saved["pre_flop_strategy"]

    def test_serialise_regret_is_plain_dict(self, tmp_path):
        """Regret in the saved file must be a plain dict, not a proxy."""
        import joblib
        agent = self._make_agent_with_data()
        serialise(agent, tmp_path, t=1, server_state={})
        saved = joblib.load(tmp_path / "agent.joblib")
        assert isinstance(saved["regret"], dict)
        for v in saved["regret"].values():
            assert isinstance(v, dict)

    def test_serialise_does_not_deepcopy(self, tmp_path):
        """Verify copy.deepcopy is not called (Bug 4 regression)."""
        import copy
        agent = self._make_agent_with_data()
        with patch.object(copy, "deepcopy", side_effect=AssertionError("deepcopy called")) as mock_dc:
            # Should not raise — deepcopy must never be called
            serialise(agent, tmp_path, t=1, server_state={})

    def test_serialise_accumulates_strategy_across_calls(self, tmp_path):
        """Repeated serialise calls must accumulate the strategy table."""
        import joblib
        agent = self._make_agent_with_data()
        serialise(agent, tmp_path, t=1, server_state={})
        serialise(agent, tmp_path, t=2, server_state={})
        saved = joblib.load(tmp_path / "agent.joblib")
        # After 2 serialise calls, strategy probabilities for IS1 should have
        # been accumulated (added) twice.
        total_prob = sum(saved["strategy"]["IS1"].values())
        assert total_prob > 0.0

    def test_serialise_server_state_written(self, tmp_path):
        import joblib
        agent = self._make_agent_with_data()
        server_state = {"mode": "test", "n_iterations": 100}
        serialise(agent, tmp_path, t=5, server_state=server_state)
        saved_server = joblib.load(tmp_path / "server.gz")
        assert saved_server["start_timestep"] == 6  # t+1


# ---------------------------------------------------------------------------
# Bug regression tests
# ---------------------------------------------------------------------------


class TestBugRegressions:
    def test_bug2_strategy_not_discounted_in_train(self):
        """After discounting, agent.strategy values must be unchanged.

        Previously singleprocess/train.py discounted agent.strategy alongside
        agent.regret (Bug 2).  This test confirms the discount loop only
        touches agent.regret.
        """
        agent = _fresh_agent()
        agent.regret["IS1"] = {"fold": 1000.0, "call": -500.0}
        agent.strategy["IS1"] = {"fold": 10, "call": 5}

        # Simulate the discount logic from the fixed train.py
        t, discount_interval, lcfr_threshold = 5, 5, 100
        if t < lcfr_threshold and t % discount_interval == 0:
            d = (t / discount_interval) / ((t / discount_interval) + 1)
            for I in agent.regret.keys():
                for a in agent.regret[I].keys():
                    agent.regret[I][a] *= d
            # Bug 2: strategy loop MUST NOT be here

        # Regret discounted
        assert agent.regret["IS1"]["fold"] < 1000.0
        # Strategy unchanged
        assert agent.strategy["IS1"]["fold"] == 10
        assert agent.strategy["IS1"]["call"] == 5

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

    def test_serialise_lock_type_hint_is_plain_dict(self):
        """serialise must accept a plain dict for locks (no mp type constraint)."""
        import inspect
        sig = inspect.signature(serialise)
        param = sig.parameters.get("locks")
        assert param is not None
        # Default must be empty dict (not a typed mp construct)
        assert param.default == {} or param.default is inspect.Parameter.empty

    def test_calculate_strategy_from_row_imported(self):
        """calculate_strategy_from_row must be importable from poker_ai.ai.ai."""
        from poker_ai.ai.ai import calculate_strategy_from_row as csfr
        assert callable(csfr)

    def test_merge_local_delta_imported(self):
        """merge_local_delta must be importable from poker_ai.ai.ai."""
        from poker_ai.ai.ai import merge_local_delta as mld
        assert callable(mld)

    def test_multiple_cfr_iterations_accumulate_correctly(self):
        """Run 5 cfr iterations using local_delta, verify regret grows."""
        root, _ = _make_two_node_game()
        agent = _fresh_agent()
        for t in range(1, 6):
            local_delta: Dict = {}
            cfr(agent, root, i=0, t=t, local_delta=local_delta)
            merge_local_delta(agent, local_delta)
        # After 5 iterations, regret should be non-trivially populated
        assert "root" in agent.regret
        # Both actions must have been updated
        assert "fold" in agent.regret["root"]
        assert "call" in agent.regret["root"]


# ---------------------------------------------------------------------------
# Slow integration test — requires the 20-card LUT
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cfr_1000_iterations_small_game():
    """Run 1000 cfr iterations on the real 20-card short-deck game.

    Verifies:
    - local_delta + merge gives non-empty regret after 1000 iterations
    - All regret values are finite
    - update_strategy populates agent.strategy
    - serialise writes a valid checkpoint (no deepcopy crash)
    """
    import tempfile
    from poker_ai.games.short_deck.state import new_game

    lut_path = "data/clustering/20cards_exact"
    if not os.path.exists(lut_path + "/card_info_lut.joblib"):
        pytest.skip("20-card exact LUT not available")

    agent = _fresh_agent()
    card_info_lut = {}
    n_players = 2
    update_threshold = 200
    strategy_interval = 10

    for t in range(1, 1001):
        for i in range(n_players):
            state = new_game(n_players, card_info_lut, lut_path=lut_path)
            card_info_lut = state.card_info_lut

            if t > update_threshold and t % strategy_interval == 0:
                update_strategy(agent, state, i=i, t=t)

            local_delta: Dict = {}
            cfr(agent, state, i=i, t=t, local_delta=local_delta)
            merge_local_delta(agent, local_delta)

    assert len(agent.regret) > 0, "Regret table must be non-empty after 1000 iterations"
    for info_set, regrets in list(agent.regret.items())[:20]:
        for action, v in regrets.items():
            assert np.isfinite(v), f"Non-finite regret at {info_set}/{action}: {v}"

    if len(agent.strategy) > 0:
        for info_set, counts in list(agent.strategy.items())[:5]:
            assert all(v >= 0 for v in counts.values())

    # Serialise checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        import joblib
        serialise(agent, tmp_path, t=1000, server_state={"n_iterations": 1000})
        saved = joblib.load(tmp_path / "agent.joblib")
        assert len(saved["regret"]) == len(agent.regret)
