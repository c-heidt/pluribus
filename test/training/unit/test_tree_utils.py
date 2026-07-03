"""Unit tests for ``poker_ai/ai/tree_utils.py``.

Covers:
- :func:`~poker_ai.blueprint.tree_utils.calculate_strategy_from_row` — pure numpy regret matching.
- :func:`~poker_ai.blueprint.tree_utils.is_terminal` — terminal detection and payout return.
- :func:`~poker_ai.blueprint.tree_utils.get_legal_actions` — legal action filtering.
- :func:`~poker_ai.blueprint.tree_utils.get_node_strategy` — regret-matching strategy lookup.

All tests use hand-crafted mock states so no LUT file is required.
"""

import numpy as np
import pytest

from poker_ai.blueprint.tree_utils import (
    calculate_strategy_from_row,
    get_legal_actions,
    get_node_strategy,
    is_terminal,
)


# ---------------------------------------------------------------------------
# Helpers — minimal mock game states (no LUT required)
# ---------------------------------------------------------------------------


class _FakePlayer:
    def __init__(self, is_active=True):
        self.is_active = is_active
        self.cards = []


class _FakeTable:
    community_cards = []


class _MockTerminal:
    def __init__(self, payout, n_players=2):
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
        raise RuntimeError("terminal")


class _MockState:
    def __init__(self, player_i, info_set, actions, n_players=2):
        self._player_i = player_i
        self._info_set = info_set
        self._actions = list(actions)
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
    def legal_actions(self):
        return list(self._actions)

    @property
    def betting_round(self):
        return 0

    def get_valid_mask(self):
        from environment.poker_env import PokerEnv
        canonical = PokerEnv.get_canonical_actions(self.betting_round)
        legal_set = set(self._actions)
        return np.array([a in legal_set for a in canonical], dtype=bool)


class _MockInactivePlayer:
    """Game where player 0 is inactive (already folded)."""
    def __init__(self):
        self.is_terminal = False
        self.players = [_FakePlayer(is_active=False), _FakePlayer(is_active=True)]
        self._payout = {0: -50, 1: 50}
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
        return "inactive"

    @property
    def legal_actions(self):
        return ["fold", "call"]


# ---------------------------------------------------------------------------
# calculate_strategy_from_row
# ---------------------------------------------------------------------------


class TestCalculateStrategyFromRow:
    def test_uniform_on_zeros(self):
        row = np.zeros(4, dtype=np.int32)
        result = calculate_strategy_from_row(row)
        np.testing.assert_allclose(result, [0.25, 0.25, 0.25, 0.25], atol=1e-6)
        assert result.dtype == np.float32

    def test_uniform_on_all_negative(self):
        row = np.array([-100, -200, -50], dtype=np.int32)
        n = len(row)
        np.testing.assert_allclose(
            calculate_strategy_from_row(row), [1 / n] * n, atol=1e-6
        )

    def test_pure_strategy_on_one_positive(self):
        row = np.array([0, 1000, 0], dtype=np.int32)
        np.testing.assert_allclose(
            calculate_strategy_from_row(row), [0.0, 1.0, 0.0], atol=1e-6
        )

    def test_proportional_to_positive_regrets(self):
        row = np.array([100, 300, 0, -50], dtype=np.int32)
        expected = np.array([100, 300, 0, 0], dtype=np.float32) / 400.0
        np.testing.assert_allclose(calculate_strategy_from_row(row), expected, atol=1e-5)

    def test_sums_to_one(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            row = rng.integers(-500, 500, size=5).astype(np.int32)
            assert abs(float(calculate_strategy_from_row(row).sum()) - 1.0) < 1e-5

    def test_output_dtype_is_float32(self):
        assert calculate_strategy_from_row(np.array([1, 2, 3], dtype=np.int32)).dtype == np.float32

    def test_single_action(self):
        np.testing.assert_allclose(
            calculate_strategy_from_row(np.array([500], dtype=np.int32)), [1.0], atol=1e-6
        )

    def test_two_equal_regrets(self):
        np.testing.assert_allclose(
            calculate_strategy_from_row(np.array([200, 200], dtype=np.int32)),
            [0.5, 0.5],
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# is_terminal
# ---------------------------------------------------------------------------


class TestIsTerminal:
    def test_returns_none_on_active_nonterminal(self):
        state = _MockState(player_i=0, info_set="active", actions=["fold", "call"])
        assert is_terminal(state, i=0) is None

    def test_returns_payout_on_terminal_state(self):
        state = _MockTerminal(payout={0: 100, 1: -100})
        result = is_terminal(state, i=0)
        assert result == pytest.approx(100.0)

    def test_returns_payout_when_player_inactive(self):
        state = _MockInactivePlayer()
        result = is_terminal(state, i=0)
        assert result == pytest.approx(-50.0)


# ---------------------------------------------------------------------------
# get_legal_actions
# ---------------------------------------------------------------------------


class TestGetLegalActions:
    def test_returns_nonempty_list(self):
        state = _MockState(player_i=0, info_set="s", actions=["fold", "call"])
        actions = get_legal_actions(state)
        assert len(actions) > 0

    def test_no_none_in_result(self):
        state = _MockState(player_i=0, info_set="s", actions=["fold", None, "call"])
        actions = get_legal_actions(state)
        assert None not in actions


# ---------------------------------------------------------------------------
# get_node_strategy
# ---------------------------------------------------------------------------


class TestGetNodeStrategy:
    def test_uniform_on_unseen_infoset(self, tmp_tables):
        state = _MockState(player_i=0, info_set="unseen_IS", actions=["fold", "call"])
        sigma, r, a_to_i, regret_row, info_set = get_node_strategy(tmp_tables, state)
        assert info_set == "unseen_IS"
        assert abs(sigma.sum() - 1.0) < 1e-5
        n_legal = len(state.legal_actions)
        for action in state.legal_actions:
            idx = a_to_i[action]
            assert sigma[idx] == pytest.approx(1.0 / n_legal, abs=1e-5)

    def test_strategy_reflects_positive_regrets(self, tmp_tables):
        state = _MockState(player_i=0, info_set="biased_IS", actions=["fold", "call"])
        # Write large positive regret for "call"
        row = tmp_tables.regret[0].get_row("biased_IS")
        from environment.action_space import ACTION_TO_IDX
        row[ACTION_TO_IDX[0]["call"]] = 10_000
        row[ACTION_TO_IDX[0]["fold"]] = 0
        sigma, _, a_to_i, _, _ = get_node_strategy(tmp_tables, state)
        assert sigma[a_to_i["call"]] > sigma[a_to_i["fold"]]
