"""Unit tests for ``poker_ai/ai/strategy.py``.

Covers:
- :func:`~poker_ai.blueprint.strategy.update_strategy` — visit-count accumulation,
  terminal skipping, postflop traversal.
- :func:`~poker_ai.blueprint.training.strategy_step` delegation.

All tests use hand-crafted mock states so no LUT file is required.
"""

import inspect
from unittest.mock import MagicMock

import numpy as np
import pytest

from poker_ai.blueprint import training
from poker_ai.blueprint.strategy import update_strategy
from poker_ai.blueprint.training import strategy_step


# ---------------------------------------------------------------------------
# Minimal mock game states (no LUT required)
# ---------------------------------------------------------------------------


class _FakePlayer:
    def __init__(self, is_active=True):
        self.is_active = is_active
        self.cards = []


class _FakeTable:
    community_cards = []


class MockTerminal:
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


class MockState:
    def __init__(self, player_i, info_set, actions, children, n_players=2):
        self._player_i = player_i
        self._info_set = info_set
        self._actions = list(actions)
        self._children = children
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
        from poker_ai.environment.poker_env import PokerEnv
        canonical = PokerEnv.get_canonical_actions(self.betting_round)
        legal_set = set(self._actions)
        return np.array([a in legal_set for a in canonical], dtype=bool)

    def apply_action(self, action):
        return self._children[action]


def _make_two_node_game():
    opp_node = MockState(
        player_i=1,
        info_set="opp",
        actions=["fold", "call"],
        children={
            "fold": MockTerminal({0: 50, 1: -50}),
            "call": MockTerminal({0: 0, 1: 0}),
        },
    )
    root = MockState(
        player_i=0,
        info_set="root",
        actions=["fold", "call"],
        children={
            "fold": MockTerminal({0: -50, 1: 50}),
            "call": opp_node,
        },
    )
    return root, opp_node


# ---------------------------------------------------------------------------
# update_strategy
# ---------------------------------------------------------------------------


class TestUpdateStrategy:
    def test_accumulates_counts(self, tmp_tables):
        root, _ = _make_two_node_game()
        for _ in range(10):
            update_strategy(tmp_tables, root, i=0)
        row = tmp_tables.strategy[0].get_row_if_exists("root")
        assert row is not None
        assert int(row.sum()) == 10

    def test_skips_terminal(self, tmp_tables):
        terminal = MockTerminal({0: 100, 1: -100})
        update_strategy(tmp_tables, terminal, i=0)
        for r in range(4):
            assert tmp_tables.strategy[r].get_row_if_exists("terminal") is None

    def test_traverses_postflop(self, tmp_tables):
        """Postflop states must be traversed (no betting_round > 0 guard)."""

        class PostflopState(MockState):
            @property
            def betting_round(self):
                return 1

            def get_valid_mask(self):
                from poker_ai.environment.poker_env import PokerEnv
                canonical = PokerEnv.get_canonical_actions(1)
                legal_set = set(self._actions)
                return np.array([a in legal_set for a in canonical], dtype=bool)

        postflop_root = PostflopState(
            player_i=0,
            info_set="postflop_root",
            actions=["fold", "call"],
            children={
                "fold": MockTerminal({0: -50, 1: 50}),
                "call": MockTerminal({0: 50, 1: -50}),
            },
        )
        update_strategy(tmp_tables, postflop_root, i=0)
        assert tmp_tables.strategy[1].get_row_if_exists("postflop_root") is not None

    def test_accepts_no_locks_arg(self):
        assert "locks" not in inspect.signature(update_strategy).parameters


# ---------------------------------------------------------------------------
# strategy_step
# ---------------------------------------------------------------------------


class TestStrategyStep:
    def test_delegates_to_update_strategy(self, monkeypatch):
        mock = MagicMock()
        monkeypatch.setattr(training, "update_strategy", mock)
        tables, state = MagicMock(), MagicMock()
        strategy_step(tables, state, i=2)
        mock.assert_called_once_with(tables, state, 2)
