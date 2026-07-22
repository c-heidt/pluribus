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
from test.training.unit.test_cfr import _MakeUndoMock


# ---------------------------------------------------------------------------
# Minimal mock game states (no LUT required)
# ---------------------------------------------------------------------------


class _FakePlayer:
    def __init__(self, is_active=True):
        self.is_active = is_active
        self.cards = []


class _FakeTable:
    community_cards = []


class MockTerminal(_MakeUndoMock):
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


class MockState(_MakeUndoMock):
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
        from environment.poker_env import PokerEnv
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

    def test_stops_at_end_of_preflop(self, tmp_tables):
        """Post-flop states must NOT be traversed (pre-flop-only guard)."""

        class PostflopState(MockState):
            @property
            def betting_round(self):
                return 1

            def get_valid_mask(self):
                from environment.poker_env import PokerEnv
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
        # The pre-flop-only guard returns immediately: nothing recorded on any
        # street, and in particular not on the flop.
        assert tmp_tables.strategy[1].get_row_if_exists("postflop_root") is None
        for r in range(4):
            assert tmp_tables.strategy[r].get_row_if_exists("postflop_root") is None

    def test_opponent_nodes_branch_all_actions(self, tmp_tables):
        """One pass must reach every traverser node behind an opponent node.

        With full pre-flop branching an opponent decision fans out into all of
        its actions, so a single ``update_strategy`` call deposits a count in
        *both* traverser rows sitting behind the opponent's two actions — the
        old single-sampled walk would have reached only one.
        """
        opp_root = MockState(
            player_i=1,  # opponent acts first
            info_set="opp_root",
            actions=["a", "b"],
            children={
                "a": MockState(
                    player_i=0, info_set="mine_a", actions=["call"],
                    children={"call": MockTerminal({0: 1, 1: -1})},
                ),
                "b": MockState(
                    player_i=0, info_set="mine_b", actions=["call"],
                    children={"call": MockTerminal({0: -1, 1: 1})},
                ),
            },
        )
        update_strategy(tmp_tables, opp_root, i=0)
        row_a = tmp_tables.strategy[0].get_row_if_exists("mine_a")
        row_b = tmp_tables.strategy[0].get_row_if_exists("mine_b")
        assert row_a is not None and int(row_a.sum()) == 1
        assert row_b is not None and int(row_b.sum()) == 1
        # The opponent node itself never writes.
        assert tmp_tables.strategy[0].get_row_if_exists("opp_root") is None

    def test_no_strategy_lookup_at_opponent_nodes(self, tmp_tables, monkeypatch):
        """``get_node_strategy`` is resolved only at traverser nodes."""
        import poker_ai.blueprint.strategy as strat_mod

        seen = []
        real = strat_mod.get_node_strategy

        def _recorder(tables, state):
            seen.append(state.info_set)
            return real(tables, state)

        monkeypatch.setattr(strat_mod, "get_node_strategy", _recorder)

        opp_root = MockState(
            player_i=1, info_set="opp_root", actions=["a"],
            children={
                "a": MockState(
                    player_i=0, info_set="mine", actions=["call"],
                    children={"call": MockTerminal({0: 0, 1: 0})},
                ),
            },
        )
        update_strategy(tmp_tables, opp_root, i=0)
        # Only the traverser node "mine" is resolved; the opponent root is not.
        assert seen == ["mine"]

    def test_local_delta_keys_are_preflop_only(self, tmp_tables):
        """Accumulator keys never carry a post-flop betting round.

        The traverser acts pre-flop, then reaches a post-flop traverser node
        whose row must not be recorded (the guard returns first).
        """

        class PostflopMine(MockState):
            @property
            def betting_round(self):
                return 1

        preflop_root = MockState(
            player_i=0, info_set="pf", actions=["call"],
            children={
                "call": PostflopMine(
                    player_i=0, info_set="postflop_mine", actions=["call"],
                    children={"call": MockTerminal({0: 0, 1: 0})},
                ),
            },
        )
        ld = {}
        update_strategy(tmp_tables, preflop_root, i=0, local_delta=ld)
        assert all(key[0] == 0 for key in ld), ld
        assert (0, "pf") in ld
        assert (1, "postflop_mine") not in ld

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
        # Default (no core, no accumulator) → the legacy direct-write path.
        mock.assert_called_once_with(tables, state, 2, local_delta=None)

    def test_accumulates_into_local_delta_when_supplied(self, monkeypatch):
        """With a local_delta, strategy_step forwards it (no direct-write)."""
        mock = MagicMock()
        monkeypatch.setattr(training, "update_strategy", mock)
        tables, state, ld = MagicMock(), MagicMock(), {}
        strategy_step(tables, state, i=1, local_delta=ld)
        mock.assert_called_once_with(tables, state, 1, local_delta=ld)

    def test_dispatches_to_core_when_supplied(self):
        """With a core driver, strategy_step routes to core.run_strategy."""
        tables, state, ld = MagicMock(), MagicMock(), {}
        core = MagicMock()
        strategy_step(tables, state, i=3, local_delta=ld, core=core)
        core.run_strategy.assert_called_once_with(state, 3, ld)
