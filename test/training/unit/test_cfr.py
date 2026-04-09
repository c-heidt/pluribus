"""Unit tests for ``poker_ai/ai/cfr.py``.

Covers:
- :func:`~poker_ai.ai.cfr.cfr` — external sampling, local delta accumulation.
- :func:`~poker_ai.ai.cfr.cfrp` — CFR with pruning (CFR-P).
- :func:`~poker_ai.ai.cfr.merge_local_delta` — delta flush into shared tables.

All tests use hand-crafted mock game trees so no LUT file is required.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from poker_ai.ai.action_space import ACTION_TO_IDX, MAX_ACTIONS_PER_STREET
from poker_ai.ai.cfr import cfr, cfrp, merge_local_delta
from poker_ai.ai.cfr_tables import CFRTables


# ---------------------------------------------------------------------------
# Minimal mock game tree (no LUT required)
# ---------------------------------------------------------------------------


class _FakePlayer:
    def __init__(self, is_active=True):
        self.is_active = is_active
        self.cards = []


class _FakeTable:
    community_cards = []


class MockTerminal:
    """A terminal game node."""

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
    """A non-terminal game node for hand-crafted game trees."""

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


class MockRiverState(MockState):
    """A non-terminal node on the river (betting_stage = 'river')."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._betting_stage = "river"

    @property
    def betting_round(self):
        return 3

    def get_valid_mask(self):
        from poker_ai.environment.poker_env import PokerEnv
        canonical = PokerEnv.get_canonical_actions(3)
        legal_set = set(self._actions)
        return np.array([a in legal_set for a in canonical], dtype=bool)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_two_node_game():
    """Minimal two-player, two-action game tree (pre-flop)."""
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


def _fresh_tables(tmp_path=None):
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    shm = str(tmp_path / "shm")
    os.makedirs(shm, exist_ok=True)
    return CFRTables(
        index_path=tmp_path / "lmdb",
        shm_dir=shm,
        actions_per_street=MAX_ACTIONS_PER_STREET,
    )


def _make_delta(r: int, **action_values) -> np.ndarray:
    arr = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int64)
    for action, val in action_values.items():
        arr[ACTION_TO_IDX[r][action]] = int(val)
    return arr


# ---------------------------------------------------------------------------
# merge_local_delta
# ---------------------------------------------------------------------------


class TestMergeLocalDelta:
    def test_new_infosets_added(self, tmp_tables):
        local_delta = {(0, "IS1"): _make_delta(0, fold=10, call=-10)}
        merge_local_delta(tmp_tables, local_delta)
        row = tmp_tables.regret[0].get_row_if_exists("IS1")
        assert row is not None
        assert row[ACTION_TO_IDX[0]["fold"]] == 10
        assert row[ACTION_TO_IDX[0]["call"]] == -10

    def test_existing_infoset_accumulated(self, tmp_tables):
        merge_local_delta(tmp_tables, {(0, "IS1"): _make_delta(0, fold=5, call=-5)})
        merge_local_delta(tmp_tables, {(0, "IS1"): _make_delta(0, fold=3, call=7)})
        row = tmp_tables.regret[0].get_row_if_exists("IS1")
        assert row[ACTION_TO_IDX[0]["fold"]] == 8
        assert row[ACTION_TO_IDX[0]["call"]] == 2

    def test_empty_delta_noop(self, tmp_tables):
        merge_local_delta(tmp_tables, {(0, "IS1"): _make_delta(0, fold=1)})
        before = int(tmp_tables.regret[0].get_row("IS1")[ACTION_TO_IDX[0]["fold"]])
        merge_local_delta(tmp_tables, {})
        assert int(tmp_tables.regret[0].get_row("IS1")[ACTION_TO_IDX[0]["fold"]]) == before

    def test_multiple_infosets(self, tmp_tables):
        merge_local_delta(tmp_tables, {
            (0, "A"): _make_delta(0, fold=1, call=2),
            (0, "B"): _make_delta(0, fold=-1),
        })
        assert tmp_tables.regret[0].get_row_if_exists("A") is not None
        assert tmp_tables.regret[0].get_row_if_exists("B") is not None

    def test_idempotent_on_repeated_merge(self, tmp_tables):
        delta = {(0, "IS1"): _make_delta(0, fold=10)}
        merge_local_delta(tmp_tables, delta)
        merge_local_delta(tmp_tables, delta)
        assert tmp_tables.regret[0].get_row_if_exists("IS1")[ACTION_TO_IDX[0]["fold"]] == 20


# ---------------------------------------------------------------------------
# CFR external sampling and local delta
# ---------------------------------------------------------------------------


class TestExternalSampling:
    def test_opponent_samples_single_action_per_call(self, tmp_tables):
        """Each cfr() call samples exactly one path through opponent nodes."""
        sampled_per_call = []

        class TrackingTerminal(MockTerminal):
            def __init__(self, action_name, payout):
                super().__init__(payout)
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
        root = MockState(
            player_i=1,
            info_set="root_ext",
            actions=["fold", "call", "raise:1.0"],
            children={"fold": opp_node, "call": opp_node, "raise:1.0": opp_node},
        )
        np.random.seed(0)
        for _ in range(30):
            sampled_per_call.append(set())
            cfr(tmp_tables, root, i=0, t=1)

        for run_idx, visited in enumerate(sampled_per_call):
            assert len(visited) == 1, f"Run {run_idx}: expected 1 action, got {visited}"

        all_visited = set().union(*sampled_per_call)
        assert len(all_visited) > 1

    def test_traversing_player_visits_all_actions(self, tmp_tables):
        """The traversing player (player_i == i) explores all legal actions."""
        visited = set()

        class TrackingTerminal(MockTerminal):
            def __init__(self, action_name, payout):
                super().__init__(payout)
                self._action_name = action_name

            @property
            def payout(self):
                visited.add(self._action_name)
                return self._payout

        root = MockState(
            player_i=0,
            info_set="root_traversing",
            actions=["fold", "call", "raise:1.0"],
            children={
                "fold": TrackingTerminal("fold", {0: -50, 1: 50}),
                "call": TrackingTerminal("call", {0: 0, 1: 0}),
                "raise:1.0": TrackingTerminal("raise:1.0", {0: 100, 1: -100}),
            },
        )
        cfr(tmp_tables, root, i=0, t=1)
        assert visited == {"fold", "call", "raise:1.0"}

    def test_same_seed_produces_same_result(self, tmp_path):
        root, _ = _make_two_node_game()
        tables1 = _fresh_tables(tmp_path / "t1")
        tables2 = _fresh_tables(tmp_path / "t2")

        np.random.seed(42)
        delta1: Dict = {}
        cfr(tables1, root, i=0, t=1, local_delta=delta1)

        np.random.seed(42)
        delta2: Dict = {}
        cfr(tables2, root, i=0, t=1, local_delta=delta2)

        assert delta1.keys() == delta2.keys()
        for key in delta1:
            np.testing.assert_array_equal(delta1[key], delta2[key])


class TestLocalDelta:
    def test_accumulates_at_traversing_player_node(self, tmp_tables):
        root, _ = _make_two_node_game()
        local_delta: Dict = {}
        cfr(tmp_tables, root, i=0, t=1, local_delta=local_delta)
        assert (0, "root") in local_delta
        assert isinstance(local_delta[(0, "root")], np.ndarray)

    def test_does_not_write_shared_tables(self, tmp_tables):
        root, _ = _make_two_node_game()
        local_delta: Dict = {}
        cfr(tmp_tables, root, i=0, t=1, local_delta=local_delta)
        assert tmp_tables.regret[0].get_row_if_exists("root") is None

    def test_direct_mode_writes_shared_tables(self, tmp_tables):
        root, _ = _make_two_node_game()
        cfr(tmp_tables, root, i=0, t=1, local_delta=None)
        assert tmp_tables.regret[0].get_row_if_exists("root") is not None

    def test_merge_after_cfr_equals_direct_mode(self, tmp_path):
        root, _ = _make_two_node_game()

        np.random.seed(7)
        t_direct = _fresh_tables(tmp_path / "d")
        cfr(t_direct, root, i=0, t=1, local_delta=None)

        np.random.seed(7)
        t_delta = _fresh_tables(tmp_path / "m")
        local_delta: Dict = {}
        cfr(t_delta, root, i=0, t=1, local_delta=local_delta)
        merge_local_delta(t_delta, local_delta)

        row_d = t_direct.regret[0].get_row_if_exists("root")
        row_m = t_delta.regret[0].get_row_if_exists("root")
        assert row_d is not None
        assert row_m is not None
        np.testing.assert_array_equal(row_d, row_m)

    def test_empty_on_terminal_state(self, tmp_tables):
        terminal = MockTerminal({0: 100, 1: -100})
        local_delta: Dict = {}
        result = cfr(tmp_tables, terminal, i=0, t=1, local_delta=local_delta)
        assert local_delta == {}
        assert result == pytest.approx(100.0)

    def test_regret_values_are_finite_after_merge(self, tmp_tables):
        root, _ = _make_two_node_game()
        local_delta: Dict = {}
        cfr(tmp_tables, root, i=0, t=1, local_delta=local_delta)
        merge_local_delta(tmp_tables, local_delta)
        row = tmp_tables.regret[0].get_row_if_exists("root")
        assert row is not None
        assert np.all(np.isfinite(row.astype(np.float64)))

    def test_payout_returned_at_terminal(self, tmp_tables):
        terminal = MockTerminal({0: 42, 1: -42})
        assert cfr(tmp_tables, terminal, i=0, t=1) == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# CFR-P (pruning)
# ---------------------------------------------------------------------------


class TestCfrP:
    def test_local_delta_accumulates(self, tmp_tables):
        root, _ = _make_two_node_game()
        local_delta: Dict = {}
        cfrp(tmp_tables, root, i=0, t=1, c=-1_000_000_000, local_delta=local_delta)
        assert (0, "root") in local_delta

    def test_direct_mode_writes_shared_tables(self, tmp_tables):
        root, _ = _make_two_node_game()
        cfrp(tmp_tables, root, i=0, t=1, c=-1_000_000_000, local_delta=None)
        assert tmp_tables.regret[0].get_row_if_exists("root") is not None

    def test_merge_equals_direct(self, tmp_path):
        root, _ = _make_two_node_game()
        c = -1_000_000_000

        np.random.seed(7)
        t_d = _fresh_tables(tmp_path / "d")
        cfrp(t_d, root, i=0, t=1, c=c, local_delta=None)

        np.random.seed(7)
        t_m = _fresh_tables(tmp_path / "m")
        local_delta: Dict = {}
        cfrp(t_m, root, i=0, t=1, c=c, local_delta=local_delta)
        merge_local_delta(t_m, local_delta)

        row_d = t_d.regret[0].get_row_if_exists("root")
        row_m = t_m.regret[0].get_row_if_exists("root")
        assert row_d is not None
        assert row_m is not None
        np.testing.assert_array_equal(row_d, row_m)

    def test_river_never_pruned(self, tmp_tables):
        """On a river node, all actions must be explored regardless of c."""
        visited = set()

        class TrackingTerminal(MockTerminal):
            def __init__(self, action_name, payout):
                super().__init__(payout)
                self._action_name = action_name

            @property
            def payout(self):
                visited.add(self._action_name)
                return self._payout

        river_root = MockRiverState(
            player_i=0,
            info_set="river_root",
            actions=["fold", "call"],
            children={
                "fold": TrackingTerminal("fold", {0: -50, 1: 50}),
                "call": TrackingTerminal("call", {0: 50, 1: -50}),
            },
        )
        # Write very negative regrets so c=-1 would prune both actions
        row = tmp_tables.regret[3].get_row("river_root")
        row[:] = -100_000
        cfrp(tmp_tables, river_root, i=0, t=1, c=-1, local_delta=None)
        assert "fold" in visited and "call" in visited

    def test_below_c_actions_pruned_preflop(self, tmp_tables):
        """Pre-flop actions with regret ≤ c must be skipped by CFR-P."""
        visited = set()

        class TrackingTerminal(MockTerminal):
            def __init__(self, action_name, payout):
                super().__init__(payout)
                self._action_name = action_name

            @property
            def payout(self):
                visited.add(self._action_name)
                return self._payout

        root = MockState(
            player_i=0,
            info_set="prune_root",
            actions=["fold", "call"],
            children={
                "fold": TrackingTerminal("fold", {0: -50, 1: 50}),
                "call": TrackingTerminal("call", {0: 50, 1: -50}),
            },
        )
        # Set fold regret well below c; call regret well above c
        row = tmp_tables.regret[0].get_row("prune_root")
        row[ACTION_TO_IDX[0]["fold"]] = -10_000
        row[ACTION_TO_IDX[0]["call"]] = 10_000
        # c = 0: fold (regret -10000 ≤ 0) should be pruned
        cfrp(tmp_tables, root, i=0, t=1, c=0, local_delta=None)
        assert "fold" not in visited
        assert "call" in visited
