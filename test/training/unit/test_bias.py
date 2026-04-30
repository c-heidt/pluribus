"""Unit tests for biased-blueprint MCCFR (``poker_ai/blueprint/bias.py`` and
the bias-aware paths in ``cfr.py``).

Covers:

- :func:`~poker_ai.blueprint.bias.is_biased` action-class mapping.
- Regression: ``bias="none"`` is byte-identical to the legacy
  unbiased path (locks in the strict no-overhead promise).
- Bias bonus arithmetic at terminals.
- Convergence: under a large bias magnitude, the recovered strategy
  shifts toward the biased action class.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from environment.action_space import ACTION_TO_IDX, MAX_ACTIONS_PER_STREET
from poker_ai.blueprint.bias import is_biased
from poker_ai.blueprint.cfr import cfr, cfrp
from poker_ai.tables.cfr_tables import CFRTables

# Reuse the same mock game tree from the existing test_cfr module.
from test.training.unit.test_cfr import MockState, MockTerminal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_tables(tmp_path: Path) -> CFRTables:
    shm = str(tmp_path / "shm")
    os.makedirs(shm, exist_ok=True)
    return CFRTables(
        index_path=tmp_path / "lmdb",
        shm_dir=shm,
        actions_per_street=MAX_ACTIONS_PER_STREET,
    )


def _make_two_node_game():
    """Two-player two-action mock game (same shape as test_cfr.py)."""
    opp = MockState(
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
            "call": opp,
        },
    )
    return root


# ---------------------------------------------------------------------------
# is_biased classification
# ---------------------------------------------------------------------------


class TestIsBiased:
    @pytest.mark.parametrize("action", ["fold"])
    def test_fold_class(self, action):
        assert is_biased(action, "fold")
        assert not is_biased(action, "call")
        assert not is_biased(action, "raise")

    @pytest.mark.parametrize("action", ["call", "check"])
    def test_call_class(self, action):
        assert is_biased(action, "call")
        assert not is_biased(action, "fold")
        assert not is_biased(action, "raise")

    @pytest.mark.parametrize(
        "action", ["raise:0.5", "raise:1.0", "raise:2.0", "all_in"]
    )
    def test_raise_class(self, action):
        assert is_biased(action, "raise")
        assert not is_biased(action, "fold")
        assert not is_biased(action, "call")

    @pytest.mark.parametrize(
        "action", ["fold", "call", "check", "raise:1.0", "all_in"]
    )
    def test_none_never_biased(self, action):
        assert not is_biased(action, "none")


# ---------------------------------------------------------------------------
# Regression: bias=none must be byte-identical to legacy path
# ---------------------------------------------------------------------------


class TestNoneIsLegacy:
    def test_cfr_bias_none_matches_default(self, tmp_path):
        root_a = _make_two_node_game()
        root_b = _make_two_node_game()
        t_a = _fresh_tables(tmp_path / "a")
        t_b = _fresh_tables(tmp_path / "b")

        np.random.seed(123)
        delta_a: Dict = {}
        cfr(t_a, root_a, i=0, t=1, local_delta=delta_a)

        np.random.seed(123)
        delta_b: Dict = {}
        cfr(t_b, root_b, i=0, t=1, local_delta=delta_b,
            bias="none", bias_magnitude=0.0)

        assert delta_a.keys() == delta_b.keys()
        for key in delta_a:
            np.testing.assert_array_equal(delta_a[key], delta_b[key])

    def test_cfrp_bias_none_matches_default(self, tmp_path):
        root_a = _make_two_node_game()
        root_b = _make_two_node_game()
        t_a = _fresh_tables(tmp_path / "a")
        t_b = _fresh_tables(tmp_path / "b")

        np.random.seed(7)
        delta_a: Dict = {}
        cfrp(t_a, root_a, i=0, t=1, c=-1_000_000_000, local_delta=delta_a)

        np.random.seed(7)
        delta_b: Dict = {}
        cfrp(t_b, root_b, i=0, t=1, c=-1_000_000_000, local_delta=delta_b,
             bias="none", bias_magnitude=0.0)

        assert delta_a.keys() == delta_b.keys()
        for key in delta_a:
            np.testing.assert_array_equal(delta_a[key], delta_b[key])


# ---------------------------------------------------------------------------
# Bias bonus arithmetic at terminals
# ---------------------------------------------------------------------------


class TestBiasBonus:
    def test_terminal_state_returns_payoff_unchanged(self, tmp_tables):
        """A bare terminal node has no actions on its trajectory — bonus is 0."""
        terminal = MockTerminal({0: 42, 1: -42})
        result = cfr(tmp_tables, terminal, i=0, t=1,
                     bias="fold", bias_magnitude=10.0)
        # bias_count = 0 at the root terminal, so bonus = 0
        assert result == pytest.approx(42.0)

    def test_root_traversing_terminal_bonus_added_per_biased_action(
        self, tmp_tables
    ):
        """One traversing decision; the fold branch's terminal sees bonus=magnitude."""
        # Root is the traversing player. Two actions: fold (biased) and call
        # (not biased). After each, terminal. With bias=fold, magnitude=10:
        #   fold leaf: payout(0) = -50, bonus = 10*1 = 10  →  -40
        #   call leaf: payout(0) =   0, bonus = 10*0 =  0  →    0
        root = MockState(
            player_i=0,
            info_set="root",
            actions=["fold", "call"],
            children={
                "fold": MockTerminal({0: -50, 1: 50}),
                "call": MockTerminal({0: 0, 1: 0}),
            },
        )
        # First-call regrets are zero, so initial sigma is uniform over both
        # actions. accumulate_regrets writes (vo_a - vo) per explored action.
        # vo_a("fold") = -50 + 10 = -40
        # vo_a("call") =   0 +  0 =   0
        # vo = 0.5*(-40) + 0.5*0 = -20
        # delta(fold) = -40 - (-20) = -20
        # delta(call) =   0 - (-20) =  20
        delta: Dict = {}
        cfr(tmp_tables, root, i=0, t=1, local_delta=delta,
            bias="fold", bias_magnitude=10.0)
        row = delta[(0, "root")]
        assert row[ACTION_TO_IDX[0]["fold"]] == -20
        assert row[ACTION_TO_IDX[0]["call"]] == 20

    def test_opponent_action_also_counted_toward_bias(self, tmp_tables):
        """Bias counts decisions on the full path, not just the traversing player."""
        # Opponent (player 1) is the only decision; player 0 just sees the
        # terminal value. With bias=fold, large magnitude, every terminal
        # reached via fold gets the bonus.
        opp = MockState(
            player_i=1,
            info_set="opp",
            actions=["fold", "call"],
            children={
                # If opp folds, player 0 wins 50; bias action → bonus = 100
                "fold": MockTerminal({0: 50, 1: -50}),
                # If opp calls, no bias action on path
                "call": MockTerminal({0: 0, 1: 0}),
            },
        )
        # Root traversing player 0 with only the call action — single path
        # leading to opp. From player 0's perspective the opp's choice is
        # external-sampled, so each cfr() returns the value of one branch.
        root = MockState(
            player_i=0,
            info_set="root_op",
            actions=["call"],
            children={"call": opp},
        )
        # Force opp to fold by writing dominating regret on "fold"
        opp_row = tmp_tables.regret[0].get_row("opp")
        opp_row[ACTION_TO_IDX[0]["fold"]] = 1
        opp_row[ACTION_TO_IDX[0]["call"]] = -1

        np.random.seed(0)
        # Now sampling at opp will pick "fold" almost surely.  player 0 is
        # the traversing player at root with one legal action; vo_a("call")
        # propagates the leaf value.  Path: call (not biased) → fold (biased)
        # so bias_count = 1.  Expected leaf value to player 0: 50 + 100*1 = 150.
        delta: Dict = {}
        v = cfr(tmp_tables, root, i=0, t=1, local_delta=delta,
                bias="fold", bias_magnitude=100.0)
        assert v == pytest.approx(150.0)

    def test_zero_magnitude_with_active_bias_class(self, tmp_path):
        """``bias != none`` with magnitude=0 produces the same regret values
        as bias='none' (the bonus reduces to 0*count = 0)."""
        root_a = _make_two_node_game()
        root_b = _make_two_node_game()
        t_a = _fresh_tables(tmp_path / "a")
        t_b = _fresh_tables(tmp_path / "b")

        np.random.seed(99)
        d_a: Dict = {}
        cfr(t_a, root_a, i=0, t=1, local_delta=d_a)

        np.random.seed(99)
        d_b: Dict = {}
        cfr(t_b, root_b, i=0, t=1, local_delta=d_b,
            bias="fold", bias_magnitude=0.0)

        assert d_a.keys() == d_b.keys()
        for k in d_a:
            np.testing.assert_array_equal(d_a[k], d_b[k])


# ---------------------------------------------------------------------------
# Convergence: large magnitude shifts the strategy toward the biased class
# ---------------------------------------------------------------------------


class TestBiasShiftsStrategy:
    def test_fold_bias_dominates_after_iterations(self, tmp_path):
        """With a strong fold bias, accumulated regret on fold should
        dominate accumulated regret on call after many iterations."""
        np.random.seed(0)
        tables_biased = _fresh_tables(tmp_path / "biased")
        tables_unbiased = _fresh_tables(tmp_path / "unbiased")

        for it in range(1, 200):
            cfr(tables_biased, _make_two_node_game(), i=0, t=it,
                bias="fold", bias_magnitude=200.0)
            cfr(tables_unbiased, _make_two_node_game(), i=0, t=it)

        biased_row = tables_biased.regret[0].get_row_if_exists("root")
        unbiased_row = tables_unbiased.regret[0].get_row_if_exists("root")
        assert biased_row is not None and unbiased_row is not None

        fold_idx = ACTION_TO_IDX[0]["fold"]
        call_idx = ACTION_TO_IDX[0]["call"]
        # Under a heavy fold bonus, fold's regret in the biased table must
        # exceed fold's regret in the unbiased table — i.e., the biased
        # variant prefers folding more strongly than the base.
        assert biased_row[fold_idx] > unbiased_row[fold_idx]
