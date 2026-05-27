"""Tests for :mod:`poker_ai.search.policy`."""

import numpy as np
import pytest

from environment.action_space import (
    ACTION_TO_IDX,
    CANONICAL_ACTIONS,
    MAX_ACTIONS_PER_STREET,
)
from poker_ai.blueprint.tree_utils import calculate_strategy_from_row
from poker_ai.search.policy import BlueprintPolicy, Policy


# ---------------------------------------------------------------------------
# Bias mask
# ---------------------------------------------------------------------------


class TestBiasMask:

    actions = ["fold", "call", "raise:0.5", "raise:1.0", "all_in"]

    def test_none_selects_nothing(self):
        mask = Policy._bias_mask(self.actions, "none")
        assert not mask.any()

    def test_fold_selects_only_fold(self):
        mask = Policy._bias_mask(self.actions, "fold")
        assert mask.tolist() == [True, False, False, False, False]

    def test_call_selects_call(self):
        mask = Policy._bias_mask(self.actions, "call")
        assert mask.tolist() == [False, True, False, False, False]

    def test_call_also_selects_check(self):
        actions = ["fold", "check", "raise:1.0"]
        mask = Policy._bias_mask(actions, "call")
        assert mask.tolist() == [False, True, False]

    def test_raise_selects_raises_and_all_in(self):
        mask = Policy._bias_mask(self.actions, "raise")
        assert mask.tolist() == [False, False, True, True, True]


# ---------------------------------------------------------------------------
# Regret matching + bias
# ---------------------------------------------------------------------------


class TestRegretMatchWithBias:

    def test_no_bias_matches_blueprint(self):
        regret = np.array([0, 10, 20, 30], dtype=np.int32)
        valid = np.array([True, True, True, False])
        bias_mask = np.array([False, True, False, False])
        sigma = Policy._regret_match_with_bias(regret, valid, bias_mask, 0.0)
        expected = calculate_strategy_from_row(regret, valid)
        np.testing.assert_array_equal(sigma, expected)

    def test_bias_shifts_mass(self):
        regret = np.array([10, 10, 10, 10], dtype=np.int32)
        valid = np.ones(4, dtype=bool)
        bias_mask = np.array([True, False, False, False])
        unbiased = Policy._regret_match_with_bias(regret, valid, bias_mask, 0.0)
        biased = Policy._regret_match_with_bias(regret, valid, bias_mask, 50.0)
        # Biased index gains; others lose.
        assert biased[0] > unbiased[0]
        assert (biased[1:] < unbiased[1:]).all()
        np.testing.assert_allclose(biased.sum(), 1.0, atol=1e-6)

    def test_uniform_fallback_no_bias(self):
        regret = np.zeros(4, dtype=np.int32)
        valid = np.array([True, True, False, False])
        bias_mask = np.zeros(4, dtype=bool)
        sigma = Policy._regret_match_with_bias(regret, valid, bias_mask, 0.0)
        np.testing.assert_allclose(sigma, [0.5, 0.5, 0.0, 0.0])

    def test_bias_ignores_illegal_actions(self):
        regret = np.zeros(4, dtype=np.int32)
        valid = np.array([True, True, False, False])
        # Bias an illegal action — it must not gain mass.
        bias_mask = np.array([False, False, True, False])
        sigma = Policy._regret_match_with_bias(regret, valid, bias_mask, 100.0)
        assert sigma[2] == 0.0
        np.testing.assert_allclose(sigma.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# BlueprintPolicy
# ---------------------------------------------------------------------------


class _FakeTable:
    """Stand-in for a ChunkedTable, returning a fixed regret row or None."""

    def __init__(self, row):
        self._row = row

    def get_row_if_exists(self, info_set):
        return self._row


class _FakeTables:
    """Stand-in for CFRTables; only exposes ``regret[r]``."""

    def __init__(self, rows_by_round):
        self.regret = {r: _FakeTable(row) for r, row in rows_by_round.items()}


class _FakeEnv:
    """Minimal duck-typed PokerEnv for policy.strategy() tests."""

    def __init__(self, betting_round, legal_actions, info_set="X"):
        self.betting_round = betting_round
        self.info_set = info_set
        self._legal = legal_actions

    @property
    def legal_actions(self):
        return self._legal

    def get_valid_mask(self):
        canon = CANONICAL_ACTIONS[self.betting_round]
        legal_set = {a for a in self._legal if a is not None}
        return np.array([a in legal_set for a in canon], dtype=bool)


def _legal_for(r):
    """Pick a representative legal subset present in the canonical set."""
    canon = CANONICAL_ACTIONS[r]
    # fold/call/all_in always exist + the first raise option.
    raise_action = next((a for a in canon if a.startswith("raise:")), None)
    legal = ["fold", "call"]
    if raise_action is not None:
        legal.append(raise_action)
    legal.append("all_in")
    return legal


class TestBlueprintPolicy:

    @pytest.mark.parametrize("r", [0, 1, 2, 3])
    def test_uniform_when_info_set_missing(self, r):
        legal = _legal_for(r)
        env = _FakeEnv(r, legal)
        tables = _FakeTables({r: None})
        policy = BlueprintPolicy(tables)
        sigma = policy.strategy(env)
        assert sigma.shape == (len(legal),)
        np.testing.assert_allclose(sigma, np.full(len(legal), 1.0 / len(legal)))

    def test_alignment_matches_legal_actions(self):
        r = 1
        legal = _legal_for(r)
        env = _FakeEnv(r, legal)
        # Plant a row that's only positive on the first legal action.
        row = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
        row[ACTION_TO_IDX[r][legal[0]]] = 100
        tables = _FakeTables({r: row})
        sigma = BlueprintPolicy(tables).strategy(env)
        assert sigma.shape == (len(legal),)
        # All mass on the first legal action.
        assert sigma[0] == pytest.approx(1.0)
        np.testing.assert_allclose(sigma[1:], 0.0)

    def test_sums_to_one(self):
        r = 2
        legal = _legal_for(r)
        env = _FakeEnv(r, legal)
        row = np.array(
            [(i * 7) % 13 for i in range(MAX_ACTIONS_PER_STREET[r])],
            dtype=np.int32,
        )
        tables = _FakeTables({r: row})
        sigma = BlueprintPolicy(tables).strategy(env)
        np.testing.assert_allclose(sigma.sum(), 1.0, atol=1e-6)

    def test_bias_shifts_mass_to_fold(self):
        r = 0
        legal = _legal_for(r)
        env = _FakeEnv(r, legal)
        # Uniform regrets across all canonical actions.
        row = np.full(MAX_ACTIONS_PER_STREET[r], 10, dtype=np.int32)
        tables = _FakeTables({r: row})
        unbiased = BlueprintPolicy(tables, bias_magnitude=50.0).strategy(env, bias="none")
        biased = BlueprintPolicy(tables, bias_magnitude=50.0).strategy(env, bias="fold")
        fold_i = legal.index("fold")
        assert biased[fold_i] > unbiased[fold_i]
        np.testing.assert_allclose(biased.sum(), 1.0, atol=1e-6)

    def test_bias_magnitude_zero_ignores_bias_arg(self):
        r = 0
        legal = _legal_for(r)
        env = _FakeEnv(r, legal)
        row = np.full(MAX_ACTIONS_PER_STREET[r], 10, dtype=np.int32)
        tables = _FakeTables({r: row})
        policy = BlueprintPolicy(tables, bias_magnitude=0.0)
        np.testing.assert_array_equal(
            policy.strategy(env, bias="none"),
            policy.strategy(env, bias="fold"),
        )
