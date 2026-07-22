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
# Multiplicative bias reweighting
# ---------------------------------------------------------------------------


class TestReweightBias:

    def test_multiplier_one_returns_unchanged(self):
        sigma = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        bias_mask = np.array([True, False, False, False])
        out = Policy._reweight_bias(sigma, bias_mask, 1.0)
        np.testing.assert_array_equal(out, sigma)

    def test_empty_mask_returns_unchanged(self):
        sigma = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        bias_mask = np.zeros(4, dtype=bool)
        out = Policy._reweight_bias(sigma, bias_mask, 5.0)
        np.testing.assert_array_equal(out, sigma)

    def test_golden_x5_reweight(self):
        # σ uniform → biased class ×5 → pre-norm [1.25, .25, .25, .25],
        # sum 2.0 → [.625, .125, .125, .125].
        sigma = np.full(4, 0.25, dtype=np.float32)
        bias_mask = np.array([True, False, False, False])
        out = Policy._reweight_bias(sigma, bias_mask, 5.0)
        np.testing.assert_allclose(out, [0.625, 0.125, 0.125, 0.125], atol=1e-6)

    def test_bias_shifts_mass(self):
        sigma = np.full(4, 0.25, dtype=np.float32)
        bias_mask = np.array([True, False, False, False])
        out = Policy._reweight_bias(sigma, bias_mask, 5.0)
        # Biased index gains; others lose; total stays 1.
        assert out[0] > sigma[0]
        assert (out[1:] < sigma[1:]).all()
        np.testing.assert_allclose(out.sum(), 1.0, atol=1e-6)

    def test_zero_mass_biased_action_stays_zero(self):
        # An action already at zero probability (e.g. illegal — zeroed by
        # regret matching) gains nothing from ×5.
        sigma = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        bias_mask = np.array([False, False, True, False])
        out = Policy._reweight_bias(sigma, bias_mask, 100.0)
        assert out[2] == 0.0
        np.testing.assert_allclose(out.sum(), 1.0, atol=1e-6)


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
    """Stand-in for CFRTables exposing ``regret[r]`` and ``strategy[r]``.

    ``strategy_rows_by_round`` defaults to no strategy rows at all, so
    tests built around regret matching exercise the policy's fallback
    path unchanged.
    """

    def __init__(self, rows_by_round, strategy_rows_by_round=None):
        self.regret = {r: _FakeTable(row) for r, row in rows_by_round.items()}
        strategy_rows_by_round = strategy_rows_by_round or {}
        self.strategy = {
            r: _FakeTable(strategy_rows_by_round.get(r))
            for r in rows_by_round
        }


def _state(betting_round, legal_actions, info_set="X", player_i=0):
    """Build a :class:`PolicyState` from the legacy ``_FakeEnv`` inputs."""
    from environment.poker_env import PolicyState
    canon = CANONICAL_ACTIONS[betting_round]
    legal_set = {a for a in legal_actions if a is not None}
    valid_mask = np.array([a in legal_set for a in canon], dtype=bool)
    legal = tuple(a for a in legal_actions if a is not None)
    return PolicyState(
        player_i=player_i,
        betting_round=betting_round,
        info_set=info_set,
        valid_mask=valid_mask,
        legal_actions=legal,
    )


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
        state = _state(r, legal)
        tables = _FakeTables({r: None})
        policy = BlueprintPolicy(tables)
        sigma = policy.strategy(state)
        assert sigma.shape == (len(legal),)
        np.testing.assert_allclose(sigma, np.full(len(legal), 1.0 / len(legal)))

    def test_alignment_matches_legal_actions(self):
        r = 1
        legal = _legal_for(r)
        state = _state(r, legal)
        # Plant a row that's only positive on the first legal action.
        row = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
        row[ACTION_TO_IDX[r][legal[0]]] = 100
        tables = _FakeTables({r: row})
        sigma = BlueprintPolicy(tables).strategy(state)
        assert sigma.shape == (len(legal),)
        # All mass on the first legal action.
        assert sigma[0] == pytest.approx(1.0)
        np.testing.assert_allclose(sigma[1:], 0.0)

    def test_sums_to_one(self):
        r = 2
        legal = _legal_for(r)
        state = _state(r, legal)
        row = np.array(
            [(i * 7) % 13 for i in range(MAX_ACTIONS_PER_STREET[r])],
            dtype=np.int32,
        )
        tables = _FakeTables({r: row})
        sigma = BlueprintPolicy(tables).strategy(state)
        np.testing.assert_allclose(sigma.sum(), 1.0, atol=1e-6)

    def test_bias_shifts_mass_to_fold(self):
        r = 0
        legal = _legal_for(r)            # fold, call, raise:<f>, all_in → 4 legal
        state = _state(r, legal)
        # Uniform regrets across all canonical actions → σ uniform 0.25 over
        # the 4 legal actions; fold ×5 → pre-norm [1.25, .25, .25, .25] / 2.0.
        row = np.full(MAX_ACTIONS_PER_STREET[r], 10, dtype=np.int32)
        tables = _FakeTables({r: row})
        biased = BlueprintPolicy(tables, bias_multiplier=5.0).strategy(state, bias="fold")
        fold_i = legal.index("fold")
        assert biased[fold_i] == pytest.approx(0.625)
        others = np.delete(biased, fold_i)
        np.testing.assert_allclose(others, 0.125, atol=1e-6)
        np.testing.assert_allclose(biased.sum(), 1.0, atol=1e-6)

    def test_bias_multiplier_one_ignores_bias_arg(self):
        r = 0
        legal = _legal_for(r)
        state = _state(r, legal)
        row = np.full(MAX_ACTIONS_PER_STREET[r], 10, dtype=np.int32)
        tables = _FakeTables({r: row})
        policy = BlueprintPolicy(tables, bias_multiplier=1.0)
        np.testing.assert_array_equal(
            policy.strategy(state, bias="none"),
            policy.strategy(state, bias="fold"),
        )

    def test_bias_multiplier_one_matches_blueprint(self):
        # bias_multiplier=1 is numerically identical to plain regret matching.
        r = 1
        legal = _legal_for(r)
        state = _state(r, legal)
        row = np.array(
            [(i * 7) % 13 for i in range(MAX_ACTIONS_PER_STREET[r])],
            dtype=np.int32,
        )
        tables = _FakeTables({r: row})
        sigma = BlueprintPolicy(tables, bias_multiplier=1.0).strategy(state, bias="fold")
        # Reference: regret-match the canonical row, then project onto legal.
        full = calculate_strategy_from_row(row, state.valid_mask)
        expected = np.array(
            [full[ACTION_TO_IDX[r][a]] for a in legal], dtype=np.float32
        )
        expected /= expected.sum()
        np.testing.assert_allclose(sigma, expected, atol=1e-6)


class TestBlueprintPolicyAverageStrategy:
    """The blueprint's base σ is the normalised average strategy; the
    regret-matched strategy is only a fallback for rows whose visit
    mass over legal actions is below ``min_strategy_mass``."""

    def test_reads_average_strategy_when_mass_sufficient(self):
        r = 1
        legal = _legal_for(r)
        state = _state(r, legal)
        # Regrets would concentrate on legal[0]; the strategy table
        # instead holds a mixed 3:1 count on the first two legal
        # actions and must win.
        regret_row = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
        regret_row[ACTION_TO_IDX[r][legal[0]]] = 100
        strat_row = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
        strat_row[ACTION_TO_IDX[r][legal[0]]] = 30
        strat_row[ACTION_TO_IDX[r][legal[1]]] = 10
        tables = _FakeTables({r: regret_row}, {r: strat_row})
        sigma = BlueprintPolicy(tables, min_strategy_mass=10).strategy(state)
        assert sigma[0] == pytest.approx(0.75)
        assert sigma[1] == pytest.approx(0.25)
        np.testing.assert_allclose(sigma[2:], 0.0)

    def test_falls_back_to_regret_matching_below_mass_threshold(self):
        r = 1
        legal = _legal_for(r)
        state = _state(r, legal)
        regret_row = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
        regret_row[ACTION_TO_IDX[r][legal[0]]] = 100
        # A single visit — one categorical sample — must not be read
        # verbatim.
        strat_row = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
        strat_row[ACTION_TO_IDX[r][legal[1]]] = 1
        tables = _FakeTables({r: regret_row}, {r: strat_row})
        sigma = BlueprintPolicy(tables, min_strategy_mass=10).strategy(state)
        assert sigma[0] == pytest.approx(1.0)

    def test_illegal_action_counts_do_not_leak(self):
        r = 1
        # Restrict legality to fold/call only; plant strategy mass on a
        # raise column that is illegal at this node.
        legal = ["fold", "call"]
        state = _state(r, legal)
        strat_row = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
        strat_row[ACTION_TO_IDX[r]["fold"]] = 10
        strat_row[ACTION_TO_IDX[r]["call"]] = 30
        strat_row[3] = 1_000  # some raise column, illegal here
        tables = _FakeTables(
            {r: np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)},
            {r: strat_row},
        )
        sigma = BlueprintPolicy(tables, min_strategy_mass=10).strategy(state)
        assert sigma[legal.index("fold")] == pytest.approx(0.25)
        assert sigma[legal.index("call")] == pytest.approx(0.75)

    def test_bias_applies_to_average_strategy(self):
        r = 0
        legal = _legal_for(r)
        state = _state(r, legal)
        strat_row = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
        for a in legal:
            strat_row[ACTION_TO_IDX[r][a]] = 10  # uniform over legal
        tables = _FakeTables(
            {r: np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)},
            {r: strat_row},
        )
        biased = BlueprintPolicy(tables, bias_multiplier=5.0).strategy(
            state, bias="fold"
        )
        fold_i = legal.index("fold")
        assert biased[fold_i] == pytest.approx(
            5.0 / (5.0 + (len(legal) - 1))
        )


class TestBlueprintPolicyOverlay:
    """F3 regression: overlay-injected actions (§6.1) appear in
    ``env.legal_actions`` but not in ``ACTION_TO_IDX``.  The
    blueprint must assign them zero mass and renormalise the
    canonical share over the full filtered legal set, never
    raising ``KeyError``."""

    def test_overlay_action_gets_zero_mass(self):
        r = 0
        canonical = _legal_for(r)            # fold, call, raise:<f>, all_in
        overlay = "raise:0.42"               # not in ACTION_TO_IDX[r]
        legal = canonical + [overlay]
        state = _state(r, legal)
        # Uniform regrets across canonical actions.
        row = np.full(MAX_ACTIONS_PER_STREET[r], 10, dtype=np.int32)
        tables = _FakeTables({r: row})
        sigma = BlueprintPolicy(tables).strategy(state)
        assert sigma.shape == (len(legal),)
        # Overlay is the last entry — must be exactly zero.
        assert sigma[-1] == 0.0
        # Canonical share sums to one (the overlay's zero leaves the
        # canonical entries to renormalise over the full vector).
        np.testing.assert_allclose(sigma.sum(), 1.0, atol=1e-6)

    def test_overlay_alone_falls_back_to_uniform(self):
        r = 0
        state = _state(r, ["raise:0.42", "raise:0.77"])
        row = np.full(MAX_ACTIONS_PER_STREET[r], 10, dtype=np.int32)
        tables = _FakeTables({r: row})
        sigma = BlueprintPolicy(tables).strategy(state)
        # Both legal actions are non-canonical → uniform fallback.
        np.testing.assert_allclose(sigma, np.full(2, 0.5), atol=1e-6)

    def test_inactive_player_returns_empty(self):
        # legal_actions == [None] → filtered legal is empty.
        # BlueprintPolicy must return an empty array, not crash on
        # the all-overlay fallback's division by zero.
        r = 0
        state = _state(r, [None])
        tables = _FakeTables({r: np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)})
        sigma = BlueprintPolicy(tables).strategy(state)
        assert sigma.shape == (0,)

    def test_canonical_only_unchanged_by_overlay_path(self):
        # When no overlay action is present, the canonical-only
        # result must match what the pre-F3 implementation produced.
        r = 1
        legal = _legal_for(r)
        state = _state(r, legal)
        # Only one positive regret entry — strategy concentrates there.
        row = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
        row[ACTION_TO_IDX[r][legal[1]]] = 100
        tables = _FakeTables({r: row})
        sigma = BlueprintPolicy(tables).strategy(state)
        assert sigma[1] == pytest.approx(1.0)
        np.testing.assert_allclose(np.delete(sigma, 1), 0.0)
