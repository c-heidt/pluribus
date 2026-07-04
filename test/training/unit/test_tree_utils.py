"""Unit tests for ``poker_ai/ai/tree_utils.py``.

Covers:
- :func:`~poker_ai.blueprint.tree_utils.calculate_strategy_from_row` — pure numpy regret matching.
- :func:`~poker_ai.blueprint.tree_utils.is_terminal` — terminal detection and payout return.
- :func:`~poker_ai.blueprint.tree_utils.get_legal_actions` — legal action filtering.
- :func:`~poker_ai.blueprint.tree_utils.get_node_strategy` — regret-matching strategy lookup.
- :func:`~poker_ai.blueprint.tree_utils.sample_index` — inverse-CDF single-index sampler.

All tests use hand-crafted mock states so no LUT file is required.
"""

import numpy as np
import pytest

from poker_ai.blueprint.tree_utils import (
    calculate_strategy_from_row,
    get_legal_actions,
    get_node_strategy,
    is_terminal,
    sample_index,
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


# ---------------------------------------------------------------------------
# sample_index — inverse-CDF single-index sampler (search hot path)
# ---------------------------------------------------------------------------
class TestSampleIndex:
    """``sample_index`` must draw from the same distribution as the
    ``rng.choice(n, p=weights/weights.sum())`` it replaces, and honour the
    unnormalised-input / degenerate-mass contracts the callers rely on."""

    def _freqs(self, rng, weights, draws):
        n = len(weights)
        counts = np.zeros(n, dtype=np.int64)
        for _ in range(draws):
            idx = sample_index(rng, weights)
            assert 0 <= idx < n
            counts[idx] += 1
        return counts / draws

    def test_matches_target_distribution(self):
        rng = np.random.default_rng(0)
        weights = np.array([0.1, 0.4, 0.05, 0.45], dtype=np.float64)
        freqs = self._freqs(rng, weights, 200_000)
        # Frequencies converge to the normalised weights.
        assert np.max(np.abs(freqs - weights / weights.sum())) < 0.005

    def test_unnormalised_weights_same_as_normalised(self):
        # The threshold is scaled by the sum, so raw and normalised weights draw
        # from the identical distribution (the whole point of dropping _as_prob).
        raw = np.array([2.0, 8.0, 0.0, 10.0], dtype=np.float64)
        f_raw = self._freqs(np.random.default_rng(7), raw, 200_000)
        f_norm = self._freqs(np.random.default_rng(7), raw / raw.sum(), 200_000)
        assert np.max(np.abs(f_raw - f_norm)) < 0.005
        # Exact target, including the zero-weight entry never being drawn.
        assert f_raw[2] == 0.0
        assert np.max(np.abs(f_raw - raw / raw.sum())) < 0.005

    def test_float32_strategy_vector(self):
        # Callers pass a float32 σ straight from calculate_strategy_from_row.
        weights = np.array([0.25, 0.25, 0.5], dtype=np.float32)
        freqs = self._freqs(np.random.default_rng(3), weights, 200_000)
        assert np.max(np.abs(freqs - np.array([0.25, 0.25, 0.5]))) < 0.005

    def test_degenerate_zero_mass_is_uniform(self):
        # All-zero σ falls back to a uniform index (matches the sampler guard).
        weights = np.zeros(5, dtype=np.float64)
        freqs = self._freqs(np.random.default_rng(1), weights, 200_000)
        assert np.max(np.abs(freqs - 0.2)) < 0.01

    def test_matches_generator_choice_distribution(self):
        # Head-to-head with the replaced primitive: same target, both within
        # sampling error of the analytic distribution.
        weights = np.array([0.3, 0.1, 0.6], dtype=np.float64)
        p = weights / weights.sum()
        f_new = self._freqs(np.random.default_rng(11), weights, 200_000)
        rng = np.random.default_rng(11)
        counts = np.zeros(3, dtype=np.int64)
        for _ in range(200_000):
            counts[int(rng.choice(3, p=p))] += 1
        f_choice = counts / 200_000
        assert np.max(np.abs(f_new - f_choice)) < 0.01

    def test_single_candidate_always_index_zero(self):
        rng = np.random.default_rng(5)
        for _ in range(100):
            assert sample_index(rng, np.array([1.0])) == 0
