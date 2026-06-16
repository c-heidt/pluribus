"""Tests for :mod:`poker_ai.search.leaf`."""

import warnings
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv, PolicyState
from poker_ai.search.context import SubgameContext
from poker_ai.search.leaf import LeafConfig, leaf_value
from poker_ai.search.policy import BiasClass, Policy


def _env(low: int = 10, high: int = 14, n_players: int = 2) -> PokerEnv:
    return PokerEnv(
        players=[Player(i, 10000) for i in range(n_players)],
        low_card_rank=low,
        high_card_rank=high,
    )


def _full_deck_env(n_players: int = 2) -> PokerEnv:
    """Full deck (2..14) — required when the rollout must reach a
    showdown / fold-terminal, because the Evaluator's lookup tables
    are sized for the full rank range."""
    return PokerEnv(players=[Player(i, 10000) for i in range(n_players)])


def _stub_lut(env: PokerEnv) -> None:
    env.card_info_lut = defaultdict(lambda: defaultdict(lambda: 0))


class UniformPolicy(Policy):
    """Returns uniform over ``state.legal_actions``.  Records every call."""

    def __init__(self) -> None:
        self.calls: List[Tuple[int, BiasClass, Tuple[str, ...]]] = []

    def strategy(self, state: PolicyState, bias: BiasClass = "none") -> np.ndarray:
        self.calls.append((state.player_i, bias, tuple(state.legal_actions)))
        n = len(state.legal_actions)
        if n == 0:
            return np.array([], dtype=np.float32)
        return np.full(n, 1.0 / n, dtype=np.float32)


class FoldOrCallPolicy(Policy):
    """Always folds if legal; otherwise picks the first legal action."""

    def strategy(self, state: PolicyState, bias: BiasClass = "none") -> np.ndarray:
        legal = state.legal_actions
        probs = np.zeros(len(legal), dtype=np.float32)
        for i, a in enumerate(legal):
            if a == "fold":
                probs[i] = 1.0
                return probs
        probs[0] = 1.0
        return probs


def _uniform_policies() -> dict:
    return {c: UniformPolicy() for c in ("none", "fold", "call", "raise")}


def _build_ctx(
    env: PokerEnv,
    *,
    n_rollouts: int = 10,
    seed=None,
    policies: dict = None,
    opponent_ranges: dict = None,
) -> SubgameContext:
    if policies is None:
        policies = _uniform_policies()
    if opponent_ranges is None:
        opponent_ranges = {1: np.ones(env.n_combos, dtype=np.float32)}
    if seed is None:
        # Draw the ctx.rng seed from the global RNG state, which is
        # itself reseeded per trial by the autouse ``_seeded`` fixture
        # in ``conftest.py``.  Tests that need a specific seed (e.g.
        # to verify reproducibility) pass it explicitly.
        seed = int(np.random.randint(0, 2**31 - 1))
    return SubgameContext.from_runtime(
        env=env,
        my_seat=0,
        my_hole=tuple(int(c) for c in env.players[0].cards),
        opponent_ranges=opponent_ranges,
        leaf=LeafConfig(policies=policies, n_rollouts=n_rollouts),
        rng=np.random.default_rng(seed),
    )


def _capture_with_hole_cards(monkeypatch):
    """Install a spy on ``PokerEnv.with_hole_cards`` that records every
    ``holes`` argument the leaf passes through, then defers to the
    original method.  Returns the recording list (list of lists).
    """
    recorded: List[List[Tuple[int, int]]] = []
    original = PokerEnv.with_hole_cards

    def spy(self, holes):
        recorded.append([tuple(int(c) for c in h) for h in holes])
        return original(self, holes)

    monkeypatch.setattr(PokerEnv, "with_hole_cards", spy)
    return recorded


class TestLeafConfig:

    def test_fields_pass_through(self):
        policies = _uniform_policies()
        cfg = LeafConfig(policies=policies, n_rollouts=3)
        assert cfg.policies is policies
        assert cfg.n_rollouts == 3

    def test_default_n_rollouts(self):
        cfg = LeafConfig(policies=_uniform_policies())
        assert cfg.n_rollouts == 20

    def test_default_n_rejection_retries(self):
        cfg = LeafConfig(policies=_uniform_policies())
        assert cfg.n_rejection_retries == 4


class TestLeafValueTerminal:

    def test_terminal_leaf_returns_payout(self):
        root_env = _full_deck_env()
        _stub_lut(root_env)
        ctx = _build_ctx(root_env, n_rollouts=5)
        leaf_env = root_env.apply_action("fold")
        assert leaf_env.is_terminal
        result = leaf_value(leaf_env, {}, {}, ctx)
        expected = np.array(
            [float(leaf_env.payout[i]) for i in range(leaf_env.n_players)],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(result, expected)

    def test_terminal_leaf_does_not_call_policy(self):
        root_env = _full_deck_env()
        _stub_lut(root_env)
        policies = _uniform_policies()
        ctx = _build_ctx(root_env, policies=policies)
        leaf_env = root_env.apply_action("fold")
        leaf_value(leaf_env, {}, {}, ctx)
        for p in policies.values():
            assert p.calls == []


class TestLeafValueDeterminism:

    def test_same_seed_same_value(self):
        # The leaf's per-rollout decisions use ctx.rng; the env's
        # deck.shuffle_undealt uses the global numpy RNG.  Pin both
        # for bit-exact reproducibility.
        env = _full_deck_env()
        _stub_lut(env)
        np.random.seed(0)
        ctx_a = _build_ctx(env, seed=42, n_rollouts=8)
        a = leaf_value(env, {1: ctx_a.opponent_ranges[1]}, {}, ctx_a)
        np.random.seed(0)
        ctx_b = _build_ctx(env, seed=42, n_rollouts=8)
        b = leaf_value(env, {1: ctx_b.opponent_ranges[1]}, {}, ctx_b)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_diverge(self):
        env = _full_deck_env()
        _stub_lut(env)
        ctx_a = _build_ctx(env, seed=1, n_rollouts=20)
        ctx_b = _build_ctx(env, seed=2, n_rollouts=20)
        a = leaf_value(env, {1: ctx_a.opponent_ranges[1]}, {}, ctx_a)
        b = leaf_value(env, {1: ctx_b.opponent_ranges[1]}, {}, ctx_b)
        assert not np.array_equal(a, b)


class TestHoleSampling:

    def test_sampled_holes_exclude_community_and_my_hole(self, monkeypatch):
        # Walk to the flop so a real board is dealt; capture every
        # ``with_hole_cards`` call's input list and verify opponent
        # entries are disjoint from board + my_hole.
        env = _full_deck_env()
        _stub_lut(env)
        env = env.apply_action("call")
        env = env.apply_action("call")
        assert env.betting_round == 1
        assert len(env.community_cards) == 3
        my_hole = set(int(c) for c in env.players[0].cards)
        board = set(int(c) for c in env.community_cards)

        recorded = _capture_with_hole_cards(monkeypatch)
        policies = {c: FoldOrCallPolicy() for c in ("none", "fold", "call", "raise")}
        ctx = _build_ctx(env, policies=policies, n_rollouts=30)
        leaf_value(env, {1: ctx.opponent_ranges[1]}, {}, ctx)
        assert recorded, "expected at least one with_hole_cards call"
        for holes in recorded:
            opp = set(holes[1])
            assert opp.isdisjoint(board), (opp, board)
            assert opp.isdisjoint(my_hole), (opp, my_hole)

    def test_sampled_holes_mutually_disjoint_three_seats(self, monkeypatch):
        env = _full_deck_env(n_players=3)
        _stub_lut(env)
        ranges = {
            1: np.ones(env.n_combos, dtype=np.float32),
            2: np.ones(env.n_combos, dtype=np.float32),
        }
        recorded = _capture_with_hole_cards(monkeypatch)
        policies = {c: FoldOrCallPolicy() for c in ("none", "fold", "call", "raise")}
        ctx = SubgameContext.from_runtime(
            env=env,
            my_seat=0,
            my_hole=tuple(int(c) for c in env.players[0].cards),
            opponent_ranges=ranges,
            leaf=LeafConfig(policies=policies, n_rollouts=15),
            rng=np.random.default_rng(7),
        )
        leaf_value(env, dict(ctx.opponent_ranges), {}, ctx)
        assert recorded
        for holes in recorded:
            assert set(holes[1]).isdisjoint(set(holes[2])), holes

    def test_uniform_fallback_on_zero_range_warns(self):
        env = _full_deck_env()
        _stub_lut(env)
        zero_range = {1: np.zeros(env.n_combos, dtype=np.float32)}
        policies = {c: FoldOrCallPolicy() for c in ("none", "fold", "call", "raise")}
        ctx = _build_ctx(
            env, n_rollouts=3, opponent_ranges=zero_range, policies=policies
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = leaf_value(env, {1: ctx.opponent_ranges[1]}, {}, ctx)
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)
        assert result.shape == (env.n_players,)
        assert np.isfinite(result).all()


class TestRolloutMechanics:

    def test_observed_biases_subset_of_classes(self):
        env = _full_deck_env()
        _stub_lut(env)
        policies = _uniform_policies()
        ctx = _build_ctx(env, policies=policies, n_rollouts=4)
        leaf_value(env, {1: ctx.opponent_ranges[1]}, {}, ctx)
        observed_biases = set()
        for p in policies.values():
            for _player_i, bias, _legal in p.calls:
                observed_biases.add(bias)
        assert observed_biases.issubset({"none", "fold", "call", "raise"})

    def test_only_active_seat_strategy_consulted(self):
        env = _full_deck_env()
        _stub_lut(env)
        policies = _uniform_policies()
        ctx = _build_ctx(env, policies=policies, n_rollouts=5)
        leaf_value(env, {1: ctx.opponent_ranges[1]}, {}, ctx)
        for p in policies.values():
            for player_i, _bias, _legal in p.calls:
                assert player_i in {0, 1}

    def test_uses_ctx_rng_for_sampling_decisions(self, monkeypatch):
        # The leaf's sampling and action choices must come from
        # ctx.rng, not the module-level ``np.random.choice``.
        # ``Deck.shuffle_undealt`` uses ``np.random.shuffle`` which
        # is a different function — only patch ``choice``.
        env = _full_deck_env()
        _stub_lut(env)

        def _explode(*args, **kwargs):
            raise AssertionError(
                "leaf_value must not touch np.random.choice; "
                "ctx.rng is the sole source of sampling randomness."
            )

        monkeypatch.setattr(np.random, "choice", _explode)
        ctx = _build_ctx(env, n_rollouts=5)
        result = leaf_value(env, {1: ctx.opponent_ranges[1]}, {}, ctx)
        assert result.shape == (env.n_players,)

    def test_payout_from_env_not_evaluator(self):
        # FoldOrCallPolicy always folds; SB folds preflop → loses SB.
        env = _full_deck_env()
        _stub_lut(env)
        policies = {c: FoldOrCallPolicy() for c in ("none", "fold", "call", "raise")}
        ctx = _build_ctx(env, policies=policies, n_rollouts=3)
        result = leaf_value(env, {1: ctx.opponent_ranges[1]}, {}, ctx)
        assert abs(result.sum()) < 1e-6
        assert result[0] * result[1] < 0


class TestLeafValuePreconditions:

    def test_live_ranges_with_my_seat_raises(self):
        env = _env()
        _stub_lut(env)
        ctx = _build_ctx(env, n_rollouts=3)
        bad_ranges = {0: np.ones(env.n_combos, dtype=np.float32)}
        with pytest.raises(ValueError, match="my_seat"):
            leaf_value(env, bad_ranges, {}, ctx)

    def test_folded_ranges_with_my_seat_raises(self):
        env = _env()
        _stub_lut(env)
        ctx = _build_ctx(env, n_rollouts=3)
        with pytest.raises(ValueError, match="my_seat"):
            leaf_value(env, {}, {0: np.ones(env.n_combos, dtype=np.float32)}, ctx)

    def test_overlap_between_live_and_folded_raises(self):
        # The same seat cannot be both live and folded simultaneously.
        env = _env()
        _stub_lut(env)
        ctx = _build_ctx(env, n_rollouts=3)
        w = np.ones(env.n_combos, dtype=np.float32)
        with pytest.raises(ValueError, match="both"):
            leaf_value(env, {1: w}, {1: w}, ctx)


class TestNonTerminalEmptyRanges:

    def test_empty_ranges_non_terminal(self):
        # Heads-up env with no live opponents (e.g. bot is HU after a
        # fold already happened upstream of the search).  No sampling
        # needed; rollout runs against env's current holes.
        env = _full_deck_env(n_players=3)
        _stub_lut(env)
        ctx = _build_ctx(
            env,
            n_rollouts=3,
            opponent_ranges={
                2: np.ones(env.n_combos, dtype=np.float32),
            },
        )
        result = leaf_value(env, {}, {}, ctx)
        assert result.shape == (env.n_players,)
        assert np.isfinite(result).all()


class TestNRolloutsEdge:

    def test_zero_rollouts_returns_zeros(self):
        env = _env()
        _stub_lut(env)
        ctx = _build_ctx(env, n_rollouts=0)
        result = leaf_value(env, {1: ctx.opponent_ranges[1]}, {}, ctx)
        np.testing.assert_array_equal(result, np.zeros(env.n_players))


class TestBiasDiversity:

    def test_all_four_classes_appear(self):
        # 50 rollouts × 2 seats = 100 bias draws.  Missing-a-class
        # probability is (3/4)^100 ≈ 3e-13.
        env = _full_deck_env()
        _stub_lut(env)
        policies = _uniform_policies()
        ctx = _build_ctx(env, policies=policies, n_rollouts=50)
        leaf_value(env, {1: ctx.opponent_ranges[1]}, {}, ctx)
        observed = set()
        for p in policies.values():
            for _player_i, bias, _legal in p.calls:
                observed.add(bias)
        assert observed == {"none", "fold", "call", "raise"}


class TestHoleResampling:

    def test_resampled_holes_reflect_one_hot_range(self, monkeypatch):
        # One-hot range pins seat 1 to a target combo; every
        # ``with_hole_cards`` call must place that combo at seat 1.
        env = _full_deck_env()
        _stub_lut(env)
        original_opp = tuple(int(c) for c in env.players[1].cards)
        my_hole = set(int(c) for c in env.players[0].cards)
        target_idx = None
        for i in range(env.n_combos):
            combo = (int(env.combo_cards[i, 0]), int(env.combo_cards[i, 1]))
            if combo != original_opp and not (set(combo) & my_hole):
                target_idx = i
                break
        assert target_idx is not None
        one_hot = np.zeros(env.n_combos, dtype=np.float32)
        one_hot[target_idx] = 1.0
        expected = (
            int(env.combo_cards[target_idx, 0]),
            int(env.combo_cards[target_idx, 1]),
        )

        recorded = _capture_with_hole_cards(monkeypatch)
        policies = {c: FoldOrCallPolicy() for c in ("none", "fold", "call", "raise")}
        ctx = _build_ctx(env, policies=policies, n_rollouts=5)
        leaf_value(env, {1: one_hot}, {}, ctx)
        assert recorded
        for holes in recorded:
            assert set(holes[1]) == set(expected), (holes[1], expected)


class TestFoldedHoleSampling:
    """Folded seats' holes are resampled from ``folded_ranges`` —
    their fold-time marginal — rather than left at the simulator's
    original deal.  Verifies the leaf calls ``with_hole_cards`` with
    folded seats' cards drawn from the supplied range."""

    def test_folded_seat_resampled_from_one_hot_range(self, monkeypatch):
        # 3-player env, bot=0, seat 1 live, seat 2 "folded" — its
        # range goes in folded_ranges.  One-hot pin verifies the
        # resampling actually consumes folded_ranges.
        env = _full_deck_env(n_players=3)
        _stub_lut(env)
        my_hole = set(int(c) for c in env.players[0].cards)
        seat1_hole = set(int(c) for c in env.players[1].cards)
        # Target combo disjoint from my_hole and from seat 1's
        # current hole (so the rejection sampler never has to retry
        # for an extreme reason).
        target_idx = None
        for i in range(env.n_combos):
            combo = (int(env.combo_cards[i, 0]), int(env.combo_cards[i, 1]))
            if not (set(combo) & my_hole) and not (set(combo) & seat1_hole):
                target_idx = i
                break
        assert target_idx is not None
        one_hot = np.zeros(env.n_combos, dtype=np.float32)
        one_hot[target_idx] = 1.0
        expected = (
            int(env.combo_cards[target_idx, 0]),
            int(env.combo_cards[target_idx, 1]),
        )

        recorded = _capture_with_hole_cards(monkeypatch)
        policies = {c: FoldOrCallPolicy() for c in ("none", "fold", "call", "raise")}
        live = {1: np.ones(env.n_combos, dtype=np.float32)}
        folded = {2: one_hot}
        ctx = SubgameContext.from_runtime(
            env=env,
            my_seat=0,
            my_hole=tuple(int(c) for c in env.players[0].cards),
            opponent_ranges=live,
            leaf=LeafConfig(policies=policies, n_rollouts=10),
            rng=np.random.default_rng(3),
        )
        leaf_value(env, dict(ctx.opponent_ranges), folded, ctx)
        assert recorded
        for holes in recorded:
            assert set(holes[2]) == set(expected), (holes[2], expected)

    def test_bot_seat_unchanged_under_folded_sampling(self, monkeypatch):
        env = _full_deck_env(n_players=3)
        _stub_lut(env)
        my_hole = tuple(int(c) for c in env.players[0].cards)
        recorded = _capture_with_hole_cards(monkeypatch)
        policies = {c: FoldOrCallPolicy() for c in ("none", "fold", "call", "raise")}
        live = {1: np.ones(env.n_combos, dtype=np.float32)}
        folded = {2: np.ones(env.n_combos, dtype=np.float32)}
        ctx = SubgameContext.from_runtime(
            env=env,
            my_seat=0,
            my_hole=my_hole,
            opponent_ranges=live,
            leaf=LeafConfig(policies=policies, n_rollouts=5),
            rng=np.random.default_rng(5),
        )
        leaf_value(env, dict(ctx.opponent_ranges), folded, ctx)
        assert recorded
        for holes in recorded:
            assert tuple(holes[0]) == my_hole


class TestRolloutAbort:

    def test_returns_zeros_when_every_rollout_aborts(self):
        env = _env()
        _stub_lut(env)
        all_cards = set()
        for i in range(env.n_combos):
            all_cards.add(int(env.combo_cards[i, 0]))
            all_cards.add(int(env.combo_cards[i, 1]))
        env.community_cards = tuple(all_cards)
        zero_range = {1: np.zeros(env.n_combos, dtype=np.float32)}
        ctx = _build_ctx(env, n_rollouts=3, opponent_ranges=zero_range)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = leaf_value(env, {1: ctx.opponent_ranges[1]}, {}, ctx)
        np.testing.assert_array_equal(result, np.zeros(env.n_players))


class TestFloat32Probabilities:

    def test_float32_probs_with_drift_succeeds(self):
        env = _full_deck_env()
        _stub_lut(env)

        class Float32ThirdsPolicy(Policy):
            def strategy(self, state: PolicyState, bias: BiasClass = "none") -> np.ndarray:
                legal = state.legal_actions
                if not legal:
                    return np.array([], dtype=np.float32)
                base = np.full(len(legal), 1.0 / len(legal), dtype=np.float32)
                return (base * np.float32(7.0)) / np.float32(7.0)

        policies = {c: Float32ThirdsPolicy() for c in ("none", "fold", "call", "raise")}
        ctx = _build_ctx(env, policies=policies, n_rollouts=10)
        result = leaf_value(env, {1: ctx.opponent_ranges[1]}, {}, ctx)
        assert result.shape == (env.n_players,)


class TestLeafConvergence:

    def test_large_sample_finite_and_zero_sum(self):
        env = _full_deck_env()
        _stub_lut(env)
        ctx = _build_ctx(env, n_rollouts=50, seed=12345)
        result = leaf_value(env, {1: ctx.opponent_ranges[1]}, {}, ctx)
        assert np.isfinite(result).all()
        assert abs(result.sum()) < 1e-6
