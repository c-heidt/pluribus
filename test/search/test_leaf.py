"""Tests for :mod:`poker_ai.search.leaf` (§6.4 continuation-value rework).

``continuation_value(frontier_env, profile, ctx)`` evaluates a **fixed**
continuation profile over the frontier's **concrete** hands by Monte-Carlo
rollout — no hole resampling, the bias per seat supplied (not drawn).  Every
rollout settles on its single sampled board via the ordinary concrete payout
(the decision-free exact-equity variant was tried and dropped; this path now
matches the compiled core's leaf rollout).
"""

import collections
from typing import List, Tuple

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv, PolicyState
from poker_ai.search.context import SubgameContext
from poker_ai.search.leaf import LeafConfig, board_rng_for, continuation_value
from poker_ai.search.policy import BiasClass, Policy
from test.abstraction_helpers import passive_action


def _full_deck_env(n_players: int = 2) -> PokerEnv:
    return PokerEnv(players=[Player(i, 10000) for i in range(n_players)])


def _stub_lut(env: PokerEnv) -> None:
    env.card_info_lut = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))


def _to_flop(env: PokerEnv) -> None:
    """Calldown pre-flop so a 3-card board is dealt."""
    while env.betting_round < 1 and not env.is_terminal:
        env.step_in_place(passive_action(env))


class UniformPolicy(Policy):
    """Uniform over ``state.legal_actions``; records every call."""

    def __init__(self) -> None:
        self.calls: List[Tuple[int, BiasClass, str]] = []

    def strategy(self, state: PolicyState, bias: BiasClass = "none") -> np.ndarray:
        self.calls.append((state.player_i, bias, state.info_set))
        n = len(state.legal_actions)
        if n == 0:
            return np.array([], dtype=np.float32)
        return np.full(n, 1.0 / n, dtype=np.float32)


class FoldOrCallPolicy(Policy):
    """Folds if legal, else the first legal action."""

    def strategy(self, state: PolicyState, bias: BiasClass = "none") -> np.ndarray:
        legal = state.legal_actions
        probs = np.zeros(len(legal), dtype=np.float32)
        for i, a in enumerate(legal):
            if a == "fold":
                probs[i] = 1.0
                return probs
        probs[0] = 1.0
        return probs


def _policies(cls=UniformPolicy) -> dict:
    return {c: cls() for c in ("none", "fold", "call", "raise")}


def _ctx(env, *, policies=None, seed=42):
    if policies is None:
        policies = _policies()
    return SubgameContext.from_runtime(
        env=env,
        my_seat=0,
        my_hole=tuple(int(c) for c in env.players[0].cards),
        ranges={1: np.ones(env.n_combos, dtype=np.float32)},
        folded_ranges={},
        leaf=LeafConfig(policies=policies),
        rng=np.random.default_rng(seed),
    )


def _profile(env, bias: BiasClass = "none") -> dict:
    return {i: bias for i in range(env.n_players)}


def _spy_with_hole_cards(monkeypatch):
    recorded: List[List[Tuple[int, ...]]] = []
    original = PokerEnv.with_hole_cards

    def spy(self, holes, **kwargs):
        recorded.append([tuple(int(c) for c in h) for h in holes])
        return original(self, holes, **kwargs)

    monkeypatch.setattr(PokerEnv, "with_hole_cards", spy)
    return recorded


class TestLeafConfig:

    def test_fields(self):
        pol = _policies()
        cfg = LeafConfig(policies=pol)
        assert cfg.policies is pol


class TestTerminal:

    def test_terminal_returns_payout(self):
        env = _full_deck_env(); _stub_lut(env)
        ctx = _ctx(env)  # build the ctx while the env is still non-terminal
        env.step_in_place("fold")
        assert env.is_terminal
        out = continuation_value(env, _profile(env), ctx)
        expected = np.array([float(env.payout[i]) for i in range(env.n_players)])
        np.testing.assert_array_equal(out, expected)

    def test_terminal_consults_no_policy(self):
        env = _full_deck_env(); _stub_lut(env)
        pol = _policies()
        ctx = _ctx(env, policies=pol)
        env.step_in_place("fold")
        continuation_value(env, _profile(env), ctx)
        for p in pol.values():
            assert p.calls == []


class TestRolloutBasics:

    def test_shape_and_finite(self):
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        out = continuation_value(env, _profile(env), _ctx(env))
        assert out.shape == (env.n_players,)
        assert np.isfinite(out).all()

    def test_finite_zero_sum(self):
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        out = continuation_value(env, _profile(env), _ctx(env, seed=123))
        assert np.isfinite(out).all()
        assert abs(out.sum()) < 1e-6

    def test_payout_from_env_fold_or_call(self):
        # SB folds pre-flop under FoldOrCall → loses the small blind only.
        env = _full_deck_env(); _stub_lut(env)
        out = continuation_value(
            env, _profile(env), _ctx(env, policies=_policies(FoldOrCallPolicy))
        )
        assert abs(out.sum()) < 1e-6
        assert out[0] * out[1] < 0

    def test_float32_drift_succeeds(self):
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)

        class Float32Thirds(Policy):
            def strategy(self, state, bias="none"):
                legal = state.legal_actions
                if not legal:
                    return np.array([], dtype=np.float32)
                base = np.full(len(legal), 1.0 / len(legal), dtype=np.float32)
                return (base * np.float32(7.0)) / np.float32(7.0)

        out = continuation_value(
            env, _profile(env), _ctx(env, policies={c: Float32Thirds() for c in
                                                    ("none", "fold", "call", "raise")})
        )
        assert out.shape == (env.n_players,)


class TestDeterminism:

    def test_same_seed_identical(self):
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        np.random.seed(0)
        a = continuation_value(env, _profile(env), _ctx(env, seed=7))
        np.random.seed(0)
        b = continuation_value(env, _profile(env), _ctx(env, seed=7))
        np.testing.assert_array_equal(a, b)

    def test_different_ctx_seed_diverges(self):
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        a = continuation_value(env, _profile(env), _ctx(env, seed=1))
        b = continuation_value(env, _profile(env), _ctx(env, seed=2))
        assert not np.array_equal(a, b)


class TestNoResampling:

    def test_rollout_uses_frontier_concrete_holes(self, monkeypatch):
        env = _full_deck_env(n_players=3); _stub_lut(env); _to_flop(env)
        expected = [tuple(int(c) for c in env.players[i].cards) for i in range(3)]
        recorded = _spy_with_hole_cards(monkeypatch)
        ctx = SubgameContext.from_runtime(
            env=env, my_seat=0, my_hole=expected[0],
            ranges={1: np.ones(env.n_combos, dtype=np.float32),
                    2: np.ones(env.n_combos, dtype=np.float32)},
            folded_ranges={}, leaf=LeafConfig(policies=_policies(FoldOrCallPolicy)),
            rng=np.random.default_rng(3),
        )
        continuation_value(env, _profile(env), ctx)
        assert recorded == [expected]  # the rollout used the frontier's own holes


class TestProfileDrivesBias:

    def test_seat_uses_its_profile_bias(self):
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        pol = _policies()
        profile = {0: "raise", 1: "fold"}
        continuation_value(env, profile, _ctx(env, policies=pol))
        # Every recorded call for a seat carries that seat's profile bias, and
        # only the policy of that bias was consulted for the seat.
        for bias, p in pol.items():
            for seat, b, _info in p.calls:
                assert b == bias                      # policy[bias] only ever queried with bias
                assert profile[seat] == bias          # and only for seats whose profile is bias

    def test_missing_seat_raises(self):
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        with pytest.raises(ValueError, match="missing acting seat"):
            continuation_value(env, {0: "none"}, _ctx(env))  # seat 1 absent


class TestBlueprintCanonicalisation:

    def test_rollout_uses_for_blueprint_lookups(self, monkeypatch):
        # Every policy query during a rollout must go through
        # ``policy_state_for(..., for_blueprint=True)`` so off-tree histories
        # canonicalise to populated blueprint rows (§6.3).
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        seen_flags = []
        original = PokerEnv.policy_state_for

        def spy(self, combo, *, for_blueprint=False):
            seen_flags.append(for_blueprint)
            return original(self, combo, for_blueprint=for_blueprint)

        monkeypatch.setattr(PokerEnv, "policy_state_for", spy)
        continuation_value(env, _profile(env), _ctx(env))
        assert seen_flags  # the rollout did query a policy
        assert all(seen_flags)  # always for_blueprint=True


# --------------------------------------------------------------------------- #
# RNG ownership (poker_ai.search.rng)
# --------------------------------------------------------------------------- #

def _global_state():
    """Hashable snapshot of the global MT19937 state (key *and* position)."""
    st = np.random.get_state()
    return (st[0], st[1].tobytes(), st[2], st[3], st[4])


class TestRngOwnership:
    """A leaf rollout is a *hypothetical* re-deal and must own its randomness.

    Drawing the rollout board from the global stream coupled the search to every
    other consumer of that stream — and, in the other direction, made AIVAT's value
    function depend on the hero's iteration count, which is what destroyed the
    evaluation's CRN pairing.
    """

    def test_rollout_does_not_consume_global_rng(self):
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        ctx = _ctx(env)
        before = _global_state()
        continuation_value(env, _profile(env), ctx)
        assert _global_state() == before

    def test_rollout_is_independent_of_global_stream_position(self):
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        np.random.seed(1)
        a = continuation_value(env, _profile(env), _ctx(env, seed=9))
        np.random.seed(2)
        np.random.random(41)                      # unrelated global consumer
        b = continuation_value(env, _profile(env), _ctx(env, seed=9))
        np.testing.assert_array_equal(a, b)

    def test_from_runtime_derives_a_board_stream(self):
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        ctx = _ctx(env)
        assert ctx.board_rng is not None
        assert ctx.board_rng is not ctx.rng

    def test_board_stream_is_distinct_from_sampling_stream(self):
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        ctx = _ctx(env)
        a = [float(ctx.rng.random()) for _ in range(8)]
        b = [float(ctx.board_rng.random()) for _ in range(8)]
        assert a != b

    def test_deriving_board_stream_leaves_sampling_stream_untouched(self):
        """Adding the board split must not change the solver's sampling draws."""
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        baseline = np.random.default_rng(42)
        ctx = _ctx(env, seed=42)
        assert [float(ctx.rng.random()) for _ in range(8)] == \
               [float(baseline.random()) for _ in range(8)]

    def test_board_rng_for_falls_back_to_ctx_rng(self):
        """A duck-typed carrier without ``board_rng`` falls back to ``rng`` —
        never to the global stream."""
        rng = np.random.default_rng(0)
        carrier = collections.namedtuple("C", "leaf rng")(None, rng)
        assert board_rng_for(carrier) is rng

    def test_board_rng_for_prefers_board_rng(self):
        rng, brng = np.random.default_rng(0), np.random.default_rng(1)
        carrier = collections.namedtuple("C", "leaf rng board_rng")(None, rng, brng)
        assert board_rng_for(carrier) is brng
