"""Tests for :mod:`poker_ai.search.leaf` (§6.4 continuation-value rework).

``continuation_value(frontier_env, profile, ctx)`` evaluates a **fixed**
continuation profile over the frontier's **concrete** hands by Monte-Carlo
rollout — no hole resampling, the bias per seat supplied (not drawn).  When a
rollout reaches an all-in showdown over an incomplete board it takes the exact
board-average via :meth:`PokerEnv.runout_equity`, gated by the
``use_decision_free_equity`` A/B toggle.
"""

import collections
import copy
from typing import List, Tuple

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv, PolicyState
from poker_ai.search.context import SubgameContext
from poker_ai.search.leaf import LeafConfig, continuation_value
from poker_ai.search.policy import BiasClass, Policy


def _full_deck_env(n_players: int = 2) -> PokerEnv:
    return PokerEnv(players=[Player(i, 10000) for i in range(n_players)])


def _stub_lut(env: PokerEnv) -> None:
    env.card_info_lut = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))


def _to_flop(env: PokerEnv) -> None:
    """Calldown pre-flop so a 3-card board is dealt."""
    while env.betting_round < 1 and not env.is_terminal:
        env.step_in_place("call" if "call" in env.legal_actions else "check")


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


class AllInPolicy(Policy):
    """Goes all-in whenever legal, else the first legal action — forces a
    decision-free runout."""

    def strategy(self, state: PolicyState, bias: BiasClass = "none") -> np.ndarray:
        legal = state.legal_actions
        probs = np.zeros(len(legal), dtype=np.float32)
        for i, a in enumerate(legal):
            if a == "all_in":
                probs[i] = 1.0
                return probs
        probs[0] = 1.0
        return probs


class CheckCallPolicy(Policy):
    """Only ever checks/calls — never folds, raises, or goes all-in, so a
    rollout reaches a normal complete-board showdown (no decision-free runout)."""

    def strategy(self, state: PolicyState, bias: BiasClass = "none") -> np.ndarray:
        legal = state.legal_actions
        probs = np.zeros(len(legal), dtype=np.float32)
        for i, a in enumerate(legal):
            if a in ("check", "call"):
                probs[i] = 1.0
                return probs
        probs[0] = 1.0
        return probs


def _policies(cls=UniformPolicy) -> dict:
    return {c: cls() for c in ("none", "fold", "call", "raise")}


def _ctx(env, *, policies=None, n_rollouts=10, seed=42, use_equity=True):
    if policies is None:
        policies = _policies()
    return SubgameContext.from_runtime(
        env=env,
        my_seat=0,
        my_hole=tuple(int(c) for c in env.players[0].cards),
        ranges={1: np.ones(env.n_combos, dtype=np.float32)},
        folded_ranges={},
        leaf=LeafConfig(policies=policies, n_rollouts=n_rollouts,
                        use_decision_free_equity=use_equity),
        rng=np.random.default_rng(seed),
    )


def _profile(env, bias: BiasClass = "none") -> dict:
    return {i: bias for i in range(env.n_players)}


def _spy_with_hole_cards(monkeypatch):
    recorded: List[List[Tuple[int, ...]]] = []
    original = PokerEnv.with_hole_cards

    def spy(self, holes):
        recorded.append([tuple(int(c) for c in h) for h in holes])
        return original(self, holes)

    monkeypatch.setattr(PokerEnv, "with_hole_cards", spy)
    return recorded


class TestLeafConfig:

    def test_fields(self):
        pol = _policies()
        cfg = LeafConfig(policies=pol, n_rollouts=3)
        assert cfg.policies is pol
        assert cfg.n_rollouts == 3

    def test_default_n_rollouts(self):
        assert LeafConfig(policies=_policies()).n_rollouts == 1

    def test_default_use_decision_free_equity_true(self):
        assert LeafConfig(policies=_policies()).use_decision_free_equity is True


class TestTerminal:

    def test_terminal_returns_payout(self):
        env = _full_deck_env(); _stub_lut(env)
        ctx = _ctx(env)  # build the ctx while the env is still non-terminal
        env.step_in_place("fold")
        assert env.is_terminal
        out = continuation_value(env, _profile(env), ctx)
        expected = np.array([float(env.payout[i]) for i in range(env.n_players)])
        np.testing.assert_array_equal(out, expected)

    def test_terminal_allin_frontier_honours_equity_flag(self):
        # An already-terminal all-in frontier is scored by the same toggle as a
        # mid-rollout terminal: flag-on → exact runout_equity, flag-off → payout.
        np.random.seed(13)
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        ctx_on = _ctx(env, n_rollouts=3, use_equity=True)
        ctx_off = _ctx(env, n_rollouts=3, use_equity=False)
        env.step_in_place("all_in")  # force-resolve → decision-free terminal
        assert env.is_terminal and env.is_decision_free
        eq = env.runout_equity()
        out_on = continuation_value(env, _profile(env), ctx_on)
        out_off = continuation_value(env, _profile(env), ctx_off)
        for i in range(env.n_players):
            assert abs(out_on[i] - eq[i]) < 1e-9
            assert abs(out_off[i] - float(env.payout[i])) < 1e-9

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

    def test_n_rollouts_zero_returns_zeros(self):
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        out = continuation_value(env, _profile(env), _ctx(env, n_rollouts=0))
        np.testing.assert_array_equal(out, np.zeros(env.n_players))

    def test_large_sample_finite_zero_sum(self):
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        out = continuation_value(env, _profile(env), _ctx(env, n_rollouts=50, seed=123))
        assert np.isfinite(out).all()
        assert abs(out.sum()) < 1e-6

    def test_payout_from_env_fold_or_call(self):
        # SB folds pre-flop under FoldOrCall → loses the small blind only.
        env = _full_deck_env(); _stub_lut(env)
        out = continuation_value(
            env, _profile(env), _ctx(env, policies=_policies(FoldOrCallPolicy), n_rollouts=3)
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
        a = continuation_value(env, _profile(env), _ctx(env, seed=7, n_rollouts=8))
        np.random.seed(0)
        b = continuation_value(env, _profile(env), _ctx(env, seed=7, n_rollouts=8))
        np.testing.assert_array_equal(a, b)

    def test_different_ctx_seed_diverges(self):
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        a = continuation_value(env, _profile(env), _ctx(env, seed=1, n_rollouts=20))
        b = continuation_value(env, _profile(env), _ctx(env, seed=2, n_rollouts=20))
        assert not np.array_equal(a, b)


class TestNoResampling:

    def test_every_rollout_uses_frontier_concrete_holes(self, monkeypatch):
        env = _full_deck_env(n_players=3); _stub_lut(env); _to_flop(env)
        expected = [tuple(int(c) for c in env.players[i].cards) for i in range(3)]
        recorded = _spy_with_hole_cards(monkeypatch)
        ctx = SubgameContext.from_runtime(
            env=env, my_seat=0, my_hole=expected[0],
            ranges={1: np.ones(env.n_combos, dtype=np.float32),
                    2: np.ones(env.n_combos, dtype=np.float32)},
            folded_ranges={}, leaf=LeafConfig(policies=_policies(FoldOrCallPolicy), n_rollouts=8),
            rng=np.random.default_rng(3),
        )
        continuation_value(env, _profile(env), ctx)
        assert recorded
        for holes in recorded:
            assert holes == expected  # identity every rollout — no resampling


class TestProfileDrivesBias:

    def test_seat_uses_its_profile_bias(self):
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        pol = _policies()
        profile = {0: "raise", 1: "fold"}
        continuation_value(env, profile, _ctx(env, policies=pol, n_rollouts=6))
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
        continuation_value(env, _profile(env), _ctx(env, n_rollouts=4))
        assert seen_flags  # the rollout did query a policy
        assert all(seen_flags)  # always for_blueprint=True


class TestDecisionFreeEquityFlag:

    def _flop_allin_frontier(self, seed):
        np.random.seed(seed)
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        return env

    def test_flag_on_equals_exact_runout_equity(self):
        # AllIn profile on the flop → decision-free 2-card runout.  Flag-on
        # value must equal the env's exact runout_equity (board-prefix fixed by
        # the frontier, so reshuffling the undealt deck cannot change it).
        env = self._flop_allin_frontier(7)
        # Reference: step the all-in directly and read exact equity.
        ref_env = copy.deepcopy(env)
        ref_env.step_in_place("all_in")
        assert ref_env.is_decision_free
        ref = ref_env.runout_equity()
        out = continuation_value(
            env, _profile(env),
            _ctx(env, policies=_policies(AllInPolicy), n_rollouts=5, use_equity=True),
        )
        for i in range(env.n_players):
            assert abs(out[i] - ref[i]) < 1e-9

    def test_runout_cache_computes_once_per_snapshot(self, monkeypatch):
        # All rollouts go all-in at the same flop frontier → one distinct
        # _runout_info, so the exact integration runs once, not once per rollout.
        env = self._flop_allin_frontier(7)
        calls = {"n": 0}
        original = PokerEnv.runout_equity

        def spy(self, *a, **k):
            calls["n"] += 1
            return original(self, *a, **k)

        monkeypatch.setattr(PokerEnv, "runout_equity", spy)
        out = continuation_value(
            env, _profile(env),
            _ctx(env, policies=_policies(AllInPolicy), n_rollouts=20, use_equity=True),
        )
        assert calls["n"] == 1
        # The cached value still equals the exact runout equity.
        ref_env = copy.deepcopy(env)
        ref_env.step_in_place("all_in")
        ref = ref_env.runout_equity()
        for i in range(env.n_players):
            assert abs(out[i] - ref[i]) < 1e-9

    def test_flag_on_is_runout_rng_independent(self):
        # With a deterministic all-in line and exact equity, the value does not
        # depend on the global board-shuffle seed (the runout is integrated, not
        # sampled).  Holes/flop are fixed; only the board-deal RNG varies.
        base = self._fixed_frontier()
        results = []
        for s in range(4):
            np.random.seed(2000 + s)
            results.append(continuation_value(
                base, _profile(base),
                _ctx(base, policies=_policies(AllInPolicy), n_rollouts=3, use_equity=True),
            ))
        for r in results[1:]:
            np.testing.assert_array_equal(results[0], r)

    def test_flag_off_samples_and_can_differ(self):
        # Flag-off uses the env's single sampled runout → varies with the
        # board-deal RNG, and differs from the exact flag-on value.
        base = self._fixed_frontier()
        ref_env = copy.deepcopy(base); ref_env.step_in_place("all_in")
        exact = ref_env.runout_equity()
        sampled = []
        for s in range(8):
            np.random.seed(3000 + s)
            sampled.append(continuation_value(
                base, _profile(base),
                _ctx(base, policies=_policies(AllInPolicy), n_rollouts=1, use_equity=False),
            ))
        # At least one sampled single-board value differs from the exact mean.
        assert any(abs(v[0] - exact[0]) > 1e-6 for v in sampled)

    def test_flag_is_noop_when_no_allin_occurs(self):
        # A check/call-down reaches a normal complete-board showdown — never a
        # decision-free runout — so flag on and off must agree bit-for-bit.
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        pol_on = _policies(CheckCallPolicy)
        pol_off = _policies(CheckCallPolicy)
        np.random.seed(5)
        on = continuation_value(env, _profile(env),
                                _ctx(env, policies=pol_on, n_rollouts=4, seed=8, use_equity=True))
        np.random.seed(5)
        off = continuation_value(env, _profile(env),
                                 _ctx(env, policies=pol_off, n_rollouts=4, seed=8, use_equity=False))
        np.testing.assert_array_equal(on, off)

    def test_three_way_decision_free_uses_exact_equity(self):
        # An all-in line in a 3-way subgame (side pots) takes the exact
        # board-average; flag-on equals the env's runout_equity.
        np.random.seed(21)
        env = _full_deck_env(n_players=3); _stub_lut(env); _to_flop(env)
        ref_env = copy.deepcopy(env)
        # Drive the same all-in line the AllInPolicy would on the flop.
        guard = 0
        while not ref_env.is_terminal and guard < 8:
            legal = [a for a in ref_env.legal_actions if a is not None]
            ref_env.step_in_place("all_in" if "all_in" in legal else legal[0])
            guard += 1
        if not ref_env.is_decision_free:
            pytest.skip("flop line did not reach a decision-free runout")
        ref = ref_env.runout_equity()
        ctx = SubgameContext.from_runtime(
            env=env, my_seat=0, my_hole=tuple(int(c) for c in env.players[0].cards),
            ranges={1: np.ones(env.n_combos, dtype=np.float32),
                    2: np.ones(env.n_combos, dtype=np.float32)},
            folded_ranges={},
            leaf=LeafConfig(policies=_policies(AllInPolicy), n_rollouts=5,
                            use_decision_free_equity=True),
            rng=np.random.default_rng(0),
        )
        out = continuation_value(env, _profile(env), ctx)
        for i in range(3):
            assert abs(out[i] - ref[i]) < 1e-9

    def _fixed_frontier(self) -> PokerEnv:
        # A frontier with deterministic holes/flop so only the runout varies.
        np.random.seed(99)
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        return env


class TestSharedRunoutCache:
    """The optional ``runout_cache`` extends the per-call runout memo across
    calls (§6.4.2): a leaf's four bias profiles — and the solver's forced-runout
    terminal — share one integration per distinct ``(holes, snapshot)``."""

    def _flop_allin_frontier(self, seed=7) -> PokerEnv:
        np.random.seed(seed)
        env = _full_deck_env(); _stub_lut(env); _to_flop(env)
        return env

    def test_shared_dict_integrates_once_across_calls(self, monkeypatch):
        # Two continuation_value calls with DIFFERENT profiles but the same
        # frontier holes share one exact integration via a passed-in dict.
        env = self._flop_allin_frontier()
        calls = {"n": 0}
        original = PokerEnv.runout_equity

        def spy(self, *a, **k):
            calls["n"] += 1
            return original(self, *a, **k)

        monkeypatch.setattr(PokerEnv, "runout_equity", spy)
        shared: dict = {}
        kw = dict(policies=_policies(AllInPolicy), n_rollouts=20, use_equity=True)
        out_a = continuation_value(
            env, _profile(env, "none"), _ctx(env, **kw), runout_cache=shared
        )
        out_b = continuation_value(
            env, _profile(env, "raise"), _ctx(env, **kw), runout_cache=shared
        )
        # Same (holes, snapshot) over 40 rollouts across two calls → ONE integration.
        assert calls["n"] == 1
        assert len(shared) == 1
        ref_env = copy.deepcopy(env); ref_env.step_in_place("all_in")
        ref = ref_env.runout_equity()
        for out in (out_a, out_b):
            for i in range(env.n_players):
                assert abs(out[i] - ref[i]) < 1e-9

    def test_cache_key_includes_holes(self):
        # The shared key carries the all-seat holes, so two frontiers with
        # different holes never collide (distinct entries, both exact).
        env_a = self._flop_allin_frontier(7)
        env_b = self._flop_allin_frontier(8)
        holes_a = tuple(tuple(int(c) for c in env_a.players[s].cards) for s in range(2))
        holes_b = tuple(tuple(int(c) for c in env_b.players[s].cards) for s in range(2))
        if holes_a == holes_b:
            pytest.skip("seeds happened to draw identical holes")
        shared: dict = {}
        kw = dict(policies=_policies(AllInPolicy), n_rollouts=4, use_equity=True)
        continuation_value(env_a, _profile(env_a), _ctx(env_a, **kw), runout_cache=shared)
        continuation_value(env_b, _profile(env_b), _ctx(env_b, **kw), runout_cache=shared)
        assert len(shared) == 2  # no collision across holes

    def test_cache_param_does_not_change_value(self):
        # Supplying a cache must never change the result (runout is exact /
        # rng-independent); equal bit-for-bit to the uncached call.
        env = self._flop_allin_frontier()
        kw = dict(policies=_policies(AllInPolicy), n_rollouts=5, use_equity=True)
        np.random.seed(5)
        without = continuation_value(env, _profile(env), _ctx(env, **kw))
        np.random.seed(5)
        withcache = continuation_value(env, _profile(env), _ctx(env, **kw), runout_cache={})
        np.testing.assert_array_equal(without, withcache)
