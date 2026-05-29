"""Tests for :mod:`poker_ai.search.translation`."""

import math

import pytest

from environment.player import Player
from environment.poker_env import (
    MAX_RAISES_PER_ROUND,
    RAISE_SIZES_BY_STAGE,
    PokerEnv,
)
from poker_ai.search.translation import (
    Classification,
    abstract_to_chips,
    canonical_raise_fractions,
    classify_observed,
)


def _env(low: int = 10, high: int = 14, n_players: int = 2):
    return PokerEnv(
        players=[Player(i, 10000) for i in range(n_players)],
        low_card_rank=low,
        high_card_rank=high,
    )


class TestCanonicalRaiseFractions:

    def test_pre_flop_matches_env_sizes(self):
        env = _env()
        env_strs = env._get_available_raise_sizes()
        expected = [
            float(s.split(":", 1)[1])
            for s in env_strs
            if s.startswith("raise:")
        ]
        assert canonical_raise_fractions(env) == expected

    def test_empty_when_max_raises_reached(self):
        env = _env()
        env._n_raises = MAX_RAISES_PER_ROUND
        assert canonical_raise_fractions(env) == []

    def test_empty_when_inactive_player(self):
        env = _env()
        env.current_player._is_active = False
        assert canonical_raise_fractions(env) == []

    def test_empty_when_call_meets_stack(self):
        # Drop actor's stack to call amount so n_chips_to_call >= stack.
        env = _env()
        biggest_bet = max(p.n_bet_chips for p in env.players)
        n_to_call = biggest_bet - env.current_player.n_bet_chips
        env.current_player.n_chips = n_to_call
        assert canonical_raise_fractions(env) == []


class TestClassifyObservedExact:

    def test_all_in_observation(self):
        env = _env()
        chips = env.current_player.n_chips
        cls, action = classify_observed(env, chips)
        assert cls is Classification.ON_TREE
        assert action == "all_in"

    def test_canonical_exact_chip_match(self):
        env = _env()
        fractions = canonical_raise_fractions(env)
        assert fractions, "need a playable canonical fraction"
        f = fractions[0]
        chips = env._compute_raise_chip_amount(f, enforce_minimum=True)
        cls, action = classify_observed(env, chips)
        assert cls is Classification.ON_TREE
        assert action == f"raise:{f}"

    def test_all_in_priority_over_canonical_match(self):
        # If chip_amount happens to equal both stack AND a canonical
        # clamp, the all-in branch wins (it's tested first).
        env = _env()
        # Force stack to equal a canonical clamp.
        f = canonical_raise_fractions(env)[0]
        chips = env._compute_raise_chip_amount(f, enforce_minimum=True)
        env.current_player.n_chips = chips
        cls, action = classify_observed(env, chips)
        assert cls is Classification.ON_TREE
        assert action == "all_in"


class TestClassifyObservedNearAndOff:

    def test_within_tolerance_snaps(self):
        env = _env()
        # f_obs = 1.05 * pot — nearest canonical is 1.0, dist = 0.05.
        chips = math.ceil(1.05 * env.pot_size)
        cls, action = classify_observed(env, chips)
        assert cls is Classification.NEAR_TREE
        assert action == "raise:1.0"

    def test_beyond_tolerance_goes_off_tree(self):
        env = _env()
        # 4.5 * pot — nearest canonical is 3.0, dist = 0.5 > 0.15.
        chips = math.ceil(4.5 * env.pot_size)
        cls, action = classify_observed(env, chips)
        assert cls is Classification.OFF_TREE
        assert action.startswith("raise:")
        # And the env will accept the same string as an injection.
        assert env.inject_action(action) is True

    def test_tolerance_boundary_inclusive(self):
        # Pick chip_amount past the highest canonical (3.0) so it
        # can fall in NEAR/OFF; use tol = actual dist; result must be
        # NEAR_TREE (boundary is inclusive).
        env = _env()
        chips = 517  # 517/150=3.4467 -> dist 0.1489 from f_near=3.0
        f_obs = chips / env.pot_size
        dist = abs(f_obs - 3.0) / 3.0
        cls, action = classify_observed(env, chips, tol=dist)
        assert cls is Classification.NEAR_TREE
        assert action == "raise:3.0"

    def test_just_inside_default_tolerance(self):
        env = _env()
        # chip=517 -> f_obs=3.4467, dist 0.1489 < 0.15 from f_near=3.0.
        cls, action = classify_observed(env, 517)
        assert cls is Classification.NEAR_TREE
        assert action == "raise:3.0"

    def test_just_outside_default_tolerance(self):
        env = _env()
        # chip=518 -> f_obs=3.4533, dist 0.1511 > 0.15 from f_near=3.0.
        cls, action = classify_observed(env, 518)
        assert cls is Classification.OFF_TREE

    def test_off_tree_string_has_stable_precision(self):
        env = _env()
        chips = math.ceil(4.5 * env.pot_size)
        _, action = classify_observed(env, chips)
        fraction_str = action.split(":", 1)[1]
        parsed = float(fraction_str)
        # At most 4 decimals: round-trip through round() is a no-op.
        assert round(parsed, 4) == parsed

    def test_no_canonical_raises_off_tree(self):
        env = _env()
        env._n_raises = MAX_RAISES_PER_ROUND
        # Pick a non-stack chip amount.
        chips = env.current_player.n_chips // 2
        cls, _ = classify_observed(env, chips)
        assert cls is Classification.OFF_TREE


class TestAbstractToChips:

    def test_fold_zero(self):
        env = _env()
        assert abstract_to_chips(env, "fold") == 0

    def test_call_matches_engine(self):
        env = _env()
        biggest = max(p.n_bet_chips for p in env.players)
        expected = biggest - env.current_player.n_bet_chips
        assert abstract_to_chips(env, "call") == expected

    def test_all_in_full_stack(self):
        env = _env()
        assert abstract_to_chips(env, "all_in") == env.current_player.n_chips

    def test_raise_round_trip(self):
        env = _env()
        for f in canonical_raise_fractions(env):
            chips = env._compute_raise_chip_amount(f, enforce_minimum=True)
            assert abstract_to_chips(env, f"raise:{f}") == chips

    def test_unknown_action_raises(self):
        env = _env()
        with pytest.raises(ValueError):
            abstract_to_chips(env, "shove")

    def test_raise_with_bad_fraction_raises(self):
        env = _env()
        with pytest.raises(ValueError):
            abstract_to_chips(env, "raise:abc")


class TestEndToEnd:

    def test_off_tree_injection_round_trip(self):
        # Observe an off-tree chip amount, inject the returned string,
        # then check it appears in env.legal_actions.
        env = _env()
        chips = math.ceil(4.5 * env.pot_size)
        cls, action = classify_observed(env, chips)
        assert cls is Classification.OFF_TREE
        assert env.inject_action(action) is True
        assert action in env.legal_actions

    def test_on_tree_chip_round_trip(self):
        # canonical f -> chips -> classify (ON_TREE) -> abstract_to_chips
        # returns the original chip count.
        env = _env()
        f = canonical_raise_fractions(env)[0]
        chips = env._compute_raise_chip_amount(f, enforce_minimum=True)
        cls, action = classify_observed(env, chips)
        assert cls is Classification.ON_TREE
        assert abstract_to_chips(env, action) == chips
