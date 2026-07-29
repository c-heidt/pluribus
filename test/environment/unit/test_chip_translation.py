"""Tests for the env's chip <-> action conversion API.

Covers :meth:`PokerEnv.canonical_raise_fractions`,
:meth:`PokerEnv.chips_to_add`, and :meth:`PokerEnv.string_for_chips`.
"""

import math

import pytest

from environment.player import Player
from environment.poker_env import PokerEnv


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
            float(s.split(":", 1)[1]) for s in env_strs if s.startswith("raise:")
        ]
        assert env.canonical_raise_fractions() == expected

    def test_empty_when_max_raises_reached(self):
        env = _env()
        env._n_raises = env._max_raises_per_round
        assert env.canonical_raise_fractions() == []

    def test_empty_when_inactive_player(self):
        env = _env()
        env.current_player._is_active = False
        assert env.canonical_raise_fractions() == []

    def test_empty_when_call_meets_stack(self):
        env = _env()
        biggest_bet = max(p.n_bet_chips for p in env.players)
        n_to_call = biggest_bet - env.current_player.n_bet_chips
        env.current_player.n_chips = n_to_call
        assert env.canonical_raise_fractions() == []

    def test_empty_and_agrees_with_legal_actions_facing_lone_all_in(self):
        """Facing a lone all-in, ``legal_actions`` offers no raise, so neither may
        ``canonical_raise_fractions`` (it must not diverge from the real tree).

        Two short stacks shove so the big stack acts with ``n_players_with_moves
        == 1`` — a raise here would only be returned uncalled.
        """
        import numpy as np

        import environment.dynamics as dynamics

        for seed in range(40):
            np.random.seed(seed)
            env = PokerEnv(
                players=[Player(0, 1000), Player(1, 40), Player(2, 55)],
                low_card_rank=11, high_card_rank=14,
            )
            steps = 0
            while not env.is_terminal and steps < 16:
                pi = env.player_i
                legal = [a for a in env.legal_actions if a]
                if pi in (1, 2) and "all_in" in legal:
                    env.step_in_place("all_in")
                elif pi == 0 and dynamics.n_players_with_moves(env) == 1:
                    # The lone-all-in node: assert agreement + emptiness.
                    la = [a for a in env.legal_actions if a]
                    assert not any(a.startswith("raise") for a in la), la
                    assert env.canonical_raise_fractions() == []
                    return
                else:
                    env.step_in_place("call" if "call" in legal else "check")
                steps += 1
        pytest.fail("could not construct a lone-all-in node")


class TestChipsToAdd:

    def test_fold_zero(self):
        env = _env()
        assert env.chips_to_add("fold") == 0

    def test_call_matches_engine(self):
        env = _env()
        biggest = max(p.n_bet_chips for p in env.players)
        expected = biggest - env.current_player.n_bet_chips
        assert env.chips_to_add("call") == expected

    def test_all_in_full_stack(self):
        env = _env()
        assert env.chips_to_add("all_in") == env.current_player.n_chips

    def test_raise_round_trip(self):
        env = _env()
        for f in env.canonical_raise_fractions():
            chips = env._compute_raise_chip_amount(f, enforce_minimum=True)
            assert env.chips_to_add(f"raise:{f}") == chips

    def test_unknown_action_raises(self):
        env = _env()
        with pytest.raises(ValueError):
            env.chips_to_add("shove")

    def test_raise_with_bad_fraction_raises(self):
        env = _env()
        with pytest.raises(ValueError):
            env.chips_to_add("raise:abc")

    def test_raise_with_empty_fraction_raises(self):
        env = _env()
        with pytest.raises(ValueError):
            env.chips_to_add("raise:")

    def test_raise_with_zero_fraction_raises(self):
        env = _env()
        with pytest.raises(ValueError, match="positive finite"):
            env.chips_to_add("raise:0")

    def test_raise_with_negative_fraction_raises(self):
        env = _env()
        with pytest.raises(ValueError, match="positive finite"):
            env.chips_to_add("raise:-2")

    def test_raise_with_infinite_fraction_raises(self):
        env = _env()
        with pytest.raises(ValueError, match="positive finite"):
            env.chips_to_add("raise:inf")

    def test_raise_with_nan_fraction_raises(self):
        env = _env()
        with pytest.raises(ValueError, match="positive finite"):
            env.chips_to_add("raise:nan")

    def test_call_zero_when_actor_is_highest_bettor(self):
        # If the actor's n_bet_chips already equals biggest_bet, the
        # call amount is zero (a "check" in poker terms).
        env = _env()
        biggest = max(p.n_bet_chips for p in env.players)
        env.current_player.n_bet_chips = biggest
        assert env.chips_to_add("call") == 0


class TestStringForChips:

    def test_all_in_when_chips_match_stack(self):
        env = _env()
        assert env.string_for_chips(env.current_player.n_chips) == "all_in"

    def test_canonical_exact_chip_match(self):
        env = _env()
        f = env.canonical_raise_fractions()[0]
        chips = env._compute_raise_chip_amount(f, enforce_minimum=True)
        assert env.string_for_chips(chips) == f"raise:{f}"

    def test_all_in_priority_over_canonical_match(self):
        # If chip_amount happens to equal both stack AND a canonical
        # clamp, the all-in branch wins (it's tested first).
        env = _env()
        f = env.canonical_raise_fractions()[0]
        chips = env._compute_raise_chip_amount(f, enforce_minimum=True)
        env.current_player.n_chips = chips
        assert env.string_for_chips(chips) == "all_in"

    def test_near_canonical_is_off_tree_not_snapped(self):
        # No tolerance snapping any more: chip=470 (f_obs≈3.13, close to
        # canonical 3.0) is returned OFF-TREE verbatim, not snapped to
        # raise:3.0.  Approximation is the translation layer's job, not
        # this chip→string boundary's.
        env = _env()
        s = env.string_for_chips(470)
        assert s != "raise:3.0"
        assert s.startswith("raise:")
        # Off-tree string is *not* in canonical legal_actions until
        # it has been injected.
        assert s not in env.legal_actions

    def test_off_tree_string_has_stable_precision(self):
        env = _env()
        chips = math.ceil(4.5 * env.pot_size)
        s = env.string_for_chips(chips)
        parsed = float(s.split(":", 1)[1])
        # At most 4 decimals: round-trip through round() is a no-op.
        assert round(parsed, 4) == parsed

    def test_no_canonical_raises_off_tree(self):
        env = _env()
        env._n_raises = env._max_raises_per_round
        chips = env.current_player.n_chips // 2
        s = env.string_for_chips(chips)
        assert s.startswith("raise:")
        assert s not in env.legal_actions

    def test_zero_chip_amount_raises(self):
        env = _env()
        with pytest.raises(ValueError, match="positive"):
            env.string_for_chips(0)

    def test_negative_chip_amount_raises(self):
        env = _env()
        with pytest.raises(ValueError, match="positive"):
            env.string_for_chips(-50)

    def test_off_tree_string_carries_observed_fraction(self):
        # Any non-exact, non-all-in chip raise is off-tree and the
        # returned string reflects the OBSERVED fraction (no snap to a
        # nearby canonical size).
        env = _env()
        env.canonical_raise_fractions = lambda: [0.5, 3.0]
        f_obs_target = 1.6
        chips = int(round(f_obs_target * env.pot_size))
        s = env.string_for_chips(chips)
        assert s.startswith("raise:")
        f_parsed = float(s.split(":", 1)[1])
        assert abs(f_parsed - 1.6) < 1e-3, s

    def test_canonical_on_flop_stage(self):
        # Walk to the flop and verify canonical mapping uses the
        # flop's raise grid, not pre-flop's.
        env = _env()
        env.step_in_place("call")
        env.step_in_place("call")
        assert env.betting_round == 1
        fractions = env.canonical_raise_fractions()
        assert fractions, "expected playable flop fractions"
        f = fractions[0]
        chips = env._compute_raise_chip_amount(f, enforce_minimum=True)
        assert env.string_for_chips(chips) == f"raise:{f}"


class TestRuntimeIntegration:

    def test_off_tree_injection_round_trip(self):
        # The runtime's caller pattern: convert chips to a string,
        # inject if not in legal_actions, then apply.
        env = _env()
        chips = math.ceil(4.5 * env.pot_size)
        s = env.string_for_chips(chips)
        assert s not in env.legal_actions
        assert env.inject_action(s) is True
        assert s in env.legal_actions

    def test_on_tree_chip_round_trip(self):
        # Canonical f -> chips -> string (already in legal_actions)
        # -> chips_to_add returns the original chip count.
        env = _env()
        f = env.canonical_raise_fractions()[0]
        chips = env._compute_raise_chip_amount(f, enforce_minimum=True)
        s = env.string_for_chips(chips)
        assert s in env.legal_actions
        assert env.chips_to_add(s) == chips
