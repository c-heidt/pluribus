"""Tests for pseudo-harmonic action translation and history canonicalisation
(docs/subgame_solving.md §6.3).

Covers :meth:`PokerEnv._pseudo_harmonic_prob` /
``_pseudo_harmonic_neighbours`` / ``_translate_fraction``,
``_canonicalize_history`` / ``_blueprint_info_set`` /
``policy_state_for(for_blueprint=...)``.
"""

from collections import defaultdict

import numpy as np
import pytest

from environment.action_space import ACTION_TO_IDX, MAX_ACTIONS_PER_STREET
from environment.player import Player
from environment.poker_env import (
    PokerEnv,
    RAISE_SIZES_BY_STAGE,
)
from poker_ai.blueprint.tree_utils import calculate_strategy_from_row
from poker_ai.search.policy import BlueprintPolicy


def _env(n_players: int = 2, low: int = 2, high: int = 14, seed: int = 0):
    np.random.seed(seed)
    return PokerEnv(
        players=[Player(i, 10000) for i in range(n_players)],
        low_card_rank=low,
        high_card_rank=high,
    )


def _stub_lut(env):
    """Make ``info_set`` / ``_compute_info_set`` resolve to cluster 0."""
    env.card_info_lut = defaultdict(lambda: defaultdict(lambda: 0))


# ---------------------------------------------------------------------------
# Pseudo-harmonic probability + neighbour location
# ---------------------------------------------------------------------------


class TestPseudoHarmonicProb:

    def test_golden_value(self):
        # ((1-0.75)(1+0.5)) / ((1-0.5)(1+0.75)) = 0.375/0.875 = 3/7.
        assert PokerEnv._pseudo_harmonic_prob(0.5, 0.75, 1.0) == pytest.approx(3 / 7)

    def test_prob_one_at_lower_endpoint(self):
        # x -> a: numerator -> (b-a)(1+a), denominator (b-a)(1+a) => 1.
        assert PokerEnv._pseudo_harmonic_prob(0.5, 0.5, 1.0) == pytest.approx(1.0)

    def test_prob_zero_at_upper_endpoint(self):
        assert PokerEnv._pseudo_harmonic_prob(0.5, 1.0, 1.0) == pytest.approx(0.0)


class TestNeighbours:

    def test_brackets_interior(self):
        a, b, p = PokerEnv._pseudo_harmonic_neighbours([0.5, 1.0], 0.75)
        assert (a, b) == (0.5, 1.0)
        assert p == pytest.approx(3 / 7)

    def test_below_smallest_returns_b_only(self):
        assert PokerEnv._pseudo_harmonic_neighbours([0.5, 1.0], 0.3) == (None, 0.5, 0.0)

    def test_above_largest_returns_a_only(self):
        assert PokerEnv._pseudo_harmonic_neighbours([0.5, 1.0], 2.0) == (1.0, None, 1.0)

    def test_exact_hit_is_identity(self):
        assert PokerEnv._pseudo_harmonic_neighbours([0.33, 0.5, 1.0], 0.5) == (0.5, 0.5, 1.0)

    def test_empty_grid(self):
        assert PokerEnv._pseudo_harmonic_neighbours([], 0.7) == (None, None, 1.0)


# ---------------------------------------------------------------------------
# _translate_fraction — the two variants
# ---------------------------------------------------------------------------


class TestTranslateFraction:

    def test_deterministic_picks_below_half_side(self):
        # turn first_raise grid [0.5, 1.0]; x=0.75 -> P_A=3/7 < 0.5 -> B.
        env = _env()
        assert env._translate_fraction(0.75, "turn", 0, randomized=False) == 1.0

    def test_deterministic_picks_above_half_side(self):
        # x=0.55 in [0.5,1.0]: P_A = 3(1-.55)/(1+.55) = 1.35/1.55 ≈ 0.871 -> A.
        env = _env()
        assert env._translate_fraction(0.55, "turn", 0, randomized=False) == 0.5

    def test_deterministic_crossing_at_half(self):
        # P_A crosses 0.5 at x=5/7≈0.7143 on the [0.5,1.0] grid.  Just
        # below the crossing P_A>0.5 -> A; just above P_A<0.5 -> B.  This
        # exercises the `>= 0.5` decision boundary without depending on an
        # exact-0.5 float (unreachable through a real grid).
        env = _env()
        assert env._pseudo_harmonic_prob(0.5, 0.71, 1.0) > 0.5
        assert env._pseudo_harmonic_prob(0.5, 0.72, 1.0) < 0.5
        assert env._translate_fraction(0.71, "turn", 0, randomized=False) == 0.5
        assert env._translate_fraction(0.72, "turn", 0, randomized=False) == 1.0

    def test_deterministic_below_and_above(self):
        env = _env()
        assert env._translate_fraction(0.2, "turn", 0, randomized=False) == 0.5
        assert env._translate_fraction(9.0, "turn", 0, randomized=False) == 1.0

    def test_empty_grid_returns_x(self):
        # subsequent_raise on turn is [1.0]; raise_index>=1 grid has one
        # element -> below/above clamp, never empty.  Force empty via a
        # stage with no cell.
        env = _env()
        assert env._translate_fraction(0.7, "show_down", 0, randomized=False) == 0.7

    def test_randomized_requires_rng(self):
        env = _env()
        with pytest.raises(ValueError, match="rng"):
            env._translate_fraction(0.75, "turn", 0, randomized=True, rng=None)

    def test_randomized_exact_hit_consumes_no_rng(self):
        env = _env()
        rng = np.random.default_rng(123)
        before = rng.bit_generator.state
        out = env._translate_fraction(0.5, "turn", 0, randomized=True, rng=rng)
        assert out == 0.5
        assert rng.bit_generator.state == before  # no draw consumed

    def test_randomized_matches_formula_frequency(self):
        env = _env()
        rng = np.random.default_rng(0)
        n = 40000
        a_count = sum(
            env._translate_fraction(0.75, "turn", 0, randomized=True, rng=rng) == 0.5
            for _ in range(n)
        )
        # Expected P(A) = 3/7 ≈ 0.4286; binomial std ≈ 0.0025, allow 5σ.
        assert abs(a_count / n - 3 / 7) < 0.0125


# ---------------------------------------------------------------------------
# History canonicalisation
# ---------------------------------------------------------------------------


class TestCanonicalizeHistory:

    def test_no_op_on_on_tree_history(self):
        env = _env()
        hist = {
            "pre_flop": ["call", "raise:1.0", "call"],
            "flop": ["raise:1.0", "call"],
        }
        out = env._canonicalize_history(hist)
        assert out == [
            ("pre_flop", ["call", "raise:1.0", "call"]),
            ("flop", ["raise:1.0", "call"]),
        ]

    def test_off_tree_raise_snaps_deterministically(self):
        # flop first_raise grid [0.5,1.0,1.5]; 0.6 in (0.5, 1.0):
        # P_A = ((1.0-0.6)(1+0.5)) / ((1.0-0.5)(1+0.6)) = 0.6/0.8 = 0.75 >= 0.5
        # -> A = 0.5.
        env = _env()
        out = env._canonicalize_history({"flop": ["raise:0.6"]})
        assert out == [("flop", ["raise:0.5"])]

    def test_raise_index_advances_to_subsequent_grid(self):
        # 1.5 is in flop first_raise but NOT in subsequent_raise
        # ([0.5, 1.0]); so the 2nd raise's 1.5 is off-tree and snaps
        # (above-grid -> 1.0), proving the index advanced.
        env = _env()
        out = env._canonicalize_history({"flop": ["raise:1.5", "raise:1.5"]})
        assert out == [("flop", ["raise:1.5", "raise:1.0"])]

    def test_all_in_advances_raise_index(self):
        # subsequent_raise on flop is [0.5, 1.0]; 0.33 is below-grid -> 0.5.
        env = _env()
        out = env._canonicalize_history({"flop": ["all_in", "raise:0.33"]})
        assert out == [("flop", ["all_in", "raise:0.5"])]

    def test_fold_call_skip_pass_through(self):
        env = _env()
        out = env._canonicalize_history({"flop": ["skip", "call", "fold"]})
        assert out == [("flop", ["skip", "call", "fold"])]


# ---------------------------------------------------------------------------
# _blueprint_info_set + policy_state_for(for_blueprint=True)
# ---------------------------------------------------------------------------


def _play_to_flop_with(env, flop_action):
    """Preflop call/call to the flop, then apply ``flop_action`` (injecting
    it first if off-tree).  Leaves the env at the opponent's flop decision."""
    env.step_in_place("call")
    env.step_in_place("call")
    assert env.betting_round == 1
    if flop_action not in env.legal_actions:
        assert env.inject_action(flop_action) is True
    env.step_in_place(flop_action)
    return env


class TestBlueprintInfoSet:

    def test_no_op_equals_compute_info_set_on_tree(self):
        env = _env()
        _stub_lut(env)
        _play_to_flop_with(env, "raise:1.0")  # on-tree
        combo = (int(env.combo_cards[0, 0]), int(env.combo_cards[0, 1]))
        assert env._blueprint_info_set(combo) == env._compute_info_set(combo)

    def test_off_tree_resolves_to_on_tree_key(self):
        # An off-tree raise:0.6 flop history must yield the SAME blueprint
        # key as the env where the canonical neighbour (0.5) was played.
        off = _play_to_flop_with(_stub_and_return(_env(seed=1)), "raise:0.6")
        on = _play_to_flop_with(_stub_and_return(_env(seed=1)), "raise:0.5")
        combo = (int(off.combo_cards[0, 0]), int(off.combo_cards[0, 1]))
        assert off._blueprint_info_set(combo) == on._compute_info_set(combo)
        # And the non-canonicalised key differs (off-tree fraction present).
        assert off._compute_info_set(combo) != on._compute_info_set(combo)

    def test_policy_state_for_blueprint_flag_threads_canonicalisation(self):
        env = _play_to_flop_with(_stub_and_return(_env(seed=2)), "raise:0.6")
        combo = (int(env.combo_cards[0, 0]), int(env.combo_cards[0, 1]))
        ps_bp = env.policy_state_for(combo, for_blueprint=True)
        ps_raw = env.policy_state_for(combo)
        assert ps_bp.info_set == env._blueprint_info_set(combo)
        assert ps_raw.info_set == env._compute_info_set(combo)
        assert ps_bp.info_set != ps_raw.info_set


def _stub_and_return(env):
    _stub_lut(env)
    return env


class TestCanonicalPublicKey:
    """``canonical_public_key``: the public-state key with off-tree raise sizes in
    the history snapped to the blueprint grid (pseudo-harmonic); a strict no-op
    when the history is already on-tree.  Lets a caller that *translated* (did not
    inject) a near-canonical off-tree raise resolve the node in the canonical
    subgame the solver built."""

    def test_no_op_on_tree(self):
        env = _play_to_flop_with(_stub_and_return(_env(seed=1)), "raise:1.0")
        assert env.canonical_public_key == env.public_key

    def test_off_tree_snaps_to_canonical_neighbour(self):
        # The off-tree (0.6) env's canonical key equals the raw key of the env
        # that actually played the canonical neighbour (0.6 -> 0.5).
        off = _play_to_flop_with(_stub_and_return(_env(seed=1)), "raise:0.6")
        on = _play_to_flop_with(_stub_and_return(_env(seed=1)), "raise:0.5")
        assert off.canonical_public_key == on.public_key
        # The raw key still differs — the off-tree fraction is present verbatim.
        assert off.public_key != on.public_key


class _KeyedTable:
    """Returns ``row`` only for one specific info_set key, else None —
    so a lookup 'hits' only when the key matches exactly."""

    def __init__(self, key, row):
        self._key, self._row = key, row

    def get_row_if_exists(self, info_set):
        return self._row if info_set == self._key else None


class _KeyedTables:
    def __init__(self, rows_by_round):
        self.regret = {r: t for r, t in rows_by_round.items()}
        # No average-strategy rows: every lookup misses, so the policy
        # exercises its regret-matching fallback (what these tests target).
        self.strategy = {
            r: _KeyedTable(None, None) for r in rows_by_round
        }


class TestBlueprintLookupHit:
    """End-to-end: canonicalisation makes a blueprint lookup on an
    off-tree history HIT the on-tree regret row instead of falling back
    to uniform."""

    def test_for_blueprint_hits_populated_row(self):
        off = _play_to_flop_with(_stub_and_return(_env(seed=3)), "raise:0.6")
        on = _play_to_flop_with(_stub_and_return(_env(seed=3)), "raise:0.5")
        combo = (int(off.combo_cards[0, 0]), int(off.combo_cards[0, 1]))
        r = 1
        key = on._compute_info_set(combo)
        # Plant a row that is all-mass-on-first-legal so it's clearly
        # non-uniform.
        ps = off.policy_state_for(combo, for_blueprint=True)
        legal = ps.legal_actions
        row = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
        row[ACTION_TO_IDX[r][legal[0]]] = 100
        tables = _KeyedTables({r: _KeyedTable(key, row)})
        policy = BlueprintPolicy(tables)

        biased = policy.strategy(ps)  # for_blueprint -> canonical key -> HIT
        assert biased[0] == pytest.approx(1.0)
        np.testing.assert_allclose(biased[1:], 0.0)

        # Without canonicalisation the off-tree key MISSES -> uniform.
        ps_raw = off.policy_state_for(combo)
        uni = policy.strategy(ps_raw)
        np.testing.assert_allclose(uni, np.full(len(legal), 1.0 / len(legal)))
