"""Tests for the compact binary info-set key encoding (v2).

Covers :func:`environment.poker_env.encode_info_set` and the two builders
(:meth:`PokerEnv._compute_info_set` / :meth:`PokerEnv._blueprint_info_set`):
injectivity, the per-street alphabet, and — the load-bearing inference
invariant — canonicalisation equivalence (an off-tree raise's pseudo-harmonic
snap encodes byte-identically to its on-tree neighbour).
"""

from collections import defaultdict

import numpy as np
import pytest

from environment.action_space import CANONICAL_ACTIONS
from environment.player import Player
from environment.poker_env import (
    PokerEnv,
    encode_info_set,
    _ACTION_BYTE,
    _INFO_SET_DEFAULT,
)


def _env(n_players: int = 2, low: int = 2, high: int = 14, seed: int = 0):
    np.random.seed(seed)
    return PokerEnv(
        players=[Player(i, 10000) for i in range(n_players)],
        low_card_rank=low,
        high_card_rank=high,
    )


def _stub_lut(env):
    env.card_info_lut = defaultdict(lambda: defaultdict(lambda: 0))
    return env


# ---------------------------------------------------------------------------
# Alphabet
# ---------------------------------------------------------------------------


class TestAlphabet:
    def test_alphabet_covers_all_canonical_actions(self):
        stage_of = {0: "pre_flop", 1: "flop", 2: "turn", 3: "river"}
        for r in range(4):
            stage = stage_of[r]
            table = _ACTION_BYTE[stage]
            assert table["skip"] == 0
            for i, action in enumerate(CANONICAL_ACTIONS[r]):
                assert table[action] == i + 1
            # No stray entries beyond canonical actions + skip.
            assert set(table) == set(CANONICAL_ACTIONS[r]) | {"skip"}


# ---------------------------------------------------------------------------
# encode_info_set injectivity
# ---------------------------------------------------------------------------


class TestEncoderInjectivity:
    def test_equal_inputs_equal_bytes(self):
        h = [("pre_flop", ["call", "raise:1.0", "call"])]
        assert encode_info_set(7, h) == encode_info_set(7, [("pre_flop", ["call", "raise:1.0", "call"])])

    def test_cluster_changes_key(self):
        h = [("pre_flop", ["call"])]
        assert encode_info_set(5, h) != encode_info_set(6, h)

    def test_action_changes_key(self):
        assert encode_info_set(5, [("pre_flop", ["raise:1.0"])]) != \
            encode_info_set(5, [("pre_flop", ["raise:2.0"])])

    def test_skip_runs_are_distinct(self):
        assert encode_info_set(5, [("flop", ["skip", "call"])]) != \
            encode_info_set(5, [("flop", ["call"])])

    def test_stage_boundary_is_distinct(self):
        # Same tokens, different stage split → different keys.
        a = encode_info_set(5, [("pre_flop", ["call", "call"])])
        b = encode_info_set(5, [("pre_flop", ["call"]), ("flop", ["call"])])
        assert a != b

    def test_random_fuzz_injective(self):
        rng = np.random.default_rng(0)
        stages = ["pre_flop", "flop", "turn", "river"]
        seen = {}
        for _ in range(4000):
            cluster = int(rng.integers(0, 250))
            n_stages = int(rng.integers(1, 5))
            history = []
            for stage in stages[:n_stages]:
                toks = list(CANONICAL_ACTIONS[stages.index(stage)]) + ["skip"]
                k = int(rng.integers(1, 5))
                history.append((stage, [toks[int(rng.integers(0, len(toks)))] for _ in range(k)]))
            key = encode_info_set(cluster, history)
            canonical = (cluster, tuple((s, tuple(a)) for s, a in history))
            if key in seen:
                assert seen[key] == canonical, f"collision: {seen[key]} vs {canonical}"
            else:
                seen[key] = canonical


# ---------------------------------------------------------------------------
# Canonicalisation equivalence (the inference invariant)
# ---------------------------------------------------------------------------


def _play_to_flop_with(env, flop_action):
    env.step_in_place("call")
    env.step_in_place("call")
    assert env.betting_round == 1
    if flop_action not in env.legal_actions:
        assert env.inject_action(flop_action) is True
    env.step_in_place(flop_action)
    return env


class TestCanonicalizationEquivalence:
    def test_off_tree_blueprint_key_equals_on_tree(self):
        # flop grid first_raise=[0.5,1.0,1.5]; 0.6 snaps to 0.5.
        off = _stub_lut(_env(seed=3))
        _play_to_flop_with(off, "raise:0.6")
        on = _stub_lut(_env(seed=3))
        _play_to_flop_with(on, "raise:0.5")
        combo = (int(off.combo_cards[0, 0]), int(off.combo_cards[0, 1]))
        # Byte-identical: off-tree blueprint key == on-tree raw key.
        assert off._blueprint_info_set(combo) == on._compute_info_set(combo)
        # And the raw off-tree key differs (off-tree fraction present).
        assert off._compute_info_set(combo) != on._compute_info_set(combo)

    def test_on_tree_blueprint_is_noop(self):
        env = _stub_lut(_env(seed=1))
        _play_to_flop_with(env, "raise:1.0")  # on-tree
        combo = (int(env.combo_cards[0, 0]), int(env.combo_cards[0, 1]))
        assert env._blueprint_info_set(combo) == env._compute_info_set(combo)


# ---------------------------------------------------------------------------
# Builders return bytes / sentinel
# ---------------------------------------------------------------------------


class TestBuilders:
    def test_compute_info_set_returns_bytes(self):
        env = _stub_lut(_env())
        combo = (int(env.combo_cards[0, 0]), int(env.combo_cards[0, 1]))
        assert isinstance(env._compute_info_set(combo), bytes)

    def test_default_sentinel_on_missing_lut_at_terminal(self):
        env = _env()
        env.card_info_lut = {}  # empty LUT
        env._betting_stage = "show_down"
        combo = (int(env.combo_cards[0, 0]), int(env.combo_cards[0, 1]))
        assert env._compute_info_set(combo) == _INFO_SET_DEFAULT

    def test_info_set_fields_none_on_missing_lut_at_terminal(self):
        env = _env()
        env.card_info_lut = {}
        env._betting_stage = "show_down"
        combo = (int(env.combo_cards[0, 0]), int(env.combo_cards[0, 1]))
        assert env.info_set_fields(combo) is None
