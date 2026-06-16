"""Tests for :mod:`poker_ai.search.context`."""

import dataclasses

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv
from poker_ai.search.context import SubgameContext, _board_compatible_mask


def _env(low: int = 2, high: int = 14, n_players: int = 2):
    return PokerEnv(
        players=[Player(i, 10000) for i in range(n_players)],
        low_card_rank=low,
        high_card_rank=high,
    )


def _ctx_from(env, leaf=None, rng=None):
    if rng is None:
        # Derive from the global state, which the autouse ``_seeded``
        # fixture reseeds per trial (see conftest.py).
        rng = np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))
    return SubgameContext.from_runtime(
        env=env,
        my_seat=0,
        my_hole=tuple(env.players[0].cards),
        opponent_ranges={1: np.ones(env.n_combos, dtype=np.float32)},
        leaf=leaf if leaf is not None else object(),
        rng=rng,
    )


class TestFromRuntime:

    def test_basic_fields_passed_through(self):
        env = _env()
        my_hole = tuple(env.players[0].cards)
        opponent_ranges = {1: np.full(env.n_combos, 0.5, dtype=np.float32)}
        rng = np.random.default_rng(42)
        leaf = object()
        ctx = SubgameContext.from_runtime(
            env=env,
            my_seat=0,
            my_hole=my_hole,
            opponent_ranges=opponent_ranges,
            leaf=leaf,
            rng=rng,
        )
        assert ctx.my_seat == 0
        assert ctx.my_hole == my_hole
        # opponent_ranges is wrapped read-only (not the same object),
        # but the contents must round-trip exactly.
        assert set(ctx.opponent_ranges) == set(opponent_ranges)
        np.testing.assert_array_equal(ctx.opponent_ranges[1], opponent_ranges[1])
        assert ctx.leaf is leaf
        assert ctx.rng is rng

    def test_street_at_root_matches_env(self):
        env = _env()
        ctx = _ctx_from(env)
        assert ctx.street_at_root == env.betting_round

    def test_board_compatible_field_populated(self):
        env = _env()
        ctx = _ctx_from(env)
        assert ctx.board_compatible.dtype == bool
        assert ctx.board_compatible.shape == (env.n_combos,)

    def test_from_runtime_on_flop_env(self):
        # Walk HU pre-flop to flop via two calls; assert from_runtime
        # reports the right street and a non-trivial board mask.
        env = _env()
        env = env.apply_action("call")
        env = env.apply_action("call")
        assert env.betting_round == 1
        assert len(env.community_cards) == 3
        ctx = _ctx_from(env)
        assert ctx.street_at_root == 1
        # At least the three board cards exclude some combos.
        assert ctx.board_compatible.sum() < env.n_combos
        # And no combo containing a board card survives.
        board = set(int(c) for c in env.community_cards)
        for i, (c0, c1) in enumerate(env.combo_cards):
            uses_board = int(c0) in board or int(c1) in board
            if uses_board:
                assert not ctx.board_compatible[i]


class TestFrozen:

    def test_cannot_mutate_field(self):
        ctx = _ctx_from(_env())
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.my_seat = 9  # type: ignore[misc]

    def test_cannot_assign_into_opponent_ranges(self):
        # MappingProxyType blocks __setitem__ — the solver can't add
        # or replace a seat's range mid-search.
        ctx = _ctx_from(_env())
        with pytest.raises(TypeError):
            ctx.opponent_ranges[2] = np.zeros(10, dtype=np.float32)  # type: ignore[index]

    def test_cannot_pop_from_opponent_ranges(self):
        ctx = _ctx_from(_env())
        with pytest.raises(AttributeError):
            ctx.opponent_ranges.pop(1)  # type: ignore[attr-defined]

    def test_cannot_mutate_a_range_array(self):
        ctx = _ctx_from(_env())
        with pytest.raises(ValueError):
            ctx.opponent_ranges[1][0] = 0.0

    def test_cannot_mutate_board_compatible(self):
        ctx = _ctx_from(_env())
        with pytest.raises(ValueError):
            ctx.board_compatible[0] = False

    def test_input_dict_unchanged_by_construction(self):
        # Defensive copies inside from_runtime mean the caller's
        # original dict / arrays stay writable.
        env = _env()
        weights = np.full(env.n_combos, 0.5, dtype=np.float32)
        input_dict = {1: weights}
        _ = SubgameContext.from_runtime(
            env=env,
            my_seat=0,
            my_hole=tuple(env.players[0].cards),
            opponent_ranges=input_dict,
            leaf=object(),
            rng=np.random.default_rng(0),
        )
        # Caller's references remain mutable.
        weights[0] = 0.0
        input_dict[2] = np.zeros(env.n_combos, dtype=np.float32)
        assert weights[0] == 0.0


class TestBoardCompatibleMask:

    def test_no_community_all_true(self):
        env = _env()
        assert env.community_cards == ()
        mask = _board_compatible_mask(env)
        assert mask.shape == (env.n_combos,)
        assert mask.all()

    def test_excludes_combos_using_board_cards(self):
        # Force a board on a short deck to avoid 1326-combo enumeration.
        env = _env(low=10, high=14)
        env.community_cards = (int(env.combo_cards[0, 0]),)
        mask = _board_compatible_mask(env)
        # Any combo containing card[0] must be False.
        board_card = int(env.combo_cards[0, 0])
        for i, (c0, c1) in enumerate(env.combo_cards):
            uses_board = int(c0) == board_card or int(c1) == board_card
            assert mask[i] == (not uses_board)

    def test_dtype_and_shape(self):
        env = _env(low=10, high=14)
        env.community_cards = (
            int(env.combo_cards[0, 0]),
            int(env.combo_cards[0, 1]),
            int(env.combo_cards[3, 0]),
        )
        mask = _board_compatible_mask(env)
        assert mask.dtype == bool
        assert mask.shape == (env.n_combos,)
