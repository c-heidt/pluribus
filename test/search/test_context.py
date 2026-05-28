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
    return SubgameContext.from_runtime(
        env=env,
        my_seat=0,
        my_hole=tuple(env.players[0].cards),
        opponent_ranges={1: np.ones(env.n_combos, dtype=np.float32)},
        leaf=leaf if leaf is not None else object(),
        rng=rng if rng is not None else np.random.default_rng(0),
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
        assert ctx.opponent_ranges is opponent_ranges
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


class TestFrozen:

    def test_cannot_mutate_field(self):
        ctx = _ctx_from(_env())
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.my_seat = 9  # type: ignore[misc]


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
