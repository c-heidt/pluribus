"""Tests for :mod:`poker_ai.search.context`."""

import dataclasses

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv
from poker_ai.search.context import (
    DepthLimit,
    SubgameContext,
    _board_compatible_mask,
)
from test.abstraction_helpers import advance_to_round


def _env(low: int = 2, high: int = 14, n_players: int = 2):
    return PokerEnv(
        players=[Player(i, 10000) for i in range(n_players)],
        low_card_rank=low,
        high_card_rank=high,
    )


def _ctx_from(env, ranges=None, folded_ranges=None, leaf=None, rng=None):
    if rng is None:
        # Derive from the global state, which the autouse ``_seeded``
        # fixture reseeds per trial (see conftest.py).
        rng = np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))
    if ranges is None:
        # New contract: ranges covers every live seat, incl. the bot.
        ranges = {
            0: np.ones(env.n_combos, dtype=np.float32),
            1: np.ones(env.n_combos, dtype=np.float32),
        }
    if folded_ranges is None:
        folded_ranges = {}
    return SubgameContext.from_runtime(
        env=env,
        my_seat=0,
        my_hole=tuple(env.players[0].cards),
        ranges=ranges,
        folded_ranges=folded_ranges,
        leaf=leaf if leaf is not None else object(),
        rng=rng,
    )


class TestFromRuntime:

    def test_basic_fields_passed_through(self):
        env = _env()
        my_hole = tuple(env.players[0].cards)
        ranges = {
            0: np.full(env.n_combos, 0.25, dtype=np.float32),
            1: np.full(env.n_combos, 0.5, dtype=np.float32),
        }
        folded_ranges = {}
        rng = np.random.default_rng(42)
        leaf = object()
        ctx = SubgameContext.from_runtime(
            env=env,
            my_seat=0,
            my_hole=my_hole,
            ranges=ranges,
            folded_ranges=folded_ranges,
            leaf=leaf,
            rng=rng,
        )
        assert ctx.my_seat == 0
        assert ctx.my_hole == my_hole
        # ranges is wrapped read-only (not the same object), but the
        # contents must round-trip exactly — including the bot's own seat.
        assert set(ctx.ranges) == set(ranges)
        np.testing.assert_array_equal(ctx.ranges[0], ranges[0])
        np.testing.assert_array_equal(ctx.ranges[1], ranges[1])
        assert ctx.leaf is leaf
        assert ctx.rng is rng

    def test_ranges_may_include_my_seat(self):
        # The context is a passive carrier: the bot's own range belongs
        # in ``ranges`` and is carried through unchanged.
        env = _env()
        ctx = _ctx_from(env)
        assert 0 in ctx.ranges  # my_seat

    def test_folded_ranges_carried_through(self):
        env = _env()
        folded = {1: np.full(env.n_combos, 0.3, dtype=np.float32)}
        ctx = _ctx_from(env, ranges={0: np.ones(env.n_combos, dtype=np.float32)},
                        folded_ranges=folded)
        assert set(ctx.folded_ranges) == {1}
        np.testing.assert_array_equal(ctx.folded_ranges[1], folded[1])

    def test_street_at_root_matches_env(self):
        env = _env()
        ctx = _ctx_from(env)
        assert ctx.street_at_root == env.betting_round

    def test_depth_limit_derived_from_env(self):
        env = _env()
        ctx = _ctx_from(env)
        assert isinstance(ctx.depth_limit, DepthLimit)
        assert ctx.depth_limit.street_at_root == env.betting_round
        assert ctx.depth_limit.n_players_at_root == env.n_players_started_round

    def test_board_compatible_field_populated(self):
        env = _env()
        ctx = _ctx_from(env)
        assert ctx.board_compatible.dtype == bool
        assert ctx.board_compatible.shape == (env.n_combos,)

    def test_from_runtime_on_flop_env(self):
        # Walk HU pre-flop to the flop (open + call — the abstraction has no
        # limp); assert from_runtime reports the right street and a
        # non-trivial board mask.
        env = _env()
        advance_to_round(env, 1)
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

    @pytest.mark.parametrize("field", ["ranges", "folded_ranges"])
    def test_cannot_assign_into_mapping(self, field):
        # MappingProxyType blocks __setitem__ — the solver can't add
        # or replace a seat's range mid-search.
        env = _env()
        ctx = _ctx_from(env, folded_ranges={1: np.ones(env.n_combos, dtype=np.float32)})
        with pytest.raises(TypeError):
            getattr(ctx, field)[2] = np.zeros(10, dtype=np.float32)  # type: ignore[index]

    @pytest.mark.parametrize("field", ["ranges", "folded_ranges"])
    def test_cannot_pop_from_mapping(self, field):
        env = _env()
        ctx = _ctx_from(env, folded_ranges={1: np.ones(env.n_combos, dtype=np.float32)})
        with pytest.raises(AttributeError):
            getattr(ctx, field).pop(1)  # type: ignore[attr-defined]

    @pytest.mark.parametrize("field", ["ranges", "folded_ranges"])
    def test_cannot_mutate_a_range_array(self, field):
        env = _env()
        ctx = _ctx_from(env, folded_ranges={1: np.ones(env.n_combos, dtype=np.float32)})
        with pytest.raises(ValueError):
            getattr(ctx, field)[1][0] = 0.0

    def test_cannot_mutate_board_compatible(self):
        ctx = _ctx_from(_env())
        with pytest.raises(ValueError):
            ctx.board_compatible[0] = False

    def test_input_dicts_unchanged_by_construction(self):
        # Defensive copies inside from_runtime mean the caller's
        # original dicts / arrays stay writable.
        env = _env()
        weights = np.full(env.n_combos, 0.5, dtype=np.float32)
        folded_weights = np.full(env.n_combos, 0.3, dtype=np.float32)
        ranges_in = {0: np.ones(env.n_combos, dtype=np.float32), 1: weights}
        folded_in = {2: folded_weights}
        _ = SubgameContext.from_runtime(
            env=env,
            my_seat=0,
            my_hole=tuple(env.players[0].cards),
            ranges=ranges_in,
            folded_ranges=folded_in,
            leaf=object(),
            rng=np.random.default_rng(0),
        )
        # Caller's references remain mutable.
        weights[0] = 0.0
        folded_weights[0] = 0.0
        ranges_in[3] = np.zeros(env.n_combos, dtype=np.float32)
        folded_in[4] = np.zeros(env.n_combos, dtype=np.float32)
        assert weights[0] == 0.0
        assert folded_weights[0] == 0.0


class _StubEnv:
    """Minimal env exposing only what :meth:`DepthLimit.classify` reads.

    ``betting_round`` raises at the terminal stage, mirroring
    :class:`PokerEnv`, so a classify that forgets to short-circuit on
    ``is_terminal`` would error instead of silently misclassifying.
    """

    def __init__(self, is_terminal, betting_round=0, n_raises=0):
        self._is_terminal = is_terminal
        self._betting_round = betting_round
        self._n_raises = n_raises

    @property
    def is_terminal(self):
        return self._is_terminal

    @property
    def betting_round(self):
        if self._is_terminal:
            raise ValueError("betting_round read at terminal stage")
        return self._betting_round

    @property
    def n_raises_this_round(self):
        return self._n_raises


class TestDepthLimit:

    def test_terminal_short_circuits_before_betting_round(self):
        # _StubEnv.betting_round raises at terminal; classify must not
        # reach it.
        dl = DepthLimit(street_at_root=0, n_players_at_root=2)
        assert dl.classify(_StubEnv(is_terminal=True)) == "terminal"

    def test_round1_search_leaf_at_end_of_round_1(self):
        dl = DepthLimit(street_at_root=0, n_players_at_root=3)
        assert dl.classify(_StubEnv(False, betting_round=0)) == "internal"
        assert dl.classify(_StubEnv(False, betting_round=1)) == "leaf"

    def test_multiway_round2_cutoff_at_turn(self):
        dl = DepthLimit(street_at_root=1, n_players_at_root=3)
        # Still on the flop, fewer than two raises → keep recursing.
        assert dl.classify(_StubEnv(False, betting_round=1, n_raises=0)) == "internal"
        assert dl.classify(_StubEnv(False, betting_round=1, n_raises=1)) == "internal"
        # Reaching the turn (start of round 3) is a depth-limit leaf.
        assert dl.classify(_StubEnv(False, betting_round=2)) == "leaf"

    def test_multiway_round2_cutoff_after_second_raise(self):
        dl = DepthLimit(street_at_root=1, n_players_at_root=3)
        assert dl.classify(_StubEnv(False, betting_round=1, n_raises=2)) == "leaf"
        assert dl.classify(_StubEnv(False, betting_round=1, n_raises=3)) == "leaf"

    def test_heads_up_round2_has_no_depth_leaf(self):
        # Extends to end of game — terminal leaves only.
        dl = DepthLimit(street_at_root=1, n_players_at_root=2)
        assert dl.classify(_StubEnv(False, betting_round=1, n_raises=2)) == "internal"
        assert dl.classify(_StubEnv(False, betting_round=2)) == "internal"
        assert dl.classify(_StubEnv(False, betting_round=3)) == "internal"
        assert dl.classify(_StubEnv(is_terminal=True)) == "terminal"

    @pytest.mark.parametrize("street_at_root", [2, 3])
    def test_late_rounds_have_no_depth_leaf(self, street_at_root):
        dl = DepthLimit(street_at_root=street_at_root, n_players_at_root=4)
        assert dl.classify(_StubEnv(False, betting_round=street_at_root)) == "internal"
        assert dl.classify(_StubEnv(False, betting_round=3)) == "internal"
        assert dl.classify(_StubEnv(is_terminal=True)) == "terminal"


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
