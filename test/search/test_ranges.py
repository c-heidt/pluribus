"""Tests for :mod:`poker_ai.search.ranges`."""

import warnings
from collections import defaultdict

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv
from poker_ai.search.ranges import (
    RangeTracker,
    _initial_uniform,
    _zero_conflicting,
)


def _env(low: int = 10, high: int = 14, n_players: int = 2):
    return PokerEnv(
        players=[Player(i, 10000) for i in range(n_players)],
        low_card_rank=low,
        high_card_rank=high,
    )


def _stub_lut(env):
    """Make ``env.info_set`` resolve for any (hole, board) combination."""
    env.card_info_lut = defaultdict(lambda: defaultdict(lambda: 0))


def _tracker(env=None, my_seat=0, live_seats=(0, 1), stub_lut=False):
    if env is None:
        env = _env()
    if stub_lut:
        _stub_lut(env)
    my_hole = tuple(int(c) for c in env.players[my_seat].cards)
    return env, RangeTracker(env, my_seat, my_hole, live_seats)


def _env_with_seat1_acting():
    """Pre-stub LUT, then advance the env so player 1 is on action.
    Returned env satisfies ``env.player_i == 1``."""
    env = _env()
    _stub_lut(env)
    env.step_in_place("call")
    assert env.player_i == 1
    return env


def _tracker_seat1(live_seats=(0, 1)):
    """Tracker rooted at an env where seat 1 (the opponent) is to act."""
    env = _env_with_seat1_acting()
    my_hole = tuple(int(c) for c in env.players[0].cards)
    return env, RangeTracker(env, my_seat=0, my_hole=my_hole, live_seats=live_seats)


def _uniform_sigma(n_actions):
    arr = np.full(n_actions, 1.0 / n_actions, dtype=np.float32)
    return lambda _h: arr


class TestInitialUniform:

    def test_excludes_my_hole(self):
        env, tracker = _tracker()
        my_hole = set(int(c) for c in env.players[0].cards)
        r = tracker.range_of(1)
        cc = env.combo_cards
        for i in range(env.n_combos):
            uses_my_hole = int(cc[i, 0]) in my_hole or int(cc[i, 1]) in my_hole
            if uses_my_hole:
                assert r[i] == 0.0
            else:
                assert r[i] > 0.0

    def test_sums_to_one(self):
        _env_, tracker = _tracker()
        np.testing.assert_allclose(tracker.range_of(1).sum(), 1.0, rtol=1e-6)

    def test_excludes_community_cards(self):
        env = _env()
        # Pick a board and a disjoint my_hole.
        board = (int(env.combo_cards[0, 0]), int(env.combo_cards[0, 1]))
        env.community_cards = board
        my_hole = None
        for i in range(env.n_combos):
            cand = (int(env.combo_cards[i, 0]), int(env.combo_cards[i, 1]))
            if set(cand).isdisjoint(board):
                my_hole = cand
                break
        assert my_hole is not None
        tracker = RangeTracker(env, my_seat=0, my_hole=my_hole, live_seats=[0, 1])
        r = tracker.range_of(1)
        forbidden = set(env.community_cards) | set(my_hole)
        for i in range(env.n_combos):
            uses = int(env.combo_cards[i, 0]) in forbidden or int(
                env.combo_cards[i, 1]
            ) in forbidden
            if uses:
                assert r[i] == 0.0
            else:
                assert r[i] > 0.0

    def test_snapshot_includes_my_seat(self):
        _env_, tracker = _tracker(my_seat=0, live_seats=[0, 1])
        snap = tracker.snapshot()
        assert 0 in snap  # the bot's own observer-perspective range
        assert 1 in snap

    def test_my_range_keeps_my_hole_combos(self):
        # The bot's own range is observer perspective: it excludes only
        # board conflicts, so the combo that *is* the bot's actual hand
        # (and every other my_hole-containing combo) keeps positive mass,
        # while every opponent's range zeroes those combos.
        env, tracker = _tracker(my_seat=0, live_seats=[0, 1])
        my_hole = set(int(c) for c in env.players[0].cards)
        my_r = tracker.range_of(0)
        opp_r = tracker.range_of(1)
        cc = env.combo_cards
        saw_my_hole_combo = False
        for i in range(env.n_combos):
            uses_my_hole = int(cc[i, 0]) in my_hole or int(cc[i, 1]) in my_hole
            if uses_my_hole:
                assert my_r[i] > 0.0
                assert opp_r[i] == 0.0
                saw_my_hole_combo = True
        assert saw_my_hole_combo
        np.testing.assert_allclose(my_r.sum(), 1.0, rtol=1e-6)

    def test_dtype_and_shape(self):
        env, tracker = _tracker()
        r = tracker.range_of(1)
        assert r.dtype == np.float32
        assert r.shape == (env.n_combos,)

    def test_observer_perspective_for_nonzero_my_seat(self):
        # Guard the seat-aware branch isn't tied to index 0: with the
        # bot at seat 1, seat 1's own range keeps its hole combos while
        # seat 0 (an opponent) excludes them.
        env, tracker = _tracker(my_seat=1, live_seats=[0, 1])
        my_hole = set(int(c) for c in env.players[1].cards)
        my_r = tracker.range_of(1)
        opp_r = tracker.range_of(0)
        cc = env.combo_cards
        for i in range(env.n_combos):
            uses_my_hole = int(cc[i, 0]) in my_hole or int(cc[i, 1]) in my_hole
            if uses_my_hole:
                assert my_r[i] > 0.0
                assert opp_r[i] == 0.0


class TestOnBoardUpdate:

    def test_zeros_new_cards(self):
        env, tracker = _tracker()
        # Pick a board of 3 cards none of which overlap my hole.
        my_hole = set(int(c) for c in env.players[0].cards)
        board = []
        for i in range(env.n_combos):
            c0, c1 = int(env.combo_cards[i, 0]), int(env.combo_cards[i, 1])
            for c in (c0, c1):
                if c not in my_hole and c not in board:
                    board.append(c)
                if len(board) == 3:
                    break
            if len(board) == 3:
                break
        tracker.on_board_update(tuple(board))
        r = tracker.range_of(1)
        forbidden = my_hole | set(board)
        for i in range(env.n_combos):
            uses = int(env.combo_cards[i, 0]) in forbidden or int(
                env.combo_cards[i, 1]
            ) in forbidden
            if uses:
                assert r[i] == 0.0
        np.testing.assert_allclose(r.sum(), 1.0, rtol=1e-6)

    def test_idempotent(self):
        # Pick a board card guaranteed not to conflict with my_hole so
        # the first call actually does work (not the empty-input no-op
        # path).  Then assert the second call leaves the range
        # invariant up to float-renormalisation drift — `w /= w.sum()`
        # can introduce ~1 ULP of jitter even when the input sums
        # mathematically to 1.0, so bit-exact equality is the wrong
        # check here.
        env, tracker = _tracker()
        my_hole = set(int(c) for c in env.players[0].cards)
        card = next(
            int(env.combo_cards[i, 0])
            for i in range(env.n_combos)
            if int(env.combo_cards[i, 0]) not in my_hole
        )
        tracker.on_board_update((card,))
        first = tracker.range_of(1).copy()
        tracker.on_board_update((card,))
        np.testing.assert_allclose(first, tracker.range_of(1), atol=1e-7)
        # The nonzero pattern must be identical (no drift sneaking
        # mass into a previously-zero combo).
        assert ((first == 0) == (tracker.range_of(1) == 0)).all()


class TestOnAction:

    def test_uniform_sigma_no_change(self):
        env, tracker = _tracker_seat1()
        legal = [a for a in env.legal_actions if a is not None]
        sigma = _uniform_sigma(len(legal))
        prior = tracker.range_of(1).copy()
        tracker.on_action(1, env, legal[0], sigma)
        np.testing.assert_allclose(tracker.range_of(1), prior, atol=1e-6)

    def test_concentrates_on_consistent_combos(self):
        env, tracker = _tracker_seat1()
        legal = [a for a in env.legal_actions if a is not None]
        n_act = len(legal)
        # Group A: even combo indices that have nonzero prior weight.
        prior = tracker.range_of(1).copy()
        group_a = {h for h in range(env.n_combos) if h % 2 == 0 and prior[h] > 0}
        likely = np.zeros(n_act, dtype=np.float32)
        likely[0] = 0.9
        likely[1:] = 0.1 / (n_act - 1)
        unlikely = np.zeros(n_act, dtype=np.float32)
        unlikely[0] = 0.05
        unlikely[1:] = 0.95 / (n_act - 1)

        def sigma(h):
            return likely if h in group_a else unlikely

        mass_a_before = sum(prior[h] for h in group_a)
        tracker.on_action(1, env, legal[0], sigma)
        post = tracker.range_of(1)
        mass_a_after = sum(post[h] for h in group_a)
        assert mass_a_after > mass_a_before
        np.testing.assert_allclose(post.sum(), 1.0, rtol=1e-6)

    def test_uniform_fallback_on_zero_collapse(self):
        env, tracker = _tracker_seat1()
        legal = [a for a in env.legal_actions if a is not None]
        n_act = len(legal)
        zero_for_obs = np.zeros(n_act, dtype=np.float32)
        zero_for_obs[1] = 1.0
        sigma = lambda _h: zero_for_obs
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker.on_action(1, env, legal[0], sigma)
        assert any(
            issubclass(w.category, RuntimeWarning) for w in caught
        ), [str(w.message) for w in caught]
        np.testing.assert_allclose(
            tracker.range_of(1),
            _initial_uniform(env, tuple(env.players[0].cards)),
            rtol=1e-6,
        )

    def test_decision_log_recorded(self):
        env, tracker = _tracker_seat1()
        legal = [a for a in env.legal_actions if a is not None]
        sigma = _uniform_sigma(len(legal))
        before_info = env.info_set
        tracker.on_action(1, env, legal[0], sigma)
        assert tracker._decision_log == [(1, before_info, legal[0])]

    def test_unknown_action_raises_value_error(self):
        env, tracker = _tracker_seat1()
        legal = [a for a in env.legal_actions if a is not None]
        sigma = _uniform_sigma(len(legal))
        with pytest.raises(ValueError):
            tracker.on_action(1, env, "raise:99.9", sigma)

    def test_with_overlay_injected_action(self):
        env, tracker = _tracker_seat1()
        env.inject_action("raise:1.1")
        legal = [a for a in env.legal_actions if a is not None]
        assert "raise:1.1" in legal
        n_act = len(legal)
        injected_idx = legal.index("raise:1.1")
        favouring = np.full(n_act, 0.01, dtype=np.float32)
        favouring[injected_idx] = 1.0 - 0.01 * (n_act - 1)
        prior = tracker.range_of(1).copy()
        tracker.on_action(1, env, "raise:1.1", lambda _h: favouring)
        post = tracker.range_of(1)
        # All originally-nonzero entries remain nonzero, and the
        # distribution stays normalised.
        np.testing.assert_allclose(post.sum(), 1.0, rtol=1e-6)
        assert (post[prior > 0] > 0).all()


class TestOnActionPreconditions:

    def test_actor_mismatch_asserts(self):
        # env.player_i == 0 but caller claims seat=1 acted — must raise.
        env, tracker = _tracker(stub_lut=True)
        legal = [a for a in env.legal_actions if a is not None]
        with pytest.raises(AssertionError):
            tracker.on_action(1, env, legal[0], _uniform_sigma(len(legal)))

    def test_on_action_updates_my_seat(self):
        # The bot's own seat is now tracked; on_action services it like
        # any other seat (round-boundary replay of the bot's own
        # actions Bayes-updates the bot's own range).
        env, tracker = _tracker(stub_lut=True, my_seat=0)
        assert env.player_i == 0
        legal = [a for a in env.legal_actions if a is not None]
        n_act = len(legal)
        prior = tracker.range_of(0).copy()
        likely = np.zeros(n_act, dtype=np.float32)
        likely[0] = 0.9
        likely[1:] = 0.1 / (n_act - 1)
        unlikely = np.zeros(n_act, dtype=np.float32)
        unlikely[0] = 0.05
        unlikely[1:] = 0.95 / (n_act - 1)
        group_a = {h for h in range(env.n_combos) if h % 2 == 0 and prior[h] > 0}

        def sigma(h):
            return likely if h in group_a else unlikely

        mass_before = sum(prior[h] for h in group_a)
        tracker.on_action(0, env, legal[0], sigma)
        post = tracker.range_of(0)
        assert sum(post[h] for h in group_a) > mass_before
        np.testing.assert_allclose(post.sum(), 1.0, rtol=1e-6)

    def test_seat_after_fold_raises_key_error(self):
        env, tracker = _tracker_seat1()
        tracker.on_seat_folded(1)
        legal = [a for a in env.legal_actions if a is not None]
        with pytest.raises(KeyError):
            tracker.on_action(1, env, legal[0], _uniform_sigma(len(legal)))


class TestOnBoardUpdateExtras:

    def test_empty_new_cards_no_op(self):
        env, tracker = _tracker()
        before = tracker.range_of(1).copy()
        # Capture warnings to assert none fired.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker.on_board_update(())
        np.testing.assert_array_equal(before, tracker.range_of(1))
        assert caught == []

    def test_accumulates_across_streets(self):
        env, tracker = _tracker()
        my_hole = set(int(c) for c in env.players[0].cards)
        # Pick three non-overlapping cards: flop subset, then turn.
        picked = []
        for i in range(env.n_combos):
            for c in (int(env.combo_cards[i, 0]), int(env.combo_cards[i, 1])):
                if c not in my_hole and c not in picked:
                    picked.append(c)
                if len(picked) == 4:
                    break
            if len(picked) == 4:
                break
        flop, turn_card = tuple(picked[:3]), picked[3]
        tracker.on_board_update(flop)
        tracker.on_board_update((turn_card,))
        r = tracker.range_of(1)
        forbidden = my_hole | set(flop) | {turn_card}
        for i in range(env.n_combos):
            uses = int(env.combo_cards[i, 0]) in forbidden or int(
                env.combo_cards[i, 1]
            ) in forbidden
            if uses:
                assert r[i] == 0.0

    def test_fallback_excludes_full_known_board(self):
        # B1 regression: force a collapse during on_board_update and
        # assert the rebuilt uniform still excludes the dealt board.
        env, tracker = _tracker()
        my_hole = set(int(c) for c in env.players[0].cards)
        # Concentrate seat 1's range to a single combo (i*).
        i_star = next(
            i for i in range(env.n_combos)
            if tracker.range_of(1)[i] > 0
        )
        w = tracker.range_of(1)
        w[:] = 0.0
        w[i_star] = 1.0
        # Deal a board card that conflicts with i*.
        board_card = int(env.combo_cards[i_star, 0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker.on_board_update((board_card,))
        assert any(issubclass(w_.category, RuntimeWarning) for w_ in caught)
        # The rebuilt range must NOT include combos using board_card.
        r = tracker.range_of(1)
        for i in range(env.n_combos):
            uses_board = (
                int(env.combo_cards[i, 0]) == board_card
                or int(env.combo_cards[i, 1]) == board_card
            )
            if uses_board:
                assert r[i] == 0.0, (
                    f"combo {i} uses dealt board card {board_card}"
                )


class TestMultipleOpponents:

    def test_independent_ranges_three_seats(self):
        env = _env(n_players=3)
        my_hole = tuple(int(c) for c in env.players[0].cards)
        tracker = RangeTracker(env, my_seat=0, my_hole=my_hole, live_seats=[0, 1, 2])
        snap = tracker.snapshot()
        # Both opponents start identical.
        np.testing.assert_array_equal(snap[1], snap[2])
        # Mutate seat 1's range in place; seat 2's unchanged.
        tracker.range_of(1)[:] = 0.0
        np.testing.assert_allclose(tracker.range_of(2).sum(), 1.0, rtol=1e-6)


class TestLiveSeatsTracking:

    def test_my_seat_is_tracked(self):
        env = _env()
        my_hole = tuple(int(c) for c in env.players[0].cards)
        tracker = RangeTracker(
            env, my_seat=0, my_hole=my_hole, live_seats=[0, 1]
        )
        assert 0 in tracker.snapshot()  # bot's own observer range
        assert 1 in tracker.snapshot()

    def test_my_seat_tracked_even_if_omitted_from_live_seats(self):
        env = _env()
        my_hole = tuple(int(c) for c in env.players[0].cards)
        tracker = RangeTracker(
            env, my_seat=0, my_hole=my_hole, live_seats=[1]
        )
        snap = tracker.snapshot()
        assert 0 in snap
        assert 1 in snap


class TestInitialUniformity:

    def test_all_nonzero_entries_equal(self):
        _env_, tracker = _tracker()
        r = tracker.range_of(1)
        nz = r[r > 0]
        np.testing.assert_allclose(nz, nz[0], rtol=1e-6)


class TestRangeOfLive:

    def test_returns_live_reference(self):
        _env_, tracker = _tracker()
        r = tracker.range_of(1)
        r[0] = 0.5
        # Same call later sees the mutation (tracker is sole writer in
        # production; this test just documents the contract).
        assert tracker.range_of(1)[0] == 0.5


class TestOnSeatFolded:

    def test_drops_seat_from_snapshot(self):
        _env_, tracker = _tracker(live_seats=[0, 1])
        tracker.on_seat_folded(1)
        assert 1 not in tracker.snapshot()

    def test_range_of_folded_seat_raises(self):
        _env_, tracker = _tracker(live_seats=[0, 1])
        tracker.on_seat_folded(1)
        with pytest.raises(KeyError):
            tracker.range_of(1)

    def test_double_fold_no_op(self):
        _env_, tracker = _tracker(live_seats=[0, 1])
        tracker.on_seat_folded(1)
        tracker.on_seat_folded(1)  # no error


class TestFoldedSnapshot:
    """``on_seat_folded`` retains the seat's range at fold time in
    a separate ``folded_snapshot()`` channel.  The leaf evaluator
    samples folded seats' holes from this marginal."""

    def test_folded_seat_appears_in_folded_snapshot(self):
        _env_, tracker = _tracker(live_seats=[0, 1])
        tracker.on_seat_folded(1)
        assert 1 in tracker.folded_snapshot()

    def test_retained_range_matches_pre_fold_snapshot(self):
        _env_, tracker = _tracker(live_seats=[0, 1])
        snap_before = tracker.snapshot()
        tracker.on_seat_folded(1)
        retained = tracker.folded_snapshot()[1]
        np.testing.assert_array_equal(retained, snap_before[1])

    def test_unknown_seat_fold_does_not_add_to_folded(self):
        _env_, tracker = _tracker(live_seats=[0, 1])
        tracker.on_seat_folded(99)  # never tracked
        assert 99 not in tracker.folded_snapshot()

    def test_double_fold_does_not_overwrite_retained(self):
        # Once a seat is folded its retained marginal must not be
        # touched by subsequent on_seat_folded calls (which find no
        # live entry and would otherwise insert an empty placeholder).
        _env_, tracker = _tracker(live_seats=[0, 1])
        tracker.on_seat_folded(1)
        retained_first = tracker.folded_snapshot()[1]
        tracker.on_seat_folded(1)
        retained_second = tracker.folded_snapshot()[1]
        np.testing.assert_array_equal(retained_first, retained_second)

    def test_snapshot_excludes_folded_seat(self):
        # The two channels are disjoint after a fold.
        _env_, tracker = _tracker(live_seats=[0, 1])
        tracker.on_seat_folded(1)
        assert 1 not in tracker.snapshot()
        assert 1 in tracker.folded_snapshot()

    def test_folded_snapshot_is_deep_copy(self):
        _env_, tracker = _tracker(live_seats=[0, 1])
        tracker.on_seat_folded(1)
        snap = tracker.folded_snapshot()
        snap[1][0] = 999.0
        # Internal state untouched.
        assert tracker.folded_snapshot()[1][0] != 999.0

    def test_folded_snapshot_empty_when_no_folds(self):
        _env_, tracker = _tracker(live_seats=[0, 1])
        assert tracker.folded_snapshot() == {}


class TestSnapshot:

    def test_is_deep_copy(self):
        _env_, tracker = _tracker()
        snap = tracker.snapshot()
        snap[1][0] = 999.0
        snap[42] = np.zeros(10, dtype=np.float32)
        # Tracker state untouched.
        assert tracker.range_of(1)[0] != 999.0
        assert 42 not in tracker.snapshot()


class TestMyRangeBoardAndFallback:
    """The bot's own range is observer perspective: board-conflict
    zeroing applies, but the bot's hole cards are never removed, so the
    actual-hand combo survives board updates and collapse fallbacks."""

    def _disjoint_board_card(self, env, my_hole):
        """A single deck card that conflicts with neither ``my_hole`` nor
        any existing community card."""
        used = set(int(c) for c in my_hole) | set(
            int(c) for c in env.community_cards
        )
        for c in env.deck._cards.tolist():
            if int(c) not in used:
                return int(c)
        raise AssertionError("no disjoint board card available")

    def test_board_update_keeps_actual_hand_combo(self):
        env, tracker = _tracker(my_seat=0, live_seats=[0, 1])
        my_hole = tuple(int(c) for c in env.players[0].cards)
        actual_idx = env.combo_index[tuple(sorted(my_hole))]
        board_card = self._disjoint_board_card(env, my_hole)
        tracker.on_board_update([board_card])
        my_r = tracker.range_of(0)
        # Combos using the new board card are zeroed in the bot's range,
        cc = env.combo_cards
        for i in range(env.n_combos):
            if board_card in (int(cc[i, 0]), int(cc[i, 1])):
                assert my_r[i] == 0.0
        # but the bot's actual hand (disjoint from the board) survives.
        assert my_r[actual_idx] > 0.0
        np.testing.assert_allclose(my_r.sum(), 1.0, rtol=1e-6)

    def test_fallback_on_my_seat_excludes_board_only(self):
        # Force the bot's own range to collapse, then confirm the
        # rebuilt uniform excludes only the board (the actual-hand combo
        # is kept) — not my_hole.
        env, tracker = _tracker(stub_lut=True, my_seat=0)
        assert env.player_i == 0
        my_hole = tuple(int(c) for c in env.players[0].cards)
        actual_idx = env.combo_index[tuple(sorted(my_hole))]
        legal = [a for a in env.legal_actions if a is not None]
        n_act = len(legal)
        # sigma puts zero mass on the observed action for every combo.
        zero_for_obs = np.zeros(n_act, dtype=np.float32)
        zero_for_obs[1] = 1.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker.on_action(0, env, legal[0], lambda _h: zero_for_obs)
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)
        np.testing.assert_allclose(
            tracker.range_of(0),
            _initial_uniform(env, ()),  # board-only, my_hole kept
            rtol=1e-6,
        )
        assert tracker.range_of(0)[actual_idx] > 0.0


class TestZeroConflictingHelper:

    def test_empty_cards_no_change(self):
        env = _env()
        w = np.ones(env.n_combos, dtype=np.float32)
        _zero_conflicting(w, env, [])
        assert (w == 1.0).all()

    def test_zeros_only_conflicting(self):
        env = _env()
        w = np.ones(env.n_combos, dtype=np.float32)
        target_card = int(env.combo_cards[0, 0])
        _zero_conflicting(w, env, [target_card])
        for i in range(env.n_combos):
            uses = (
                int(env.combo_cards[i, 0]) == target_card
                or int(env.combo_cards[i, 1]) == target_card
            )
            assert (w[i] == 0.0) == uses
