"""Tests for ``PokerEnv.with_hole_cards`` (batched full-sequence form).

The method takes one ``(c0, c1)`` tuple per seat (length
``n_players``) and applies them atomically.  Validation rejects
within-hole duplicates, pairwise collisions across seats, and any
overlap with the community.  After ``Deck.replace_drawn`` the
undealt segment is shuffled so per-rollout community deals don't
prefer positions touched by the swap.
"""

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv


def _env(n_players: int = 2) -> PokerEnv:
    return PokerEnv(players=[Player(i, 10000) for i in range(n_players)])


def _current_holes(env: PokerEnv):
    return [tuple(int(c) for c in env.players[i]._cards) for i in range(env.n_players)]


def _holes_with(env: PokerEnv, seat: int, cards):
    """Length-N list with ``seat``'s hole swapped for ``cards``."""
    holes = _current_holes(env)
    holes[seat] = (int(cards[0]), int(cards[1]))
    return holes


def _disjoint_combo(env: PokerEnv, seat: int):
    """Pick a combo whose cards share nothing with the community or any
    other seat's hole.  Tests that need a *valid* single-seat
    replacement derive one from the env rather than hard-coding
    ``combo_cards[0]``.
    """
    forbidden = set(int(c) for c in env.community_cards)
    for other_seat, player in enumerate(env.players):
        if other_seat == seat:
            continue
        forbidden.update(int(c) for c in player._cards)
    for i in range(env.n_combos):
        c0, c1 = int(env.combo_cards[i, 0]), int(env.combo_cards[i, 1])
        if c0 not in forbidden and c1 not in forbidden:
            return (c0, c1)
    raise AssertionError("no disjoint combo available")


class TestWithHoleCards:

    def test_returns_independent_env(self):
        env = _env()
        original_cards = env.players[0].cards
        replacement = _disjoint_combo(env, 0)
        new_env = env.with_hole_cards(_holes_with(env, 0, replacement))
        assert new_env is not env
        assert new_env.pot_size == env.pot_size
        assert env.players[0].cards == original_cards

    def test_seat_cards_replaced(self):
        env = _env()
        replacement = _disjoint_combo(env, 0)
        new_env = env.with_hole_cards(_holes_with(env, 0, replacement))
        assert tuple(new_env.players[0].cards) == replacement

    def test_other_seats_unchanged_when_passed_through(self):
        env = _env(n_players=3)
        original_seat_1 = env.players[1].cards
        original_seat_2 = env.players[2].cards
        replacement = _disjoint_combo(env, 0)
        new_env = env.with_hole_cards(_holes_with(env, 0, replacement))
        assert new_env.players[1].cards == original_seat_1
        assert new_env.players[2].cards == original_seat_2

    def test_card_info_lut_preserved(self):
        env = _env()
        env.card_info_lut = {"pre_flop": {"some_key": 42}}
        replacement = _disjoint_combo(env, 0)
        new_env = env.with_hole_cards(_holes_with(env, 0, replacement))
        assert new_env.card_info_lut is env.card_info_lut

    def test_overlay_shared(self):
        env = _env()
        env.inject_action("raise:1.1")
        replacement = _disjoint_combo(env, 0)
        new_env = env.with_hole_cards(_holes_with(env, 0, replacement))
        assert new_env.has_overlay_at_current_node

    def test_info_set_reflects_replaced_cards(self):
        env = _env()
        actual_hole = tuple(sorted(int(c) for c in env.players[env.player_i]._cards))
        replacement = None
        for i in range(env.n_combos):
            combo = tuple(sorted(int(c) for c in env.combo_cards[i]))
            if combo != actual_hole:
                other = set(int(c) for c in env.players[1 - env.player_i]._cards)
                if not (set(combo) & other):
                    replacement = combo
                    break
        assert replacement is not None
        env.card_info_lut = {
            "pre_flop": {
                actual_hole: 111,
                replacement: 222,
            }
        }
        info_before = env.info_set
        new_env = env.with_hole_cards(_holes_with(env, env.player_i, replacement))
        info_after = new_env.info_set
        assert info_before != info_after
        assert '"cards_cluster":111' in info_before
        assert '"cards_cluster":222' in info_after


class TestValidation:
    """``with_hole_cards`` rejects holes that aren't physically realisable."""

    def test_wrong_length_raises(self):
        env = _env(n_players=3)
        too_short = _current_holes(env)[:2]
        with pytest.raises(ValueError, match="expected 3 hole tuples"):
            env.with_hole_cards(too_short)

    def test_internal_duplicate_raises(self):
        env = _env()
        card = int(env.combo_cards[0, 0])
        holes = _holes_with(env, 0, (card, card))
        with pytest.raises(ValueError, match="distinct"):
            env.with_hole_cards(holes)

    def test_community_overlap_raises(self):
        env = _env()
        board_card = int(env.combo_cards[0, 0])
        env.community_cards = (board_card,)
        forbidden = {board_card} | set(int(c) for c in env.players[1]._cards)
        second = next(
            int(env.combo_cards[i, 0])
            for i in range(env.n_combos)
            if int(env.combo_cards[i, 0]) not in forbidden
        )
        holes = _holes_with(env, 0, (board_card, second))
        with pytest.raises(ValueError, match="overlap the community"):
            env.with_hole_cards(holes)

    def test_pairwise_collision_within_batch_raises(self):
        # Two seats specifying the same card across their holes is rejected.
        env = _env(n_players=3)
        shared = int(env.players[0]._cards[0])
        # Pick a second card for seat 1 that doesn't collide otherwise.
        forbidden = set(int(c) for c in env.community_cards)
        for p in env.players:
            forbidden.update(int(c) for c in p._cards)
        # Seat 1's new hole reuses seat 0's existing card.
        second = next(
            int(env.combo_cards[i, 0])
            for i in range(env.n_combos)
            if int(env.combo_cards[i, 0]) not in forbidden
        )
        holes = _current_holes(env)
        holes[1] = (shared, second)
        with pytest.raises(ValueError, match="pairwise disjoint"):
            env.with_hole_cards(holes)

    def test_replacing_all_seats_with_current_is_valid(self):
        # Identity batch (every seat keeps its current cards) is a no-op.
        env = _env(n_players=3)
        original = _current_holes(env)
        new_env = env.with_hole_cards(original)
        for i in range(env.n_players):
            assert tuple(new_env.players[i]._cards) == original[i]


class TestDeckSync:
    """After ``with_hole_cards`` the env is indistinguishable from a
    regular state that was dealt those cards from the start."""

    def test_deck_drawn_segment_contains_new_hole(self):
        env = _env()
        replacement = _disjoint_combo(env, 0)
        new_env = env.with_hole_cards(_holes_with(env, 0, replacement))
        drawn = set(int(c) for c in new_env.deck._cards[: new_env.deck._idx])
        assert replacement[0] in drawn
        assert replacement[1] in drawn

    def test_deck_undealt_segment_excludes_new_hole(self):
        env = _env()
        replacement = _disjoint_combo(env, 0)
        new_env = env.with_hole_cards(_holes_with(env, 0, replacement))
        undealt = set(int(c) for c in new_env.deck._cards[new_env.deck._idx:])
        assert replacement[0] not in undealt
        assert replacement[1] not in undealt

    def test_deck_undealt_segment_includes_displaced_cards(self):
        env = _env()
        original = tuple(int(c) for c in env.players[0]._cards)
        replacement = _disjoint_combo(env, 0)
        new_env = env.with_hole_cards(_holes_with(env, 0, replacement))
        undealt = set(int(c) for c in new_env.deck._cards[new_env.deck._idx:])
        displaced = set(original) - set(replacement)
        assert displaced <= undealt

    def test_deck_multiset_preserved(self):
        env = _env()
        before = sorted(int(c) for c in env.deck._cards)
        replacement = _disjoint_combo(env, 0)
        new_env = env.with_hole_cards(_holes_with(env, 0, replacement))
        after = sorted(int(c) for c in new_env.deck._cards)
        assert before == after

    def test_deck_idx_unchanged(self):
        env = _env()
        idx_before = env.deck._idx
        replacement = _disjoint_combo(env, 0)
        new_env = env.with_hole_cards(_holes_with(env, 0, replacement))
        assert new_env.deck._idx == idx_before

    def test_apply_action_after_swap_reaches_terminal_safely(self):
        # End-to-end: a fold-terminal after a hole swap must not
        # produce a duplicate-card 7-card hand for the surviving seat.
        env = _env()
        replacement = _disjoint_combo(env, 1)
        new_env = env.with_hole_cards(_holes_with(env, 1, replacement))
        terminal = new_env.apply_action("fold")
        assert terminal.is_terminal
        survivor_cards = (
            tuple(int(c) for c in terminal.players[1]._cards)
            + tuple(int(c) for c in terminal.community_cards)
        )
        assert len(set(survivor_cards)) == len(survivor_cards)


class TestAtomicBatch:
    """Cases the old single-seat form couldn't express because per-call
    validation tripped on cards held by another seat in the batch."""

    def test_two_seats_swap_holes(self):
        # new[0] == old[1] and new[1] == old[0] simultaneously.
        env = _env()
        old0 = tuple(int(c) for c in env.players[0]._cards)
        old1 = tuple(int(c) for c in env.players[1]._cards)
        new_env = env.with_hole_cards([old1, old0])
        assert tuple(new_env.players[0]._cards) == old1
        assert tuple(new_env.players[1]._cards) == old0
        # Deck still consistent: drawn segment holds the same set, no duplicates.
        drawn = list(int(c) for c in new_env.deck._cards[: new_env.deck._idx])
        assert len(drawn) == len(set(drawn))

    def test_three_seat_rotation(self):
        env = _env(n_players=3)
        old0 = tuple(int(c) for c in env.players[0]._cards)
        old1 = tuple(int(c) for c in env.players[1]._cards)
        old2 = tuple(int(c) for c in env.players[2]._cards)
        # Rotate: 0 ← old1, 1 ← old2, 2 ← old0.
        new_env = env.with_hole_cards([old1, old2, old0])
        assert tuple(new_env.players[0]._cards) == old1
        assert tuple(new_env.players[1]._cards) == old2
        assert tuple(new_env.players[2]._cards) == old0
        # Multiset preserved.
        before = sorted(int(c) for c in env.deck._cards)
        after = sorted(int(c) for c in new_env.deck._cards)
        assert before == after

    def test_new_hole_reuses_own_card_keeps_env_consistent(self):
        env = _env()
        old = tuple(int(c) for c in env.players[1]._cards)
        forbidden = set(env.community_cards)
        for p in env.players:
            forbidden.update(int(c) for c in p._cards)
        new_card = next(
            int(c)
            for c in env.deck._cards[env.deck._idx:]
            if int(c) not in forbidden
        )
        # Seat 1's new hole reuses one of its own current cards.
        new_env = env.with_hole_cards(_holes_with(env, 1, (old[1], new_card)))
        assert tuple(int(c) for c in new_env.players[1]._cards) == (
            old[1],
            new_card,
        )
        terminal = new_env.apply_action("fold")
        survivor_cards = (
            tuple(int(c) for c in terminal.players[1]._cards)
            + tuple(int(c) for c in terminal.community_cards)
        )
        assert len(set(survivor_cards)) == len(survivor_cards)


class TestShuffleUndealtInvoked:
    """``with_hole_cards`` must call ``Deck.shuffle_undealt`` so the
    undealt segment is not preferentially exposing displaced cards.

    Verifying that the post-swap order is *different* from the
    replace-drawn-only order is enough — the unit-level correctness
    of ``shuffle_undealt`` itself is covered in ``test_deck.py``.
    """

    def test_undealt_order_differs_from_replace_drawn_only(self):
        env = _env()
        replacement = _disjoint_combo(env, 0)

        # Branch A: with_hole_cards (replace_drawn + shuffle_undealt).
        np.random.seed(123)
        new_env = env.with_hole_cards(_holes_with(env, 0, replacement))
        seq_with_shuffle = list(int(c) for c in new_env.deck._cards[new_env.deck._idx:])

        # Branch B: manually run only replace_drawn on a deepcopy.
        import copy
        env_b = copy.deepcopy(env)
        old0 = tuple(int(c) for c in env_b.players[0]._cards)
        env_b.deck.replace_drawn(old0, replacement)
        seq_no_shuffle = list(int(c) for c in env_b.deck._cards[env_b.deck._idx:])

        # Same set, different order.
        assert sorted(seq_with_shuffle) == sorted(seq_no_shuffle)
        assert seq_with_shuffle != seq_no_shuffle


class TestDeckReplaceDrawn:
    """Direct tests on the underlying ``Deck.replace_drawn`` helper.

    These exercise the deck primitive itself, not ``with_hole_cards``;
    they stay unchanged across the env-API refactor.
    """

    def test_swaps_two_specific_cards(self):
        env = _env()
        deck = env.deck
        drawn = int(deck._cards[0])
        undealt = int(deck._cards[deck._idx + 3])
        deck.replace_drawn((drawn,), (undealt,))
        assert int(deck._cards[0]) == undealt
        assert int(deck._cards[deck._idx + 3]) == drawn

    def test_identical_pair_is_noop(self):
        env = _env()
        deck = env.deck
        before = list(int(c) for c in deck._cards)
        card = int(deck._cards[0])
        deck.replace_drawn((card,), (card,))
        assert list(int(c) for c in deck._cards) == before

    def test_idx_unchanged(self):
        env = _env()
        deck = env.deck
        before = deck._idx
        drawn = int(deck._cards[0])
        undealt = int(deck._cards[deck._idx + 3])
        deck.replace_drawn((drawn,), (undealt,))
        assert deck._idx == before

    def test_new_reuses_own_old_card_preserves_multiset(self):
        env = _env()
        deck = env.deck
        before = sorted(int(c) for c in deck._cards)
        a = int(deck._cards[0])
        b = int(deck._cards[1])
        x = int(deck._cards[deck._idx + 5])
        deck.replace_drawn((a, b), (b, x))
        after = sorted(int(c) for c in deck._cards)
        assert before == after, "multiset broken by overlapping swap"
        drawn = set(int(c) for c in deck._cards[: deck._idx])
        assert b in drawn
        assert x in drawn
        undealt = set(int(c) for c in deck._cards[deck._idx:])
        assert a in undealt

    def test_full_swap_preserves_multiset(self):
        env = _env()
        deck = env.deck
        before = sorted(int(c) for c in deck._cards)
        a = int(deck._cards[0])
        b = int(deck._cards[1])
        x = int(deck._cards[deck._idx + 2])
        y = int(deck._cards[deck._idx + 3])
        deck.replace_drawn((a, b), (x, y))
        after = sorted(int(c) for c in deck._cards)
        assert before == after
