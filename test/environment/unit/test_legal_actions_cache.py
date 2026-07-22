"""Tests for the public-state-keyed ``legal_actions`` memo on :class:`PokerEnv`.

``legal_actions`` is a pure function of the current public state plus the
overlay (``_extra_legal_actions``), so it is memoised in ``_legal_actions_cache``
keyed by :meth:`PokerEnv._current_public_state`.  The memo must be transparent —
every read returns exactly what an uncached derivation would — and must stay
coherent across the make/undo walk and the overlay mutators.  These tests pin
that contract.
"""

import copy

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv


def _env(n_players: int = 2, seed: int = 0) -> PokerEnv:
    np.random.seed(seed)
    return PokerEnv(players=[Player(i, 10000) for i in range(n_players)])


def _uncached(env: PokerEnv):
    """What ``legal_actions`` would return with no memo (fresh derivation)."""
    return env._compute_legal_actions(env._current_public_state())


class TestLegalActionsCacheTransparency:

    def test_first_and_repeat_reads_match_uncached(self):
        env = _env()
        expected = _uncached(env)
        # Cold read populates the memo; warm read serves from it.  Both must
        # equal the uncached derivation.
        assert env.legal_actions == expected
        assert env.legal_actions == expected

    def test_matches_uncached_at_every_node_of_a_walk(self):
        env = _env(seed=3)
        for _ in range(8):
            if env.is_terminal:
                break
            assert env.legal_actions == _uncached(env)
            legal = [a for a in env.legal_actions if a is not None]
            env.step_in_place(legal[0])

    def test_returned_list_is_a_defensive_copy(self):
        env = _env()
        first = env.legal_actions
        first.append("raise:9.9")  # mutate the returned list
        first[0] = "TAMPERED"
        # The memo is untouched; a fresh read is pristine.
        assert env.legal_actions == _uncached(env)
        assert "TAMPERED" not in env.legal_actions


class TestLegalActionsCacheMakeUndo:

    def test_step_reflects_new_state_not_stale_parent(self):
        env = _env(seed=1)
        parent = env.legal_actions  # populate memo for the parent node
        legal = [a for a in parent if a is not None]
        env.step_in_place(legal[0])
        # After advancing, the read must reflect the child's state.
        assert env.legal_actions == _uncached(env)

    def test_undo_restores_parent_legal_set(self):
        env = _env(seed=2)
        parent = env.legal_actions
        legal = [a for a in parent if a is not None]
        token = env.step_in_place(legal[0])
        _ = env.legal_actions  # populate the child entry too
        env.undo(token)
        # Ascending re-selects the parent entry by its public-state key — no
        # cache bookkeeping in step/undo is needed for this to hold.
        assert env.legal_actions == parent
        assert env.legal_actions == _uncached(env)

    def test_explore_all_actions_then_undo_each(self):
        # Mirrors the traverser loop: legal must be correct for the parent on
        # every iteration despite child excursions overwriting nothing.
        env = _env(seed=4)
        parent = env.legal_actions
        legal = [a for a in parent if a is not None]
        for action in legal:
            token = env.step_in_place(action)
            _ = env.legal_actions
            env.undo(token)
            assert env.legal_actions == parent


class TestLegalActionsCacheInvalidation:

    def test_inject_action_is_visible_after_a_cold_read(self):
        env = _env()
        before = env.legal_actions  # populate memo without the overlay
        injected = "raise:1.1"
        assert env.inject_action(injected) is True
        # The memo entry for this state was dropped, so the injected action
        # shows up immediately.
        assert injected in env.legal_actions
        assert injected not in before

    def test_reset_overlay_drops_injected_action(self):
        env = _env()
        env.inject_action("raise:1.1")
        assert "raise:1.1" in env.legal_actions  # populate memo *with* overlay
        env.reset_overlay()
        assert "raise:1.1" not in env.legal_actions
        assert env.legal_actions == _uncached(env)


class TestLegalActionsCacheDeepcopy:

    def test_deepcopy_starts_with_empty_independent_memo(self):
        env = _env(seed=5)
        _ = env.legal_actions  # fill the original's memo
        clone = copy.deepcopy(env)
        # Fresh empty memo on the copy — never shared by reference (so no stale
        # entry can survive a copy taken to start a new hand).
        assert clone._legal_actions_cache == {}
        assert clone.legal_actions == _uncached(clone)
        # Advancing the clone fills *its* memo; the original's is a separate dict
        # object, so it is unaffected and still serves correct values.
        orig_memo = env._legal_actions_cache
        clone.step_in_place([a for a in clone.legal_actions if a is not None][0])
        _ = clone.legal_actions
        assert clone._legal_actions_cache is not env._legal_actions_cache
        assert env._legal_actions_cache is orig_memo
        assert env.legal_actions == _uncached(env)
