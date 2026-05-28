"""Tests for the env's off-tree action overlay."""

import copy

from environment.player import Player
from environment.poker_env import PokerEnv


def _env(n_players: int = 2):
    return PokerEnv(players=[Player(i, 10000) for i in range(n_players)])


class TestInjectAction:

    def test_appears_in_legal_actions(self):
        env = _env()
        assert "raise:0.42" not in env.legal_actions
        env.inject_action("raise:0.42")
        assert env.legal_actions.count("raise:0.42") == 1

    def test_idempotent(self):
        env = _env()
        env.inject_action("raise:0.42")
        before = list(env.legal_actions)
        env.inject_action("raise:0.42")
        assert env.legal_actions == before

    def test_dedupes_against_canonical(self):
        env = _env()
        env.inject_action("fold")
        assert env.legal_actions.count("fold") == 1


class TestOverlayScopedToPublicState:

    def test_injection_not_visible_after_action(self):
        env = _env()
        env.inject_action("raise:0.42")
        assert env.has_overlay_at_current_node
        # Step the env — public state (history) changes.
        env_next = env.apply_action("call")
        assert not env_next.has_overlay_at_current_node
        assert "raise:0.42" not in env_next.legal_actions

    def test_original_state_still_sees_injection(self):
        env = _env()
        env.inject_action("raise:0.42")
        # Advance via deepcopy, leaving original env untouched.
        _ = env.apply_action("call")
        assert env.has_overlay_at_current_node
        assert "raise:0.42" in env.legal_actions


class TestResetOverlay:

    def test_reset_removes_injections(self):
        env = _env()
        env.inject_action("raise:0.42")
        assert env.has_overlay_at_current_node
        env.reset_overlay()
        assert not env.has_overlay_at_current_node
        assert "raise:0.42" not in env.legal_actions

    def test_reset_on_fresh_env_noop(self):
        env = _env()
        env.reset_overlay()
        assert not env.has_overlay_at_current_node


class TestHasOverlayProperty:

    def test_false_on_fresh_env(self):
        assert not _env().has_overlay_at_current_node

    def test_true_after_inject(self):
        env = _env()
        env.inject_action("raise:0.42")
        assert env.has_overlay_at_current_node

    def test_false_after_reset(self):
        env = _env()
        env.inject_action("raise:0.42")
        env.reset_overlay()
        assert not env.has_overlay_at_current_node


class TestOverlaySharedByReference:

    def test_deepcopy_shares_overlay(self):
        env = _env()
        env_copy = copy.deepcopy(env)
        env_copy.inject_action("raise:0.42")
        # In-place mutation, shared by reference: both envs see it.
        assert env.has_overlay_at_current_node
        assert "raise:0.42" in env.legal_actions

    def test_apply_action_descendant_sees_overlay_at_returning_state(self):
        # Injection on parent should be visible to any descendant env that
        # later reaches the same public state.  We can't easily return to
        # the same public state in real play, so this asserts the simpler
        # property: a deepcopy of a post-inject env preserves the injection
        # at the matching public state.
        env = _env()
        env.inject_action("raise:0.42")
        env_copy = copy.deepcopy(env)
        assert env_copy.has_overlay_at_current_node
        assert "raise:0.42" in env_copy.legal_actions
