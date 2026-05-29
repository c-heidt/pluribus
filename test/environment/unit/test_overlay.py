"""Tests for the env's off-tree action overlay."""

import copy

from environment.action_space import ACTION_TO_IDX, CANONICAL_ACTIONS
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


class TestLegalActionsOrdering:
    """Pin down the legal_actions ordering contract.

    Critical for the blueprint: ACTION_TO_IDX is built from
    CANONICAL_ACTIONS, so the blueprint's column indexing only works
    if canonical actions appear in canonical order in legal_actions
    regardless of overlay state.  The subgame solver additionally
    requires overlay actions to appear in a deterministic order
    across processes (frozenset iteration is NOT deterministic; we
    sort).
    """

    def test_canonical_subset_order_unchanged_by_overlay(self):
        env = _env()
        before = [
            a for a in env.legal_actions
            if a is not None and a in ACTION_TO_IDX[env.betting_round]
        ]
        env.inject_action("raise:0.42")
        env.inject_action("raise:9.9")
        env.inject_action("raise:0.123")
        after = [
            a for a in env.legal_actions
            if a is not None and a in ACTION_TO_IDX[env.betting_round]
        ]
        # Same canonical actions, in the same canonical order, regardless
        # of how many off-tree injections we add.  This is the property
        # blueprint regret-row column indexing relies on.
        assert before == after

    def test_overlay_actions_appear_after_canonical(self):
        env = _env()
        env.inject_action("raise:0.42")
        env.inject_action("raise:9.9")
        legal = [a for a in env.legal_actions if a is not None]
        canonical_keys = set(ACTION_TO_IDX[env.betting_round])
        last_canonical_idx = max(
            i for i, a in enumerate(legal) if a in canonical_keys
        )
        first_overlay_idx = min(
            i for i, a in enumerate(legal) if a not in canonical_keys
        )
        assert first_overlay_idx > last_canonical_idx

    def test_overlay_actions_sorted_lex(self):
        env = _env()
        # Inject in a non-sorted order to prove the env re-sorts.
        for a in ("raise:9.9", "raise:0.123", "raise:0.42", "raise:0.07"):
            env.inject_action(a)
        legal = [a for a in env.legal_actions if a is not None]
        overlay_in_legal = [
            a for a in legal if a not in ACTION_TO_IDX[env.betting_round]
        ]
        assert overlay_in_legal == sorted(overlay_in_legal)

    def test_legal_actions_deterministic_across_calls(self):
        env = _env()
        for a in ("raise:0.42", "raise:9.9", "raise:0.07", "raise:1.23"):
            env.inject_action(a)
        # Multiple calls on the same env must return identical lists.
        first = env.legal_actions
        for _ in range(5):
            assert env.legal_actions == first

    def test_order_independent_of_insertion_order(self):
        env_a = _env()
        env_b = _env()
        for a in ("raise:0.42", "raise:9.9", "raise:0.07", "raise:1.23"):
            env_a.inject_action(a)
        for a in ("raise:1.23", "raise:0.07", "raise:9.9", "raise:0.42"):
            env_b.inject_action(a)
        assert env_a.legal_actions == env_b.legal_actions

    def test_action_to_idx_lookup_safe_for_canonical_subset(self):
        # The blueprint's BlueprintPolicy does
        #   idx = [ACTION_TO_IDX[r][a] for a in legal]
        # to map per-action slots back to canonical regret-row columns.
        # The mapping is per-action, so legal_actions order need NOT
        # match canonical order — but every canonical action present
        # in legal_actions must be looked up without KeyError, and
        # the resulting column indices must be unique (no two actions
        # collapse to the same column).
        env = _env()
        env.inject_action("raise:0.42")
        legal = [a for a in env.legal_actions if a is not None]
        a_to_i = ACTION_TO_IDX[env.betting_round]
        canonical_part = [a for a in legal if a in a_to_i]
        idx = [a_to_i[a] for a in canonical_part]
        assert len(set(idx)) == len(idx)
        # And: every canonical action present is a known canonical key.
        for a in canonical_part:
            assert a in CANONICAL_ACTIONS[env.betting_round]

    def test_apply_action_uses_overlay_action(self):
        # End-to-end round-trip: an injected action must actually
        # execute via apply_action.  Without this, the overlay is
        # decorative.  Use a fraction that wouldn't normally be in
        # the pre-flop abstraction.
        env = _env()
        env.inject_action("raise:0.42")
        assert "raise:0.42" in env.legal_actions
        env_next = env.apply_action("raise:0.42")
        # Pot grew; the action was real.
        assert env_next.pot_size > env.pot_size
