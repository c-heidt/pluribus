"""Tests for the env's off-tree action overlay."""

import copy

import pytest

from environment.action_space import ACTION_TO_IDX, CANONICAL_ACTIONS
from environment.player import Player
from environment.poker_env import PokerEnv


def _env(n_players: int = 2):
    return PokerEnv(players=[Player(i, 10000) for i in range(n_players)])


class TestInjectAction:

    def test_appears_in_legal_actions(self):
        env = _env()
        assert "raise:1.1" not in env.legal_actions
        env.inject_action("raise:1.1")
        assert env.legal_actions.count("raise:1.1") == 1

    def test_idempotent(self):
        env = _env()
        env.inject_action("raise:1.1")
        before = list(env.legal_actions)
        env.inject_action("raise:1.1")
        assert env.legal_actions == before

    def test_dedupes_against_canonical(self):
        env = _env()
        env.inject_action("fold")
        assert env.legal_actions.count("fold") == 1


class TestOverlayScopedToPublicState:

    def test_injection_not_visible_after_action(self):
        env = _env()
        env.inject_action("raise:1.1")
        assert env.has_overlay_at_current_node
        # Step the env — public state (history) changes.
        env_next = env.apply_action("call")
        assert not env_next.has_overlay_at_current_node
        assert "raise:1.1" not in env_next.legal_actions

    def test_original_state_still_sees_injection(self):
        env = _env()
        env.inject_action("raise:1.1")
        # Advance via deepcopy, leaving original env untouched.
        _ = env.apply_action("call")
        assert env.has_overlay_at_current_node
        assert "raise:1.1" in env.legal_actions


class TestResetOverlay:

    def test_reset_removes_injections(self):
        env = _env()
        env.inject_action("raise:1.1")
        assert env.has_overlay_at_current_node
        env.reset_overlay()
        assert not env.has_overlay_at_current_node
        assert "raise:1.1" not in env.legal_actions

    def test_reset_on_fresh_env_noop(self):
        env = _env()
        env.reset_overlay()
        assert not env.has_overlay_at_current_node


class TestHasOverlayProperty:

    def test_false_on_fresh_env(self):
        assert not _env().has_overlay_at_current_node

    def test_true_after_inject(self):
        env = _env()
        env.inject_action("raise:1.1")
        assert env.has_overlay_at_current_node

    def test_false_after_reset(self):
        env = _env()
        env.inject_action("raise:1.1")
        env.reset_overlay()
        assert not env.has_overlay_at_current_node

    def test_false_when_current_player_inactive(self):
        # legal_actions short-circuits to [None] for an inactive player
        # and exposes no overlay; has_overlay_at_current_node must agree
        # so callers can use either to gate the round-1 fast path.
        env = _env()
        env.inject_action("raise:1.1")
        assert env.has_overlay_at_current_node     # baseline
        env.current_player._is_active = False
        assert env.legal_actions == [None]
        assert not env.has_overlay_at_current_node


class TestOverlaySharedByReference:

    def test_deepcopy_shares_overlay(self):
        env = _env()
        env_copy = copy.deepcopy(env)
        env_copy.inject_action("raise:1.1")
        # In-place mutation, shared by reference: both envs see it.
        assert env.has_overlay_at_current_node
        assert "raise:1.1" in env.legal_actions

    def test_apply_action_descendant_sees_overlay_at_returning_state(self):
        # Injection on parent should be visible to any descendant env that
        # later reaches the same public state.  We can't easily return to
        # the same public state in real play, so this asserts the simpler
        # property: a deepcopy of a post-inject env preserves the injection
        # at the matching public state.
        env = _env()
        env.inject_action("raise:1.1")
        env_copy = copy.deepcopy(env)
        assert env_copy.has_overlay_at_current_node
        assert "raise:1.1" in env_copy.legal_actions

    def test_reset_overlay_clears_for_whole_lineage(self):
        # The overlay is shared by reference across the deepcopy lineage,
        # so reset_overlay() on ANY env in the lineage clears it for all.
        # Documented foot-gun: search-time deepcopies that call
        # reset_overlay would wipe the runtime env's overlay too.
        env = _env()
        env.inject_action("raise:1.1")
        env_copy = copy.deepcopy(env)
        env_copy.reset_overlay()
        assert not env.has_overlay_at_current_node
        assert "raise:1.1" not in env.legal_actions


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
        env.inject_action("raise:1.1")
        env.inject_action("raise:9.9")
        env.inject_action("raise:2.3")
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
        env.inject_action("raise:1.1")
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
        for a in ("raise:9.9", "raise:2.3", "raise:1.1", "raise:1.7"):
            env.inject_action(a)
        legal = [a for a in env.legal_actions if a is not None]
        overlay_in_legal = [
            a for a in legal if a not in ACTION_TO_IDX[env.betting_round]
        ]
        assert overlay_in_legal == sorted(overlay_in_legal)

    def test_legal_actions_deterministic_across_calls(self):
        env = _env()
        for a in ("raise:1.1", "raise:9.9", "raise:1.7", "raise:1.3"):
            env.inject_action(a)
        # Multiple calls on the same env must return identical lists.
        first = env.legal_actions
        for _ in range(5):
            assert env.legal_actions == first

    def test_order_independent_of_insertion_order(self):
        env_a = _env()
        env_b = _env()
        for a in ("raise:1.1", "raise:9.9", "raise:1.7", "raise:1.3"):
            env_a.inject_action(a)
        for a in ("raise:1.3", "raise:1.7", "raise:9.9", "raise:1.1"):
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
        env.inject_action("raise:1.1")
        legal = [a for a in env.legal_actions if a is not None]
        a_to_i = ACTION_TO_IDX[env.betting_round]
        canonical_part = [a for a in legal if a in a_to_i]
        idx = [a_to_i[a] for a in canonical_part]
        assert len(set(idx)) == len(idx)
        # And: every canonical action present is a known canonical key.
        for a in canonical_part:
            assert a in CANONICAL_ACTIONS[env.betting_round]

    def test_apply_action_uses_overlay_action(self):
        # Smoke-only check (full round-trip lives in TestInjectApplyRoundTrip).
        env = _env()
        env.inject_action("raise:1.1")
        assert "raise:1.1" in env.legal_actions
        env_next = env.apply_action("raise:1.1")
        assert env_next.pot_size > env.pot_size


class TestInjectApplyRoundTrip:
    """End-to-end: every action that passes inject_action's sanity check
    must also execute cleanly via apply_action and leave the env in a
    consistent state.  Without this, the overlay is decorative — a
    sanity check that admits actions apply_action then mishandles would
    be a hidden divergence between the search abstraction and the
    game's actual dynamics.
    """

    def test_chip_accounting_consistent(self):
        # Pot, player stack, and player bet must all move by the
        # expected amount.  At HU pre-flop with pot=150,
        # raise:1.1 → ceil(150*1.1)=165 chips to add.
        env = _env()
        env.inject_action("raise:1.1")
        player = env.current_player
        chips_before = player.n_chips
        bet_before = player.n_bet_chips
        pot_before = env.pot_size

        env_next = env.apply_action("raise:1.1")
        new_player = env_next.players[player.player_i]

        added = chips_before - new_player.n_chips
        assert added == 165
        assert new_player.n_bet_chips - bet_before == 165
        assert env_next.pot_size - pot_before == 165

    def test_raise_increments_n_raises(self):
        env = _env()
        env.inject_action("raise:1.1")
        assert env._n_raises == 0
        env_next = env.apply_action("raise:1.1")
        assert env_next._n_raises == 1

    def test_action_recorded_in_history(self):
        env = _env()
        env.inject_action("raise:1.1")
        env_next = env.apply_action("raise:1.1")
        assert env_next._history["pre_flop"][-1] == "raise:1.1"

    def test_next_state_is_terminal_legal(self):
        # The opponent's response options after an injected raise are
        # well-formed: at least fold and call/all_in available.
        env = _env()
        env.inject_action("raise:1.1")
        env_next = env.apply_action("raise:1.1")
        legal = [a for a in env_next.legal_actions if a is not None]
        assert "fold" in legal
        assert "call" in legal or "all_in" in legal

    def test_injection_does_not_carry_to_next_state(self):
        # The injection was at the pre-action public state; after
        # apply_action the public state changes and the overlay
        # should not show up.
        env = _env()
        env.inject_action("raise:1.1")
        env_next = env.apply_action("raise:1.1")
        assert not env_next.has_overlay_at_current_node
        assert "raise:1.1" not in env_next.legal_actions

    def test_multiple_off_tree_raises_all_executable(self):
        # Inject several distinct off-tree raises and verify each one
        # round-trips cleanly from a fresh env.  Catches edge cases
        # where one fraction works but a neighboring one corrupts state.
        for action in ("raise:1.1", "raise:1.3", "raise:1.7", "raise:9.9"):
            env = _env()
            env.inject_action(action)
            assert action in env.legal_actions, action
            env_next = env.apply_action(action)
            assert env_next.pot_size > env.pot_size, action
            # Final consistency: next state is well-formed.
            assert env_next.current_player.player_i != env.current_player.player_i


class TestInjectActionReturnValue:
    """`inject_action` returns True iff the action is in the legal
    set after the call.  Translation (§6.3) branches on this."""

    def test_returns_true_on_new_valid_injection(self):
        env = _env()
        assert env.inject_action("raise:1.1") is True
        assert "raise:1.1" in env.legal_actions

    def test_returns_true_on_repeat_valid_injection(self):
        env = _env()
        env.inject_action("raise:1.1")
        # Already present — still returns True (the action IS injected).
        assert env.inject_action("raise:1.1") is True

    def test_returns_true_for_canonical_already_legal(self):
        # fold / call / all_in are always canonical; inject is a no-op.
        env = _env()
        before = list(env.legal_actions)
        assert env.inject_action("fold") is True
        # No overlay was actually written.
        assert not env.has_overlay_at_current_node
        assert env.legal_actions == before

    def test_returns_false_for_canonical_when_player_inactive(self):
        # F2 regression: canonical actions are not unconditionally
        # legal — an inactive player's legal_actions is [None], so
        # inject_action("fold") must report False, matching the
        # documented "True iff in legal_actions" contract.
        env = _env()
        env.current_player._is_active = False
        assert env.inject_action("fold") is False
        assert env.inject_action("call") is False
        assert env.inject_action("all_in") is False
        # And no overlay was written.
        assert env._extra_legal_actions == {}

    def test_returns_false_for_all_in_when_stack_is_zero(self):
        # all_in is gated on chips_available > 0 in legal_actions
        # ([poker_env.py:949-951]); stack-0 means no all_in is legal.
        env = _env()
        env.current_player.n_chips = 0
        # Sanity: legal_actions excludes all_in here.
        assert "all_in" not in env.legal_actions
        assert env.inject_action("all_in") is False
        assert env._extra_legal_actions == {}


class TestInjectActionSanityChecks:
    """Game-state rejections return False without mutating the overlay."""

    def test_rejects_below_min_raise(self):
        # At HU pre-flop, pot=150, last_raise_amount=100.
        # raise:0.42 → ceil(150*0.42)=63 chips → actual_raise=13 < 100.
        env = _env()
        assert env.inject_action("raise:0.42") is False
        assert "raise:0.42" not in env.legal_actions
        assert not env.has_overlay_at_current_node

    def test_rejects_at_or_above_stack(self):
        # Player has 9950 chips available; "raise:99.9" → 14985 chips.
        env = _env()
        assert env.inject_action("raise:99.9") is False
        assert "raise:99.9" not in env.legal_actions
        assert not env.has_overlay_at_current_node

    def test_rejects_when_max_raises_reached(self):
        env = _env()
        # Force three raises into the round.
        env._n_raises = 3
        assert env.inject_action("raise:1.5") is False
        assert not env.has_overlay_at_current_node

    def test_rejection_does_not_taint_subsequent_valid_inject(self):
        env = _env()
        assert env.inject_action("raise:0.42") is False
        # A valid raise after a rejected one still works.
        assert env.inject_action("raise:1.1") is True
        assert "raise:1.1" in env.legal_actions
        assert "raise:0.42" not in env.legal_actions


class TestInjectActionMalformedInput:
    """Programmer-error inputs raise ValueError, not silent rejection."""

    def test_unknown_action_prefix_raises(self):
        env = _env()
        with pytest.raises(ValueError):
            env.inject_action("strange_action")

    def test_unparseable_fraction_raises(self):
        env = _env()
        with pytest.raises(ValueError):
            env.inject_action("raise:abc")

    def test_empty_fraction_raises(self):
        env = _env()
        with pytest.raises(ValueError):
            env.inject_action("raise:")

    def test_zero_fraction_raises(self):
        env = _env()
        with pytest.raises(ValueError):
            env.inject_action("raise:0")

    def test_negative_fraction_raises(self):
        env = _env()
        with pytest.raises(ValueError):
            env.inject_action("raise:-1.5")

    def test_infinite_fraction_raises(self):
        env = _env()
        with pytest.raises(ValueError):
            env.inject_action("raise:inf")

    def test_malformed_input_does_not_mutate_overlay(self):
        env = _env()
        with pytest.raises(ValueError):
            env.inject_action("raise:abc")
        assert not env.has_overlay_at_current_node
