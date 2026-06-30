"""Tests for the env make/undo traversal (`step_in_place` / `undo`).

The contract: ``step_in_place(a)`` advances the env in place, and ``undo``
restores the exact pre-step state (the mutable per-hand footprint
``__deepcopy__`` copies), field-identical to a pre-step ``deepcopy``.
"""

import copy

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv


def _env(n_players: int = 2, low: int = 2, high: int = 14, seed: int = 0) -> PokerEnv:
    # Full deck (2..14) so terminal transitions can rank hands via the
    # Evaluator.  Seed the global RNG so the shuffled deck is deterministic.
    np.random.seed(seed)
    return PokerEnv(
        players=[Player(i, 10000) for i in range(n_players)],
        low_card_rank=low,
        high_card_rank=high,
    )


def _mutable_view(env: PokerEnv) -> dict:
    """Everything the undo token is supposed to restore — the field set
    ``__deepcopy__`` deep-copies, excluding the by-reference shares
    (``card_info_lut`` and ``_extra_legal_actions`` are shared by
    reference)."""
    return {
        "betting_stage": env._betting_stage,
        "skip_counter": env._skip_counter,
        "first_move": env._first_move_of_current_round,
        "last_raise_amount": env._last_raise_amount,
        "all_acted": env._all_players_have_made_action,
        "n_actions": env._n_actions,
        "n_raises": env._n_raises,
        "player_i_index": env._player_i_index,
        "n_players_started_round": env._n_players_started_round,
        "community_cards": tuple(env.community_cards),
        "pot_chips": list(env.pot._chips),
        "deck_idx": env.deck._idx,
        "history": {k: list(v) for k, v in env._history.items()},
        "players": [
            (p.n_chips, p.n_bet_chips, p.is_active, p.is_turn, tuple(p.cards))
            for p in env.players
        ],
    }


def assert_env_equal(a: PokerEnv, b: PokerEnv) -> None:
    assert _mutable_view(a) == _mutable_view(b)
    # The deck's card array must never be mutated by a step (only the cursor
    # moves); compared separately because numpy arrays need array_equal.
    assert np.array_equal(a.deck._cards, b.deck._cards)


def _generic_state(env: PokerEnv) -> dict:
    """Full env state, derived by walking ``env.__dict__`` generically
    (excluding ``card_info_lut``, which is shared by reference).  Unlike
    ``_mutable_view``'s hand-picked field list, this auto-includes any
    *new* env attribute, so it catches a future mutable field that gets
    added to ``__deepcopy__`` but forgotten in the ``UndoToken``."""
    out = {}
    for k, v in env.__dict__.items():
        if k == "card_info_lut":
            continue
        if k == "players":
            out[k] = [(p.n_chips, p.n_bet_chips, p.is_active, p.is_turn,
                       tuple(p.cards)) for p in v]
        elif k == "pot":
            out[k] = list(v._chips)
        elif k == "deck":
            out[k] = (v._idx, tuple(v._cards.tolist()))
        elif k == "_history":
            out[k] = {s: list(a) for s, a in v.items()}
        else:
            out[k] = v
    return out


def _actions_to_test(env: PokerEnv):
    """A representative subset of legal actions bounding branching: every
    non-raise action plus the smallest and largest raise size."""
    legal = [a for a in env.legal_actions if a is not None]
    picked = [a for a in legal if a in ("fold", "call", "check", "all_in")]
    raises = [a for a in legal if a.startswith("raise")]
    if raises:
        picked.append(raises[0])
        if raises[-1] != raises[0]:
            picked.append(raises[-1])
    return picked


def _check_node(env: PokerEnv, depth: int, max_depth: int) -> None:
    """Recursively assert step/undo round-trip equivalence at every
    edge of the subtree rooted at ``env`` (bounded depth)."""
    if env.is_terminal or depth >= max_depth:
        return
    for action in _actions_to_test(env):
        before = copy.deepcopy(env)
        token = env.step_in_place(action)
        # Recurse on the mutated env (nested step/undo exercises LIFO).
        _check_node(env, depth + 1, max_depth)
        env.undo(token)
        # Round-trip: undo restores the exact pre-step state.
        assert_env_equal(env, before)


class TestStepUndo:

    @pytest.mark.parametrize("n_players", [2, 3])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_subtree_round_trip_and_equivalence(self, n_players, seed):
        env = _env(n_players=n_players, seed=seed)
        _check_node(env, depth=0, max_depth=5)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_linear_calldown_to_terminal(self, seed):
        # Play check/call to a showdown terminal; at every node test
        # step/undo round-trip + equivalence for each legal action, then
        # advance.  Guarantees coverage of the terminal transition
        # (deal remaining board + compute_winners).
        env = _env(n_players=2, seed=seed)
        steps = 0
        while not env.is_terminal and steps < 50:
            for action in _actions_to_test(env):
                before = copy.deepcopy(env)
                token = env.step_in_place(action)
                env.undo(token)
                assert_env_equal(env, before)
            nxt = "check" if "check" in env.legal_actions else "call"
            env.step_in_place(nxt)
            steps += 1
        assert env.is_terminal

    def test_lifo_nested_unwind(self):
        # Push a sequence of in-place steps, then undo in reverse order;
        # the env must return to the original root state.
        env = _env(n_players=3, seed=7)
        root = copy.deepcopy(env)
        tokens = []
        for _ in range(4):
            if env.is_terminal:
                break
            action = "call" if "call" in env.legal_actions else (
                "check" if "check" in env.legal_actions else
                [a for a in env.legal_actions if a][0]
            )
            tokens.append(env.step_in_place(action))
        for token in reversed(tokens):
            env.undo(token)
        assert_env_equal(env, root)

    def test_fold_terminal_round_trip(self):
        env = _env(n_players=2, seed=0)
        before = copy.deepcopy(env)
        token = env.step_in_place("fold")
        assert env.is_terminal  # HU fold ends the hand
        env.undo(token)
        assert_env_equal(env, before)

    def test_all_in_round_trip(self):
        env = _env(n_players=2, seed=3)
        before = copy.deepcopy(env)
        token = env.step_in_place("all_in")
        env.undo(token)
        assert_env_equal(env, before)

    def test_none_action_inactive_seat_round_trips(self):
        # An inactive (folded) actor is stepped with a ``None`` action; the
        # env supports it (skip), and undo must restore it like any other.
        env = _env(n_players=3, seed=1)
        env.current_player._is_active = False
        before = copy.deepcopy(env)
        token = env.step_in_place(None)
        env.undo(token)
        assert_env_equal(env, before)

    def test_undo_restores_full_env_dict_generically(self):
        # Maintainability guard: compare the *entire* env state (every
        # __dict__ attr), not a curated field list, so a future mutable
        # field missing from the UndoToken is caught here.
        env = _env(n_players=3, seed=0)
        env.step_in_place("call")
        env.step_in_place("call")  # to the flop: richer mid-hand state
        before = _generic_state(env)
        token = env.step_in_place(
            "call" if "call" in env.legal_actions else "check"
        )
        env.undo(token)
        assert _generic_state(env) == before


class TestCardDealingTransitions:
    """Dedicated coverage for the critical state changes a single step can
    trigger via the betting-round/terminal machinery: dealing community
    cards (round close: flop = +3, turn/river = +1) and the all-in runout
    that deals the remaining board *and* settles the pot in one step.  Each
    asserts the specific forward mutation, then that ``undo`` reverses it
    (deck cursor, community cards, per-round bets, pot, stacks)."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_round_close_deals_cards_and_reverses(self, seed):
        env = _env(n_players=2, seed=seed)
        seen = {1: 0, 3: 0}  # turn/river deal 1; flop deals 3
        steps = 0
        while not env.is_terminal and steps < 60:
            for a in _actions_to_test(env):
                before = copy.deepcopy(env)
                token = env.step_in_place(a)
                dealt = len(env.community_cards) - len(before.community_cards)
                if dealt > 0:
                    # The deck cursor advanced by exactly the cards dealt,
                    assert env.deck._idx == before.deck._idx + dealt
                    # the prior board is preserved as a prefix,
                    assert (env.community_cards[: len(before.community_cards)]
                            == before.community_cards)
                    # and a new street zeroes every player's round bet.
                    if not env.is_terminal:
                        assert all(p.n_bet_chips == 0 for p in env.players)
                    seen[dealt] = seen.get(dealt, 0) + 1
                env.undo(token)
                # Full reversal incl. deck cursor + community + bets.
                assert_env_equal(env, before)
            env.step_in_place(
                "check" if "check" in env.legal_actions else "call"
            )
            steps += 1
        # A calldown crosses all three post-flop streets, so both a 3-card
        # (flop) and a 1-card (turn/river) deal must have been exercised.
        assert seen[3] > 0 and seen[1] > 0

    def test_all_in_runout_terminal_round_trips(self):
        # The all-in runout sets the stage terminal, deals the remaining
        # board to 5 cards, AND runs compute_winners — all in one step.
        # This is the only path that deals *and* settles the pot in a single
        # action, so undo must reverse the multi-card deal and the payout.
        found = 0
        for seed in range(20):
            for n_players in (2, 3, 6):   # incl. 6-max (multiway side pots)
                env = _env(n_players=n_players, seed=seed)
                steps = 0
                while not env.is_terminal and steps < 40:
                    for a in _actions_to_test(env):
                        before = copy.deepcopy(env)
                        token = env.step_in_place(a)
                        if (env.is_terminal
                                and len(env.community_cards) == 5
                                and len(before.community_cards) < 5):
                            dealt = 5 - len(before.community_cards)
                            assert env.deck._idx == before.deck._idx + dealt
                            # Pot consumed; stacks received exactly the pot.
                            assert sum(env.pot._chips) == 0
                            assert (sum(p.n_chips for p in env.players)
                                    == sum(p.n_chips for p in before.players)
                                    + sum(before.pot._chips))
                            found += 1
                        env.undo(token)
                        assert_env_equal(env, before)
                    nxt = next(
                        (a for a in ("all_in", "call", "check")
                         if a in env.legal_actions),
                        next(a for a in env.legal_actions if a is not None),
                    )
                    env.step_in_place(nxt)
                    steps += 1
        assert found > 0, "expected to exercise a terminal board-completion step"


class TestHistoryRestoration:

    def test_undo_restores_history_key_presence(self):
        # A step appends to the current stage's history list; undo must
        # restore the exact _history, including not leaving an empty stage
        # behind that would perturb a later info_set key.
        env = _env(n_players=2, seed=0)
        hist_before = {k: list(v) for k, v in env._history.items()}
        token = env.step_in_place("call")
        env.undo(token)
        assert {k: list(v) for k, v in env._history.items()} == hist_before


class TestSettleWinnersFlag:
    """``settle_winners=False`` skips the concrete terminal settlement (hand
    ranking + chip distribution) that range-valued callers (vector regime)
    discard, but still snapshots ``terminal_contributions`` (the stake the
    vectorised payout reads).  Default ``True`` is unchanged."""

    def _to_terminal_step(self, seed):
        """Drive a HU env to a terminal; return the pre-terminal env (deepcopy)
        and the action that ends the hand."""
        env = _env(n_players=2, seed=seed)
        prev, last = None, None
        steps = 0
        while not env.is_terminal and steps < 60:
            a = "call" if "call" in env.legal_actions else "check"
            prev, last = copy.deepcopy(env), a
            env.step_in_place(a)
            steps += 1
        assert env.is_terminal and prev is not None
        return prev, last

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_false_sets_contributions_but_skips_distribution(self, seed):
        prev, action = self._to_terminal_step(seed)
        e_true, e_false = copy.deepcopy(prev), copy.deepcopy(prev)
        e_true.step_in_place(action, settle_winners=True)
        e_false.step_in_place(action, settle_winners=False)

        # Both reach the terminal and snapshot the *same* matched-stake stake.
        assert e_true.is_terminal and e_false.is_terminal
        assert e_false.terminal_contributions is not None
        assert e_false.terminal_contributions == e_true.terminal_contributions

        # settle_winners=True distributes (pot reset, winner paid); False does not.
        assert e_true.pot.total == 0
        assert e_false.pot.total > 0
        true_chips = [p.n_chips for p in e_true.players]
        false_chips = [p.n_chips for p in e_false.players]
        assert true_chips != false_chips

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_false_still_round_trips_under_undo(self, seed):
        prev, action = self._to_terminal_step(seed)
        env = copy.deepcopy(prev)
        token = env.step_in_place(action, settle_winners=False)
        env.undo(token)
        assert_env_equal(env, prev)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_vector_payout_identical_regardless_of_flag(self, seed):
        # The value the vector regime actually reads must not depend on the flag.
        prev, action = self._to_terminal_step(seed)
        e_true, e_false = copy.deepcopy(prev), copy.deepcopy(prev)
        e_true.step_in_place(action, settle_winners=True)
        e_false.step_in_place(action, settle_winners=False)
        opp = np.ones(e_true.n_combos) / e_true.n_combos
        vt = e_true.vector_payout(0, 1, opp, river=None)
        vf = e_false.vector_payout(0, 1, opp, river=None)
        np.testing.assert_array_equal(vt, vf)
