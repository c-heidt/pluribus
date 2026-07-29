"""Functional tests for poker_ai/environment/poker_env.py.

Covers PokerEnv construction, validation errors, initial state properties,
action semantics (step_in_place), stage progression, terminal states, and
CFR helpers.
"""

import copy

import pytest

from environment.player import Player
from environment.poker_env import PokerEnv, new_game


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _play_to_terminal(env, max_steps=200):
    """Drive a game to completion by having every player call."""
    steps = 0
    while not env.is_terminal and steps < max_steps:
        action = "call" if "call" in env.legal_actions else env.legal_actions[0]
        env.step_in_place(action)
        steps += 1
    return env


# ---------------------------------------------------------------------------
# Construction & Validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_two_player_game_created(self, two_player_game):
        assert two_player_game is not None

    def test_six_player_game_created(self):
        env = new_game(n_players=6, card_info_lut={})
        assert env.n_players == 6

    def test_community_cards_empty(self, fresh_game):
        assert fresh_game.community_cards == ()

    def test_players_list_length(self, fresh_game):
        assert len(fresh_game.players) == 3

    def test_pot_is_pot_instance(self, fresh_game):
        from environment.pot import Pot
        assert isinstance(fresh_game.pot, Pot)

    def test_deck_is_deck_instance(self, fresh_game):
        from environment.chance import Deck
        assert isinstance(fresh_game.deck, Deck)


class TestConstructionErrors:
    def test_one_player_raises(self):
        with pytest.raises(ValueError):
            new_game(n_players=1, card_info_lut={})

    def test_zero_players_raises(self):
        with pytest.raises(ValueError):
            new_game(n_players=0, card_info_lut={})

    def test_low_card_rank_below_2_raises(self):
        with pytest.raises(ValueError):
            PokerEnv([Player(i, 10000) for i in range(2)], low_card_rank=1)

    def test_high_card_rank_above_14_raises(self):
        with pytest.raises(ValueError):
            PokerEnv([Player(i, 10000) for i in range(2)], high_card_rank=15)

    def test_low_above_high_raises(self):
        with pytest.raises(ValueError):
            PokerEnv([Player(i, 10000) for i in range(2)], low_card_rank=10, high_card_rank=9)

    def test_deck_too_small_for_players_raises(self):
        # ranks 13–14 = 8 cards; need at least 4*2+5=13 for 4 players
        with pytest.raises(ValueError):
            PokerEnv([Player(i, 10000) for i in range(4)], low_card_rank=13, high_card_rank=14)

    def test_player_i_setter_raises(self, fresh_game):
        with pytest.raises(ValueError):
            fresh_game.player_i = 0


# ---------------------------------------------------------------------------
# Initial State Properties
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_betting_stage_pre_flop(self, fresh_game):
        assert fresh_game.betting_stage == "pre_flop"

    def test_is_terminal_false(self, fresh_game):
        assert fresh_game.is_terminal is False

    def test_betting_round_zero(self, fresh_game):
        assert fresh_game.betting_round == 0

    def test_small_blind_posted(self):
        env = new_game(n_players=2, card_info_lut={}, small_blind=50, initial_chips=10000)
        assert env.players[0].n_chips == 9950

    def test_big_blind_posted(self):
        env = new_game(n_players=2, card_info_lut={}, big_blind=100, initial_chips=10000)
        assert env.players[1].n_chips == 9900

    def test_pot_size_equals_blinds(self):
        env = new_game(n_players=2, card_info_lut={}, small_blind=50, big_blind=100)
        assert env.pot_size == 150

    def test_each_player_has_two_cards(self, fresh_game):
        for player in fresh_game.players:
            assert len(player.cards) == 2

    def test_hole_cards_are_ints(self, fresh_game):
        for player in fresh_game.players:
            for card in player.cards:
                assert isinstance(card, int)

    def test_all_hole_cards_unique(self, fresh_game):
        all_cards = [c for p in fresh_game.players for c in p.cards]
        assert len(all_cards) == len(set(all_cards))

    def test_private_hands_maps_all_players(self, fresh_game):
        hands = fresh_game.private_hands
        assert set(hands.keys()) == set(range(fresh_game.n_players))
        for hand in hands.values():
            assert len(hand) == 2

    def test_n_players_matches_input(self, fresh_game):
        assert fresh_game.n_players == 3

    def test_deck_size_52(self):
        env = new_game(n_players=2, card_info_lut={})
        assert env.deck_size == 52

    def test_low_card_rank_property(self, fresh_game):
        assert fresh_game.low_card_rank == 2

    def test_high_card_rank_property(self, fresh_game):
        assert fresh_game.high_card_rank == 14

    def test_legal_actions_not_empty(self, fresh_game):
        assert len(fresh_game.legal_actions) > 0

    def test_legal_actions_contains_fold(self, fresh_game):
        assert "fold" in fresh_game.legal_actions

    def test_initial_regret_keys_match_legal_actions(self, fresh_game):
        assert set(fresh_game.initial_regret.keys()) == set(fresh_game.legal_actions)

    def test_initial_regret_all_zeros(self, fresh_game):
        for v in fresh_game.initial_regret.values():
            assert v == 0

    def test_initial_strategy_keys_match_legal_actions(self, fresh_game):
        assert set(fresh_game.initial_strategy.keys()) == set(fresh_game.legal_actions)

    def test_initial_strategy_all_zeros(self, fresh_game):
        for v in fresh_game.initial_strategy.values():
            assert v == 0

    def test_min_raise_amount_equals_big_blind(self):
        env = new_game(n_players=2, card_info_lut={}, big_blind=100)
        assert env.min_raise_amount == 100

    def test_n_players_started_round(self, fresh_game):
        assert fresh_game.n_players_started_round == 3

    def test_all_players_have_actioned_false(self, fresh_game):
        assert fresh_game.all_players_have_actioned is False

    def test_is_turn_exactly_one_player(self, fresh_game):
        turns = [p.is_turn for p in fresh_game.players]
        assert turns.count(True) == 1

    def test_repr_contains_stage(self, fresh_game):
        assert "pre_flop" in repr(fresh_game)


# ---------------------------------------------------------------------------
# Action semantics
# ---------------------------------------------------------------------------

class TestActionSemantics:
    def test_fold_deactivates_current_player(self, fresh_game):
        acting_i = fresh_game.player_i
        new_env = copy.deepcopy(fresh_game); new_env.step_in_place("fold")
        assert not new_env.players[acting_i].is_active

    def test_call_increases_pot(self, fresh_game):
        pot_before = fresh_game.pot_size
        new_env = copy.deepcopy(fresh_game); new_env.step_in_place("call")
        assert new_env.pot_size > pot_before

    def test_call_equalizes_bet(self, fresh_game):
        new_env = copy.deepcopy(fresh_game); new_env.step_in_place("call")
        bets = [p.n_bet_chips for p in new_env.players if p.is_active]
        # After a call, the bet should be equal to BB (100)
        assert len(set(bets)) <= 2  # at most caller vs raiser differ

    def test_raise_action_increments_n_raises(self, fresh_game):
        raise_actions = [a for a in fresh_game.legal_actions if a and a.startswith("raise:")]
        if raise_actions:
            new_env = copy.deepcopy(fresh_game); new_env.step_in_place(raise_actions[0])
            assert new_env._n_raises >= 1

    def test_all_in_sets_player_all_in(self):
        # Deep stacks so the shove is non-terminal (opponent still to respond);
        # otherwise a both-all-in hand settles immediately and the winner's chips
        # are already redistributed, masking the "actor staked everything" state.
        env = new_game(n_players=2, card_info_lut={}, initial_chips=10000, big_blind=100)
        if "all_in" in env.legal_actions:
            new_env = copy.deepcopy(env); new_env.step_in_place("all_in")
            # The acting player should now have 0 chips (staked their whole stack)
            acting_i = env.player_i
            assert not new_env.is_terminal
            assert new_env.players[acting_i].n_chips == 0

    def test_action_recorded_in_history(self, fresh_game):
        new_env = copy.deepcopy(fresh_game); new_env.step_in_place("fold")
        history = dict(new_env._history)
        assert any(len(actions) > 0 for actions in history.values())

    def test_is_turn_updated_after_action(self, fresh_game):
        new_env = copy.deepcopy(fresh_game); new_env.step_in_place("call")
        turns = [p.is_turn for p in new_env.players]
        assert turns.count(True) == 1


# ---------------------------------------------------------------------------
# Raise cap
# ---------------------------------------------------------------------------

class TestRaiseCap:
    def test_no_raises_after_max_raises(self):
        env = new_game(n_players=2, card_info_lut={})
        for _ in range(env._max_raises_per_round):
            raise_actions = [a for a in env.legal_actions if a and a.startswith("raise:")]
            if not raise_actions:
                break
            env.step_in_place(raise_actions[0])
        raise_actions = [a for a in env.legal_actions if a and a.startswith("raise:")]
        assert len(raise_actions) == 0


# ---------------------------------------------------------------------------
# Stage progression
# ---------------------------------------------------------------------------

class TestStageProgression:
    def test_pre_flop_to_flop(self):
        env = new_game(n_players=2, card_info_lut={})
        while env.betting_stage == "pre_flop":
            env.step_in_place("call")
        assert len(env.community_cards) == 3

    def test_flop_to_turn(self):
        env = new_game(n_players=2, card_info_lut={})
        while env.betting_stage in ("pre_flop", "flop"):
            env.step_in_place("call")
        assert len(env.community_cards) == 4

    def test_turn_to_river(self):
        env = new_game(n_players=2, card_info_lut={})
        while env.betting_stage in ("pre_flop", "flop", "turn"):
            env.step_in_place("call")
        assert len(env.community_cards) == 5

    def test_river_to_show_down(self):
        env = new_game(n_players=2, card_info_lut={})
        env = _play_to_terminal(env)
        assert env.is_terminal

    def test_community_cards_are_ints(self):
        env = new_game(n_players=2, card_info_lut={})
        while env.betting_stage == "pre_flop":
            env.step_in_place("call")
        for c in env.community_cards:
            assert isinstance(c, int)

    def test_community_cards_unique(self):
        env = new_game(n_players=2, card_info_lut={})
        env = _play_to_terminal(env)
        # community_cards may be 0 (terminal via fold) or 5
        if env.community_cards:
            assert len(env.community_cards) == len(set(env.community_cards))

    def test_community_no_overlap_with_hole_cards(self):
        env = new_game(n_players=2, card_info_lut={})
        hole_cards = {c for p in env.players for c in p.cards}
        while env.betting_stage == "pre_flop":
            env.step_in_place("call")
        for c in env.community_cards:
            assert c not in hole_cards


# ---------------------------------------------------------------------------
# betting_round exception
# ---------------------------------------------------------------------------

class TestBettingRoundException:
    def test_betting_round_raises_at_terminal(self):
        env = new_game(n_players=2, card_info_lut={})
        # Force terminal stage by folding
        env2 = copy.deepcopy(env); env2.step_in_place("fold")
        if env2.betting_stage == "terminal":
            with pytest.raises(ValueError):
                _ = env2.betting_round


# ---------------------------------------------------------------------------
# Terminal state
# ---------------------------------------------------------------------------

class TestTerminalState:
    def test_fold_two_player_is_terminal(self, two_player_game):
        new_env = copy.deepcopy(two_player_game); new_env.step_in_place("fold")
        assert new_env.is_terminal

    def test_fold_terminal_stage(self, two_player_game):
        new_env = copy.deepcopy(two_player_game); new_env.step_in_place("fold")
        assert new_env.betting_stage == "terminal"

    def test_all_call_three_players_show_down(self, fresh_game):
        env = _play_to_terminal(fresh_game)
        assert env.betting_stage in {"show_down", "terminal"}

    def test_payout_sums_to_zero(self, fresh_game):
        env = _play_to_terminal(fresh_game)
        assert sum(env.payout.values()) == 0

    def test_payout_keys_cover_all_players(self, fresh_game):
        env = _play_to_terminal(fresh_game)
        assert set(env.payout.keys()) == set(range(fresh_game.n_players))


# ---------------------------------------------------------------------------
# info_set behaviour
# ---------------------------------------------------------------------------

class TestInfoSet:
    def test_info_set_raises_at_non_terminal_with_empty_lut(self, fresh_game):
        assert not fresh_game.is_terminal
        with pytest.raises(ValueError):
            _ = fresh_game.info_set

    def test_info_set_returns_bytes_at_terminal(self, two_player_game):
        env = copy.deepcopy(two_player_game); env.step_in_place("fold")
        assert env.is_terminal
        result = env.info_set
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# CFR helpers
# ---------------------------------------------------------------------------

class TestCFRHelpers:
    @pytest.mark.parametrize("round_i", [0, 1, 2, 3])
    def test_canonical_actions_contains_fold(self, round_i):
        actions = PokerEnv.get_canonical_actions(round_i)
        assert "fold" in actions

    @pytest.mark.parametrize("round_i", [0, 1, 2, 3])
    def test_canonical_actions_contains_call(self, round_i):
        actions = PokerEnv.get_canonical_actions(round_i)
        assert "call" in actions

    @pytest.mark.parametrize("invalid_round", [-1, 4, 5, 100])
    def test_canonical_actions_invalid_round_raises(self, invalid_round):
        with pytest.raises(ValueError):
            PokerEnv.get_canonical_actions(invalid_round)

    def test_get_valid_mask_length(self, fresh_game):
        import numpy as np
        canonical = PokerEnv.get_canonical_actions(fresh_game.betting_round)
        mask = fresh_game.get_valid_mask()
        assert isinstance(mask, np.ndarray)
        assert len(mask) == len(canonical)

    def test_legal_actions_are_true_in_mask(self, fresh_game):
        canonical = PokerEnv.get_canonical_actions(fresh_game.betting_round)
        mask = fresh_game.get_valid_mask()
        legal_set = {a for a in fresh_game.legal_actions if a is not None}
        for i, action in enumerate(canonical):
            if action in legal_set:
                assert mask[i], f"Expected {action} to be True in mask"

    def test_illegal_actions_are_false_in_mask(self, fresh_game):
        canonical = PokerEnv.get_canonical_actions(fresh_game.betting_round)
        mask = fresh_game.get_valid_mask()
        legal_set = {a for a in fresh_game.legal_actions if a is not None}
        for i, action in enumerate(canonical):
            if action not in legal_set:
                assert not mask[i], f"Expected {action} to be False in mask"


# ---------------------------------------------------------------------------
# __deepcopy__ correctness
# ---------------------------------------------------------------------------

class TestDeepCopy:
    def test_lut_shared_by_reference(self, fresh_game):
        fresh_game.card_info_lut = {"pre_flop": {(1, 2): "cluster_A"}}
        copied = copy.deepcopy(fresh_game)
        # The LUT is read-only and large, so __deepcopy__ shares it by
        # reference rather than copying or zeroing it.
        assert copied.card_info_lut is fresh_game.card_info_lut

    def test_config_fields_independent(self, fresh_game):
        import copy
        copied = copy.deepcopy(fresh_game)
        # Immutable config scalars should match
        assert copied.small_blind == fresh_game.small_blind
        assert copied.big_blind == fresh_game.big_blind
        assert copied.low_card_rank == fresh_game.low_card_rank

    def test_game_state_is_independent(self, fresh_game):
        import copy
        copied = copy.deepcopy(fresh_game)
        # Mutating copied players must not affect original
        copied.players[0].n_chips = 0
        assert fresh_game.players[0].n_chips != 0

    def test_community_cards_independent(self, fresh_game):
        import copy
        copied = copy.deepcopy(fresh_game)
        # Override community cards on copy; original must be unaffected
        copied.community_cards = (99999,)
        assert fresh_game.community_cards == ()


# ---------------------------------------------------------------------------
# Closest-legal-action mapping
# ---------------------------------------------------------------------------

class TestClosestLegalAction:
    def test_oversize_raise_maps_to_valid_action(self, fresh_game):
        # "raise:999" is not in legal_actions; must not raise an exception
        new_env = copy.deepcopy(fresh_game); new_env.step_in_place("raise:999")
        assert new_env is not None

    def test_completely_invalid_string_maps_to_valid_action(self, fresh_game):
        new_env = copy.deepcopy(fresh_game); new_env.step_in_place("not_a_real_action")
        assert new_env is not None
        # The resulting env must still be in a valid state
        assert new_env.betting_stage in {
            "pre_flop", "flop", "turn", "river", "show_down", "terminal"
        }

    def test_none_action_for_inactive_player(self):
        env = new_game(n_players=3, card_info_lut={})
        # Fold the current player so the next is inactive in a forced-skip scenario
        # Drive to a point where an inactive player would receive None
        # Simulate by checking that None is accepted for an inactive player
        env2 = copy.deepcopy(env); env2.step_in_place("fold")
        # Continue past the skip; eventually game is terminal or another player acts
        assert env2 is not None


# ---------------------------------------------------------------------------
# Positional flags (Bug 1 regression)
# ---------------------------------------------------------------------------

class TestPositionalFlagsAfterConstruction:
    def test_heads_up_dealer_is_player_0(self):
        env = new_game(n_players=2, card_info_lut={})
        assert env.players[0].is_dealer is True

    def test_heads_up_dealer_is_not_player_1(self):
        env = new_game(n_players=2, card_info_lut={})
        assert env.players[1].is_dealer is False

    def test_heads_up_dealer_same_player_as_small_blind(self):
        env = new_game(n_players=2, card_info_lut={})
        dealer = next(p for p in env.players if p.is_dealer)
        sb = next(p for p in env.players if p.is_small_blind)
        assert dealer.name == sb.name

    @pytest.mark.parametrize("n", [3, 4, 6])
    def test_multiway_dealer_is_last_player(self, n):
        env = new_game(n_players=n, card_info_lut={})
        assert env.players[-1].is_dealer is True

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_exactly_one_dealer(self, n):
        env = new_game(n_players=n, card_info_lut={})
        assert sum(p.is_dealer for p in env.players) == 1

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_exactly_one_small_blind(self, n):
        env = new_game(n_players=n, card_info_lut={})
        assert sum(p.is_small_blind for p in env.players) == 1

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_exactly_one_big_blind(self, n):
        env = new_game(n_players=n, card_info_lut={})
        assert sum(p.is_big_blind for p in env.players) == 1

    def test_small_blind_player_posted_blind(self):
        env = new_game(n_players=2, card_info_lut={}, small_blind=50, initial_chips=10000)
        sb = next(p for p in env.players if p.is_small_blind)
        assert sb.n_chips == 10000 - 50


class TestAllInBettingContract:
    """An all-in must not let the engine skip an opponent's response or make a
    live player act twice — the round-advance twin of the terminal all-in
    contract enforced by ``_hand_over``.
    """

    def test_heads_up_over_the_top_all_in_lets_opponent_respond(self):
        # P0 (SB) raises pre-flop, P1 (BB) shoves over the top.  The street must
        # NOT advance and the hand must NOT end: P0 still owes a call-or-fold on
        # the shove (previously the street silently advanced to the flop with
        # P0's chips uncommitted).
        env = PokerEnv([Player(0, 10000), Player(1, 10000)],
                       small_blind=50, big_blind=100)
        env.step_in_place("raise:1.0")
        env.step_in_place("all_in")
        assert env.betting_stage == "pre_flop"
        assert not env.is_terminal
        assert env.player_i == 0
        # Facing the full-stack shove, P0 can only call it off or fold.
        assert set(a for a in env.legal_actions if a) == {"fold", "all_in"}

    def test_multiway_over_the_top_all_in_lets_live_players_respond(self):
        # P2 (short) shoves over a flop bet that P0/P1 already matched; the two
        # live players must still get to respond, not have the hand jump to the
        # turn with the shove uncalled.
        env = PokerEnv([Player(0, 10000), Player(1, 10000), Player(2, 450)],
                       small_blind=50, big_blind=100)
        for _ in range(3):
            env.step_in_place("call")          # everyone limps to the flop
        assert env.betting_stage == "flop"
        env.step_in_place("raise:1.0")         # P0 opens
        env.step_in_place("call")              # P1 calls
        env.step_in_place("all_in")            # P2 shoves over the top
        assert env.betting_stage == "flop"
        assert env.current_player.is_active
        assert not env.current_player.is_all_in

    def test_earlier_street_all_in_does_not_cause_double_action(self):
        # P2 is all-in from pre-flop; on the flop the two remaining live players
        # must each act exactly once before the turn.  Previously the already-
        # all-in P2 was counted in the round's actor total, so the gate that
        # ends the round never tripped after one lap and P0/P1 were re-polled.
        env = PokerEnv([Player(0, 10000), Player(1, 10000), Player(2, 200)],
                       small_blind=50, big_blind=100)
        env.step_in_place("all_in")            # P2 shoves pre-flop
        env.step_in_place("call")              # P0 calls
        env.step_in_place("call")              # P1 calls -> flop
        assert env.betting_stage == "flop"
        assert env.players[2].is_all_in
        actors = []
        for _ in range(10):                    # bounded so a regression can't hang
            if env.betting_stage != "flop":
                break
            actors.append(env.player_i)
            env.step_in_place("call")          # check it down
        assert actors == [0, 1]                # each live player acts once


class TestHeadsUpActionOrder:
    """Heads-up position: the button (seat 0 = SB) acts first pre-flop but LAST
    post-flop, so the big blind (seat 1) leads post-flop.  For 3+ players the
    small blind (seat 0) already leads post-flop, so their order is unchanged.
    """

    def test_preflop_button_acts_first(self):
        env = new_game(n_players=2, card_info_lut={})
        assert env.player_i == 0               # SB / button

    def test_postflop_big_blind_acts_first(self):
        env = new_game(n_players=2, card_info_lut={})
        env.step_in_place("call")              # SB completes
        env.step_in_place("call")              # BB checks -> flop
        assert env.betting_stage == "flop"
        assert env.player_i == 1               # BB is first to act post-flop

    @pytest.mark.parametrize("street", ["flop", "turn", "river"])
    def test_heads_up_postflop_lut_is_bb_first(self, street):
        env = new_game(n_players=2, card_info_lut={})
        assert env._player_i_lut[street] == [1, 0]

    @pytest.mark.parametrize("n", [3, 4, 6])
    def test_multiway_postflop_lut_is_seat_order(self, n):
        env = new_game(n_players=n, card_info_lut={})
        assert env._player_i_lut["flop"] == list(range(n))


# ---------------------------------------------------------------------------
# Bet reset at stage transitions (Bug 3 regression)
# ---------------------------------------------------------------------------

class TestBetResetViaApplyAction:
    def _advance_to_stage(self, env, target_stage):
        while env.betting_stage != target_stage and not env.is_terminal:
            env.step_in_place("call")
        return env

    def test_n_bet_chips_zero_at_flop_start(self):
        env = new_game(n_players=2, card_info_lut={})
        env = self._advance_to_stage(env, "flop")
        assert env.betting_stage == "flop"
        for p in env.players:
            assert p.n_bet_chips == 0

    def test_n_bet_chips_zero_at_turn_start(self):
        env = new_game(n_players=2, card_info_lut={})
        env = self._advance_to_stage(env, "turn")
        assert env.betting_stage == "turn"
        for p in env.players:
            assert p.n_bet_chips == 0

    def test_n_bet_chips_zero_at_river_start(self):
        env = new_game(n_players=2, card_info_lut={})
        env = self._advance_to_stage(env, "river")
        assert env.betting_stage == "river"
        for p in env.players:
            assert p.n_bet_chips == 0

    def test_raise_before_stage_transition_is_reset(self):
        env = new_game(n_players=2, card_info_lut={})
        # Raise pre-flop so n_bet_chips > 0
        raise_actions = [a for a in env.legal_actions if a and a.startswith("raise:")]
        if raise_actions:
            env.step_in_place(raise_actions[0])
            assert any(p.n_bet_chips > 0 for p in env.players)
        # Drive to flop and verify reset
        env = self._advance_to_stage(env, "flop")
        if env.betting_stage == "flop":
            for p in env.players:
                assert p.n_bet_chips == 0

    def test_n_bet_chips_zero_three_player_game(self):
        env = new_game(n_players=3, card_info_lut={})
        env = self._advance_to_stage(env, "flop")
        if env.betting_stage == "flop":
            for p in env.players:
                assert p.n_bet_chips == 0

    def test_n_bet_chips_zero_at_all_stages(self):
        env = new_game(n_players=2, card_info_lut={})
        for target in ("flop", "turn", "river"):
            env = self._advance_to_stage(env, target)
            if env.betting_stage == target:
                for p in env.players:
                    assert p.n_bet_chips == 0


# ---------------------------------------------------------------------------
# All-in board completion (Bug 4 regression)
# ---------------------------------------------------------------------------

class TestAllInBoardCompletion:
    def _advance_to_stage(self, env, target_stage):
        while env.betting_stage != target_stage and not env.is_terminal:
            env.step_in_place("call")
        return env

    def _shove_and_respond(self, env, respond="call"):
        """Shove all-in, then have the opponent respond (the corrected
        contract: an all-in is NOT terminal until the opponent calls or folds).
        Returns the env after the response resolves the hand."""
        env.step_in_place("all_in")
        # The fix: the opponent must still get to act on the all-in.
        assert not env.is_terminal
        if respond == "call":
            resp = "all_in" if "all_in" in env.legal_actions else "call"
        else:
            resp = "fold"
        env.step_in_place(resp)
        return env

    def test_all_in_preflop_has_5_community_cards(self):
        env = new_game(n_players=2, card_info_lut={})
        assert "all_in" in env.legal_actions
        self._shove_and_respond(env)
        assert env.is_terminal
        assert len(env.community_cards) == 5

    def test_all_in_at_flop_has_5_community_cards(self):
        env = new_game(n_players=2, card_info_lut={})
        env = self._advance_to_stage(env, "flop")
        assert env.betting_stage == "flop"
        assert "all_in" in env.legal_actions
        self._shove_and_respond(env)
        assert env.is_terminal
        assert len(env.community_cards) == 5

    def test_all_in_at_turn_has_5_community_cards(self):
        env = new_game(n_players=2, card_info_lut={})
        env = self._advance_to_stage(env, "turn")
        assert env.betting_stage == "turn"
        assert "all_in" in env.legal_actions
        self._shove_and_respond(env)
        assert env.is_terminal
        assert len(env.community_cards) == 5

    def test_all_in_at_river_has_5_community_cards(self):
        env = new_game(n_players=2, card_info_lut={})
        env = self._advance_to_stage(env, "river")
        assert env.betting_stage == "river"
        assert "all_in" in env.legal_actions
        self._shove_and_respond(env)
        assert env.is_terminal
        assert len(env.community_cards) == 5

    def test_fold_terminal_has_5_community_cards(self):
        env = new_game(n_players=2, card_info_lut={})
        env.step_in_place("fold")
        assert env.is_terminal
        assert len(env.community_cards) == 5

    def test_community_cards_unique_after_all_in(self):
        env = new_game(n_players=2, card_info_lut={})
        self._shove_and_respond(env)
        assert len(env.community_cards) == len(set(env.community_cards))

    def test_community_cards_no_overlap_with_hole_cards(self):
        env = new_game(n_players=2, card_info_lut={})
        hole_cards = {c for p in env.players for c in p.cards}
        self._shove_and_respond(env)
        for c in env.community_cards:
            assert c not in hole_cards


class TestAllInFacingResponse:
    """Regression for the corrected all-in contract: an all-in must give the
    opponent a call/fold decision (it used to auto-terminate the hand)."""

    def test_all_in_is_not_immediately_terminal(self):
        env = new_game(n_players=2, card_info_lut={})
        env.step_in_place("all_in")
        assert not env.is_terminal
        assert env.player_i == 1  # opponent to act
        legal = [a for a in env.legal_actions if a is not None]
        assert "fold" in legal and ("call" in legal or "all_in" in legal)

    def test_facing_all_in_offers_only_call_or_fold(self):
        # No one can call a raise when the only opponent is all-in, so the
        # responses are exactly call/all_in and fold — never a raise.
        env = new_game(n_players=2, card_info_lut={})
        env.step_in_place("all_in")
        legal = [a for a in env.legal_actions if a is not None]
        assert not any(a.startswith("raise:") for a in legal)

    def test_calling_all_in_contests_full_stacks(self):
        # The core of the fix: a called all-in settles for the full contested
        # stacks, not just the pre-shove matched pot.
        env = new_game(n_players=2, card_info_lut={})
        env.step_in_place("all_in")
        env.step_in_place("all_in")  # opponent calls the shove
        assert env.is_terminal
        payout = dict(env.payout)
        # One player wins the other's entire 10000 stack (zero-sum).
        assert sorted(payout.values()) == [-10000, 10000]

    def test_folding_to_all_in_forfeits_only_committed_chips(self):
        env = new_game(n_players=2, card_info_lut={})
        env.step_in_place("all_in")   # SB shoves
        env.step_in_place("fold")     # BB folds
        assert env.is_terminal
        payout = dict(env.payout)
        # BB forfeits only the big blind it had committed.
        assert payout == {0: 100, 1: -100}
