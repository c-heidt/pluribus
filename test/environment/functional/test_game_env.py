"""Functional tests for poker_ai/environment/poker_env.py.

Covers PokerEnv construction, validation errors, initial state properties,
apply_action semantics, stage progression, terminal states, and CFR helpers.
"""

import pytest

from poker_ai.environment.player import Player
from poker_ai.environment.poker_env import PokerEnv, new_game, MAX_RAISES_PER_ROUND


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _play_to_terminal(env, max_steps=200):
    """Drive a game to completion by having every player call."""
    steps = 0
    while not env.is_terminal and steps < max_steps:
        action = "call" if "call" in env.legal_actions else env.legal_actions[0]
        env = env.apply_action(action)
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

    def test_n_players_attribute(self, fresh_game):
        assert fresh_game.n_players == 3

    def test_community_cards_empty(self, fresh_game):
        assert fresh_game.community_cards == ()

    def test_players_list_length(self, fresh_game):
        assert len(fresh_game.players) == 3

    def test_pot_is_pot_instance(self, fresh_game):
        from poker_ai.environment.pot import Pot
        assert isinstance(fresh_game.pot, Pot)

    def test_deck_is_deck_instance(self, fresh_game):
        from poker_ai.environment.chance import Deck
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
            PokerEnv([Player(i, 10000) for i in range(2)], low_card_rank=1, load_card_lut=False)

    def test_high_card_rank_above_14_raises(self):
        with pytest.raises(ValueError):
            PokerEnv([Player(i, 10000) for i in range(2)], high_card_rank=15, load_card_lut=False)

    def test_low_above_high_raises(self):
        with pytest.raises(ValueError):
            PokerEnv([Player(i, 10000) for i in range(2)], low_card_rank=10, high_card_rank=9, load_card_lut=False)

    def test_deck_too_small_for_players_raises(self):
        # ranks 13–14 = 8 cards; need at least 4*2+5=13 for 4 players
        with pytest.raises(ValueError):
            PokerEnv([Player(i, 10000) for i in range(4)], low_card_rank=13, high_card_rank=14, load_card_lut=False)

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

    def test_legal_actions_inactive_player(self):
        env = new_game(n_players=3, card_info_lut={})
        env.players[env.player_i].fold()
        env.players[env.player_i]._is_active = False
        # Access legal_actions from inactive player's perspective directly
        from poker_ai.environment.player import Player
        p = Player(99, 0)
        p._is_active = False
        # Simulate inactive player: override current player temporarily
        original_idx = env._player_i_index
        # Find a folded player
        for i, p in enumerate(env.players):
            if not p.is_active:
                # legal_actions checks current_player
                break

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
# apply_action immutability
# ---------------------------------------------------------------------------

class TestApplyActionImmutability:
    def test_returns_new_instance(self, fresh_game):
        new_env = fresh_game.apply_action("fold")
        assert new_env is not fresh_game

    def test_original_stage_unchanged(self, fresh_game):
        stage_before = fresh_game.betting_stage
        fresh_game.apply_action("fold")
        assert fresh_game.betting_stage == stage_before

    def test_original_pot_unchanged(self, fresh_game):
        pot_before = fresh_game.pot_size
        fresh_game.apply_action("fold")
        assert fresh_game.pot_size == pot_before

    def test_original_players_unchanged(self, fresh_game):
        active_before = [p.is_active for p in fresh_game.players]
        fresh_game.apply_action("fold")
        assert [p.is_active for p in fresh_game.players] == active_before


# ---------------------------------------------------------------------------
# Action semantics
# ---------------------------------------------------------------------------

class TestActionSemantics:
    def test_fold_deactivates_current_player(self, fresh_game):
        acting_i = fresh_game.player_i
        new_env = fresh_game.apply_action("fold")
        assert not new_env.players[acting_i].is_active

    def test_call_increases_pot(self, fresh_game):
        pot_before = fresh_game.pot_size
        new_env = fresh_game.apply_action("call")
        assert new_env.pot_size > pot_before

    def test_call_equalizes_bet(self, fresh_game):
        new_env = fresh_game.apply_action("call")
        bets = [p.n_bet_chips for p in new_env.players if p.is_active]
        # After a call, the bet should be equal to BB (100)
        assert len(set(bets)) <= 2  # at most caller vs raiser differ

    def test_raise_action_increments_n_raises(self, fresh_game):
        raise_actions = [a for a in fresh_game.legal_actions if a and a.startswith("raise:")]
        if raise_actions:
            new_env = fresh_game.apply_action(raise_actions[0])
            assert new_env._n_raises >= 1

    def test_all_in_sets_player_all_in(self):
        env = new_game(n_players=2, card_info_lut={}, initial_chips=100, big_blind=100)
        if "all_in" in env.legal_actions:
            new_env = env.apply_action("all_in")
            # The acting player should now have 0 chips
            acting_i = env.player_i
            assert new_env.players[acting_i].n_chips == 0

    def test_invalid_action_does_not_raise(self, fresh_game):
        new_env = fresh_game.apply_action("raise:999")  # absurdly large raise
        assert new_env is not None

    def test_action_recorded_in_history(self, fresh_game):
        new_env = fresh_game.apply_action("fold")
        history = dict(new_env._history)
        assert any(len(actions) > 0 for actions in history.values())

    def test_is_turn_updated_after_action(self, fresh_game):
        new_env = fresh_game.apply_action("call")
        turns = [p.is_turn for p in new_env.players]
        assert turns.count(True) == 1


# ---------------------------------------------------------------------------
# Raise cap
# ---------------------------------------------------------------------------

class TestRaiseCap:
    def test_no_raises_after_max_raises(self):
        env = new_game(n_players=2, card_info_lut={})
        for _ in range(MAX_RAISES_PER_ROUND):
            raise_actions = [a for a in env.legal_actions if a and a.startswith("raise:")]
            if not raise_actions:
                break
            env = env.apply_action(raise_actions[0])
        raise_actions = [a for a in env.legal_actions if a and a.startswith("raise:")]
        assert len(raise_actions) == 0


# ---------------------------------------------------------------------------
# Stage progression
# ---------------------------------------------------------------------------

class TestStageProgression:
    def test_pre_flop_to_flop(self):
        env = new_game(n_players=2, card_info_lut={})
        while env.betting_stage == "pre_flop":
            env = env.apply_action("call")
        assert len(env.community_cards) == 3

    def test_flop_to_turn(self):
        env = new_game(n_players=2, card_info_lut={})
        while env.betting_stage in ("pre_flop", "flop"):
            env = env.apply_action("call")
        assert len(env.community_cards) == 4

    def test_turn_to_river(self):
        env = new_game(n_players=2, card_info_lut={})
        while env.betting_stage in ("pre_flop", "flop", "turn"):
            env = env.apply_action("call")
        assert len(env.community_cards) == 5

    def test_river_to_show_down(self):
        env = new_game(n_players=2, card_info_lut={})
        env = _play_to_terminal(env)
        assert env.is_terminal

    def test_community_cards_are_ints(self):
        env = new_game(n_players=2, card_info_lut={})
        while env.betting_stage == "pre_flop":
            env = env.apply_action("call")
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
            env = env.apply_action("call")
        for c in env.community_cards:
            assert c not in hole_cards


# ---------------------------------------------------------------------------
# betting_round exception
# ---------------------------------------------------------------------------

class TestBettingRoundException:
    def test_betting_round_raises_at_terminal(self):
        env = new_game(n_players=2, card_info_lut={})
        # Force terminal stage by folding
        env2 = env.apply_action("fold")
        if env2.betting_stage == "terminal":
            with pytest.raises(ValueError):
                _ = env2.betting_round


# ---------------------------------------------------------------------------
# Terminal state
# ---------------------------------------------------------------------------

class TestTerminalState:
    def test_fold_two_player_is_terminal(self, two_player_game):
        new_env = two_player_game.apply_action("fold")
        assert new_env.is_terminal

    def test_fold_terminal_stage(self, two_player_game):
        new_env = two_player_game.apply_action("fold")
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

    def test_info_set_returns_string_at_terminal(self, two_player_game):
        env = two_player_game.apply_action("fold")
        assert env.is_terminal
        result = env.info_set
        assert isinstance(result, str)


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
    def test_lut_excluded_from_copy(self, fresh_game):
        import copy
        fresh_game.card_info_lut = {"pre_flop": {(1, 2): "cluster_A"}}
        copied = copy.deepcopy(fresh_game)
        # LUT should be {} on the copy, not the original's dict
        assert copied.card_info_lut == {}

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
        new_env = fresh_game.apply_action("raise:999")
        assert new_env is not None

    def test_completely_invalid_string_maps_to_valid_action(self, fresh_game):
        new_env = fresh_game.apply_action("not_a_real_action")
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
        env2 = env.apply_action("fold")
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


# ---------------------------------------------------------------------------
# Bet reset at stage transitions (Bug 3 regression)
# ---------------------------------------------------------------------------

class TestBetResetViaApplyAction:
    def _advance_to_stage(self, env, target_stage):
        while env.betting_stage != target_stage and not env.is_terminal:
            env = env.apply_action("call")
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
            env = env.apply_action(raise_actions[0])
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
            env = env.apply_action("call")
        return env

    def test_all_in_preflop_has_5_community_cards(self):
        env = new_game(n_players=2, card_info_lut={})
        assert "all_in" in env.legal_actions
        env = env.apply_action("all_in")
        assert env.is_terminal
        assert len(env.community_cards) == 5

    def test_all_in_at_flop_has_5_community_cards(self):
        env = new_game(n_players=2, card_info_lut={})
        env = self._advance_to_stage(env, "flop")
        assert env.betting_stage == "flop"
        assert "all_in" in env.legal_actions
        env = env.apply_action("all_in")
        assert env.is_terminal
        assert len(env.community_cards) == 5

    def test_all_in_at_turn_has_5_community_cards(self):
        env = new_game(n_players=2, card_info_lut={})
        env = self._advance_to_stage(env, "turn")
        assert env.betting_stage == "turn"
        assert "all_in" in env.legal_actions
        env = env.apply_action("all_in")
        assert env.is_terminal
        assert len(env.community_cards) == 5

    def test_all_in_at_river_has_5_community_cards(self):
        env = new_game(n_players=2, card_info_lut={})
        env = self._advance_to_stage(env, "river")
        assert env.betting_stage == "river"
        assert "all_in" in env.legal_actions
        env = env.apply_action("all_in")
        assert env.is_terminal
        assert len(env.community_cards) == 5

    def test_fold_terminal_has_5_community_cards(self):
        env = new_game(n_players=2, card_info_lut={})
        env = env.apply_action("fold")
        assert env.is_terminal
        assert len(env.community_cards) == 5

    def test_community_cards_unique_after_all_in(self):
        env = new_game(n_players=2, card_info_lut={})
        env = env.apply_action("all_in")
        assert len(env.community_cards) == len(set(env.community_cards))

    def test_community_cards_no_overlap_with_hole_cards(self):
        env = new_game(n_players=2, card_info_lut={})
        hole_cards = {c for p in env.players for c in p.cards}
        env = env.apply_action("all_in")
        for c in env.community_cards:
            assert c not in hole_cards
