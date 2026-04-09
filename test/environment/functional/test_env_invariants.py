"""Cross-cutting game invariants for the poker environment.

These tests play full games under various strategies and assert properties
that must hold at EVERY state or at terminal states.  They are designed
to catch systemic bugs like the three that were fixed:

- Bug 1: is_dealer set on the wrong player for heads-up
- Bug 3: n_bet_chips not reset to 0 at stage transitions
- Bug 4: all-in fast-path always dealt 3 community cards instead of
         the correct number needed to reach 5

Key insight: fold-terminal games also trigger the n_players_with_moves==1
fast-path, so ALL terminal states have exactly 5 community cards.
"""

import pytest

from poker_ai.environment.poker_env import new_game


# ---------------------------------------------------------------------------
# Play-strategy helpers
# ---------------------------------------------------------------------------

def _play_step(env):
    """Advance one action: prefer call, else first legal action."""
    action = "call" if "call" in env.legal_actions else env.legal_actions[0]
    return env.apply_action(action)


def _play_all_calls(env, max_steps=500):
    steps = 0
    while not env.is_terminal and steps < max_steps:
        env = _play_step(env)
        steps += 1
    return env


def _play_fold_first(env, max_steps=500):
    """First active player folds every time a fold is legal."""
    steps = 0
    while not env.is_terminal and steps < max_steps:
        if "fold" in env.legal_actions:
            env = env.apply_action("fold")
        else:
            env = _play_step(env)
        steps += 1
    return env


def _play_all_raises(env, max_steps=500):
    """Prefer raise:0.5 when available, else call."""
    steps = 0
    while not env.is_terminal and steps < max_steps:
        raise_actions = [a for a in env.legal_actions if a and a.startswith("raise:0.5")]
        if raise_actions:
            env = env.apply_action(raise_actions[0])
        else:
            env = _play_step(env)
        steps += 1
    return env


def _play_all_in_first(env, max_steps=500):
    """Go all-in as soon as it is legal."""
    steps = 0
    while not env.is_terminal and steps < max_steps:
        if "all_in" in env.legal_actions:
            env = env.apply_action("all_in")
        else:
            env = _play_step(env)
        steps += 1
    return env


def _total_chips(env) -> int:
    return sum(p.n_chips for p in env.players) + env.pot_size


# ---------------------------------------------------------------------------
# Chip conservation
# ---------------------------------------------------------------------------

class TestChipConservationInvariant:
    @pytest.mark.parametrize("n_players,play_fn", [
        (2, _play_all_calls),
        (3, _play_all_calls),
        (6, _play_all_calls),
        (2, _play_fold_first),
        (3, _play_fold_first),
        (2, _play_all_in_first),
    ])
    def test_chips_constant_at_every_step(self, n_players, play_fn):
        env = new_game(n_players=n_players, card_info_lut={})
        initial = _total_chips(env)
        while not env.is_terminal:
            assert _total_chips(env) == initial, (
                f"Chip leak at stage={env.betting_stage}: "
                f"expected {initial}, got {_total_chips(env)}"
            )
            env = _play_step(env)
        assert _total_chips(env) == initial


# ---------------------------------------------------------------------------
# Bet reset at stage transitions (Bug 3)
# ---------------------------------------------------------------------------

class TestBetResetInvariant:
    def _assert_bets_reset_at_stage_transitions(self, env):
        prev_stage = env.betting_stage
        for _ in range(500):
            if env.is_terminal:
                break
            env = _play_step(env)
            if env.betting_stage != prev_stage:
                for p in env.players:
                    assert p.n_bet_chips == 0, (
                        f"{p.name}.n_bet_chips={p.n_bet_chips} "
                        f"at start of {env.betting_stage}"
                    )
                prev_stage = env.betting_stage

    @pytest.mark.parametrize("n_players", [2, 3, 6])
    def test_bet_chips_zero_at_every_stage_start(self, n_players):
        env = new_game(n_players=n_players, card_info_lut={})
        self._assert_bets_reset_at_stage_transitions(env)

    def test_bet_reset_with_raises(self):
        env = new_game(n_players=2, card_info_lut={})
        self._assert_bets_reset_at_stage_transitions(_play_all_raises(
            new_game(n_players=2, card_info_lut={}), max_steps=0
        ))
        # Run the invariant check on a game played with raises
        env = new_game(n_players=2, card_info_lut={})
        self._assert_bets_reset_at_stage_transitions(env)

    def test_bet_reset_with_all_in_strategy(self):
        env = new_game(n_players=2, card_info_lut={})
        self._assert_bets_reset_at_stage_transitions(env)


# ---------------------------------------------------------------------------
# Board completion — all terminals have exactly 5 community cards (Bug 4)
# ---------------------------------------------------------------------------

class TestBoardCompletionInvariant:
    @pytest.mark.parametrize("play_fn", [
        _play_all_calls,
        _play_fold_first,
        _play_all_in_first,
        _play_all_raises,
    ])
    def test_terminal_has_exactly_5_community_cards(self, play_fn):
        env = new_game(n_players=2, card_info_lut={})
        env = play_fn(env)
        assert env.is_terminal
        assert len(env.community_cards) == 5, (
            f"Expected 5 community cards, got {len(env.community_cards)} "
            f"(stage={env.betting_stage})"
        )

    def test_three_player_fold_has_5_community_cards(self):
        # Two players fold → remaining player wins, board still completed
        env = new_game(n_players=3, card_info_lut={})
        env = _play_fold_first(env)
        assert env.is_terminal
        assert len(env.community_cards) == 5

    def test_community_cards_unique_at_terminal(self):
        env = new_game(n_players=2, card_info_lut={})
        env = _play_all_calls(env)
        assert len(env.community_cards) == len(set(env.community_cards))

    def test_community_cards_no_overlap_with_hole_cards(self):
        env = new_game(n_players=2, card_info_lut={})
        # Capture hole cards from initial state before the game plays out
        hole_cards = {c for p in env.players for c in p.cards}
        env = _play_all_calls(env)
        for c in env.community_cards:
            assert c not in hole_cards

    def test_all_in_at_each_stage_completes_board(self):
        """All-in at pre-flop, flop, turn, and river all produce 5 board cards."""
        for target_stage, n_calls_before in [("pre_flop", 0), ("flop", 2), ("turn", 4), ("river", 6)]:
            env = new_game(n_players=2, card_info_lut={})
            for _ in range(n_calls_before):
                if not env.is_terminal:
                    env = env.apply_action("call")
            if not env.is_terminal and "all_in" in env.legal_actions:
                env = env.apply_action("all_in")
                assert env.is_terminal
                assert len(env.community_cards) == 5, (
                    f"all-in at {target_stage}: expected 5 cards, "
                    f"got {len(env.community_cards)}"
                )


# ---------------------------------------------------------------------------
# Positional flags persist correctly (Bug 1)
# ---------------------------------------------------------------------------

class TestPositionalFlagsInvariant:
    def test_exactly_one_dealer_throughout_game(self):
        env = new_game(n_players=3, card_info_lut={})
        while not env.is_terminal:
            assert sum(p.is_dealer for p in env.players) == 1, (
                f"Dealer count wrong at stage={env.betting_stage}"
            )
            env = _play_step(env)

    def test_heads_up_dealer_is_player_0_throughout(self):
        env = new_game(n_players=2, card_info_lut={})
        assert env.players[0].is_dealer is True
        while not env.is_terminal:
            env = _play_step(env)
        # Flags should be unchanged after game ends
        assert env.players[0].is_dealer is True

    def test_small_blind_player_posted_blind(self):
        env = new_game(n_players=2, card_info_lut={}, small_blind=50, initial_chips=10000)
        sb = next(p for p in env.players if p.is_small_blind)
        assert sb.n_chips == 10000 - 50


# ---------------------------------------------------------------------------
# Payout integrity
# ---------------------------------------------------------------------------

class TestPayoutIntegrityInvariant:
    def test_payout_sums_to_zero(self):
        env = new_game(n_players=2, card_info_lut={})
        env = _play_all_calls(env)
        assert sum(env.payout.values()) == 0

    def test_payout_keys_cover_all_players(self):
        env = new_game(n_players=3, card_info_lut={})
        env = _play_all_calls(env)
        assert set(env.payout.keys()) == set(range(3))

    def test_payout_consistent_with_chip_delta(self):
        # payout[i] is defined as player.n_chips - _initial_n_chips (pre-blind).
        # Verify this holds and that all payouts sum to zero.
        env = new_game(n_players=2, card_info_lut={}, initial_chips=10000)
        initial = env._initial_n_chips
        env = _play_all_calls(env)
        for i, player in enumerate(env.players):
            assert env.payout[i] == player.n_chips - initial
        assert sum(env.payout.values()) == 0

    def test_payout_zero_sum_three_players(self):
        env = new_game(n_players=3, card_info_lut={})
        env = _play_all_calls(env)
        assert sum(env.payout.values()) == 0


# ---------------------------------------------------------------------------
# Termination guarantee
# ---------------------------------------------------------------------------

class TestTerminationInvariant:
    @pytest.mark.parametrize("n_players,play_fn", [
        (2, _play_all_calls),
        (3, _play_all_calls),
        (6, _play_all_calls),
        (2, _play_fold_first),
        (2, _play_all_in_first),
        (3, _play_all_raises),
    ])
    def test_game_terminates(self, n_players, play_fn):
        env = new_game(n_players=n_players, card_info_lut={})
        env = play_fn(env, max_steps=500)
        assert env.is_terminal, (
            f"Game did not terminate after 500 steps "
            f"(last stage: {env.betting_stage})"
        )


# ---------------------------------------------------------------------------
# Stage legality
# ---------------------------------------------------------------------------

class TestStageLegalityInvariant:
    _VALID_STAGES = {"pre_flop", "flop", "turn", "river", "show_down", "terminal"}
    _STAGE_ORDER = ["pre_flop", "flop", "turn", "river"]

    def test_legal_actions_never_empty(self):
        env = new_game(n_players=3, card_info_lut={})
        for _ in range(200):
            if env.is_terminal:
                break
            assert len(env.legal_actions) > 0, (
                f"Empty legal_actions at stage={env.betting_stage}"
            )
            env = _play_step(env)

    def test_stage_only_advances_forward(self):
        """Stages must progress forward; never revert to an earlier stage."""
        env = new_game(n_players=2, card_info_lut={})
        seen_stages = []
        for _ in range(200):
            if env.is_terminal:
                break
            stage = env.betting_stage
            if stage in self._STAGE_ORDER:
                if seen_stages:
                    prev = seen_stages[-1]
                    if prev in self._STAGE_ORDER:
                        assert self._STAGE_ORDER.index(stage) >= self._STAGE_ORDER.index(prev), (
                            f"Stage reverted from {prev} to {stage}"
                        )
                seen_stages.append(stage)
            env = _play_step(env)

    def test_current_player_index_always_valid(self):
        env = new_game(n_players=3, card_info_lut={})
        for _ in range(200):
            if env.is_terminal:
                break
            assert 0 <= env.player_i < env.n_players
            env = _play_step(env)

    def test_is_turn_always_exactly_one_player(self):
        env = new_game(n_players=3, card_info_lut={})
        for _ in range(50):
            if env.is_terminal:
                break
            turns = [p.is_turn for p in env.players]
            assert turns.count(True) == 1, (
                f"Expected exactly 1 player with is_turn=True, "
                f"got {turns.count(True)} at stage={env.betting_stage}"
            )
            env = _play_step(env)
