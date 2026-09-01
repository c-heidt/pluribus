"""Integration tests that require a pre-built card info LUT on disk.

All tests in this file are marked with ``requires_lut`` and are skipped
automatically when the LUT file is absent.  To run only these tests:

    pytest -m requires_lut

To exclude them:

    pytest -m "not requires_lut"

The LUT directory is resolved from the environment variable
``PLURIBUS_LUT_PATH`` (default: ``data/20cards_exact``).
"""

import copy
import pytest

from environment.poker_env import new_game
from test.abstraction_helpers import passive_action
from environment.utils import card_rank_int


pytestmark = pytest.mark.requires_lut


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _play_to_terminal(env, max_steps=500):
    steps = 0
    while not env.is_terminal and steps < max_steps:
        action = "call" if "call" in env.legal_actions else env.legal_actions[0]
        env.step_in_place(action)
        steps += 1
    return env


# ---------------------------------------------------------------------------
# Deck inference from LUT
# ---------------------------------------------------------------------------

class TestDeckInferenceFromLUT:
    def test_new_game_uses_20_card_deck(self, lut):
        env = new_game(n_players=2, card_info_lut=lut)
        assert env.deck_size == 20

    def test_new_game_infers_low_card_rank(self, lut):
        env = new_game(n_players=2, card_info_lut=lut)
        assert env.low_card_rank == 10

    def test_new_game_infers_high_card_rank(self, lut):
        env = new_game(n_players=2, card_info_lut=lut)
        assert env.high_card_rank == 14

    def test_hole_cards_within_lut_rank_range(self, lut):
        for _ in range(10):
            env = new_game(n_players=2, card_info_lut=lut)
            for player in env.players:
                for card in player.cards:
                    assert 10 <= card_rank_int(card) <= 14

    def test_community_cards_within_lut_rank_range(self, lut):
        env = new_game(n_players=2, card_info_lut=lut)
        env = _play_to_terminal(env)
        for card in env.community_cards:
            assert 10 <= card_rank_int(card) <= 14


# ---------------------------------------------------------------------------
# info_set lookup
# ---------------------------------------------------------------------------

class TestInfoSetWithLUT:
    def test_info_set_returns_bytes_at_preflop(self, lut):
        env = new_game(n_players=2, card_info_lut=lut)
        result = env.info_set
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_info_set_fields_expose_cluster(self, lut):
        # The key itself is now opaque compact bytes; the structured
        # accessor exposes the pre-encoding (cluster, history) fields.
        env = new_game(n_players=2, card_info_lut=lut)
        cluster, history = env.info_set_fields()
        assert isinstance(cluster, int)

    def test_info_set_fields_expose_history(self, lut):
        env = new_game(n_players=2, card_info_lut=lut)
        cluster, history = env.info_set_fields()
        assert isinstance(history, list)

    def test_info_set_differs_between_players(self, lut):
        env = new_game(n_players=2, card_info_lut=lut)
        # Player 0 info set
        info_0 = env.info_set
        # Advance to player 1's turn
        env2 = copy.deepcopy(env); env2.step_in_place(passive_action(env2))
        if not env2.is_terminal:
            info_1 = env2.info_set
            # Different players hold different hole cards so info sets differ
            assert info_0 != info_1

    def test_info_set_at_flop(self, lut):
        env = new_game(n_players=2, card_info_lut=lut)
        # Play through pre-flop
        while env.betting_stage == "pre_flop":
            env.step_in_place(passive_action(env))
        if env.betting_stage == "flop":
            cluster, _ = env.info_set_fields()
            assert isinstance(cluster, int)

    def test_info_set_raises_at_nonterminal_without_lut(self, lut):
        # Strip the LUT after construction to simulate a missing entry
        env = new_game(n_players=2, card_info_lut=lut)
        env.card_info_lut = {}
        with pytest.raises(ValueError):
            _ = env.info_set

    def test_info_set_all_streets(self, lut):
        """info_set resolves without error for every non-terminal state."""
        env = new_game(n_players=2, card_info_lut=lut)
        visited_stages = set()
        steps = 0
        while not env.is_terminal and steps < 200:
            visited_stages.add(env.betting_stage)
            assert isinstance(env.info_set, bytes) and len(env.info_set) > 0
            cluster, _ = env.info_set_fields()
            assert isinstance(cluster, int)
            env.step_in_place(passive_action(env))
            steps += 1
        # Must have passed through at least pre_flop and flop
        assert "pre_flop" in visited_stages
        assert "flop" in visited_stages


# ---------------------------------------------------------------------------
# Full game with LUT
# ---------------------------------------------------------------------------

class TestFullGameWithLUT:
    def test_game_reaches_terminal(self, lut):
        env = new_game(n_players=2, card_info_lut=lut)
        env = _play_to_terminal(env)
        assert env.is_terminal

    def test_chip_conservation(self, lut):
        env = new_game(n_players=2, card_info_lut=lut)
        initial_total = sum(p.n_chips for p in env.players) + env.pot_size
        env = _play_to_terminal(env)
        final_total = sum(p.n_chips for p in env.players) + env.pot_size
        assert final_total == initial_total

    def test_three_player_game_reaches_terminal(self, lut):
        env = new_game(n_players=3, card_info_lut=lut)
        env = _play_to_terminal(env)
        assert env.is_terminal

    def test_payout_sums_to_zero(self, lut):
        env = new_game(n_players=2, card_info_lut=lut)
        env = _play_to_terminal(env)
        assert sum(env.payout.values()) == 0
