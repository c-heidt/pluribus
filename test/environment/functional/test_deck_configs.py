"""Functional tests for multi-deck-configuration gameplay.

Verifies that PokerEnv correctly constrains card ranks and raises on invalid
configurations, and that full games play to completion for each deck size.
Deck configuration is set via PokerEnv directly (low_card_rank / high_card_rank);
new_game derives the deck from the LUT automatically.
"""

import pytest

from environment.player import Player
from environment.poker_env import PokerEnv, new_game
from environment.utils import card_rank_int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(low: int, high: int, n_players: int = 2, chips: int = 10000) -> PokerEnv:
    """Create a PokerEnv with explicit deck bounds (no LUT)."""
    players = [Player(i, chips) for i in range(n_players)]
    return PokerEnv(players=players, low_card_rank=low, high_card_rank=high, load_card_lut=False)


def _play_to_terminal(env, max_steps=300):
    steps = 0
    while not env.is_terminal and steps < max_steps:
        action = "call" if "call" in env.legal_actions else env.legal_actions[0]
        env = env.apply_action(action)
        steps += 1
    return env


# ---------------------------------------------------------------------------
# Deck size properties
# ---------------------------------------------------------------------------

class TestDeckSizeProperties:
    def test_20_card_deck_size(self):
        assert _env(10, 14).deck_size == 20

    def test_20_card_low_card_rank(self):
        assert _env(10, 14).low_card_rank == 10

    def test_20_card_high_card_rank(self):
        assert _env(10, 14).high_card_rank == 14

    def test_36_card_deck_size(self):
        assert _env(6, 14).deck_size == 36

    def test_36_card_low_card_rank(self):
        assert _env(6, 14).low_card_rank == 6

    def test_52_card_deck_size(self):
        # new_game with no LUT defaults to full deck
        env = new_game(n_players=2, card_info_lut={})
        assert env.deck_size == 52

    def test_custom_28_card_deck(self):
        assert _env(8, 14).deck_size == 28


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestDeckConfigValidationErrors:
    def test_deck_too_small_two_players(self):
        # ranks 13–14 = 8 cards; need at least 2*2+5=9
        with pytest.raises(ValueError):
            _env(13, 14)

    def test_deck_too_small_four_players(self):
        # ranks 13–14 = 8 cards; need at least 4*2+5=13
        with pytest.raises(ValueError):
            _env(13, 14, n_players=4)

    def test_low_rank_above_high_rank_raises(self):
        with pytest.raises(ValueError):
            _env(12, 10)

    def test_rank_below_2_raises(self):
        with pytest.raises(ValueError):
            _env(1, 14)

    def test_rank_above_14_raises(self):
        with pytest.raises(ValueError):
            _env(2, 15)


# ---------------------------------------------------------------------------
# Card rank range enforcement
# ---------------------------------------------------------------------------

class TestCardRankRangeEnforcement:
    @pytest.mark.parametrize("low,high,n_games", [
        (10, 14, 5),  # 20-card
        (6, 14, 5),   # 36-card
        (2, 14, 5),   # 52-card
    ])
    def test_hole_cards_within_rank_range(self, low, high, n_games):
        for _ in range(n_games):
            env = _env(low, high)
            for player in env.players:
                for card in player.cards:
                    assert low <= card_rank_int(card) <= high, (
                        f"Rank {card_rank_int(card)} outside [{low},{high}]"
                    )

    @pytest.mark.parametrize("low,high,n_games", [
        (10, 14, 5),
        (6, 14, 5),
        (2, 14, 5),
    ])
    def test_community_cards_within_rank_range(self, low, high, n_games):
        for _ in range(n_games):
            env = _env(low, high)
            while env.betting_stage == "pre_flop":
                env = env.apply_action("call")
            for card in env.community_cards:
                assert low <= card_rank_int(card) <= high


# ---------------------------------------------------------------------------
# Full gameplay to terminal
# ---------------------------------------------------------------------------

class TestFullGameplayToTerminal:
    def test_20_card_game_reaches_terminal(self):
        assert _play_to_terminal(_env(10, 14)).is_terminal

    def test_36_card_game_reaches_terminal(self):
        assert _play_to_terminal(_env(6, 14)).is_terminal

    def test_52_card_game_reaches_terminal(self):
        assert _play_to_terminal(new_game(n_players=3, card_info_lut={})).is_terminal

    def test_custom_28_card_game_reaches_terminal(self):
        assert _play_to_terminal(_env(8, 14)).is_terminal

    def test_chip_conservation_across_deck_configs(self):
        for low in [10, 6, 2]:
            env = _env(low, 14)
            initial_total = sum(p.n_chips for p in env.players) + env.pot_size
            env = _play_to_terminal(env)
            final_total = sum(p.n_chips for p in env.players) + env.pot_size
            assert final_total == initial_total
