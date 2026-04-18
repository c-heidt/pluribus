"""Shared fixtures for the environment test suite.

Provides game objects (players, decks, pots, game states) used across
``test/environment/unit/`` and ``test/environment/functional/``.
"""

import pytest

from environment.player import Player
from environment.pot import Pot
from environment.chance import Deck
from environment.evaluator import Evaluator
from environment.poker_env import new_game


@pytest.fixture
def two_players():
    return [Player(i, 10000) for i in range(2)]


@pytest.fixture
def three_players():
    return [Player(i, 10000) for i in range(3)]


@pytest.fixture
def six_players():
    return [Player(i, 10000) for i in range(6)]


@pytest.fixture
def full_deck():
    return Deck(2, 14)


@pytest.fixture
def short_deck():
    return Deck(10, 14)


@pytest.fixture
def fresh_pot():
    return Pot(3)


@pytest.fixture
def evaluator():
    return Evaluator()


@pytest.fixture
def fresh_game():
    return new_game(n_players=3, card_info_lut={})


@pytest.fixture
def two_player_game():
    return new_game(n_players=2, card_info_lut={})
