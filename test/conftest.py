"""Shared pytest fixtures for the environment test suite."""

import os
from pathlib import Path

import joblib
import pytest

from poker_ai.environment.player import Player
from poker_ai.environment.pot import Pot
from poker_ai.environment.chance import Deck
from poker_ai.environment.evaluator import Evaluator
from poker_ai.environment.poker_env import new_game


# ---------------------------------------------------------------------------
# LUT discovery and skip logic
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_LUT_DIR = _REPO_ROOT / "data" / "clustering" / "20cards_exact"
_LUT_PATH = Path(os.environ.get("PLURIBUS_LUT_PATH", str(_DEFAULT_LUT_DIR)))
_LUT_FILE = _LUT_PATH / "card_info_lut.joblib"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_lut: marks tests that need a pre-built LUT on disk "
        "(skip with '-m \"not requires_lut\"')",
    )


def pytest_collection_modifyitems(config, items):
    if _LUT_FILE.exists():
        return  # LUT present — nothing to skip
    skip = pytest.mark.skip(reason=f"LUT not found at {_LUT_FILE}")
    for item in items:
        if item.get_closest_marker("requires_lut"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def lut():
    """Load the card info LUT once per test session."""
    if not _LUT_FILE.exists():
        pytest.skip(f"LUT not found at {_LUT_FILE}")
    return joblib.load(str(_LUT_FILE))


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
