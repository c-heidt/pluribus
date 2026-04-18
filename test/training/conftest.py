"""Shared pytest fixtures for the training test suite.

Provides lightweight :class:`~poker_ai.tables.cfr_tables.CFRTables` instances
and starter game states used across ``test/training/unit/`` and
``test/training/functional/``.  All I/O lands in pytest's ``tmp_path`` so
there is no cross-test contamination and no need to touch ``/dev/shm``.
"""

import os

import pytest

from environment.action_space import MAX_ACTIONS_PER_STREET
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.tables.index import InfosetIndex
from environment.poker_env import new_game


@pytest.fixture
def tmp_tables(tmp_path):
    """A :class:`CFRTables` instance backed entirely by *tmp_path*.

    The ``PLURIBUS_SHM_DIR`` environment variable is set for the duration
    of the fixture so chunk files land in pytest's tempdir rather than
    ``/dev/shm``, keeping tests hermetic and cleaning up automatically.
    """
    shm_dir = str(tmp_path / "shm")
    os.makedirs(shm_dir, exist_ok=True)
    tables = CFRTables(
        index_path=tmp_path / "lmdb_index",
        shm_dir=shm_dir,
        actions_per_street=MAX_ACTIONS_PER_STREET,
    )
    yield tables
    tables.close()


@pytest.fixture
def tmp_index(tmp_path):
    """A temporary :class:`~poker_ai.tables.index.InfosetIndex`."""
    idx = InfosetIndex(tmp_path / "idx", lmdb_map_size=10 * 1024 ** 3)
    yield idx
    idx.close()


@pytest.fixture
def two_player_state():
    """A fresh two-player game with an empty card-info LUT."""
    return new_game(n_players=2, card_info_lut={}, lut_path=".", pickle_dir=False)


@pytest.fixture
def three_player_state():
    """A fresh three-player game with an empty card-info LUT."""
    return new_game(n_players=3, card_info_lut={}, lut_path=".", pickle_dir=False)
