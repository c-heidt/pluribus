"""Shared fixtures for the information_abstraction test suite."""
import pytest

from poker_ai.environment.evaluator import Evaluator
from poker_ai.information_abstraction.build.card_combos import CardCombos


@pytest.fixture
def evaluator():
    return Evaluator()


@pytest.fixture
def tiny_combos():
    """Smallest legal deck: rank 12-14 (3 ranks = 12 cards).

    Small enough to enumerate every street exhaustively in tests while still
    exercising the combinadic indexing and the full (hole, board) shape.
    """
    return CardCombos(
        low_card_rank=12, high_card_rank=14,
        parallel=False, n_workers=1,
    )
