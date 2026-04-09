"""Root pytest configuration — LUT discovery and shared markers.

Only the session-scoped LUT fixture and the ``requires_lut`` skip logic live
here so they are available to tests in any subdirectory.  Environment-specific
game fixtures (players, decks, pots, game states) are in
``test/environment/conftest.py``.
"""

import os
from pathlib import Path

import joblib
import pytest


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
    config.addinivalue_line(
        "markers",
        "slow: marks tests that take a long time to run "
        "(skip with '-m \"not slow\"')",
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
