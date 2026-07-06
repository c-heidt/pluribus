"""Functional-search fixtures.

The parent ``test/search/conftest.py`` defines an autouse ``_seeded`` fixture
parametrised over five RNG seeds, so every search test runs five times.  The
golden-trace and harness self-tests here pin their **own** seeds internally (the
whole point is a fixed, reproducible fingerprint), so five identical repetitions
would be pure waste.  Overriding ``_seeded`` with a single non-parametrised
autouse fixture (the closest conftest wins) makes them run once while preserving
the reseed contract for any test that reads the returned seed.
"""

import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seeded():
    """Single-seed reseed of Python + numpy globals (overrides the 5× parent)."""
    random.seed(0)
    np.random.seed(0)
    return 0
