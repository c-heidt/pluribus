"""Unit-test fixtures for the environment package.

Adds a per-trial seeding autouse fixture so every test below this
directory is executed multiple times with different RNG states.  The
env's ``Deck`` shuffles via :func:`numpy.random.shuffle` against the
global RNG, so each trial produces a different card layout — flakes
that depend on rare card patterns surface in a single ``pytest`` run
instead of requiring dozens of consecutive invocations.

Pure-logic tests (combos, evaluator, pot, utils) run identically
across trials at very small per-trial cost; the parametrisation is
opt-in via a small fixed trial count.
"""

import random

import numpy as np
import pytest


_TRIALS = tuple(range(5))


@pytest.fixture(autouse=True, params=_TRIALS, ids=lambda t: f"seed{t}")
def _seeded(request):
    seed = request.param
    random.seed(seed)
    np.random.seed(seed)
    return seed
