"""Functional-test fixtures for the environment package.

Every test below this directory is executed multiple times with
different RNG states.  The env's ``Deck`` shuffles via
:func:`numpy.random.shuffle` against the global RNG, so each trial
produces a different card layout — card-pattern-dependent flakes
that previously required repeated ``pytest`` invocations to surface
now show up in a single run with the trial id printed alongside the
failing test.

Pure-logic tests run identically across trials at a small constant
overhead; the parametrisation is opt-in via a small fixed trial
count.
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
