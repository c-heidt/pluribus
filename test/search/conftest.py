"""Shared search-test fixtures.

Every test below this directory is parametrised over multiple RNG
seeds via the :func:`_seeded` autouse fixture.  Each parametrised
instance reseeds Python's ``random`` and numpy's global RNG so the
env's deck shuffle (which uses :func:`numpy.random.shuffle`) yields a
different card layout per trial.  This way a single ``pytest`` run
already exercises a representative spread of card patterns — flakes
that previously needed dozens of consecutive runs to surface now
show up reliably in one invocation.

Helpers like :func:`_build_ctx` in the per-module test files derive
the per-test ``ctx.rng`` seed from the global RNG state so Monte-Carlo
sampling also varies with the trial unless a test pins it explicitly.
"""

import random

import numpy as np
import pytest


_SEARCH_TRIALS = tuple(range(5))


@pytest.fixture(autouse=True, params=_SEARCH_TRIALS, ids=lambda t: f"seed{t}")
def _seeded(request):
    """Reseed Python and numpy's global RNG per trial.

    Returns the seed value so tests that want to thread it explicitly
    (e.g. into :class:`SubgameContext`'s ``rng`` field) can do so by
    declaring ``_seeded`` in their signature.  Tests that ignore the
    fixture still benefit from the reseed because env construction
    uses :func:`numpy.random.shuffle` against the global state.
    """
    seed = request.param
    random.seed(seed)
    np.random.seed(seed)
    return seed
