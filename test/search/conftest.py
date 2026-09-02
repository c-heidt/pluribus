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


#: Seeds are chosen for STRATEGIC SPREAD, not taken as ``range(5)``.
#:
#: The oracle fixtures deal seat 0 the two lowest board-free cards and seat 1 the next
#: four, so the seed picks the board but the rank-ordered slice pins who holds what.
#: With the old 16-card (J..A) deck that made most trials the same spot wearing a
#: different board — seeds 0 and 3 produced byte-identical oracle values, solver values
#: AND exploitability on genuinely different boards, because seat 0 was drawing dead in
#: both.  Five trials, but nothing like five independent tests.
#:
#: These five were selected on the 20-card deck by seat 0's exact equity in each of the
#: four hole matchups on each street (12 numbers per seed), requiring both vector
#: production cells (turn AND river) to be non-degenerate and then taking the
#: farthest-point spread, turn weighted double since both the vector gate and the
#: MCCFR gate root there.  What they cover, by seat 0's turn/river equity:
#:
#:     4   turn 0.44 (spread 0.75)   river 0.25   balanced, one nut matchup
#:     5   turn 0.43 (spread 0.50)   river 0.38   balanced, mixed river
#:    23   turn 0.12 (spread 0.25)   river 0.62   BEHIND on the turn, ahead by the river
#:    35   turn 0.61 (spread 0.42)   river 0.50   ahead, polarised river (0/0/1/1)
#:    37   turn 0.96 (spread 0.17)   river 0.75   near-nuts, one live loser
#:
#: Mean pairwise signature distance 2.57 vs 1.99 for ``range(5)``; minimum 1.96 vs 1.22.
#: Re-derive with the scan in the commit that introduced this if the fixtures change:
#: distinct BOARDS are not the property that matters, distinct EQUITY SIGNATURES are.
_SEARCH_TRIALS = (4, 5, 23, 35, 37)


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
