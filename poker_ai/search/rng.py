"""Independent RNG sub-stream derivation (the "one stream per consumer" rule).

Three consumers draw randomness while a hand is played and must not share a stream:
the **game simulation** (the deal, which owns the global ``np.random`` exclusively),
the **acting approach** (play sampling and solver internals, separately), and
**AIVAT** (:mod:`evaluation.aivat`).  Sharing makes each one's draws depend on how
much work the other did, which breaks the evaluation's CRN pairing: two arms that
play a hand identically still burn different amounts of solver randomness.

:func:`spawn` takes children off the parent's
:class:`~numpy.random.SeedSequence` **without advancing the parent**, so adding a
consumer never perturbs an existing one's byte-stream.
"""

from __future__ import annotations

from typing import List

import numpy as np


def spawn(rng: np.random.Generator, n: int = 1) -> List[np.random.Generator]:
    """``n`` independent child generators derived from ``rng``.

    The parent is **not advanced**, so spawning a new sub-stream leaves every
    previously-derived stream bit-identical.  A generator exposing no seed sequence
    falls back to seeding from a draw, which does advance ``rng``.

    .. warning::
       ``SeedSequence.spawn`` alone is **not** a correct derivation on the pinned
       numpy: 1.17.4 does not mix ``spawn_key`` into the generated state, so the
       first spawned child is bit-identical to its parent.  This folds the key into
       the entropy itself; call it rather than spawning directly.
    """
    if n < 1:
        raise ValueError(f"spawn: n must be >= 1, got {n}")
    try:
        parent_seq = rng.bit_generator._seed_seq
        # ``spawn`` for the child COUNTER only (so repeated spawns off one parent
        # keep diverging); the seeds themselves are built below.
        spawn_keys = [tuple(int(k) for k in c.spawn_key)
                      for c in parent_seq.spawn(n)]
        # Do NOT seed children from the spawned SeedSequences directly: on numpy
        # 1.17.4 ``SeedSequence(e, spawn_key=(0,))`` yields the SAME stream as
        # ``SeedSequence(e)``, so the "separate stream" would be no separation at
        # all.  Folding the key into the entropy is correct on every version.
        pool = [int(w) for w in parent_seq.generate_state(4, dtype=np.uint32)]
        seeds = [np.random.SeedSequence(pool + list(key)) for key in spawn_keys]
    except AttributeError:
        # No exposed seed sequence — derive from draws instead (advances ``rng``).
        seeds = [
            np.random.SeedSequence(int(rng.integers(0, 2 ** 63 - 1)))
            for _ in range(n)
        ]
    return [np.random.default_rng(s) for s in seeds]


def spawn_one(rng: np.random.Generator) -> np.random.Generator:
    """Single-child convenience wrapper around :func:`spawn`."""
    return spawn(rng, 1)[0]
