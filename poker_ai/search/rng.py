"""Independent RNG sub-stream derivation (the "one stream per consumer" rule).

Three different consumers draw randomness while one hand is played, and they must
not share a stream:

- **the game simulation** — the deal.  Owns the *global* ``np.random`` (the engine
  shuffles ``Deck`` at construction off it), and nothing else may draw from it.
- **the acting approach** (vanilla / DBR / OX) — its play sampling and, separately,
  its solver internals.
- **AIVAT** (:mod:`evaluation.aivat`) — a passive post-hoc estimator.

Sharing a stream across two of these makes each one's draws depend on *how much
work the other did*.  That is not a cosmetic concern: it is what broke the
evaluation's CRN pairing, because two arms that play a hand identically still burn
different amounts of solver randomness, which silently shifted the boards AIVAT
drew and destroyed the exact cancellation the paired Δ relies on.

:func:`spawn` is the derivation primitive.  It takes children off the parent's
:class:`~numpy.random.SeedSequence` **without advancing the parent**, so adding a
new consumer never perturbs an existing one's byte-stream — the property that lets
a board-draw stream be introduced next to a sampling stream without changing the
sampling stream's output.  This mirrors the rationale already recorded at
:class:`poker_ai.search.mccfr._MCCFRSolver` for its board-shuffle RNG.
"""

from __future__ import annotations

from typing import List

import numpy as np


def spawn(rng: np.random.Generator, n: int = 1) -> List[np.random.Generator]:
    """``n`` independent child generators derived from ``rng``.

    The parent is **not advanced**: children come off its seed sequence, so a
    caller that starts spawning a new sub-stream leaves every previously-derived
    stream bit-identical.

    Falls back to seeding from a draw when the generator exposes no seed sequence
    (a hand-built ``Generator``, or one restored from raw bit-generator state).
    That path *does* advance ``rng`` — unavoidable, and acceptable because it only
    occurs for generators that were never seeded reproducibly in the first place.

    .. warning::
       ``SeedSequence.spawn`` alone is **not** a correct derivation on the pinned
       numpy: 1.17.4 does not mix ``spawn_key`` into the generated state, so the
       first spawned child is bit-identical to its parent.  This function folds the
       key into the entropy itself; call it rather than spawning directly.

    Parameters
    ----------
    rng : numpy.random.Generator
        Parent stream.
    n : int
        Number of children to derive.

    Returns
    -------
    list[numpy.random.Generator]
        ``n`` independent generators.
    """
    if n < 1:
        raise ValueError(f"spawn: n must be >= 1, got {n}")
    try:
        parent_seq = rng.bit_generator._seed_seq
        # ``spawn`` for the child COUNTER only (so repeated spawns off one parent
        # keep diverging); the seeds themselves are built below.
        spawn_keys = [tuple(int(k) for k in c.spawn_key)
                      for c in parent_seq.spawn(n)]
        # ⚠️ Do NOT seed the children from those spawned SeedSequences directly.
        # On the pinned numpy (1.17.4) ``spawn_key`` is not mixed into the
        # generated state, so ``SeedSequence(e, spawn_key=(0,))`` yields the SAME
        # stream as ``SeedSequence(e)`` — the first child comes back bit-identical
        # to its parent and the "separate stream" is silently no separation at all.
        # Folding the key into the entropy explicitly is correct on every version.
        pool = [int(w) for w in parent_seq.generate_state(4, dtype=np.uint32)]
        seeds = [np.random.SeedSequence(pool + list(key)) for key in spawn_keys]
    except AttributeError:
        # No exposed seed sequence — derive from draws instead.  This advances
        # ``rng``; see the docstring for why that is tolerable here.
        seeds = [
            np.random.SeedSequence(int(rng.integers(0, 2 ** 63 - 1)))
            for _ in range(n)
        ]
    return [np.random.default_rng(s) for s in seeds]


def spawn_one(rng: np.random.Generator) -> np.random.Generator:
    """Single-child convenience wrapper around :func:`spawn`."""
    return spawn(rng, 1)[0]
