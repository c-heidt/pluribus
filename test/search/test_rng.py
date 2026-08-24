"""Tests for :mod:`poker_ai.search.rng` — RNG sub-stream derivation.

The module exists to enforce one rule: **one stream per consumer**.  The two
properties every caller depends on are that children are independent of each other,
and that spawning them does *not* advance the parent — the latter is what lets a new
sub-stream (a board-runout stream next to a sampling stream) be introduced without
changing any existing consumer's byte-stream.
"""

import numpy as np
import pytest

from poker_ai.search.rng import spawn, spawn_one


def _draws(rng, n=8):
    return [float(rng.random()) for _ in range(n)]


class TestSpawn:

    def test_returns_requested_count(self):
        assert len(spawn(np.random.default_rng(0), 3)) == 3

    def test_children_are_independent(self):
        a, b = spawn(np.random.default_rng(7), 2)
        assert _draws(a) != _draws(b)

    def test_children_differ_from_parent(self):
        parent = np.random.default_rng(7)
        child = spawn_one(parent)
        assert _draws(child) != _draws(parent)

    def test_parent_is_not_advanced(self):
        """The property the whole split relies on.

        Spawning a board stream off a sampling stream must leave the sampling
        stream's output bit-identical, or introducing the split would silently
        change every existing search trajectory.
        """
        untouched = _draws(np.random.default_rng(11))
        parent = np.random.default_rng(11)
        spawn(parent, 2)
        assert _draws(parent) == untouched

    def test_reproducible_from_same_parent_seed(self):
        assert _draws(spawn_one(np.random.default_rng(3))) == \
               _draws(spawn_one(np.random.default_rng(3)))

    def test_repeated_spawns_do_not_collide(self):
        parent = np.random.default_rng(5)
        first, second = spawn_one(parent), spawn_one(parent)
        assert _draws(first) != _draws(second)

    def test_rejects_non_positive_n(self):
        with pytest.raises(ValueError):
            spawn(np.random.default_rng(0), 0)

    def test_falls_back_when_no_seed_sequence(self):
        """A generator exposing no seed sequence still yields usable children.

        This path derives from draws and so *does* advance the parent
        (documented); it only has to work, not preserve the parent's stream.
        """
        class _NoSeedSeq:
            """Duck-typed generator whose ``bit_generator`` has no ``_seed_seq``."""

            def __init__(self, seed):
                self._inner = np.random.default_rng(seed)
                self.bit_generator = object()

            def integers(self, *args, **kwargs):
                return self._inner.integers(*args, **kwargs)

        children = spawn(_NoSeedSeq(0), 2)
        assert len(children) == 2
        assert _draws(children[0]) != _draws(children[1])


class TestNumpySpawnKeyQuirk:
    """Why :func:`spawn` cannot just return ``SeedSequence.spawn``'s children.

    On the pinned numpy (1.17.4) ``spawn_key`` is not mixed into the generated
    state, so ``SeedSequence(e, spawn_key=(0,))`` produces the SAME stream as
    ``SeedSequence(e)`` — a bare ``spawn(1)`` hands back a child bit-identical to
    its parent, and a "dedicated" sub-stream is silently no separation at all.

    The first test documents the quirk (it is expected to start failing on a newer
    numpy, at which point the workaround is merely redundant, not wrong); the
    second pins the property that actually matters.
    """

    def test_bare_spawn_first_child_may_collide_with_parent(self):
        parent_seq = np.random.SeedSequence(7)
        child = parent_seq.spawn(1)[0]
        bare = np.random.default_rng(child)
        parent = np.random.default_rng(np.random.SeedSequence(7))
        if _draws(bare) == _draws(parent):
            pytest.xfail("numpy drops spawn_key (1.17.x) — this is why spawn() "
                         "folds the key into the entropy itself")

    def test_spawn_one_never_collides_with_parent(self):
        for seed in range(25):
            parent = np.random.default_rng(seed)
            child = spawn_one(parent)
            assert _draws(child) != _draws(np.random.default_rng(seed))


class TestNeverGlobal:

    def test_spawn_does_not_touch_global_np_random(self):
        before = np.random.get_state()[1].copy()
        spawn(np.random.default_rng(1), 4)
        np.testing.assert_array_equal(np.random.get_state()[1], before)
