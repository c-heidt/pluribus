"""Tests for the opponent-model providers (poker_ai/modeling/model.py, §4.1).

Covers :class:`SyntheticOpponentModel`: exact pass-through of the wrapped policy,
the seeded ℓ1 perturbation (valid distribution, bounded distance, reproducible), and
the confidence schedule scaled by ``p_max`` as DBR's mixture ``c = p_max · g``
(naive best response is ``g ≡ 1`` at ``p_max = 1``).
"""

import numpy as np

from environment.poker_env import PolicyState
from poker_ai.modeling.model import SyntheticOpponentModel


class FakePolicy:
    """A fixed distribution, ignoring the state — the exactly-known opponent."""

    def __init__(self, dist):
        self._dist = np.asarray(dist, dtype=np.float32)

    def strategy(self, state, bias="none"):
        return self._dist


def _state(info_set=b"k", legal=("fold", "call", "raise:1.0"), r=1):
    n = len(legal)
    return PolicyState(
        player_i=0,
        betting_round=r,
        info_set=info_set,
        valid_mask=np.ones(n, dtype=bool),
        legal_actions=tuple(legal),
    )


class TestSynthetic:

    def test_exact_when_error_zero(self):
        p = [0.5, 0.3, 0.2]
        m = SyntheticOpponentModel(FakePolicy(p), error=0.0)
        assert np.allclose(m.strategy(_state()), p)

    def test_perturbation_stays_a_distribution_and_is_bounded(self):
        p = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        m = SyntheticOpponentModel(FakePolicy(p), error=0.3, seed=1)
        out = m.strategy(_state())
        assert np.isclose(out.sum(), 1.0, atol=1e-5) and (out >= 0).all()
        dist = float(np.abs(out - p).sum())
        assert 0.0 < dist <= 0.3 + 1e-5           # never exceeds the target ℓ1

    def test_perturbation_is_seeded_reproducible_and_infoset_local(self):
        p = [0.4, 0.4, 0.2]
        m = SyntheticOpponentModel(FakePolicy(p), error=0.4, seed=7)
        a = m.strategy(_state(info_set=b"A"))
        a2 = m.strategy(_state(info_set=b"A"))
        b = m.strategy(_state(info_set=b"B"))
        assert np.array_equal(a, a2)               # same (seed, infoset) → identical
        assert not np.allclose(a, b)               # different infoset → different draw
        # A different seed gives a different perturbation for the same infoset.
        other = SyntheticOpponentModel(FakePolicy(p), error=0.4, seed=8)
        assert not np.allclose(a, other.strategy(_state(info_set=b"A")))

    def test_mean_distance_tracks_the_target(self):
        # Over many infosets the mean ℓ1 distance should sit near the target (the
        # cap only bites when a random q lands closer than the target).
        p = [0.34, 0.33, 0.33]
        m = SyntheticOpponentModel(FakePolicy(p), error=0.25, seed=3)
        dists = [
            float(np.abs(m.strategy(_state(info_set=bytes([i]))) - p).sum())
            for i in range(200)
        ]
        assert 0.15 < np.mean(dists) <= 0.25 + 1e-6

    def test_perturbation_preserves_zeros(self):
        # A base row with a zero (an overlay / never-taken action) must stay zero
        # after perturbation — the overlay contract.
        p = [0.5, 0.0, 0.5]
        m = SyntheticOpponentModel(FakePolicy(p), error=0.6, seed=2)
        for i in range(50):
            out = m.strategy(_state(info_set=bytes([i])))
            assert out[1] == 0.0                          # zero support stays zero
            assert np.isclose(out.sum(), 1.0, atol=1e-5)

    def test_point_mass_is_unperturbed(self):
        m = SyntheticOpponentModel(FakePolicy([0.0, 1.0, 0.0]), error=0.5)
        assert np.allclose(m.strategy(_state()), [0.0, 1.0, 0.0])

    def test_confidence_scales_by_p_max(self):
        """``c = p_max · g``: the paper's mixture, so p_max scales rather than caps.

        A clamp would return 0.8 here (0.9 capped); the product returns 0.72.  The
        difference is the whole safety behaviour — under a clamp, p_max stops mattering
        once the schedule exceeds it, and a low p_max would ignore the schedule entirely.
        """
        m = SyntheticOpponentModel(FakePolicy([1.0]), confidence=0.9, p_max=0.8)
        assert np.isclose(m.confidence(_state()), 0.72)
        m2 = SyntheticOpponentModel(FakePolicy([1.0]), confidence=-0.5, p_max=0.8)
        assert m2.confidence(_state()) == 0.0      # the shape g is clipped to >= 0

    def test_p_max_one_is_unchanged(self):
        """The shipped default is behaviour-identical: ``min(1, g) == 1 · g``."""
        for g in (0.0, 0.37, 1.0):
            m = SyntheticOpponentModel(FakePolicy([1.0]), confidence=g, p_max=1.0)
            assert np.isclose(m.confidence(_state()), g)

    def test_confidence_b1_is_one(self):
        m = SyntheticOpponentModel(FakePolicy([1.0]), confidence=1.0, p_max=1.0)
        assert m.confidence(_state()) == 1.0

    def test_confidence_schedule_callable(self):
        # g(state) = 0.1 * betting_round, scaled by p_max=0.95 and clipped at g=1.
        m = SyntheticOpponentModel(
            FakePolicy([1.0]), confidence=lambda s: 0.1 * s.betting_round, p_max=0.95
        )
        assert np.isclose(m.confidence(_state(r=3)), 0.95 * 0.3)
        # g saturates at 1, so the confidence approaches p_max — never exceeds it.
        assert np.isclose(m.confidence(_state(r=20)), 0.95)
