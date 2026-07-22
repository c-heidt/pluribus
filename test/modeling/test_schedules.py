"""Tests for the synthetic-sweep error/confidence schedules (poker_ai/modeling/schedules.py).

Pins the *pure vs noisy* axis: street-graded and per-infoset-noisy error magnitude, the
calibrated / anti-calibrated / flat confidence regimes, their reproducibility (paired
evaluation needs the injected model identical for a given seed), and that
:class:`SyntheticOpponentModel` honours a scheduled (non-scalar) error.
"""

import numpy as np

from environment.poker_env import PolicyState
from poker_ai.modeling.model import SyntheticOpponentModel
from poker_ai.modeling import schedules as sch


def _state(info_set=b"k", betting_round=3, legal=("fold", "call", "raise:1.0")):
    n = len(legal)
    return PolicyState(
        player_i=0,
        betting_round=betting_round,
        info_set=info_set,
        valid_mask=np.ones(n, dtype=bool),
        legal_actions=tuple(legal),
    )


class FakePolicy:
    def __init__(self, dist):
        self._d = np.asarray(dist, dtype=np.float32)

    def strategy(self, state, bias="none"):
        return self._d


class TestErrorMagnitude:

    def test_uniform_is_the_scalar(self):
        assert sch.uniform_error(0.2) == 0.2

    def test_street_error_grades_by_round(self):
        s = sch.street_error({0: 0.05, 1: 0.1, 2: 0.2, 3: 0.4}, default=0.99)
        assert s(_state(betting_round=0)) == 0.05
        assert s(_state(betting_round=3)) == 0.4
        assert s(_state(betting_round=7)) == 0.99          # unknown round → default

    def test_infoset_noise_is_reproducible_and_heterogeneous(self):
        s = sch.with_infoset_noise(0.2, sigma=0.5, seed=3)
        a = s(_state(info_set=b"A"))
        assert a == s(_state(info_set=b"A"))               # same (seed, infoset) → identical
        b = s(_state(info_set=b"B"))
        assert a != b                                       # different infoset → different draw

    def test_infoset_noise_respects_clip_and_zero_sigma(self):
        # sigma=0 reproduces the base exactly.
        s0 = sch.with_infoset_noise(0.3, sigma=0.0)
        assert s0(_state()) == 0.3
        # Heavy spread stays within [lo, hi] for every infoset.
        s = sch.with_infoset_noise(0.5, sigma=2.0, seed=1, lo=0.0, hi=1.0)
        vals = [s(_state(info_set=bytes([i]))) for i in range(200)]
        assert all(0.0 <= v <= 1.0 for v in vals)
        assert np.std(vals) > 0.0                           # genuinely heterogeneous

    def test_noise_on_zero_base_stays_zero(self):
        s = sch.with_infoset_noise(sch.street_error({0: 0.0, 3: 0.4}), sigma=1.0, seed=2)
        assert s(_state(betting_round=0)) == 0.0            # no error to jitter preflop


class TestConfidenceRegimes:

    def test_calibrated_falls_as_error_rises(self):
        err = sch.street_error({0: 0.05, 3: 0.45})
        c = sch.calibrated(err)
        assert c(_state(betting_round=0)) > c(_state(betting_round=3))
        assert np.isclose(c(_state(betting_round=0)), 0.95)     # 1 - 0.05
        assert np.isclose(c(_state(betting_round=3)), 0.55)     # 1 - 0.45

    def test_anti_calibrated_rises_with_error(self):
        err = sch.street_error({0: 0.05, 3: 0.45})
        c = sch.anti_calibrated(err, base=0.4)
        assert c(_state(betting_round=3)) > c(_state(betting_round=0))
        assert np.isclose(c(_state(betting_round=0)), 0.45)     # 0.4 + 0.05

    def test_flat_is_constant(self):
        c = sch.flat(0.7)
        assert c(_state(betting_round=0)) == c(_state(betting_round=3)) == 0.7

    def test_all_regimes_clip_to_unit_interval(self):
        err = sch.uniform_error(2.0)                        # absurd error to force clipping
        assert sch.calibrated(err)(_state()) == 0.0        # 1 - 2 → clipped to 0
        assert sch.anti_calibrated(err, base=0.5)(_state()) == 1.0   # 0.5 + 2 → clipped to 1

    def test_calibrated_noise_is_reproducible(self):
        c = sch.calibrated(sch.uniform_error(0.2), noise=0.2, seed=5)
        assert c(_state(info_set=b"X")) == c(_state(info_set=b"X"))
        # Noise makes otherwise-identical-error infosets differ (confidently-wrong tail).
        assert c(_state(info_set=b"X")) != c(_state(info_set=b"Y"))


class TestModelHonoursSchedule:

    def test_scheduled_error_perturbs_only_where_nonzero(self):
        # error 0 preflop / 0.4 river → preflop exact, river perturbed.
        err = sch.street_error({0: 0.0, 3: 0.4})
        m = SyntheticOpponentModel(FakePolicy([0.5, 0.3, 0.2]), error=err, seed=1)
        pre = m.strategy(_state(info_set=b"p", betting_round=0))
        assert np.allclose(pre, [0.5, 0.3, 0.2])           # unperturbed where error=0
        riv = m.strategy(_state(info_set=b"r", betting_round=3))
        assert not np.allclose(riv, [0.5, 0.3, 0.2])       # perturbed where error>0
        assert np.isclose(riv.sum(), 1.0) and (riv >= 0).all()

    def test_calibrated_confidence_tracks_scheduled_error(self):
        err = sch.street_error({0: 0.05, 3: 0.45})
        m = SyntheticOpponentModel(
            FakePolicy([1.0]), confidence=sch.calibrated(err), error=err, p_max=1.0
        )
        assert m.confidence(_state(betting_round=0)) > m.confidence(_state(betting_round=3))
