"""Tests for the range-tracking quality hook (evaluation/range_quality.py, doc §7).

Two layers:

- :func:`compute_metrics` — the pure §7 metric block, checked on hand-built beliefs
  (peaked / uniform / collapsed) so the arithmetic (log-loss clamp, net info gain
  sign, rank, entropy) is pinned independently of the play loop.
- :class:`RangeQualityRecorder` — buffering live opponent beliefs per street and
  resolving them against a stub env's revealed holes (resolved vs. folded).
"""

import math

import numpy as np
import pytest

from evaluation.range_quality import (
    BeliefSnapshot,
    RangeQualityRecorder,
    compute_metrics,
)
from poker_ai.search.ranges import _NUMERICAL_FLOOR


# --------------------------------------------------------------------------- #
# compute_metrics — the §7 arithmetic
# --------------------------------------------------------------------------- #

class TestComputeMetrics:

    def test_uniform_belief_has_zero_net_gain(self):
        # A uniform belief over the full board-compatible support == the baseline,
        # so net info gain is 0 and log-loss == log-loss-uniform == log(support).
        w = np.ones(10, dtype=np.float64) / 10.0
        m = compute_metrics(w, true_combo=3, baseline_support=10)
        assert m["effective_support"] == 10
        assert m["true_combo_mass"] == pytest.approx(0.1)
        assert m["log_loss"] == pytest.approx(math.log(10))
        assert m["log_loss_uniform"] == pytest.approx(math.log(10))
        assert m["net_info_gain"] == pytest.approx(0.0)
        assert m["collapsed_truth"] == 0

    def test_concentrated_on_truth_beats_uniform(self):
        # Belief peaked on the true combo → positive net info gain, rank 1.0 (mode).
        w = np.array([0.9, 0.05, 0.05], dtype=np.float64)
        m = compute_metrics(w, true_combo=0, baseline_support=3)
        assert m["log_loss"] == pytest.approx(-math.log(0.9))
        assert m["net_info_gain"] > 0.0
        assert m["true_combo_rank"] == pytest.approx(1.0)   # true mass is the max

    def test_concentrated_away_from_truth_hurts(self):
        # Confident but wrong → negative net info gain and a low rank.
        w = np.array([0.9, 0.05, 0.05], dtype=np.float64)
        m = compute_metrics(w, true_combo=1, baseline_support=3)  # truth is a tail
        assert m["net_info_gain"] < 0.0
        assert m["true_combo_rank"] < 1.0

    def test_baseline_is_board_support_not_live_support(self):
        # Regression: the baseline is the no-update board-compatible support, NOT
        # the belief's post-Bayes live support.  A belief that concentrates onto the
        # truth by *zeroing* combos (live support 2) must still be scored against the
        # full board support (50), so net info gain is large — not ~log(2).
        w = np.zeros(50, dtype=np.float64)
        w[0], w[1] = 0.99, 0.01                       # live support == 2
        m = compute_metrics(w, true_combo=0, baseline_support=50)
        assert m["effective_support"] == 50           # baseline, not the live 2
        assert m["log_loss_uniform"] == pytest.approx(math.log(50))
        assert m["net_info_gain"] > math.log(2)       # would be ~log(2) under the bug

    def test_collapsed_truth_is_flagged_and_finite(self):
        # Truth zeroed: collapsed flag set, log-loss clamped (finite, not +inf),
        # rank 0, net info gain strongly negative.
        w = np.array([0.5, 0.5, 0.0], dtype=np.float64)
        m = compute_metrics(w, true_combo=2, baseline_support=3)
        assert m["collapsed_truth"] == 1
        assert m["true_combo_mass"] == 0.0
        assert math.isfinite(m["log_loss"])
        assert m["log_loss"] == pytest.approx(-math.log(_NUMERICAL_FLOOR))
        assert m["true_combo_rank"] == pytest.approx(0.0)
        assert m["net_info_gain"] < 0.0

    def test_renormalises_unnormalised_input(self):
        # Raw (unnormalised) weights are renormalised before scoring.
        w = np.array([2.0, 2.0, 0.0], dtype=np.float64)     # → [0.5, 0.5, 0]
        m = compute_metrics(w, true_combo=0, baseline_support=2)
        assert m["true_combo_mass"] == pytest.approx(0.5)
        assert m["effective_support"] == 2

    def test_entropy_is_zero_for_point_mass(self):
        w = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        m = compute_metrics(w, true_combo=1, baseline_support=1)
        assert m["entropy"] == pytest.approx(0.0)
        assert m["effective_support"] == 1


# --------------------------------------------------------------------------- #
# RangeQualityRecorder — buffering + resolution
# --------------------------------------------------------------------------- #

class _StubPlayer:
    def __init__(self, cards, is_active):
        self.cards = cards
        self.is_active = is_active


class _StubEnv:
    """Minimal env exposing what :meth:`RangeQualityRecorder.resolve` reads."""

    def __init__(self, players, combo_index):
        self.players = players
        self.combo_index = combo_index


class _StubTracker:
    """A tracker whose ``snapshot`` returns fixed per-seat beliefs."""

    def __init__(self, ranges, replay=None, fallback=None, baseline=None):
        self._ranges = ranges
        self._replay = replay or {}
        self._fallback = fallback or {}
        self._baseline = baseline or {}

    def snapshot(self):
        return {s: w.copy() for s, w in self._ranges.items()}

    def replay_count(self, seat):
        return self._replay.get(seat, 0)

    def fallback_count(self, seat):
        return self._fallback.get(seat, 0)

    def baseline_support(self, seat):
        # Default: the range's own nonzero count (no Bayes zeroing in the stub).
        return self._baseline.get(seat, int(np.count_nonzero(self._ranges[seat])))


class TestRecorder:

    def _combo_index(self):
        # Two seats, holes (10,11) → combo 0 and (12,13) → combo 1, over a size-3 vec.
        return {(10, 11): 0, (12, 13): 1, (10, 12): 2}

    def test_drops_hero_seat_and_keeps_opponents(self):
        rec = RangeQualityRecorder(hero_seat=0, opponent_seats=[0, 1, 2])
        tracker = _StubTracker({
            0: np.ones(3) / 3, 1: np.ones(3) / 3, 2: np.ones(3) / 3,
        })
        rec.capture(tracker, "flop")
        seats = sorted(s.seat for s in rec._snaps)
        assert seats == [1, 2]                         # hero (0) excluded

    def test_folded_seat_absent_from_snapshot_is_skipped(self):
        rec = RangeQualityRecorder(hero_seat=0, opponent_seats=[1, 2])
        tracker = _StubTracker({1: np.ones(3) / 3})    # seat 2 already folded out
        rec.capture(tracker, "flop")
        assert [s.seat for s in rec._snaps] == [1]

    def test_resolves_showdown_seat_against_revealed_hole(self):
        idx = self._combo_index()
        rec = RangeQualityRecorder(hero_seat=0, opponent_seats=[1])
        # Seat 1's belief peaks on combo 1 == its actual hole (12, 13) → helps.  The
        # belief zeroed combo 2 (live support 2), but the baseline is the full
        # board-compatible support (3), so the recorder must carry that through.
        tracker = _StubTracker(
            {1: np.array([0.1, 0.9, 0.0])}, replay={1: 2}, fallback={1: 0},
            baseline={1: 3},
        )
        rec.capture(tracker, "turn")
        env = _StubEnv(
            players={0: _StubPlayer((0, 1), True), 1: _StubPlayer((12, 13), True)},
            combo_index=idx,
        )
        rows = rec.resolve(env, went_to_showdown=1)
        assert len(rows) == 1
        row = rows[0]
        assert row.resolved == 1 and row.seat == 1 and row.betting_stage == "turn"
        assert row.true_combo == 1
        assert row.n_actions_replayed == 2
        assert row.effective_support == 3              # baseline threaded through
        assert row.net_info_gain > 0.0                 # peaked on the truth

    def test_folded_before_showdown_is_unresolved_with_null_metrics(self):
        idx = self._combo_index()
        rec = RangeQualityRecorder(hero_seat=0, opponent_seats=[1])
        tracker = _StubTracker({1: np.array([0.5, 0.5, 0.0])}, replay={1: 1})
        rec.capture(tracker, "flop")
        # Seat 1 folded (inactive at terminal) → unverifiable.
        env = _StubEnv(
            players={0: _StubPlayer((0, 1), True), 1: _StubPlayer((12, 13), False)},
            combo_index=idx,
        )
        rows = rec.resolve(env, went_to_showdown=0)
        assert len(rows) == 1
        row = rows[0]
        assert row.resolved == 0
        assert row.true_combo is None and row.net_info_gain is None
        assert row.n_actions_replayed == 1             # belief covariates still kept

    def test_showdown_but_seat_folded_is_unresolved(self):
        # Hand reached showdown among *other* seats, but this seat folded → its hole
        # is not revealed, so resolved=0 even though went_to_showdown=1.
        idx = self._combo_index()
        rec = RangeQualityRecorder(hero_seat=0, opponent_seats=[1])
        tracker = _StubTracker({1: np.ones(3) / 3})
        rec.capture(tracker, "river")
        env = _StubEnv(
            players={0: _StubPlayer((0, 1), True), 1: _StubPlayer((12, 13), False)},
            combo_index=idx,
        )
        rows = rec.resolve(env, went_to_showdown=1)
        assert rows[0].resolved == 0

    def test_multiple_streets_buffer_independently(self):
        rec = RangeQualityRecorder(hero_seat=0, opponent_seats=[1])
        tracker = _StubTracker({1: np.ones(3) / 3}, replay={1: 1})
        rec.capture(tracker, "flop")
        tracker._replay[1] = 3                          # more actions folded in
        rec.capture(tracker, "turn")
        stages = [s.betting_stage for s in rec._snaps]
        assert stages == ["flop", "turn"]
        assert [s.n_actions_replayed for s in rec._snaps] == [1, 3]
