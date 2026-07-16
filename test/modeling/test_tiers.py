"""Tests for the strength-tier derivation (poker_ai/modeling/tiers.py, §4.4).

Two styles:

- **Hand-built centroids** — a tiny centroids dict with a *known* equity ordering,
  so tier monotonicity, range, and the potential-aware roll-back
  (river → turn → flop) are asserted against arithmetic we control.
- **Real LUT** — the committed ``data/20cards_exact/centroids.joblib``, to confirm
  the derivation runs on the real structure and the equal-frequency bins are
  balanced.
"""

from pathlib import Path

import numpy as np
import pytest

from poker_ai.modeling.tiers import (
    StrengthTiers,
    build_tiers,
    build_tiers_from_centroids,
)

_LUT_DIR = Path(__file__).resolve().parents[2] / "data" / "20cards_exact"


def _toy_centroids():
    """A centroids dict whose per-cluster equity ordering is known by construction.

    river equity = win (tie=0): 4 clusters at [0.1, 0.4, 0.6, 0.9].
    turn / flop are one-hot histograms over the next street, so their equity is
    exactly the selected next-street cluster's equity.
    """
    river = np.array(
        [[0.1, 0.9, 0.0], [0.4, 0.6, 0.0], [0.6, 0.4, 0.0], [0.9, 0.1, 0.0]]
    )                                                   # river_eq = [.1,.4,.6,.9]
    turn = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )                                                   # turn_eq  = [.1,.6,.9]
    flop = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )                                                   # flop_eq  = [.1,.6,.9]
    return {"river": river, "turn": turn, "flop": flop}


class TestEquities:

    def test_rollback_matches_known_construction(self):
        from poker_ai.modeling.tiers import _cluster_equities
        e = _cluster_equities(_toy_centroids())
        assert np.allclose(e["river"], [0.1, 0.4, 0.6, 0.9])
        assert np.allclose(e["turn"], [0.1, 0.6, 0.9])   # potential-aware roll-back
        assert np.allclose(e["flop"], [0.1, 0.6, 0.9])

    def test_mismatched_widths_raise(self):
        from poker_ai.modeling.tiers import _cluster_equities
        bad = _toy_centroids()
        bad["turn"] = np.ones((3, 5))                    # width 5 != 4 river clusters
        with pytest.raises(ValueError, match="histogram over river"):
            _cluster_equities(bad)


class TestTiers:

    def test_tiers_are_monotone_in_equity(self):
        st = build_tiers_from_centroids(_toy_centroids(), n_tiers=4)
        from poker_ai.modeling.tiers import _cluster_equities
        eq = _cluster_equities(_toy_centroids())
        for r, key in ((1, "flop"), (2, "turn"), (3, "river")):
            e = eq[key]
            tiers = np.array([st.tier(r, c) for c in range(len(e))])
            order = np.argsort(e)
            # Sorted by equity, tiers must be non-decreasing.
            assert np.all(np.diff(tiers[order]) >= 0)

    def test_every_cluster_maps_into_range(self):
        st = build_tiers_from_centroids(_toy_centroids(), n_tiers=4)
        for r in (1, 2, 3):
            table = st.tables[r]
            assert table.min() >= 0 and table.max() < st.n_tiers

    def test_preflop_is_lossless(self):
        st = build_tiers_from_centroids(_toy_centroids(), n_tiers=4)
        # No preflop table → the raw cluster is returned unchanged.
        assert 0 not in st.tables
        assert st.tier(0, 137) == 137

    def test_out_of_range_postflop_cluster_raises(self):
        st = build_tiers_from_centroids(_toy_centroids(), n_tiers=4)
        with pytest.raises(IndexError):
            st.tier(3, 999)

    def test_save_load_round_trip(self, tmp_path):
        st = build_tiers_from_centroids(_toy_centroids(), n_tiers=4)
        p = tmp_path / "tiers.npz"
        st.save(p)
        back = StrengthTiers.load(p)
        assert back.n_tiers == st.n_tiers
        for r in (1, 2, 3):
            assert np.array_equal(back.tables[r], st.tables[r])


class TestRealLut:

    @pytest.mark.skipif(
        not (_LUT_DIR / "centroids.joblib").exists(),
        reason="20-card LUT centroids not present",
    )
    def test_builds_and_bins_are_balanced(self):
        st = build_tiers(_LUT_DIR, n_tiers=10)
        for r in (1, 2, 3):
            table = st.tables[r]
            assert table.min() >= 0 and table.max() < 10
            counts = np.bincount(table, minlength=10)
            # Equal-frequency bins: the largest tier is within ~2x the smallest
            # (exact balance is impossible with ties / non-divisible counts).
            nonzero = counts[counts > 0]
            assert nonzero.max() <= 2 * nonzero.min() + 2
