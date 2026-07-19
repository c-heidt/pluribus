"""Unit gate for :class:`poker_ai.search.cluster_maps.ClusterMapper`.

The traverser-vectorized MCCFR walk stores future-street nodes per LUT cluster and
relies on ClusterMapper for the dense row layout, the per-combo gather index, the
board-feasibility mask, and the segment-sum scatter.  The Phase-2 walk gate runs
on a stub LUT that collapses every combo to ONE cluster, so this test exercises
the **multi-cluster** gather/scatter path (>1 row) that the walk needs on a real
LUT — validated against the ``clusters_for_board`` ground truth.
"""

import numpy as np
import pytest

from environment.utils import enumerate_combos
from information_abstraction.lookup import clusters_for_board
from poker_ai.search.cluster_maps import ClusterMapper


class _MultiClusterLUT:
    """A dict-street stand-in mapping each combo to one of ``n`` clusters.

    ``clusters_for_board`` treats any non-``MemmapLookup`` entry as a dict and
    indexes it by ``tuple(sorted(hole) + sorted(board))``; the cluster here
    depends only on the hole cards, so a fixed set of distinct clusters is
    reachable on every board (exercising a >1-row universe)."""

    def __init__(self, n: int = 4):
        self.n = n

    def __getitem__(self, key):
        return (int(key[0]) + int(key[1])) % self.n


LOW, HIGH = 11, 14                       # 16-card deck, C(16,2) = 120 combos


def _cc():
    cards, _ = enumerate_combos(LOW, HIGH)
    return cards


def _root_board_turn():
    """Four turn cards disjoint from a broad set of combos."""
    cc = _cc()
    # Take the four lowest distinct card ints as the turn board.
    return sorted(set(int(c) for c in np.unique(cc)))[:4]


def test_universe_is_sorted_unique_reachable_ids():
    cc = _cc()
    board = _root_board_turn()
    lut = {"river": _MultiClusterLUT(4)}
    cm = ClusterMapper(lut, cc, board, street_at_root=2)
    assert cm.n_completion == 1                      # turn root → one river card

    # Recompute the universe: union of cluster ids over every 1-card completion.
    avail = cm.avail.tolist()
    ids = set()
    for card in avail:
        b = np.array(board + [card], dtype=np.int64)
        raw = clusters_for_board(lut["river"], cc, b)
        ids.update(int(x) for x in np.unique(raw[raw >= 0]))
    assert cm.n_rows(3) == len(ids)
    assert len(ids) > 1                               # genuinely multi-cluster


@pytest.mark.parametrize("river_pick", [0, 3, 7])
def test_refresh_cluster_of_and_feas_match_ground_truth(river_pick):
    cc = _cc()
    board = _root_board_turn()
    lut = {"river": _MultiClusterLUT(4)}
    cm = ClusterMapper(lut, cc, board, street_at_root=2)
    river = int(cm.avail[river_pick % len(cm.avail)])
    cm.refresh((river,))

    full_board = np.array(board + [river], dtype=np.int64)
    raw = clusters_for_board(lut["river"], cc, full_board)   # global ids, -1 infeasible
    valid = raw >= 0

    # Feasibility matches board-compatibility exactly.
    assert np.array_equal(cm.feas(3) > 0, valid)

    # Dense row = searchsorted into the (sorted) universe for feasible combos; -1 else.
    cof = cm.cluster_of(3)
    universe = np.array(sorted(set(int(x) for x in np.unique(raw[valid]))), dtype=np.int64)
    # (universe here is over ONE completion — a subset of cm's full-search universe.)
    expected = np.full(cc.shape[0], -1, dtype=np.int64)
    from information_abstraction.lookup import clusters_for_board as _cfb  # noqa
    # Rebuild against cm's actual universe (union over all completions).
    full_universe = _reconstruct_universe(cm, lut, cc, board)
    expected[valid] = np.searchsorted(full_universe, raw[valid])
    assert np.array_equal(cof, expected)
    assert np.array_equal(cof < 0, ~valid)


def _reconstruct_universe(cm, lut, cc, board):
    ids = set()
    for card in cm.avail.tolist():
        b = np.array(board + [card], dtype=np.int64)
        raw = clusters_for_board(lut["river"], cc, b)
        ids.update(int(x) for x in np.unique(raw[raw >= 0]))
    return np.array(sorted(ids), dtype=np.int64)


def test_scatter_add_segment_sum_matches_reference():
    """``scatter_add`` must equal a plain per-combo grouped add into cluster rows."""
    cc = _cc()
    board = _root_board_turn()
    lut = {"river": _MultiClusterLUT(4)}
    cm = ClusterMapper(lut, cc, board, street_at_root=2)
    river = int(cm.avail[0])
    cm.refresh((river,))

    n_rows = cm.n_rows(3)
    width = 5
    rng = np.random.default_rng(0)
    per_combo = rng.standard_normal((cc.shape[0], width))

    table = np.zeros((n_rows, width), dtype=np.float64)
    cm.scatter_add(table, per_combo, 3)

    # Reference: for each feasible combo, add its row into its dense cluster row.
    cof = cm.cluster_of(3)
    ref = np.zeros((n_rows, width), dtype=np.float64)
    for c in range(cc.shape[0]):
        k = int(cof[c])
        if k >= 0:
            ref[k] += per_combo[c]
    assert np.allclose(table, ref)
    # Infeasible combos contribute nothing (their rows never appear in a segment).
    assert np.count_nonzero(cof < 0) > 0             # some combos share the board


def test_river_root_has_no_completion():
    cc = _cc()
    board = sorted(set(int(c) for c in np.unique(cc)))[:5]      # 5-card river board
    lut = {}
    cm = ClusterMapper(lut, cc, board, street_at_root=3)
    assert cm.n_completion == 0
    assert cm.avail.size == 0
    cm.refresh(())                                    # no-op, must not raise
    assert cm.feas_full is None
