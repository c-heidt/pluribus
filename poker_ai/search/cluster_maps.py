"""Per-iteration future-street cluster machinery (§6.5).

Both vector-form regimes — the heads-up :mod:`poker_ai.search.vector` regime and
the traverser-vectorized :mod:`poker_ai.search.mccfr` walk — store root-street
decision nodes lossless (one row per ``combo_index``) and future-street nodes per
LUT cluster, folding the sampled/frozen board into the cluster id (no explicit
river axis).  This class owns that mapping so the two regimes share one
implementation:

- the deterministic per-street cluster **universe** (the union of cluster ids
  reachable over *every* candidate completion), fixed for the whole search so
  parallel replicas derive an identical local row layout and
  :meth:`SolverState.accumulate` sums aligned rows; and
- refreshed each iteration for the sampled/frozen completion: the dense
  combo→cluster row map (``-1`` on board-infeasible combos), the board
  feasibility mask, and a presorted scatter plan that turns a cluster-row update
  into an ``np.add.reduceat`` segment-sum (cheaper than ``np.add.at``).

Bucket-count-agnostic: it reads only the ids the LUT actually produces on the
reachable boards, never a cluster total.
"""

from __future__ import annotations

import itertools
from typing import Dict, Optional, Tuple

import numpy as np

from information_abstraction.lookup import clusters_for_board

# Street index -> LUT street key, for the future-street cluster lookups.
_STREET_NAME = {0: "pre_flop", 1: "flop", 2: "turn", 3: "river"}


class ClusterMapper:
    """Future-street cluster rows / feasibility / scatter plans for one subgame.

    Parameters
    ----------
    lut :
        The env's ``card_info_lut`` (street key -> board/combo -> cluster id).
    combo_cards : numpy.ndarray
        ``(n_combos, 2)`` hole-card pairs (``env.combo_cards``).
    root_community : Sequence[int]
        The community cards public at the subgame root.
    street_at_root : int
        The root street (0..3); future streets are ``street_at_root+1 .. 3``.
    """

    def __init__(self, lut, combo_cards, root_community, street_at_root: int) -> None:
        self._lut = lut
        self._combo_cards = np.asarray(combo_cards, dtype=np.int64)
        self._n_combos = int(self._combo_cards.shape[0])
        self._root_comm = [int(c) for c in root_community]
        self._street_at_root = int(street_at_root)
        self._future = list(range(self._street_at_root + 1, 4))
        self.n_completion = len(self._future)          # 0 river / 1 turn / 2 flop
        if self.n_completion:
            avail = sorted(
                set(int(c) for c in np.unique(self._combo_cards)) - set(self._root_comm)
            )
            self._avail = np.array(avail, dtype=np.int64)
            self._universe = self._build_universes()
        else:
            self._avail = np.empty(0, dtype=np.int64)
            self._universe = {}
        # Per-iteration maps (filled by :meth:`refresh`).
        self._cluster_of: Dict[int, np.ndarray] = {}    # street -> (n_combos,) dense row, -1 infeasible
        self._feas: Dict[int, np.ndarray] = {}          # street -> (n_combos,) 0/1 board-feasibility
        self._scatter: Dict[int, Tuple] = {}            # street -> (sorted_combos, seg_starts, seg_cluster)
        self._feas_full: Optional[np.ndarray] = None

    # -- static layout ---------------------------------------------------

    @property
    def avail(self) -> np.ndarray:
        """Cards available for the runout (deck minus the root community)."""
        return self._avail

    def n_rows(self, street: int) -> int:
        """Number of distinct cluster rows reachable at ``street``."""
        return len(self._universe[street])

    def _build_universes(self) -> Dict[int, np.ndarray]:
        """Sorted unique LUT cluster ids reachable at each future street."""
        universe: Dict[int, np.ndarray] = {}
        avail = self._avail.tolist()
        for s in self._future:
            name = _STREET_NAME[s]
            depth = s - self._street_at_root
            ids: set = set()
            for comp in itertools.combinations(avail, depth):
                board = np.array(self._root_comm + list(comp), dtype=np.int64)
                raw = clusters_for_board(self._lut[name], self._combo_cards, board)
                ids.update(int(x) for x in np.unique(raw[raw >= 0]))
            universe[s] = np.array(sorted(ids), dtype=np.int64)
        return universe

    # -- per-iteration maps ---------------------------------------------

    def refresh(self, completion: Tuple[int, ...]) -> None:
        """Rebuild the dense cluster row + feasibility + scatter plan per street.

        ``completion`` is the sampled/frozen runout (turn[, river]); at street
        ``s`` the board carries its first ``s - street_at_root`` cards.
        """
        for s in self._future:
            name = _STREET_NAME[s]
            depth = s - self._street_at_root
            board = np.array(self._root_comm + list(completion[:depth]), dtype=np.int64)
            raw = clusters_for_board(self._lut[name], self._combo_cards, board)
            valid = raw >= 0
            dense = np.full(self._n_combos, -1, dtype=np.int64)
            dense[valid] = np.searchsorted(self._universe[s], raw[valid])
            self._cluster_of[s] = dense
            self._feas[s] = valid.astype(np.float64)
            fcombos = np.flatnonzero(valid)
            fclusters = dense[fcombos]
            order = np.argsort(fclusters, kind="stable")
            sorted_combos = fcombos[order]
            sorted_clusters = fclusters[order]
            seg_starts = np.concatenate(
                ([0], np.flatnonzero(np.diff(sorted_clusters)) + 1)
            ).astype(np.intp)
            seg_cluster = sorted_clusters[seg_starts]
            self._scatter[s] = (sorted_combos, seg_starts, seg_cluster)
        # Full-completion feasibility (deepest future street) — used to mask the
        # opponent reach at a showdown that completes the whole board at once.
        self._feas_full = self._feas[self._future[-1]] if self._future else None

    def cluster_of(self, street: int) -> np.ndarray:
        """``(n_combos,)`` dense cluster row per combo at ``street`` (-1 infeasible)."""
        return self._cluster_of[street]

    def feas(self, street: int) -> np.ndarray:
        """``(n_combos,)`` 0/1 board feasibility per combo at ``street``."""
        return self._feas[street]

    @property
    def feas_full(self) -> Optional[np.ndarray]:
        """Feasibility for the full (deepest) completion; ``None`` at a river root."""
        return self._feas_full

    def scatter_add(self, table: np.ndarray, per_combo: np.ndarray, street: int) -> None:
        """Segment-sum the feasible combos' rows of ``per_combo`` into ``table``.

        The combo→cluster map is fixed for the whole iteration, so the sort +
        segment boundaries are precomputed in :meth:`refresh` and a ``reduceat``
        does the grouping — cheaper than ``np.add.at`` on every node.
        """
        sorted_combos, seg_starts, seg_cluster = self._scatter[street]
        grouped = per_combo[sorted_combos]                # (n_feasible, width)
        seg_sums = np.add.reduceat(grouped, seg_starts, axis=0)
        table[seg_cluster] += seg_sums
