"""Per-street feature extractors for the build pipeline.

Three tiny classes, one per street, each callable as
``extractor(combo) -> np.ndarray``:

- :class:`RiverEHS` — three-element ``[win, loss, tie]`` expected hand
  strength.  Branches internally on ``method`` (``"exact"`` or
  ``"monte_carlo"``).
- :class:`TurnEHS` — histogram over river clusters.
- :class:`FlopEHS` — histogram over turn clusters.

Turn and flop extractors lazily load the upstream street's ``cluster_ids.dat``
memmap on first use, caching it at module scope so multiple worker processes
sharing a pool each load it at most once.  Extractors are picklable and
designed to be passed directly to
:meth:`ChunkStore.process_chunks_parallel`.
"""
import logging
import os
import threading
from itertools import combinations
from pathlib import Path
from typing import Dict

import numpy as np

from poker_ai.information_abstraction._combinatorics import comb, lex_rank

log = logging.getLogger("poker_ai.information_abstraction.build.ehs")


# ---------------------------------------------------------------------------
# Per-worker memmap cache for upstream cluster ids
# ---------------------------------------------------------------------------
# Workers built by ProcessPoolExecutor live across many chunks of the same
# street.  Lazy-loading the memmap once and caching it at module scope keeps
# the per-chunk dispatch cost ~0 after the first call.  Keyed on save_dir so
# a worker reused across runs doesn't return stale data.

_PROCESS_CACHE: Dict[str, object] = {
    "river_cluster_ids": None,
    "turn_cluster_ids": None,
    "save_dir": None,
}
_CACHE_LOAD_LOCK = threading.Lock()


def clear_process_cache_for_street(street: str) -> None:
    _PROCESS_CACHE[f"{street}_cluster_ids"] = None
    log.debug("Cleared process cache for %s", street)


def get_cluster_id_cache(
    street: str, save_dir: str, n_rows: int,
) -> np.memmap:
    """Return (lazy-loading) the upstream cluster-id memmap for ``street``."""
    if _PROCESS_CACHE["save_dir"] != save_dir:
        _PROCESS_CACHE["save_dir"] = save_dir
        _PROCESS_CACHE["river_cluster_ids"] = None
        _PROCESS_CACHE["turn_cluster_ids"] = None

    cache_key = f"{street}_cluster_ids"
    if _PROCESS_CACHE[cache_key] is not None:
        return _PROCESS_CACHE[cache_key]

    with _CACHE_LOAD_LOCK:
        if _PROCESS_CACHE[cache_key] is not None:
            return _PROCESS_CACHE[cache_key]

        ids_path = Path(save_dir) / street / "cluster_ids.dat"
        if not ids_path.exists():
            raise FileNotFoundError(
                f"cluster_ids.dat not found for {street} at {ids_path}. "
                "Re-run clustering from scratch."
            )
        cluster_ids = np.memmap(
            ids_path, dtype=np.uint16, mode="r", shape=(n_rows,),
        )
        _PROCESS_CACHE[cache_key] = cluster_ids
        log.debug(
            "Worker %d loaded %s cluster_ids (%d entries)",
            os.getpid(), street, n_rows,
        )
        return cluster_ids


def _row_index(
    hole_ints, public_ints, card_to_idx: Dict[int, int], n: int,
) -> int:
    h_idx = sorted(card_to_idx[int(c)] for c in hole_ints)
    p_idx = sorted(card_to_idx[int(c)] for c in public_ints)
    hole_rank = lex_rank(tuple(h_idx), n)
    h0, h1 = h_idx[0], h_idx[1]
    p_reindexed = tuple(p - (h0 < p) - (h1 < p) for p in p_idx)
    n_remaining = n - 2
    k_public = len(p_idx)
    return hole_rank * comb(n_remaining, k_public) + lex_rank(
        p_reindexed, n_remaining,
    )


# ---------------------------------------------------------------------------
# River — exact / monte-carlo branch happens inside one class
# ---------------------------------------------------------------------------


class RiverEHS:
    """Compute ``[win, loss, tie]`` for a river combo.

    Parameters
    ----------
    evaluator
        Hand evaluator (needs ``_five`` and ``_seven``).
    card_ints : np.ndarray
        Deck as ascending eval-card ints.
    method : str
        ``"exact"`` enumerates every opponent pair; ``"monte_carlo"``
        samples ``n_simulations`` pairs.
    n_simulations : int
        Opponent samples (MC mode only).
    """

    def __init__(
        self,
        evaluator,
        card_ints: np.ndarray,
        method: str = "monte_carlo",
        n_simulations: int = 6,
    ):
        self._evaluator = evaluator
        self._card_ints = np.asarray(card_ints)
        self.method = method
        self.n_simulations = n_simulations

    def __call__(self, public: np.ndarray) -> np.ndarray:
        our_hand = public[:2]
        board = public[2:7]
        if self.method == "exact":
            return self._exact(our_hand, board)
        return self._monte_carlo(our_hand, board)

    # ------------------------------------------------------------------

    def _precompute_single_best(
        self, available, board_ints, board_only_rank, board_4,
    ):
        """Pre-compute Category 1 (board + 1 hole card) best rank."""
        _five = self._evaluator._five
        single_best = {}
        for c in available:
            ci = int(c)
            best = board_only_rank
            for b4 in board_4:
                s = _five(b4 + (ci,))
                if s < best:
                    best = s
            single_best[c] = best
        return single_best

    def _setup(self, our_hand, board):
        unavailable = set(our_hand.tolist() + board.tolist())
        available = [c for c in self._card_ints if c not in unavailable]
        board_ints = [int(c) for c in board]
        our_ints = [int(c) for c in our_hand]

        _five = self._evaluator._five
        our_rank = self._evaluator._seven(our_ints + board_ints)
        board_only_rank = _five(board_ints)
        board_4 = list(combinations(board_ints, 4))
        board_3 = list(combinations(board_ints, 3))
        single_best = self._precompute_single_best(
            available, board_ints, board_only_rank, board_4,
        )
        return available, our_rank, board_3, single_best

    def _exact(self, our_hand, board) -> np.ndarray:
        available, our_rank, board_3, single_best = self._setup(
            our_hand, board,
        )
        _five = self._evaluator._five

        wins = losses = ties = 0
        for o1, o2 in combinations(available, 2):
            opp_rank = min(single_best[o1], single_best[o2])
            if opp_rank >= our_rank:
                pair = (int(o1), int(o2))
                for b3 in board_3:
                    s = _five(b3 + pair)
                    if s < opp_rank:
                        opp_rank = s
                        if opp_rank < our_rank:
                            break
            if our_rank > opp_rank:
                wins += 1
            elif our_rank < opp_rank:
                losses += 1
            else:
                ties += 1

        total = wins + losses + ties
        if total == 0:
            return np.array([0.0, 0.0, 0.0])
        return np.array([wins / total, losses / total, ties / total])

    def _monte_carlo(self, our_hand, board) -> np.ndarray:
        available, our_rank, board_3, single_best = self._setup(
            our_hand, board,
        )
        _five = self._evaluator._five
        available_arr = np.array(available)
        n = self.n_simulations
        wins = losses = ties = 0

        for _ in range(n):
            o1, o2 = np.random.choice(available_arr, 2, replace=False)
            opp_rank = min(single_best[o1], single_best[o2])
            if opp_rank >= our_rank:
                pair = (int(o1), int(o2))
                for b3 in board_3:
                    s = _five(b3 + pair)
                    if s < opp_rank:
                        opp_rank = s
                        if opp_rank < our_rank:
                            break
            if our_rank > opp_rank:
                wins += 1
            elif our_rank < opp_rank:
                losses += 1
            else:
                ties += 1

        return np.array([wins / n, losses / n, ties / n])


# ---------------------------------------------------------------------------
# Turn — histogram over river clusters
# ---------------------------------------------------------------------------


class TurnEHS:
    """Compute the river-cluster histogram for a turn combo.

    On first call inside a worker, lazily loads the river
    ``cluster_ids.dat`` memmap at module scope.
    """

    def __init__(
        self,
        card_ints: np.ndarray,
        card_to_idx: Dict[int, int],
        n_cards: int,
        save_dir: str,
        n_river_clusters: int,
    ):
        self._card_ints = np.asarray(card_ints)
        self._card_to_idx = card_to_idx
        self._n_cards = n_cards
        self._save_dir = save_dir
        self._n_river_clusters = n_river_clusters

    def __call__(self, public: np.ndarray) -> np.ndarray:
        our_hand = public[:2]
        board = public[2:6]

        n = self._n_cards
        n_river_rows = comb(n, 2) * comb(n - 2, 5)
        river_cluster_ids = get_cluster_id_cache(
            "river", self._save_dir, n_river_rows,
        )

        unavailable = set(our_hand.tolist() + board.tolist())
        available = [c for c in self._card_ints if c not in unavailable]
        dist = np.zeros(self._n_river_clusters)

        river_board = np.empty(5, dtype=np.int32)
        river_board[:4] = board
        for river_card in available:
            river_board[4] = river_card
            row = _row_index(
                our_hand, river_board, self._card_to_idx, n,
            )
            dist[int(river_cluster_ids[row])] += 1

        total = dist.sum()
        if total > 0:
            dist /= total
        return dist


# ---------------------------------------------------------------------------
# Flop — histogram over turn clusters
# ---------------------------------------------------------------------------


class FlopEHS:
    """Compute the turn-cluster histogram for a flop combo."""

    def __init__(
        self,
        card_ints: np.ndarray,
        card_to_idx: Dict[int, int],
        n_cards: int,
        save_dir: str,
        n_turn_clusters: int,
    ):
        self._card_ints = np.asarray(card_ints)
        self._card_to_idx = card_to_idx
        self._n_cards = n_cards
        self._save_dir = save_dir
        self._n_turn_clusters = n_turn_clusters

    def __call__(self, public: np.ndarray) -> np.ndarray:
        our_hand = public[:2]
        board = public[2:5]

        # River cluster ids aren't needed past the turn stage — drop them
        # from the worker cache so the OS can reclaim the pages.
        clear_process_cache_for_street("river")

        n = self._n_cards
        n_turn_rows = comb(n, 2) * comb(n - 2, 4)
        turn_cluster_ids = get_cluster_id_cache(
            "turn", self._save_dir, n_turn_rows,
        )

        unavailable = set(our_hand.tolist() + board.tolist())
        available = [c for c in self._card_ints if c not in unavailable]
        dist = np.zeros(self._n_turn_clusters)

        turn_board = np.empty(4, dtype=np.int32)
        turn_board[:3] = board
        for turn_card in available:
            turn_board[3] = turn_card
            row = _row_index(
                our_hand, turn_board, self._card_to_idx, n,
            )
            dist[int(turn_cluster_ids[row])] += 1

        total = dist.sum()
        if total > 0:
            dist /= total
        return dist
