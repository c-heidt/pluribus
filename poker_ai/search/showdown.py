"""Vectorised range-vs-range showdown for the vector-form CFR regime (§6.5).

The vector regime (heads-up turn/river, §6.5) carries a per-combo *reach* vector
per player and, at a showdown terminal over a completed board, must value an
entire range against an entire range at once — the paper's **F2 vectorised
hand-vs-hand showdown**.  Rather than the O(n_combos^2) all-pairs sum, this
module settles every acting combo in **O(n_combos log n_combos)** via two sorted
sweeps with exact card removal (the "ref. 42" trick): the value of acting combo
``i`` against the opponent's reach-weighted range is

    v_i = stake * (W_i - L_i),
      W_i = sum of opp reach on combos i beats   (rank[j] > rank[i]),
      L_i = sum of opp reach on combos i loses to (rank[j] < rank[i]),

restricted to opponent combos that share **no** card with ``i`` (card removal).
Ties net zero (heads-up showdown is winner-takes-pot with equal contributions).

Scope is **heads-up only** — a single opponent range, so there is no
opponent-opponent card-removal term and no side pots (§10).  The hand *ranking*
(:func:`rank_combos_on_board`) is player-count-agnostic; only the settlement is
heads-up.  Both halves are pure and deterministic (no RNG, no env mutation): the
hand evaluator is the same scalar :class:`Evaluator` the rest of the engine uses
(``compute_winners`` / ``runout_equity``), so a showdown here is scored
identically to a concrete resolution.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

import environment.dynamics as dynamics

# A rank strictly worse than any real hand rank (Evaluator returns [1, 7462],
# lower = stronger).  Board-incompatible combos are parked here and carry zero
# reach, so they never enter a win/lose sum.
_SENTINEL_RANK = 1 << 30


def rank_combos_on_board(
    combo_cards: np.ndarray,
    board: Sequence[int],
    evaluator=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rank every hole combo on a fixed board.

    Parameters
    ----------
    combo_cards : numpy.ndarray
        Shape ``(n_combos, 2)`` int array of hole-card pairs (``env.combo_cards``).
    board : Sequence[int]
        Community cards (card integers).  Any length the evaluator accepts; in
        the vector regime this is the completed 5-card runout.
    evaluator : optional
        Scalar hand evaluator; defaults to ``environment.dynamics._evaluator``
        (the same object ``compute_winners`` and ``runout_equity`` use).

    Returns
    -------
    ranks : numpy.ndarray
        Shape ``(n_combos,)`` int64.  ``ranks[i]`` is the best-5-of-7 rank of
        combo ``i`` (lower = stronger); board-incompatible combos hold
        :data:`_SENTINEL_RANK`.
    valid : numpy.ndarray
        Shape ``(n_combos,)`` bool.  ``False`` iff combo ``i`` shares a card with
        the board (impossible to hold, excluded from the showdown).
    """
    if evaluator is None:
        evaluator = dynamics._evaluator
    board_list = [int(c) for c in board]

    n = combo_cards.shape[0]
    # Vectorised validity: a combo is impossible iff either card is on the board.
    board_arr = np.asarray(board_list, dtype=combo_cards.dtype)
    valid = ~(
        np.isin(combo_cards[:, 0], board_arr)
        | np.isin(combo_cards[:, 1], board_arr)
    )

    ranks = np.full(n, _SENTINEL_RANK, dtype=np.int64)
    # Rank only the board-compatible combos (the rest stay at the sentinel and
    # carry zero reach), in a single vectorised batch: each row is the combo's
    # two cards followed by the shared board (K = 2 + len(board) in {5, 6, 7}).
    vidx = np.flatnonzero(valid)
    if vidx.size:
        board_row = np.asarray(board_list, dtype=np.int64)
        hands = np.empty((vidx.size, 2 + board_row.shape[0]), dtype=np.int64)
        hands[:, 0] = combo_cards[vidx, 0]
        hands[:, 1] = combo_cards[vidx, 1]
        hands[:, 2:] = board_row
        ranks[vidx] = evaluator.evaluate_batch(hands)
    return ranks, valid


def removal_index(combo_cards: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Densify card integers to contiguous slot indices ``0..deck_size-1``.

    Card integers are Cactus-Kev encoded (not ``0..51``), so the per-card
    removal accumulators are indexed by a densified slot.  Depends only on the
    deck (board-independent), so a caller settling many terminals on one deck
    should compute this **once** and pass it to :func:`showdown_cfv`.

    Returns the two per-combo slot arrays (int64) and the deck size.
    """
    uniq = np.unique(combo_cards)
    s0 = np.searchsorted(uniq, combo_cards[:, 0])
    s1 = np.searchsorted(uniq, combo_cards[:, 1])
    return s0.astype(np.int64), s1.astype(np.int64), int(uniq.shape[0])


def showdown_cfv(
    ranks: np.ndarray,
    valid: np.ndarray,
    combo_cards: np.ndarray,
    opp_reach: np.ndarray,
    stake: float,
    removal: Optional[Tuple[np.ndarray, np.ndarray, int]] = None,
) -> np.ndarray:
    """Counterfactual value per acting combo vs an opponent reach-weighted range.

    Implements ``v_i = stake * (W_i - L_i)`` with exact card removal in
    O(n_combos log n_combos) — fully vectorised (sort + ``bincount`` + ``cumsum``
    over rank groups), never forming an n^2 matrix and with no per-group loop.

    Parameters
    ----------
    ranks, valid : numpy.ndarray
        Output of :func:`rank_combos_on_board` for this board.
    combo_cards : numpy.ndarray
        ``(n_combos, 2)`` hole-card pairs; used only to derive ``removal`` when
        it is not supplied.
    opp_reach : numpy.ndarray
        ``(n_combos,)`` opponent reach (unnormalised); board-incompatible entries
        are ignored (re-zeroed internally for safety).
    stake : float
        Each player's matched contribution to the contested pot — the amount the
        winner gains and the loser forfeits (heads-up, winner-takes-pot).
    removal : optional
        Precomputed :func:`removal_index` for ``combo_cards``.  Pass it when
        settling many terminals on the same deck to skip the per-call densify.

    Returns
    -------
    numpy.ndarray
        ``(n_combos,)`` float64 counterfactual value per acting combo; ``0`` for
        board-incompatible combos.
    """
    n = ranks.shape[0]
    s0, s1, deck_size = removal if removal is not None else removal_index(combo_cards)

    # Opponent reach contributes only from holdable combos.
    w = np.where(valid, np.asarray(opp_reach, dtype=np.float64), 0.0)

    # Dense group id per combo (one group per distinct rank, ascending = strong
    # first).  Equal ranks share a group, so they fall into neither sum below.
    order = np.argsort(ranks, kind="stable")
    r_sorted = ranks[order]
    grp_sorted = np.zeros(n, dtype=np.int64)
    if n > 1:
        grp_sorted[1:] = np.cumsum(r_sorted[1:] != r_sorted[:-1])
    group_of = np.empty(n, dtype=np.int64)
    group_of[order] = grp_sorted
    n_groups = int(grp_sorted[-1]) + 1 if n else 0

    # Per-group total reach, and per-group per-card reach (G x deck_size), built
    # with bincount (C-level scatter) instead of a Python loop / np.add.at.
    gt = np.bincount(group_of, weights=w, minlength=n_groups)
    flat = np.concatenate(
        [group_of * deck_size + s0, group_of * deck_size + s1]
    )
    M = np.bincount(
        flat, weights=np.concatenate([w, w]), minlength=n_groups * deck_size
    ).reshape(n_groups, deck_size)

    # Exclusive prefix/suffix sums across groups give the strictly-stronger and
    # strictly-weaker reach, both in total and per card (for removal).
    M_cum = np.cumsum(M, axis=0)
    gt_cum = np.cumsum(gt)

    lower_total = gt_cum - gt              # reach in groups strictly before g
    lower_card = M_cum - M                 # same, per card
    upper_total = gt_cum[-1] - gt_cum      # reach in groups strictly after g
    upper_card = M_cum[-1] - M_cum         # same, per card

    g = group_of
    loses = lower_total[g] - lower_card[g, s0] - lower_card[g, s1]
    beats = upper_total[g] - upper_card[g, s0] - upper_card[g, s1]

    v = stake * (beats - loses)
    v[~valid] = 0.0
    return v


def showdown_values(
    combo_cards: np.ndarray,
    board: Sequence[int],
    reach_a: np.ndarray,
    reach_b: np.ndarray,
    stake: float,
    evaluator=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Heads-up showdown: per-combo CFV for both players on a completed board.

    Ranks the board once, then settles each player against the *other* player's
    reach.  This is the entry point the vector regime calls at a showdown
    terminal.

    Returns
    -------
    (cfv_a, cfv_b) : Tuple[numpy.ndarray, numpy.ndarray]
        Per-combo counterfactual values for player A and player B respectively.
    """
    ranks, valid = rank_combos_on_board(combo_cards, board, evaluator)
    removal = removal_index(combo_cards)
    cfv_a = showdown_cfv(ranks, valid, combo_cards, reach_b, stake, removal=removal)
    cfv_b = showdown_cfv(ranks, valid, combo_cards, reach_a, stake, removal=removal)
    return cfv_a, cfv_b
