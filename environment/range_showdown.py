"""Vectorised range-vs-range showdown — the env's per-combo terminal payout.

This is the third of the environment's terminal-payout evaluators, alongside the
**concrete** settlement (:func:`environment.dynamics.compute_winners` /
``PokerEnv.payout``) and the **decision-free** board-average (``PokerEnv.runout_equity``).
Where those score a single dealt hand, this one values an **entire range against an
entire range** on a completed board at once — the paper's F2 vectorised hand-vs-hand
showdown — and is consumed by ``PokerEnv.vector_payout``.

Rather than the O(n_combos^2) all-pairs sum, every acting combo is settled in
**O(n_combos log n_combos)** via two sorted sweeps with exact card removal (the
"ref. 42" trick): the value of acting combo ``i`` against the opponent's
reach-weighted range is

    v_i = stake * (W_i - L_i),
      W_i = sum of opp reach on combos i beats   (rank[j] > rank[i]),
      L_i = sum of opp reach on combos i loses to (rank[j] < rank[i]),

restricted to opponent combos that share **no** card with ``i`` (card removal).
Ties net zero.  Folds (no showdown) use :func:`reach_after_removal` instead.

The hand *ranking* is player-count-agnostic; only the settlement is heads-up (a
single opponent range → no opponent-opponent removal term).  Everything is pure and
deterministic, and ranks hands with the **same shared** :data:`environment.evaluator.default_evaluator`
the concrete path uses — so a range showdown is scored identically to a concrete
resolution (aligned by construction).  Board rankings are memoised (board-keyed
LRU) so the env settles many terminals on the same deck without re-ranking.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np

from environment.evaluator import default_evaluator
from environment.utils import enumerate_combos

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
        Scalar hand evaluator; defaults to the shared
        :data:`environment.evaluator.default_evaluator` (the same object
        ``compute_winners`` and ``runout_equity`` use).

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
        evaluator = default_evaluator
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


# --------------------------------------------------------------------------- #
# Board-keyed memoisation (the env's "fast" responsibility).  Keyed by the deck
# rank range + the board tuple; ``combo_cards`` is rebuilt from the (already
# cached) :func:`enumerate_combos`, so callers pass only hashable identifiers.
# Returned arrays are read-only — callers must not mutate them.
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1024)
def _ranked_cached(low_card_rank: int, high_card_rank: int, board: Tuple[int, ...]):
    cards, _ = enumerate_combos(low_card_rank, high_card_rank)
    ranks, valid = rank_combos_on_board(cards, board)
    ranks.flags.writeable = False
    valid.flags.writeable = False
    return ranks, valid


@lru_cache(maxsize=1024)
def _valid_cached(low_card_rank: int, high_card_rank: int, board: Tuple[int, ...]):
    cards, _ = enumerate_combos(low_card_rank, high_card_rank)
    board_arr = np.asarray(board, dtype=cards.dtype)
    valid = ~(
        np.isin(cards[:, 0], board_arr) | np.isin(cards[:, 1], board_arr)
    )
    valid.flags.writeable = False
    return valid


@lru_cache(maxsize=64)
def _removal_cached(low_card_rank: int, high_card_rank: int):
    cards, _ = enumerate_combos(low_card_rank, high_card_rank)
    return removal_index(cards)


def ranked_board(
    low_card_rank: int, high_card_rank: int, board: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Memoised :func:`rank_combos_on_board` for a deck's combo set on ``board``.

    Ranks the ≤~46 candidate boards of a subgame **once** each and reuses them
    across CFR iterations.  ``(ranks, valid)`` are read-only — do not mutate.
    """
    return _ranked_cached(int(low_card_rank), int(high_card_rank),
                          tuple(int(c) for c in board))


def removal_for(low_card_rank: int, high_card_rank: int):
    """Memoised :func:`removal_index` for a deck's combo set (board-independent)."""
    return _removal_cached(int(low_card_rank), int(high_card_rank))


def board_valid_mask(
    low_card_rank: int, high_card_rank: int, board: Sequence[int]
) -> np.ndarray:
    """Memoised board-compatibility mask for a deck's combo set on ``board``.

    The rank-free counterpart of :func:`ranked_board`'s ``valid`` output: ``True``
    iff the combo shares no card with ``board``.  A **fold** terminal needs only
    this (no showdown, so no hand ranking), so settling it this way avoids the
    evaluator pass — and avoids ranking a partial (pre-river) board, keeping the
    "rank each completed board once" invariant intact.  The returned array is
    read-only — do not mutate.
    """
    return _valid_cached(int(low_card_rank), int(high_card_rank),
                         tuple(int(c) for c in board))


def reach_after_removal(
    combo_cards: np.ndarray,
    opp_reach: np.ndarray,
    removal: Optional[Tuple[np.ndarray, np.ndarray, int]] = None,
) -> np.ndarray:
    """Per acting combo, the opponent reach on combos sharing **no** card with it.

    The rank-independent counterpart of :func:`showdown_cfv`, for **fold** terminals
    (no showdown: the pot is decided by the fold, but card removal between the two
    ranges still applies).  For acting combo ``i = {c0, c1}`` it returns

        available_i = total - on_card[c0] - on_card[c1] + opp_reach[i]

    where ``total`` is the summed opponent reach and ``on_card[c]`` is the opponent
    reach on combos containing card ``c``.  The trailing ``+ opp_reach[i]`` is the
    **inclusion-exclusion add-back**: combo ``i`` itself contains *both* ``c0`` and
    ``c1``, so the two per-card subtractions remove it twice — it must be added back
    once so it is excluded exactly once.  (:func:`showdown_cfv` needs no such add-back
    because ``i`` shares its rank group and is excluded from both the strictly-stronger
    and strictly-weaker sums anyway.)  Fully vectorised, O(n_combos), no n^2 matrix.

    Parameters
    ----------
    combo_cards : numpy.ndarray
        ``(n_combos, 2)`` hole-card pairs; used only to derive ``removal`` when it is
        not supplied.
    opp_reach : numpy.ndarray
        ``(n_combos,)`` opponent reach (unnormalised).  Pass a board-masked vector
        (zeros on board-incompatible combos) for correct card removal at the terminal.
    removal : optional
        Precomputed :func:`removal_index` for ``combo_cards``.

    Returns
    -------
    numpy.ndarray
        ``(n_combos,)`` float64 available opponent reach per acting combo.
    """
    s0, s1, deck_size = removal if removal is not None else removal_index(combo_cards)
    w = np.asarray(opp_reach, dtype=np.float64)
    total = float(w.sum())
    on_card = np.bincount(
        np.concatenate([s0, s1]), weights=np.concatenate([w, w]), minlength=deck_size
    )
    return total - on_card[s0] - on_card[s1] + w


def showdown_cfv(
    ranks: np.ndarray,
    valid: np.ndarray,
    combo_cards: np.ndarray,
    opp_reach: np.ndarray,
    stake: float,
    dead: float = 0.0,
    removal: Optional[Tuple[np.ndarray, np.ndarray, int]] = None,
) -> np.ndarray:
    """Counterfactual value per acting combo vs an opponent reach-weighted range.

    Implements ``v_i = stake * (W_i - L_i) + dead * (W_i + T_i / 2)`` with exact
    card removal in O(n_combos log n_combos) — fully vectorised (sort +
    ``bincount`` + ``cumsum`` over rank groups), never forming an n^2 matrix and
    with no per-group loop.  ``W/L/T`` are the opponent reach the combo beats /
    loses to / ties: the winner collects the opponent's matched stake **plus**
    the dead money, the loser forfeits only its stake, and a chopped pot splits
    the dead money.

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
        winner gains from the opponent and the loser forfeits (heads-up,
        winner-takes-pot).
    dead : float
        Chips in the pot contributed by seats **outside** the heads-up contest
        (folded players' blinds and abandoned bets).  They carry no side-pot
        claim: the winner collects them in full, a chop splits them evenly.
        ``0`` for a pot with no third-party contributions.
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
    if dead != 0.0 and n_groups:
        # Dead money goes to the winner (half each on a chop): + dead per unit of
        # beaten reach, + dead/2 per unit of tied reach.  Tied reach = available
        # reach (card removal, with the own-combo add-back — cf.
        # :func:`reach_after_removal`) minus the strictly-stronger/-weaker parts.
        card_total = M_cum[-1]
        avail = gt_cum[-1] - card_total[s0] - card_total[s1] + w
        ties = avail - beats - loses
        v = v + dead * (beats + 0.5 * ties)
    v[~valid] = 0.0
    return v


def showdown_values(
    combo_cards: np.ndarray,
    board: Sequence[int],
    reach_a: np.ndarray,
    reach_b: np.ndarray,
    stake: float,
    dead: float = 0.0,
    evaluator=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Heads-up showdown: per-combo CFV for both players on a completed board.

    Ranks the board once, then settles each player against the *other* player's
    reach.  ``dead`` is third-party (folded seats') pot money, per
    :func:`showdown_cfv`.

    Returns
    -------
    (cfv_a, cfv_b) : Tuple[numpy.ndarray, numpy.ndarray]
        Per-combo counterfactual values for player A and player B respectively.
    """
    ranks, valid = rank_combos_on_board(combo_cards, board, evaluator)
    removal = removal_index(combo_cards)
    cfv_a = showdown_cfv(
        ranks, valid, combo_cards, reach_b, stake, dead=dead, removal=removal
    )
    cfv_b = showdown_cfv(
        ranks, valid, combo_cards, reach_a, stake, dead=dead, removal=removal
    )
    return cfv_a, cfv_b
