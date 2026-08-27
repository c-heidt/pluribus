"""Per-opponent range tracking for the depth-limited subgame solver (§6.2).

:class:`RangeTracker` maintains a Bayesian belief over each opponent's hole cards
across one hand: uniform at the start (modulo board conflicts and the bot's own hole),
zeroed against new board cards as streets advance, and Bayes-updated whenever a seat
acts under a per-combo strategy oracle ``sigma_for_combo`` supplied by the caller.

That oracle is deliberately opaque — this module does not import the policy package —
which keeps ``ranges.py`` free of search dependencies beyond :class:`PokerEnv`.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from environment.poker_env import PokerEnv


Range = np.ndarray
"""Float32 vector of shape ``(env.n_combos,)`` representing per-combo
weights for one seat's hole.  Sums to 1 when normalised; zero entries
mark combos ruled out by the board or by prior observations."""


_NUMERICAL_FLOOR = 1e-12


def _zero_conflicting(
    weights: np.ndarray, env: PokerEnv, cards: Iterable[int]
) -> None:
    """In-place: zero every entry of ``weights`` whose combo shares a card with any
    element of ``cards``.  No-op when ``cards`` is empty."""
    card_list = list(cards)
    if not card_list:
        return
    cc = env.combo_cards
    forbidden = np.fromiter(card_list, dtype=np.int32)
    conflict = np.isin(cc[:, 0], forbidden) | np.isin(cc[:, 1], forbidden)
    weights[conflict] = 0.0


def _initial_uniform(
    env: PokerEnv, my_hole: Tuple[int, int], community: Iterable[int] = ()
) -> np.ndarray:
    """Uniform float32 range over combos that share no card with
    ``my_hole`` or ``community``."""
    weights = np.ones(env.n_combos, dtype=np.float32)
    forbidden = set(int(c) for c in my_hole) | set(int(c) for c in community)
    _zero_conflicting(weights, env, forbidden)
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


class RangeTracker:
    """Dense-per-combo range estimates for one hand, every live seat.

    A range is kept for **every** live seat **including the bot**.  The bot's own
    range is observer-perspective: it excludes board conflicts only, so the combo that
    *is* its actual hand is **kept** (the search solves over the whole range and the
    agent plays the actual hand's row).  Opponent ranges additionally exclude the bot's
    hole cards (known card removal).

    Parameters
    ----------
    env : PokerEnv
        Reference env, read for combo enumeration / community cards.  Not deepcopied.
    my_seat : int
        Seat index of the bot; tracked even if omitted from ``live_seats``.
    my_hole : tuple[int, int]
        Bot's hole cards.
    live_seats : Iterable[int]
        Seats with a range to track.
    """

    def __init__(
        self,
        env: PokerEnv,
        my_seat: int,
        my_hole: Tuple[int, int],
        live_seats: Iterable[int],
    ) -> None:
        self._env_ref = env
        self._my_seat = my_seat
        self._my_hole = (int(my_hole[0]), int(my_hole[1]))
        self._community: Set[int] = set(int(c) for c in env.community_cards)
        seats = {int(s) for s in live_seats}
        seats.add(int(my_seat))
        self._ranges: Dict[int, np.ndarray] = {
            s: self._initial_for_seat(s) for s in seats
        }
        # Folded seats' marginals, moved out of ``_ranges`` by
        # :meth:`on_seat_folded`, so the leaf evaluator can sample them from their
        # fold-time belief rather than a uniform fallback.
        self._folded_ranges: Dict[int, np.ndarray] = {}
        self._decision_log: List[Tuple[int, str, str]] = []
        # Times :meth:`_uniform_fallback` fired this hand, per seat — a counted
        # signal for the range-quality metric (docs/evaluation.md §7).
        self._fallback_counts: Dict[int, int] = {}

    def _initial_for_seat(self, seat: int) -> np.ndarray:
        """Uniform initial range for ``seat`` over the current board.

        The bot excludes board conflicts only; every other seat also excludes the
        bot's known hole cards.
        """
        exclude = () if seat == self._my_seat else self._my_hole
        return _initial_uniform(self._env_ref, exclude, self._community)

    def on_board_update(self, new_cards: Sequence[int]) -> None:
        """Zero every combo sharing a card with ``new_cards`` and renormalise.

        ``new_cards`` also updates ``self._community`` so a later
        :meth:`_uniform_fallback` rebuilds against the full known board rather than
        the env captured at construction.  A seat whose weight collapses below
        ``_NUMERICAL_FLOOR`` triggers that fallback.
        """
        if not new_cards:
            return
        self._community.update(int(c) for c in new_cards)
        for seat, w in list(self._ranges.items()):
            _zero_conflicting(w, self._env_ref, new_cards)
            total = w.sum()
            if total < _NUMERICAL_FLOOR:
                self._uniform_fallback(seat)
            else:
                w /= total

    def on_action(
        self,
        seat: int,
        env_before: PokerEnv,
        action: str,
        sigma_for_combo: Callable[[int], np.ndarray],
    ) -> None:
        """Bayes-update ``seat``'s range given the observed ``action``.

        The round-boundary **replay primitive**: the agent buffers
        ``(seat, env_before, action)`` during a round and replays them here at the
        boundary under the last search's average policy.  Services any tracked seat
        including the bot.

        ``sigma_for_combo(h)`` must return a probability vector aligned with
        ``[a for a in env_before.legal_actions if a is not None]``, for ``seat``'s hole
        being ``env.combo_cards[h]``.  ``env_before.player_i`` must equal ``seat`` —
        ``legal_actions`` is computed for the current actor, so a mismatch would
        silently apply the update at the wrong column.
        """
        assert env_before.player_i == seat, (
            f"on_action: env_before.player_i={env_before.player_i} "
            f"but seat={seat}; caller must pass an env where seat is to act."
        )
        filtered = [a for a in env_before.legal_actions if a is not None]
        action_idx = filtered.index(action)
        w = self._ranges[seat]
        nz = np.nonzero(w)[0]
        for h in nz:
            w[h] *= float(sigma_for_combo(int(h))[action_idx])
        total = w.sum()
        if total < _NUMERICAL_FLOOR:
            self._uniform_fallback(seat)
        else:
            w /= total
        self._decision_log.append((seat, env_before.info_set, action))

    def on_seat_folded(self, seat: int) -> None:
        """Move ``seat`` from live ranges into :meth:`folded_snapshot`.

        The retained marginal is whatever ``_ranges`` held at the moment of the move:
        callers that ran :meth:`on_action` with ``"fold"`` first get the post-fold
        posterior, others the pre-fold belief.  Both are valid — the caller's choice.
        No-op if ``seat`` is already absent.
        """
        range_at_fold = self._ranges.pop(seat, None)
        if range_at_fold is not None:
            self._folded_ranges[seat] = range_at_fold

    def range_of(self, seat: int) -> Range:
        """The in-place range vector for ``seat``; treat as read-only (the tracker is
        the sole writer).  Raises ``KeyError`` for unknown / folded seats."""
        return self._ranges[seat]

    def snapshot(self) -> Dict[int, Range]:
        """Deep copy of every live seat's range, including the bot's — the ``ranges``
        argument of :meth:`SubgameContext.from_runtime`."""
        return {seat: w.copy() for seat, w in self._ranges.items()}

    def folded_snapshot(self) -> Dict[int, Range]:
        """Deep copy of all folded opponent ranges at fold time, so the leaf evaluator
        can sample those seats from their fold-time marginal rather than a uniform
        prior."""
        return {seat: w.copy() for seat, w in self._folded_ranges.items()}

    def fallback_count(self, seat: int) -> int:
        """How many times ``seat``'s belief collapsed to uniform this hand
        (docs/evaluation.md §7); ``0`` means it never did."""
        return self._fallback_counts.get(int(seat), 0)

    def replay_count(self, seat: int) -> int:
        """Number of observed actions Bayes-replayed into ``seat``'s belief.

        Counts folded seats too, so the range-quality metric can see whether error
        compounds as more actions are folded in.
        """
        s = int(seat)
        return sum(1 for logged_seat, _, _ in self._decision_log if logged_seat == s)

    def baseline_support(self, seat: int) -> int:
        """Size of ``seat``'s no-update uniform prior over the current board.

        The range-quality baseline the tracked belief must beat (docs/evaluation.md §7,
        ``-log(1/|support|)``).  Bayes replay can only zero combos, never add them, so
        this is always ``>=`` the tracked belief's live support.
        """
        return int(np.count_nonzero(self._initial_for_seat(int(seat))))

    def _uniform_fallback(self, seat: int) -> None:
        warnings.warn(
            f"Range for seat {seat} collapsed below floor; "
            "resetting to uniform over board-compatible combos.",
            RuntimeWarning,
            stacklevel=2,
        )
        self._fallback_counts[seat] = self._fallback_counts.get(seat, 0) + 1
        self._ranges[seat] = self._initial_for_seat(seat)
