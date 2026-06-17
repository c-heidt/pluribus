"""Per-opponent range tracking for the depth-limited subgame solver.

:class:`RangeTracker` maintains a Bayesian belief over each opponent's
hole cards across one hand.  It starts uniform (modulo board conflicts
and the bot's own hole), gets zeroed against new board cards as streets
advance, and is updated each time an opponent acts under a per-combo
strategy oracle ``sigma_for_combo`` provided by the caller.

The tracker has no search-package dependencies beyond
:class:`environment.poker_env.PokerEnv`.  In particular, it does NOT
import the policy module: ``sigma_for_combo`` is an opaque callable
constructed by the caller (typically :class:`SearchAgent`).  This
keeps ``ranges.py`` test-friendly and re-usable.

See §6.2 of ``docs/subgame_solving.md`` for the design rationale.
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
    """In-place: zero every entry of ``weights`` whose combo shares a
    card with any element of ``cards``.

    Vectorised via :func:`numpy.isin`.  No-op when ``cards`` is empty.
    """
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

    The tracker maintains a range for **every** live seat **including
    the bot** (``my_seat``).  The bot's own range is its
    observer-perspective distribution: uniform over board-compatible
    combos, excluding only board conflicts — the combo that *is* the
    bot's actual hand is **kept** (the search solves over the bot's
    whole range and the agent plays the actual hand's row).  Opponent
    ranges additionally exclude the bot's actual hole cards (known card
    removal).

    Parameters
    ----------
    env : PokerEnv
        Reference env, used only for combo enumeration / community
        cards.  Not deepcopied; the tracker reads from it.
    my_seat : int
        Seat index of the bot.  Tracked (observer perspective) and
        present in :meth:`snapshot`.
    my_hole : tuple[int, int]
        Bot's hole cards.  Combos sharing any of these cards start with
        zero weight in every *opponent's* range, but are kept in the
        bot's own range.
    live_seats : Iterable[int]
        Seats with a range to track, including ``my_seat``.  ``my_seat``
        is tracked even if omitted here.
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
        # The bot's own range excludes only board conflicts (observer
        # perspective — its actual hand stays in the range); every
        # opponent additionally excludes the bot's known hole cards.
        seats = {int(s) for s in live_seats}
        seats.add(int(my_seat))
        self._ranges: Dict[int, np.ndarray] = {
            s: self._initial_for_seat(s) for s in seats
        }
        # Folded seats' marginals are retained here (moved from
        # ``_ranges`` by :meth:`on_seat_folded`) so the leaf evaluator
        # can sample folded seats from their fold-time belief instead
        # of from a uniform fallback.  The principled "fold-time
        # posterior" is obtained when callers invoke
        # :meth:`on_action` with ``"fold"`` *before*
        # :meth:`on_seat_folded` — otherwise the retained range is
        # the pre-fold belief.
        self._folded_ranges: Dict[int, np.ndarray] = {}
        self._decision_log: List[Tuple[int, str, str]] = []

    def _initial_for_seat(self, seat: int) -> np.ndarray:
        """Uniform initial range for ``seat`` over the current board.

        The bot (``my_seat``) excludes only board conflicts (its actual
        hand is kept — observer perspective); every other seat also
        excludes the bot's known hole cards (card removal).
        """
        exclude = () if seat == self._my_seat else self._my_hole
        return _initial_uniform(self._env_ref, exclude, self._community)

    def on_board_update(self, new_cards: Sequence[int]) -> None:
        """Zero every combo sharing a card with ``new_cards`` and
        renormalise each opponent's range.

        ``new_cards`` is also folded into the tracker's running
        ``self._community`` so that any later :meth:`_uniform_fallback`
        rebuilds against the full known board, not the env reference
        captured at construction.

        Triggers :meth:`_uniform_fallback` for any seat whose total
        weight collapses below ``_NUMERICAL_FLOOR``.
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

        This is the round-boundary **replay primitive** (§6.2): the
        agent buffers ``(seat, env_before, action)`` tuples during a
        round and replays them through this method at the boundary
        under the last search's average policy.  It services any
        tracked seat, **including the bot** (``my_seat``) — the bot's
        own actions Bayes-update its own range exactly like an
        opponent's.

        ``sigma_for_combo(h)`` must return a probability vector aligned
        with ``[a for a in env_before.legal_actions if a is not None]``
        for the case where ``seat``'s hole is ``env.combo_cards[h]``.

        ``env_before.player_i`` must equal ``seat``: ``legal_actions``
        is computed for the current actor, and a mismatch silently
        applies the update at the wrong column.
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
        """Move ``seat`` from live ranges into :attr:`folded_snapshot`.

        After this call ``seat`` is absent from :meth:`snapshot` and
        present in :meth:`folded_snapshot` carrying the range it had
        in ``_ranges`` at the moment of the move.  The retained
        marginal is the seat's fold-time belief: callers that ran
        :meth:`on_action` with ``"fold"`` immediately before get the
        post-fold Bayes posterior; callers that didn't get the
        pre-fold belief.  Both are valid; the choice is the caller's
        modelling policy.

        No-op if ``seat`` is already absent (already folded or never
        tracked).
        """
        range_at_fold = self._ranges.pop(seat, None)
        if range_at_fold is not None:
            self._folded_ranges[seat] = range_at_fold

    def range_of(self, seat: int) -> Range:
        """Return the in-place range vector for ``seat``.

        Treat the returned array as read-only; the tracker is the
        sole writer.  Raises ``KeyError`` for unknown / folded seats.
        """
        return self._ranges[seat]

    def snapshot(self) -> Dict[int, Range]:
        """Deep copy of every live seat's range — **including the bot**
        (``my_seat``, observer perspective) — suitable for handing to
        :meth:`SubgameContext.from_runtime` as its ``ranges`` argument."""
        return {seat: w.copy() for seat, w in self._ranges.items()}

    def folded_snapshot(self) -> Dict[int, Range]:
        """Deep copy of all folded opponent ranges at fold time.

        Symmetric to :meth:`snapshot`; consumed by the leaf evaluator
        to sample folded seats' holes from their fold-time marginal
        rather than from a uniform prior.
        """
        return {seat: w.copy() for seat, w in self._folded_ranges.items()}

    def _uniform_fallback(self, seat: int) -> None:
        warnings.warn(
            f"Range for seat {seat} collapsed below floor; "
            "resetting to uniform over board-compatible combos.",
            RuntimeWarning,
            stacklevel=2,
        )
        self._ranges[seat] = self._initial_for_seat(seat)
