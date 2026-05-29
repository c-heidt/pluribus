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
    """Per-opponent dense-per-combo range estimates for one hand.

    Parameters
    ----------
    env : PokerEnv
        Reference env, used only for combo enumeration / community
        cards.  Not deepcopied; the tracker reads from it.
    my_seat : int
        Seat index of the bot.  Excluded from :meth:`snapshot`.
    my_hole : tuple[int, int]
        Bot's hole cards.  Combos sharing any of these cards start with
        zero weight in every opponent's range.
    live_seats : Iterable[int]
        Seats with a range to track.  ``my_seat`` is silently filtered
        out if included.
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
        initial = _initial_uniform(env, self._my_hole, self._community)
        self._ranges: Dict[int, np.ndarray] = {
            int(s): initial.copy() for s in live_seats if int(s) != my_seat
        }
        self._decision_log: List[Tuple[int, str, str]] = []

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
        """Drop ``seat`` from the tracker.  No-op if already absent."""
        self._ranges.pop(seat, None)

    def range_of(self, seat: int) -> Range:
        """Return the in-place range vector for ``seat``.

        Treat the returned array as read-only; the tracker is the
        sole writer.  Raises ``KeyError`` for unknown / folded seats.
        """
        return self._ranges[seat]

    def snapshot(self) -> Dict[int, Range]:
        """Deep copy of all live opponent ranges, suitable for handing
        to :meth:`SubgameContext.from_runtime`."""
        return {seat: w.copy() for seat, w in self._ranges.items()}

    def _uniform_fallback(self, seat: int) -> None:
        warnings.warn(
            f"Range for seat {seat} collapsed below floor; "
            "resetting to uniform over board-compatible combos.",
            RuntimeWarning,
            stacklevel=2,
        )
        self._ranges[seat] = _initial_uniform(
            self._env_ref, self._my_hole, self._community
        )
