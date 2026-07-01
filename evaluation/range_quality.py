"""Range-tracking quality: buffer beliefs, resolve them at showdown (doc §7, step 4).

The solver conditions on a per-combo belief over every live opponent's hole cards
([RangeTracker](../poker_ai/search/ranges.py)).  A belief that does not resemble the
hand actually held solves the subgame against a fiction; past a point that *hurts
more than it helps* versus a plain uniform prior, and today nothing measures it.

This module is the measurement.  During a hand the runner calls
:meth:`RangeQualityRecorder.capture` at each round boundary — right after the agent's
belief update, before the hero acts — buffering a copy of every live **opponent**
seat's belief tagged with the street it was held at.  At hand end
:meth:`RangeQualityRecorder.resolve` compares each buffered belief against the hole
actually revealed at showdown and emits one :class:`RangeQualityRow` per snapshot:

- **resolved** (seat reached showdown) — the full metric block below vs. the true
  combo.
- **unresolved** (seat folded before showdown, so the truth is unverifiable) — the
  belief-shape covariates (``n_actions_replayed`` / ``uniform_fallback``) only, with
  ``resolved = 0`` and the quality metrics NULL; kept for the resolved-fraction
  denominator (§8/§11), excluded from quality aggregates.

Only reaches inside the play loop to read revealed holes; the belief math is pure
(no search-package state beyond the buffered vectors + the env's combo enumeration).

Metric definitions (all from the normalised belief ``w`` and the true combo ``h*``),
faithful to the schema comments in §6 / the metric list in §7:

- ``true_combo_mass`` = ``w[h*]`` — belief weight on reality.
- ``effective_support`` = the **board-compatible** support size — the number of
  combos the no-update uniform prior spreads over (board + card removal), NOT the
  tracked belief's post-Bayes nonzero count.  This is the baseline's size (§7:
  "uniform over board-compatible combos"); using the post-update support instead
  would understate the baseline and so understate ``net_info_gain`` exactly when the
  belief helps by ruling combos out.  Supplied by ``RangeTracker.baseline_support``.
- ``log_loss`` = ``-log(w[h*])`` — the single most informative scalar; the true mass
  is clamped at the numerical floor so a collapsed truth yields a large-but-finite
  penalty (≈27.6 nats) that still averages, rather than ``+inf`` poisoning the mean.
- ``log_loss_uniform`` = ``-log(1/effective_support)`` = ``log(effective_support)`` —
  the belief the tracker would have had with no updates; the baseline to beat.
- ``net_info_gain`` = ``log_loss_uniform - log_loss`` — **the headline (§7)**: > 0
  means tracking helped, < 0 means it hurt.
- ``true_combo_rank`` = fraction of the belief's *live* support with mass ≤ the true
  combo's (1.0 = the true combo is the belief's mode, 0.0 = collapsed) —
  calibration-free.
- ``collapsed_truth`` = 1 iff ``w[h*] < floor`` — the catastrophic case: the solve
  ruled out reality.
- ``entropy`` = ``-Σ w log w`` over the belief's live support (nats) — a confidently
  *wrong* belief (low entropy, low true mass) is the most damaging; contextualises
  the log-loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

# The per-combo floor below which the tracker treats a belief as collapsed and
# resets to uniform (ranges.py).  Reused here so ``collapsed_truth`` keys off the
# same threshold the tracker itself uses — one source of truth for "ruled out".
from poker_ai.search.ranges import _NUMERICAL_FLOOR
from evaluation.sqlite_logging import RangeQualityRow


@dataclass
class BeliefSnapshot:
    """One live-opponent belief buffered at a round boundary (pre-resolution)."""

    seat: int
    betting_stage: str
    belief: np.ndarray            # a copy (tracker.snapshot()), normalised to sum 1
    baseline_support: int         # board-compatible (no-update) support — the baseline size
    n_actions_replayed: int
    uniform_fallback: int


def compute_metrics(
    belief: np.ndarray, true_combo: int, baseline_support: int
) -> Dict[str, object]:
    """The §7 metric block for a resolved belief vs. its revealed ``true_combo``.

    ``belief`` is the per-combo weight vector (any nonnegative scaling — it is
    renormalised here defensively); ``true_combo`` indexes the revealed hole in the
    same combo enumeration the belief was built over.  ``baseline_support`` is the
    size of the no-update uniform prior (board- and card-removal-compatible combos,
    from :meth:`RangeTracker.baseline_support`) — the baseline the belief must beat,
    NOT the belief's own post-Bayes support (§7).  Returns a dict keyed by the
    ``range_quality`` metric columns (§6).
    """
    w = np.asarray(belief, dtype=np.float64)
    total = w.sum()
    if total > 0.0:
        w = w / total

    live_mask = w > 0.0                          # the belief's own (post-Bayes) support
    effective_support = int(baseline_support)    # the no-update baseline's size (§7)
    mass = float(w[true_combo])
    collapsed = 1 if mass < _NUMERICAL_FLOOR else 0

    # Clamp at the floor so a zeroed truth is a large finite penalty, not +inf.
    log_loss = float(-np.log(max(mass, _NUMERICAL_FLOOR)))
    if effective_support > 0:
        log_loss_uniform: Optional[float] = float(np.log(effective_support))
        net_info_gain: Optional[float] = log_loss_uniform - log_loss
    else:
        log_loss_uniform = net_info_gain = None

    if live_mask.any():
        live = w[live_mask]
        entropy: Optional[float] = float(-np.sum(live * np.log(live)))
        # Ordering check: where the true mass falls among the live support (1.0 =
        # mode).  A collapsed truth (mass ~ 0, not in the support) lands at 0.0.
        rank: Optional[float] = float(np.mean(live <= mass)) if mass > 0.0 else 0.0
    else:
        entropy = rank = None

    return {
        "true_combo_mass": mass,
        "true_combo_rank": rank,
        "effective_support": effective_support,
        "log_loss": log_loss,
        "log_loss_uniform": log_loss_uniform,
        "net_info_gain": net_info_gain,
        "collapsed_truth": collapsed,
        "entropy": entropy,
    }


class RangeQualityRecorder:
    """Buffers opponent beliefs across a hand and resolves them at showdown (§7).

    Constructed once per hand with the hero seat and the opponent seats.  The runner
    drives :meth:`capture` at each round boundary (post belief-update) and
    :meth:`resolve` once, at the terminal state.  Snapshots are only taken while the
    hero is live — once it folds the tracker goes dormant (its ``on_board_update``
    no-ops), so any later belief is stale and must not be measured; the runner
    enforces that gate before calling :meth:`capture`.
    """

    def __init__(self, hero_seat: int, opponent_seats: Iterable[int]) -> None:
        self._hero_seat = int(hero_seat)
        self._opp_seats = sorted(int(s) for s in opponent_seats if int(s) != int(hero_seat))
        self._snaps: List[BeliefSnapshot] = []

    def capture(self, tracker, betting_stage: str) -> None:
        """Buffer every live opponent seat's current belief at ``betting_stage``.

        ``tracker.snapshot()`` returns a fresh copy per live seat (including the
        hero, which we drop), so the buffered vector is immune to later in-place
        updates.  A seat that has already folded is absent from the snapshot and so
        is silently skipped — its earlier-street snapshots are already buffered.
        """
        live = tracker.snapshot()
        for seat in self._opp_seats:
            belief = live.get(seat)
            if belief is None:
                continue
            self._snaps.append(
                BeliefSnapshot(
                    seat=seat,
                    betting_stage=betting_stage,
                    belief=belief,
                    baseline_support=tracker.baseline_support(seat),
                    n_actions_replayed=tracker.replay_count(seat),
                    uniform_fallback=tracker.fallback_count(seat),
                )
            )

    def resolve(self, env, went_to_showdown: int) -> List[RangeQualityRow]:
        """Resolve every buffered snapshot into a :class:`RangeQualityRow` (§7).

        A snapshot is *resolved* iff the hand reached showdown and the snapshot's
        seat is still live at the terminal state — i.e. its hole is revealed.  The
        revealed hole is looked up in ``env.combo_index`` (the same deck-level
        enumeration the belief was built over, so the index aligns).  Every buffered
        snapshot yields a row, resolved or not, so the resolved-fraction denominator
        (§8/§11) is complete.
        """
        combo_index = env.combo_index
        rows: List[RangeQualityRow] = []
        for snap in self._snaps:
            true_combo: Optional[int] = None
            if went_to_showdown and env.players[snap.seat].is_active:
                hole = tuple(sorted(int(c) for c in env.players[snap.seat].cards))
                idx = combo_index.get(hole)
                if idx is not None:
                    true_combo = int(idx)
            rows.append(_to_row(snap, true_combo))
        return rows


def _to_row(snap: BeliefSnapshot, true_combo: Optional[int]) -> RangeQualityRow:
    """A resolved (metrics filled) or unresolved (metrics NULL) row for ``snap``."""
    common = dict(
        seat=snap.seat,
        betting_stage=snap.betting_stage,
        n_actions_replayed=snap.n_actions_replayed,
        uniform_fallback=snap.uniform_fallback,
    )
    if true_combo is None:
        return RangeQualityRow(resolved=0, **common)
    metrics = compute_metrics(snap.belief, true_combo, snap.baseline_support)
    return RangeQualityRow(resolved=1, true_combo=true_combo, **common, **metrics)


__all__ = ["BeliefSnapshot", "RangeQualityRecorder", "compute_metrics"]
