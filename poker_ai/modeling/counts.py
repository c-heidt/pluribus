"""Coarse-bucket soft counts for the learned opponent model (opponent_modeling §3, §4.2).

The learned model keys its counts on the **coarse behavioral bucket**
``k = (street, s, ctx)`` — *not* the blueprint's ``(betting_round, info_set)`` — so a
~10k-game budget saturates confidence (doc §3):

- ``street`` = ``state.betting_round``;
- ``s`` = the strength tier (:mod:`poker_ai.modeling.tiers`); preflop stays the raw
  lossless cluster;
- ``ctx`` = ``(raise_bucket ∈ {0,1,2+}, facing_bet ∈ {0,1})`` — a coarse betting
  context read from the info-set history and the legal set.

Actions collapse to **4 classes** ``{fold, call/check, raise, all_in}``.  Counts are
fractional (soft attribution at the belief replay, §6.3): each observed action
credits its class in the projected bucket by the tracker's pre-action belief mass.

The key ``π`` is a pure function of the :class:`PolicyState` (via
:func:`~environment.poker_env.decode_info_set`), so the model can be queried on the
belief-vectorized and leaf paths with no env handle (doc §4.1).
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np

from environment.poker_env import PolicyState, decode_info_set

# The 4 coarse action classes and the count-row width.
ACTION_CLASSES: Tuple[str, ...] = ("fold", "call", "raise", "all_in")
N_ACTION_CLASSES: int = 4

# betting_round index → stage name (for reading the current street's history).
_STREET_STAGE: Mapping[int, str] = {0: "pre_flop", 1: "flop", 2: "turn", 3: "river"}

# The coarse model key: (street, tier-or-cluster, raise_bucket, facing_bet).
ModelKey = Tuple[int, int, int, int]


def action_class(action: str) -> int:
    """Map a canonical action token to its coarse class index in ``ACTION_CLASSES``.

    ``call``/``check`` → 1 (they are the same "don't raise, stay in" class);
    any ``raise:<f>`` → 2; ``all_in`` → 3; ``fold`` → 0.
    """
    if action == "fold":
        return 0
    if action in ("call", "check"):
        return 1
    if action == "all_in":
        return 3
    if action.startswith("raise"):
        return 2
    raise ValueError(f"unclassifiable action token {action!r}")


def _raise_bucket(n_aggressive: int) -> int:
    """Bucket the number of aggressive actions this street into ``{0, 1, 2+}``."""
    return 0 if n_aggressive == 0 else (1 if n_aggressive == 1 else 2)


def _aggressive_this_street(history, street: int) -> int:
    """Count raise/all-in tokens in the current street's action list (a coarse ``ctx``).

    A slight over-count is possible (an all-in that is really a call still logs the
    ``all_in`` token), but ``ctx`` is a coarse bucket — the imprecision is bounded and
    harmless for separating betting situations (doc §4.4 accepts the coarse context).
    """
    stage = _STREET_STAGE.get(int(street))
    for st, actions in history:
        if st == stage:
            return sum(1 for a in actions if a == "all_in" or a.startswith("raise"))
    return 0


def model_key(state: PolicyState, tiers) -> ModelKey:
    """Project a :class:`PolicyState` onto its coarse model key ``k = (street, s, ctx)``.

    ``s`` is the strength tier for postflop clusters and the raw cluster preflop
    (:meth:`StrengthTiers.tier`); ``ctx`` is ``(raise_bucket, facing_bet)`` with
    ``facing_bet`` read from the legal set (``call`` legal ⇒ facing a bet, vs
    ``check``).  A terminal/sentinel info-set (no cluster) maps to tier ``-1`` — it is
    never a live decision node, so it only needs to be well-defined.
    """
    street = int(state.betting_round)
    facing_bet = 1 if "call" in state.legal_actions else 0
    decoded = decode_info_set(state.info_set)
    if decoded is None:
        return (street, -1, 0, facing_bet)
    cluster, history = decoded
    s = tiers.tier(street, cluster)
    n_aggr = _aggressive_this_street(history, street)
    return (street, int(s), _raise_bucket(n_aggr), facing_bet)


class CountsView:
    """Read-only view of a committed counts table — the per-hand frozen snapshot.

    Bound to the committed dict at snapshot time; because :meth:`CountsTable.commit`
    replaces (never mutates) committed rows, a held view stays frozen for the hand
    even if the next hand commits (doc §3 freeze cadence).
    """

    __slots__ = ("_committed",)

    def __init__(self, committed: Mapping[ModelKey, np.ndarray]) -> None:
        self._committed = committed

    def count(self, key: ModelKey) -> np.ndarray:
        """The 4-class soft-count row for ``key`` (a zero row if unseen)."""
        row = self._committed.get(key)
        return row.copy() if row is not None else np.zeros(N_ACTION_CLASSES, dtype=np.float64)

    def total(self, key: ModelKey) -> float:
        """Total soft-count mass ``n(k)`` at ``key`` (drives confidence, §3)."""
        row = self._committed.get(key)
        return float(row.sum()) if row is not None else 0.0


class CountsTable:
    """Coarse-key → ``float64[4]`` soft counts with hand-boundary buffer/commit.

    Observations during a hand accumulate in a buffer; :meth:`commit` folds them into
    the committed table between hands.  Committed rows are never mutated in place —
    :meth:`commit` writes *new* arrays for changed keys and shares the rest — so a
    :class:`CountsView` taken before the commit stays frozen (doc §3).
    """

    def __init__(self) -> None:
        self._committed: Dict[ModelKey, np.ndarray] = {}
        self._buffer: Dict[ModelKey, np.ndarray] = {}

    def buffer_observation(self, key: ModelKey, action_class_idx: int, mass: float) -> None:
        """Credit ``mass`` to ``action_class_idx`` at ``key`` (soft count, §6.3)."""
        row = self._buffer.get(key)
        if row is None:
            row = np.zeros(N_ACTION_CLASSES, dtype=np.float64)
            self._buffer[key] = row
        row[action_class_idx] += float(mass)

    def commit(self) -> None:
        """Fold the buffered observations into the committed table (hand boundary)."""
        if not self._buffer:
            return
        new = dict(self._committed)                      # shallow: shares frozen rows
        for key, row in self._buffer.items():
            base = new.get(key)
            new[key] = row.copy() if base is None else base + row   # new array for changed
        self._committed = new
        self._buffer = {}

    def snapshot(self) -> CountsView:
        """A frozen read-only view of the committed table (the per-hand snapshot)."""
        return CountsView(self._committed)

    # ------------------------------------------------------------------ #
    # Persistence (committed table only; the buffer is transient)
    # ------------------------------------------------------------------ #
    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Serialise the committed table to ``(keys[n,4] int64, rows[n,4] float64)``."""
        if not self._committed:
            return (np.zeros((0, 4), np.int64), np.zeros((0, 4), np.float64))
        items = list(self._committed.items())            # one pass, no re-lookup
        keys = np.array([k for k, _ in items], dtype=np.int64)
        rows = np.stack([v for _, v in items]).astype(np.float64)
        return keys, rows

    @classmethod
    def from_arrays(cls, keys: np.ndarray, rows: np.ndarray) -> "CountsTable":
        """Rebuild a :class:`CountsTable` from :meth:`to_arrays` output."""
        t = cls()
        t._committed = {tuple(int(x) for x in k): r.astype(np.float64) for k, r in zip(keys, rows)}
        return t


__all__ = [
    "ACTION_CLASSES",
    "N_ACTION_CLASSES",
    "ModelKey",
    "action_class",
    "model_key",
    "CountsView",
    "CountsTable",
]
