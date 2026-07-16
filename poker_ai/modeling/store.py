"""Per-opponent model registry with hand-boundary commit (opponent_modeling §4.2).

:class:`ModelStore` owns one :class:`~poker_ai.modeling.counts.CountsTable` per stable
``opponent_id`` and mediates the freeze cadence of doc §3: observations buffer during
a hand and :meth:`commit_hand` folds them in at the boundary, so the within-hand model
stays frozen (breaks the count↔belief feedback loop).  It persists as an ``npz`` of the
count tables (the blueprint itself is not duplicated).

The ``snapshot(opponent_id) -> OpponentModel`` frozen view is wired once
``BayesOpponentModel`` lands (the Bayes step); until then :meth:`counts_snapshot`
exposes the raw frozen :class:`~poker_ai.modeling.counts.CountsView`.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from poker_ai.modeling.counts import CountsTable, CountsView, ModelKey


class ModelStore:
    """One counts table + model per ``opponent_id``; buffers and commits per hand.

    Parameters
    ----------
    blueprint_policy
        The base blueprint backing every model's per-state prior; required for
        :meth:`snapshot`.  The counts-only path (buffer/commit/save) does not need it.
    tiers
        The :class:`~poker_ai.modeling.tiers.StrengthTiers` for the model key ``π``;
        required for :meth:`snapshot`.
    tau, p_max
        The Dirichlet prior strength and confidence cap handed to each
        :class:`~poker_ai.modeling.model.BayesOpponentModel` (doc §3 defaults).
    """

    def __init__(
        self, *, blueprint_policy=None, tiers=None, tau: float = 50.0, p_max: float = 0.8
    ) -> None:
        self._blueprint_policy = blueprint_policy
        self._tiers = tiers
        self._tau = float(tau)
        self._p_max = float(p_max)
        self._tables: Dict[str, CountsTable] = {}

    def _table(self, opponent_id: str) -> CountsTable:
        t = self._tables.get(opponent_id)
        if t is None:
            t = CountsTable()
            self._tables[opponent_id] = t
        return t

    def opponent_ids(self) -> List[str]:
        """The registered opponent ids (insertion order)."""
        return list(self._tables)

    # ------------------------------------------------------------------ #
    # Observation buffering + hand-boundary commit (§3 freeze cadence)
    # ------------------------------------------------------------------ #
    def buffer_observation(
        self, opponent_id: str, key: ModelKey, action_class_idx: int, mass: float
    ) -> None:
        """Buffer one soft-count observation for ``opponent_id`` (committed at hand end)."""
        self._table(opponent_id).buffer_observation(key, action_class_idx, mass)

    def commit_hand(self) -> None:
        """Commit every opponent's buffered observations (run at the hand boundary)."""
        for t in self._tables.values():
            t.commit()

    def counts_snapshot(self, opponent_id: str) -> CountsView:
        """The frozen committed counts view for ``opponent_id`` (creates an empty one)."""
        return self._table(opponent_id).snapshot()

    def snapshot(self, opponent_id: str):
        """The frozen per-hand :class:`BayesOpponentModel` view for ``opponent_id`` (§3).

        Binds the current committed counts (a frozen :class:`CountsView`) into a
        model over the store's blueprint + tiers; the returned model reads that
        snapshot for the whole hand while new observations buffer separately.
        """
        if self._blueprint_policy is None or self._tiers is None:
            raise ValueError(
                "ModelStore.snapshot needs blueprint_policy and tiers "
                "(pass them to ModelStore(...))"
            )
        from poker_ai.modeling.model import BayesOpponentModel

        return BayesOpponentModel(
            self._blueprint_policy,
            self._table(opponent_id).snapshot(),
            self._tiers,
            tau=self._tau,
            p_max=self._p_max,
        )

    # ------------------------------------------------------------------ #
    # Persistence — npz of the count tables (§4.2)
    # ------------------------------------------------------------------ #
    def save(self, path) -> None:
        """Persist all opponents' committed count tables to ``path`` (npz)."""
        ids = list(self._tables)
        arrays: Dict[str, np.ndarray] = {"opponent_ids": np.array(ids)}
        for i, oid in enumerate(ids):
            keys, rows = self._tables[oid].to_arrays()
            arrays[f"keys_{i}"] = keys
            arrays[f"rows_{i}"] = rows
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path, *, tiers=None) -> "ModelStore":
        """Load a :meth:`save`d ``npz`` back into a :class:`ModelStore`."""
        store = cls(tiers=tiers)
        with np.load(path, allow_pickle=False) as data:
            ids = [str(x) for x in data["opponent_ids"]]
            for i, oid in enumerate(ids):
                store._tables[oid] = CountsTable.from_arrays(
                    data[f"keys_{i}"], data[f"rows_{i}"]
                )
        return store


__all__ = ["ModelStore"]
