"""Strength tiers from the clustering's own LUT centroids (opponent_modeling §4.4).

The learned opponent model keys its coarse buckets on ``(street, s, ctx)`` where
``s`` is a **strength tier**: a ~10-bin coarsening of the blueprint's postflop card
clusters, so a ~10k-game budget saturates the per-bucket confidence instead of
spreading counts across the full cluster abstraction (doc §3, §10).

**No new abstraction is built.** The tiers are read straight out of the LUT the
codebase already ships (``<lut_dir>/centroids.joblib``). The clustering's features
already encode strength:

- the **river** centroid is ``[win, loss, tie]``
  (:mod:`information_abstraction.build.ehs`), so a river cluster's equity is
  ``win + tie/2``;
- the **turn** / **flop** centroids are histograms over the *next* street's
  clusters, so their equity is the histogram dotted with the next street's equity
  (equity-of-equity — a potential-aware strength scalar).

Roll those back to one scalar per cluster and bin into ``n_tiers`` equal-frequency
bins per street. **Preflop stays lossless** — it is already coarse and the
highest-traffic street — so no preflop tier table is produced; the model keys
preflop on the raw bucket (doc §4.4).

The identical derivation runs unchanged on the production LUT (same centroid
structure). Build once, persist next to the LUT, and load into the model.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

# betting_round index → centroid-dict street key.  Preflop (0) is intentionally
# absent: it stays lossless (§4.4).  These are the streets that carry a tier table.
_STREET_KEY: Mapping[int, str] = {1: "flop", 2: "turn", 3: "river"}
_POSTFLOP_ROUNDS: Tuple[int, ...] = (1, 2, 3)

DEFAULT_N_TIERS = 10


def _cluster_equities(centroids: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-cluster scalar equity for flop/turn/river from the raw centroids.

    ``river_eq = win + tie/2``; ``turn_eq = turn_hist · river_eq``;
    ``flop_eq = flop_hist · turn_eq`` (potential-aware roll-back, §4.4).  The turn /
    flop histogram width must equal the number of clusters on the next street (the
    histogram is *over* those clusters); a mismatch is a corrupt / mismatched LUT and
    raises rather than silently producing garbage.
    """
    river = np.asarray(centroids["river"], dtype=np.float64)
    if river.ndim != 2 or river.shape[1] != 3:
        raise ValueError(
            f"river centroid must be (n_clusters, 3) = [win, loss, tie]; "
            f"got shape {river.shape}"
        )
    river_eq = river[:, 0] + 0.5 * river[:, 2]

    turn = np.asarray(centroids["turn"], dtype=np.float64)
    if turn.shape[1] != river_eq.shape[0]:
        raise ValueError(
            f"turn centroid width {turn.shape[1]} != n_river_clusters "
            f"{river_eq.shape[0]} — the turn feature is a histogram over river "
            "clusters; the LUT streets are inconsistent."
        )
    turn_eq = turn @ river_eq

    flop = np.asarray(centroids["flop"], dtype=np.float64)
    if flop.shape[1] != turn_eq.shape[0]:
        raise ValueError(
            f"flop centroid width {flop.shape[1]} != n_turn_clusters "
            f"{turn_eq.shape[0]} — the flop feature is a histogram over turn "
            "clusters; the LUT streets are inconsistent."
        )
    flop_eq = flop @ turn_eq

    return {"flop": flop_eq, "turn": turn_eq, "river": river_eq}


def _equal_frequency_tiers(eq: np.ndarray, n_tiers: int) -> np.ndarray:
    """Bin ``eq`` into ``n_tiers`` equal-frequency tiers, monotone in equity.

    Interior quantile edges + ``np.digitize`` (right-open), so tier ``t`` holds the
    ``t``-th equity decile — low tier = weak, high tier = strong.  Fewer distinct
    equities than tiers simply leaves the upper tiers empty (still well-defined).
    """
    if n_tiers < 1:
        raise ValueError(f"n_tiers must be >= 1, got {n_tiers}")
    if n_tiers == 1:
        return np.zeros(eq.shape[0], dtype=np.int16)
    edges = np.quantile(eq, np.linspace(0.0, 1.0, n_tiers + 1)[1:-1])
    return np.digitize(eq, edges).astype(np.int16)


@dataclasses.dataclass(frozen=True)
class StrengthTiers:
    """Postflop ``cluster → tier`` tables — the ``s`` dimension of the model key.

    ``tables`` maps a betting-round index (1=flop, 2=turn, 3=river) to an
    ``int16`` array indexed by cluster id, giving that cluster's tier in
    ``[0, n_tiers)``.  Preflop (round 0) is absent by construction — it stays
    lossless (§4.4); :meth:`tier` returns the raw cluster there so the model key
    degrades to the lossless bucket.
    """

    tables: Mapping[int, np.ndarray]
    n_tiers: int

    def tier(self, betting_round: int, cluster: int) -> int:
        """Tier for ``cluster`` on ``betting_round``.

        Postflop: the binned tier in ``[0, n_tiers)``.  Preflop (no table): the raw
        ``cluster`` unchanged (lossless key, §4.4).  An out-of-range cluster on a
        postflop street is a caller/LUT-mismatch bug and raises.
        """
        table = self.tables.get(int(betting_round))
        if table is None:
            return int(cluster)                       # preflop stays lossless
        c = int(cluster)
        if c < 0 or c >= table.shape[0]:
            raise IndexError(
                f"cluster {c} out of range for round {betting_round} "
                f"(n_clusters={table.shape[0]})"
            )
        return int(table[c])

    def save(self, path) -> None:
        """Persist as an ``npz`` (``n_tiers`` + one ``street_<r>`` array per round)."""
        arrays = {f"street_{r}": np.asarray(t) for r, t in self.tables.items()}
        np.savez(path, n_tiers=np.int64(self.n_tiers), **arrays)

    @classmethod
    def load(cls, path) -> "StrengthTiers":
        """Load a :meth:`save`d ``npz`` back into a :class:`StrengthTiers`."""
        with np.load(path) as data:
            n_tiers = int(data["n_tiers"])
            tables = {
                int(k.split("_")[1]): data[k].astype(np.int16)
                for k in data.files
                if k.startswith("street_")
            }
        return cls(tables=tables, n_tiers=n_tiers)


def build_tiers_from_centroids(
    centroids: Mapping[str, np.ndarray],
    n_tiers: int = DEFAULT_N_TIERS,
) -> StrengthTiers:
    """Build the postflop tier tables from an in-memory centroids dict (§4.4).

    Separated from :func:`build_tiers` so tests can drive it with hand-built
    centroids without touching disk.
    """
    eq = _cluster_equities(centroids)
    tables = {
        r: _equal_frequency_tiers(eq[_STREET_KEY[r]], n_tiers)
        for r in _POSTFLOP_ROUNDS
    }
    return StrengthTiers(tables=tables, n_tiers=int(n_tiers))


def build_tiers(lut_dir, n_tiers: int = DEFAULT_N_TIERS) -> StrengthTiers:
    """Build the tier tables from ``<lut_dir>/centroids.joblib`` (§4.4).

    Loads the clustering's centroids (the same artifact the LUT build produced) and
    bins each postflop street's rolled-back equity into ``n_tiers`` tiers.  Pure
    read — never rewrites the LUT.
    """
    import os

    import joblib

    centroids = joblib.load(os.path.join(str(lut_dir), "centroids.joblib"))
    return build_tiers_from_centroids(centroids, n_tiers=n_tiers)


def cluster_equities(lut_dir) -> Dict[str, np.ndarray]:
    """Expose the per-cluster equity scalars for ``<lut_dir>`` (diagnostics/tests)."""
    import os

    import joblib

    centroids = joblib.load(os.path.join(str(lut_dir), "centroids.joblib"))
    return _cluster_equities(centroids)


__all__ = [
    "StrengthTiers",
    "build_tiers",
    "build_tiers_from_centroids",
    "cluster_equities",
    "DEFAULT_N_TIERS",
]
