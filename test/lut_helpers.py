"""Shared ``card_info_lut`` stand-ins for tests.

**The single-cluster LUT is banned.**  ``card_info_lut = defaultdict(lambda:
defaultdict(lambda: 0))`` — every hand on every board mapping to cluster 0 — used to be
copy-pasted into fifteen test modules, and it makes a whole bug class structurally
invisible:

* every future street collapses to ``n_rows == 1``, so anything that depends on the
  cluster→dense-row mapping (``ClusterMapper._universe`` / ``cof`` / the gather-scatter
  seam) is trivially correct;
* every board maps a hole to the same cluster, so a cache that goes stale when the board
  changes still returns the right answer.

Both have bitten.  A board-stale model-row cache passed the entire stub-LUT suite
unchanged.  And on 2026-09-01 the compiled search core was found to diverge from the
Python vector walk *only* when a future street has more than one cluster row — the
parity gate had been building its envs on the single-cluster stub, i.e. the one
configuration in which the two engines agree by construction.

:func:`install_cluster_lut` is the replacement: deterministic, data-free (no LUT file,
so nothing skips), multi-cluster, and keyed on **both** the hole and the board.  Use the
real LUT (``test.search._helpers._real_lut``) where a test needs true cluster semantics;
use this where it just needs ``info_set`` to resolve.
"""

import collections

#: Street order of ``card_info_lut``, and the default cluster count per street.  The
#: numbers mirror the shipped 20-card LUT (25/50/50/45) so row counts are in a realistic
#: range rather than degenerate at either end.
STREETS = ("pre_flop", "flop", "turn", "river")
DEFAULT_CLUSTERS = (25, 50, 50, 45)


class ClusterLUT:
    """One street's ``(hole, board) -> cluster`` map: deterministic, data-free.

    Keys arrive as ``tuple(sorted(hole) + sorted(board))`` — the shape
    :func:`information_abstraction.lookup.clusters_for_board` builds for a plain-dict
    street and the shape ``PokerEnv.cluster_for`` uses.  The cluster id is a pure
    function of the WHOLE key, so the same hole lands in different clusters on different
    boards; that is the property a single-cluster or hole-only stand-in throws away.

    Board-conflicting combos are not this object's problem — ``clusters_for_board``
    masks them to ``-1`` from the board itself, whatever the LUT says.
    """

    __slots__ = ("n",)

    def __init__(self, n_clusters: int) -> None:
        self.n = int(n_clusters)

    def __getitem__(self, key):
        try:
            h = 0
            for c in key:
                h = (h * 31 + int(c)) & 0xFFFFFFFF
        except TypeError:                      # a scalar key (some callers pass one)
            h = int(key) & 0xFFFFFFFF
        return h % self.n

    def get(self, key, default=None):
        try:
            return self[key]
        except (TypeError, ValueError):
            return default

    def __contains__(self, key) -> bool:
        return True

    def __repr__(self) -> str:
        return f"ClusterLUT(n_clusters={self.n})"


class HoleClusterLUT:
    """A street stand-in whose cluster depends on the HOLE ONLY, not the board.

    Deliberately board-independent, and the one case where that is the point: it makes
    the same fixed set of clusters reachable on *every* board, which is what
    ``ClusterMapper._universe`` needs in order to be exercised with more than one row.

    ⚠️ Not a general-purpose stand-in — use :class:`ClusterLUT` for that.  Board
    independence is one of the two degeneracies that made the old single-cluster stub
    hide bugs (a board-stale cache still returns the right answer under it), so this is
    for tests that specifically want a board-invariant universe and nothing else.
    """

    __slots__ = ("n",)

    def __init__(self, n: int = 4) -> None:
        self.n = int(n)

    def __getitem__(self, key):
        return (int(key[0]) + int(key[1])) % self.n

    def __repr__(self) -> str:
        return f"HoleClusterLUT(n={self.n})"


class LosslessClusterLUT:
    """A lossless street map: a unique, ORDER-INDEPENDENT cluster id per (hole, board).

    The obvious lossless stub — ``defaultdict(itertools.count().__next__)`` — hands out
    ids in *first-access order*, so it is stateful: the id a key receives depends on the
    order the LUT happens to be queried in.  That is fine for a single solve, but it
    makes the LUT unusable for comparing two engines, because a different traversal
    order relabels the clusters and every downstream dense-row index shifts.  It cost
    real debugging time on 2026-09-01, when it looked like the compiled search core
    diverged from the Python walk.

    Here the id is a positional encoding of the key's card ranks within the deck, so it
    is injective (lossless) AND a pure function of the key.
    """

    __slots__ = ("_rank", "_base")

    def __init__(self, deck):
        cards = sorted({int(c) for c in deck})
        self._rank = {c: i for i, c in enumerate(cards)}
        self._base = len(cards) + 1

    def __getitem__(self, key):
        out = 0
        for c in key:
            out = out * self._base + self._rank[int(c)] + 1
        return out

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, TypeError):
            return default

    def __contains__(self, key) -> bool:
        try:
            self[key]
            return True
        except (KeyError, TypeError):
            return False


def lossless_lut(deck):
    """A lossless, order-independent ``card_info_lut`` over ``deck``'s cards."""
    return collections.defaultdict(
        lambda: LosslessClusterLUT(deck),
        {name: LosslessClusterLUT(deck) for name in STREETS},
    )


def cluster_lut(n_clusters=DEFAULT_CLUSTERS):
    """A multi-cluster ``card_info_lut``, street-keyed.

    Standalone so callers that build a LUT without an env to install it on
    (evaluation configs, leaf fixtures) share one implementation with
    :func:`install_cluster_lut`.  An unknown street key falls back to the river count.
    """
    sizes = dict(zip(STREETS, n_clusters))
    river = sizes.get("river", DEFAULT_CLUSTERS[-1])
    return collections.defaultdict(
        lambda: ClusterLUT(river),
        {name: ClusterLUT(size) for name, size in sizes.items()},
    )


def install_cluster_lut(env, n_clusters=DEFAULT_CLUSTERS):
    """Give ``env`` a multi-cluster ``card_info_lut``; returns ``env``.

    Drop-in for the banned single-cluster stub: ``info_set`` resolves for any
    ``(hole, board)`` as before, but the future streets now carry many cluster rows.
    """
    env.card_info_lut = cluster_lut(n_clusters)
    return env
