"""Shared state and configuration for the depth-limited subgame solver (§6.5).

This module is the **regime-agnostic foundation** of the solver: it owns the
in-memory CFR tables and the operations on them (regret matching, accumulation,
Linear-CFR discount, averaging, freezing), plus the config and the ``Key``
helpers.  It deliberately imports **nothing** from the rest of the ``search``
package at runtime — the two CFR regimes (``mccfr.py`` / ``vector.py``) and the
orchestrator (``solver.py``) import *it*, so keeping it a leaf of the import
graph is what lets the package split cleanly without an import cycle (the
``LeafConfig`` reference on :class:`SolverConfig` is a ``TYPE_CHECKING``-only
annotation for exactly this reason).

The tables are keyed by ``Key = (public_key, hand_row)`` where ``public_key`` is
``env.public_key`` (the shared public-state identifier) and ``hand_row`` is the
acting seat's combo index on the **root street** (lossless) or its LUT cluster id
on later streets (:func:`_hand_row`).  Rows are **per-node width** — the number of
legal actions at that public node — so the same dicts hold both the MCCFR regime's
scalar rows and (later) the vector regime's per-combo rows without any reshape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Sequence, Tuple

import numpy as np

from poker_ai.blueprint.tree_utils import calculate_strategy_from_row

if TYPE_CHECKING:  # avoid a runtime import cycle (leaf -> policy -> solver_state)
    from environment.poker_env import PokerEnv
    from poker_ai.search.leaf import LeafConfig


# A public node identifier.  ``env.public_key`` is ``(betting_stage, history)``;
# the meta-game extends it to ``(env.public_key, "META", seat)``.  Either way it
# is a hashable tuple — the alias documents intent, it is not enforced.
PublicKey = Tuple
# ``(public_key, hand_row)`` — hand_row is a combo index (root street) or an LUT
# cluster id (later streets); see :func:`_hand_row`.
Key = Tuple[PublicKey, int]


@dataclass(frozen=True)
class SolverConfig:
    """Static solver hyperparameters (§6.5).

    ``leaf`` is required and carries the continuation-strategy fleet + the
    ``use_decision_free_equity`` toggle consumed at depth-limit leaves and
    forced-runout terminals.
    """

    leaf: "LeafConfig"
    # Defaults tuned for a 6-player game on a ~48-core node, early-testing grade
    # (not paper-accurate).  At ``LeafConfig.n_rollouts == 1`` an MCCFR iteration is
    # ~35 ms on a multiway flop and ~5 ms on late/heads-up subgames, so wall time —
    # not the iteration cap — is still the binding stop: within ``max_wall_seconds``
    # a replica reaches ~285 (flop) to ~2000 (late) iterations, and ``max_iterations``
    # stays a generous, effectively-inert safety cap.  (Cutting rollouts 8→1 banks
    # as ~5× more iterations at the same wall — better hole coverage — rather than a
    # shorter wall; lower ``max_wall_seconds`` if you want turnaround over coverage.)
    max_iterations: int = 5_000
    max_wall_seconds: float = 10.0  # per-search wall budget (the binding stop)
    # Linear-CFR discount cadence.  Kept well below the per-replica iteration count
    # of the *expensive* subgames (multiway flop ~285/replica) so the discount fires
    # several times there — at the old 100 it barely engaged on those (and never at
    # n_rollouts=8, ~50 iters/replica).  Cheap late subgames just discount more often.
    discount_interval: int = 50
    # Parallel search (§6.7 row 11).  ``None`` → resolve to a cpu-based default
    # (cpu_count-1, SLURM-aware) — the sanctioned way to spend the wall budget is W
    # independent replicas merged once, so on a 48-core node this fans out to ~47
    # replicas.  ``1`` → the serial loop (bit-for-bit identical to the pre-parallel
    # solver); ``>1`` → that many independent MCCFR replicas, merged once at the end
    # (:meth:`SolverState.accumulate`).  See :mod:`poker_ai.search.parallel`.
    workers: "int | None" = None


def _hand_row(env: "PokerEnv", combo: Sequence[int], street_at_root: int) -> int:
    """Row id for ``combo`` at ``env``'s current node.

    Per-combo (lossless) on the root street, an LUT cluster id on later streets
    (§6.5).  ``combo`` is sorted to match ``env.combo_index`` / ``cluster_for``
    key order.  The combo is assumed board-compatible (the solver only ever keys
    on the acting seat's sampled, board-disjoint hole).
    """
    c = tuple(sorted(int(x) for x in combo))
    if env.betting_round == street_at_root:
        return int(env.combo_index[c])
    return int(env.cluster_for(c))


class _CountingCache:
    """A dict-backed memo that counts ``get`` hits and misses (eval doc §6, §9.1).

    Wraps the two search-lifetime caches (``leaf_value_cache`` / ``runout_cache``)
    so their hit/miss rate can be logged without threading counters through every
    call site: the leaf evaluator (:mod:`poker_ai.search.leaf`) receives the
    ``runout_cache`` as a bare mapping, and its ``.get`` / ``[]`` accesses count
    automatically.  Only the operations the caches (and the tests) actually use
    are implemented — ``get``/``[]``/``in``/``len``/iteration — so a plain
    ``dict`` stays a valid drop-in wherever counting is not wanted (leaf's
    standalone per-call memo when no shared cache is supplied).  ``__slots__``
    keeps it deepcopy-/pickle-friendly for the parallel replicas (§6.7).
    """

    __slots__ = ("_data", "hits", "misses")

    def __init__(self) -> None:
        self._data: Dict = {}
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key, default=None):
        if key in self._data:
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return default

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value) -> None:
        self._data[key] = value

    def __contains__(self, key) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


@dataclass
class SearchStats:
    """Per-search instrumentation counters (eval doc §6 ``decisions`` grain, §9.1).

    Snapshotted off a solved :class:`SolverState` (:meth:`SolverState.stats_snapshot`)
    and carried on :class:`~poker_ai.search.solver.SearchResult` so the evaluation
    logger (doc §9.2) can persist tree shape (``node_count`` / ``unique_pubkeys``)
    and cache behaviour without reaching into solver internals.  ``cache_hits`` /
    ``cache_misses`` roll the three caches together for the single schema columns of
    the same name; the per-cache fields stay available for finer analysis.
    """

    node_count: int = 0           # decision-node visits over the whole search
    unique_pubkeys: int = 0       # distinct public keys (== len(legal_at))
    legal_at_hits: int = 0        # node revisits (legal set already registered)
    legal_at_misses: int = 0      # first registrations (== unique_pubkeys)
    leaf_cache_hits: int = 0
    leaf_cache_misses: int = 0
    leaf_cache_size: int = 0
    runout_cache_hits: int = 0
    runout_cache_misses: int = 0
    runout_cache_size: int = 0

    @property
    def cache_hits(self) -> int:
        """All three caches' hits, for the ``decisions.cache_hits`` column."""
        return self.legal_at_hits + self.leaf_cache_hits + self.runout_cache_hits

    @property
    def cache_misses(self) -> int:
        """All three caches' misses, for the ``decisions.cache_misses`` column."""
        return self.legal_at_misses + self.leaf_cache_misses + self.runout_cache_misses

    def combined_with(self, other: "SearchStats") -> "SearchStats":
        """Sum two snapshots (parallel replicas, §6.7 row 11).

        Additive counters (visits, hits/misses, cache sizes) sum; ``unique_pubkeys``
        does **not** — distinct-key counts overlap across replicas that walk the
        same tree, so the caller sets it from the merged state's ``legal_at``.
        """
        merged = SearchStats(
            node_count=self.node_count + other.node_count,
            unique_pubkeys=max(self.unique_pubkeys, other.unique_pubkeys),
            legal_at_hits=self.legal_at_hits + other.legal_at_hits,
            legal_at_misses=self.legal_at_misses + other.legal_at_misses,
            leaf_cache_hits=self.leaf_cache_hits + other.leaf_cache_hits,
            leaf_cache_misses=self.leaf_cache_misses + other.leaf_cache_misses,
            leaf_cache_size=self.leaf_cache_size + other.leaf_cache_size,
            runout_cache_hits=self.runout_cache_hits + other.runout_cache_hits,
            runout_cache_misses=self.runout_cache_misses + other.runout_cache_misses,
            runout_cache_size=self.runout_cache_size + other.runout_cache_size,
        )
        return merged


@dataclass
class SolverState:
    """In-memory CFR tables for one search (warm-start carrier, §6.5).

    All four dicts are mutated in place across iterations.  ``regret`` and
    ``strat_sum`` rows are float64 and exactly ``width(public_key)`` wide;
    ``frozen`` pins a probability vector for the bot's actual-hand rows (§5).
    """

    regret: Dict[Key, np.ndarray] = field(default_factory=dict)
    strat_sum: Dict[Key, np.ndarray] = field(default_factory=dict)
    legal_at: Dict[PublicKey, Tuple[str, ...]] = field(default_factory=dict)
    actor_at: Dict[PublicKey, int] = field(default_factory=dict)
    frozen: Dict[Key, np.ndarray] = field(default_factory=dict)
    # Vector regime (§6.5): per-public-node ``(n_combos, width)`` matrices, the
    # combo axis indexed by ``combo_index`` (lossless, every depth).  These are the
    # vector regime's native storage — the hot loop is one vectorised op per node
    # rather than ~n_combos dict rows.  A given ``SolverState`` is only ever written
    # by one regime, so ``regret``/``strat_sum`` (MCCFR) and ``vregret``/``vstrat``
    # (vector) never both populate; the readers below dispatch on which is present.
    vregret: Dict[PublicKey, np.ndarray] = field(default_factory=dict)
    vstrat: Dict[PublicKey, np.ndarray] = field(default_factory=dict)
    # Search-lifetime caches (§6.4.2, §6.7 Tier 1).  Both hold values that are
    # **invariant across CFR iterations** for a fixed key, so they are *not*
    # touched by ``discount`` and persist across a warm-started re-search:
    #   ``leaf_value_cache`` — ``continuation_value`` keyed by
    #     ``(leaf public_key, all-seat holes, profile)`` (§6.4.1).
    #   ``runout_cache`` — exact decision-free ``runout_equity`` keyed by
    #     ``(all-seat holes, runout snapshot)``; shared by the leaf rollouts
    #     (a leaf's four bias calls) and the forced-runout terminal (§6.4.2).
    leaf_value_cache: _CountingCache = field(default_factory=_CountingCache)
    runout_cache: _CountingCache = field(default_factory=_CountingCache)
    # Walk instrumentation (eval doc §9.1) — cumulative over the search; the
    # legal-action cache (``legal_at``) is a plain dict, so its hit/miss and the
    # node-visit tally are counted explicitly in :meth:`ensure_node` rather than by
    # a ``_CountingCache`` wrapper (``legal_at`` is also read incidentally by the
    # policy reader and strategy ops, which must not count as node visits).
    node_count: int = 0
    legal_at_hits: int = 0
    legal_at_misses: int = 0

    @classmethod
    def empty(cls) -> "SolverState":
        return cls()

    def reset_counters(self) -> None:
        """Zero the per-search walk / cache tallies, preserving tables + caches.

        The instrumentation counters (``node_count``, ``legal_at`` hits/misses, and
        the two value caches' hit/miss tallies) are cumulative over a state's
        lifetime.  A **warm-started re-search** reuses the state for a *new* search
        invocation whose ``decisions`` row (eval doc §6, §9.1) must report only that
        invocation's work — matching ``iterations_run``, which ``run_loop`` counts
        fresh per ``solve()`` call.  Without this reset a re-search's snapshot would
        report its own work *plus* every prior solve on the same state (serial), or
        the warm baseline's counters multiplied by the replica count (parallel).

        Only the tallies reset: ``legal_at`` / ``regret`` / ``strat_sum`` / ``frozen``
        and the cache **contents** are preserved, so a value cached by the prior
        solve correctly scores as a *hit* for the re-search.  ``unique_pubkeys`` is
        read from ``len(legal_at)`` (the whole widened tree), so it is unaffected.
        """
        self.node_count = 0
        self.legal_at_hits = 0
        self.legal_at_misses = 0
        for cache in (self.leaf_value_cache, self.runout_cache):
            if hasattr(cache, "hits"):
                cache.hits = 0
                cache.misses = 0

    @classmethod
    def accumulate(
        cls,
        states: Sequence["SolverState"],
        *,
        baseline: "SolverState | None" = None,
    ) -> "SolverState":
        """Fold independent-replica tables into one merged state (§6.7 row 11).

        Each parallel MCCFR replica runs the full traverser rotation over its own
        seeded substream and returns its :class:`SolverState`; this folds them into
        a single state by **summing** the cumulative ``regret`` / ``strat_sum`` (and
        the vector regime's ``vregret`` / ``vstrat``) over the union of keys.  Rows
        for a given key share a width because the legal set is a deterministic
        function of ``public_key`` and every replica inherits the same warm-start
        widening, so the per-key arrays add directly.  ``average_sigma`` normalises
        the summed ``strat_sum`` per node, so the merged average policy is a valid
        reach-weighted average over all ``W × iterations``.

        ``baseline`` is the warm-start state every replica was seeded from (``None``
        for a fresh search).  When given it is added **exactly once** and each
        replica contributes only its delta over it (``replica − baseline``) —
        otherwise the shared warm-start regrets would be counted ``W`` times.  Its
        ``frozen`` rows (pinned actual hands) and node structure carry through.  The
        per-iteration leaf/runout value caches are dropped (recomputed on demand).

        Note: a replica also discounts the baseline rows along with its own
        accumulation, so ``replica − baseline`` is only approximately the replica's
        fresh contribution under Linear-CFR; the fresh-search path (no baseline) is
        exact, and a warm re-search is a short refinement where the drift is small.
        """
        _TABLES = ("regret", "strat_sum", "vregret", "vstrat")
        out = cls()
        if baseline is not None:
            for pk, legal in baseline.legal_at.items():
                out.legal_at[pk] = legal
            for pk, actor in baseline.actor_at.items():
                out.actor_at[pk] = actor
            out.frozen = {k: v.copy() for k, v in baseline.frozen.items()}
            for name in _TABLES:
                dst = getattr(out, name)
                for key, row in getattr(baseline, name).items():
                    dst[key] = row.copy()
        for st in states:
            for pk, legal in st.legal_at.items():
                out.legal_at.setdefault(pk, legal)
            for pk, actor in st.actor_at.items():
                out.actor_at.setdefault(pk, actor)
            for name in _TABLES:
                dst = getattr(out, name)
                base = getattr(baseline, name) if baseline is not None else None
                for key, row in getattr(st, name).items():
                    b = base.get(key) if base is not None else None
                    delta = row - b if b is not None else row
                    acc = dst.get(key)
                    if acc is None:
                        dst[key] = delta.copy()
                    else:
                        acc += delta
        return out

    # ------------------------------------------------------------------
    # Node registration (+ warm-start widening)
    # ------------------------------------------------------------------

    def width(self, public_key: PublicKey) -> int:
        return len(self.legal_at[public_key])

    def ensure_node(
        self, public_key: PublicKey, legal_actions: Sequence[str], actor: int
    ) -> None:
        """Register a public node's legal-action list and acting seat.

        First visit records ``legal_at``/``actor_at``.  On a **widened** legal
        set (warm-start re-search after an off-tree action was injected at this
        node) every row under ``public_key`` is grown to the new width, with old
        columns preserved and injected actions zero-initialised — so prior
        regrets/strategy survive the re-search.  Widening only ever *adds*
        actions; the action→column mapping is by string, so a reordered legal
        set is handled correctly.
        """
        legal = tuple(legal_actions)
        prev = self.legal_at.get(public_key)
        # Node-visit + legal_at hit/miss tally (eval doc §9.1): a first
        # registration is a miss (and a new distinct public key), a revisit — even
        # a widening re-search — is a hit.  ``node_count`` == hits + misses is the
        # decision-node-visit count; ``len(legal_at)`` == misses is unique_pubkeys.
        self.node_count += 1
        if prev is None:
            self.legal_at_misses += 1
            self.legal_at[public_key] = legal
            self.actor_at[public_key] = actor
            return
        self.legal_at_hits += 1
        if prev != legal:
            self._grow_rows(public_key, prev, legal)
            self.legal_at[public_key] = legal

    def ensure_vnode(
        self,
        public_key: PublicKey,
        legal_actions: Sequence[str],
        actor: int,
        n_combos: int,
        n_rivers: "int | None" = None,
    ) -> None:
        """Register a public node and lazily allocate its vector-regime matrices.

        Like :meth:`ensure_node` (it delegates the ``legal_at``/``actor_at``
        bookkeeping and warm-start widening — which now also grows the matrix
        **columns**), but additionally allocates the ``vregret``/``vstrat``
        matrices for ``public_key`` on first visit.  The combo axis is indexed by
        ``combo_index`` (lossless at every depth, §6.5).

        ``n_rivers`` adds a **river axis** for the river-conditioned turn regime
        (§6.5): a river-stage node below the turn→river chance node carries a
        per-river strategy, so its matrices are ``(n_combos, n_rivers, width)``.
        Turn-stage nodes (and every river-subgame node) pass ``None`` and stay
        ``(n_combos, width)``.  The action ``width`` is always the **last** axis,
        so the rest of the table ops (regret matching, widening) are axis-aware.
        """
        self.ensure_node(public_key, legal_actions, actor)
        if public_key not in self.vregret:
            width = len(self.legal_at[public_key])
            shape = (n_combos, width) if n_rivers is None else (n_combos, n_rivers, width)
            self.vregret[public_key] = np.zeros(shape, dtype=np.float64)
            self.vstrat[public_key] = np.zeros(shape, dtype=np.float64)

    def _grow_rows(
        self,
        public_key: PublicKey,
        prev_legal: Tuple[str, ...],
        new_legal: Tuple[str, ...],
    ) -> None:
        new_index = {a: i for i, a in enumerate(new_legal)}
        cols = [new_index[a] for a in prev_legal]  # old col -> new col
        width = len(new_legal)
        for table in (self.regret, self.strat_sum, self.frozen):
            for key in list(table):
                if key[0] == public_key:
                    old = table[key]
                    grown = np.zeros(width, dtype=old.dtype)
                    grown[cols] = old
                    table[key] = grown
        # Vector regime: grow the per-public-node matrices along the **last**
        # (action) axis — works for both 2-D ``(n_combos, width)`` turn/river
        # nodes and 3-D ``(n_combos, n_rivers, width)`` river-conditioned nodes.
        for table in (self.vregret, self.vstrat):
            mat = table.get(public_key)
            if mat is not None:
                grown = np.zeros(mat.shape[:-1] + (width,), dtype=mat.dtype)
                grown[..., cols] = mat
                table[public_key] = grown

    # ------------------------------------------------------------------
    # Strategy / regret operations
    # ------------------------------------------------------------------

    def sigma(self, key: Key) -> np.ndarray:
        """Pure regret-matched strategy at ``key``.

        The row *is* the legal set, so no ``valid_mask`` is needed — an unseen
        or all-non-positive row falls back to uniform over the node's actions.

        Freezing is **not** applied here: on later streets a row is keyed by an
        LUT *cluster*, which many holes share, so the bot's pinned strategy must
        only be substituted when the *sampled hole equals the bot's actual hole*
        — a condition the caller checks (it is encoded in the key only on the
        root street).  ``frozen`` is therefore consulted by the consumers
        (:mod:`mccfr`, :class:`SearchPolicy`) under that guard, not here.

        Vector regime: when ``key``'s public node has a regret **matrix**
        (``vregret``), regret-match its ``combo_index`` row.
        """
        mat = self.vregret.get(key[0])
        if mat is not None:
            if mat.ndim != 2:
                raise ValueError(
                    "sigma() read a river-conditioned vector node "
                    f"{key[0]!r} (ndim={mat.ndim}); such turn-subgame river "
                    "nodes are internal to the solve and must not be read "
                    "externally (the river is played from a fresh river subgame)."
                )
            return calculate_strategy_from_row(mat[key[1]])
        width = len(self.legal_at[key[0]])
        row = self.regret.get(key)
        if row is None:
            row = np.zeros(width, dtype=np.float64)
        return calculate_strategy_from_row(row)

    def add_regret(self, key: Key, delta: np.ndarray) -> None:
        row = self.regret.get(key)
        if row is None:
            row = np.zeros(len(delta), dtype=np.float64)
            self.regret[key] = row
        row += np.asarray(delta, dtype=np.float64)

    def add_strat(self, key: Key, sigma: np.ndarray) -> None:
        row = self.strat_sum.get(key)
        if row is None:
            row = np.zeros(len(sigma), dtype=np.float64)
            self.strat_sum[key] = row
        row += np.asarray(sigma, dtype=np.float64)

    def discount(self, factor: float) -> None:
        """Linear-CFR discount: scale every regret and strategy-sum row.

        Mirrors ``CFRTables.apply_discount`` (in-memory, no floor clamp — the
        blueprint's regret floor guards a long training run, not a short search).
        Scales the vector regime's per-node matrices too.
        """
        for table in (self.regret, self.strat_sum, self.vregret, self.vstrat):
            for row in table.values():
                row *= factor

    def average_sigma(self, key: Key) -> "np.ndarray | None":
        """Normalised cumulative strategy at ``key``; ``None`` if unaccumulated.

        Vector regime: read the ``combo_index`` row of the ``vstrat`` matrix.
        """
        mat = self.vstrat.get(key[0])
        if mat is not None:
            if mat.ndim != 2:
                raise ValueError(
                    "average_sigma() read a river-conditioned vector node "
                    f"{key[0]!r} (ndim={mat.ndim}); such turn-subgame river "
                    "nodes are internal to the solve and must not be read "
                    "externally."
                )
            row = mat[key[1]]
        else:
            row = self.strat_sum.get(key)
        if row is None:
            return None
        total = row.sum()
        if total <= 0.0:
            return None
        return (row / total).astype(np.float32)

    # ------------------------------------------------------------------
    # Instrumentation snapshot (eval doc §9.1)
    # ------------------------------------------------------------------

    def stats_snapshot(self) -> "SearchStats":
        """Read the walk / cache counters off this state into a :class:`SearchStats`.

        ``getattr(..., 0)`` guards a cache that a caller replaced with a plain
        ``dict`` (no counters) — the leaf/runout caches default to
        :class:`_CountingCache`, but the accessor stays robust either way.
        """
        return SearchStats(
            node_count=self.node_count,
            unique_pubkeys=len(self.legal_at),
            legal_at_hits=self.legal_at_hits,
            legal_at_misses=self.legal_at_misses,
            leaf_cache_hits=getattr(self.leaf_value_cache, "hits", 0),
            leaf_cache_misses=getattr(self.leaf_value_cache, "misses", 0),
            leaf_cache_size=len(self.leaf_value_cache),
            runout_cache_hits=getattr(self.runout_cache, "hits", 0),
            runout_cache_misses=getattr(self.runout_cache, "misses", 0),
            runout_cache_size=len(self.runout_cache),
        )
