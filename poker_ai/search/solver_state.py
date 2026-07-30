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

Both regimes store their tables as per-``public_key`` ``(n_rows, width)`` matrices
(``vregret``/``vstrat``): the row axis is the acting seat's combo index on the
**root street** (lossless) or its LUT cluster id on later streets, and ``width`` is
the node's legal-action count.  ``Key = (public_key, hand_row)`` still names a
single row — used by ``frozen`` and the policy readers to index one combo's row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Sequence, Tuple

import numpy as np

from poker_ai.blueprint.tree_utils import calculate_strategy_from_row

if TYPE_CHECKING:  # avoid a runtime import cycle (leaf -> policy -> solver_state)
    from poker_ai.search.leaf import LeafConfig


# A public node identifier.  ``env.public_key`` is ``(betting_stage, history)``;
# the meta-game extends it to ``(env.public_key, "META", seat)``.  Either way it
# is a hashable tuple — the alias documents intent, it is not enforced.
PublicKey = Tuple
# ``(public_key, hand_row)`` — hand_row is a combo index (root street) or an LUT
# cluster id (later streets), the row axis of the ``vregret``/``vstrat`` matrix.
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
    max_wall_seconds: float = 20.0  # per-search wall budget (the binding stop)
    # Linear-CFR discount cadence.  Kept well below the per-replica iteration count
    # of the *expensive* subgames (multiway flop ~285/replica) so the discount fires
    # several times there — at the old 100 it barely engaged on those (and never at
    # n_rollouts=8, ~50 iters/replica).  Cheap late subgames just discount more often.
    discount_interval: int = 10
    # Parallel search (§6.7 row 11).  ``None`` → resolve to a cpu-based default
    # (cpu_count-1, SLURM-aware) — the sanctioned way to spend the wall budget is W
    # independent replicas merged once, so on a 48-core node this fans out to ~47
    # replicas.  ``1`` → the serial loop (bit-for-bit identical to the pre-parallel
    # solver); ``>1`` → that many independent MCCFR replicas, merged once at the end
    # (:meth:`SolverState.accumulate`).  See :mod:`poker_ai.search.parallel`.
    workers: "int | None" = None
    # Blueprint-prior shrinkage strength (§6.6).  At read time a solved row's
    # strategy is pulled toward the blueprint by weight ``kappa / (mass + kappa)``,
    # where ``mass`` is the row's reach-weighted cumulative-strategy sum.  Genuinely
    # trained rows (mass in the hundreds–thousands) are essentially untouched; barely
    # reached off-path rows (mass ~0.01, one-sample noise) fall back almost entirely
    # to the blueprint instead of a near-uniform under-trained guess.  ``kappa`` sits
    # in the empirical bimodal gap between noise and genuine mass.  ``0.0`` disables
    # the shrinkage (pure search rows, the pre-shrinkage behaviour).
    blueprint_prior_kappa: float = 5.0
    # Structural iteration budget (§6.5) — the *primary* stop.  A real subgame is far
    # too large for any single sampled replica to reach a tight equilibrium online, so
    # there is **no online convergence test**; instead the per-subgame iteration count
    # is derived from the subgame's *structure*, which is machine-independent (unlike a
    # wall cap) and known up front.  ``poker_ai.search.budget.iteration_budget`` reads
    # these; ``max_iterations`` is the absolute safety ceiling and ``max_wall_seconds``
    # a loose backstop.  **Off by default** so a flat ``max_iterations`` is honoured
    # verbatim (every existing test / pinned digest that sets an explicit iteration
    # count is unchanged); production turns it on in ``build_blueprint_session``.
    auto_budget: bool = True
    # Vector regime (heads-up flop/turn/river): **full-width**, so iterations-to-
    # converge is driven by tree *depth* (streets left to resolve), not the infoset
    # count — a per-stage constant ``(flop, turn, river)``.  The 200-buckets/street
    # infoset counts (~746k flop / ~89k turn / ~26k river) only bound the per-iteration
    # wall (flop ≈ 1500 it × 746k rows ≈ tens of s/replica — the paper's 1–33 s).
    vector_budget_by_street: tuple = (1500, 1000, 500)  # (flop, turn, river)
    # MCCFR regime (multiway, or heads-up pre-flop): **sampled**, and only the HOT PATH
    # needs to converge — rarely-reached infosets fall back to the blueprint via the
    # ``blueprint_prior_kappa`` shrinkage — so the budget is a **GLOBAL** (pooled-over-
    # all-W-replicas) iteration count, split among the replicas: per-replica =
    # ``ceil(global / workers)``.  More workers ⇒ **shorter wall at ~constant total
    # work**, because the merged average pools every replica's samples so what matters
    # is bounding the *total* sampled work — unlike the full-width vector regime, whose
    # replicas each need the whole per-replica learning horizon (so vector stays a
    # per-replica constant, not divided).  The global budget grows ~linearly with the
    # live-player count (bigger hot path); clamped to ``[min, max]``.
    #
    # Per-street (preflop, flop, turn, river) base pooled work **per live player** —
    # ``global = base[street] * n_live``, clamped to ``[min, max]`` then split across
    # the W replicas.  Per-street because a deep multiway flop needs far more sampled
    # work to cover its hot path than a river.  Indexed by ``street_at_root``
    # (0=preflop … 3=river).  Defaults are the old flat 3000 for every street (an
    # un-tuned starting point — recalibrate per street on the multiway blueprint via
    # the calibration harness).
    mccfr_global_per_player_by_street: tuple = (3000, 3000, 3000, 3000)
    mccfr_global_min: int = 6000
    mccfr_global_max: int = 30000
    # Per-replica **learning floor**, per street: every replica runs at least this many
    # iterations so it learns properly even when the global budget divided by a large
    # ``workers`` would otherwise starve it (an under-learned replica pollutes the
    # merged average).  On the 64-core target this floor is what actually **binds** —
    # ``global / 63`` falls below it for every street and live-count — so this tuple is
    # the primary per-street budget dial *at cluster scale* (the per-street global above
    # governs low-W runs and the n_live scaling).  Indexed by ``street_at_root``.
    mccfr_min_per_replica_by_street: tuple = (750, 750, 750, 750)
    # OX-Search safety parameter β (Approach B, PO-CES-HU; Ge et al. ICML 2024,
    # Thm 4.6: ``exp(σ₂ˢ) − exp(σ) ≤ Δ/β``).  ``None`` → OX-Search is OFF and the
    # vector solve is byte-for-byte the vanilla/DBR path (no gadget root, no opt-out
    # row, no ``CBV_ref`` pass).  A finite β turns on the gadget root
    # (:mod:`poker_ai.search.vector`): smaller β ⇒ more exploitation of the belief
    # ``p̂``; larger β ⇒ safer (β is an upper bound; OX auto-balances).  Only the
    # heads-up turn/river **vector** regime consumes it; the MCCFR regime ignores it.
    beta: "float | None" = None


class _CountingCache:
    """A dict-backed memo that counts ``get`` hits and misses (eval doc §6, §9.1).

    Wraps the search-lifetime ``leaf_value_cache`` so its hit/miss rate can be
    logged without threading counters through every call site.  Only the
    operations the cache (and the tests) actually use are implemented —
    ``get``/``[]``/``in``/``len``/iteration — so a plain ``dict`` stays a valid
    drop-in wherever counting is not wanted (leaf's standalone per-call memo when
    no shared cache is supplied).  ``__slots__`` keeps it deepcopy-/pickle-friendly
    for the parallel replicas (§6.7).
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
    ``cache_misses`` roll both caches together for the single schema columns of
    the same name; the per-cache fields stay available for finer analysis.
    """

    node_count: int = 0           # decision-node visits over the whole search
    unique_pubkeys: int = 0       # distinct public keys (== len(legal_at))
    legal_at_hits: int = 0        # node revisits (legal set already registered)
    legal_at_misses: int = 0      # first registrations (== unique_pubkeys)
    leaf_cache_hits: int = 0
    leaf_cache_misses: int = 0
    leaf_cache_size: int = 0

    @property
    def cache_hits(self) -> int:
        """Both caches' hits, for the ``decisions.cache_hits`` column."""
        return self.legal_at_hits + self.leaf_cache_hits

    @property
    def cache_misses(self) -> int:
        """Both caches' misses, for the ``decisions.cache_misses`` column."""
        return self.legal_at_misses + self.leaf_cache_misses

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
        )
        return merged


@dataclass
class SolverState:
    """In-memory CFR tables for one search (warm-start carrier, §6.5).

    Both regimes store the traverser's regret/strategy in the per-public-node
    ``vregret``/``vstrat`` matrices (below); ``frozen`` pins a probability vector
    for the bot's actual-hand rows (§5).  All are mutated in place across iterations.
    """

    legal_at: Dict[PublicKey, Tuple[str, ...]] = field(default_factory=dict)
    actor_at: Dict[PublicKey, int] = field(default_factory=dict)
    frozen: Dict[Key, np.ndarray] = field(default_factory=dict)
    # Per-public-node ``(n_rows, width)`` matrices — the shared native storage for
    # BOTH regimes (the vector regime and the traverser-vectorized MCCFR walk): one
    # vectorised op per node rather than ~n_rows dict rows.  On the root street the
    # first axis is the lossless ``combo_index`` (externally readable by
    # :class:`SearchPolicy`); on a future street it is the LUT-cluster index.
    vregret: Dict[PublicKey, np.ndarray] = field(default_factory=dict)
    vstrat: Dict[PublicKey, np.ndarray] = field(default_factory=dict)
    # Row space of each vector node: ``"combo"`` (root street — lossless, one row
    # per ``combo_index``, externally readable by :class:`SearchPolicy`) or
    # ``"cluster"`` (a future street — one row per LUT cluster reachable in the
    # subgame, §6.5 lossy abstraction; internal to the solve, never read
    # externally).  The read guards in :meth:`sigma` / :meth:`average_sigma`
    # consult this so a cluster node cannot be mis-read as if keyed by combo.
    vrow_space: Dict[PublicKey, str] = field(default_factory=dict)
    # Search-lifetime leaf cache (§6.4.2, §6.7 Tier 1).  Holds values that are
    # **invariant across CFR iterations** for a fixed key, so it is *not* touched
    # by ``discount`` and persists across a warm-started re-search:
    # ``continuation_value_vector`` keyed by ``(leaf public_key, other-seat holes,
    # profile)`` — one ``(n_combos,)`` vector per key spans every traverser combo
    # (§6.4.1).
    leaf_value_cache: _CountingCache = field(default_factory=_CountingCache)
    # Opponent-model rows for the solver clamp (opponent_modeling §5.3), keyed by
    # ``(seat, public_key)`` → ``(σ̂ (n_combos, width), c (n_combos, 1))``, aligned to
    # the node's legal set at insert time.  The model is frozen for the search's
    # lifetime, so warm-started re-searches within a hand reuse it; the cache dies
    # with the ``SolverState`` at hand end.  **Untouched when ``ctx.models`` is empty**
    # — the clamp early-outs before any lookup, so an unmodeled solve neither reads
    # nor writes it (keeping the baseline bit-for-bit and its counters at zero).
    model_sigma_cache: _CountingCache = field(default_factory=_CountingCache)
    # Walk instrumentation (eval doc §9.1) — cumulative over the search; the
    # legal-action cache (``legal_at``) is a plain dict, so its hit/miss and the
    # node-visit tally are counted explicitly in :meth:`ensure_node` rather than by
    # a ``_CountingCache`` wrapper (``legal_at`` is also read incidentally by the
    # policy reader and strategy ops, which must not count as node visits).
    node_count: int = 0
    legal_at_hits: int = 0
    legal_at_misses: int = 0
    # Root-value convergence signal (calibration; eval doc §9).  A linearly-weighted
    # running estimate of the hero's root counterfactual value for its ACTUAL hand,
    # normalised to read as the played hand's conditional EV vs the belief opponent
    # (chips) — accumulated by whichever regime runs.  ``num``/``den`` are summed
    # across parallel replicas in :meth:`accumulate` and read as ``num/den`` by
    # ``solve`` (``None`` when ``den == 0``: no played-combo value was tracked).
    # These are **write-only side counters** — the regimes only ever add to them off a
    # value the walk already computed, touching no table and no RNG stream, so a solve
    # is byte-for-byte identical whether or not the value is tracked.
    root_value_num: float = 0.0
    root_value_den: float = 0.0

    @classmethod
    def empty(cls) -> "SolverState":
        return cls()

    def reset_counters(self) -> None:
        """Zero the per-search walk / cache tallies, preserving tables + caches.

        The instrumentation counters (``node_count``, ``legal_at`` hits/misses, and
        the leaf cache's hit/miss tally) are cumulative over a state's
        lifetime.  A **warm-started re-search** reuses the state for a *new* search
        invocation whose ``decisions`` row (eval doc §6, §9.1) must report only that
        invocation's work — matching ``iterations_run``, which ``run_loop`` counts
        fresh per ``solve()`` call.  Without this reset a re-search's snapshot would
        report its own work *plus* every prior solve on the same state (serial), or
        the warm baseline's counters multiplied by the replica count (parallel).

        Only the tallies reset: ``legal_at`` / ``vregret`` / ``vstrat`` / ``frozen``
        and the cache **contents** are preserved, so a value cached by the prior
        solve correctly scores as a *hit* for the re-search.  ``unique_pubkeys`` is
        read from ``len(legal_at)`` (the whole widened tree), so it is unaffected.
        """
        self.node_count = 0
        self.legal_at_hits = 0
        self.legal_at_misses = 0
        self.root_value_num = 0.0
        self.root_value_den = 0.0
        if hasattr(self.leaf_value_cache, "hits"):
            self.leaf_value_cache.hits = 0
            self.leaf_value_cache.misses = 0

    @classmethod
    def accumulate(
        cls,
        states: Sequence["SolverState"],
        *,
        baseline: "SolverState | None" = None,
    ) -> "SolverState":
        """Fold independent-replica tables into one merged state (§6.7 row 11).

        Each parallel replica runs the full traverser rotation over its own
        seeded substream and returns its :class:`SolverState`; this folds them into
        a single state by **summing** the cumulative ``vregret`` / ``vstrat``
        matrices over the union of public keys.  Rows for a given key share a shape
        because the legal set is a deterministic function of ``public_key`` and every
        replica inherits the same warm-start widening, so the per-key arrays add
        directly.  ``average_sigma`` normalises the summed ``vstrat`` per node, so the
        merged average policy is a valid reach-weighted average over all
        ``W × iterations``.

        ``baseline`` is the warm-start state every replica was seeded from (``None``
        for a fresh search).  When given it is added **exactly once** and each
        replica contributes only its delta over it (``replica − baseline``) —
        otherwise the shared warm-start regrets would be counted ``W`` times.  Its
        ``frozen`` rows (pinned actual hands) and node structure carry through.  The
        per-search ``leaf_value_cache`` is dropped (recomputed on demand).

        Note: a replica also discounts the baseline rows along with its own
        accumulation, so ``replica − baseline`` is only approximately the replica's
        fresh contribution under Linear-CFR; the fresh-search path (no baseline) is
        exact, and a warm re-search is a short refinement where the drift is small.
        """
        _TABLES = ("vregret", "vstrat")
        out = cls()
        if baseline is not None:
            for pk, legal in baseline.legal_at.items():
                out.legal_at[pk] = legal
            for pk, actor in baseline.actor_at.items():
                out.actor_at[pk] = actor
            out.vrow_space.update(baseline.vrow_space)
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
            for pk, rs in st.vrow_space.items():
                out.vrow_space.setdefault(pk, rs)
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
        # Root-value estimate: pool the replicas' linearly-weighted sums (summing
        # num + den merges their per-iteration estimates into one pooled linear
        # average).  The baseline is a *prior* search's estimate, so it is NOT added —
        # each replica already reset these to zero and accrues only this search's work.
        for st in states:
            out.root_value_num += st.root_value_num
            out.root_value_den += st.root_value_den
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
        n_rows: int,
        row_space: str,
    ) -> None:
        """Register a public node and lazily allocate its vector-regime matrices.

        Like :meth:`ensure_node` (it delegates the ``legal_at``/``actor_at``
        bookkeeping and warm-start widening — which now also grows the matrix
        **columns**), but additionally allocates the ``vregret``/``vstrat``
        matrices ``(n_rows, width)`` for ``public_key`` on first visit.

        The first (``n_rows``) axis is the node's **row space** (§6.5):

        - ``row_space == "combo"`` — a **root-street** node, ``n_rows == n_combos``,
          one lossless row per ``combo_index`` (the rows :class:`SearchPolicy`
          reads for the bot's actual hand);
        - ``row_space == "cluster"`` — a **future-street** node, ``n_rows`` = the
          number of LUT clusters reachable in this subgame, one row per cluster
          (lossy abstraction; the sampled board is folded into the cluster id, so
          no explicit river axis is needed).

        The action ``width`` is always the **last** axis, so the table ops
        (regret matching, widening) are axis-agnostic.
        """
        self.ensure_node(public_key, legal_actions, actor)
        if public_key not in self.vregret:
            width = len(self.legal_at[public_key])
            self.vregret[public_key] = np.zeros((n_rows, width), dtype=np.float64)
            self.vstrat[public_key] = np.zeros((n_rows, width), dtype=np.float64)
            self.vrow_space[public_key] = row_space

    def _grow_rows(
        self,
        public_key: PublicKey,
        prev_legal: Tuple[str, ...],
        new_legal: Tuple[str, ...],
    ) -> None:
        new_index = {a: i for i, a in enumerate(new_legal)}
        cols = [new_index[a] for a in prev_legal]  # old col -> new col
        width = len(new_legal)
        for key in list(self.frozen):
            if key[0] == public_key:
                old = self.frozen[key]
                grown = np.zeros(width, dtype=old.dtype)
                grown[cols] = old
                self.frozen[key] = grown
        # Vector regime: grow the per-public-node matrices along the **last**
        # (action) axis.  Both row spaces are 2-D ``(n_rows, width)``, so the
        # leading-axis-preserving grow below is uniform.
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
        (``vregret``), regret-match its ``combo_index`` row — but only for a
        **combo**-keyed (root-street) node.  A cluster-keyed future-street node is
        internal to the solve (its rows are LUT clusters, not combos, and the
        next round is played from a fresh subgame) and must not be read here.
        """
        mat = self.vregret.get(key[0])
        if mat is not None:
            if self.vrow_space.get(key[0]) == "cluster":
                raise ValueError(
                    "sigma() read a cluster-keyed vector node "
                    f"{key[0]!r}; such future-street nodes are internal to the "
                    "solve (rows are LUT clusters, not combos) and must not be "
                    "read externally (the next round is a fresh subgame)."
                )
            return calculate_strategy_from_row(mat[key[1]])
        # No matrix for this node → uniform over its legal set.  Every live decision
        # node is allocated via ``ensure_vnode``, so this fallback only fires for an
        # out-of-tree read (an unseen node the caller still queried).
        width = len(self.legal_at[key[0]])
        return calculate_strategy_from_row(np.zeros(width, dtype=np.float64))

    def discount(self, factor: float) -> None:
        """Linear-CFR discount: scale every ``vregret`` / ``vstrat`` matrix.

        Mirrors ``CFRTables.apply_discount`` (in-memory, no floor clamp — the
        blueprint's regret floor guards a long training run, not a short search).
        """
        for table in (self.vregret, self.vstrat):
            for mat in table.values():
                mat *= factor

    def average_sigma(self, key: Key) -> "np.ndarray | None":
        """Normalised cumulative strategy at ``key``; ``None`` if unaccumulated.

        Vector regime: read the ``combo_index`` row of the ``vstrat`` matrix — a
        combo-keyed (root-street) node only; a cluster-keyed future-street node is
        internal to the solve and must not be read externally.
        """
        mat = self.vstrat.get(key[0])
        if mat is None:
            return None
        if self.vrow_space.get(key[0]) == "cluster":
            raise ValueError(
                "average_sigma() read a cluster-keyed vector node "
                f"{key[0]!r}; such future-street nodes are internal to the "
                "solve and must not be read externally."
            )
        row = mat[key[1]]
        total = row.sum()
        if total <= 0.0:
            return None
        return (row / total).astype(np.float32)

    def mass(self, key: Key) -> float:
        """Reach-weighted cumulative-strategy mass at ``key`` (0.0 if unaccumulated).

        This is the un-normalised denominator of :meth:`average_sigma` — the
        row's ``vstrat`` combo-row sum.  It is
        the search's *confidence* at this infoset: high on the on-path rows it
        trained full-width, near-zero on barely-reached off-path rows.  The read
        seam (:class:`SearchPolicy` / :class:`~poker_ai.search.agent.SearchAgent`)
        uses it to shrink the strategy toward the blueprint.  Combo-keyed
        (root-street) reads only, mirroring :meth:`average_sigma`; a cluster-keyed
        future-street node is internal to the solve and never read here.
        """
        mat = self.vstrat.get(key[0])
        if mat is None or self.vrow_space.get(key[0]) == "cluster":
            return 0.0
        return float(mat[key[1]].sum())

    # ------------------------------------------------------------------
    # Instrumentation snapshot (eval doc §9.1)
    # ------------------------------------------------------------------

    def stats_snapshot(self) -> "SearchStats":
        """Read the walk / cache counters off this state into a :class:`SearchStats`.

        ``getattr(..., 0)`` guards a cache that a caller replaced with a plain
        ``dict`` (no counters) — ``leaf_value_cache`` defaults to
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
        )
