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
from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, Mapping, Sequence, Tuple

import numpy as np

from poker_ai.blueprint.tree_utils import calculate_strategy_from_row

if TYPE_CHECKING:  # avoid a runtime import cycle (leaf -> policy -> solver_state)
    from poker_ai.search.leaf import LeafConfig


# The three search approaches, as the budget tables key them.  Inferred from the solve's
# own inputs (``poker_ai.search.budget.search_approach``), never passed in: DBR is the
# approach that carries opponent models, OX-Search the one that sets ``beta``, and the two
# are mutually exclusive by construction (``evaluation.runner.for_condition`` enforces it).
VANILLA = "vanilla"
DBR = "dbr"
OX = "ox"
APPROACHES = (VANILLA, DBR, OX)

# ---------------------------------------------------------------------------- #
# Per-cell iteration budgets — EXPLICIT, one number per (approach, street, n_live)
# ---------------------------------------------------------------------------- #
# There is deliberately NO formula here: no per-live-player base multiplied out, no
# per-approach scale factor.  Those were convenient but dishonest — they implied a
# structure the measurements do not have (DBR's cost is not a fixed multiple of
# vanilla's; it is ~2x slower per iteration on the flop and ~2.1x on the turn, and its
# convergence point differs per street), and they made a change for one wall-bound cell
# silently move every other cell that shared the base.  One number per cell means each
# is independently traceable to what it came from, and editing one edits exactly one.
#
# DERIVATION (2026-08 v7 calibration, single worker):
#     budget = min(v7 convergence suggestion, iterations that fit 630 s at the measured
#                  throughput of THAT approach on THAT cell), rounded to 50
# 630 s leaves ~5% headroom under the ~660 s reference (Pluribus's 30 s x 22-core top =
# 660 core-seconds/search; 4p < 6p).  Cells whose convergence point sits under the wall
# take the convergence point; cells where the wall bites first are WALL-CLIPPED and
# marked below — for those the number is what fits, not what converges.
#
# OX-Search is SPLIT across the two tables, because the gadget only exists in one regime:
#   - VECTOR (its real home, HU turn/river): DBR's numbers.  The gadget root is live there
#     and costs about what DBR's modelled solve does, so DBR's budget is the right size.
#   - MCCFR: VANILLA's numbers.  ``beta`` is inert outside the vector regime — solver.py
#     warns that such a solve is "an ordinary best response, NOT adaptation-safe" — so an
#     OX-labelled MCCFR solve IS a vanilla solve and should be budgeted as one.  Giving it
#     DBR's number would spend DBR's wall on a search doing none of DBR's work.
# Kept as explicit tables rather than aliases so any of the three can diverge later.

# (street, n_live) -> per-replica iterations.  street: 0=preflop 1=flop 2=turn 3=river.
_MCCFR_VANILLA = {
    # Preflop is played from the BLUEPRINT in production and never searched, so these are
    # unmeasured carry-forwards, kept only so a forced preflop solve has a sane number.
    (0, 2): 6000, (0, 3): 9000, (0, 4): 12000,
    (1, 2): 36250,   # WALL-CLIPPED (630 s @ 57.5 it/s); v7 wanted 36436 — essentially at it
    (1, 3): 41500,   # v7 convergence point (287 s) — wall not binding
    (1, 4): 47250,   # EXTRAPOLATED from n3 (see LIVE-COUNT EXTRAPOLATION below), 327 s
    (2, 3): 59450,   # WALL-CLIPPED (630 s @ 94.4 it/s); v7 wanted 69627
    (2, 4): 59450,   # EXTRAPOLATED, then WALL-CLIPPED — n4 wants 67672, 630 s allows 59441
    (3, 3): 8150,    # v7 convergence point (19 s) — wall nowhere near binding
    (3, 4): 9300,    # EXTRAPOLATED from n3 (22 s)
}
_MCCFR_DBR = {
    (0, 2): 6000, (0, 3): 9000, (0, 4): 12000,   # unmeasured ⇒ same as vanilla
    (1, 2): 29800,   # WALL-CLIPPED (630 s @ 47.3 it/s) — cannot afford vanilla's 36250
    (1, 3): 27250,   # v7 CEILING, not a measurement: this cell converged BELOW the ladder
    (1, 4): 31000,   # EXTRAPOLATED from n3 (242 s).  n3 was itself a below-ladder CEILING,
                     # so this inherits that — an upper bound propagated, not a measurement
    (2, 3): 28500,   # WALL-CLIPPED (630 s @ 45.2 it/s); v7 wanted 56441
    (2, 4): 28500,   # EXTRAPOLATED, then WALL-CLIPPED — n4 wants 32441, 630 s allows 28481
    (3, 3): 15350,   # v7 convergence point (39 s) — DBR genuinely needs ~1.9x vanilla here
    (3, 4): 17450,   # EXTRAPOLATED from n3 (45 s)
}
# The vector regime only runs heads-up (n_live == 2); the flop entry is oracle/A-B only,
# since a real HU flop routes to MCCFR.
_VECTOR_VANILLA = {
    (1, 2): 1500,    # oracle / calibration A-B only — never routed in production
    (2, 2): 2550,    # v7 convergence point (328 s @ 7.8 it/s)
    (3, 2): 600,     # v7: converged BELOW the whole ladder, and EXACT here (no chance node
                     # left; measured cross-seed spread 0.0), so this is a real bound (3 s)
}
_VECTOR_DBR = {
    (1, 2): 1500,
    (2, 2): 1700,    # WALL-CLIPPED (630 s @ 3.1 it/s) — DBR cannot afford vanilla's 2550
    (3, 2): 600,     # exact, same as vanilla
}

# LIVE-COUNT EXTRAPOLATION (the ``n_live == 4`` entries).  v7 ran n_live=2,3 only, so the
# 4-player cells are DERIVED, not measured:
#
#     budget(n4) = min( budget(n3) * 1.138 ,  630 s * throughput(n3) )
#
# The 1.138 is the one live-count trend v7 actually resolved — vanilla flop, 36436 (n2) ->
# 41475 (n3).  It is a SINGLE pair of points on ONE street: the turn and river ran at n3
# only, and DBR's flop n3 landed below its ladder (a ceiling, not a convergence point), so
# neither yields a second estimate.  Treat 1.138 as the best available reading of the
# pattern, not as an established growth law.
#
# Note how much flatter that is than the ``base * n_live`` model these tables replaced,
# which grew the budget by 1.333 from n3 to n4: the hot path widens with the live count far
# more slowly than per-player scaling assumed.  That is why the n4 flop/river budgets move
# so much here — the old carry-forwards were shaped by the steeper law, not by evidence.
#
# Throughput at n4 is ASSUMED EQUAL to n3 rather than extrapolated.  v7 measured throughput
# RISING with the live count (flop 57.5 -> 144.4 it/s from n2 to n3), so holding it flat is
# the conservative direction: if n4 is in fact faster, these budgets merely leave wall
# unused; had the rise been extrapolated and been wrong, the solves would overrun the wall
# backstop and truncate silently.  Both turn cells are wall-clipped under that assumption
# and would grow if a real n4 throughput measurement came in higher.

# ⚠ CROSS-ARM COMPARABILITY: on the wall-bound cells DBR runs FEWER iterations than
# vanilla (flop n2 29800 vs 36250; turn n2 vector 1700 vs 2550; turn n3 28500 vs 59450)
# purely because it is slower per iteration.  An evaluation that compares the two is
# therefore NOT budget-controlled on those cells, and a DBR loss there is confounded with
# the shorter search.  Equalising would mean cutting vanilla to DBR's number — a real
# option, deliberately not taken here since it would weaken the paper baseline.
MCCFR_BUDGET: Mapping[str, Mapping[Tuple[int, int], int]] = MappingProxyType({
    VANILLA: MappingProxyType(dict(_MCCFR_VANILLA)),
    DBR: MappingProxyType(dict(_MCCFR_DBR)),
    # OX in the MCCFR regime has no gadget ⇒ it is a vanilla solve (see above).
    OX: MappingProxyType(dict(_MCCFR_VANILLA)),
})
VECTOR_BUDGET: Mapping[str, Mapping[Tuple[int, int], int]] = MappingProxyType({
    VANILLA: MappingProxyType(dict(_VECTOR_VANILLA)),
    DBR: MappingProxyType(dict(_VECTOR_DBR)),
    # OX's real home: the gadget is live here, so it is sized like DBR (see above).
    OX: MappingProxyType(dict(_VECTOR_DBR)),
})


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

    ``leaf`` is required and carries the continuation-strategy fleet consumed
    at depth-limit leaves.
    """

    leaf: "LeafConfig"
    # ``max_iterations`` is the ABSOLUTE per-replica hard ceiling and the SINGLE upper
    # bound on the structural budget (:mod:`poker_ai.search.budget`) — the primary stop
    # is the structural budget itself, and this only clips it in pathology (e.g. a deep
    # multiway flop that would otherwise request more).  Set below where MCCFR converges
    # it silently throttles every HU-flop / multiway search — the pre-2026-08 bug, where
    # a 5_000 cap capped them regardless of the per-street numbers below.  It doubles as
    # the pinned iteration count when ``auto_budget`` is off (tests/digests).
    max_iterations: int = 60_000  # the one ceiling; a true bound, not a throttle (raised
                                  # 2026-08 from 30k so the DBR deep-cell scale below can
                                  # actually reach depth on HU flop / multiway; 4p vanilla
                                  # is unaffected — its budgets are all ≤24k)
    # LOOSE per-search wall backstop, NOT the primary stop: a single flat cap sized
    # above the DEEPEST shipped search's wall cost so it does not clip the iteration
    # budget in normal operation — it only catches a genuinely stuck subgame.  Flat (not
    # per-street): the 2026-08 calibration measured only the HU vector turn/river, whose
    # tiny river cap would butcher multiway river MCCFR; the eval path sizes this to the
    # multiway-flop worst case (~1000 s, see evaluation/runner.py).
    max_wall_seconds: float = 300.0
    # Linear-CFR discount cadence.  Each firing scales EVERY vregret/vstrat matrix
    # (O(table)), so under the 2026-08 structural budgets (10k-40k iters/solve) the old
    # cadence of 10 fired 1000-4000 times per solve — a nontrivial runtime slice for a
    # weighting profile that barely differs at these depths.  100 keeps the Linear-CFR
    # weighting (piecewise k/(k+1), k = t/interval) with 10-400 firings per solve; even
    # the smallest structural budget (vector river 850) still discounts 8 times.
    # ⚠ Results-affecting (different discount points → different tables): golden digests
    # regenerated for this change; the equilibrium oracle is the correctness gate.
    discount_interval: int = 100
    # Structural iteration budget (§6.5) — the *primary* stop.  A real subgame is far
    # too large for any single sampled replica to reach a tight equilibrium online, so
    # there is **no online convergence test**; instead the per-subgame iteration count
    # is derived from the subgame's *structure*, which is machine-independent (unlike a
    # wall cap) and known up front.  ``poker_ai.search.budget.iteration_budget`` reads
    # these; ``max_iterations`` is the absolute safety ceiling and ``max_wall_seconds``
    # a loose backstop.  **On by default** so every production solve (search + eval) is
    # driven by the structural budget without the caller having to opt in.  The tests /
    # pinned digests that need a fixed iteration count set ``auto_budget=False``
    # explicitly, and then a flat ``max_iterations`` is honoured verbatim.
    auto_budget: bool = True
    # Per-cell iteration budgets — EXPLICIT tables, one number per
    # ``(approach, street, n_live)``; see MCCFR_BUDGET / VECTOR_BUDGET at module level for
    # the numbers and their derivation.  There is no formula and no multiplier: the budget
    # is looked up, not computed.  ``poker_ai.search.budget.iteration_budget`` reads these,
    # picking the table by regime and the row by the approach it infers from the solve's
    # own inputs (models ⇒ DBR, ``beta`` ⇒ OX, neither ⇒ vanilla).
    #
    # Per-replica, and production runs one replica per hand (``workers=1``, one hand per
    # core), so these ARE the per-search numbers.  With ``workers > 1`` the MCCFR budget is
    # divided across replicas (more cores ⇒ shorter wall at ~constant work); the vector
    # budget is not divisible — each full-width replica needs the whole horizon.
    #
    # Overridable per solve, so a caller can pin a cell without editing the module tables.
    mccfr_budget: "Mapping[str, Mapping[Tuple[int, int], int]]" = MCCFR_BUDGET
    vector_budget: "Mapping[str, Mapping[Tuple[int, int], int]]" = VECTOR_BUDGET
    # OX-Search safety parameter β (Approach B, PO-CES-HU; Ge et al. ICML 2024,
    # Thm 4.6: ``exp(σ₂ˢ) − exp(σ) ≤ Δ/β``).  ``None`` → OX-Search is OFF and the
    # vector solve is byte-for-byte the vanilla/DBR path (no gadget root, no opt-out
    # row, no ``CBV_ref`` pass).  A finite β turns on the gadget root
    # (:mod:`poker_ai.search.vector`): smaller β ⇒ more exploitation of the belief
    # ``p̂``; larger β ⇒ safer (β is an upper bound; OX auto-balances).  Only the
    # heads-up turn/river **vector** regime consumes it; the MCCFR regime ignores it.
    beta: "float | None" = None
    # VR-MCCFR variance reduction (opponent_modeling §5.5) — a control-variate baseline
    # on the sampled opponent-action counterfactual values in the MCCFR walk.  DBR-only:
    # it activates ONLY when this flag is set AND the subgame carries opponent models
    # (``ctx.models``), so the vanilla/paper baseline is byte-for-byte untouched (no
    # models ⇒ inert regardless of the flag).  Unbiased (same best response, less
    # estimator variance).  ``vr_baseline_decay`` is the EMA weight on each new sample
    # when updating the per-node baseline (0 ⇒ frozen at init, 1 ⇒ last-sample only).
    variance_reduction: bool = False
    vr_baseline_decay: float = 0.5


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
