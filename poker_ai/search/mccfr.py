"""Traverser-vectorized external-sampling Linear MCCFR regime (§6.5).

The MCCFR regime is the *large / early* path — round 1, all of round 2, and any
large multiway later subgame.  The traverser's private hand is solved in **vector
form**: one walk updates every combo's row at once (killing the per-row
starvation the scalar external-sampling walk suffered), while opponents still
**sample one** action from their single sampled hole's regret-matched row (keeping
multiway tractable) and chance is the engine's frozen per-iteration board runout.
Regret and average strategy fold into a single pass — ``pi_p`` is exact because the
traverser's actions are expanded — writing the shared ``vregret``/``vstrat``
matrices (keyed by ``public_key``; combo rows on the root street, LUT-cluster rows
on future streets) that the vector regime and :class:`SearchPolicy` also read.

Depth-limit leaves are the §6.4 continuation meta-game, valued per-combo by
:func:`poker_ai.search.leaf.continuation_value_vector`; terminals settle against
the concrete sampled opponents via :meth:`PokerEnv.vector_payout_concrete`.  This
mirrors :meth:`poker_ai.search.vector._VectorSolver._walk`, specialized to sampled
(rather than range-expanded) opponents.  CFR-P pruning is omitted (it never
engages in a short search, §6.5).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

from environment import range_showdown
from environment.utils import make_deck_arr
from information_abstraction.lookup import clusters_for_board
from poker_ai.blueprint.tree_utils import sample_index
from poker_ai.search.cluster_maps import _STREET_NAME, ClusterMapper
from poker_ai.search.context import SubgameContext
from poker_ai.search.fast_env import build_fast_mccfr_env
from poker_ai.search.leaf import continuation_value_vector
from poker_ai.search.policy import BiasClass
from poker_ai.search.solver_state import SolverConfig, SolverState
from poker_ai.search.vform import (
    apply_model_clamp,
    freeze_combo,
    node_sigma,
    regret_match_matrix,
    traverser_update,
    vr_baseline_estimate,
)

# Search Cython core (Phase 2): when built AND ``PLURIBUS_SEARCH_CORE=1``, the
# depth-limit leaf rollout runs on the compiled ``FastState`` engine, settling each
# terminal per-combo via ``FastState.vector_payout_concrete``.  Rebinds the module
# global the walk resolves at call time (``_vleaf_value``), so the swap is
# transparent; falls back to the pure-Python ``continuation_value_vector`` when the
# core is unavailable or the flag is off.  Equilibrium-gated (not byte-identical):
# the rollout board draw consumes ``ctx.rng`` differently from the env shuffle.
try:
    from poker_ai._core import CORE_AVAILABLE as _CORE_AVAILABLE
    from poker_ai._core.flags import search_core_enabled as _search_core_enabled

    if _CORE_AVAILABLE and _search_core_enabled():
        from poker_ai.search.leaf_fast import (
            continuation_value_vector_fast as continuation_value_vector,
        )
except ImportError:
    pass


def _leaf_fn(frontier_env):
    """The leaf rollout to use for ``frontier_env`` — the walk engine decides.

    The module-global binding above is resolved at **import** time, but whether the
    walk actually runs on the core is decided per solver (``_use_core``) and per
    iteration (``build_fast_mccfr_env`` may return ``None``).  When the flag is set
    after this module is imported, those disagree: the walk hands the leaf a
    ``FastMCCFRAdapter`` while the global still points at the pure-Python rollout,
    which reaches for ``PokerEnv``-only members (``with_hole_cards``) and raises —
    swallowed by ``SearchAgent._solve_and_store`` into a silent blueprint fallback.

    So: a compiled frontier always gets the compiled leaf.  Deliberately one-way —
    never downgrade a ``PokerEnv`` frontier to the Python rollout when the global
    selected the fast one, because the two consume ``ctx.rng`` differently and that
    would move ``GOLDEN_DIGEST_MCCFR``.
    """
    if getattr(frontier_env, "_fast", None) is not None:
        from poker_ai.search.leaf_fast import continuation_value_vector_fast
        return continuation_value_vector_fast
    return continuation_value_vector


logger = logging.getLogger(__name__)

# The four §4 continuation strategies, in the canonical meta-action order.
_BIAS_CLASSES: Tuple[BiasClass, ...] = ("none", "fold", "call", "raise")


class _MCCFRSolver:
    """One external-sampling Linear MCCFR search over a fixed subgame root.

    Holds the shared :class:`SolverState` plus the immutable search inputs; one
    :meth:`iterate` call is one traversal.  The orchestrator (:func:`solve`)
    drives the iteration count, the Linear-CFR discount, and result extraction.
    """

    def __init__(self, root_env, state: SolverState, ctx: SubgameContext,
                 cfg: SolverConfig, rng: np.random.Generator) -> None:
        self.root_env = root_env
        self.state = state
        self.ctx = ctx
        self.cfg = cfg
        self.rng = rng
        self.n_players = root_env.n_players
        self._combo_cards = root_env.combo_cards
        self._my_hole = tuple(sorted(int(c) for c in ctx.my_hole))

        # Densified card-removal slots (once per solver, not per node/iteration):
        # card ints are Cactus-Kev encoded (not ``0..51``), so a dense boolean
        # exclusion mask needs each combo's two cards mapped to a compact
        # ``0..deck_size-1`` slot first — the same densification
        # ``range_showdown.removal_index`` uses for showdown settlement, reused
        # here so root-hole sampling's card-removal checks (`np.isin` against a
        # tiny per-call exclusion set, confirmed hot by profiling) become a
        # dense-mask lookup instead. ``removal_for``/``deck_slots`` are
        # ``lru_cache``d per deck, so this is cheap even recomputed elsewhere.
        low, high = root_env.low_card_rank, root_env.high_card_rank
        self._combo_slot0, self._combo_slot1, self._deck_size = (
            range_showdown.removal_for(low, high)
        )
        self._deck_uniq = range_showdown.deck_slots(low, high)
        # Fixed per-iteration payout-feasibility mask (recomputed at the top of
        # every ``_vectorized_iterate`` call; ``None`` iff the current root is
        # leaf-containing, where the board deals progressively and no single
        # fixed mask is valid for a whole iteration — see ``_vectorized_iterate``).
        self._iter_feasible = None

        self._live_seats = sorted(ctx.ranges.keys())
        all_seats = set(ctx.ranges) | set(ctx.folded_ranges)
        if all_seats != set(range(self.n_players)):
            # with_hole_cards replaces *every* seat, so the joint draw must cover
            # all of them; live + folded is the full dealt field by construction.
            raise ValueError(
                f"MCCFR root: live+folded seats {sorted(all_seats)} must cover "
                f"every seat 0..{self.n_players - 1}."
            )
        self._all_seats = sorted(all_seats)
        # Per-seat board-masked, normalised belief weights for the joint draw.
        bc = np.asarray(ctx.board_compatible, dtype=np.float64)
        self._weights: Dict[int, np.ndarray] = {}
        for s in self._all_seats:
            src = ctx.ranges[s] if s in ctx.ranges else ctx.folded_ranges[s]
            w = np.asarray(src, dtype=np.float64) * bc
            total = w.sum()
            if total <= 0.0:
                raise ValueError(f"MCCFR root: seat {s} has zero board-compatible reach.")
            self._weights[s] = w / total
        # Precomputed inverse-CDF for the root-hole draw: the joint belief weights
        # are fixed for the whole solve, so a per-seat cumulative built once here
        # lets ``_sample_root_holes`` draw with one ``rng.random()`` + searchsorted
        # instead of ``rng.choice(p=)`` rebuilding and revalidating a CDF per call.
        # The cumulative is normalised to end exactly at 1.0 so the draw mirrors
        # ``Generator.choice`` bit-for-bit (numpy's own p-path is cumsum → /cdf[-1]
        # → random() → searchsorted(side="right")); the search stays byte-identical.
        self._cdf: Dict[int, np.ndarray] = {}
        for s, w in self._weights.items():
            cdf = np.cumsum(w)
            cdf /= cdf[-1]
            self._cdf[s] = cdf
        # Single-copy walk: the traversal reseat-mutates ``root_env`` in place per
        # iteration (§6.5) instead of deepcopying it, so snapshot its pristine
        # card-state now — before any walk — to rewind afterwards, honouring
        # solve()'s "never mutates root_env" contract and keeping warm re-search /
        # downstream reads correct.  Only the cards + deal cursor change (reseat);
        # the betting state is restored by the walk's own make/undo balance.
        self._pristine_holes = [tuple(int(c) for c in p._cards) for p in root_env.players]
        self._pristine_cards = root_env.deck._cards.copy()
        self._pristine_idx = int(root_env.deck._idx)
        # Canonical full deck, built once and reused by every reseat so the
        # per-iteration re-hole never rebuilds it (it is invariant for the solve).
        self._full_deck = make_deck_arr(
            root_env._low_card_rank, root_env._high_card_rank
        )
        # Dedicated board-shuffle RNG.  reseat re-randomises the undealt deck each
        # iteration, but that draw must NOT come from ``self.rng`` — that stream
        # drives hole + action sampling, and polluting it with per-iteration board
        # shuffles desyncs the traversal and wrecks convergence.  The old
        # per-iteration ``with_hole_cards`` shuffled on the *global* ``np.random``
        # (a separate stream) for exactly this reason; we keep the separation but
        # derive an independent child from ``self.rng``'s seed sequence so parallel
        # replicas stay decorrelated (each replica's ``rng`` is its own seeded
        # substream), rather than sharing fork-inherited global state.  Spawn from
        # the seed sequence so ``self.rng`` itself is NOT advanced — the sampling
        # stream (hole + action draws) must stay bit-identical to a board-free
        # baseline, or the traversal desyncs.
        try:
            _board_seed = self.rng.bit_generator._seed_seq.spawn(1)[0]
            self._board_rng = np.random.default_rng(_board_seed)
        except AttributeError:  # generator without an exposed seed sequence
            self._board_rng = np.random.default_rng(int(self.rng.integers(0, 2 ** 63 - 1)))

        # ---- Traverser-vectorized walk (all MCCFR subgames) ------------------
        # The traverser's private hand is solved in VECTOR form: every combo's row
        # updates every iteration (killing the per-row starvation), while opponents
        # still SAMPLE one action (external sampling — keeps multiway tractable) and
        # the board is the frozen reseat runout.  Decision + strategy fold into one
        # walk.  Two mutually-exclusive shapes by root street:
        #   * turn/river root (>=2): leaf-free, so the walk plays to real terminals
        #     and crosses chance nodes → future streets stored per LUT cluster
        #     (``_cmaps``); it never reaches a depth-limit leaf.
        #   * pre-flop / multiway-flop root (<2): the depth limit cuts to a leaf
        #     BEFORE any future-street decision node, so the walk only ever visits
        #     root-street (combo) decision nodes + the continuation meta-game leaf
        #     (also combo-keyed) — no cluster machinery is reached.
        self._n_combos = int(self._combo_cards.shape[0])
        self._combo_index = root_env.combo_index
        self._my_seat = ctx.my_seat
        self._my_combo = root_env.combo_index.get(self._my_hole)
        # Per-live-seat board-masked reach (the traverser's own-reach weight for the
        # average strategy; matches the vector regime's ``_reach`` scale).
        self._reach: Dict[int, np.ndarray] = {
            s: np.asarray(ctx.ranges[s], dtype=np.float64) * bc
            for s in self._live_seats
        }
        self._root_len = len(root_env.community_cards)
        # Cluster machinery is needed exactly when the subgame is LEAF-FREE, i.e.
        # the walk can cross to a future (clustered) street — every subgame except
        # the depth-limited round-1 (pre-flop) and multiway round-2 (flop) roots
        # (§3).  Note a HEADS-UP flop is leaf-free (the multiway guard needs >2
        # seats), so it too crosses turn+river and needs ``_cmaps`` — the router
        # sends it to the vector regime in production, but the MCCFR walk must still
        # handle it correctly (the differential harness drives it directly).
        street = ctx.street_at_root
        n_at_root = (ctx.depth_limit.n_players_at_root if ctx.depth_limit is not None
                     else root_env.n_players_started_round)
        leaf_containing = (street == 0) or (street == 1 and n_at_root > 2)
        self._cmaps = (
            None if leaf_containing
            else ClusterMapper(root_env.card_info_lut, root_env.combo_cards,
                               root_env.community_cards, street)
        )
        # Search Cython core (Phase 3): when enabled and the search is overlay-free,
        # the walk runs on the compiled FastState engine (make/undo + concrete
        # settlement in-core, leaf rollout via the cloned frontier).  The walk
        # re-holes the root every iteration, so the FastState is rebuilt per
        # traversal (:meth:`_make_walk_env`); a build that returns ``None`` falls
        # back to the PokerEnv walk for that iteration.  Equilibrium-gated, not
        # byte-identical (the FastState leaf's board draw diverges the RNG stream).
        # Modeled solves run on the core too: the clamp keys the model by *cluster*
        # (``policy_state_for_cluster``), served identically by both engines.
        self._use_core = False
        try:
            from poker_ai._core import CORE_AVAILABLE
            from poker_ai._core.flags import search_core_enabled
            self._use_core = bool(CORE_AVAILABLE and search_core_enabled())
        except ImportError:
            pass

        # Root-street LUT cluster per combo, for the opponent-model clamp's info-set
        # key.  Built only when models are attached, so a vanilla solve pays nothing.
        # ``_cmaps`` is ``None`` on the leaf-containing roots (pre-flop, multiway
        # flop) — where every walked node is a root-street node — so the map is
        # derived straight from the LUT there.
        self._root_cluster = None
        if getattr(ctx, "models", None):
            self._root_cluster = (
                self._cmaps.root_cluster_of() if self._cmaps is not None
                else clusters_for_board(
                    root_env.card_info_lut[_STREET_NAME[street]],
                    root_env.combo_cards,
                    np.array(root_env.community_cards, dtype=np.int64),
                )
            )
        self._iter = 0
        # Root-value convergence signal (calibration; see the vector regime).  MCCFR's
        # per-combo value is already the EV vs the belief-sampled opponents (their
        # holes are drawn from the beliefs), so no reach-mass normalisation is applied.
        # Only iterations where the played combo is card-consistent with the sampled
        # field contribute (an overlapping draw is an impossible joint, not a sample).
        # Write-only side counter — the walk is untouched.
        self._val_iter = 0
        self._track_root_value = (self._my_seat in self._live_seats
                                  and self._my_combo is not None)

        # VR-MCCFR baseline (opponent_modeling §5.5): a control variate on the sampled
        # opponent-action counterfactual values.  DBR-only — gated on both the config
        # flag and the presence of opponent models — so a vanilla solve (no models) is
        # byte-identical regardless of the flag.  ``_vbaseline[(p, pk)]`` is a per-
        # traverser-per-opponent-node ``(n_legal, n_combos)`` baseline, EMA-updated from
        # observed child values and reset per search (fresh solve).  Unbiased for any
        # baseline.  Keyed by ``(p, pk)`` and not just ``pk``: with 3+ live seats the
        # traverser rotates every iteration, and a bare ``pk`` node is "opponent" from
        # more than one traverser's perspective — those are different value functions
        # (different traverser's per-combo payoff) over the same combo-index space, so
        # collapsing them into one baseline array would silently mix incompatible
        # values instead of reducing variance.
        self._vr = (bool(getattr(cfg, "variance_reduction", False))
                    and bool(getattr(ctx, "models", None)))
        self._vr_decay = float(getattr(cfg, "vr_baseline_decay", 0.5))
        self._vbaseline: Dict[object, np.ndarray] = {}

    def _make_walk_env(self, env):
        """The env the walk traverses this iteration: a fresh FastState adapter over
        the just-reseated root under the search core, else the ``PokerEnv`` itself."""
        if self._use_core:
            fast = build_fast_mccfr_env(env)
            if fast is not None:
                return fast
        return env

    # ------------------------------------------------------------------
    # One traversal
    # ------------------------------------------------------------------

    def iterate(self) -> None:
        # Traverser first (rotation over live seats) — the vectorized draw needs it
        # to sample the phantom traverser hole.
        i = self._live_seats[self._iter % len(self._live_seats)]
        self._iter += 1
        # The traverser is swept in vector form, so its hole is a phantom drawn
        # UNIFORMLY (unbiased frozen runout); opponents keep their belief draw.
        holes = self._sample_root_holes_vectorized(i)
        holes_list = [holes[s] for s in range(self.n_players)]
        # In-place re-hole of the single walk env (no per-iteration deepcopy):
        # atomically resets every seat's holes + the deck as one permutation,
        # board frozen (§6.5).  The prior iteration's walk left the betting state
        # root-restored (make/undo balance); reseat rebuilds only the cards, so
        # ``root_env`` is a valid fresh sampled root again.
        env = self.root_env
        env.reseat_private_cards(holes_list, rng=self._board_rng, full_deck=self._full_deck)
        # Single vectorized walk: regret + average strategy in one pass over the
        # traverser's whole range (opponents/chance still sampled).  Under the search
        # core the walk runs on a FastState adapter built from the reseated root.
        v = self._vectorized_iterate(self._make_walk_env(env), i, holes)
        if self._track_root_value and i == self._my_seat:
            self._record_root_value(v, holes)

    def _record_root_value(self, v: np.ndarray,
                           holes: Dict[int, Tuple[int, int]]) -> None:
        """Accumulate the hero's linearly-weighted root-EV estimate (calibration).

        ``v`` is the root per-combo counterfactual value the traverser walk returned
        (already the EV vs the belief-sampled opponents, so no normalisation).  Skips
        iterations where the played hand shares a card with a sampled opponent hole —
        an impossible joint whose value is not a valid sample.  Pure side-effect:
        reads one float, writes two counters, never advances the RNG (byte-identical).
        """
        mine = set(self._my_hole)
        for s in range(self.n_players):
            if s != self._my_seat and mine & {int(c) for c in holes[s]}:
                return
        self._val_iter += 1
        w = float(self._val_iter)
        self.state.root_value_num += w * float(v[self._my_combo])
        self.state.root_value_den += w

    def restore_root(self) -> None:
        """Rewind ``root_env`` to its pristine card-state after the in-place walk.

        The traversal reseat-mutates ``root_env`` (holes + deck order + cursor)
        rather than deepcopying it, so ``solve()`` calls this once the loop ends to
        leave ``root_env`` byte-identical — the documented "never mutates root_env"
        contract that warm re-search and any downstream reader rely on.  The
        betting state is already root-restored by the walk's make/undo balance, so
        only the cards + deal cursor are rewound here.  Idempotent; a no-op if the
        solve ran zero iterations (root_env never reseated).
        """
        deck = self.root_env.deck
        deck._cards[:] = self._pristine_cards
        deck._idx = self._pristine_idx
        for p, h in zip(self.root_env.players, self._pristine_holes):
            p._cards = h

    # ------------------------------------------------------------------
    # Joint root sampling (§6.5 step 1)
    # ------------------------------------------------------------------

    def _draw_index(self, cdf: np.ndarray) -> int:
        """Inverse-CDF draw ∝ the weights whose cumulative sum is ``cdf``.

        One ``rng.random()`` + ``searchsorted`` in place of ``rng.choice(p=w)``:
        the weights are fixed for the whole solve, so the cumulative is built once
        (:attr:`_cdf`, normalised to end at 1.0) rather than rebuilt and revalidated
        on every draw.  This is exactly what ``Generator.choice`` does internally, so
        the draw is bit-for-bit identical; ``side="right"`` gives ``P(i) == w[i]`` and
        never selects a zero-mass entry (a flat cdf step).  ``cdf[-1] == 1.0`` exactly
        and ``rng.random() < 1.0``, so the index can never run past the last bin.
        """
        return int(np.searchsorted(cdf, self.rng.random(), side="right"))

    def _sample_root_holes(self) -> Dict[int, Tuple[int, int]]:
        """Draw one card-disjoint assignment from the joint belief.

        Rejection sampling: draw each seat's combo independently ∝ its marginal,
        reject the whole draw on any inter-seat card conflict, retry.  This is
        exact for the joint ``P(h) ∝ Π_s w_s[h_s]·1[disjoint]`` (the product
        conditioned on the no-conflict event).  Zero-mass hands are never drawn.
        A bounded retry budget falls back to sequential-with-removal (a mild,
        logged approximation) only under pathological range concentration.
        """
        cc = self._combo_cards
        for _ in range(256):
            choice: Dict[int, Tuple[int, int]] = {}
            used: set = set()
            ok = True
            for s in self._all_seats:
                ci = self._draw_index(self._cdf[s])
                c0, c1 = int(cc[ci, 0]), int(cc[ci, 1])
                if c0 in used or c1 in used:
                    ok = False
                    break
                used.add(c0)
                used.add(c1)
                choice[s] = (c0, c1)
            if ok:
                return choice
        logger.warning(
            "MCCFR joint sampler: rejection budget exhausted; falling back to "
            "sequential-with-removal (approximate) for this traversal."
        )
        return self._sequential_sample()

    def _disjoint_mask(self, excluded) -> np.ndarray:
        """``(n_combos,)`` bool: combos sharing no card with any card in ``excluded``.

        Card-removal check via the precomputed densified slots (``__init__``)
        instead of ``np.isin`` against the small, per-call ``excluded`` set —
        confirmed by profiling as a hot cost (12-20% of MCCFR wall) across every
        call site below. ``excluded`` may be empty.
        """
        mask = np.zeros(self._deck_size, dtype=bool)
        if excluded:
            mask[np.searchsorted(self._deck_uniq, list(excluded))] = True
        return ~(mask[self._combo_slot0] | mask[self._combo_slot1])

    def _sample_root_holes_vectorized(self, traverser: int) -> Dict[int, Tuple[int, int]]:
        """Root holes for the vectorized walk: real opponents, phantom traverser.

        Every seat except ``traverser`` is drawn from its belief (used concretely —
        opponent action sampling + the terminal ranks).  The traverser's hole is a
        **phantom**: the walk sweeps all of its combos and never reads it, so it
        serves only to partition the deck for ``reseat_private_cards``.  It is drawn
        **uniformly** over combos card-disjoint from the opponents and board — NOT
        from the traverser's range — because reseat excludes the traverser hole from
        the frozen runout pool, so a range-weighted phantom would bias the future
        board's marginal (range-frequent cards under-sampled as board cards).  A
        uniform phantom keeps the runout hole-independent: the public-chance-sampling
        property the vector regime has by construction.  Opponents are sampled
        *without* conditioning on the phantom, so no combo's card removal is skewed
        either.  Falls back to the range-weighted joint sampler only if the (rare)
        rejection budget is exhausted.
        """
        cc = self._combo_cards
        opp_seats = [s for s in self._all_seats if s != traverser]
        board = set(int(c) for c in self.root_env.community_cards)
        for _ in range(256):
            choice: Dict[int, Tuple[int, int]] = {}
            used: set = set()
            ok = True
            for s in opp_seats:
                ci = self._draw_index(self._cdf[s])
                c0, c1 = int(cc[ci, 0]), int(cc[ci, 1])
                if c0 in used or c1 in used:
                    ok = False
                    break
                used.add(c0)
                used.add(c1)
                choice[s] = (c0, c1)
            if not ok:
                continue
            excl = used | board
            feas = np.flatnonzero(self._disjoint_mask(excl))
            if feas.size == 0:
                continue
            row = int(feas[self.rng.integers(feas.size)])
            choice[traverser] = (int(cc[row, 0]), int(cc[row, 1]))
            return choice
        logger.warning(
            "MCCFR vectorized sampler: rejection budget exhausted; falling back to "
            "the range-weighted joint sampler (phantom traverser hole range-biased)."
        )
        return self._sample_root_holes()

    def _sequential_sample(self) -> Dict[int, Tuple[int, int]]:
        cc = self._combo_cards
        order = np.array(self._all_seats)
        self.rng.shuffle(order)
        used: set = set()
        choice: Dict[int, Tuple[int, int]] = {}
        for s in order.tolist():
            w = self._weights[s].copy()
            if used:
                w[~self._disjoint_mask(used)] = 0.0
            total = w.sum()
            if total <= 0.0:
                raise ValueError("MCCFR joint sampler: card exhaustion in fallback.")
            w /= total
            ci = int(self.rng.choice(len(w), p=w))
            c0, c1 = int(cc[ci, 0]), int(cc[ci, 1])
            used.add(c0)
            used.add(c1)
            choice[s] = (c0, c1)
        return choice

    # ------------------------------------------------------------------
    # Traverser-vectorized walk (§6.5 — leaf-free turn/river subgames)
    # ------------------------------------------------------------------

    def _vectorized_iterate(self, env, i: int,
                            holes: Dict[int, Tuple[int, int]]) -> np.ndarray:
        """One vectorized external-sampling walk: all of ``i``'s combos at once.

        Regret and average strategy accrue in a single pass — ``pi_p`` is exact
        because the traverser's actions are expanded, so no separate sampled
        strategy pass is needed.  Opponents sample one action from their single
        sampled hole's row; the board is the frozen reseat runout.  Writes the
        shared ``vregret``/``vstrat`` matrices — the same tables the vector regime
        and :class:`SearchPolicy` read.  Returns the traverser's ``(n_combos,)`` root
        counterfactual value (used by the calibration root-value signal).
        """
        runout = None
        if self._cmaps is not None:
            # Frozen board for this iteration — reseat has already dealt the WHOLE
            # 5-card runout before the walk starts, and it never changes again this
            # iteration (leaf-free root: no depth-limit leaf re-deals mid-walk).
            # Read it off the PokerEnv root (``env`` here may be the deck-less
            # FastState adapter). Only turn/river/HU-flop roots reach here;
            # leaf-containing roots (preflop / multiway-flop, cmaps is None) deal
            # the board PROGRESSIVELY as betting advances instead — see below.
            runout = self.root_env.deck.board_runout(5)
            if self._cmaps.n_completion:
                # Future-street cluster ids past the root community, folded in
                # exactly as the vector regime folds its sampled completion.
                completion = tuple(int(c) for c in runout[self._root_len:])
                self._cmaps.refresh(completion)
        # Traverser root reach: its board-masked range, additionally masked to
        # combos card-disjoint from every OTHER seat's sampled hole (card removal
        # — the vector analogue of the joint sampler's disjointness rejection).
        excl: set = set()
        for s in range(self.n_players):
            if s != i:
                excl.update(int(c) for c in holes[s])
        disjoint = self._disjoint_mask(excl)
        pi_p0 = self._reach[i] * disjoint

        # Fixed per-iteration payout-feasibility mask: every terminal reached
        # during THIS iteration's walk settles the SAME traverser (``i``) against
        # the SAME frozen board + the SAME other-seats' holes, so the card-removal
        # mask ``vector_payout_concrete`` needs is identical at every one of those
        # terminals (measured ~33 calls/iteration on a HU-flop cell) — compute it
        # ONCE here instead of rebuilding it at each terminal.  Valid ONLY when the
        # board is genuinely fixed for the whole iteration (leaf-free roots,
        # ``runout is not None``): a leaf-containing root (preflop / multiway-flop)
        # deals its board PROGRESSIVELY as the walk advances streets, so a terminal
        # reached mid-walk may hold fewer board cards than the eventual full
        # board — precomputing against the wrong (future) board there would
        # silently rule out combos on cards not yet actually dealt at that
        # terminal. Those roots pass ``feasible=None`` and keep computing the mask
        # fresh per terminal (the pre-existing, always-correct behavior).
        self._iter_feasible = (
            self._disjoint_mask(set(int(c) for c in runout) | excl)
            if runout is not None else None
        )
        return self._vwalk(env, i, pi_p0, holes)

    def _vwalk(self, env, p: int, pi_p: np.ndarray,
               holes: Dict[int, Tuple[int, int]]) -> np.ndarray:
        """One vectorized CFR pass for traverser ``p`` (entered on a decision node).

        Returns the ``(n_combos,)`` per-combo counterfactual value.  Mirrors
        :meth:`poker_ai.search.vector._VectorSolver._walk` but external-sampled:
        the opponent samples ONE action from its single sampled hole's row instead
        of expanding a range, and terminals settle against concrete opponents
        (:meth:`PokerEnv.vector_payout_concrete`).  Only the traverser's rows
        accrue regret/strategy.
        """
        street = env.betting_round
        actor = env.player_i
        legal = tuple(a for a in env.legal_actions if a is not None)
        pk = env.public_key
        is_root = street == self.ctx.street_at_root
        if is_root:
            n_rows, row_space, cof, gof = self._n_combos, "combo", None, None
        else:
            cof = self._cmaps.cluster_of(street)
            gof = self._cmaps.gather_of(street)      # hoisted gather index
            n_rows, row_space = self._cmaps.n_rows(street), "cluster"
        # Shared vector-form preamble + freezing (§6.5, §5 — see :mod:`vform`).
        sigma, regret, strat = node_sigma(
            self.state, pk, legal, actor, is_root, n_rows, row_space, cof, gof
        )
        # Opponent-model clamp (opponent_modeling §5.2) — no-op without models.
        # Before `freeze_combo` so the bot's pinned actual-hand row always wins.
        sigma = apply_model_clamp(
            sigma, self.ctx, self.state, env, pk, actor, len(legal),
            self._combo_cards, cof, n_rows, is_root, gof,
            self._cmaps, street, self._root_cluster,
        )
        frozen_combo = freeze_combo(
            self.state, pk, sigma, is_root, actor, self._my_seat, self._my_combo
        )

        if actor != p:
            # Opponent node: sample one action from its single sampled hole's row
            # (external sampling).  The traverser's per-combo value below is the
            # opponent-strategy-weighted expectation ``Σ_a σ_opp(a)·v(·a)``; the sampled
            # child ``v`` is an unbiased single-sample estimate of it.
            # ``holes[actor]`` is already ``(c0, c1)`` with ``c0 < c1`` by
            # construction (every sampler draws it straight from ``combo_cards``
            # rows, which ``combo_index`` keys match verbatim) — sorting it here
            # would be a no-op on every call.
            opp_ci = self._combo_index[holes[actor]]
            opp_row = sigma[opp_ci]                       # (n_legal,) opponent strategy
            a_idx = sample_index(self.rng, opp_row)
            token = env.step_in_place(legal[a_idx], settle_winners=False)
            v = self._vchild(env, p, pi_p, holes, street)
            env.undo(token)
            if not self._vr:
                return v
            # VR-MCCFR baseline (opponent_modeling §5.5).  Sampling is ON-POLICY
            # (q = σ_opp), so the baseline-corrected estimator reduces to the
            # σ-weighted baseline expectation plus the sampled child's deviation from
            # its baseline — no ``1/q`` blow-up.  Unbiased for ANY baseline:
            #   E_a*[Σ_a σ(a)·b(a) + (v(·a*) − b(a*))] = Σ_a σ(a)·v(·a).
            # ``b = 0`` (first visit) recovers the plain single-sample value exactly, so
            # this only ever *reduces* variance as the baseline learns.  EMA-update the
            # sampled action's baseline AFTER forming the estimate (keeps it independent
            # of this sample → unbiased).  Both the returned value and — via the parent
            # traverser node's ``child_vs`` — its regret update inherit the lower variance.
            bkey = (p, pk)
            b = self._vbaseline.get(bkey)
            if b is None:
                b = np.zeros((len(legal), self._n_combos), dtype=np.float64)
                self._vbaseline[bkey] = b
            corrected = vr_baseline_estimate(opp_row, b, a_idx, v)
            b[a_idx] += self._vr_decay * (v - b[a_idx])   # EMA-update after the estimate
            return corrected

        # Traverser node: expand every action, weighting children by p's strategy.
        child_vs = np.empty((len(legal), self._n_combos), dtype=np.float64)
        for a_idx, action in enumerate(legal):
            token = env.step_in_place(action, settle_winners=False)
            child_vs[a_idx] = self._vchild(env, p, pi_p * sigma[:, a_idx], holes, street)
            env.undo(token)
        # All combos in a cluster share one row (§6.5), so a future-street update is
        # a presorted segment-sum; a root node adds combo rows directly.
        scatter = None if is_root else (
            lambda t, pc: self._cmaps.scatter_add(t, pc, street)
        )
        return traverser_update(
            regret, strat, sigma, child_vs, pi_p, frozen_combo, scatter
        )

    def _vchild(self, env, p: int, pi_p: np.ndarray,
                holes: Dict[int, Tuple[int, int]], parent_street: int) -> np.ndarray:
        """Descend one edge: settle a terminal / leaf, or recurse a decision node."""
        verdict = self.ctx.depth_limit.classify(env)
        if verdict == "terminal":
            return env.vector_payout_concrete(p, feasible=self._iter_feasible)
        if verdict == "leaf":
            # Depth-limit continuation meta-game (only pre-flop / multiway-flop
            # roots reach it; those never cross to a future street first).
            return self._vmeta_game(env, p, pi_p, holes)
        if env.betting_round > parent_street:
            # Crossed a chance node (board grew): a combo holding the freshly dealt
            # card carries no reach below it.  No 1/R — the board is frozen this
            # iteration (chance is the reseat runout).
            feas = self._cmaps.feas(env.betting_round)
            return self._vwalk(env, p, pi_p * feas, holes)
        return self._vwalk(env, p, pi_p, holes)

    # ------------------------------------------------------------------
    # Vectorized continuation meta-game leaf (§6.5 step 4)
    # ------------------------------------------------------------------

    def _vmeta_game(self, env, p: int, pi_p: np.ndarray,
                    holes: Dict[int, Tuple[int, int]]) -> np.ndarray:
        """Vectorized depth-limit continuation meta-game → ``(n_combos,)``.

        The traverser explores all four §4 bias classes (per combo); every other
        active seat samples one bias from its single sampled hole's meta-row.  A
        fully-chosen profile is scored per-combo by :func:`continuation_value_vector`.
        Meta-rows are keyed ``((public_key, "META", seat), combo)`` and stored
        **lossless per combo** (a clean deviation from the scalar path's leaf-street
        cluster key — it needs no future-street universe, which for a pre-flop root
        would mean enumerating every flop).  Meta-rows are never frozen (freezing
        pins only real decisions), so no freeze substitution here.
        """
        pk_base = env.public_key
        active = [s for s in range(self.n_players) if env.players[s].is_active]
        # Sample the non-traverser active seats' biases once, from their meta-row.
        sampled: Dict[int, BiasClass] = {}
        for s in active:
            if s != p:
                sampled[s] = _BIAS_CLASSES[self._vmeta_sample(env, holes, s, pk_base)]

        if p not in active:
            # Traverser cannot act here; score the sampled profile as-is (no update).
            return self._vleaf_value(env, dict(sampled), pk_base, p)

        meta_pk = (pk_base, "META", p)
        self.state.ensure_vnode(meta_pk, _BIAS_CLASSES, p, self._n_combos, "combo")
        regret = self.state.vregret[meta_pk]              # (n_combos, 4)
        strat = self.state.vstrat[meta_pk]
        sigma = regret_match_matrix(regret)              # (n_combos, 4)
        child_vs = np.empty((len(_BIAS_CLASSES), self._n_combos), dtype=np.float64)
        for b, bias in enumerate(_BIAS_CLASSES):
            profile = dict(sampled)
            profile[p] = bias
            child_vs[b] = self._vleaf_value(env, profile, pk_base, p)
        cv = np.moveaxis(child_vs, 0, -1)                 # (n_combos, 4)
        v = (sigma * cv).sum(axis=-1)                     # (n_combos,)
        regret += cv - v[:, None]
        # Own-reach-weighted strategy sum, masked to combos holdable given the leaf
        # frontier's board (combos sharing a board card carry no strategy mass; the
        # regret needs no mask — infeasible combos already have cv == 0).
        strat += (pi_p * self._board_feasible_mask(env))[:, None] * sigma
        return v

    def _vmeta_sample(self, env, holes, seat: int, pk_base) -> int:
        """Sample one bias index for ``seat`` from its single sampled hole's meta-row."""
        meta_pk = (pk_base, "META", seat)
        self.state.ensure_vnode(meta_pk, _BIAS_CLASSES, seat, self._n_combos, "combo")
        sigma = regret_match_matrix(self.state.vregret[meta_pk])
        opp_ci = self._combo_index[holes[seat]]  # already (c0, c1), c0 < c1
        return sample_index(self.rng, sigma[opp_ci])

    def _vleaf_value(self, env, profile: Dict[int, BiasClass], pk_base,
                     traverser: int) -> np.ndarray:
        """Per-combo continuation value at this leaf, memoised search-wide (§6.4.1).

        Keyed on ``(leaf pk, traverser, OTHER-seat holes, profile)`` — the traverser
        hole is dropped because the returned ``(n_combos,)`` vector already spans
        every traverser combo, so one entry serves them all (a big cache-hit win over
        the scalar per-hole key).
        """
        holes_key = tuple(
            tuple(int(c) for c in env.players[s].cards)
            for s in range(self.n_players) if s != traverser
        )
        ck = (pk_base, traverser, holes_key, tuple(sorted(profile.items())))
        val = self.state.leaf_value_cache.get(ck)
        if val is None:
            val = _leaf_fn(env)(env, profile, self.ctx, traverser)
            self.state.leaf_value_cache[ck] = val
        return val

    def _board_feasible_mask(self, env) -> np.ndarray:
        """``(n_combos,)`` bool: combos sharing no card with ``env``'s board."""
        board = list(env.community_cards)
        if not board:
            return np.ones(self._n_combos, dtype=bool)
        return self._disjoint_mask(board)

