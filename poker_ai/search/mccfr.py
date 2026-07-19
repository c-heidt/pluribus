"""External-sampling Linear MCCFR regime for the subgame solver (§6.5).

The MCCFR regime is the *large / early* path — round 1, all of round 2, and any
large multiway later subgame.  Per traversal it samples **one** card-disjoint
hole assignment from the joint belief (so inter-seat card removal is reflected in
the draw), then runs an external-sampling CFR pass on the make/undo env: the
**traverser explores all of its actions**, every opponent **samples one** action
from its regret-matched strategy, and chance (the board deal) is the engine's own
one-outcome-per-round draw.  Depth-limit leaves are the §6.4 continuation
meta-game, solved as an ordinary action; terminals score with ``env.payout`` (or
``runout_equity`` at a decision-free all-in).  Regret accrues only on the
traverser's rows, which are variable-width (per-node legal set), keyed
``(public_key, hand_row)`` in the shared :class:`SolverState`.

This mirrors :func:`poker_ai.blueprint.cfr._traverse` but with variable-width
rows, the belief-sampled root, the meta-game action, and freezing.  CFR-P pruning
is omitted (it never engages in a short search, §6.5).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

from environment.utils import make_deck_arr
from poker_ai.blueprint.tree_utils import sample_index
from poker_ai.search.cluster_maps import ClusterMapper
from poker_ai.search.context import SubgameContext
from poker_ai.search.leaf import continuation_value, continuation_value_vector
from poker_ai.search.policy import BiasClass
from poker_ai.search.solver_state import Key, SolverConfig, SolverState, _hand_row
from poker_ai.search.vector import _regret_match_matrix

logger = logging.getLogger(__name__)

# Search Cython core (Phase 4b): under ``PLURIBUS_SEARCH_CORE=1`` swap the leaf
# rollout (the ~74% MCCFR cost) for the FastState-driven rollout, which builds the
# leaf env once and make/undo-walks it per rollout, settling terminals in-core.
# ``_leaf_value`` calls the module global ``continuation_value`` so the rebind is
# transparent; the pure-Python reference is retained (and used as the fallback the
# fast path delegates to for unrepresentable frontiers).  Equilibrium-gated.
_continuation_value_py = continuation_value
try:
    from poker_ai._core.flags import search_core_enabled as _search_core_enabled

    if _search_core_enabled():
        from poker_ai.search.leaf_fast import continuation_value_fast as continuation_value
except ImportError:
    pass

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

        # Decision-free runout equity is exact and cheap once a flop is out (<=2
        # cards to come), but a *preflop* (round-1) all-in is a 5-card runout that
        # blows past runout_equity's enumeration cap and falls back to sampling
        # thousands of boards *per terminal, per iteration*.  For a preflop root we
        # therefore score all-in terminals by the env's single sampled board
        # (env.payout) regardless of the flag — far cheaper, and the per-terminal
        # variance is absorbed across iterations.  (continuation_value keeps the
        # flag: its rollouts only reach all-ins on the flop or later, where the
        # exact path is cheap.)
        self._use_equity = (
            bool(cfg.leaf.use_decision_free_equity) and ctx.street_at_root != 0
        )

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
        self._iter = 0

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
        # traverser's whole range (opponents/chance still sampled).
        self._vectorized_iterate(env, i, holes)

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
            excl_arr = np.fromiter(excl, dtype=cc.dtype, count=len(excl))
            feas = np.flatnonzero(
                ~(np.isin(cc[:, 0], excl_arr) | np.isin(cc[:, 1], excl_arr))
            )
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
                forbidden = np.fromiter(used, dtype=np.int64)
                conflict = np.isin(cc[:, 0], forbidden) | np.isin(cc[:, 1], forbidden)
                w[conflict] = 0.0
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
    # Recursion (§6.5 steps 2-4)
    # ------------------------------------------------------------------

    def _traverse(self, env, i: int, holes: Dict[int, Tuple[int, int]]) -> float:
        verdict = self.ctx.depth_limit.classify(env)
        if verdict == "terminal":
            return self._terminal_value(env, i)
        if verdict == "leaf":
            return self._meta_game(env, i, holes)

        actor = env.player_i
        legal = tuple(a for a in env.legal_actions if a is not None)
        pk = env.public_key
        key = (pk, _hand_row(env, holes[actor], self.ctx.street_at_root))
        self.state.ensure_node(pk, legal, actor)
        sig = self._node_sigma(key, actor, holes)

        if actor != i:
            # External sampling: one opponent action from its current strategy.
            a_idx = sample_index(self.rng, sig)
            token = env.step_in_place(legal[a_idx])
            value = self._traverse(env, i, holes)
            env.undo(token)
            return value

        # Traverser: explore every action.
        va = np.empty(len(legal), dtype=np.float64)
        for a_idx, action in enumerate(legal):
            token = env.step_in_place(action)
            va[a_idx] = self._traverse(env, i, holes)
            env.undo(token)
        node_v = float(np.dot(sig, va))
        if not self._is_frozen(key, actor, holes):
            self.state.add_regret(key, va - node_v)
        return node_v

    def _update_strategy(self, env, i: int, holes: Dict[int, Tuple[int, int]]) -> None:
        """Average-strategy pass: one sampled playthrough, accumulating ``i``'s rows.

        Mirrors :func:`poker_ai.blueprint.strategy.update_strategy`.  Every node
        samples a single action (a single playthrough, not a branching tree); when
        the actor is the traverser ``i`` its current strategy is accumulated into
        ``strat_sum`` (unless the row is frozen).  Because ``i``'s own actions are
        **sampled** here, visits to ``i``'s infosets carry the π_i reach weight,
        which is exactly what makes the normalised ``strat_sum`` the correct
        average strategy.  The regret pass must therefore **not** accumulate it —
        there ``i`` explores all actions, which would mis-weight by the opponent
        reach.  (Accumulating the full ``sig`` rather than a unit count of the
        sampled action is the same in expectation, with lower variance.)
        """
        verdict = self.ctx.depth_limit.classify(env)
        if verdict == "terminal":
            return
        if verdict == "leaf":
            # Continuation meta-game: accumulate ``i``'s meta-row, then stop — the
            # continuation below the leaf is not part of the search tree.
            if env.players[i].is_active:
                key, sig = self._meta_node(env, holes, i, env.public_key)
                if not self._is_frozen(key, i, holes):
                    self.state.add_strat(key, self._frozen_or(sig, key, i, holes))
            return

        actor = env.player_i
        legal = tuple(a for a in env.legal_actions if a is not None)
        pk = env.public_key
        key = (pk, _hand_row(env, holes[actor], self.ctx.street_at_root))
        self.state.ensure_node(pk, legal, actor)
        sig = self._node_sigma(key, actor, holes)
        if actor == i and not self._is_frozen(key, actor, holes):
            self.state.add_strat(key, sig)
        a_idx = sample_index(self.rng, sig)
        token = env.step_in_place(legal[a_idx])
        self._update_strategy(env, i, holes)
        env.undo(token)

    # ------------------------------------------------------------------
    # Traverser-vectorized walk (§6.5 — leaf-free turn/river subgames)
    # ------------------------------------------------------------------

    def _vectorized_iterate(self, env, i: int, holes: Dict[int, Tuple[int, int]]) -> None:
        """One vectorized external-sampling walk: all of ``i``'s combos at once.

        Regret and average strategy accrue in a single pass — ``pi_p`` is exact
        because the traverser's actions are expanded, so no separate sampled
        strategy pass is needed.  Opponents sample one action from their single
        sampled hole's row; the board is the frozen reseat runout.  Writes the
        shared ``vregret``/``vstrat`` matrices — the same tables the vector regime
        and :class:`SearchPolicy` read.
        """
        if self._cmaps is not None and self._cmaps.n_completion:
            # Frozen future runout for this iteration (the board cards past the
            # root community), folded into the future-street cluster ids exactly
            # as the vector regime folds its sampled completion.  Only turn/river
            # roots reach future streets; leaf-containing roots (cmaps is None) cut
            # to a leaf first.
            runout = env.deck.board_runout(5)
            completion = tuple(int(c) for c in runout[self._root_len:])
            self._cmaps.refresh(completion)
        # Traverser root reach: its board-masked range, additionally masked to
        # combos card-disjoint from every OTHER seat's sampled hole (card removal
        # — the vector analogue of the joint sampler's disjointness rejection).
        cc = self._combo_cards
        excl: set = set()
        for s in range(self.n_players):
            if s != i:
                excl.update(int(c) for c in holes[s])
        excl_arr = np.fromiter(excl, dtype=cc.dtype, count=len(excl))
        disjoint = ~(np.isin(cc[:, 0], excl_arr) | np.isin(cc[:, 1], excl_arr))
        pi_p0 = self._reach[i] * disjoint
        self._vwalk(env, i, pi_p0, holes)

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
            n_rows, row_space, cof = self._n_combos, "combo", None
        else:
            cof = self._cmaps.cluster_of(street)
            n_rows, row_space = self._cmaps.n_rows(street), "cluster"
        self.state.ensure_vnode(pk, legal, actor, n_rows, row_space)
        regret = self.state.vregret[pk]                 # (n_rows, width)
        strat = self.state.vstrat[pk]
        sigma_rows = _regret_match_matrix(regret)       # (n_rows, width)
        # Per-combo strategy: identity for a root node, a cluster gather otherwise
        # (infeasible combos map to row 0 — harmless, their reach is zeroed).
        sigma = sigma_rows if is_root else sigma_rows[np.where(cof >= 0, cof, 0)]

        # Freezing (§5): substitute the bot's pinned actual-hand row at every visit
        # to the bot's (root-street) node — for whichever seat is acting, so a
        # sampled opponent that IS the frozen bot also plays the pinned strategy.
        apply_frozen = (
            is_root
            and actor == self._my_seat
            and self._my_combo is not None
            and (pk, self._my_combo) in self.state.frozen
        )
        if apply_frozen:
            sigma[self._my_combo] = self.state.frozen[(pk, self._my_combo)]

        if actor != p:
            # Opponent node: sample one action from its single sampled hole's row
            # (external sampling); return the per-combo value unchanged, no update.
            opp_ci = self._combo_index[tuple(sorted(holes[actor]))]
            a_idx = sample_index(self.rng, sigma[opp_ci])
            token = env.step_in_place(legal[a_idx], settle_winners=False)
            v = self._vchild(env, p, pi_p, holes, street)
            env.undo(token)
            return v

        # Traverser node: expand every action, weighting children by p's strategy.
        child_vs = np.empty((len(legal), self._n_combos), dtype=np.float64)
        for a_idx, action in enumerate(legal):
            token = env.step_in_place(action, settle_winners=False)
            child_vs[a_idx] = self._vchild(env, p, pi_p * sigma[:, a_idx], holes, street)
            env.undo(token)
        cv = np.moveaxis(child_vs, 0, -1)             # (n_combos, width)
        v = (sigma * cv).sum(axis=-1)                 # (n_combos,)
        delta = cv - v[:, None]                       # regret: v_a - v
        strat_delta = pi_p[:, None] * sigma           # strat-sum: own reach * sigma
        if apply_frozen:
            delta[self._my_combo] = 0.0
            strat_delta[self._my_combo] = 0.0
        if is_root:
            regret += delta                           # in-place: writes the store
            strat += strat_delta
        else:
            self._cmaps.scatter_add(regret, delta, street)
            self._cmaps.scatter_add(strat, strat_delta, street)
        return v

    def _vchild(self, env, p: int, pi_p: np.ndarray,
                holes: Dict[int, Tuple[int, int]], parent_street: int) -> np.ndarray:
        """Descend one edge: settle a terminal / leaf, or recurse a decision node."""
        verdict = self.ctx.depth_limit.classify(env)
        if verdict == "terminal":
            return env.vector_payout_concrete(p)
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
        sigma = _regret_match_matrix(regret)              # (n_combos, 4)
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
        sigma = _regret_match_matrix(self.state.vregret[meta_pk])
        opp_ci = self._combo_index[tuple(sorted(holes[seat]))]
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
            val = continuation_value_vector(
                env, profile, self.ctx, traverser, runout_cache=self.state.runout_cache
            )
            self.state.leaf_value_cache[ck] = val
        return val

    def _board_feasible_mask(self, env) -> np.ndarray:
        """``(n_combos,)`` bool: combos sharing no card with ``env``'s board."""
        cc = self._combo_cards
        board = list(env.community_cards)
        if not board:
            return np.ones(self._n_combos, dtype=bool)
        barr = np.fromiter((int(c) for c in board), dtype=cc.dtype, count=len(board))
        return ~(np.isin(cc[:, 0], barr) | np.isin(cc[:, 1], barr))

    def _terminal_value(self, env, i: int) -> float:
        if self._use_equity and env.is_decision_free:
            # Search-lifetime runout memo (§6.4.2): the same all-in reached from
            # many lines/iterations integrates once.  Key on the all-seat holes
            # plus the exact pre-runout snapshot, built exactly as the leaf
            # rollouts do (``leaf.py``) so the two call sites share entries.
            holes_key = tuple(
                tuple(int(c) for c in env.players[s].cards)
                for s in range(self.n_players)
            )
            key = (holes_key, env.runout_key)
            eq = self.state.runout_cache.get(key)
            if eq is None:
                eq = env.runout_equity(rng=self.rng)
                self.state.runout_cache[key] = eq
            self.state.term_runout += 1
            return float(eq[i])
        self.state.term_payout += 1
        return float(env.payout[i])

    def _leaf_value(self, env, profile: Dict[int, BiasClass], pk_base) -> np.ndarray:
        """Continuation value at this leaf, memoised search-wide (§6.4.1).

        For a fixed ``(leaf public_key, all-seat holes, profile)`` the value is
        invariant across CFR iterations — only the meta-game's weighting over
        profiles changes — so the ``n_rollouts`` estimate is computed once per
        key and reused.  The estimate stored is the **first** draw (subsequent
        visits consume no ``ctx.rng``); the search stays deterministic per seed.
        The shared ``runout_cache`` is threaded so a leaf's four bias profiles
        share their decision-free runout integrations.
        """
        holes_key = tuple(
            tuple(int(c) for c in env.players[s].cards)
            for s in range(self.n_players)
        )
        ck = (pk_base, holes_key, tuple(sorted(profile.items())))
        val = self.state.leaf_value_cache.get(ck)
        if val is None:
            val = continuation_value(
                env, profile, self.ctx, runout_cache=self.state.runout_cache
            )
            self.state.leaf_value_cache[ck] = val
        return val

    # ------------------------------------------------------------------
    # Continuation meta-game leaf (§6.5 step 4)
    # ------------------------------------------------------------------

    def _meta_game(self, env, i: int, holes: Dict[int, Tuple[int, int]]) -> float:
        """Value to ``i`` of the depth-limit continuation meta-game.

        Each active seat picks one of the four §4 bias classes as an ordinary
        action.  The traverser explores all four; every other active seat's
        choice is sampled from its current meta-strategy.  A fully-chosen profile
        is scored by :func:`continuation_value`.  Meta-rows are keyed
        ``((public_key, "META", seat), hand_row)`` so no other seat's choice
        leaks into the row (infoset-consistent simultaneous choice).
        """
        pk_base = env.public_key
        active = [s for s in range(self.n_players) if env.players[s].is_active]

        # Sample the non-traverser active seats' biases once.
        sampled: Dict[int, BiasClass] = {}
        for s in active:
            if s == i:
                continue
            _, sig = self._meta_node(env, holes, s, pk_base)
            b = sample_index(self.rng, sig)
            sampled[s] = _BIAS_CLASSES[b]

        if i not in active:
            # Traverser cannot act here; score the sampled profile as-is.
            return float(self._leaf_value(env, dict(sampled), pk_base)[i])

        key, sig = self._meta_node(env, holes, i, pk_base)
        sig = self._frozen_or(sig, key, i, holes)
        va = np.empty(len(_BIAS_CLASSES), dtype=np.float64)
        for b, bias in enumerate(_BIAS_CLASSES):
            profile = dict(sampled)
            profile[i] = bias
            va[b] = float(self._leaf_value(env, profile, pk_base)[i])
        node_v = float(np.dot(sig, va))
        if not self._is_frozen(key, i, holes):
            self.state.add_regret(key, va - node_v)
        return node_v

    def _meta_node(self, env, holes, seat: int, pk_base) -> Tuple[Key, np.ndarray]:
        meta_pk = (pk_base, "META", seat)
        key = (meta_pk, _hand_row(env, holes[seat], self.ctx.street_at_root))
        self.state.ensure_node(meta_pk, _BIAS_CLASSES, seat)
        return key, self.state.sigma(key)

    # ------------------------------------------------------------------
    # Freezing (§5) — only when the sampled hole is the bot's actual hole.
    # ------------------------------------------------------------------

    def _is_actual_bot(self, seat: int, holes: Dict[int, Tuple[int, int]]) -> bool:
        return seat == self.ctx.my_seat and tuple(sorted(holes[seat])) == self._my_hole

    def _is_frozen(self, key: Key, seat: int, holes: Dict[int, Tuple[int, int]]) -> bool:
        return self._is_actual_bot(seat, holes) and key in self.state.frozen

    def _frozen_or(self, sig: np.ndarray, key: Key, seat: int, holes) -> np.ndarray:
        if self._is_frozen(key, seat, holes):
            return self.state.frozen[key]
        return sig

    def _node_sigma(self, key: Key, actor: int, holes) -> np.ndarray:
        return self._frozen_or(self.state.sigma(key), key, actor, holes)
