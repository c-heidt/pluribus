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

from poker_ai.blueprint.tree_utils import sample_index
from poker_ai.search.context import SubgameContext
from poker_ai.search.leaf import continuation_value
from poker_ai.search.policy import BiasClass
from poker_ai.search.solver_state import Key, SolverConfig, SolverState, _hand_row

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
        self._iter = 0

    # ------------------------------------------------------------------
    # One traversal
    # ------------------------------------------------------------------

    def iterate(self) -> None:
        holes = self._sample_root_holes()
        holes_list = [holes[s] for s in range(self.n_players)]
        env = self.root_env.with_hole_cards(holes_list)
        i = self._live_seats[self._iter % len(self._live_seats)]
        self._iter += 1
        # Regret pass: explore the traverser's actions, sample opponents/chance;
        # accumulates regret only (mirrors blueprint/cfr.py:_traverse).
        self._traverse(env, i, holes)
        # Strategy pass: a single sampled playthrough whose visits to the
        # traverser's own (sampled) infosets accumulate the average strategy —
        # the π_i-reach weighting comes from sampling i's own actions, exactly as
        # blueprint/strategy.py:update_strategy.  The regret pass leaves ``env``
        # restored (make/undo), so it is reused here.
        self._update_strategy(env, i, holes)

    # ------------------------------------------------------------------
    # Joint root sampling (§6.5 step 1)
    # ------------------------------------------------------------------

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
                ci = int(self.rng.choice(len(self._weights[s]), p=self._weights[s]))
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
            return float(eq[i])
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
