"""Search-aware play agent (§6.6) — the online orchestrator (Algorithm 2).

:class:`SearchAgent` ties the already-landed search pieces (range tracking,
subgame context, the two CFR regimes behind :func:`solve`, the policy readers and
the leaf evaluator) into a per-hand play lifecycle.  It owns no game dynamics and
**never sees chips** — action strings in, an action string out:

- ``on_hand_start`` resets the per-hand state and snapshots the round-1 root.
- ``on_board_update`` runs the round-boundary Bayes belief update over every
  seat's range, then immediately solves the new round's subgame (so the search is
  ready *before* the bot is asked to act).
- ``on_observed_action`` buffers each observed action for the next boundary update
  and, on rounds 2–4, re-searches the same root (warm-started) when an opponent
  takes an action far enough outside the subgame's action abstraction (pot-fraction
  gap > ``offtree_threshold``); a near-canonical off-tree raise is translated onto
  the solved canonical branch (pseudo-harmonic) instead, at no re-solve cost.
- ``act`` plays the blueprint on round 1 (unless a round-1 search was triggered)
  and the searched final-iteration strategy on rounds 2–4, pinning the bot's
  actual-hand action so a re-search keeps it fixed.

Runner / play-loop wiring lives outside this module: a runner instantiates the
agent and drives these four hooks (plus :meth:`search_round1` if it owns the
chip-denominated round-1 trigger).  See ``docs/subgame_solving.md`` §4–§6.6.
"""

from __future__ import annotations

import copy
import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from environment.poker_env import PokerEnv
from information_abstraction.lookup import clusters_for_board
from poker_ai.search.cluster_maps import _STREET_NAME
from poker_ai.search.context import SubgameContext
from poker_ai.search.policy import BiasClass, Policy
from poker_ai.search.ranges import RangeTracker
from poker_ai.search.solver import SearchResult, SolverConfig, solve

if TYPE_CHECKING:
    from poker_ai.modeling.model import OpponentModel

logger = logging.getLogger(__name__)


class SearchAgent:
    """Real-time-search play agent (§6.6)."""

    def __init__(
        self,
        leaf_policies: Mapping[BiasClass, Policy],
        blueprint_policy: Policy,
        solver_cfg: SolverConfig,
        rng: np.random.Generator,
        *,
        round1_offtree_threshold: float = 0.25,
        round1_max_players: int = 4,
        offtree_threshold: float = 0.25,
        models: Optional[Mapping[int, "OpponentModel"]] = None,
        search_enabled: bool = True,
    ) -> None:
        # --- session-static ---
        self._leaf_policies = leaf_policies     # the four §4 variants (== cfg.leaf.policies)
        self._blueprint = blueprint_policy      # round-1 play + round-1->2 Bayes
        self._cfg = solver_cfg                  # carries .leaf (the LeafConfig)
        self._rng = rng
        # When ``False`` the agent never solves — every boundary/observed-action
        # search trigger is skipped, so ``play_distribution`` always returns the
        # blueprint.  This is the **blueprint-only pipeline test** (no search): NOT
        # an approach and NOT the baseline — real vanilla Pluribus *searches*.  The
        # default ``True`` leaves every search path exactly as before (byte-identical),
        # so vanilla (search, no model) and A (search + model) are unaffected.  Belief
        # tracking still runs (cheap, harmless), but with no search it is never used.
        self._search_enabled = bool(search_enabled)
        self._round1_threshold = float(round1_offtree_threshold)
        self._round1_max_players = int(round1_max_players)
        # Rounds 2-4: an off-tree opponent raise within this pot-fraction of a
        # canonical size is *translated* (pseudo-harmonic, via the solved
        # canonical branch) instead of triggering a warm re-search — bounding
        # re-search frequency at a negligible exploitability cost (§7).
        self._offtree_threshold = float(offtree_threshold)
        # Opponent models, seat → model (opponent_modeling §6.3).  Session-static
        # source; ``on_hand_start`` freezes a per-hand snapshot into ``self._models``.
        # ``None``/empty ⇒ every model code path is inert and the agent is exactly
        # vanilla Pluribus (search, no opponent model — the baseline).
        self._models_source: Mapping[int, "OpponentModel"] = models or {}

        # --- per-hand (initialised in on_hand_start) ---
        #: The hand-start model snapshot.  **The invariant of §6.3:** the belief
        #: tracker's likelihood and the solver's clamp read the *same* frozen
        #: mapping, so the bot infers an opponent's range under the very strategy
        #: it then best-responds to.  Never contains the bot's own seat.
        self._models: Mapping[int, "OpponentModel"] = MappingProxyType({})
        self.my_seat: int = -1
        self.my_hole: Tuple[int, int] = (-1, -1)
        self.tracker: Optional[RangeTracker] = None
        self.pending_actions: List[Tuple[int, PokerEnv, str]] = []
        self.last_search: Optional[SearchResult] = None
        self._root_env: Optional[PokerEnv] = None
        self._ctx: Optional[SubgameContext] = None
        self._searched_this_round: bool = False
        self._folded: bool = False              # bot has folded → agent dormant

    @property
    def search_enabled(self) -> bool:
        """True for vanilla Pluribus and DBR (both search); False only for the
        no-search blueprint-only pipeline test."""
        return self._search_enabled

    @property
    def has_models(self) -> bool:
        """Whether this hand's frozen snapshot models any opponent (a DBR hand).

        Read after :meth:`on_hand_start`.  The runner uses it to mark a hero
        decision as model-informed (``decisions.modeled_decision``) so the summary
        can restrict the exploitation slice to decisions where the model applied.
        """
        return bool(self._models)

    # ----------------------------------------------------------------- #
    # Lifecycle (Algorithm 2)
    # ----------------------------------------------------------------- #
    def on_hand_start(self, env: PokerEnv, my_seat: int) -> None:
        """Reset per-hand state at the start of a new hand (round 1, pre-flop).

        Snapshots the round-1 root (``self._root_env``) so a round-1 search can be
        triggered later, and clears any off-tree injections left over from the
        previous hand.  No search runs — round 1 plays the blueprint by default.
        """
        self.my_seat = int(my_seat)
        self.my_hole = tuple(sorted(int(c) for c in env.players[my_seat].cards))
        live = [i for i in range(env.n_players) if env.players[i].is_active]
        self.tracker = RangeTracker(env, self.my_seat, self.my_hole, live)
        # Freeze the per-hand model snapshot (§6.3 invariant — one mapping feeds
        # both the belief likelihood and the solver clamp for the whole hand).
        # The bot's own seat is dropped defensively: hero is never modeled, and
        # `apply_model_clamp` must never blend hero's rows.
        self._models = MappingProxyType(
            {int(s): m for s, m in self._models_source.items() if int(s) != self.my_seat}
        )
        self.pending_actions = []
        self.last_search = None
        self._ctx = None
        self._searched_this_round = False
        self._folded = False
        # Frozen at pre-flop start; the overlay dict is shared by reference so
        # later injections into the live env are visible here automatically.
        self._root_env = copy.deepcopy(env)
        env.reset_overlay()

    def on_board_update(self, env: PokerEnv, new_cards) -> None:
        """A new betting round has begun; ``env`` is the round-start root.

        Runs the round-boundary belief update (replay buffered actions under the
        round-just-ended's strategy, then zero new-board-conflicting combos), then
        roots and solves the new round's subgame so it is ready before ``act``.

        No-op once the bot has folded — it has no further decisions this hand, so
        there is nothing to search.
        """
        if self._folded:
            return
        self._apply_boundary_belief_update()
        self.tracker.on_board_update(new_cards)
        # The blueprint-only test (search disabled) plays the blueprint at every node,
        # so skip the solve; the belief replay above still runs (cheap, and keeps
        # range-quality logging meaningful) but nothing consumes ``last_search``.
        # (Vanilla Pluribus keeps ``search_enabled=True`` and DOES solve here.)
        if self._search_enabled:
            self._solve_and_store(copy.deepcopy(env))
        self.pending_actions = []

    def on_observed_action(self, env_before: PokerEnv, seat: int, action: str) -> None:
        """Record an observed action and re-search on an off-tree opponent raise.

        ``env_before`` must be the pre-action env where ``env_before.player_i ==
        seat`` (the caller deepcopies before stepping).  The action is buffered for
        the next boundary's Bayes update.  A (re-)search is a reaction to an
        **opponent's** action only — the bot's own action never triggers one, and a
        bot fold puts the agent dormant for the rest of the hand.  On rounds 2–4 a
        genuinely off-abstraction opponent raise (one the runtime injected into the
        shared overlay, so legal here yet off the canonical set) triggers a
        warm-started re-search of the same root **only when it is far enough off
        the canonical grid** (pot-fraction gap > ``offtree_threshold``); a
        near-canonical off-tree raise is instead translated onto the solved
        canonical branch at read time (§7), so it costs no re-solve.  On round 1 it
        may trigger a round-1 search.
        """
        if self._folded:
            return
        self.pending_actions.append((seat, env_before, action))
        if seat == self.my_seat:
            if action == "fold":
                self._folded = True
            return
        if not self._search_enabled:
            return                              # vanilla: never (re-)searches
        if self.last_search is not None:
            # Re-search only when an off-canonical opponent raise was actually
            # injected into the tree (present in ``legal_actions`` via the overlay,
            # yet off the canonical set) AND is far enough off-grid to matter — a
            # near-canonical size is translated (pseudo-harmonic) onto the existing
            # canonical branch by ``_solved_public_key`` instead of paying a full
            # re-solve.  Same root + ctx; the injection is visible via the overlay.
            if (
                self._is_off_tree(env_before, action)
                and action in env_before.legal_actions
                and self._offtree_gap(env_before, action) > self._offtree_threshold
            ):
                try:
                    self.last_search = solve(
                        self._root_env, self._ctx, self._cfg,
                        warm_start=self.last_search.state,
                    )
                except Exception:
                    # A warm re-search failure keeps the prior search: it is a valid
                    # solve of the same root (just without this off-tree branch), so
                    # it still plays better than the blueprint. Logged, not silent.
                    logger.exception(
                        "warm re-search failed (seat %s) — keeping the prior search",
                        self.my_seat,
                    )
        else:
            self._maybe_trigger_round1(env_before, action)

    def act(self, env: PokerEnv) -> str:
        """Return the action the bot plays at its current decision.

        Delegates to :meth:`play_distribution` — the single source of truth for the
        played σ (searched final-iteration strategy, or the blueprint fallback when
        the search has no strategy for this node) — samples it, and, when the play
        came from the search, pins the actual-hand row into the freeze map so a
        within-round re-search keeps it fixed (§5).
        """
        legal, prob, searched = self.play_distribution(env)
        action = self._sample(prob, legal)
        if searched:
            # Freeze the actual-hand row at the (normalised) mixed σ just played —
            # not the realised sample — so a within-round re-search keeps it pinned.
            pk = self._solved_public_key(env)
            hr = self._hand_row(env)
            st = self.last_search.state
            st.legal_at.setdefault(pk, tuple(legal))
            st.frozen[(pk, hr)] = prob
        return action

    def play_distribution(
        self, env: PokerEnv
    ) -> Tuple[List[str], np.ndarray, bool]:
        """The exact ``(legal, probs, searched)`` the bot plays at ``env``.

        The single source of truth for the played σ — shared by :meth:`act` and the
        runner's decision logging / AIVAT correction, which must agree with what was
        actually played.  Returns the searched strategy for the bot's actual hand
        (``searched=True``) whenever the search covers this node; the played σ is the
        raw search read, no blend toward the blueprint (a covered row is trained —
        traverser-vectorized MCCFR updates every root-street combo every iteration).

        The blueprint fallback fires **only** on a hard failure — the search produced
        **no** usable strategy for this decision: round 1 with no search, a failed
        solve (``last_search is None``), or a node the solved tree does not contain: a
        decision *past a depth-limit leaf* (the multiway within-round gap), or an
        off-tree line neither injected nor translatable (``_solved_public_key``'s
        key absent from ``legal_at``).  In every such case the bot plays the
        blueprint (``searched=False``) rather than a uniform guess over its legal
        actions (§6.6).
        """
        if self.last_search is not None:
            pk = self._solved_public_key(env)
            if pk in self.last_search.state.legal_at:
                hr = self._hand_row(env)
                legal = [a for a in env.legal_actions if a is not None]
                # OX-Search (Approach B, decision 5) plays the weighted-AVERAGE — every
                # §4.2 safety guarantee attaches to CFR's average, not the final iterate.
                # ``ox_enter_prob`` is set exactly when the gadget ran (vector regime +
                # β); vanilla/DBR leave it ``None`` and play the final iterate as before.
                played = (self.last_search.average_policy
                          if self.last_search.ox_enter_prob is not None
                          else self.last_search.policy)
                prob = np.asarray(
                    played.strategy_for(pk, hr, legal),
                    dtype=np.float64,
                )
                total = prob.sum()
                prob = prob / total if total > 0 else np.full(len(legal), 1.0 / len(legal))
                return legal, prob, True
        # Blueprint fallback (no search, failed solve, or a node the search does
        # not cover) — play the blueprint, never a uniform guess.
        state = env.policy_state_for(self.my_hole, for_blueprint=True)
        prob = np.asarray(self._blueprint.strategy(state, "none"), dtype=np.float64)
        return list(state.legal_actions), prob, False

    # ----------------------------------------------------------------- #
    # Public mechanism (a chip-aware runner may also call this directly)
    # ----------------------------------------------------------------- #
    def search_round1(self) -> None:
        """Run a round-1 subgame search and store it so ``act`` reads the search.

        Always roots at the pre-flop root snapshotted in :meth:`on_hand_start` (its
        ``street_at_root == 0`` makes the depth limit cut leaves at the end of
        round 1) — it takes no env argument because the round-1 root is fixed for
        the hand.  The *decision* of when to call this is the caller's; the agent
        provides a chip-free trigger in :meth:`_maybe_trigger_round1`.
        """
        self._solve_and_store(self._root_env)

    # ----------------------------------------------------------------- #
    # Belief update + solve
    # ----------------------------------------------------------------- #
    def _apply_boundary_belief_update(self) -> None:
        """Replay the round's buffered actions through the range tracker.

        Each ``(seat, env_before, action)`` is Bayes-applied under the strategy of
        the round just ending (the last search's average policy, or the blueprint
        for the round-1→2 boundary).  A fold is replayed *then* the seat is moved
        to the folded set, yielding the post-fold posterior for card removal.
        """
        for seat, env_before, action in self.pending_actions:
            sigma = self._make_sigma_for_combo(env_before, seat)
            self.tracker.on_action(seat, env_before, action, sigma)
            if action == "fold":
                self.tracker.on_seat_folded(seat)

    def _solve_and_store(self, root_env: PokerEnv) -> None:
        """Build the ctx from the current ranges, solve, and store the result.

        A solve that raises *for any reason* must not crash the hand: it is logged
        loudly and ``last_search`` is cleared, so :meth:`act` / :meth:`play_distribution`
        fall back to the blueprint for this round rather than propagating the error
        or reading a stale prior-round search (§6.6 robustness).
        """
        self._root_env = root_env
        self._ctx = SubgameContext.from_runtime(
            root_env,
            self.my_seat,
            self.my_hole,
            self.tracker.snapshot(),
            self.tracker.folded_snapshot(),
            self._cfg.leaf,
            self._rng,
            # The SAME hand-start snapshot the belief likelihood used (§6.3
            # invariant): infer the opponent's range under the strategy we clamp to.
            models=self._models,
        )
        try:
            self.last_search = solve(root_env, self._ctx, self._cfg)
        except Exception:
            logger.exception(
                "subgame solve failed (seat %s, street %s) — falling back to the "
                "blueprint for this round", self.my_seat, root_env.betting_round,
            )
            self.last_search = None
        self._searched_this_round = True

    def _make_sigma_for_combo(
        self, env_before: PokerEnv, seat: Optional[int] = None
    ) -> Callable[[int], np.ndarray]:
        """Closure mapping a combo row ``h`` → its action distribution at ``env_before``.

        Aligned to ``[a for a in env_before.legal_actions if a is not None]`` for
        the hypothetical hole ``env_before.combo_cards[h]`` — the contract
        :meth:`RangeTracker.on_action` requires.  ``ranges.py`` never imports
        ``policy.py``; the agent owns this bridge.

        **Per-seat (opponent_modeling §6.3, the belief-likelihood swap).**  When
        ``seat`` has a model in the hand-start snapshot, the likelihood is the model
        ``σ̂`` — *not* the solver's mixture ``σ̃``: beliefs estimate what the opponent
        **actually does**, while the mixture is only the solver's hedge.  This is
        design-doc §5 integration point 1 — reach beliefs and behavioral models
        agree, so the bot infers a modeled opponent's range under the same strategy
        it best-responds to.  Unmodeled seats and the bot's own range keep the
        baseline path below (last search's average, else the blueprint) unchanged.
        """
        model = self._models.get(int(seat)) if seat is not None else None
        if model is not None:
            combo_cards = env_before.combo_cards
            # Combo-independent fields computed once (this sweep runs per combo).
            public = env_before.policy_public_fields()

            def sigma_model(h: int) -> np.ndarray:
                # ``for_blueprint=True`` matches the blueprint path below and the
                # solver clamp, so an off-tree history canonicalises identically in
                # all three — the model is queried at one and the same info-set key.
                state = env_before.policy_state_for(
                    tuple(int(c) for c in combo_cards[h]),
                    for_blueprint=True,
                    public=public,
                )
                return np.asarray(model.strategy(state), dtype=np.float64)

            # The model row is a function of the info-set ``(cluster, history)``, so
            # combos sharing a cluster share it — dedupe the sweep by cluster (P1).
            return self._dedupe_by_cluster(env_before, combo_cards, sigma_model)

        if self._searched_this_round and self.last_search is not None:
            pk = self._solved_public_key(env_before)
            legal = [a for a in env_before.legal_actions if a is not None]
            avg = self.last_search.average_policy

            def sigma(h: int) -> np.ndarray:
                return np.asarray(avg.strategy_for(pk, h, legal), dtype=np.float64)

            return sigma

        combo_cards = env_before.combo_cards
        # The legal set + valid mask are combo-independent, so compute them once
        # for this env state rather than per combo (the belief-update sweep calls
        # ``sigma_blueprint`` ~1326×); only ``info_set`` varies per combo.
        public = env_before.policy_public_fields()

        def sigma_blueprint(h: int) -> np.ndarray:
            state = env_before.policy_state_for(
                tuple(int(c) for c in combo_cards[h]),
                for_blueprint=True,
                public=public,
            )
            return self._blueprint.strategy(state, "none")

        # info_set == (cluster, history) ⇒ every combo in a cluster returns the SAME
        # blueprint row, so collapse the per-combo LMDB sweep to one read per distinct
        # cluster (P1, mirroring the solver clamp's dedup — the vanilla belief win).
        return self._dedupe_by_cluster(env_before, combo_cards, sigma_blueprint)

    def _dedupe_by_cluster(
        self,
        env_before: PokerEnv,
        combo_cards: np.ndarray,
        row_fn: Callable[[int], np.ndarray],
    ) -> Callable[[int], np.ndarray]:
        """Memoise a per-combo policy read ``row_fn(h)`` by LUT cluster.

        The belief sweep queries a blueprint / model row for every combo still in a
        seat's range (:meth:`RangeTracker.on_action`), but that row is a function of
        the info-set ``(cluster, history)`` alone — the history is fixed at
        ``env_before`` and only the cluster varies with the hole.  So combos sharing a
        cluster share the row, and one read per distinct cluster suffices (P1, the same
        collapse the solver clamp does via :meth:`ClusterMapper.root_cluster_of`).  At
        production bucket counts this is ~``n_combos / n_buckets`` fewer LMDB reads.

        ``clusters_for_board`` is bit-exact with the cluster :meth:`PokerEnv.policy_state_for`
        embeds in the info-set (both are the same LUT combinadic lookup — the seam
        tests gate it), so the grouping is exact, not approximate.  Board-conflicting
        combos (cluster ``-1``) share no info-set key, so they bypass the cache and read
        directly; in practice they are never queried (they carry zero range mass, so
        ``on_action`` skips them).

        The cached row is returned by reference to multiple combos; callers must treat
        it read-only, which the :meth:`RangeTracker.on_action` contract already does
        (it reads a single action column and never mutates the vector).
        """
        clusters = clusters_for_board(
            env_before.card_info_lut[_STREET_NAME[env_before.betting_round]],
            combo_cards,
            np.asarray(env_before.community_cards, dtype=np.int64),
        )
        cache: Dict[int, np.ndarray] = {}

        def memoized(h: int) -> np.ndarray:
            cl = int(clusters[h])
            if cl < 0:                       # board-conflicting: no shared info-set
                return row_fn(h)
            row = cache.get(cl)
            if row is None:
                row = row_fn(h)
                cache[cl] = row
            return row

        return memoized

    # ----------------------------------------------------------------- #
    # Round-1 trigger (pot-relative fraction-gap; chip-free)
    # ----------------------------------------------------------------- #
    def _maybe_trigger_round1(self, env_before: PokerEnv, action: str) -> None:
        """Trigger a round-1 search on a sufficiently off-abstraction raise.

        Pot-relative analog of §5's "> $100 from every blueprint size": the
        observed off-tree raise fraction ``f_obs`` (= chips/pot, straight from the
        injected ``raise:<f_obs>`` string) must differ from *every* canonical
        abstraction fraction by more than ``round1_offtree_threshold`` (pot
        fractions), with no more than ``round1_max_players`` seats live.  Both
        conditions are chip-free.
        """
        if self.last_search is not None:
            return
        if not self._is_off_tree(env_before, action):
            return
        gap = self._offtree_gap(env_before, action)
        n_live = sum(1 for p in env_before.players if p.is_active)
        if gap > self._round1_threshold and n_live <= self._round1_max_players:
            self.search_round1()

    # ----------------------------------------------------------------- #
    # Pure helpers
    # ----------------------------------------------------------------- #
    def _is_off_tree(self, env_before: PokerEnv, action: str) -> bool:
        """True iff ``action`` is a raise off the blueprint (canonical) raise set.

        The solver branches on the full canonical abstraction (``legal_actions``),
        and the runtime injects exactly the raises that fall off it, so off-tree
        detection uses the same set: fold / call / check / ``all_in`` are always
        on-tree, and only a ``raise:<f>`` whose fraction is not a currently-playable
        canonical fraction is off-tree (and therefore worth a re-search).  Matches
        the ``"raise:"`` prefix specifically so a malformed bare ``"raise"`` is
        treated as on-tree rather than parsed downstream.
        """
        if not action.startswith("raise:"):
            return False
        canonical = {f"raise:{f}" for f in env_before.canonical_raise_fractions()}
        return action not in canonical

    def _offtree_gap(self, env_before: PokerEnv, action: str) -> float:
        """Pot-fraction distance from an off-tree ``raise:<f>`` to the nearest
        canonical size (``inf`` if no canonical raise is playable).

        The chip-free trigger shared by the round-1 and rounds-2–4 off-tree gates:
        a small gap means the size is near-canonical (translate), a large one means
        it is genuinely off-grid (inject + re-search).
        """
        f_obs = float(action.split(":", 1)[1])
        sizes = env_before.canonical_raise_fractions()
        return min((abs(f_obs - a) for a in sizes), default=float("inf"))

    def _solved_public_key(self, env: PokerEnv):
        """Public key under which to read the solved policy for ``env``'s node.

        Prefers the **raw** key: on-tree nodes and injected (re-searched) off-tree
        branches live in the solved tree under their exact history.  Falls back to
        the **pseudo-harmonic canonical** key (:attr:`PokerEnv.canonical_public_key`)
        so a *translated* — near-canonical, un-injected — off-tree raise resolves to
        the canonical branch the solver built.  If neither is present (e.g. a mixed
        line with both an injected and a translated off-tree raise), returns the raw
        key and the policy reader uniform-falls-back.
        """
        pk = env.public_key
        state = self.last_search.state
        if pk in state.legal_at:
            return pk
        canon = env.canonical_public_key
        return canon if canon in state.legal_at else pk

    def _sample(self, prob, legal: List[str]) -> str:
        """Sample a legal action from ``prob`` (renormalised in float64)."""
        if not legal:
            raise ValueError("SearchAgent._sample: no legal actions to sample from")
        prob = np.asarray(prob, dtype=np.float64)
        total = prob.sum()
        if total > 0:
            prob = prob / total
        else:
            prob = np.full(len(legal), 1.0 / len(legal))
        idx = int(self._rng.choice(len(legal), p=prob))
        return legal[idx]

    def _hand_row(self, env: PokerEnv) -> int:
        """Combo index of the bot's actual hand on the (current = root) street."""
        return int(env.combo_index[self.my_hole])
