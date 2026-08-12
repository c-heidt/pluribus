"""Vector-form Linear CFR regime for the subgame solver (§6.5).

The vector regime is the *small / late* path: **heads-up** subgames rooted on the
flop, turn, or river.  It is the designed-in counterpart to :mod:`mccfr`, and
persists into the **same** :class:`SolverState` (read by the same
:class:`SearchPolicy`) — only the *storage shape* differs (per-public-node
matrices, see below).

Design (per §6.5):

- Carry a **per-combo reach vector per player** (``reach[p] = ctx.ranges[p]`` at
  the root), not a sampled concrete hand.
- **Expand every action at every decision node** (no action sampling); **sample
  one board completion per iteration** at the chance nodes (chance-sampled CFR),
  leaving the tree deterministic for that iteration.  A flop root samples a
  *(turn, river)* pair; a turn root samples a river; a river root samples
  nothing.  The betting tree is hole- and card-independent, so a single make/undo
  env walk serves every combo at once; the engine's own board deal is **ignored**
  and the completion sampled from the deck (via ``ctx.rng``) is used as that
  iteration's public runout — the correct chance distribution over ranges, free
  of global-RNG dependence, and only one completion's work per iteration.

Storage — root street lossless, future streets clustered (§6.5):

- A **root-street** decision node is ``(n_combos, width)``, one row per
  ``combo_index`` (lossless).  These are the rows :class:`SearchPolicy` reads for
  the bot's actual hand, keyed ``(public_key, combo_index)`` — unchanged.
- A **future-street** decision node is ``(n_clusters, width)``, one row per LUT
  cluster reachable in the subgame.  The sampled board is folded into the cluster
  id (``cluster_for`` keys on the full community), so there is **no explicit river
  axis** — different runouts land in different clusters, and holes sharing a
  cluster share a row (the paper's lossy future-street abstraction).  The walk
  stays per-combo: it *gathers* each combo's cluster row to form a per-combo
  strategy, then *scatters* the per-combo regret/strategy deltas back into the
  cluster rows (a segment-sum).  These nodes are internal to the solve (the next
  round is a fresh subgame) and are never read externally — the ``vrow_space``
  guard in :class:`SolverState` enforces that.

- Regret / strategy-sum updates are **reach-weighted per combo** (float64).  The
  per-combo counterfactual value carries the opponent reach (folded into the
  terminals, with card removal); the strategy sum carries the player's own reach.
  Alternating updates: each ``iterate`` runs one tree pass per seat.
- **Settlement is the env's job.**  At a terminal the regime calls
  :meth:`environment.poker_env.PokerEnv.vector_payout`, passing the traverser
  seat, the opponent reach, and the sampled runout.  The env handles the matched
  stake, showdown vs fold, board completion, card removal, and ranking (cached);
  this regime does no settlement and never imports the showdown primitives.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from poker_ai.search.cluster_maps import ClusterMapper
from poker_ai.search.context import SubgameContext
from poker_ai.search.reference import compute_cbv_ref
from poker_ai.search.solver_state import SolverConfig, SolverState
from poker_ai.search.vform import (
    _regret_match_matrix_py,
    apply_model_clamp,
    freeze_combo,
    node_sigma,
    regret_match_matrix,
    traverser_update,
)

# The vector-form CFR kernel is shared with the MCCFR walk (:mod:`vform`).  It is
# re-exported here (``_regret_match_matrix`` / ``_regret_match_matrix_py``) for the
# compiled-kernel byte-parity tests and any caller that historically imported it
# from this module.
_regret_match_matrix = regret_match_matrix

# OX-Search opt-out row columns (Approach B gadget, §11.3 step 11): the opponent's
# per-combo choice to ENTER the subgame or take the CBV_ref alternative (OUT).
_OX_ENTER, _OX_OUT = 0, 1


def ox_optout_key(root_env):
    """The synthetic ``state.vregret`` key holding the OX-Search opt-out regret row.

    Distinct from every real node key (a ``(betting_stage:str, history:tuple)`` pair):
    here element 0 is itself the root public key and element 1 is a sentinel string.
    """
    return (root_env.public_key, "OX_OPTOUT")


def ox_enter_prob(state, root_env, board_compatible):
    """Opt-out saturation (Thm 4.5 guard): mean ENTER-probability over the feasible
    opponent infosets, read off the solved ``state``; ``None`` if no OX solve ran.

    ≈ 1 ⇒ the safety branch never opts out ⇒ β is too small (raise it).  Reads the
    final (serial or merged-replica) opt-out row, so it is regime- and worker-agnostic.
    """
    optr = state.vregret.get(ox_optout_key(root_env))
    if optr is None:
        return None
    feas = np.asarray(board_compatible, dtype=bool)
    if not feas.any():
        return float("nan")
    q = regret_match_matrix(optr)
    return float(q[feas, _OX_ENTER].mean())


class _VectorSolver:
    """Heads-up turn/river vector-form Linear CFR over a fixed subgame root.

    Production routes a HU **turn or river** subgame here (``_select_regime``): at
    most one future chance node remains, so the full-width per-combo walk is cheap.
    A HU **flop** root (two future chance nodes) is routed to sampled MCCFR instead —
    the vector walk still *supports* a flop root (the code below is street-general and
    the oracle harness drives it directly), it is just too slow full-width to route
    there in production.

    Constructed with the same ``(root_env, state, ctx, cfg, rng)`` surface as
    :class:`poker_ai.search.mccfr._MCCFRSolver` so :func:`solve` dispatches on the
    selected regime uniformly.  One :meth:`iterate` is two tree passes (one per
    seat, alternating updates); the orchestrator drives the iteration count, the
    Linear-CFR discount, and result extraction.

    Root-street decision nodes are stored lossless (one row per combo); future
    streets are stored per LUT cluster (§6.5), the sampled board folded into the
    cluster id.  The walk stays per-combo — it gathers each combo's cluster row to
    a per-combo strategy and scatters the per-combo deltas back into cluster rows.
    """

    def __init__(self, root_env, state: SolverState, ctx: SubgameContext,
                 cfg: SolverConfig, rng: np.random.Generator) -> None:
        self.root_env = root_env
        self.state = state
        self.ctx = ctx
        self.cfg = cfg
        self.rng = rng

        # Search Cython core (Phase 3): when ``PLURIBUS_SEARCH_CORE=1`` and the
        # search has no off-tree injections, drive the walk on the compiled
        # ``FastState`` betting engine (make/undo collapses out of Python).  The
        # Python ``_walk`` is unchanged and byte-identical either way; falls back
        # to the ``PokerEnv`` root when the flag is off, the core is unavailable,
        # or an off-tree action was injected (see ``fast_env``).
        self._walk_env = root_env
        try:
            from poker_ai._core.flags import search_core_enabled
            # Modeled solves run on the core too: the clamp keys the model by
            # *cluster* (``policy_state_for_cluster``), which both engines serve
            # identically, so DBR is not handicapped against vanilla Pluribus by engine.
            if search_core_enabled():
                from poker_ai.search.fast_env import build_fast_walk_env
                fast_env = build_fast_walk_env(root_env)
                if fast_env is not None:
                    self._walk_env = fast_env
        except ImportError:
            pass

        live = sorted(ctx.ranges)
        if len(live) != 2:
            raise ValueError(
                f"vector regime requires exactly two live seats, got {live}."
            )
        self._seats: Tuple[int, int] = (live[0], live[1])
        self._n_combos = int(root_env.combo_cards.shape[0])
        self._combo_cards = root_env.combo_cards   # (n_combos, 2), for the model clamp

        # Per-seat board-masked reach (float64 copies of the §6.2 ranges).
        bc = np.asarray(ctx.board_compatible, dtype=np.float64)
        self._reach: Dict[int, np.ndarray] = {
            s: np.asarray(ctx.ranges[s], dtype=np.float64) * bc for s in self._seats
        }
        # Fail fast on a degenerate range (mirrors the MCCFR regime's guard): a
        # seat with no board-compatible reach would otherwise silently yield
        # all-zero regrets and a junk uniform strategy.
        for s in self._seats:
            if self._reach[s].sum() <= 0.0:
                raise ValueError(
                    f"vector regime: seat {s} has zero board-compatible reach."
                )

        # Bot's actual-hand combo row + seat (freezing, §5).
        my = tuple(sorted(int(c) for c in ctx.my_hole))
        self._my_combo: Optional[int] = root_env.combo_index.get(my)
        self._my_seat = ctx.my_seat

        # Root-value convergence signal (calibration): the hero's per-combo root
        # counterfactual value, normalised by the opponent's board-masked reach mass
        # so it reads as the played hand's conditional EV vs the belief opponent
        # (chips).  Tracked only when the bot is a live seat holding a combo row; a
        # write-only side counter that never touches a table or the RNG stream.
        self._val_iter = 0
        self._opp_mass: Optional[float] = None
        if self._my_seat in self._seats and self._my_combo is not None:
            opp = self._seats[0] if self._seats[1] == self._my_seat else self._seats[1]
            m = float(self._reach[opp].sum())
            self._opp_mass = m if m > 0.0 else None

        # Future streets (§6.5): the per-LUT-cluster storage machinery — the
        # deterministic per-street cluster universes, the per-iteration dense
        # combo→cluster rows / board feasibility / scatter plans — is shared with the
        # MCCFR walk via :class:`ClusterMapper`.  ``street == street_at_root`` is
        # lossless (combo rows); deeper streets fold the sampled runout into the
        # cluster id, so there is no explicit river axis.
        self._street_at_root = ctx.street_at_root
        self._cmaps = ClusterMapper(
            root_env.card_info_lut, root_env.combo_cards,
            root_env.community_cards, self._street_at_root,
        )
        self._n_completion = self._cmaps.n_completion   # 0 (river) / 1 (turn) / 2 (flop)
        # The sampled board runout for the current iteration (filled in :meth:`iterate`).
        self._completion: Tuple[int, ...] = ()

        # OX-Search gadget root (Approach B, §11.3 step 11): OFF unless ``cfg.beta``
        # is set, in which case the solve is byte-for-byte the vanilla/DBR walk.  A
        # finite β swaps the opponent's fixed root reach for the belief/opt-out
        # gadget mix and adds the CBV_ref-anchored opt-out row (see :meth:`_ox_setup`).
        self._ox = getattr(cfg, "beta", None) is not None
        if self._ox:
            # Defense in depth for the adaptation-safety guarantee (Thm 4.3): the
            # in-subgame opponent must be a true regret-matching adversary, with the
            # DBR model entering ONLY via the reach belief p̂ at the gadget root
            # (docs/opponent_modeling.md Part II — no confidence/mixture machinery
            # inside S). `_walk` calls `apply_model_clamp` unconditionally, so a
            # populated `ctx.models` would silently blend the opponent's in-subgame
            # strategy toward its model at every interior node, invalidating the
            # safety proof with no crash. The production driver (runner.py's
            # `EvalConfig.for_condition`) already rejects OX+model_spec combinations
            # upstream — this raises here too so the invariant can't be silently
            # violated by a future/alternate caller that skips that guard.
            if getattr(ctx, "models", None):
                raise ValueError(
                    "OX-Search (cfg.beta set) requires ctx.models to be empty — the "
                    "gadget's safety guarantee assumes a model-free adversarial "
                    "opponent inside the subgame; got models for seats "
                    f"{sorted(ctx.models)}."
                )
            self._ox_setup()

    # ------------------------------------------------------------------
    # OX-Search gadget root (Approach B — adaptation-safe exploitation, §11.3)
    # ------------------------------------------------------------------

    def _ox_setup(self) -> None:
        """Prepare the gadget-root references (fixed for the solve).

        Computes the belief entry distribution ``p̂``, the exploitation/safety chance
        coefficients from ``(k, β)``, and the exact ``CBV_ref`` pass (anchored to the
        blueprint — the locked decision), and allocates the per-combo opt-out regret
        row **inside** ``state.vregret`` so it is Linear-CFR discounted and
        cross-replica accumulated exactly like every other regret table.
        """
        beta = float(self.cfg.beta)
        if beta < 0.0:
            raise ValueError(f"OX-Search beta must be >= 0, got {beta}.")
        bot = int(self.ctx.my_seat)
        if bot not in self._seats:
            raise ValueError(f"OX-Search: bot seat {bot} not live ({self._seats}).")
        self._ox_bot = bot
        self._ox_opp = self._seats[0] if self._seats[1] == bot else self._seats[1]

        bc = np.asarray(self.ctx.board_compatible, dtype=np.float64)
        self._ox_bc = bc
        # k = board-compatible opponent root combos (root street lossless ⇒ cluster ==
        # combo).  The uniform-1/k safety entry folds into ``c_safe`` (= kβ/(kβ+1)·1/k),
        # so k enters ONLY through the two chance coefficients.
        k = float(bc.sum())
        if k <= 0.0:
            raise ValueError("OX-Search: no board-compatible root combos (k = 0).")
        denom = k * beta + 1.0
        self._ox_c_expl = 1.0 / denom          # exploitation branch: chance, entry ∝ p̂
        self._ox_c_safe = beta / denom         # safety branch: chance·(1/k), opt-out q

        # p̂: the opponent's believed entry distribution (the A4 belief lives in
        # ctx.ranges[opp]); board-masked and normalised to a distribution over the k
        # feasible infosets so it balances against the uniform-1/k safety entry.
        phat = self._reach[self._ox_opp].copy()      # already ranges[opp] * bc
        tot = phat.sum()
        self._ox_phat = phat / tot if tot > 0.0 else phat

        # CBV_ref: the opponent's exact best-response counterfactual value vs the bot
        # playing the BLUEPRINT, per opponent root combo — the opt-out alternative
        # payoff.  Fixed for the solve (a property of blueprint + subgame), re-anchored
        # to the blueprint every solve.  (At workers > 1 each replica recomputes this
        # deterministic pass; hoisting it ahead of the fan-out is a later perf lever.)
        self._ox_cbv = np.asarray(
            compute_cbv_ref(self.root_env, self.ctx, self.cfg), dtype=np.float64
        )

        # Per-combo opt-out regret row (width 2: [enter, out]) as a combo-keyed root
        # node in state.vregret under a synthetic key, so state.discount / accumulate
        # sweep it uniformly with the real tables.  Reused on a warm re-search.
        self._ox_optout_key = ox_optout_key(self.root_env)
        if self._ox_optout_key not in self.state.vregret:
            self.state.vregret[self._ox_optout_key] = np.zeros(
                (self._n_combos, 2), dtype=np.float64
            )
            self.state.vrow_space[self._ox_optout_key] = "combo"
        # Opt-out saturation (Thm 4.5 guard): mean enter-prob over the feasible
        # infosets, refreshed each iteration; ≈ 1 ⇒ the safety branch never opts out
        # ⇒ raise β.  Read off the solver by the agent/eval (step 12).
        self.ox_enter_prob = float("nan")

    def _iterate_ox(self) -> None:
        """One gadget-root iteration (§11.3 step 11).

        The walk itself is unchanged: the opponent plays its adversarial regret-matched
        strategy inside the subgame (reach-only exploitation).  Only the opponent's
        **root entry reach** becomes the gadget mix ``c_expl·p̂ + c_safe·q_enter``, and
        the opt-out row is updated from the opponent's per-combo subgame CFV vs
        ``CBV_ref`` (weighted by the safety-branch chance ``c_safe``).  The per-infoset
        ``CBV_ref`` shift cancels in every interior/bot regret, so it appears ONLY here.
        """
        bot, opp = self._ox_bot, self._ox_opp
        optr = self.state.vregret[self._ox_optout_key]      # (n_combos, 2)
        q = regret_match_matrix(optr)                        # [:,0]=enter, [:,1]=out
        q_enter = q[:, _OX_ENTER]
        opp_entry = (self._ox_c_expl * self._ox_phat
                     + self._ox_c_safe * q_enter) * self._ox_bc
        # Bot best-responds to the gadget-weighted opponent (bot regrets update here).
        v_bot = self._walk(self._walk_env, bot, self._reach[bot], opp_entry)
        # Calibration root-value signal: the bot's conditional EV vs the gadget-weighted
        # opponent, normalised by the entry mass its walk actually used (the OX analogue
        # of the vanilla ``_opp_mass``).  Pure side-effect; solve stays byte-identical.
        self._record_root_value(bot, v_bot, opp_mass=float(opp_entry.sum()))
        # Opponent adapts; its per-combo subgame root value is the opt-out ENTER value.
        v_enter = self._walk(self._walk_env, opp, opp_entry, self._reach[bot])
        # Opt-out CFR update: ENTER → subgame CFV, OUT → CBV_ref; counterfactual weight
        # is the safety-branch chance reach ``c_safe``.  Infeasible combos have
        # v_enter == CBV_ref == 0, so their rows never move.
        v_out = self._ox_cbv
        node_v = q_enter * v_enter + q[:, _OX_OUT] * v_out
        optr[:, _OX_ENTER] += self._ox_c_safe * (v_enter - node_v)
        optr[:, _OX_OUT] += self._ox_c_safe * (v_out - node_v)
        feas = self._ox_bc > 0.0
        self.ox_enter_prob = float(q_enter[feas].mean()) if feas.any() else float("nan")

    # ------------------------------------------------------------------
    # One iteration (§6.5 vector regime)
    # ------------------------------------------------------------------

    def iterate(self) -> None:
        # Chance-sampled runout: draw the whole board completion once for this
        # iteration (a (turn, river) pair from a flop root, a river from a turn
        # root, nothing from a river root).  Sampling without replacement over the
        # available deck gives the uniform chance measure over ordered runouts.
        if self._n_completion:
            idx = self.rng.choice(
                len(self._cmaps.avail), size=self._n_completion, replace=False
            )
            self._completion = tuple(int(self._cmaps.avail[i]) for i in idx)
            self._cmaps.refresh(self._completion)
        else:
            self._completion = ()
        if self._ox:
            # OX-Search: the opponent enters via the belief/opt-out gadget mix
            # (bot pass then opponent pass, sharing the sampled completion above).
            self._iterate_ox()
            return
        s0, s1 = self._seats
        # Alternating updates: one full tree pass per traverser.  ``_walk_env`` is
        # the compiled FastState adapter under PLURIBUS_SEARCH_CORE (else the
        # PokerEnv root) — same walk, only make/undo speed differs.
        v0 = self._walk(self._walk_env, s0, self._reach[s0], self._reach[s1])
        v1 = self._walk(self._walk_env, s1, self._reach[s1], self._reach[s0])
        if self._opp_mass is not None:
            self._record_root_value(s0, v0)
            self._record_root_value(s1, v1)

    def _record_root_value(self, seat: int, v: np.ndarray,
                           opp_mass: "float | None" = None) -> None:
        """Accumulate the hero's linearly-weighted root-EV estimate (calibration).

        Fires only for the bot's seat holding a combo row; ``v`` is the root per-combo
        counterfactual value the walk just returned.  Linear (Linear-CFR-matching)
        weighting downweights the noisy early iterates.  Dividing by the opponent reach
        mass **actually used in that walk** turns the counterfactual value into the
        played hand's conditional EV (chips), so it is comparable across budgets and
        interpretable in mbb.  ``opp_mass`` defaults to the fixed belief reach mass
        ``self._opp_mass`` (vanilla/DBR); the OX gadget passes its per-iteration entry
        mass (``opp_entry.sum()``) since that is the reach its bot walk was weighted by.
        A pure side-effect: reads one already-computed float, writes two counters, and
        never advances the RNG or a table (solve stays byte-identical).
        """
        if seat != self._my_seat or self._my_combo is None:
            return
        om = self._opp_mass if opp_mass is None else opp_mass
        if om is None or om <= 0.0:
            return
        self._val_iter += 1
        w = float(self._val_iter)
        self.state.root_value_num += w * float(v[self._my_combo]) / om
        self.state.root_value_den += w

    # ------------------------------------------------------------------
    # Recursion (always entered on a non-terminal node)
    # ------------------------------------------------------------------

    def _walk(self, env, p: int, pi_p: np.ndarray, pi_o: np.ndarray) -> np.ndarray:
        """One CFR tree pass for traverser ``p``.

        Reach vectors are always ``(n_combos,)``.  The node's row space is set by
        its street: the root street is lossless (row == ``combo_index``); a future
        street is clustered (``self._cmaps.cluster_of(street)`` maps each combo to
        its dense cluster row).  For a clustered node the per-combo strategy is a
        gather of the cluster rows, and the per-combo regret/strategy deltas are
        scattered (segment-summed) back into the cluster rows.
        """
        street = env.betting_round
        actor = env.player_i
        legal = tuple(a for a in env.legal_actions if a is not None)
        pk = env.public_key
        is_root = street == self._street_at_root
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
            self._cmaps, street,
        )
        frozen_combo = freeze_combo(
            self.state, pk, sigma, is_root, actor, self._my_seat, self._my_combo
        )

        if actor != p:
            # Opponent node: expand all actions, fold opp mixing into pi_o, and
            # SUM the children (no regret/strategy update when traversing p).
            v = np.zeros_like(pi_o)
            for a_idx, action in enumerate(legal):
                # settle_winners=False: this regime values terminals over ranges
                # (vector_payout), so the env's concrete hand ranking + chip
                # distribution at a terminal is discarded — skip it (§6.5).
                token = env.step_in_place(action, settle_winners=False)
                v += self._child(env, p, pi_p, pi_o * sigma[:, a_idx], street)
                env.undo(token)
            return v

        # Traverser node: expand all actions, weighting children by p's strategy.
        child_vs = np.empty((len(legal), self._n_combos), dtype=np.float64)
        for a_idx, action in enumerate(legal):
            token = env.step_in_place(action, settle_winners=False)
            child_vs[a_idx] = self._child(env, p, pi_p * sigma[:, a_idx], pi_o, street)
            env.undo(token)
        # All combos in a cluster share one row (§6.5), so a future-street update is
        # a presorted segment-sum; a root node adds combo rows directly.
        scatter = None if is_root else (
            lambda t, pc: self._cmaps.scatter_add(t, pc, street)
        )
        return traverser_update(
            regret, strat, sigma, child_vs, pi_p, frozen_combo, scatter
        )

    def _child(self, env, p: int, pi_p: np.ndarray, pi_o: np.ndarray,
               parent_street: int) -> np.ndarray:
        s0, s1 = self._seats
        opp = s1 if p == s0 else s0

        if self.ctx.depth_limit.classify(env) == "terminal":
            runout = self._completion or None
            if env.players[s0].is_active and env.players[s1].is_active:
                # Showdown: the board completes to five with the whole sampled
                # runout.  Mask pi_o by full-completion feasibility so an all-in
                # that reached the terminal *before* the chance nodes were walked
                # still excludes combos holding a completion card (mirrors the
                # per-crossing masking below).  No-op for a river root.
                reach = (pi_o if self._cmaps.feas_full is None
                         else pi_o * self._cmaps.feas_full)
                return env.vector_payout(p, opp, reach, runout=runout)
            # Fold: pi_o is already masked to the fold's street by the crossings
            # below; vector_payout's fold path selects the board it saw via
            # terminal_board_len.
            return env.vector_payout(p, opp, pi_o, runout=runout)

        if env.betting_round > parent_street:
            # Crossed a chance node (the board grew by one card): mask both reaches
            # by the new street's board feasibility, so a combo holding the freshly
            # dealt completion card carries no reach below it.  No 1/R — chance
            # sampling already gives the expectation.
            feas = self._cmaps.feas(env.betting_round)
            return self._walk(env, p, pi_p * feas, pi_o * feas)

        return self._walk(env, p, pi_p, pi_o)
