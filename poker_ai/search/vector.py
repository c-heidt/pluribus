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


class _VectorSolver:
    """Heads-up flop/turn/river vector-form Linear CFR over a fixed subgame root.

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
            # identically, so condition A is not handicapped against B0 by engine.
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
        s0, s1 = self._seats
        # Alternating updates: one full tree pass per traverser.  ``_walk_env`` is
        # the compiled FastState adapter under PLURIBUS_SEARCH_CORE (else the
        # PokerEnv root) — same walk, only make/undo speed differs.
        self._walk(self._walk_env, s0, self._reach[s0], self._reach[s1])
        self._walk(self._walk_env, s1, self._reach[s1], self._reach[s0])

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
