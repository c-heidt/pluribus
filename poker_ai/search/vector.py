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

import itertools
from typing import Dict, Optional, Tuple

import numpy as np

from information_abstraction.lookup import clusters_for_board
from poker_ai.search.context import SubgameContext
from poker_ai.search.solver_state import SolverConfig, SolverState

# Street index -> LUT street key, for the future-street cluster lookups.
_STREET_NAME = {0: "pre_flop", 1: "flop", 2: "turn", 3: "river"}


def _regret_match_matrix(regret: np.ndarray) -> np.ndarray:
    """Row-wise regret matching over the **last** axis of a regret tensor.

    Vectorised counterpart of
    :func:`poker_ai.blueprint.tree_utils.calculate_strategy_from_row`: each row's
    strategy is proportional to its positive cumulative regret, falling back to
    uniform over the ``width`` actions when a row has no positive regret.

    Action ``width`` is always the last axis, so this serves both a
    root-street ``(n_combos, width)`` node and a future-street
    ``(n_clusters, width)`` node (§6.5) without reshaping.
    """
    pos = np.maximum(regret, 0.0)
    total = pos.sum(axis=-1, keepdims=True)
    width = regret.shape[-1]
    safe = np.where(total > 0.0, total, 1.0)
    return np.where(total > 0.0, pos / safe, 1.0 / width)


# Compiled-core wiring (Phase 1): when built AND enabled
# (``PLURIBUS_CORE_KERNELS`` includes ``regret_match_matrix``), swap the batched
# regret-matcher for its byte-identical Cython kernel.  ``_walk`` calls it as the
# module global ``_regret_match_matrix``, so the rebind is transparent; the
# pure-Python reference is kept as ``_regret_match_matrix_py`` (the oracle).
_regret_match_matrix_py = _regret_match_matrix
try:
    from poker_ai._core import CORE_AVAILABLE as _CORE_AVAILABLE
    from poker_ai._core.flags import kernel_enabled as _kernel_enabled

    if _CORE_AVAILABLE and _kernel_enabled("regret_match_matrix"):
        from poker_ai._core._regret import (
            calculate_strategy_matrix as _regret_match_matrix,
        )
except ImportError:
    pass


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

        # Future streets (§6.5): each is stored per LUT cluster.  ``street ==
        # street_at_root`` is lossless (combo rows); deeper streets fold the
        # sampled runout into the cluster id, so no explicit river axis is needed.
        self._street_at_root = ctx.street_at_root
        self._future = list(range(self._street_at_root + 1, 4))
        self._n_completion = len(self._future)          # 0 (river) / 1 (turn) / 2 (flop)

        self._combo_cards = np.asarray(root_env.combo_cards, dtype=np.int64)
        self._lut = root_env.card_info_lut
        self._root_comm = [int(c) for c in root_env.community_cards]
        if self._n_completion:
            avail = sorted(
                set(int(c) for c in np.unique(self._combo_cards)) - set(self._root_comm)
            )
            self._avail = np.array(avail, dtype=np.int64)
            # Deterministic per-street cluster universes (union over every
            # candidate completion) → identical local row layout across parallel
            # replicas, so ``SolverState.accumulate`` sums aligned rows.
            self._universe = self._build_universes()
        else:
            self._avail = np.empty(0, dtype=np.int64)
            self._universe = {}

        # Per-iteration cluster maps + scatter plans (filled in :meth:`iterate`).
        self._completion: Tuple[int, ...] = ()
        self._cluster_of: Dict[int, np.ndarray] = {}    # street -> (n_combos,) dense row, -1 infeasible
        self._feas: Dict[int, np.ndarray] = {}          # street -> (n_combos,) 0/1 board-feasibility
        self._scatter: Dict[int, Tuple] = {}            # street -> (sorted_combos, seg_starts, seg_cluster)
        self._feas_full: Optional[np.ndarray] = None    # feasibility for the full completion

    def _build_universes(self) -> Dict[int, np.ndarray]:
        """Sorted unique LUT cluster ids reachable at each future street.

        For street ``s`` the board is the root community plus the first
        ``s - street_at_root`` completion cards; the universe unions the clusters
        over **every** candidate completion of that depth, so the dense local row
        index (a ``searchsorted`` into this array) is fixed for the whole search
        and identical across replicas.  Bucket-count-agnostic: it reads only the
        ids the LUT actually produces here, never a cluster total.
        """
        universe: Dict[int, np.ndarray] = {}
        avail = self._avail.tolist()
        for s in self._future:
            name = _STREET_NAME[s]
            depth = s - self._street_at_root
            ids: set = set()
            for comp in itertools.combinations(avail, depth):
                board = np.array(self._root_comm + list(comp), dtype=np.int64)
                raw = clusters_for_board(self._lut[name], self._combo_cards, board)
                ids.update(int(x) for x in np.unique(raw[raw >= 0]))
            universe[s] = np.array(sorted(ids), dtype=np.int64)
        return universe

    def _refresh_cluster_maps(self, completion: Tuple[int, ...]) -> None:
        """Per-iteration: dense cluster row + feasibility + scatter plan per street.

        ``completion`` is the sampled runout (turn[, river]); at street ``s`` the
        board carries its first ``s - street_at_root`` cards.  The scatter plan
        pre-sorts the feasible combos by cluster row so a node's regret/strategy
        update is a ``reduceat`` segment-sum rather than an ``np.add.at``.
        """
        for s in self._future:
            name = _STREET_NAME[s]
            depth = s - self._street_at_root
            board = np.array(self._root_comm + list(completion[:depth]), dtype=np.int64)
            raw = clusters_for_board(self._lut[name], self._combo_cards, board)
            valid = raw >= 0
            dense = np.full(self._n_combos, -1, dtype=np.int64)
            dense[valid] = np.searchsorted(self._universe[s], raw[valid])
            self._cluster_of[s] = dense
            self._feas[s] = valid.astype(np.float64)
            fcombos = np.flatnonzero(valid)
            fclusters = dense[fcombos]
            order = np.argsort(fclusters, kind="stable")
            sorted_combos = fcombos[order]
            sorted_clusters = fclusters[order]
            seg_starts = np.concatenate(
                ([0], np.flatnonzero(np.diff(sorted_clusters)) + 1)
            ).astype(np.intp)
            seg_cluster = sorted_clusters[seg_starts]
            self._scatter[s] = (sorted_combos, seg_starts, seg_cluster)
        # Full-completion feasibility (deepest future street) — used to mask the
        # opponent reach at a showdown that completes the whole board at once.
        self._feas_full = self._feas[self._future[-1]]

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
                len(self._avail), size=self._n_completion, replace=False
            )
            self._completion = tuple(int(self._avail[i]) for i in idx)
            self._refresh_cluster_maps(self._completion)
        else:
            self._completion = ()
            self._feas_full = None
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
        street is clustered (``self._cluster_of[street]`` maps each combo to its
        dense cluster row).  For a clustered node the per-combo strategy is a
        gather of the cluster rows, and the per-combo regret/strategy deltas are
        scattered (segment-summed) back into the cluster rows.
        """
        street = env.betting_round
        actor = env.player_i
        legal = tuple(a for a in env.legal_actions if a is not None)
        pk = env.public_key
        is_root = street == self._street_at_root
        if is_root:
            n_rows, row_space, cof = self._n_combos, "combo", None
        else:
            cof = self._cluster_of[street]
            n_rows, row_space = len(self._universe[street]), "cluster"
        self.state.ensure_vnode(pk, legal, actor, n_rows, row_space)
        regret = self.state.vregret[pk]                 # (n_rows, width)
        strat = self.state.vstrat[pk]
        sigma_rows = _regret_match_matrix(regret)       # (n_rows, width)
        # Per-combo strategy: identity for a root node, a cluster gather otherwise
        # (infeasible combos map to row 0 — harmless, their reach is zeroed below).
        sigma = sigma_rows if is_root else sigma_rows[np.where(cof >= 0, cof, 0)]

        # Freezing (§5): the bot's pinned actual-hand row is substituted at every
        # visit to the bot's node.  Frozen rows are only ever set on the bot's
        # current (root-street) decisions, so this fires on combo nodes only.
        apply_frozen = (
            is_root
            and actor == self._my_seat
            and self._my_combo is not None
            and (pk, self._my_combo) in self.state.frozen
        )
        if apply_frozen:
            sigma[self._my_combo] = self.state.frozen[(pk, self._my_combo)]

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
        cv = np.moveaxis(child_vs, 0, -1)             # (n_combos, width)
        v = (sigma * cv).sum(axis=-1)                 # (n_combos,)
        delta = cv - v[:, None]                       # regret: v_a - v
        strat_delta = pi_p[:, None] * sigma           # strat-sum: own reach * sigma
        if apply_frozen:
            # The pinned actual-hand row neither accrues regret nor strategy (here
            # ``actor == p``, so this fires only when the bot traverses itself).
            delta[self._my_combo] = 0.0
            strat_delta[self._my_combo] = 0.0
        if is_root:
            regret += delta                           # in-place: writes the store
            strat += strat_delta
        else:
            # All combos in a cluster share one row (§6.5), so the row update is
            # the sum of their per-combo deltas — a presorted segment-sum.
            self._scatter_add(regret, delta, street)
            self._scatter_add(strat, strat_delta, street)
        return v

    def _scatter_add(self, table: np.ndarray, per_combo: np.ndarray, street: int) -> None:
        """Segment-sum the feasible combos' rows of ``per_combo`` into ``table``.

        The combo→cluster map is fixed for the whole iteration, so the sort +
        segment boundaries are precomputed once (``_refresh_cluster_maps``) and a
        ``reduceat`` does the grouping — cheaper than ``np.add.at`` on every node.
        ``seg_cluster`` holds each segment's distinct cluster row, so the final
        fancy-index add has no duplicate targets.
        """
        sorted_combos, seg_starts, seg_cluster = self._scatter[street]
        grouped = per_combo[sorted_combos]            # (n_feasible, width)
        seg_sums = np.add.reduceat(grouped, seg_starts, axis=0)
        table[seg_cluster] += seg_sums

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
                reach = pi_o if self._feas_full is None else pi_o * self._feas_full
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
            feas = self._feas[env.betting_round]
            return self._walk(env, p, pi_p * feas, pi_o * feas)

        return self._walk(env, p, pi_p, pi_o)
