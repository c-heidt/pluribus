"""Vector-form Linear CFR regime for the subgame solver (§6.5).

The vector regime is the *small / late* path: **heads-up turn/river** subgames.
It is the designed-in counterpart to :mod:`mccfr`, and persists into the **same**
:class:`SolverState` (and is read by the same :class:`SearchPolicy`) — only the
*storage shape* differs (per-public-node matrices, see below).

Design (per §6.5):

- Carry a **per-combo reach vector per player** (``reach[p] = ctx.ranges[p]`` at
  the root), not a sampled concrete hand.
- **Expand every action at every decision node** (no action sampling); **sample
  one river per iteration** at the river chance node (chance-sampled MCCFR),
  leaving the tree deterministic for that iteration.  The betting tree is hole-
  and card-independent, so a single make/undo env walk serves every combo at
  once; the engine's own river deal is **ignored** and a river sampled from the
  ranges' candidate set (via ``ctx.rng``) is used as that iteration's public
  river — the correct chance distribution over ranges, free of global-RNG
  dependence, and only one river's worth of work per iteration (not all ``R``).
- **River betting still conditions on the river card.**  Each candidate river has
  its own regret / strategy-sum slice (storage is ``(n_combos, n_rivers, width)``
  for river-stage nodes); an iteration touches only the **sampled** river's slice,
  so over iterations every river converges separately — the same river-conditioned
  equilibrium the exact enumeration reaches, estimated by chance sampling.
- Regret / strategy-sum updates are **reach-weighted per combo** (float64).  The
  per-combo counterfactual value carries the opponent reach (folded into the
  terminals, with card removal); the strategy sum carries the player's own reach.
  Alternating updates: each ``iterate`` runs one tree pass per seat.
- **Settlement is the env's job.**  At a terminal the regime calls
  :meth:`environment.poker_env.PokerEnv.vector_payout` — the env-owned vectorised
  payout — passing only the CFR quantities (the traverser seat, the opponent
  reach, and the sampled river).  The env handles the matched stake, showdown vs
  fold, board completion, card removal, and ranking (cached); this regime does no
  settlement and never imports the showdown primitives.

These per-combo rows persist into the **same** :class:`SolverState`: turn/river
**decision** nodes are ``(n_combos, width)`` matrices keyed by ``public_key`` whose
combo axis is ``combo_index`` (lossless at every depth) — so :class:`SearchPolicy`
reads them keyed ``(public_key, combo_index)`` with no changes.  A turn subgame's
internal **river-stage** nodes carry an extra river axis (``(n_combos, n_rivers,
width)``) for per-river conditioning; these are never read by the policy (the agent
re-solves a river subgame for the actual river), so the river axis is internal.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from poker_ai.search.context import SubgameContext
from poker_ai.search.solver_state import SolverConfig, SolverState


def _regret_match_matrix(regret: np.ndarray) -> np.ndarray:
    """Row-wise regret matching over the **last** axis of a regret tensor.

    Vectorised counterpart of
    :func:`poker_ai.blueprint.tree_utils.calculate_strategy_from_row`: each row's
    strategy is proportional to its positive cumulative regret, falling back to
    uniform over the ``width`` actions when a row has no positive regret.

    Action ``width`` is always the last axis, so this serves both the
    ``(n_combos, width)`` turn/river nodes and the river-conditioned
    ``(n_combos, n_rivers, width)`` nodes (§6.5) without reshaping.
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
    """Heads-up turn/river vector-form Linear CFR over a fixed subgame root.

    Constructed with the same ``(root_env, state, ctx, cfg, rng)`` surface as
    :class:`poker_ai.search.mccfr._MCCFRSolver` so :func:`solve` dispatches on the
    selected regime uniformly.  One :meth:`iterate` is two tree passes (one per
    seat, alternating updates); the orchestrator drives the iteration count, the
    Linear-CFR discount, and result extraction.
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

        # The river index sampled for the current iteration (set in ``iterate``);
        # ``None`` for a river subgame (no chance node) or before the first pass.
        self._sampled_k: Optional[int] = None

        # River chance node (§6.5): a turn subgame's river is an explicit chance
        # node, **sampled** one card per iteration (chance-sampled MCCFR) while
        # each river keeps its own conditioned regret/strategy slice.  A river
        # subgame has no chance node.
        if ctx.street_at_root >= 3:
            self._rivers: Optional[np.ndarray] = None
            self._feas: Optional[np.ndarray] = None
        else:
            board = set(int(c) for c in root_env.community_cards)
            self._rivers = np.array(
                [int(c) for c in np.unique(root_env.combo_cards) if int(c) not in board],
                dtype=np.int64,
            )
            # Per-(combo, river) feasibility: a combo cannot be held when the
            # river is one of its hole cards.  Board conflicts are already zeroed
            # in ``self._reach`` (board_compatible), so ``feas`` need only exclude
            # the river card itself.  Shape ``(n_combos, R)``, float for masking.
            cc = root_env.combo_cards
            rv = self._rivers
            conflict = (cc[:, 0:1] == rv[None, :]) | (cc[:, 1:2] == rv[None, :])
            self._feas = (~conflict).astype(np.float64)

    # ------------------------------------------------------------------
    # One iteration (§6.5 vector regime)
    # ------------------------------------------------------------------

    def iterate(self) -> None:
        # Chance-sampled river: a turn subgame's river chance node draws one
        # public river for the whole iteration (used wherever the river is
        # resolved this pass), so the cost is one river's work, not R.  Each
        # river still owns its conditioned regret/strategy slice, so over
        # iterations every river converges.  A river subgame has no chance node.
        self._sampled_k = (
            int(self.rng.integers(len(self._rivers)))
            if self._rivers is not None
            else None
        )
        s0, s1 = self._seats
        # Alternating updates: one full tree pass per traverser (river_k=None at
        # the root — turn betting / river-subgame nodes sit above the chance node).
        # ``_walk_env`` is the compiled FastState adapter under PLURIBUS_SEARCH_CORE
        # (else the PokerEnv root) — byte-identical, only make/undo speed differs.
        self._walk(self._walk_env, s0, self._reach[s0], self._reach[s1], None)
        self._walk(self._walk_env, s1, self._reach[s1], self._reach[s0], None)

    # ------------------------------------------------------------------
    # Recursion (always entered on a non-terminal node)
    # ------------------------------------------------------------------

    def _walk(self, env, p: int, pi_p: np.ndarray, pi_o: np.ndarray,
              river_k: Optional[int]) -> np.ndarray:
        """One CFR tree pass for traverser ``p``.

        Reach vectors are always ``(n_combos,)`` — the river chance node is
        *sampled*, so a single river is in play per iteration and the walk never
        carries a river axis.  ``river_k`` selects the storage slice: ``None``
        above the chance node (turn betting / river-subgame nodes are 2-D
        ``(n_combos, width)``), and the sampled river index below it (river-stage
        nodes are stored ``(n_combos, n_rivers, width)`` for per-river
        conditioning, but only the ``river_k`` slice — a 2-D view — is read and
        written this iteration).
        """
        actor = env.player_i
        legal = tuple(a for a in env.legal_actions if a is not None)
        pk = env.public_key
        below_chance = river_k is not None
        # River-stage nodes allocate a river axis (per-river conditioning); turn
        # / river-subgame nodes stay 2-D.
        n_rivers = (
            len(self._rivers) if (below_chance and self._rivers is not None) else None
        )
        self.state.ensure_vnode(pk, legal, actor, self._n_combos, n_rivers)
        # Below the chance node, operate on the sampled river's 2-D slice (a view,
        # so in-place ``+=`` writes back into the 3-D store); above it, the node is
        # already 2-D.
        if below_chance:
            regret = self.state.vregret[pk][:, river_k, :]
            strat = self.state.vstrat[pk][:, river_k, :]
        else:
            regret = self.state.vregret[pk]
            strat = self.state.vstrat[pk]
        sigma = _regret_match_matrix(regret)

        # Freezing (§5): the bot's pinned actual-hand row is substituted at **every**
        # visit to the bot's node — including when the opponent is the traverser, so
        # the bot's frozen reach propagates into ``pi_o`` (mirrors the MCCFR
        # ``_node_sigma`` / ``_frozen_or`` behaviour, which is traverser-agnostic).
        # Frozen rows are only ever set on the bot's *current-round* (turn / river)
        # decisions, never on a turn subgame's internal river-continuation nodes —
        # so in practice this fires only on 2-D nodes; the broadcast keeps it safe.
        apply_frozen = (
            actor == self._my_seat
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
                v += self._child(env, p, pi_p, pi_o * sigma[:, a_idx], river_k)
                env.undo(token)
            return v

        # Traverser node: expand all actions, weighting children by p's strategy.
        child_vs = np.empty((len(legal), self._n_combos), dtype=np.float64)
        for a_idx, action in enumerate(legal):
            token = env.step_in_place(action, settle_winners=False)
            child_vs[a_idx] = self._child(env, p, pi_p * sigma[:, a_idx], pi_o, river_k)
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
        regret += delta       # in-place; via the slice view this writes the store
        strat += strat_delta
        return v

    def _child(self, env, p: int, pi_p: np.ndarray, pi_o: np.ndarray,
               river_k: Optional[int]) -> np.ndarray:
        s0, s1 = self._seats
        opp = s1 if p == s0 else s0
        terminal = self.ctx.depth_limit.classify(env) == "terminal"

        if terminal:
            if river_k is not None:
                # Below the chance node: settle for the iteration's sampled river.
                return env.vector_payout(p, opp, pi_o, int(self._rivers[river_k]))
            if self._rivers is not None and self._terminal_needs_river(env):
                # The turn→river chance node *at* a terminal: a turn all-in
                # showdown (board completed by the river) or a river-side fold
                # reached directly.  Settle for the sampled river, feasibility-
                # masking the opponent reach for that river.
                k = self._sampled_k
                return env.vector_payout(
                    p, opp, pi_o * self._feas[:, k], int(self._rivers[k])
                )
            # Turn-side terminal (river-independent), or a river subgame terminal.
            return env.vector_payout(p, opp, pi_o, None)

        if (
            self._rivers is not None
            and river_k is None
            and env.betting_round > self.ctx.street_at_root
        ):
            # The turn→river chance node *before* river betting: descend into the
            # sampled river's slice, feasibility-masking BOTH reach vectors so
            # impossible (combo, river) rows never carry reach at interior river-
            # betting nodes (mandatory — the terminal showdown mask alone is not
            # enough).  No 1/R: chance sampling already gives the expectation.
            k = self._sampled_k
            feas_k = self._feas[:, k]
            return self._walk(env, p, pi_p * feas_k, pi_o * feas_k, k)

        return self._walk(env, p, pi_p, pi_o, river_k)

    # ------------------------------------------------------------------
    # River chance node helper (turn subgame only)
    # ------------------------------------------------------------------

    def _terminal_needs_river(self, env) -> bool:
        """Does the river chance affect this (turn-subgame) terminal?

        A showdown always integrates the river (it completes the board); a fold
        does so only if it is a genuine river-side fold (the real board reached
        five) — a turn-side fold is river-independent.
        """
        s0, s1 = self._seats
        if env.players[s0].is_active and env.players[s1].is_active:
            return True
        return env.terminal_board_len == 5
