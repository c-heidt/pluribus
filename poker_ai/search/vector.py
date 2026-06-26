"""Vector-form Linear CFR regime for the subgame solver (§6.5).

The vector regime is the *small / late* path: **heads-up turn/river** subgames.
It is the designed-in counterpart to :mod:`mccfr`, and persists into the **same**
:class:`SolverState` (and is read by the same :class:`SearchPolicy`) — only the
*storage shape* differs (per-public-node matrices, see below).

Design (per §6.5):

- Carry a **per-combo reach vector per player** (``reach[p] = ctx.ranges[p]`` at
  the root), not a sampled concrete hand.
- **Expand every action at every decision node** (no action sampling); **sample
  one board runout per iteration** at the chance node (the river), leaving the
  tree deterministic for that iteration.  The betting tree is hole- and
  card-independent, so a single make/undo env walk serves every combo at once;
  the engine's own river deal is **ignored** and a river sampled from the ranges'
  candidate set (via ``ctx.rng``) is used for the showdown board instead — the
  correct chance distribution over ranges, and free of global-RNG dependence.
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

These per-combo rows persist into the **same** :class:`SolverState`, as
``(n_combos, width)`` matrices keyed by ``public_key`` whose combo axis is
``combo_index`` (lossless at every depth) — so :class:`SearchPolicy` reads a
vector-regime result keyed ``(public_key, combo_index)`` with no changes.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from poker_ai.search.context import SubgameContext
from poker_ai.search.solver_state import SolverConfig, SolverState


def _regret_match_matrix(regret: np.ndarray) -> np.ndarray:
    """Row-wise regret matching over a ``(n_combos, width)`` regret matrix.

    Vectorised counterpart of
    :func:`poker_ai.blueprint.tree_utils.calculate_strategy_from_row`: each row's
    strategy is proportional to its positive cumulative regret, falling back to
    uniform over the ``width`` actions when a row has no positive regret.
    """
    pos = np.maximum(regret, 0.0)
    total = pos.sum(axis=1, keepdims=True)
    width = regret.shape[1]
    safe = np.where(total > 0.0, total, 1.0)
    return np.where(total > 0.0, pos / safe, 1.0 / width)


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

        # Bot's actual-hand combo row + seat (freezing, §5).
        my = tuple(sorted(int(c) for c in ctx.my_hole))
        self._my_combo: Optional[int] = root_env.combo_index.get(my)
        self._my_seat = ctx.my_seat

        # River sampling: a turn subgame samples one of the candidate (non-board)
        # cards per iteration; a river subgame has no chance node.
        if ctx.street_at_root >= 3:
            self._rivers: Optional[np.ndarray] = None
        else:
            board = set(int(c) for c in root_env.community_cards)
            self._rivers = np.array(
                [int(c) for c in np.unique(root_env.combo_cards) if int(c) not in board],
                dtype=np.int64,
            )
        self._river: Optional[int] = None

    # ------------------------------------------------------------------
    # One iteration (§6.5 vector regime)
    # ------------------------------------------------------------------

    def iterate(self) -> None:
        # Sample one river runout for this iteration (turn subgames only).
        if self._rivers is not None:
            self._river = int(self.rng.choice(self._rivers))
        else:
            self._river = None
        s0, s1 = self._seats
        # Alternating updates: one full tree pass per traverser.
        self._walk(self.root_env, s0, self._reach[s0], self._reach[s1])
        self._walk(self.root_env, s1, self._reach[s1], self._reach[s0])

    # ------------------------------------------------------------------
    # Recursion (always entered on a non-terminal node)
    # ------------------------------------------------------------------

    def _walk(self, env, p: int, pi_p: np.ndarray, pi_o: np.ndarray) -> np.ndarray:
        actor = env.player_i
        legal = tuple(a for a in env.legal_actions if a is not None)
        pk = env.public_key
        self.state.ensure_vnode(pk, legal, actor, self._n_combos)
        regret = self.state.vregret[pk]
        sigma = _regret_match_matrix(regret)

        # Freezing (§5): the bot's pinned actual-hand row is substituted at **every**
        # visit to the bot's node — including when the opponent is the traverser, so
        # the bot's frozen reach propagates into ``pi_o`` (mirrors the MCCFR
        # ``_node_sigma`` / ``_frozen_or`` behaviour, which is traverser-agnostic).
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
            v = np.zeros(self._n_combos, dtype=np.float64)
            for a_idx, action in enumerate(legal):
                token = env.step_in_place(action)
                v += self._child(env, p, pi_p, pi_o * sigma[:, a_idx])
                env.undo(token)
            return v

        # Traverser node: expand all actions, weighting children by p's strategy.
        child_vs = np.empty((len(legal), self._n_combos), dtype=np.float64)
        for a_idx, action in enumerate(legal):
            token = env.step_in_place(action)
            child_vs[a_idx] = self._child(env, p, pi_p * sigma[:, a_idx], pi_o)
            env.undo(token)
        v = (sigma.T * child_vs).sum(axis=0)          # (n_combos,)
        delta = child_vs.T - v[:, None]               # regret: v_a - v
        strat_delta = pi_p[:, None] * sigma           # strat-sum: own reach * sigma
        if apply_frozen:
            # The pinned actual-hand row neither accrues regret nor strategy (here
            # ``actor == p``, so this fires only when the bot traverses itself).
            delta[self._my_combo] = 0.0
            strat_delta[self._my_combo] = 0.0
        regret += delta
        self.state.vstrat[pk] += strat_delta
        return v

    def _child(self, env, p: int, pi_p: np.ndarray, pi_o: np.ndarray) -> np.ndarray:
        if self.ctx.depth_limit.classify(env) == "terminal":
            # Terminal value is the env-owned vectorised payout: per-combo value to
            # ``p`` against the opponent's reach on the sampled river.  All
            # settlement (stake, showdown/fold, board, card removal) is the env's.
            s0, s1 = self._seats
            opp = s1 if p == s0 else s0
            return env.vector_payout(p, opp, pi_o, self._river)
        return self._walk(env, p, pi_p, pi_o)
