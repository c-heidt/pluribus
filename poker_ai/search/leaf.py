"""Depth-limit leaf continuation-value evaluation (§6.4).

At a depth-limit leaf the subgame solver has already sampled a **concrete**
hand for every seat (the joint root draw, §6.5) and the continuation
meta-game has already fixed each active seat's bias class.  :func:`continuation_value`
evaluates that **fixed profile of concrete hands** by Monte-Carlo rollout to
terminal, returning the mean per-seat chip delta.

The leaf does **no** hole resampling — the hands are inherited from
``frontier_env`` and only the future board runout (and the sampled action
line) vary across rollouts.  When a rollout reaches an all-in showdown over an
incomplete board, the value of that decision-free runout is taken **exactly**
via :meth:`PokerEnv.runout_equity` (§6.4) rather than from a single sampled
board — gated by ``LeafConfig.use_decision_free_equity`` so the improvement can
be A/B-measured against the paper's sampled runout.

The module is a pure consumer of :class:`PokerEnv`, :class:`Policy`, and
:class:`SubgameContext`; ``ctx.rng`` drives the action sampling (the board
runout uses the env's global-``np.random`` deal, as elsewhere in the engine).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Tuple

import numpy as np

from environment.poker_env import PokerEnv
from poker_ai.search.context import SubgameContext
from poker_ai.search.policy import BiasClass, Policy


@dataclass
class LeafConfig:
    """Per-session leaf-evaluator config.

    Attributes
    ----------
    policies : Mapping[BiasClass, Policy]
        One policy per bias class — the four §4 continuation variants.
        :func:`continuation_value` consults ``policies[profile[seat]]`` for the
        acting seat.
    n_rollouts : int
        Monte-Carlo rollout count per :func:`continuation_value` call (default
        20).  Each rollout re-randomises the board runout; the hands are fixed.
    use_decision_free_equity : bool
        A/B toggle for the over-the-paper improvement (§6.4).  ``True`` (default)
        replaces an all-in showdown's single sampled board with the exact
        board-average via :meth:`PokerEnv.runout_equity`; ``False`` reproduces
        the paper's sampled single-board runout.  The same field is read by the
        solver (row 6) for its forced-runout terminals, so flipping it measures
        the whole effect.
    """

    policies: Mapping[BiasClass, Policy]
    n_rollouts: int = 20
    use_decision_free_equity: bool = True


def continuation_value(
    frontier_env: PokerEnv,
    profile: Mapping[int, BiasClass],
    ctx: SubgameContext,
) -> np.ndarray:
    """Expected per-seat chip delta of a fixed continuation profile (§6.4).

    Parameters
    ----------
    frontier_env : PokerEnv
        State at the depth-limit leaf.  Every seat already holds a concrete
        hole (the solver's root draw); an already-terminal env is handled by a
        fast path returning :attr:`PokerEnv.payout`.
    profile : Mapping[int, BiasClass]
        One bias class per seat that can still act, fixed by the continuation
        meta-game.  The acting seat plays ``ctx.leaf.policies[profile[seat]]``
        with ``bias=profile[seat]``.  A seat that acts but is absent raises
        ``ValueError``.
    ctx : SubgameContext
        Search context; ``ctx.rng`` is the sole RNG for action sampling and
        ``ctx.leaf`` carries the policy fleet, rollout count, and the
        decision-free-equity toggle.

    Returns
    -------
    numpy.ndarray
        Float64 vector of shape ``(frontier_env.n_players,)``; entry ``i`` is
        the mean chip delta for seat ``i`` over the rollouts.  ``n_rollouts ==
        0`` returns zeros.
    """
    n = frontier_env.n_players
    cfg = ctx.leaf
    rng = ctx.rng
    use_equity = cfg.use_decision_free_equity
    if frontier_env.is_terminal:
        # Defensive fast path (leaves are normally non-terminal).  Honour the
        # decision-free toggle here too so an already-resolved all-in frontier
        # is scored consistently with mid-rollout terminals.
        if use_equity and frontier_env.is_decision_free:
            eq = frontier_env.runout_equity(rng=rng)
            return np.array([eq[i] for i in range(n)], dtype=np.float64)
        return np.array(
            [float(frontier_env.payout[i]) for i in range(n)], dtype=np.float64
        )

    if cfg.n_rollouts <= 0:
        return np.zeros(n, dtype=np.float64)

    # Concrete holes are fixed; re-applying them via ``with_hole_cards`` per
    # rollout reshuffles only the undealt board, so each rollout draws a fresh
    # runout while the hands stay put.
    holes: List[Tuple[int, int]] = [
        tuple(int(c) for c in frontier_env.players[i].cards) for i in range(n)
    ]

    accum = np.zeros(n, dtype=np.float64)
    # Holes are fixed across the rollouts, so a decision-free runout's exact
    # board-average depends only on the env's pre-runout snapshot ``_runout_info``
    # (board prefix, frozen pot contributions, active mask) — identical for any
    # two rollouts that reach the same all-in.  Memoise on that snapshot so the
    # integration runs once per distinct all-in, not once per rollout.  In a leaf
    # the runout is always <=2 board cards (the exact path), so the cached value
    # is a deterministic function of the key.
    runout_cache: dict = {}
    for _ in range(cfg.n_rollouts):
        e = frontier_env.with_hole_cards(holes)
        while not e.is_terminal:
            seat = e.player_i
            if seat not in profile:
                raise ValueError(
                    f"continuation_value: profile is missing acting seat "
                    f"{seat}; it must cover every seat that can act."
                )
            c = profile[seat]
            # Blueprint lookups canonicalise off-tree histories (§6.3); the
            # actor plays its own concrete hole.
            state = e.policy_state_for(
                tuple(int(x) for x in e.current_player.cards), for_blueprint=True
            )
            probs = cfg.policies[c].strategy(state, bias=c)
            # Policy.strategy returns float32; cast + renormalise so
            # Generator.choice accepts the vector (float32 sums can drift
            # beyond its ~1e-8 tolerance).
            probs = np.asarray(probs, dtype=np.float64)
            probs /= probs.sum()
            idx = int(rng.choice(len(state.legal_actions), p=probs))
            e.step_in_place(state.legal_actions[idx])
        # Decision-free all-in showdown over an incomplete board: take the
        # exact board-average instead of the single dealt runout (§6.4),
        # unless the A/B toggle reproduces the sampled-runout baseline.
        if use_equity and e.is_decision_free:
            key = e._runout_info
            eq = runout_cache.get(key)
            if eq is None:
                eq = e.runout_equity(rng=rng)
                runout_cache[key] = eq
            for i in range(n):
                accum[i] += eq[i]
        else:
            for i in range(n):
                accum[i] += float(e.payout[i])
    return accum / cfg.n_rollouts
