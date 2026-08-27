"""Depth-limit leaf continuation-value evaluation (§6.4).

At a depth-limit leaf the solver has already sampled a **concrete** hand for every
seat and the continuation meta-game has fixed each active seat's bias class.
:func:`continuation_value` evaluates that fixed profile by Monte-Carlo rollout to
terminal, returning the mean per-seat chip delta.

The leaf does **no** hole resampling — hands are inherited from ``frontier_env``, and
only the sampled action line and board runout vary.  Exactly **one** action line is
played on one sampled board, matching the paper (stochasticity is recovered in
aggregate across CFR's per-iteration hole draws, not by per-leaf averaging).  A
decision-free board-average and a multi-rollout average were both tried and dropped on
perf, so this path and the compiled core agree exactly.

A pure consumer of :class:`PokerEnv`, :class:`Policy` and :class:`SubgameContext`.
``ctx.rng`` drives action sampling, ``ctx.board_rng`` the board runout — separate
streams, and **neither is the global** ``np.random``: a rollout board is a hypothetical
re-deal, so drawing it from the played game's stream would couple the search's
randomness to unrelated consumers.  See :mod:`poker_ai.search.rng`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Tuple

import numpy as np

from environment.poker_env import PokerEnv
from poker_ai.blueprint.tree_utils import sample_index
from poker_ai.search.context import SubgameContext
from poker_ai.search.policy import BiasClass, Policy


def board_rng_for(ctx) -> np.random.Generator:
    """The stream a leaf rollout draws its board runout from.

    ``ctx.board_rng`` when the context carries one (the norm —
    :meth:`SubgameContext.from_runtime` always derives it), else ``ctx.rng``.
    Never the global ``np.random``: see the module docstring.

    Takes a duck-typed ``ctx`` because the leaf machinery is reused outside the
    solver by lightweight carriers (e.g. ``evaluation.aivat._LeafCtx``).
    """
    rng = getattr(ctx, "board_rng", None)
    return ctx.rng if rng is None else rng


@dataclass
class LeafConfig:
    """Per-session leaf-evaluator config.

    Attributes
    ----------
    policies : Mapping[BiasClass, Policy]
        One policy per bias class — the four §4 continuation variants.
        :func:`continuation_value` consults ``policies[profile[seat]]`` for the
        acting seat.
    """

    policies: Mapping[BiasClass, Policy]


def continuation_value(
    frontier_env: PokerEnv,
    profile: Mapping[int, BiasClass],
    ctx: SubgameContext,
) -> np.ndarray:
    """Expected per-seat chip delta of a fixed continuation profile (§6.4).

    Parameters
    ----------
    frontier_env : PokerEnv
        State at the depth-limit leaf; every seat already holds a concrete hole.  An
        already-terminal env takes a fast path returning :attr:`PokerEnv.payout`.
    profile : Mapping[int, BiasClass]
        One bias class per seat that can still act.  The acting seat plays
        ``ctx.leaf.policies[profile[seat]]``; a seat that acts but is absent raises.
    ctx : SubgameContext
        ``ctx.rng`` drives action sampling, ``ctx.leaf`` carries the policy fleet.

    Returns
    -------
    numpy.ndarray
        Float64 ``(n_players,)`` chip deltas on the single sampled continuation.
    """
    n = frontier_env.n_players
    cfg = ctx.leaf
    rng = ctx.rng
    if frontier_env.is_terminal:
        # Defensive fast path (leaves are normally non-terminal).
        return np.array(
            [float(frontier_env.payout[i]) for i in range(n)], dtype=np.float64
        )

    # Concrete holes are fixed at the leaf; play one sampled action line to a
    # terminal on one sampled board.
    holes: List[Tuple[int, int]] = [
        tuple(int(c) for c in frontier_env.players[i].cards) for i in range(n)
    ]
    e = frontier_env.with_hole_cards(holes, rng=board_rng_for(ctx))
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
        # Single inverse-CDF draw over the (short, per-node) strategy vector;
        # sums in float64 internally and scales by the total, so the float32
        # σ needs no copy-and-renormalise (the old Generator.choice path did).
        idx = sample_index(rng, probs)
        e.step_in_place(state.legal_actions[idx])
    return np.array([float(e.payout[i]) for i in range(n)], dtype=np.float64)


def continuation_value_vector(
    frontier_env: PokerEnv,
    profile: Mapping[int, BiasClass],
    ctx: SubgameContext,
    traverser_seat: int,
) -> np.ndarray:
    """Per-traverser-combo continuation value of a fixed profile (§6.4, vectorized).

    The vector twin of :func:`continuation_value` for the traverser-vectorized
    MCCFR walk: ``traverser_seat``'s hand is swept over **all** combos while every
    other seat holds its concrete sampled hole.  A single action line is played
    out using the frontier's concrete holes — the phantom traverser hole
    picks the line (the shared-line approximation, in the spirit of the §6.4 leaf
    already being a sampled estimate) — and the reached terminal is settled for
    **every** traverser combo at once via
    :meth:`environment.poker_env.PokerEnv.vector_payout_concrete`.

    Returns ``(n_combos,)`` float64: entry ``c`` is the chip delta to
    ``traverser_seat`` holding combo ``c`` on the single sampled continuation
    (``0`` on combos it cannot hold given the board and the other seats' cards).
    v1 scores a decision-free all-in on its single force-dealt board (variance
    absorbed across iterations), not the per-combo exact board-average — a
    bounded later refinement.
    """
    cfg = ctx.leaf
    rng = ctx.rng
    if frontier_env.is_terminal:
        return frontier_env.vector_payout_concrete(traverser_seat)

    n = frontier_env.n_players
    holes: List[Tuple[int, int]] = [
        tuple(int(c) for c in frontier_env.players[i].cards) for i in range(n)
    ]
    e = frontier_env.with_hole_cards(holes, rng=board_rng_for(ctx))
    while not e.is_terminal:
        seat = e.player_i
        if seat not in profile:
            raise ValueError(
                f"continuation_value_vector: profile is missing acting seat "
                f"{seat}; it must cover every seat that can act."
            )
        c = profile[seat]
        state = e.policy_state_for(
            tuple(int(x) for x in e.current_player.cards), for_blueprint=True
        )
        probs = cfg.policies[c].strategy(state, bias=c)
        idx = sample_index(rng, probs)
        # settle_winners=False: the terminal is valued per-combo below, so the
        # engine's concrete single-hand settlement is skipped (as the vector walk).
        e.step_in_place(state.legal_actions[idx], settle_winners=False)
    return e.vector_payout_concrete(traverser_seat)
