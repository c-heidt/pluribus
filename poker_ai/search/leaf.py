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
from poker_ai.blueprint.tree_utils import sample_index
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
        1).  Each rollout re-randomises the board runout; the hands are fixed.
        The leaf value is memoised per ``(leaf public_key, holes, profile)``, so
        ``1`` scores each leaf-profile by a **single sampled playout** — matching
        the paper's design (one action sampled per infoset; stochasticity recovered
        in aggregate across CFR's per-iteration hole draws, not by per-leaf
        averaging).  ``>1`` averages per leaf: an over-the-paper variance reduction
        that costs proportionally more per iteration (the rollout walk scales
        linearly with this) and thus buys fewer iterations under a fixed wall
        budget.  In the low-iteration 6p regime, spending that compute on more
        iterations (better hole coverage) generally beats per-leaf averaging.
        Decision-free all-in runouts are scored exactly (``use_decision_free_equity``),
        so this only affects the sampled action-line / non-all-in variance.
    use_decision_free_equity : bool
        A/B toggle for the over-the-paper improvement (§6.4).  ``True`` (default)
        replaces an all-in showdown's single sampled board with the exact
        board-average via :meth:`PokerEnv.runout_equity`; ``False`` reproduces
        the paper's sampled single-board runout.  The same field is read by the
        solver (row 6) for its forced-runout terminals, so flipping it measures
        the whole effect.
    """

    policies: Mapping[BiasClass, Policy]
    n_rollouts: int = 1
    use_decision_free_equity: bool = True


def continuation_value(
    frontier_env: PokerEnv,
    profile: Mapping[int, BiasClass],
    ctx: SubgameContext,
    runout_cache: dict | None = None,
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
    runout_cache : dict, optional
        Memo for exact decision-free runout equities (§6.4.2).  When supplied
        (the solver passes its search-lifetime ``SolverState.runout_cache``),
        the integration for a given ``(all-seat holes, runout snapshot)`` runs
        once and is reused across calls — so a leaf's four bias profiles share
        one integration per distinct all-in.  When ``None`` a fresh per-call
        dict is used (the standalone behaviour); the runout value is exact and
        rng-independent either way, so caching never changes the result.

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

    # Concrete holes are fixed across the rollouts; the env below is built with
    # them once and reused (rewound between rollouts), the undealt board reshuffled
    # per rollout so each draws a fresh runout while the hands stay put.
    holes: List[Tuple[int, int]] = [
        tuple(int(c) for c in frontier_env.players[i].cards) for i in range(n)
    ]

    accum = np.zeros(n, dtype=np.float64)
    # Holes are fixed across the rollouts, so a decision-free runout's exact
    # board-average depends only on ``(holes, runout snapshot)`` — the snapshot
    # being the env's ``runout_key`` (board prefix, frozen pot contributions,
    # active mask).  Memoise on that pair so the integration runs once per
    # distinct all-in, not once per rollout.  A caller-supplied ``runout_cache``
    # extends the memo across calls (the solver's search-lifetime table, shared
    # by a leaf's four bias profiles); ``holes`` is in the key so entries from
    # other leaves/hands never collide.  In a leaf the runout is always <=2
    # board cards (the exact path), so the cached value is rng-independent.
    cache = runout_cache if runout_cache is not None else {}
    holes_key = tuple(holes)
    # Pay the env construction **once**: build the holes-applied env before the
    # loop and rewind it after each rollout via the make/undo path instead of
    # deepcopying per rollout (``step_in_place`` returns an ``UndoToken``; ``undo``
    # restores the deck cursor, board, pot and terminal snapshot).  The holes are
    # fixed across the rollouts, so only ``shuffle_undealt`` need vary the board —
    # each rollout still draws an independent runout, but the ~O(n_rollouts)
    # deepcopies collapse to one.  ``with_hole_cards`` already shuffles the undealt
    # deck once (used by rollout 0); later rollouts reshuffle after rewinding.
    e = frontier_env.with_hole_cards(holes)
    for r in range(cfg.n_rollouts):
        if r > 0:
            e.deck.shuffle_undealt()
        tokens: List = []
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
            tokens.append(e.step_in_place(state.legal_actions[idx]))
        # Decision-free all-in showdown over an incomplete board: take the
        # exact board-average instead of the single dealt runout (§6.4),
        # unless the A/B toggle reproduces the sampled-runout baseline.
        if use_equity and e.is_decision_free:
            key = (holes_key, e.runout_key)
            eq = cache.get(key)
            if eq is None:
                eq = e.runout_equity(rng=rng)
                cache[key] = eq
            for i in range(n):
                accum[i] += eq[i]
        else:
            for i in range(n):
                accum[i] += float(e.payout[i])
        # Rewind to the frontier (strict LIFO) so the next rollout starts clean.
        for tok in reversed(tokens):
            e.undo(tok)
    return accum / cfg.n_rollouts
