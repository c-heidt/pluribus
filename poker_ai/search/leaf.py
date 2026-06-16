"""Depth-limit leaf continuation-value evaluation.

:func:`leaf_value` is the depth-limited subgame solver's leaf
evaluator: at a search-tree node where ``env.betting_round >
ctx.street_at_root``, it runs Monte-Carlo rollouts to terminal under
a mixed fleet of bias-class policies, returning the mean per-seat
chip delta.

The module is a pure consumer of :class:`PokerEnv`,
:class:`Policy`, :class:`SubgameContext`, and per-opponent range
mappings.  Every rollout samples a fresh hole for every non-bot
seat — live opponents from ``live_ranges`` and folded opponents
from ``folded_ranges`` — and applies all the samples in one
batched :meth:`PokerEnv.with_hole_cards` call.

See §6.4 of ``docs/subgame_solving.md`` for the design.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from typing_extensions import get_args

from environment.poker_env import PokerEnv
from poker_ai.search.context import SubgameContext, _board_compatible_mask
from poker_ai.search.policy import BiasClass, Policy
from poker_ai.search.ranges import Range


_BIAS_CLASSES: Tuple[BiasClass, ...] = get_args(BiasClass)

logger = logging.getLogger(__name__)


@dataclass
class LeafConfig:
    """Per-session leaf evaluator config.

    Attributes
    ----------
    policies : Mapping[BiasClass, Policy]
        One policy per bias class.  Phase 1 wires four
        :class:`BlueprintPolicy` instances at varying
        ``bias_magnitude``; Phase 2 wires four precomputed
        ``BiasedBlueprintPolicy`` variants.  Same shape, swapped by
        the runtime.
    n_rollouts : int
        Monte-Carlo sample count per :func:`leaf_value` call.
        Default 20; small because the solver calls ``leaf_value`` at
        every visited leaf across thousands of CFR iterations.
    n_rejection_retries : int
        Number of rejection-sampling attempts per rollout before
        falling back to randomised-order sequential masked sampling.
        Each attempt draws every non-bot seat's hole independently
        from its (board-and-fixed-card-masked) range and accepts
        only if all sampled holes are pairwise disjoint; the
        fallback is biased on the marginals but always produces a
        feasible state.  Default 4.
    """

    policies: Mapping[BiasClass, Policy]
    n_rollouts: int = 20
    n_rejection_retries: int = 4


def leaf_value(
    env: PokerEnv,
    live_ranges: Mapping[int, Range],
    folded_ranges: Mapping[int, Range],
    ctx: SubgameContext,
) -> np.ndarray:
    """Expected per-seat chip delta at the leaf.

    Parameters
    ----------
    env : PokerEnv
        State at the leaf.  Typically ``env.betting_round >
        ctx.street_at_root``; an already-terminal env is handled by
        a fast path that returns ``env.payout`` directly.
    live_ranges : Mapping[int, Range]
        Per-live-opponent ranges after the search path's
        conditioning.  Excludes the bot's seat; ``my_seat`` in
        ``live_ranges`` raises ``ValueError``.
    folded_ranges : Mapping[int, Range]
        Per-folded-opponent fold-time marginals from
        :meth:`RangeTracker.folded_snapshot`.  Disjoint from
        ``live_ranges``; an empty dict means no folded seats.  These
        seats are resampled too so the leaf doesn't leak the folded
        seats' real dealt cards into per-rollout community variance.
    ctx : SubgameContext
        Search context.  ``ctx.rng`` is the sole RNG; ``ctx.leaf``
        carries the policy fleet and rollout count.

    Returns
    -------
    numpy.ndarray
        Float64 vector of shape ``(env.n_players,)``.  Entry ``i`` is
        the mean of ``env.payout[i]`` across completed rollouts.  If
        every rollout aborts (genuinely infeasible hole sampling),
        the per-seat result is zero.
    """
    n = env.n_players
    if ctx.my_seat in live_ranges:
        raise ValueError(
            f"leaf_value: live_ranges must not contain ctx.my_seat="
            f"{ctx.my_seat}; the bot's hole is fixed at ctx.my_hole."
        )
    if ctx.my_seat in folded_ranges:
        raise ValueError(
            f"leaf_value: folded_ranges must not contain ctx.my_seat="
            f"{ctx.my_seat}."
        )
    overlap = set(live_ranges) & set(folded_ranges)
    if overlap:
        raise ValueError(
            f"leaf_value: a seat cannot be in both live_ranges and "
            f"folded_ranges; got overlap {sorted(overlap)}."
        )
    if env.is_terminal:
        return np.array(
            [float(env.payout[i]) for i in range(n)], dtype=np.float64
        )

    rng = ctx.rng
    cfg = ctx.leaf
    accum = np.zeros(n, dtype=np.float64)
    completed = 0
    for _ in range(cfg.n_rollouts):
        bias = {i: str(rng.choice(_BIAS_CLASSES)) for i in range(n)}
        sampled = _sample_all_holes(env, live_ranges, folded_ranges, ctx)
        if sampled is None:
            continue
        e = sampled
        while not e.is_terminal:
            state = e.policy_state
            c = bias[e.player_i]
            probs = cfg.policies[c].strategy(state, bias=c)
            # Cast + renormalise: Policy.strategy returns float32; the
            # generator's choice() rejects probability vectors whose
            # sum deviates from 1.0 by more than ~1e-8, which float32
            # normalisation can violate for non-power-of-two
            # denominators.
            probs = np.asarray(probs, dtype=np.float64)
            probs /= probs.sum()
            idx = int(rng.choice(len(state.legal_actions), p=probs))
            e = e.apply_action(state.legal_actions[idx])
        for i in range(n):
            accum[i] += float(e.payout[i])
        completed += 1
    if completed == 0:
        return np.zeros(n, dtype=np.float64)
    return accum / completed


def _sample_all_holes(
    env: PokerEnv,
    live_ranges: Mapping[int, Range],
    folded_ranges: Mapping[int, Range],
    ctx: SubgameContext,
) -> Optional[PokerEnv]:
    """Draw one hole per non-bot seat and return an env reflecting it.

    Two-tier sampler.  All non-bot seats (live ∪ folded) are
    resampled together; the bot's seat is filled with
    ``ctx.my_hole``.  The resulting length-``n_players`` hole list
    is applied via a single batched :meth:`PokerEnv.with_hole_cards`
    call.

    **Primary path — rejection sampling.**  Each seat drawn
    independently from its unconditional range, masked only against
    fixed cards (community + ``ctx.my_hole``).  Accepted iff every
    pair of holes is disjoint.  Up to
    ``ctx.leaf.n_rejection_retries`` attempts.  Unbiased over the
    physically-realisable joint when accepted.

    **Fallback — randomised-order sequential masked sampling.**  If
    every rejection attempt collides or any seat's range collapses
    against fixed cards, iterates the seats in a randomised order
    and masks each draw against every other seat's *current* (or
    already-sampled) hole.  Biased on the marginals but always
    produces a feasible state.  A :mod:`logging` ``INFO`` line
    records each fallback so rejection-vs-fallback frequency can be
    evaluated post-run via
    ``logging.getLogger("poker_ai.search.leaf")``.

    Returns ``None`` only when even the fallback's uniform-over-
    feasible degenerate path is empty.
    """
    seats_to_sample: Dict[int, Range] = {}
    seats_to_sample.update(live_ranges)
    seats_to_sample.update(folded_ranges)
    if not seats_to_sample:
        # No non-bot seats need resampling.  Identity batch keeps
        # every seat's current hole but still triggers
        # ``shuffle_undealt`` so the next community deal is
        # uniformly random — matching the leaf-EV semantics the rest
        # of the rollout loop relies on.
        return env.with_hole_cards(
            [tuple(env.players[i].cards) for i in range(env.n_players)]
        )

    rng = ctx.rng
    cc = env.combo_cards
    board_mask = _board_compatible_mask(env)
    # Fixed cards = community plus the bot's hole.  Non-bot seats'
    # current holes are *not* fixed: they're about to be overwritten
    # by the sample (live or folded alike).
    fixed_used = set(int(c) for c in env.community_cards)
    fixed_used.update(int(c) for c in ctx.my_hole)

    seats = sorted(seats_to_sample)
    seat_weights: List[np.ndarray] = []
    for seat in seats:
        w = _feasible_weights(
            seats_to_sample[seat], board_mask, fixed_used, cc
        )
        total = w.sum()
        if total <= 0.0:
            logger.info(
                "leaf rejection-sampling skipped: seat %d range "
                "collapsed against fixed cards (street %d, %d sampled "
                "seats); falling back to sequential",
                seat, env.betting_round, len(seats_to_sample),
            )
            return _sample_sequential(env, seats_to_sample, board_mask, ctx)
        seat_weights.append(w / total)

    n_retries = max(1, int(ctx.leaf.n_rejection_retries))
    for _ in range(n_retries):
        samples = _try_rejection_attempt(seats, seat_weights, cc, rng)
        if samples is not None:
            return _apply_with_bot(env, samples, ctx.my_seat, ctx.my_hole)

    logger.info(
        "leaf rejection-sampling exhausted after %d retries "
        "(street %d, %d sampled seats); falling back to sequential",
        n_retries, env.betting_round, len(seats_to_sample),
    )
    return _sample_sequential(env, seats_to_sample, board_mask, ctx)


def _try_rejection_attempt(
    seats: List[int],
    seat_weights: List[np.ndarray],
    combo_cards: np.ndarray,
    rng: np.random.Generator,
) -> Optional[List[Tuple[int, Tuple[int, int]]]]:
    """One rejection-sampling attempt.

    Draws every seat from its precomputed normalised weights, then
    checks that no card appears in more than one drawn hole.  Returns
    the list of ``(seat, (c0, c1))`` on success; ``None`` on any
    pairwise collision.
    """
    samples: List[Tuple[int, Tuple[int, int]]] = []
    used: set = set()
    n_combos = combo_cards.shape[0]
    for seat, w in zip(seats, seat_weights):
        idx = int(rng.choice(n_combos, p=w))
        c0 = int(combo_cards[idx, 0])
        c1 = int(combo_cards[idx, 1])
        if c0 in used or c1 in used:
            return None
        used.add(c0)
        used.add(c1)
        samples.append((seat, (c0, c1)))
    return samples


def _apply_with_bot(
    env: PokerEnv,
    samples: List[Tuple[int, Tuple[int, int]]],
    my_seat: int,
    my_hole: Tuple[int, int],
) -> PokerEnv:
    """Apply the rejection samples + the bot's fixed hole atomically.

    Builds a length-``n_players`` holes list (sampled entries for
    non-bot seats, ``my_hole`` for the bot, original cards for any
    seat not in ``samples`` — though in practice every non-bot seat
    is always in ``samples``).  Single batched
    :meth:`PokerEnv.with_hole_cards` call handles the deck sync and
    undealt-segment shuffle in one go.
    """
    by_seat = {seat: cards for seat, cards in samples}
    holes = []
    for i in range(env.n_players):
        if i == my_seat:
            holes.append((int(my_hole[0]), int(my_hole[1])))
        elif i in by_seat:
            holes.append(by_seat[i])
        else:
            holes.append(tuple(int(c) for c in env.players[i].cards))
    return env.with_hole_cards(holes)


def _sample_sequential(
    env: PokerEnv,
    seats_to_sample: Mapping[int, Range],
    board_mask: np.ndarray,
    ctx: SubgameContext,
) -> Optional[PokerEnv]:
    """Fallback sampler — sequential masked draw in randomised order.

    Walks ``seats_to_sample`` in a per-call random permutation so
    the marginal bias toward the first-drawn seat is averaged out
    across rollouts.  Each draw is masked against community, the
    bot's hole, every other non-bot seat's *currently-pending*
    sample (or original hole if not yet drawn), and the bot's own
    hole.  On a collapsed range, falls back to uniform over the
    still-feasible combos and emits a :class:`RuntimeWarning`.
    Returns ``None`` only when even the uniform fallback is empty
    for some seat.

    After every seat has been drawn, the single batched
    :meth:`PokerEnv.with_hole_cards` call applies the result.
    """
    rng = ctx.rng
    cc = env.combo_cards
    fixed_used_base = set(int(c) for c in env.community_cards)
    fixed_used_base.update(int(c) for c in ctx.my_hole)
    # Track the holes we've already committed for not-yet-applied
    # seats; the masking grows monotonically so each successive draw
    # avoids prior draws.
    committed: Dict[int, Tuple[int, int]] = {}
    order = list(seats_to_sample)
    rng.shuffle(order)
    for seat in order:
        used = set(fixed_used_base)
        for s, h in committed.items():
            used.update(int(c) for c in h)
        # Also exclude any not-yet-sampled non-bot seat's current
        # cards from this seat's mask, so the sequential draws stay
        # consistent with how the leaf's caller built the env
        # (matches pre-refactor behaviour).
        for s in seats_to_sample:
            if s in committed or s == seat:
                continue
            used.update(int(c) for c in env.players[s].cards)
        w = _feasible_weights(seats_to_sample[seat], board_mask, used, cc)
        total = w.sum()
        if total <= 0.0:
            warnings.warn(
                f"leaf_value: range for seat {seat} collapsed at the "
                "leaf board; falling back to uniform over feasible "
                "combos.",
                RuntimeWarning,
                stacklevel=2,
            )
            w = _feasible_weights(board_mask, board_mask, used, cc)
            total = w.sum()
            if total <= 0.0:
                return None
        w /= total
        idx = int(rng.choice(env.n_combos, p=w))
        committed[seat] = (int(cc[idx, 0]), int(cc[idx, 1]))
    samples = list(committed.items())
    return _apply_with_bot(env, samples, ctx.my_seat, ctx.my_hole)


def _feasible_weights(
    base: np.ndarray,
    board_mask: np.ndarray,
    used: set,
    combo_cards: np.ndarray,
) -> np.ndarray:
    """Float64 weights with combos using any card in ``used`` zeroed
    and the per-leaf board mask applied.

    Pure helper used by both the primary range-conditioned draw and
    the uniform-fallback draw; same masking semantics, different
    starting weights.
    """
    w = np.asarray(base, dtype=np.float64).copy()
    w *= board_mask
    if used:
        used_arr = np.fromiter(used, dtype=np.int32)
        conflict = (
            np.isin(combo_cards[:, 0], used_arr)
            | np.isin(combo_cards[:, 1], used_arr)
        )
        w[conflict] = 0.0
    return w
