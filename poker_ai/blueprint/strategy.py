"""Strategy-sampling traversal for Linear Monte Carlo CFR.

In Linear MCCFR the *average* strategy is the quantity that converges
to the Nash equilibrium; the per-iteration strategy derived by regret
matching is only a stepping stone.  The codebase stores the average
strategy as a running sum of visit counts in
``tables.strategy[betting_round]`` and normalises it at play time by
dividing each action's count by the row sum — the standard "accumulate
during training, normalise at read" pattern.

This module implements the traversal that increments those counts:

1. Walk the game tree from the root.
2. At every non-terminal node, compute the current mixed strategy
   from the regrets and sample a single action proportional to it.
3. If the current player is the traversing player, increment the
   sampled action's count in ``tables.strategy[r]`` by 1.
4. Recurse into the sampled successor state.

Accumulation is unweighted (each visit adds ``1``) because linear
weighting of the average strategy is produced by the periodic LCFR
discount applied by
:meth:`~poker_ai.tables.cfr_tables.CFRTables.apply_discount`, exactly
symmetric with how regrets are linearly weighted.  Doing both at once
(weighting by ``t`` *and* discounting) would produce super-linear
weighting.

The traversal terminates early when either the hand has ended or the
traversing player is no longer active — beyond that point the
traversing player cannot make any further decisions that would update
their average strategy.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from environment.action_space import MAX_ACTIONS_PER_STREET
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.blueprint.tree_utils import (
    get_legal_actions,
    get_node_strategy,
    is_terminal,
    sample_action,
)
from environment.poker_env import PokerEnv as PokerState

log = logging.getLogger("poker_ai.blueprint.strategy")


def update_strategy(
    tables: CFRTables,
    state: PokerState,
    i: int,
    local_delta: Optional[Dict[Tuple[int, bytes], np.ndarray]] = None,
) -> None:
    """Accumulate player *i*'s average strategy on the **first betting round**.

    Implements Pluribus's UPDATE-STRATEGY pass (Science supplement,
    Algorithm 1, line 2): the average strategy is tracked *only* on the
    pre-flop betting round, and within it every opponent action is
    traversed (full branching) rather than sampled, so a single pass
    covers the whole pre-flop opponent sub-tree.  At each **player-*i***
    node one action is drawn from the current regret-matching strategy
    and its visit count is incremented by 1; the traversal then descends
    the sampled action.  The walk returns as soon as the hand leaves the
    pre-flop round (``betting_round != 0``) — the post-flop average is
    reconstructed offline from checkpoint snapshots
    (:mod:`poker_ai.blueprint.offline_average`), not tracked online.

    This replaces the earlier all-streets *sampled* walk: measured A/B
    showed that online average read worse than the last iterate (its
    post-flop rows were starved and its early iterations polluted the
    running mean).  Restricting to the pre-flop round with full opponent
    branching gives a dense, correctly-averaged pre-flop table cheaply,
    and drops the per-iteration cost of the post-flop strategy walk.

    The function returns nothing; the visit-count increments land either
    directly in the shared ``tables.strategy`` tables or in a
    caller-owned accumulator, depending on ``local_delta``.

    Parameters
    ----------
    tables : CFRTables
        CFR tables being trained.  ``tables.regret`` is read to compute
        the sampling distribution; ``tables.strategy[0]`` is written
        only when ``local_delta is None``.
    state : PokerState
        Starting game state for this strategy-sampling pass.
    i : int
        Traversing player whose average strategy is being updated.
    local_delta : dict, optional
        Caller-owned visit-count accumulator keyed by ``(betting_round,
        info_set)`` with ``int64`` delta rows — symmetric with the
        ``local_delta`` :func:`poker_ai.blueprint.cfr.cfr` accumulates into.
        Every key produced here has ``betting_round == 0``.  When supplied,
        increments are written here (lock-free) instead of into the shared
        tables and the caller flushes via
        :func:`poker_ai.blueprint.cfr.merge_local_strategy_delta`.  When
        ``None`` (default) the increment is applied directly to
        ``tables.strategy[0]`` via the stripe-locked ``update_row``.
    """
    # Terminal check first: it also fires when player ``i`` has folded, and it
    # must precede the betting-round read because ``betting_round`` raises for a
    # terminal state (there is no active betting round there).
    if is_terminal(state, i) is not None:
        return
    # Stop as soon as the hand leaves the pre-flop round — the average strategy
    # is tracked pre-flop only.
    if state.betting_round != 0:
        return

    legal_actions = get_legal_actions(state)
    if not legal_actions:
        return

    if state.player_i == i:
        # Traverser node: sample one action from the regret-matching strategy,
        # record it, and descend that action only.
        sigma, r, a_to_i, _, info_set = get_node_strategy(tables, state)
        action = sample_action(legal_actions, sigma, a_to_i)
        log.debug("ACTION SAMPLED: ph %s ACTION: %s", state.player_i, action)
        if local_delta is None:
            tables.strategy[r].update_row(info_set, a_to_i[action], 1)
        else:
            key = (r, info_set)
            row = local_delta.get(key)
            if row is None:
                row = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int64)
                local_delta[key] = row
            row[a_to_i[action]] += 1

        # Descend in place and restore on the way back so the caller's ``state``
        # is left unchanged (the same non-mutating contract as before).
        token = state.step_in_place(action)
        update_strategy(tables, state, i, local_delta)
        state.undo(token)
    else:
        # Opponent node: traverse EVERY legal action (Pluribus full pre-flop
        # branching).  No regret read or info-set resolution here — opponent
        # nodes write nothing, which is the per-iteration cost saving.
        for action in legal_actions:
            token = state.step_in_place(action)
            update_strategy(tables, state, i, local_delta)
            state.undo(token)
