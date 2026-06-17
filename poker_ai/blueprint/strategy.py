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
) -> None:
    """Sample a play-through from *state* and record player *i*'s actions.

    Recursively walks the game tree starting from *state*.  At every
    node the function draws one action from the current regret-matching
    strategy; when the current player is *i* the sampled action's
    visit count is incremented in ``tables.strategy[r]`` by 1, where
    ``r`` is the betting round of the current node.  The traversal
    then descends into the sampled successor.

    Opponent decisions are also sampled — the traversal follows a
    single playthrough rather than branching — but they do not write
    anything to the strategy tables.

    The function returns nothing; all state updates happen in place on
    ``tables.strategy``.

    Parameters
    ----------
    tables : CFRTables
        CFR tables being trained.  Only ``tables.strategy[r]`` is
        modified; regret tables are read-only in this traversal.
    state : PokerState
        Starting game state for this strategy-sampling pass.
    i : int
        Traversing player whose average strategy is being updated.
    """
    if is_terminal(state, i) is not None:
        return

    legal_actions = get_legal_actions(state)
    if not legal_actions:
        return

    sigma, r, a_to_i, _ = get_node_strategy(tables, state)
    action = sample_action(legal_actions, sigma, a_to_i)

    if state.player_i == i:
        log.debug("ACTION SAMPLED: ph %s ACTION: %s", state.player_i, action)
        tables.strategy[r].update_row(state.info_set, a_to_i[action], 1)

    # Single sampled action: descend in place and restore on the way back,
    # so this function leaves its ``state`` argument unchanged (the same
    # non-mutating contract the old copy-on-write traversal had).
    token = state.step_in_place(action)
    update_strategy(tables, state, i)
    state.undo(token)
