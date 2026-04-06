"""Strategy traversal — the strategy-sampling phase of Linear MCCFR.

``update_strategy`` is architecturally separate from regret accumulation:
it reads regrets to compute a current strategy, samples one action per
information set for the traversing player, and increments that action's
visit count in the strategy table.  It never modifies regret tables.

Accumulation is unweighted (``amount=1``) — linear weighting of the
average strategy is produced by the periodic LCFR discount applied in
:meth:`CFRTables.apply_discount`, exactly symmetric with how regrets
are linearly weighted.
"""

import logging

from poker_ai.ai.cfr_tables import CFRTables
from poker_ai.ai.tree_utils import (
    get_legal_actions,
    get_node_strategy,
    is_terminal,
    sample_action,
)
from poker_ai.environment.poker_env import PokerEnv as PokerState

log = logging.getLogger("poker_ai.ai.strategy")


def update_strategy(
    tables: CFRTables,
    state: PokerState,
    i: int,
) -> None:
    """Sample an action and record it in the strategy table for player *i*.

    Recursively traverses the game tree.  At every node:
    - Computes the current mixed strategy from regrets.
    - Samples one action proportional to that strategy.
    - If the current player is the traversing player *i*, increments the
      sampled action's visit count in ``tables.strategy[r]`` by 1.
    - Recurses into the sampled successor state.

    Parameters
    ----------
    tables:
        CFR tables being trained.
    state:
        Current game state.
    i:
        Traversing player index.
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

    update_strategy(tables, state.apply_action(action), i)
