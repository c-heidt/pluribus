"""CFR training algorithms: standard CFR and CFR with pruning (CFR-P).

Both variants are implemented via a single ``_traverse()`` function
parameterised by an ``explore_fn`` callback that decides whether to
descend into a given action at a traversing-player node.  This keeps the
game-tree skeleton in one place and makes it trivial to add new search
variants by supplying a different ``explore_fn``.

External sampling is used for opponent nodes: the traversing player
explores all (un-pruned) actions while each opponent node samples a
single action from the current strategy.
"""

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from poker_ai.ai.action_space import ACTION_TO_IDX, MAX_ACTIONS_PER_STREET
from poker_ai.ai.cfr_tables import CFRTables
from poker_ai.ai.tree_utils import (
    accumulate_regrets,
    get_legal_actions,
    get_node_strategy,
    is_terminal,
    sample_action,
)
from poker_ai.environment.poker_env import PokerEnv as PokerState

log = logging.getLogger("poker_ai.ai.cfr")

# Type alias for the exploration predicate passed to _traverse.
# Signature: (action, regret_row, action_to_idx, state) -> bool
ExploreFn = Callable[[str, np.ndarray, Dict[str, int], PokerState], bool]

_EXPLORE_ALL: ExploreFn = lambda a, row, idx, s: True
"""Exploration predicate for standard CFR: always explore every action."""


def merge_local_delta(
    tables: CFRTables,
    local_delta: Dict[Tuple[int, str], np.ndarray],
) -> None:
    """Merge a local CFR regret accumulator into the shared regret tables.

    For each ``(betting_round, info_set)`` key, the corresponding numpy
    delta array is added to the row in ``tables.regret[betting_round]``
    under the chunk's stripe lock.

    Parameters
    ----------
    tables:
        Tables whose regret arrays will be updated in-place.
    local_delta:
        Per-infoset regret increments keyed by ``(betting_round, info_set)``.
        Values are int64 delta arrays (may be positive or negative).
    """
    for (r, info_set), delta in local_delta.items():
        tables.regret[r].merge_delta_row(info_set, delta)


def cfr(
    tables: CFRTables,
    state: PokerState,
    i: int,
    t: int,
    local_delta: Optional[Dict[Tuple[int, str], np.ndarray]] = None,
) -> float:
    """Counterfactual regret minimisation with external sampling.

    The traversing player explores all legal actions; each opponent node
    samples a single action from the current strategy.  This gives
    O(B^{depth/2}) cost per traversal instead of O(B^depth) for vanilla CFR.

    Parameters
    ----------
    tables:
        CFR tables being trained.
    state:
        Current game state.
    i:
        Traversing player index.
    t:
        Training iteration (used for Linear MCCFR weighting).
    local_delta:
        Per-infoset regret accumulator keyed by ``(betting_round, info_set)``.
        When provided, regret updates are written here lock-free; the caller
        must call :func:`merge_local_delta` when the traversal batch is done.
        When ``None``, a temporary buffer is created and merged before returning.
    """
    _own_delta = local_delta is None
    if _own_delta:
        local_delta = {}
    try:
        return _traverse(tables, state, i, t, local_delta, _EXPLORE_ALL)
    finally:
        if _own_delta:
            merge_local_delta(tables, local_delta)


def cfrp(
    tables: CFRTables,
    state: PokerState,
    i: int,
    t: int,
    c: int,
    local_delta: Optional[Dict[Tuple[int, str], np.ndarray]] = None,
) -> float:
    """CFR with pruning (CFR-P) using external sampling.

    Actions whose cumulative regret is at or below the pruning threshold *c*
    are skipped, except on the river where all actions are always explored.
    Pruning actions that have been consistently bad speeds up convergence
    without losing theoretical guarantees.

    Parameters
    ----------
    tables:
        CFR tables being trained.
    state:
        Current game state.
    i:
        Traversing player index.
    t:
        Training iteration.
    c:
        Pruning threshold.  Actions with ``regret <= c`` are skipped
        (unless on the river).
    local_delta:
        Per-infoset regret accumulator.  See :func:`cfr` for details.
    """
    _own_delta = local_delta is None
    if _own_delta:
        local_delta = {}

    def _prune(
        action: str,
        regret_row: np.ndarray,
        a_to_i: Dict[str, int],
        state: PokerState,
    ) -> bool:
        if state._betting_stage == "river":
            return True
        return int(regret_row[a_to_i[action]]) > c

    try:
        return _traverse(tables, state, i, t, local_delta, _prune)
    finally:
        if _own_delta:
            merge_local_delta(tables, local_delta)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _traverse(
    tables: CFRTables,
    state: PokerState,
    i: int,
    t: int,
    local_delta: Dict[Tuple[int, str], np.ndarray],
    explore_fn: ExploreFn,
) -> float:
    """Generic game-tree traversal used by both ``cfr`` and ``cfrp``.

    At traversing-player nodes, iterates over all actions for which
    ``explore_fn`` returns ``True`` and accumulates regret updates.
    At opponent nodes, samples a single action (external sampling).

    Parameters
    ----------
    tables:
        CFR tables.
    state:
        Current game state.
    i:
        Traversing player index.
    t:
        Training iteration.
    local_delta:
        Lock-free regret accumulator.
    explore_fn:
        ``(action, regret_row, a_to_i, state) -> bool`` — returns ``True``
        if the action should be explored at this node.
    """
    _debug = log.isEnabledFor(logging.DEBUG)

    terminal_value = is_terminal(state, i)
    if terminal_value is not None:
        return terminal_value

    legal_actions = get_legal_actions(state)
    if not legal_actions:
        return float(state.payout[i])

    sigma, r, a_to_i, regret_row = get_node_strategy(tables, state)

    if state.player_i == i:
        # Traversing player: iterate over explore_fn-filtered actions.
        vo = 0.0
        voa: Dict[str, float] = {}
        for action in legal_actions:
            if not explore_fn(action, regret_row, a_to_i, state):
                continue
            if _debug:
                log.debug("ACTION TRAVERSED FOR REGRET: ph %s ACTION: %s", state.player_i, action)
            new_state = state.apply_action(action)
            voa[action] = _traverse(tables, new_state, i, t, local_delta, explore_fn)
            if _debug:
                log.debug("Got EV for %s: %s", action, voa[action])
            vo += sigma[a_to_i[action]] * voa[action]
            if _debug:
                log.debug(
                    "Added to Node EV for ACTION: %s INFOSET: %s\nSTRATEGY: %s: %s",
                    action, state.info_set,
                    sigma[a_to_i[action]], sigma[a_to_i[action]] * voa[action],
                )
        if _debug:
            log.debug("Updated EV at %s: %s", state.info_set, vo)
        # Only explored actions (those in voa) receive regret updates.
        accumulate_regrets(local_delta, r, state.info_set, voa, vo, a_to_i)
        return vo
    else:
        # Opponent node: external sampling — descend one sampled branch.
        action = sample_action(legal_actions, sigma, a_to_i)
        if _debug:
            log.debug(
                "EXTERNAL SAMPLE: opponent ph %s sampled ACTION: %s",
                state.player_i, action,
            )
        return _traverse(tables, state.apply_action(action), i, t, local_delta, explore_fn)
