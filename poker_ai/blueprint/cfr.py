"""Counterfactual regret minimisation traversals for training.

This module implements the two CFR variants used during training:

- :func:`cfr` — Monte Carlo CFR with external sampling.
- :func:`cfrp` — CFR with pruning (CFR-P), which skips subtrees whose
  cumulative regret has fallen below a user-supplied threshold.

Both variants share a single recursive traversal
(:func:`_traverse`) parameterised by an *exploration predicate*.  The
predicate decides, at each traversing-player node and for each legal
action, whether the action should be explored this iteration.  Passing
``lambda *_: True`` recovers standard CFR; passing a regret-threshold
check recovers CFR-P.  Any future variant (e.g. abstraction-aware
pruning, regret bounding, action-space restriction) can be expressed
as a new predicate without touching the tree-walking code.

External sampling is used at opponent nodes: instead of recursing into
every opponent action, the traversal draws a single action
proportional to the current strategy.  This reduces the per-traversal
branching factor from ``O(B^depth)`` to ``O(B^(depth/2))`` in a
two-player setting, which is the efficiency win that makes
self-play CFR practical.

Regret updates are written into a caller-owned ``local_delta``
dictionary so workers can accumulate across many traversals and flush
to the shared tables at a later sync barrier.  If the caller does not
supply a ``local_delta``, the wrappers create a temporary one and
merge it immediately.
"""

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.blueprint.tree_utils import (
    accumulate_regrets,
    get_legal_actions,
    get_node_strategy,
    is_terminal,
    sample_action,
)
from poker_ai.environment.poker_env import PokerEnv as PokerState

log = logging.getLogger("poker_ai.blueprint.cfr")

ExploreFn = Callable[[str, np.ndarray, Dict[str, int], PokerState], bool]
"""Signature of the exploration predicate passed to :func:`_traverse`.

Called at every traversing-player node with arguments
``(action, regret_row, a_to_i, state)`` and must return ``True`` if the
action should be explored (and its regret updated) and ``False`` if the
action should be pruned this iteration.
"""

_EXPLORE_ALL: ExploreFn = lambda a, row, idx, s: True
"""Predicate used by :func:`cfr`: explore every action unconditionally."""


def merge_local_delta(
    tables: CFRTables,
    local_delta: Dict[Tuple[int, str], np.ndarray],
) -> None:
    """Flush a local regret accumulator into the shared regret tables.

    Walks the per-infoset deltas accumulated during one or more CFR
    traversals and adds each one into the corresponding row of
    ``tables.regret[betting_round]``.  The underlying
    :meth:`~poker_ai.tables.chunked_table.ChunkedTable.merge_delta_row` call
    acquires the chunk's stripe lock, so this function is safe to call
    concurrently from multiple workers.

    After this call the caller is expected to clear or discard
    ``local_delta``; this function does not do so itself.

    Parameters
    ----------
    tables : CFRTables
        Shared regret and strategy tables.
    local_delta : dict[tuple[int, str], np.ndarray]
        Per-infoset regret increments keyed by ``(betting_round,
        info_set)``.  Values are int64 delta arrays (positive or
        negative) produced by :func:`cfr` / :func:`cfrp`.
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
    """Run one CFR traversal from *state* for player *i*.

    Uses Monte Carlo CFR with external sampling: the traversing player
    recurses into every legal action while each opponent node samples
    a single action from the current strategy.

    Parameters
    ----------
    tables : CFRTables
        CFR tables being trained.
    state : PokerState
        Root game state for this traversal (one full hand).
    i : int
        Traversing player index.  Only this player's regrets are
        updated by this call.
    t : int
        Current training iteration.  Passed through to the recursive
        body for logging; the regret increments themselves are
        unweighted — linear CFR weighting is produced by the periodic
        LCFR discount applied by
        :meth:`~poker_ai.tables.cfr_tables.CFRTables.apply_discount`.
    local_delta : dict, optional
        Caller-owned regret accumulator.  When supplied, regret
        updates are written here lock-free and the caller is
        responsible for calling :func:`merge_local_delta` later to
        flush into the shared tables.  When ``None`` a temporary
        accumulator is created and merged immediately before the
        function returns.

    Returns
    -------
    float
        The counterfactual value of *state* under the current strategy
        — useful for diagnostics; most callers discard this.
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
    """Run one CFR-P traversal from *state* for player *i*.

    CFR with pruning skips actions whose cumulative regret has already
    fallen to or below the threshold *c*, on the assumption that such
    actions are unlikely to become optimal in the near future.  The
    river is always explored in full because its regrets contribute
    directly to the terminal payout and are not worth pruning.

    Parameters
    ----------
    tables : CFRTables
        CFR tables being trained.
    state : PokerState
        Root game state for this traversal.
    i : int
        Traversing player index.
    t : int
        Current training iteration.
    c : int
        Pruning threshold.  An action is explored iff its cumulative
        regret is strictly greater than *c* or the current betting
        stage is the river.  The caller is responsible for choosing a
        value consistent with
        :data:`~poker_ai.tables.cfr_tables.REGRET_FLOOR` — a threshold
        below the floor effectively disables pruning.
    local_delta : dict, optional
        Caller-owned regret accumulator.  See :func:`cfr` for details.

    Returns
    -------
    float
        Counterfactual value of *state* under the current strategy.
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
        """Explore *action* iff past the river or its regret exceeds *c*."""
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
    """Generic recursive CFR traversal shared by :func:`cfr` and :func:`cfrp`.

    The structure mirrors the textbook external-sampling CFR pseudocode:

    1. If the node is terminal for player *i*, return the locked-in
       payout.
    2. Otherwise compute the current strategy via regret matching.
    3. At a traversing-player node, recurse into every action for
       which ``explore_fn`` returns ``True``, record per-action
       counterfactual values, and accumulate regret updates into
       ``local_delta``.
    4. At an opponent node, draw a single action proportional to the
       current strategy (external sampling) and recurse.

    ``vo`` — the node value returned to the parent — is the
    strategy-weighted sum of the counterfactual values of the
    *explored* actions only.  Pruned actions do not contribute to
    ``vo`` and do not receive regret updates.

    Parameters
    ----------
    tables : CFRTables
        CFR tables being trained.
    state : PokerState
        Current game state.
    i : int
        Traversing player index.
    t : int
        Current training iteration (for logging).
    local_delta : dict[tuple[int, str], np.ndarray]
        Caller-owned regret accumulator written in place.
    explore_fn : ExploreFn
        Predicate deciding which actions to explore at traversing-player
        nodes.

    Returns
    -------
    float
        Counterfactual value of *state* for player *i*.
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
