import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from poker_ai.ai.agent import Agent
from poker_ai.environment.poker_env import PokerEnv as PokerState


log = logging.getLogger("sync.ai")

# ---------------------------------------------------------------------------
# Action abstraction constants — built once at import time.
# ---------------------------------------------------------------------------

CANONICAL_ACTIONS: Dict[int, List[str]] = {
    r: PokerState.get_canonical_actions(r) for r in range(4)
}
"""Full abstract action list per betting round, in stable order."""

ACTION_TO_IDX: Dict[int, Dict[str, int]] = {
    r: {a: i for i, a in enumerate(CANONICAL_ACTIONS[r])} for r in range(4)
}
"""Mapping from action string to column index for a given betting round."""

MAX_ACTIONS_PER_STREET: Dict[int, int] = {
    r: len(CANONICAL_ACTIONS[r]) for r in range(4)
}
"""Row width of each SparseRegretTable, keyed by betting round."""


def calculate_strategy_from_row(
    regret_row: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculate strategy from a 1-D numpy regret array using regret matching.

    Parameters
    ----------
    regret_row : np.ndarray
        1-D int32 (or float-compatible) array of per-action regrets indexed
        by the canonical action ordering for this street.
    valid_mask : np.ndarray, optional
        Boolean mask of length ``n_actions``.  When provided, invalid actions
        are zeroed before regret matching so the uniform fallback distributes
        probability only over legal actions.

    Returns
    -------
    np.ndarray
        1-D float32 probability array of shape ``(n_actions,)`` summing to 1.
    """
    if valid_mask is not None:
        masked = regret_row.astype(np.float32)
        masked[~valid_mask] = 0.0
    else:
        masked = regret_row.astype(np.float32)
    positive = np.maximum(masked, 0.0)
    total = float(positive.sum())
    if total > 0.0:
        return (positive / total).astype(np.float32)
    # Uniform fallback — distribute over valid actions only
    result = np.zeros(len(regret_row), dtype=np.float32)
    if valid_mask is not None:
        n_valid = int(valid_mask.sum())
        if n_valid > 0:
            result[valid_mask] = 1.0 / n_valid
    else:
        n = len(regret_row)
        result[:] = 1.0 / n
    return result


def merge_local_delta(
    agent: Agent,
    local_delta: Dict[Tuple[int, str], np.ndarray],
) -> None:
    """Merge a local CFR regret accumulator into the agent's regret tables.

    For each ``(betting_round, info_set)`` key, the corresponding numpy delta
    array is added to the row in ``agent.regret_tables[betting_round]`` under
    the chunk's stripe lock.

    Parameters
    ----------
    agent : Agent
        Agent whose ``regret_tables`` will be updated in-place.
    local_delta : Dict[Tuple[int, str], np.ndarray]
        Per-infoset regret increments keyed by ``(betting_round, info_set)``.
        Values are int64 delta arrays (positive or negative), not absolute
        regrets.
    """
    for (r, info_set), delta in local_delta.items():
        agent.regret_tables[r].merge_delta_row(info_set, delta)


def update_strategy(
    agent: Agent,
    state: PokerState,
    i: int,
    t: int,
) -> None:
    """Update strategy visit counts for all streets.

    Reads regret from ``agent.regret_tables[r]`` and increments the sampled
    action's visit count in ``agent.strategy_tables[r]`` for the current
    betting round ``r``.

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    state : PokerState
        Current game state.
    i : int
        The traversing player index.
    t : int
        The iteration.
    """
    ph = state.player_i
    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand:
        return

    r = state.betting_round
    raw_actions = state.legal_actions
    legal_actions: List[str] = [a for a in raw_actions if a is not None]
    if not legal_actions:
        return

    canonical = CANONICAL_ACTIONS[r]
    legal_set = set(legal_actions)
    valid_mask = np.array([a in legal_set for a in canonical], dtype=bool)
    row = agent.regret_tables[r].get_row_if_exists(state.info_set)
    regret_row = (
        row if row is not None
        else np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
    )
    sigma = calculate_strategy_from_row(regret_row, valid_mask)
    a_to_i = ACTION_TO_IDX[r]
    probs = np.array([sigma[a_to_i[a]] for a in legal_actions], dtype=np.float64)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs /= prob_sum
    else:
        probs[:] = 1.0 / len(legal_actions)
    action: str = np.random.choice(legal_actions, p=probs)

    if ph == i:
        log.debug("ACTION SAMPLED: ph %s ACTION: %s", state.player_i, action)
        # Increment the strategy table visit count for the sampled action,
        # weighted by iteration t for Linear MCCFR.
        agent.strategy_tables[r].update_row(state.info_set, a_to_i[action], t)

    new_state: PokerState = state.apply_action(action)
    update_strategy(agent, new_state, i, t)


def cfr(
    agent: Agent,
    state: PokerState,
    i: int,
    t: int,
    local_delta: Optional[Dict[Tuple[int, str], np.ndarray]] = None,
) -> float:
    """Regular counter-factual regret minimisation (external sampling).

    Uses **external sampling** for opponent nodes: the traversing player
    explores all its own actions while each opponent node samples a single
    action.  This gives O(B^{depth/2}) cost per traversal vs. O(B^depth)
    for vanilla CFR.

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    state : PokerState
        Current game state.
    i : int
        The traversing player index.
    t : int
        The iteration.
    local_delta : Dict[Tuple[int, str], np.ndarray], optional
        Per-infoset regret accumulator keyed by ``(betting_round, info_set)``.
        Values are ``int64`` delta arrays of length
        ``MAX_ACTIONS_PER_STREET[betting_round]``.  When provided, regret
        updates are written here (lock-free); the caller must call
        ``merge_local_delta`` after the traversal.  When ``None``, a
        temporary buffer is created and merged before returning.
    """
    _own_delta = local_delta is None
    if _own_delta:
        local_delta = {}
    try:
        return _cfr_body(agent, state, i, t, local_delta)
    finally:
        if _own_delta:
            merge_local_delta(agent, local_delta)


def _cfr_body(
    agent: Agent,
    state: PokerState,
    i: int,
    t: int,
    local_delta: Dict[Tuple[int, str], np.ndarray],
) -> float:
    """Internal recursive implementation of ``cfr``."""
    _debug = log.isEnabledFor(logging.DEBUG)
    if _debug:
        log.debug("CFR")
        log.debug("########")
        log.debug("Iteration: %d", t)
        log.debug("Player Set to Update Regret: %d", i)
        log.debug("P(h): %s", state.player_i)
        log.debug("P(h) Updating Regret? %s", state.player_i == i)
        log.debug("Betting Round %s", state._betting_stage)
        log.debug("Community Cards %s", state._table.community_cards)
        for player_idx, player in enumerate(state.players):
            log.debug("Player %d hole cards: %s", player_idx, player.cards)
        try:
            log.debug("I(h): %s", state.info_set)
        except KeyError:
            pass
        log.debug("Betting Action Correct?: %s", state.players)

    ph = state.player_i
    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand:
        return state.payout[i]

    r = state.betting_round
    # Call legal_actions once; build valid_mask from the same list to avoid
    # iterating state.legal_actions twice.
    raw_actions = state.legal_actions
    legal_actions: List[str] = [a for a in raw_actions if a is not None]
    if not legal_actions:
        return state.payout[i]

    canonical = CANONICAL_ACTIONS[r]
    legal_set = set(legal_actions)
    valid_mask = np.array([a in legal_set for a in canonical], dtype=bool)

    row = agent.regret_tables[r].get_row_if_exists(state.info_set)
    regret_row = (
        row if row is not None
        else np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
    )
    sigma = calculate_strategy_from_row(regret_row, valid_mask)
    a_to_i = ACTION_TO_IDX[r]

    if ph == i:
        # Traversing player: iterate all actions, compute counterfactual value.
        vo = 0.0
        voa: Dict[str, float] = {}
        for action in legal_actions:
            if _debug:
                log.debug("ACTION TRAVERSED FOR REGRET: ph %s ACTION: %s", state.player_i, action)
            new_state: PokerState = state.apply_action(action)
            voa[action] = _cfr_body(agent, new_state, i, t, local_delta)
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
        # Accumulate increments into the local delta buffer.
        key = (r, state.info_set)
        if key not in local_delta:
            local_delta[key] = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int64)
        for action in legal_actions:
            local_delta[key][a_to_i[action]] += int(round(voa[action] - vo))
        return vo
    else:
        # External sampling: sample ONE opponent action from current strategy.
        if _debug:
            log.debug("Calculated Strategy for %s: %s", state.info_set, sigma)
        action_probs = np.array([sigma[a_to_i[a]] for a in legal_actions], dtype=np.float64)
        prob_sum = action_probs.sum()
        if prob_sum > 0:
            action_probs /= prob_sum
        else:
            action_probs[:] = 1.0 / len(legal_actions)
        action: str = np.random.choice(legal_actions, p=action_probs)
        if _debug:
            log.debug("EXTERNAL SAMPLE: opponent ph %s sampled ACTION: %s", state.player_i, action)
        new_state: PokerState = state.apply_action(action)
        return _cfr_body(agent, new_state, i, t, local_delta)


def cfrp(
    agent: Agent,
    state: PokerState,
    i: int,
    t: int,
    c: int,
    local_delta: Optional[Dict[Tuple[int, str], np.ndarray]] = None,
) -> float:
    """Counter-factual regret minimisation with pruning (external sampling).

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    state : PokerState
        Current game state.
    i : int
        The traversing player index.
    t : int
        The iteration.
    c : int
        Floor for regret below which we do not search a node.
    local_delta : Dict[Tuple[int, str], np.ndarray], optional
        Per-infoset regret accumulator.  See ``cfr()`` for full description.
    """
    _own_delta = local_delta is None
    if _own_delta:
        local_delta = {}
    try:
        return _cfrp_body(agent, state, i, t, c, local_delta)
    finally:
        if _own_delta:
            merge_local_delta(agent, local_delta)


def _cfrp_body(
    agent: Agent,
    state: PokerState,
    i: int,
    t: int,
    c: int,
    local_delta: Dict[Tuple[int, str], np.ndarray],
) -> float:
    """Internal recursive implementation of ``cfrp``."""
    ph = state.player_i
    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand:
        return state.payout[i]

    r = state.betting_round
    raw_actions = state.legal_actions
    legal_actions: List[str] = [a for a in raw_actions if a is not None]
    if not legal_actions:
        return state.payout[i]

    canonical = CANONICAL_ACTIONS[r]
    legal_set = set(legal_actions)
    valid_mask = np.array([a in legal_set for a in canonical], dtype=bool)

    row = agent.regret_tables[r].get_row_if_exists(state.info_set)
    regret_row = (
        row if row is not None
        else np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
    )
    sigma = calculate_strategy_from_row(regret_row, valid_mask)
    a_to_i = ACTION_TO_IDX[r]

    if ph == i:
        vo = 0.0
        voa: Dict[str, float] = {}
        explored: Dict[str, bool] = {action: False for action in legal_actions}
        # Disable pruning on the river — explore all actions regardless.
        is_river = state._betting_stage == "river"
        for action in legal_actions:
            regret_at_action = int(regret_row[a_to_i[action]])
            if is_river or regret_at_action > c:
                new_state: PokerState = state.apply_action(action)
                voa[action] = _cfrp_body(agent, new_state, i, t, c, local_delta)
                explored[action] = True
                vo += sigma[a_to_i[action]] * voa[action]
        key = (r, state.info_set)
        if key not in local_delta:
            local_delta[key] = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int64)
        for action in legal_actions:
            if explored[action]:
                local_delta[key][a_to_i[action]] += int(round(voa[action] - vo))
        return vo
    else:
        # External sampling: sample ONE opponent action from current strategy.
        action_probs = np.array([sigma[a_to_i[a]] for a in legal_actions], dtype=np.float64)
        prob_sum = action_probs.sum()
        if prob_sum > 0:
            action_probs /= prob_sum
        else:
            action_probs[:] = 1.0 / len(legal_actions)
        action: str = np.random.choice(legal_actions, p=action_probs)
        new_state: PokerState = state.apply_action(action)
        return _cfrp_body(agent, new_state, i, t, c, local_delta)


def serialise(
    agent: Agent,
    save_path: Path,
    t: int,
    server_state: Dict[str, Union[str, float, int, None]],
    locks: dict = {},
) -> None:
    """Stub — full checkpointing is implemented in Phase 6 (CheckpointManager).

    No files are written; a warning is emitted so callers know the call was
    a no-op.
    """
    log.warning(
        "serialise() is a stub in Phase 5 — no checkpoint written. "
        "See Phase 6 / CheckpointManager."
    )
