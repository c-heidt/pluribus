import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np

from poker_ai.ai.agent import Agent
from poker_ai.games.short_deck.state import ShortDeckPokerState


log = logging.getLogger("sync.ai")


def calculate_strategy(this_info_sets_regret: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate the strategy based on the current information sets regret.

    ...

    Parameters
    ----------
    this_info_sets_regret : Dict[str, float]
        Regret for each action at this info set.

    Returns
    -------
    strategy : Dict[str, float]
        Strategy as a probability over actions.
    """
    # TODO: Could we instanciate a state object from an info set?
    actions = this_info_sets_regret.keys()
    regret_sum = sum([max(regret, 0) for regret in this_info_sets_regret.values()])
    if regret_sum > 0:
        strategy: Dict[str, float] = {
            action: max(this_info_sets_regret[action], 0) / regret_sum
            for action in actions
        }
    else:
        default_probability = 1 / len(actions)
        strategy: Dict[str, float] = {action: default_probability for action in actions}
    return strategy


def calculate_strategy_from_row(regret_row: np.ndarray) -> np.ndarray:
    """Calculate strategy from a 1-D numpy regret array using regret matching.

    Pure-numpy implementation intended for use with ``SparseRegretTable``
    rows.  Falls back to a uniform distribution when all regrets are
    non-positive.

    Parameters
    ----------
    regret_row : np.ndarray
        1-D int32 (or float-compatible) array of per-action regrets.

    Returns
    -------
    np.ndarray
        1-D float32 array of action probabilities that sums to 1.0.
    """
    positive = np.maximum(regret_row, 0).astype(np.float32)
    total = float(positive.sum())
    if total > 0.0:
        return positive / total
    n = len(regret_row)
    return np.full(n, 1.0 / n, dtype=np.float32)


def merge_local_delta(
    agent: Agent,
    local_delta: Dict[str, Dict[str, float]],
) -> None:
    """Merge a local CFR regret accumulator into ``agent.regret``.

    After each ``cfr()`` or ``cfrp()`` call with a ``local_delta`` buffer the
    caller must merge the buffer into the shared regret table.  In
    single-process mode this is a plain dict update.  In multi-process mode
    the caller should hold the regret lock around this call so that concurrent
    writes from competing workers do not lose increments.

    Parameters
    ----------
    agent : Agent
        Agent whose ``regret`` dict will be updated in-place.
    local_delta : Dict[str, Dict[str, float]]
        Per-infoset regret increments produced by ``cfr()`` / ``cfrp()``.
        Values are deltas (positive or negative), not absolute regrets.
    """
    for info_set, delta in local_delta.items():
        current = dict(agent.regret.get(info_set, {}))
        for action, regret_delta in delta.items():
            current[action] = current.get(action, 0.0) + regret_delta
        agent.regret[info_set] = current


def update_strategy(
    agent: Agent,
    state: ShortDeckPokerState,
    i: int,
    t: int,
) -> None:
    """
    Update pre flop strategy using a more theoretically sound approach.

    Reads regret from ``agent.regret`` and updates preflop visit counts in
    ``agent.strategy``.  Only called for preflop states — all postflop states
    are deliberately skipped (``state.betting_round > 0``) following the
    Pluribus blueprint (Bug 5 — intentional design, not a bug).

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    state : ShortDeckPokerState
        Current game state.
    i : int
        The Player.
    t : int
        The iteration.
    """
    ph = state.player_i  # this is always the case no matter what i is

    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand or state.betting_round > 0:
        return

    elif ph == i:
        # calculate regret
        this_info_sets_regret = {**state.initial_regret, **agent.regret.get(state.info_set, {})}
        sigma = calculate_strategy(this_info_sets_regret)
        log.debug(f"Calculated Strategy for {state.info_set}: {sigma}")
        # choose an action based of sigma
        available_actions: List[str] = list(sigma.keys())
        action_probabilities: np.ndarray = np.array(list(sigma.values()))
        action: str = np.random.choice(available_actions, p=action_probabilities)
        log.debug(f"ACTION SAMPLED: ph {state.player_i} ACTION: {action}")
        # Increment the action counter.
        this_states_strategy = {**state.initial_strategy, **agent.strategy.get(state.info_set, {})}
        this_states_strategy[action] += 1
        # Update the master strategy by assigning.
        agent.strategy[state.info_set] = this_states_strategy
        new_state: ShortDeckPokerState = state.apply_action(action)
        update_strategy(agent, new_state, i, t)
    else:
        # Traverse each action.
        for action in state.legal_actions:
            log.debug(f"Going to Traverse {action} for opponent")
            new_state: ShortDeckPokerState = state.apply_action(action)
            update_strategy(agent, new_state, i, t)


def cfr(
    agent: Agent,
    state: ShortDeckPokerState,
    i: int,
    t: int,
    local_delta: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """
    Regular counter factual regret minimization algorithm.

    Uses **external sampling** for opponent nodes — iterates all opponent
    actions weighted by current strategy, eliminating the high-variance
    single-sample used by the previous outcome-sampling scheme.

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    state : ShortDeckPokerState
        Current game state.
    i : int
        The traversing player index.
    t : int
        The iteration.
    local_delta : Dict[str, Dict[str, float]], optional
        Per-infoset regret accumulator owned by the caller.  When provided,
        all regret updates are written here rather than directly to
        ``agent.regret``, keeping traversal completely lock-free.  The caller
        is responsible for merging via ``merge_local_delta`` after the call
        returns.  When ``None`` (default), updates are written directly to
        ``agent.regret`` for single-process training compatibility.
    """
    log.debug("CFR")
    log.debug("########")
    log.debug(f"Iteration: {t}")
    log.debug(f"Player Set to Update Regret: {i}")
    log.debug(f"P(h): {state.player_i}")
    log.debug(f"P(h) Updating Regret? {state.player_i == i}")
    log.debug(f"Betting Round {state._betting_stage}")
    log.debug(f"Community Cards {state._table.community_cards}")
    for player_idx, player in enumerate(state.players):
        log.debug(f"Player {player_idx} hole cards: {player.cards}")
    try:
        log.debug(f"I(h): {state.info_set}")
    except KeyError:
        pass
    log.debug(f"Betting Action Correct?: {state.players}")

    ph = state.player_i

    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand:
        return state.payout[i]

    elif ph == i:
        # Traversing player: iterate all actions, compute counterfactual value.
        this_info_sets_regret = {**state.initial_regret, **agent.regret.get(state.info_set, {})}
        sigma = calculate_strategy(this_info_sets_regret)
        log.debug(f"Calculated Strategy for {state.info_set}: {sigma}")

        vo = 0.0
        voa: Dict[str, float] = {}
        for action in state.legal_actions:
            log.debug(
                f"ACTION TRAVERSED FOR REGRET: ph {state.player_i} ACTION: {action}"
            )
            new_state: ShortDeckPokerState = state.apply_action(action)
            voa[action] = cfr(agent, new_state, i, t, local_delta)
            log.debug(f"Got EV for {action}: {voa[action]}")
            vo += sigma[action] * voa[action]
            log.debug(
                f"Added to Node EV for ACTION: {action} INFOSET: {state.info_set}\n"
                f"STRATEGY: {sigma[action]}: {sigma[action] * voa[action]}"
            )
        log.debug(f"Updated EV at {state.info_set}: {vo}")
        if local_delta is not None:
            # Lock-free path: accumulate regret increments into the caller's
            # local buffer.  No writes to agent.regret during traversal.
            infoset_delta = local_delta.setdefault(state.info_set, {})
            for action in state.legal_actions:
                infoset_delta[action] = infoset_delta.get(action, 0.0) + (voa[action] - vo)
        else:
            # Direct path (single-process): write to agent.regret.
            # Re-read to pick up any changes made since sigma was computed.
            this_info_sets_regret = {**state.initial_regret, **agent.regret.get(state.info_set, {})}
            for action in state.legal_actions:
                this_info_sets_regret[action] += voa[action] - vo
            agent.regret[state.info_set] = this_info_sets_regret
        return vo
    else:
        # External sampling: iterate all opponent actions weighted by their
        # current strategy probability, replacing the old outcome-sampling
        # single-action sample and dramatically reducing per-iteration variance.
        this_info_sets_regret = {**state.initial_regret, **agent.regret.get(state.info_set, {})}
        sigma = calculate_strategy(this_info_sets_regret)
        log.debug(f"Calculated Strategy for {state.info_set}: {sigma}")
        vo = 0.0
        for action in state.legal_actions:
            log.debug(f"EXTERNAL SAMPLE: opponent ph {state.player_i} ACTION: {action}")
            new_state: ShortDeckPokerState = state.apply_action(action)
            vo += sigma[action] * cfr(agent, new_state, i, t, local_delta)
        return vo


def cfrp(
    agent: Agent,
    state: ShortDeckPokerState,
    i: int,
    t: int,
    c: int,
    local_delta: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """
    Counter factual regret minimization with pruning.

    Uses **external sampling** for opponent nodes.  Pruning skips actions
    whose regret is at or below the threshold ``c`` (except on the river,
    where all actions are always explored).

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    state : ShortDeckPokerState
        Current game state.
    i : int
        The traversing player index.
    t : int
        The iteration.
    c : int
        Floor for regret below which we do not search a node.
    local_delta : Dict[str, Dict[str, float]], optional
        Per-infoset regret accumulator.  See ``cfr()`` for full description.
    """
    ph = state.player_i

    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand:
        return state.payout[i]

    elif ph == i:
        # calculate strategy
        this_info_sets_regret = {**state.initial_regret, **agent.regret.get(state.info_set, {})}
        sigma = calculate_strategy(this_info_sets_regret)
        vo = 0.0
        voa: Dict[str, float] = {}
        # Explored dict tracks which actions were not pruned.
        explored: Dict[str, bool] = {action: False for action in state.legal_actions}
        # Disable pruning on the river — explore all actions regardless.
        is_river = state._betting_stage == "river"
        for action in state.legal_actions:
            if is_river or this_info_sets_regret[action] > c:
                new_state: ShortDeckPokerState = state.apply_action(action)
                voa[action] = cfrp(agent, new_state, i, t, c, local_delta)
                explored[action] = True
                vo += sigma[action] * voa[action]
        if local_delta is not None:
            infoset_delta = local_delta.setdefault(state.info_set, {})
            for action in state.legal_actions:
                if explored[action]:
                    infoset_delta[action] = infoset_delta.get(action, 0.0) + (voa[action] - vo)
        else:
            this_info_sets_regret = {**state.initial_regret, **agent.regret.get(state.info_set, {})}
            for action in state.legal_actions:
                if explored[action]:
                    this_info_sets_regret[action] += voa[action] - vo
            agent.regret[state.info_set] = this_info_sets_regret
        return vo
    else:
        # External sampling: iterate all opponent actions weighted by strategy.
        this_info_sets_regret = {**state.initial_regret, **agent.regret.get(state.info_set, {})}
        sigma = calculate_strategy(this_info_sets_regret)
        vo = 0.0
        for action in state.legal_actions:
            new_state: ShortDeckPokerState = state.apply_action(action)
            vo += sigma[action] * cfrp(agent, new_state, i, t, c, local_delta)
        return vo


def serialise(
    agent: Agent,
    save_path: Path,
    t: int,
    server_state: Dict[str, Union[str, float, int, None]],
    locks: dict = {},
):
    """
    Write progress of optimising agent (and server state) to file.

    Takes consistent snapshots of ``agent.regret`` and ``agent.strategy``
    under their respective locks (short critical sections — no processing
    under lock).  Builds offline dicts from snapshots without
    ``copy.deepcopy``, fixing Bug 4.

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    save_path : Path
        Directory to save checkpoint files into.
    t : int
        The iteration.
    server_state : Dict[str, Union[str, float, int, None]]
        All the variables required to resume training.
    locks : dict, optional
        Named lock mapping (``"regret"``, ``"pre_flop_strategy"``).
        Pass an empty dict (default) in single-process mode.
    """
    # Load the shared strategy that we accumulate into.
    agent_path = os.path.abspath(str(save_path / f"agent.joblib"))
    agent_path_tmp = agent_path + ".tmp"
    if os.path.isfile(agent_path):
        try:
            offline_agent = joblib.load(agent_path)
        except (EOFError, Exception):
            offline_agent = {
                "regret": {},
                "timestep": t,
                "strategy": {},
                "pre_flop_strategy": {}
            }
    else:
        offline_agent = {
            "regret": {},
            "timestep": t,
            "strategy": {},
            "pre_flop_strategy": {}
        }
    # Take short-lived snapshots under their respective locks.
    # No processing happens under the lock — only the list() call itself.
    if locks:
        locks["regret"].acquire()
    regret_snapshot = list(agent.regret.items())
    if locks:
        locks["regret"].release()
    if locks:
        locks["pre_flop_strategy"].acquire()
    strategy_snapshot = list(agent.strategy.items())
    if locks:
        locks["pre_flop_strategy"].release()
    # Calculate the strategy for each info sets regret, and accumulate in
    # the offline agent's strategy.
    for info_set, this_info_sets_regret in sorted(regret_snapshot):
        strategy = calculate_strategy(this_info_sets_regret)
        if info_set not in offline_agent["strategy"]:
            offline_agent["strategy"][info_set] = {
                action: probability for action, probability in strategy.items()
            }
        else:
            for action, probability in strategy.items():
                offline_agent["strategy"][info_set][action] = (
                    offline_agent["strategy"][info_set].get(action, 0) + probability
                )
    # Build regret and pre_flop_strategy from snapshots — no copy.deepcopy
    # (Bug 4 fix: removes the full in-memory duplicate of the regret table).
    offline_agent["regret"] = {
        info_set: dict(regret_vals) for info_set, regret_vals in regret_snapshot
    }
    offline_agent["pre_flop_strategy"] = {
        info_set: dict(strat_vals) for info_set, strat_vals in strategy_snapshot
    }
    joblib.dump(offline_agent, agent_path_tmp)
    os.replace(agent_path_tmp, agent_path)
    # Dump the server state to file too, but first update a few bits of the
    # state so when we load it next time, we start from the right place in
    # the optimisation process.
    server_path = save_path / f"server.gz"
    server_path_tmp = str(server_path) + ".tmp"
    server_state["agent_path"] = agent_path
    server_state["start_timestep"] = t + 1
    joblib.dump(server_state, server_path_tmp)
    os.replace(server_path_tmp, str(server_path))
