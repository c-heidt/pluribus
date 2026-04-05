"""Shared game-tree primitives for CFR training and real-time search.

All functions in this module are stateless and have no dependency on
CFR-specific concepts like regret deltas or training iterations.
Real-time search algorithms can import directly from here without
touching the CFR training machinery.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from poker_ai.ai.action_space import (
    ACTION_TO_IDX,
    CANONICAL_ACTIONS,
    MAX_ACTIONS_PER_STREET,
)
from poker_ai.ai.cfr_tables import CFRTables
from poker_ai.environment.poker_env import PokerEnv as PokerState


def is_terminal(state: PokerState, i: int) -> Optional[float]:
    """Return the terminal payout for player *i* if the node is terminal.

    A node is treated as terminal when the hand has ended *or* when the
    traversing player has folded (they can no longer affect the outcome).

    Returns ``None`` at non-terminal nodes so callers can write::

        value = is_terminal(state, i)
        if value is not None:
            return value
    """
    if state.is_terminal or not state.players[i].is_active:
        return float(state.payout[i])
    return None


def get_legal_actions(state: PokerState) -> List[str]:
    """Return the list of legal action strings at *state*, filtering out ``None``.

    ``PokerState.legal_actions`` may contain ``None`` entries for inactive
    players; this function strips them so callers always receive clean strings.
    """
    return [a for a in state.legal_actions if a is not None]


def get_node_strategy(
    tables: CFRTables,
    state: PokerState,
) -> Tuple[np.ndarray, int, Dict[str, int], np.ndarray]:
    """Compute the current mixed strategy at *state* via regret matching.

    Looks up the cumulative regret row for the current information set from
    ``tables.regret[r]``.  If the infoset has never been visited the regret
    row is treated as all-zeros (uniform strategy).

    Parameters
    ----------
    tables:
        CFR tables containing per-street regret data.
    state:
        Current game state.

    Returns
    -------
    sigma : np.ndarray
        Float32 strategy vector over canonical actions (sums to 1).
    r : int
        Betting round index (0–3).
    a_to_i : Dict[str, int]
        Canonical action → column index for this street.
    regret_row : np.ndarray
        Int32 cumulative regret vector (zero-vector if unseen).
    """
    r = state.betting_round
    legal_actions = get_legal_actions(state)
    canonical = CANONICAL_ACTIONS[r]
    legal_set = set(legal_actions)
    valid_mask = np.array([a in legal_set for a in canonical], dtype=bool)

    row = tables.regret[r].get_row_if_exists(state.info_set)
    regret_row = (
        row if row is not None
        else np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int32)
    )
    sigma = calculate_strategy_from_row(regret_row, valid_mask)
    a_to_i = ACTION_TO_IDX[r]
    return sigma, r, a_to_i, regret_row


def sample_action(
    legal_actions: List[str],
    sigma: np.ndarray,
    a_to_i: Dict[str, int],
) -> str:
    """Sample one action from the current strategy (external sampling).

    Used for opponent nodes: instead of exploring all opponent actions,
    external sampling draws a single action proportional to the current
    strategy, reducing the branching factor by the number of players.

    Falls back to uniform if all strategy probabilities are zero.
    """
    probs = np.array([sigma[a_to_i[a]] for a in legal_actions], dtype=np.float64)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs /= prob_sum
    else:
        probs[:] = 1.0 / len(legal_actions)
    return np.random.choice(legal_actions, p=probs)


def accumulate_regrets(
    local_delta: Dict[Tuple[int, str], np.ndarray],
    r: int,
    info_set: str,
    voa: Dict[str, float],
    vo: float,
    a_to_i: Dict[str, int],
) -> None:
    """Write counterfactual regret increments into *local_delta*.

    Only actions present in *voa* receive an update — actions that were
    pruned (i.e. not explored) are simply not included in *voa* and are
    therefore skipped, which is the correct behaviour for CFR-P.

    Parameters
    ----------
    local_delta:
        Lock-free per-infoset accumulator keyed by ``(r, info_set)``.
    r:
        Betting round index.
    info_set:
        Information set string for the current node.
    voa:
        Counterfactual value per explored action.
    vo:
        Node value (weighted sum over all explored actions).
    a_to_i:
        Canonical action → column index for this street.
    """
    key = (r, info_set)
    if key not in local_delta:
        local_delta[key] = np.zeros(MAX_ACTIONS_PER_STREET[r], dtype=np.int64)
    for action, cfv in voa.items():
        local_delta[key][a_to_i[action]] += int(round(cfv - vo))


def calculate_strategy_from_row(
    regret_row: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert a cumulative regret array into a probability distribution.

    Applies regret matching: only positive regrets contribute to the
    strategy; if all regrets are non-positive, the result is uniform over
    valid actions.

    Parameters
    ----------
    regret_row:
        1-D int32 (or float-compatible) array of per-action cumulative regrets
        indexed by the canonical action ordering for this street.
    valid_mask:
        Boolean mask of length ``n_actions``.  When provided, invalid actions
        are zeroed before regret matching so the uniform fallback distributes
        probability only over legal actions.

    Returns
    -------
    np.ndarray
        1-D float32 probability array of shape ``(n_actions,)`` summing to 1.
    """
    masked = regret_row.astype(np.float32)
    if valid_mask is not None:
        masked[~valid_mask] = 0.0
    positive = np.maximum(masked, 0.0)
    total = float(positive.sum())
    if total > 0.0:
        return (positive / total).astype(np.float32)
    result = np.zeros(len(regret_row), dtype=np.float32)
    if valid_mask is not None:
        n_valid = int(valid_mask.sum())
        if n_valid > 0:
            result[valid_mask] = 1.0 / n_valid
    else:
        result[:] = 1.0 / len(regret_row)
    return result
