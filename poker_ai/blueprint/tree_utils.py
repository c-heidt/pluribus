"""Stateless game-tree primitives used by CFR training and search.

This module holds the building blocks that any CFR-style tree traversal
needs — terminal detection, legal-action extraction, regret matching,
strategy lookup, opponent sampling, and regret accumulation — in a form
that is decoupled from the training loop.  Nothing in this file
mutates training-wide state; callers pass in the tables and
accumulators explicitly.

The same primitives are used by:

- :mod:`poker_ai.blueprint.cfr` — the training traversal
  (``cfr`` / ``cfrp``).
- :mod:`poker_ai.blueprint.strategy` — the strategy-sampling traversal
  (``update_strategy``).
- Any future real-time search algorithm that needs to walk the game
  tree against a trained strategy without re-implementing the
  primitives.

All functions are pure with respect to their inputs.  ``accumulate_regrets``
is the only function that writes to a mutable container — the
caller-owned ``local_delta`` dictionary — and does not touch any shared
table.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from poker_ai.environment.action_space import (
    ACTION_TO_IDX,
    CANONICAL_ACTIONS,
    MAX_ACTIONS_PER_STREET,
)
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.environment.poker_env import PokerEnv as PokerState


def is_terminal(state: PokerState, i: int) -> Optional[float]:
    """Return the terminal payout for player *i* if the node is terminal.

    A node is terminal for the traversing player when either the hand
    has ended or the traversing player has already folded — once the
    traversing player is inactive, no future decision can change their
    payout, so the traversal can stop here and return the locked-in
    value.

    Returning ``None`` at non-terminal nodes lets callers use the
    idiom::

        value = is_terminal(state, i)
        if value is not None:
            return value

    which avoids an explicit boolean check.

    Parameters
    ----------
    state : PokerState
        Current game state.
    i : int
        Traversing player index.

    Returns
    -------
    float or None
        Player *i*'s payout at a terminal node, or ``None`` at
        non-terminal nodes.
    """
    if state.is_terminal or not state.players[i].is_active:
        return float(state.payout[i])
    return None


def get_legal_actions(state: PokerState) -> List[str]:
    """Return the legal action strings at *state*, filtering ``None`` entries.

    :attr:`PokerState.legal_actions <poker_ai.environment.poker_env.PokerEnv.legal_actions>`
    may contain ``None`` placeholders for slots that do not apply at
    the current node (e.g. inactive players).  Callers always want a
    clean list of action strings, so this helper strips the placeholders.

    Parameters
    ----------
    state : PokerState
        Current game state.

    Returns
    -------
    list[str]
        Legal abstract action strings at *state*.
    """
    return [a for a in state.legal_actions if a is not None]


def get_node_strategy(
    tables: CFRTables,
    state: PokerState,
) -> Tuple[np.ndarray, int, Dict[str, int], np.ndarray]:
    """Compute the current mixed strategy at *state* via regret matching.

    Looks up the cumulative regret row for the current information set
    from ``tables.regret[r]``.  Information sets that have never been
    visited are treated as all-zero regret vectors, yielding the
    uniform strategy.  The result is passed through
    :func:`calculate_strategy_from_row` with a mask constructed from the
    legal actions at *state* so illegal actions are guaranteed zero
    probability.

    Parameters
    ----------
    tables : CFRTables
        Per-street regret and strategy tables.
    state : PokerState
        Current game state.

    Returns
    -------
    sigma : np.ndarray
        Float32 mixed strategy over canonical actions (sums to 1).
    r : int
        Betting round index (0 = pre-flop, 3 = river).
    a_to_i : dict[str, int]
        Canonical action → column index for this street (shared reference,
        do not mutate).
    regret_row : np.ndarray
        Int32 cumulative regret vector, either the live row from the
        table or a fresh zero vector when the infoset is unseen.
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

    External sampling is the variance-reduction technique at the heart
    of MCCFR: at opponent nodes we draw a single action proportional to
    the current strategy instead of recursing into every opponent
    action.  This reduces the per-traversal branching factor by the
    number of non-traversing players.

    When all strategy probabilities at the node are zero (i.e. every
    entry of the regret row is non-positive and the caller did not mask
    the illegal actions), the function falls back to a uniform
    distribution over ``legal_actions`` so the sampler never sees a
    degenerate probability vector.

    Parameters
    ----------
    legal_actions : list[str]
        Legal abstract action strings to sample from.
    sigma : np.ndarray
        Float32 mixed strategy over *all* canonical actions for this
        street (may include zero entries for illegal actions).
    a_to_i : dict[str, int]
        Canonical action → column index for this street.

    Returns
    -------
    str
        One action string drawn from ``legal_actions``.
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

    For each explored action ``a``, the counterfactual regret
    increment is ``round(voa[a] - vo)``, where *vo* is the node value
    (strategy-weighted expected value of the explored actions) and
    ``voa[a]`` is the value obtained by playing ``a``.  Only actions
    present in ``voa`` receive an update — actions that were pruned
    (not explored) are simply not in the dict, which is the correct
    behaviour for CFR-P.

    The target row in ``local_delta`` is lazily allocated as a zero
    int64 vector on first write; subsequent calls for the same infoset
    accumulate into the same array.

    Parameters
    ----------
    local_delta : dict[tuple[int, str], np.ndarray]
        Caller-owned regret accumulator keyed by ``(betting_round,
        info_set)``.  Flushed into the shared tables later via
        :func:`poker_ai.blueprint.cfr.merge_local_delta`.
    r : int
        Betting round index for *info_set*.
    info_set : str
        Information set string for the current node.
    voa : dict[str, float]
        Counterfactual value per explored action.
    vo : float
        Node value — strategy-weighted sum of ``voa`` values.
    a_to_i : dict[str, int]
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
    """Convert a cumulative regret vector into a probability distribution.

    Implements standard regret matching: the probability mass on each
    action is proportional to its positive cumulative regret.  If every
    positive regret is zero (e.g. a fresh information set) the
    distribution falls back to uniform over the valid actions.

    When ``valid_mask`` is provided, illegal actions are zeroed out
    before the regret-matching step so the uniform fallback only
    distributes probability over the legal subset.  This keeps the
    returned vector a valid mixed strategy even at nodes where most
    canonical actions are illegal (e.g. pre-flop where only
    fold/call/raise apply).

    Parameters
    ----------
    regret_row : np.ndarray
        1-D array of per-action cumulative regrets indexed by the
        canonical action ordering for this street.  Typically int32 but
        any numeric dtype is accepted and internally promoted to float32.
    valid_mask : np.ndarray, optional
        Boolean mask of the same length as ``regret_row`` marking legal
        actions.  When omitted, all actions are treated as legal.

    Returns
    -------
    np.ndarray
        Float32 probability vector of shape ``(n_actions,)`` that sums
        to 1 (or to 0 if no actions are valid, which should not occur
        in a well-formed game state).
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
