"""Action-space constants for all four betting streets.

Built once at import time from ``PokerState.get_canonical_actions()``.
These are pure lookup tables — no algorithmic logic.

Imported by ``tree_utils``, ``cfr``, ``strategy``, ``server``, and
``train`` to share a single, consistent action ordering.
"""

from typing import Dict, List

from poker_ai.environment.poker_env import PokerEnv as PokerState

CANONICAL_ACTIONS: Dict[int, List[str]] = {
    r: PokerState.get_canonical_actions(r) for r in range(4)
}
"""Full abstract action list per betting round (0=pre_flop … 3=river), in stable order."""

ACTION_TO_IDX: Dict[int, Dict[str, int]] = {
    r: {a: i for i, a in enumerate(CANONICAL_ACTIONS[r])} for r in range(4)
}
"""Mapping from action string to column index for a given betting round."""

MAX_ACTIONS_PER_STREET: Dict[int, int] = {
    r: len(CANONICAL_ACTIONS[r]) for r in range(4)
}
"""Column width of each ChunkedTable, keyed by betting round."""
