"""Per-street action-space lookup tables.

CFR works over a finite abstract action set that must stay identical
across every code path touching the regret and strategy tables.  Each
information set's row in a :class:`~poker_ai.tables.chunked_table.ChunkedTable`
is indexed by the *canonical* action ordering of its betting round;
mixing up the ordering between writer and reader would silently corrupt
regrets.

This module builds three lookup tables once at import time from
:meth:`PokerEnv.get_canonical_actions <poker_ai.environment.poker_env.PokerEnv.get_canonical_actions>`
so every downstream module (``tree_utils``, ``cfr``, ``strategy``, the
server and the training loop) can import a single source of truth.

The tables are immutable dictionaries; callers never need to recompute
them and must not mutate them in place.

Attributes
----------
CANONICAL_ACTIONS : dict[int, list[str]]
    Ordered list of abstract action strings for each betting round
    (``0`` = pre-flop … ``3`` = river).  The order is stable across
    runs — it defines the column layout of the regret/strategy rows.
ACTION_TO_IDX : dict[int, dict[str, int]]
    Mapping from action string to its column index, per betting round.
    Used by CFR code to write a regret delta into the correct slot and
    by strategy traversal to look up the sampled action's slot.
MAX_ACTIONS_PER_STREET : dict[int, int]
    Row width (number of action columns) of each
    :class:`~poker_ai.tables.chunked_table.ChunkedTable`, keyed by betting
    round.  Used to size the per-infoset regret/strategy arrays and to
    pre-allocate empty rows.
"""

from typing import Dict, List

from poker_ai.environment.poker_env import PokerEnv as PokerState

CANONICAL_ACTIONS: Dict[int, List[str]] = {
    r: PokerState.get_canonical_actions(r) for r in range(4)
}
"""Canonical abstract action list per betting round, in stable order."""

ACTION_TO_IDX: Dict[int, Dict[str, int]] = {
    r: {a: i for i, a in enumerate(CANONICAL_ACTIONS[r])} for r in range(4)
}
"""Mapping from action string to column index for a given betting round."""

MAX_ACTIONS_PER_STREET: Dict[int, int] = {
    r: len(CANONICAL_ACTIONS[r]) for r in range(4)
}
"""Column width of each ChunkedTable, keyed by betting round."""
