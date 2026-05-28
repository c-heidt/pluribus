"""Per-search context for the depth-limited subgame solver.

:class:`SubgameContext` carries the inputs to one ``solve()`` call that
do not change during the CFR walk: opponent ranges, board-conflict
mask, bot identity, depth limit, leaf config, and RNG.  Construction
lives on the dataclass as a classmethod because most fields are derived
from the env or are simple per-hand state — see §6.1 of
``docs/subgame_solving.md`` for the rationale.

The class is frozen; the solver is expected to be a pure function of
``(root_env, ctx, cfg)``.  All per-iteration state lives on the solver's
stack, not on the context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

from environment.poker_env import PokerEnv

if TYPE_CHECKING:
    # LeafConfig is introduced in §6.4 (poker_ai/search/leaf.py) which
    # is not yet implemented.  The forward reference keeps this module
    # importable in the meantime; downstream code passes any object
    # whose duck-typed interface matches what leaf.py will expose.
    from poker_ai.search.leaf import LeafConfig


Range = np.ndarray
"""Float32 vector of shape ``(env.n_combos,)`` representing per-combo
weights.  Owned by :mod:`poker_ai.search.ranges` (§6.2)."""


@dataclass(frozen=True)
class SubgameContext:
    """Static-for-one-search inputs to the depth-limited subgame solver.

    Attributes
    ----------
    my_seat : int
        Seat index of the bot.
    my_hole : tuple[int, int]
        Bot's hole cards.  The bot's range is implicitly the Dirac at
        this combo — not represented as an entry in ``opponent_ranges``.
    opponent_ranges : dict[int, Range]
        Per-seat ranges for every opponent still live in the hand.
        Seats that have folded or busted are omitted.
    board_compatible : numpy.ndarray
        Boolean mask of shape ``(env.n_combos,)``.  ``True`` for combos
        that share no card with the current community.
    street_at_root : int
        ``env.betting_round`` at the root of the search.  The solver
        halts and calls ``leaf.leaf_value`` when
        ``env.betting_round > street_at_root``.
    leaf : LeafConfig
        Continuation-strategy configuration for depth-limit leaves.
    rng : numpy.random.Generator
        Source of randomness; passed to the leaf rollouts and any
        sampling done by the solver.
    """

    my_seat: int
    my_hole: Tuple[int, int]
    opponent_ranges: Dict[int, Range]
    board_compatible: np.ndarray
    street_at_root: int
    leaf: "LeafConfig"
    rng: np.random.Generator

    @classmethod
    def from_runtime(
        cls,
        env: PokerEnv,
        my_seat: int,
        my_hole: Tuple[int, int],
        opponent_ranges: Dict[int, Range],
        leaf: "LeafConfig",
        rng: np.random.Generator,
    ) -> "SubgameContext":
        """Build a context for a search rooted at ``env``.

        Derives ``board_compatible`` from the env's combo table and
        community cards; sets ``street_at_root`` to ``env.betting_round``.
        Does not deepcopy the env — the solver's caller (typically
        ``SearchAgent``) owns that.
        """
        return cls(
            my_seat=my_seat,
            my_hole=my_hole,
            opponent_ranges=opponent_ranges,
            board_compatible=_board_compatible_mask(env),
            street_at_root=env.betting_round,
            leaf=leaf,
            rng=rng,
        )


def _board_compatible_mask(env: PokerEnv) -> np.ndarray:
    """Boolean mask over ``env.combo_cards``: True iff the combo shares
    no card with the current community.

    Pre-flop (empty community) every combo is compatible.
    """
    if not env.community_cards:
        return np.ones(env.n_combos, dtype=bool)
    board = np.asarray(env.community_cards, dtype=np.int32)
    cc = env.combo_cards
    return ~(np.isin(cc[:, 0], board) | np.isin(cc[:, 1], board))
