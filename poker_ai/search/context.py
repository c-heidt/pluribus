"""Per-search context for the depth-limited subgame solver.

:class:`SubgameContext` carries the inputs to one ``solve()`` call that
do not change during the CFR walk: per-seat ranges (every live seat,
including the bot), the fold-time ranges of seats that folded before the
root, the board-conflict mask, bot identity, the depth-limit descriptor,
leaf config, and RNG.  Construction lives on the dataclass as a
classmethod because most fields are derived from the env or are simple
per-hand state — see §6.1 of ``docs/subgame_solving.md`` for the rationale.

The class is frozen and a **passive carrier**: it does not validate seat
membership (the "every live seat incl. the bot" contract is upheld by the
agent / range tracker that populates it).  In addition, the constructed
instance defends against accidental mutation of the *contents* of its
container fields: ``ranges`` and ``folded_ranges`` are exposed as
read-only mappings over arrays whose ``writeable`` flag has been cleared,
and ``board_compatible`` is a read-only view.  This keeps the solver a
pure function of ``(root_env, ctx, cfg)`` — an inner loop that
accidentally writes to a range or the board mask fails fast instead of
silently corrupting the search.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, Mapping, Tuple

import numpy as np
from typing_extensions import Literal

from environment.poker_env import PokerEnv
from poker_ai.search.ranges import Range

if TYPE_CHECKING:
    # LeafConfig is introduced in §6.4 (poker_ai/search/leaf.py) which
    # is not yet implemented.  The forward reference keeps this module
    # importable in the meantime; downstream code passes any object
    # whose duck-typed interface matches what leaf.py will expose.
    from poker_ai.search.leaf import LeafConfig


DepthVerdict = Literal["internal", "leaf", "terminal"]
"""Classification of an env reached during the CFR walk (§3 depth limits)."""


@dataclass(frozen=True)
class DepthLimit:
    """Round-dependent depth-limit descriptor (§3 table).

    Derived once at the root from ``(street_at_root, players live at round
    start)``, it classifies any env reached during the walk into one of
    three verdicts via :meth:`classify`:

    - ``"terminal"``  — the hand is over (``env.payout`` scores it);
    - ``"leaf"``      — a depth-limit leaf (continuation meta-game, §6.5);
    - ``"internal"``  — keep recursing.

    Attributes
    ----------
    street_at_root : int
        ``env.betting_round`` at the subgame root (0=pre-flop .. 3=river).
    n_players_at_root : int
        ``env.n_players_started_round`` at the root — distinguishes the
        multiway round-2 case (the only one with the after-2nd-raise
        cutoff) from heads-up round 2.
    """

    street_at_root: int
    n_players_at_root: int

    def classify(self, env: PokerEnv) -> DepthVerdict:
        """Verdict for ``env`` under the §3 depth-limit rules.

        ``is_terminal`` is checked **before** ``betting_round`` because
        the latter raises at the terminal stage.
        """
        if env.is_terminal:
            return "terminal"
        r = env.betting_round
        if self.street_at_root == 0:
            # Round-1 search: subgame ends at the end of round 1.
            return "leaf" if r > 0 else "internal"
        if self.street_at_root == 1 and self.n_players_at_root > 2:
            # Multiway round-2: cut off at the start of round 3 (the turn)
            # or immediately after the 2nd raise of the flop, whichever
            # comes first.
            if r > 1:
                return "leaf"
            if r == 1 and env.n_raises_this_round >= 2:
                return "leaf"
            return "internal"
        # Round-2 heads-up and rounds 3-4: subgame extends to the end of
        # the game — terminal leaves only, never a depth-limit leaf.
        return "internal"


@dataclass(frozen=True)
class SubgameContext:
    """Static-for-one-search inputs to the depth-limited subgame solver.

    Attributes
    ----------
    my_seat : int
        Seat index of the bot.
    my_hole : tuple[int, int]
        Bot's actual hole cards — the row the agent ultimately plays.
        The bot's *range* is carried separately in ``ranges`` (observer
        perspective), so the search solves over its whole range.
    ranges : Mapping[int, Range]
        Per-seat ranges for every seat still live in the hand, **including
        ``my_seat``** (the bot's observer-perspective range, excluding only
        board conflicts).  Seats that folded or busted before the root are
        not here (see ``folded_ranges``).
    folded_ranges : Mapping[int, Range]
        Fold-time marginals of seats that folded before the root, retained
        for card removal.  May be empty.
    board_compatible : numpy.ndarray
        Boolean mask of shape ``(env.n_combos,)``.  ``True`` for combos
        that share no card with the current community.
    street_at_root : int
        ``env.betting_round`` at the root of the search.
    depth_limit : DepthLimit
        Descriptor implementing the §3 depth-limit table; classifies each
        env reached during the walk as internal / leaf / terminal.
    leaf : LeafConfig
        Continuation-strategy configuration for depth-limit leaves.
    rng : numpy.random.Generator
        Source of randomness; passed to the leaf rollouts and any
        sampling done by the solver.
    """

    my_seat: int
    my_hole: Tuple[int, int]
    ranges: Mapping[int, Range]
    folded_ranges: Mapping[int, Range]
    board_compatible: np.ndarray
    street_at_root: int
    depth_limit: DepthLimit
    leaf: "LeafConfig"
    rng: np.random.Generator

    @classmethod
    def from_runtime(
        cls,
        env: PokerEnv,
        my_seat: int,
        my_hole: Tuple[int, int],
        ranges: Dict[int, Range],
        folded_ranges: Dict[int, Range],
        leaf: "LeafConfig",
        rng: np.random.Generator,
    ) -> "SubgameContext":
        """Build a context for a search rooted at ``env``.

        Derives ``board_compatible`` from the env's combo table and
        community cards; sets ``street_at_root`` to ``env.betting_round``
        and ``depth_limit`` from ``env.betting_round`` /
        ``env.n_players_started_round``.  Does not deepcopy the env — the
        solver's caller (typically ``SearchAgent``) owns that.

        Container fields are wrapped read-only so the solver cannot
        accidentally mutate ranges or the board mask mid-search:
        ``ranges`` and ``folded_ranges`` are each exposed as a
        :class:`MappingProxyType` over defensive array copies whose
        ``writeable`` flag is cleared, and ``board_compatible`` is set
        non-writeable.  The caller's original dicts and arrays are
        unaffected.
        """
        board_mask = _board_compatible_mask(env)
        board_mask.flags.writeable = False
        return cls(
            my_seat=my_seat,
            my_hole=my_hole,
            ranges=_freeze_ranges(ranges),
            folded_ranges=_freeze_ranges(folded_ranges),
            board_compatible=board_mask,
            street_at_root=env.betting_round,
            depth_limit=DepthLimit(env.betting_round, env.n_players_started_round),
            leaf=leaf,
            rng=rng,
        )


def _freeze_ranges(ranges: Mapping[int, Range]) -> Mapping[int, Range]:
    """Read-only view over defensive, non-writeable copies of ``ranges``.

    The caller's input dict and arrays stay mutable; only the returned
    mapping (and the arrays it holds) are frozen.
    """
    frozen: Dict[int, Range] = {}
    for seat, weights in ranges.items():
        ro = np.array(weights, copy=True)
        ro.flags.writeable = False
        frozen[seat] = ro
    return MappingProxyType(frozen)


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
