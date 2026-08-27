"""Per-search context for the depth-limited subgame solver (§6.1).

:class:`SubgameContext` carries the inputs to one ``solve()`` call that do not change
during the CFR walk.  It is frozen and a **passive carrier**: it does not validate
seat membership (the "every live seat incl. the bot" contract is upheld by the agent /
range tracker that populates it).

Container *contents* are frozen too — ``ranges`` / ``folded_ranges`` are read-only
mappings over non-writeable arrays, and ``board_compatible`` is a read-only view — so
the solver stays a pure function of ``(root_env, ctx, cfg)`` and an inner loop that
writes to a range fails fast instead of silently corrupting the search.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Tuple

import numpy as np
from typing_extensions import Literal

from environment.poker_env import PokerEnv
from poker_ai.search.ranges import Range
from poker_ai.search.rng import spawn_one

if TYPE_CHECKING:
    from poker_ai.search.leaf import LeafConfig
    from poker_ai.modeling.model import OpponentModel


DepthVerdict = Literal["internal", "leaf", "terminal"]
"""Classification of an env reached during the CFR walk (§3 depth limits)."""


@dataclass(frozen=True)
class DepthLimit:
    """Round-dependent depth-limit descriptor (§3 table).

    Derived once at the root, it classifies any env reached during the walk as
    ``"terminal"`` (``env.payout`` scores it), ``"leaf"`` (a depth-limit leaf, §6.5)
    or ``"internal"`` (keep recursing).

    ``n_players_at_root`` is ``env.n_players_started_round``, which distinguishes the
    multiway round-2 case (the only one with the after-2nd-raise cutoff) from heads-up
    round 2.
    """

    street_at_root: int
    n_players_at_root: int

    def classify(self, env: PokerEnv) -> DepthVerdict:
        """Verdict for ``env`` under the §3 depth-limit rules.

        ``is_terminal`` is checked **before** ``betting_round`` because the latter
        raises at the terminal stage.
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
    my_seat, my_hole
        The bot's seat and its actual hole cards — the row the agent ultimately
        plays.  Its *range* is carried separately in ``ranges``, so the search
        solves over the whole range.
    ranges : Mapping[int, Range]
        Per-seat ranges for every seat still live, **including ``my_seat``**
        (observer perspective, excluding only board conflicts).
    folded_ranges : Mapping[int, Range]
        Fold-time marginals of seats that folded before the root, retained for card
        removal.  May be empty.
    board_compatible : numpy.ndarray
        Shape ``(env.n_combos,)``; True for combos sharing no card with the community.
    street_at_root : int
        ``env.betting_round`` at the root.
    depth_limit : DepthLimit
        Classifies each env reached during the walk as internal / leaf / terminal.
    leaf : LeafConfig
        Continuation-strategy configuration for depth-limit leaves.
    rng : numpy.random.Generator
        Sampling stream: hole draws and the leaf rollout's action draws.
    board_rng : numpy.random.Generator, optional
        Separate stream for the leaf rollout's **board** runout, because a board draw
        interleaved into the sampling stream desyncs the sampling trajectory.
        :meth:`from_runtime` derives one; ``None`` falls back to ``rng`` (never to the
        global ``np.random`` — see :mod:`poker_ai.search.rng`).
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
    #: seat → opponent model (opponent_modeling §5.1).  Empty by default, and an empty
    #: mapping activates no code path — the clamp early-outs, so an unmodeled solve is
    #: bit-for-bit vanilla.  See :func:`poker_ai.search.vform.apply_model_clamp`.
    models: Mapping[int, "OpponentModel"] = MappingProxyType({})
    #: Board-runout stream for leaf rollouts; see the class docstring.
    board_rng: Optional[np.random.Generator] = None

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
        models: Optional[Mapping[int, "OpponentModel"]] = None,
        board_rng: Optional[np.random.Generator] = None,
    ) -> "SubgameContext":
        """Build a context for a search rooted at ``env``.

        Does not deepcopy the env — the caller (typically ``SearchAgent``) owns that.
        Container fields are wrapped read-only (the caller's own dicts and arrays are
        unaffected).

        ``board_rng`` defaults to a child of ``rng``, *spawned* rather than drawn from,
        so ``rng`` is not advanced and the solver's sampling trajectory is unchanged by
        the existence of a separate board stream.
        """
        board_mask = _board_compatible_mask(env)
        board_mask.flags.writeable = False
        return cls(
            board_rng=board_rng if board_rng is not None else spawn_one(rng),
            my_seat=my_seat,
            my_hole=my_hole,
            ranges=_freeze_ranges(ranges),
            folded_ranges=_freeze_ranges(folded_ranges),
            board_compatible=board_mask,
            street_at_root=env.betting_round,
            depth_limit=DepthLimit(env.betting_round, env.n_players_started_round),
            leaf=leaf,
            rng=rng,
            models=MappingProxyType(dict(models)) if models else MappingProxyType({}),
        )


def _freeze_ranges(ranges: Mapping[int, Range]) -> Mapping[int, Range]:
    """Read-only view over defensive, non-writeable copies of ``ranges``.

    The caller's input dict and arrays stay mutable.
    """
    frozen: Dict[int, Range] = {}
    for seat, weights in ranges.items():
        ro = np.array(weights, copy=True)
        ro.flags.writeable = False
        frozen[seat] = ro
    return MappingProxyType(frozen)


def _board_compatible_mask(env: PokerEnv) -> np.ndarray:
    """Boolean mask over ``env.combo_cards``: True iff the combo shares no card with
    the current community (every combo pre-flop)."""
    if not env.community_cards:
        return np.ones(env.n_combos, dtype=bool)
    board = np.asarray(env.community_cards, dtype=np.int32)
    cc = env.combo_cards
    return ~(np.isin(cc[:, 0], board) | np.isin(cc[:, 1], board))
