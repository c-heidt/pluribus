"""Deterministic game transition functions for the poker environment.

All functions in this module operate on a ``PokerEnv`` instance passed
as their first argument.  A single ``Evaluator`` instance is created at
module import time and shared across all calls.
"""

from __future__ import annotations

import collections
import logging
from typing import TYPE_CHECKING

from environment.evaluator import Evaluator

if TYPE_CHECKING:
    from environment.poker_env import PokerEnv
    from environment.player import Player

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton: immutable after init, NEVER deep-copied.
# ---------------------------------------------------------------------------
_evaluator: Evaluator = Evaluator()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def assign_blinds(env: PokerEnv) -> None:
    """Post small and big blinds for the first two players.

    Parameters
    ----------
    env : PokerEnv
        The game environment. ``env.players[0]`` posts the small blind
        and ``env.players[1]`` posts the big blind.
    """
    env.players[0].add_to_pot(env.pot, env.small_blind)
    env.players[1].add_to_pot(env.pot, env.big_blind)
    logger.debug(
        "Assigned blinds to players %s (SB) and %s (BB)",
        env.players[0].name,
        env.players[1].name,
    )


def assign_order(env: PokerEnv) -> None:
    """Set ``player.order`` for every player based on list position.

    Parameters
    ----------
    env : PokerEnv
        The game environment.
    """
    for idx, player in enumerate(env.players):
        player.order = idx


def rotate_blinds(env: PokerEnv) -> None:
    """Rotate the player list so the next player posts the small blind.

    The player at index 0 is moved to the end of the list.

    Parameters
    ----------
    env : PokerEnv
        The game environment.
    """
    env.players.append(env.players.pop(0))


def advance_stage(env: PokerEnv) -> None:
    """Deal community cards for the next betting stage.

    Called at the end of a betting round to transition the board
    state. Also resets ``n_bet_chips`` on every player so
    bet-equality checks start fresh for the new round.

    Parameters
    ----------
    env : PokerEnv
        The game environment. ``env._betting_stage`` determines how
        many cards to deal (3 for pre_flop→flop, 1 for subsequent
        transitions).
    """
    stage = env._betting_stage
    if stage == "pre_flop":
        env.community_cards += env.deck.deal_community(3)
    elif stage == "flop":
        env.community_cards += env.deck.deal_community(1)
    elif stage == "turn":
        env.community_cards += env.deck.deal_community(1)
    # "river" → "show_down" needs no deal
    for player in env.players:
        player.n_bet_chips = 0


def rank_players_by_best_hand(env: PokerEnv) -> list:
    """Rank active players by hand strength.

    Parameters
    ----------
    env : PokerEnv
        The game environment. ``env.community_cards`` and each
        player's ``_cards`` must be populated before calling.

    Returns
    -------
    list[list[Player]]
        Players grouped by hand rank, ordered best-hand-first.
        Players who share the same rank appear in the same inner list.
    """
    grouped: dict = collections.defaultdict(list)
    for player in env.players:
        if player.is_active:
            rank = _evaluator.evaluate(
                list(env.community_cards), list(player._cards)
            )
            hand_class = _evaluator.get_rank_class(rank)
            hand_desc = _evaluator.class_to_string(hand_class).lower()
            logger.debug("Rank #%d  %s  %s", rank, player, hand_desc)
            grouped[rank].append(player)
    return [grouped[r] for r in sorted(grouped.keys())]


def compute_winners(env: PokerEnv) -> None:
    """Determine hand winners, distribute chips, and reset the pot.

    Ranks all active players' hands, delegates payout computation to
    ``env.pot.compute_utility``, then adds winnings back to each
    player's chip stack.

    Parameters
    ----------
    env : PokerEnv
        The game environment. Modified in-place: player chip counts
        are updated and the pot is reset to zero.
    """
    ranked = rank_players_by_best_hand(env)
    payouts = env.pot.compute_utility(env.players, ranked)
    env.pot.reset()
    for player in env.players:
        player.add_chips(payouts[player.player_i])
    logger.debug("Winnings computation complete.")
    for player in env.players:
        logger.debug("%s", player)


# ---------------------------------------------------------------------------
# Properties / computed state (called from PokerEnv properties)
# ---------------------------------------------------------------------------


def n_active_players(env: PokerEnv) -> int:
    """Count players who have not folded.

    Parameters
    ----------
    env : PokerEnv
        The game environment.

    Returns
    -------
    int
        Number of players with ``is_active == True``.
    """
    return sum(1 for p in env.players if p.is_active)


def n_players_with_moves(env: PokerEnv) -> int:
    """Count players who can still make a betting decision.

    A player can act if they are active (not folded) and not all-in.

    Parameters
    ----------
    env : PokerEnv
        The game environment.

    Returns
    -------
    int
        Number of active, non-all-in players.
    """
    return sum(1 for p in env.players if p.is_active and not p.is_all_in)


def more_betting_needed(env: PokerEnv) -> bool:
    """Return True if active non-all-in players have unequal bets.

    Parameters
    ----------
    env : PokerEnv
        The game environment.

    Returns
    -------
    bool
        ``True`` if at least two active non-all-in players have
        contributed different amounts this round; ``False`` otherwise.
    """
    active_bets = [
        p.n_bet_chips for p in env.players if p.is_active and not p.is_all_in
    ]
    if len(active_bets) <= 1:
        return False
    return not all(b == active_bets[0] for b in active_bets)
