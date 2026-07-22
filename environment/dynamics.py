"""Deterministic game transition functions for the poker environment.

All functions in this module operate on a ``PokerEnv`` instance passed
as their first argument.  A single ``Evaluator`` instance is created at
module import time and shared across all calls.
"""

from __future__ import annotations

import collections
import logging
from typing import TYPE_CHECKING

from environment.evaluator import default_evaluator as _evaluator

if TYPE_CHECKING:
    from environment.poker_env import PokerEnv
    from environment.player import Player

logger = logging.getLogger(__name__)

# The shared evaluator now lives in :mod:`environment.evaluator` as
# ``default_evaluator`` (imported above as ``_evaluator`` for backward
# compatibility).  Hosting it at the leaf evaluator layer keeps the
# range-showdown settlement free of any dependency on this module.


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
    # Snapshot the per-seat contributions *before* the pot is reset, so the
    # terminal's matched/contested stake stays available to the payout evaluators
    # (the smaller of two heads-up contributions is the winner-takes amount).
    env._terminal_contributions = tuple(env.pot.capture())
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
    """Return True if any live player has not yet matched the largest bet.

    "Live" means active (not folded) and not all-in — a player who still has a
    betting decision.  The comparison is against the maximum bet among **all**
    active players, *including all-in players*: an all-in raise over the top of
    the current bet leaves the live players owing a call/fold decision, so more
    betting is still needed even though the live players' bets are equal *to
    each other*.  Comparing live bets only to each other (the previous
    behaviour) silently treated an unmatched over-the-top all-in as "betting
    complete" and advanced the street without giving the opponents a chance to
    respond — the round-advance twin of the ``_hand_over`` all-in contract.

    Parameters
    ----------
    env : PokerEnv
        The game environment.

    Returns
    -------
    bool
        ``True`` if at least one active non-all-in player has bet less than the
        largest amount committed by any active player this round; ``False``
        otherwise (every live player has matched the top bet, or no live player
        remains).
    """
    active = [p for p in env.players if p.is_active]
    live = [p for p in active if not p.is_all_in]
    if not live:
        return False
    max_bet = max(p.n_bet_chips for p in active)
    return any(p.n_bet_chips < max_bet for p in live)
