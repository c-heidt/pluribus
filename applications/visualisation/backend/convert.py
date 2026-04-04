from typing import Any, Dict

from poker_ai.environment.player import Player
from poker_ai.environment.utils import card_rank_char, card_suit_str

_colours = ["cyan", "lightcoral", "crimson", "#444", "forestgreen", "goldenrod", "gold"]
_suit_lut = {"spades": "P", "diamonds": "D", "clubs": "C", "hearts": "H"}


def to_player_dict(player_i: int, player: Player) -> Dict[str, Any]:
    """Create dictionary to describe player for frontend."""
    return {
        "name": player.name,
        "color": _colours[player_i],
        "bank": player.n_chips,
        "onTable": player.n_bet_chips,
        "hasCards": True,
    }


def to_card_dict(card: int) -> Dict[str, str]:
    """Create dictionary to describe card for frontend."""
    return {
        "f": _suit_lut[card_suit_str(card)],
        "v": card_rank_char(card),
    }
