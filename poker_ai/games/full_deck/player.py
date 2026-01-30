from poker_ai.poker.player import Player
from poker_ai.poker.pot import Pot


class FullDeckPokerPlayer(Player):
    """Player for Full Deck (Texas Hold'em) poker.

    Inherits from Player which interfaces with the PokerEngine.
    Manages player state including chips, private cards, and fold status.
    """

    def __init__(self, player_i: int, initial_chips: int, pot: Pot):
        """Instantiate a player for Full Deck poker."""
        super().__init__(
            name=f"player_{player_i}", initial_chips=initial_chips, pot=pot,
        )
        self.is_turn = False
