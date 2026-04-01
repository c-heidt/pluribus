import time

from plot import PokerPlot
from poker_ai.environment.player import Player
from poker_ai.environment.game_state import PokerState
from poker_ai.environment.pot import Pot


def get_state() -> PokerState:
    """Gets a state to visualise"""
    n_players = 6
    pot = Pot()
    players = [
        Player(player_i=player_i, initial_chips=10000, pot=pot)
        for player_i in range(n_players)
    ]
    return PokerState(players=players, load_card_lut=False)


pp: PokerPlot = PokerPlot()
state: PokerState = get_state()
time.sleep(5)
print("updating state")
pp.update_state(state)
