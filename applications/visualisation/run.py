import time

from plot import PokerPlot
from poker_ai.environment.poker_env import PokerEnv, new_game


def get_state() -> PokerEnv:
    """Gets a state to visualise"""
    return new_game(n_players=6, info_set_lut={})


pp: PokerPlot = PokerPlot()
state: PokerEnv = get_state()
time.sleep(5)
print("updating state")
pp.update_state(state)
