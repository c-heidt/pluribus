from poker_ai.environment.player import Player
from poker_ai.environment.game_state import PokerState
from poker_ai.environment.pot import Pot


def default_state_to_visualise() -> PokerState:
    """"""
    pot = Pot()
    n_players = 3
    players = [
        Player(player_i=player_i, initial_chips=10000, pot=pot)
        for player_i in range(n_players)
    ]
    return PokerState(
        players=players, pickle_dir="../../research/blueprint_algo/"
    )


