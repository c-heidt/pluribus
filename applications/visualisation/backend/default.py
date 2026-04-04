from poker_ai.environment.poker_env import PokerEnv, new_game


def default_state_to_visualise() -> PokerEnv:
    """"""
    return new_game(
        n_players=3,
        info_set_lut={},
        pickle_dir="../../research/blueprint_algo/",
    )
