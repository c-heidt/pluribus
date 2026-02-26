from __future__ import annotations

import json
import logging
import operator
import os
from typing import Dict, List, Optional, Tuple

import joblib

from poker_ai import utils
from poker_ai.games.base.state import PokerState, InfoSetLookupTable
from poker_ai.games.full_deck.player import FullDeckPokerPlayer
from poker_ai.poker.evaluation.evaluator import Evaluator
from poker_ai.poker.pot import Pot

logger = logging.getLogger("poker_ai.games.full_deck.state")


def new_game(
    n_players: int, card_info_lut: InfoSetLookupTable = None, **kwargs
) -> FullDeckPokerState:
    """
    Create a new game of full deck poker (Texas Hold'em).

    ...

    Parameters
    ----------
    n_players : int
        Number of players.
    card_info_lut : InfoSetLookupTable
        Card information cluster lookup table.

    Returns
    -------
    state : FullDeckPokerState
        Current state of the game
    """
    pot = Pot()
    players = [
        FullDeckPokerPlayer(player_i=player_i, initial_chips=10000, pot=pot)
        for player_i in range(n_players)
    ]
    if card_info_lut is not None:
        # Don't reload massive files, it takes ages.
        state = FullDeckPokerState(
            players=players,
            load_card_lut=False,
            **kwargs
        )
        state.card_info_lut = card_info_lut
    else:
        # Load massive files.
        state = FullDeckPokerState(
            players=players,
            **kwargs
        )
    return state


class FullDeckPokerState(PokerState):
    """The state of a Full Deck Poker (Texas Hold'em) game at some point in time.

    The class is immutable and new state can be instantiated from once an
    action is applied via the `apply_action` method.
    
    Full Deck poker uses all standard ranks 2-A (52 cards total).
    """

    def _get_deck_ranks(self) -> List[int]:
        """Return ranks for Full Deck: 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A.
        
        Returns
        -------
        ranks : List[int]
            List containing [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14].
        """
        return [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    def _get_evaluator(self):
        """Return standard hand evaluator with traditional poker rankings.
        
        Returns
        -------
        evaluator : Evaluator
            Standard evaluator with traditional poker hand rankings
            (Full House > Flush, Straight > Three of a Kind).
        """
        return Evaluator()

    @staticmethod
    def load_card_lut(
        lut_path: str = ".",
        pickle_dir: bool = False
    ) -> Dict[str, Dict[Tuple[int, ...], str]]:
        """Load card information lookup table for Full Deck poker.

        Parameters
        ----------
        lut_path : str
            Path to lookup table.
        pickle_dir : bool
            Whether the lut_path is a path to pickle files or not. Pickle files
            are deprecated for the lut.

        Returns
        -------
        card_info_lut : InfoSetLookupTable
            Card information cluster lookup table.
        """
        if pickle_dir:
            logger.info("Loading card information lut in deprecated way")
            file_names = [
                "preflop_lossless.pkl",
                "flop_lossy_2.pkl",
                "turn_lossy_2.pkl",
                "river_lossy_2.pkl",
            ]
            betting_stages = ["pre_flop", "flop", "turn", "river"]
            card_info_lut: Dict[str, Dict[Tuple[int, ...], str]] = {}
            for file_name, betting_stage in zip(file_names, betting_stages):
                file_path = os.path.join(lut_path, file_name)
                if not os.path.isfile(file_path):
                    raise ValueError(
                        f"File path not found {file_path}. Ensure lut_path is "
                        f"set to directory containing pickle files"
                    )
                with open(file_path, "rb") as fp:
                    card_info_lut[betting_stage] = joblib.load(fp)
        elif lut_path:
            logger.info(f"Loading card from single file at path: {lut_path}")
            card_info_lut = joblib.load(lut_path + '/card_info_lut.joblib')
        else:
            card_info_lut = {}
        return card_info_lut

    @property
    def info_set(self) -> str:
        """Get the information set for the current player."""
        cards = sorted(
            self.current_player.cards,
            key=operator.attrgetter("eval_card"),
        )
        cards += sorted(
            self._table.community_cards,
            key=operator.attrgetter("eval_card"),
        )
        if self._pickle_dir:
            lookup_cards = tuple([card.eval_card for card in cards])
        else:
            lookup_cards = tuple(cards)
        try:
            cards_cluster = self.card_info_lut[self._betting_stage][lookup_cards]
        except KeyError:
            if self.betting_stage not in {"terminal", "show_down"}:
                raise ValueError("You should have these cards in your lut.")
            return "default info set, please ensure you load it correctly"
        # Convert history from a dict of lists to a list of dicts as I'm
        # paranoid about JSON's lack of care with insertion order.
        info_set_dict = {
            "cards_cluster": cards_cluster,
            "history": [
                {betting_stage: [str(action) for action in actions]}
                for betting_stage, actions in self._history.items()
            ],
        }
        return json.dumps(
            info_set_dict, separators=(",", ":"), cls=utils.io.NumpyJSONEncoder
        )
