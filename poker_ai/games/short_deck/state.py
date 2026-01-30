from __future__ import annotations

import json
import logging
import operator
import os
from typing import Dict, List, Optional, Tuple

import joblib

from poker_ai import utils
from poker_ai.games.base.state import PokerState, InfoSetLookupTable
from poker_ai.games.short_deck.player import ShortDeckPokerPlayer
from poker_ai.poker.evaluation.short_deck_evaluator import ShortDeckEvaluator
from poker_ai.poker.pot import Pot

logger = logging.getLogger("poker_ai.games.short_deck.state")


def new_game(
    n_players: int, 
    card_info_lut: InfoSetLookupTable = {}, 
    deck_size: int = 20,
    **kwargs
) -> ShortDeckPokerState:
    """
    Create a new game of short deck poker.

    ...

    Parameters
    ----------
    n_players : int
        Number of players.
    card_info_lut : InfoSetLookupTable
        Card information cluster lookup table.
    deck_size : int
        Size of deck to use. Options:
        - 20: Ranks 10-A (5 ranks, 20 cards)
        - 36: Ranks 6-A (9 ranks, 36 cards, standard Short Deck)
        Default is 20.

    Returns
    -------
    state : ShortDeckPokerState
        Current state of the game
    """
    pot = Pot()
    players = [
        ShortDeckPokerPlayer(player_i=player_i, initial_chips=10000, pot=pot)
        for player_i in range(n_players)
    ]
    if card_info_lut:
        # Don't reload massive files, it takes ages.
        state = ShortDeckPokerState(
            players=players,
            load_card_lut=False,
            deck_size=deck_size,
            **kwargs
        )
        state.card_info_lut = card_info_lut
    else:
        # Load massive files.
        state = ShortDeckPokerState(
            players=players,
            deck_size=deck_size,
            **kwargs
        )
    return state


class ShortDeckPokerState(PokerState):
    """The state of a Short Deck Poker game at some given point in time.

    The class is immutable and new state can be instantiated from once an
    action is applied via the `apply_action` method.
    
    Supports two deck configurations:
    - 20 cards (ranks 10-A): 5 ranks × 4 suits
    - 36 cards (ranks 6-A): 9 ranks × 4 suits (standard Short Deck)
    
    Note: With 20 cards, any flush requires all 5 cards of a suit (10-J-Q-K-A),
    which is always a straight flush. Therefore, only the "Three of a Kind > Straight"
    ranking change applies in 20-card gameplay. With 36 cards, both ranking changes
    (Flush > Full House and Three of a Kind > Straight) apply in actual gameplay.
    """

    def __init__(self, deck_size: int = 20, **kwargs):
        """Initialize Short Deck Poker state.
        
        Parameters
        ----------
        deck_size : int
            Size of deck to use. Options:
            - 20: Ranks 10-A (5 ranks, 20 cards)
            - 36: Ranks 6-A (9 ranks, 36 cards, standard Short Deck)
            Default is 20.
        **kwargs
            Additional arguments passed to PokerState base class.
        """
        if deck_size not in (20, 36):
            raise ValueError(
                f"deck_size must be 20 or 36, got {deck_size}. "
                f"Use 20 for ranks 10-A or 36 for ranks 6-A."
            )
        self._deck_size = deck_size
        super().__init__(**kwargs)

    def _get_deck_ranks(self) -> List[int]:
        """Return ranks for Short Deck based on deck configuration.
        
        Returns
        -------
        ranks : List[int]
            - For 20-card deck: [10, 11, 12, 13, 14] (10, J, Q, K, A)
            - For 36-card deck: [6, 7, 8, 9, 10, 11, 12, 13, 14] (6-A)
        """
        if self._deck_size == 20:
            return [10, 11, 12, 13, 14]
        else:  # 36-card deck
            return [6, 7, 8, 9, 10, 11, 12, 13, 14]

    def _get_evaluator(self):
        """Return Short Deck hand evaluator with adjusted rankings.
        
        Returns
        -------
        evaluator : ShortDeckEvaluator
            Evaluator that correctly ranks hands for Short Deck poker
            (Flush > Full House, Three of a Kind > Straight).
        """
        return ShortDeckEvaluator()

    @staticmethod
    def load_card_lut(
        lut_path: str = ".",
        pickle_dir: bool = False
    ) -> Dict[str, Dict[Tuple[int, ...], str]]:
        """Load card information lookup table for Short Deck poker.

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
            reverse=True,
        )
        cards += sorted(
            self._table.community_cards,
            key=operator.attrgetter("eval_card"),
            reverse=True,
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


