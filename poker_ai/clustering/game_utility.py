from typing import List, Optional

import numpy as np

from poker_ai.poker.evaluation import Evaluator
from poker_ai.poker.evaluation.short_deck_evaluator import ShortDeckEvaluator


class GameUtility:
    """This class takes care of some game related functions."""

    def __init__(
        self, 
        our_hand: np.ndarray, 
        board: np.ndarray, 
        cards: np.ndarray,
        evaluator: Optional[Evaluator] = None
    ):
        """
        Initialize GameUtility with optional evaluator.
        
        Parameters
        ----------
        our_hand : np.ndarray
            Our hole cards
        board : np.ndarray
            Board cards
        cards : np.ndarray
            All cards in the deck
        evaluator : Optional[Evaluator]
            Evaluator to use. If None, selects based on deck size:
            - 20 or 36 cards: ShortDeckEvaluator
            - 52 cards: Standard Evaluator
        """
        if evaluator is None:
            # Auto-select evaluator based on deck size
            n_cards = len(cards)
            if n_cards in [20, 36]:
                self._evaluator = ShortDeckEvaluator()
            else:
                self._evaluator = Evaluator()
        else:
            self._evaluator = evaluator
        
        unavailable_cards = np.concatenate([board, our_hand], axis=0)
        self.available_cards = np.array(
            [c for c in cards if c not in unavailable_cards]
        )
        self.our_hand = our_hand
        self.board = board

    def evaluate_hand(self, hand: np.ndarray) -> int:
        """
        Evaluate a hand.

        Parameters
        ----------
        hand : np.ndarray
            Hand to evaluate.

        Returns
        -------
            Evaluation of hand
        """
        return self._evaluator.evaluate(
            board=self.board.astype(np.int).tolist(),
            cards=hand.astype(np.int).tolist(),
        )

    def get_winner(self) -> int:
        """Get the winner.

        Returns
        -------
            int of win (0), lose (1) or tie (2) - this is an index in the
            expected hand strength array
        """
        our_hand_rank = self.evaluate_hand(self.our_hand)
        opp_hand_rank = self.evaluate_hand(self.opp_hand)
        if our_hand_rank > opp_hand_rank:
            return 0
        elif our_hand_rank < opp_hand_rank:
            return 1
        else:
            return 2

    @property
    def opp_hand(self) -> List[int]:
        """Get random card.

        Returns
        -------
            Two cards for the opponent (Card)
        """
        return np.random.choice(self.available_cards, 2, replace=False)
