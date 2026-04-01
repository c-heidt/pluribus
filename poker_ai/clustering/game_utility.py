from typing import List, Optional

import numpy as np

from poker_ai.environment.evaluation import Evaluator


class GameUtility:
    """
    This class takes care of some game related functions.
    
    Works with integer card representation (eval_card values) for memory efficiency.
    """

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
            Our hole cards (as integers/eval_card values).
        board : np.ndarray
            Board cards (as integers/eval_card values).
        cards : np.ndarray
            All cards in the deck (as integers/eval_card values).
        evaluator : Optional[Evaluator]
            Evaluator to use. If None, uses standard Evaluator.
        """
        if evaluator is None:
            self._evaluator = Evaluator()
        else:
            self._evaluator = evaluator
        
        # Convert to sets for O(1) lookup
        unavailable_set = set(int(c) for c in our_hand) | set(int(c) for c in board)
        self.available_cards = np.array(
            [c for c in cards if int(c) not in unavailable_set],
            dtype=np.int32
        )
        self.our_hand = np.asarray(our_hand, dtype=np.int32)
        self.board = np.asarray(board, dtype=np.int32)
        # Pre-compute our rank once — avoid re-evaluating it on every get_winner() call
        self.our_hand_rank = self.evaluate_hand(self.our_hand)

    def evaluate_hand(self, hand: np.ndarray) -> int:
        """
        Evaluate a hand.

        Parameters
        ----------
        hand : np.ndarray
            Hand to evaluate (as integers/eval_card values).

        Returns
        -------
            Evaluation of hand (lower is better).
        """
        return self._evaluator.evaluate(
            board=self.board.astype(np.int64).tolist(),
            cards=[int(c) for c in hand],
        )

    def get_winner(self) -> int:
        """Get the winner.

        Returns
        -------
            int of win (0), lose (1) or tie (2) - this is an index in the
            expected hand strength array
        """
        opp_hand_rank = self.evaluate_hand(self.opp_hand)
        if self.our_hand_rank > opp_hand_rank:
            return 0
        elif self.our_hand_rank < opp_hand_rank:
            return 1
        else:
            return 2

    @property
    def opp_hand(self) -> np.ndarray:
        """Get random opponent hand.

        Returns
        -------
            Two cards for the opponent (as integers).
        """
        return np.random.choice(self.available_cards, 2, replace=False)
