import logging
from typing import Dict, List, Tuple
from itertools import combinations
import operator

import numpy as np
from tqdm import tqdm

from poker_ai.poker.card import Card
from poker_ai.poker.deck import get_all_suits


log = logging.getLogger("poker_ai.clustering.runner")


class CardCombos:
    """
    This class stores combinations of cards (histories) per street.
    
    Uses integer card representation (eval_card values) for memory efficiency.
    The integer representation reduces memory usage by ~18x compared to Card objects.
    
    Attributes
    ----------
    _cards : np.ndarray
        Array of Card objects (kept for preflop bucket calculation).
    _card_ints : np.ndarray
        Array of integer card values (eval_card). Used for all combo arrays.
    _int_to_card : Dict[int, Card]
        Mapping from integer to Card object for final lookup table creation.
    starting_hands : np.ndarray
        2D array of starting hand combos as integers, shape (n_hands, 2).
    flop : np.ndarray
        2D array of flop combos as integers, shape (n_combos, 5).
    turn : np.ndarray
        2D array of turn combos as integers, shape (n_combos, 6).
    river : np.ndarray
        2D array of river combos as integers, shape (n_combos, 7).
    """

    def __init__(
        self, low_card_rank: int, high_card_rank: int,
    ):
        super().__init__()
        # Sort for caching.
        suits: List[str] = sorted(list(get_all_suits()))
        ranks: List[int] = sorted(list(range(low_card_rank, high_card_rank + 1)))
        
        # Create Card objects (needed for preflop bucket calculation)
        self._cards = np.array(
            [Card(rank, suit) for suit in suits for rank in ranks]
        )
        
        # Create integer representation (eval_card values) - 18x more memory efficient
        self._card_ints = np.array(
            [c.eval_card for c in self._cards], dtype=np.int32
        )
        
        # Mapping from integer back to Card object
        self._int_to_card: Dict[int, Card] = {
            c.eval_card: c for c in self._cards
        }
        
        # Generate combos using integers
        log.info(f"Generating card combos for {len(self._cards)} cards...")
        self.starting_hands = self._get_int_combos(2)
        log.info(f"Starting hands: {len(self.starting_hands):,}")
        
        self.flop = self._create_int_info_combos(
            self.starting_hands, self._get_int_combos(3), "flop"
        )
        log.info(f"Created flop: {len(self.flop):,} combos")
        
        self.turn = self._create_int_info_combos(
            self.starting_hands, self._get_int_combos(4), "turn"
        )
        log.info(f"Created turn: {len(self.turn):,} combos")
        
        self.river = self._create_int_info_combos(
            self.starting_hands, self._get_int_combos(5), "river"
        )
        log.info(f"Created river: {len(self.river):,} combos")

    def _get_int_combos(self, num_cards: int) -> np.ndarray:
        """
        Get card combinations as integer arrays.

        Parameters
        ----------
        num_cards : int
            Number of cards per combination.

        Returns
        -------
        np.ndarray
            2D array of shape (n_combos, num_cards) with integer card values.
        """
        combos = list(combinations(self._card_ints, num_cards))
        return np.array(combos, dtype=np.int32)

    def _create_int_info_combos(
        self,
        start_combos: np.ndarray,
        publics: np.ndarray,
        betting_stage: str = "unknown",
    ) -> np.ndarray:
        """
        Combinations of private info (hole cards) and public info (board).
        
        Uses integer card representation for memory efficiency.
        Cards are sorted by value (descending) for canonical representation.

        Parameters
        ----------
        start_combos : np.ndarray
            Starting combinations (hole cards) as integers, shape (n, 2).
        publics : np.ndarray
            Public card combinations as integers, shape (m, num_public).
        betting_stage : str
            Name of the betting stage for progress logging.
            
        Returns
        -------
        np.ndarray
            2D array of combined hole + board cards as integers.
        """
        num_hole = start_combos.shape[1]
        num_public = publics.shape[1]
        total_cards = num_hole + num_public
        
        # Pre-allocate result list
        result = []
        
        # Convert publics to set for faster lookup
        for hole_combo in tqdm(
            start_combos,
            dynamic_ncols=True,
            desc=f"Creating {betting_stage} info combos",
        ):
            # Sort hole cards descending
            sorted_hole = np.sort(hole_combo)[::-1]
            hole_set = set(sorted_hole.tolist())
            
            for public_combo in publics:
                # Check for overlap with hole cards
                if not any(c in hole_set for c in public_combo):
                    # Sort public cards descending
                    sorted_public = np.sort(public_combo)[::-1]
                    # Combine: hole cards first, then public
                    combined = np.concatenate([sorted_hole, sorted_public])
                    result.append(combined)
        
        return np.array(result, dtype=np.int32)

    # Legacy methods for backward compatibility
    def get_card_combos(self, num_cards: int) -> np.ndarray:
        """
        Get the card combinations for a given street.
        
        DEPRECATED: Use _get_int_combos for new code.
        Kept for backward compatibility with preflop calculation.

        Parameters
        ----------
        num_cards : int
            Number of cards you want returned

        Returns
        -------
            Combos of cards (Card) -> np.ndarray
        """
        return np.array([c for c in combinations(self._cards, num_cards)])

    def create_info_combos(
        self, start_combos: np.ndarray, publics: np.ndarray
    ) -> np.ndarray:
        """
        DEPRECATED: Use _create_int_info_combos for new code.
        Kept for backward compatibility.
        """
        if publics.shape[1] == 3:
            betting_stage = "flop"
        elif publics.shape[1] == 4:
            betting_stage = "turn"
        elif publics.shape[1] == 5:
            betting_stage = "river"
        else:
            betting_stage = "unknown"
        our_cards: List[int] = []
        for combos in tqdm(
            start_combos,
            dynamic_ncols=True,
            desc=f"Creating {betting_stage} info combos",
        ):
            # Descending sort combos.
            sorted_combos: List[Card] = sorted(
                list(combos),
                key=operator.attrgetter("eval_card"),
                reverse=True,
            )
            for public_combo in publics:
                # Descending sort public_combo.
                sorted_public_combo: List[Card] = sorted(
                    list(public_combo),
                    key=operator.attrgetter("eval_card"),
                    reverse=True,
                )
                if not np.any(np.isin(sorted_combos, sorted_public_combo)):
                    # Combine hand and public cards.
                    hand: np.array = np.array(
                        sorted_combos + sorted_public_combo
                    )
                    our_cards.append(hand)
        return np.array(our_cards)

    def int_to_cards(self, int_combo: np.ndarray) -> Tuple[Card, ...]:
        """
        Convert an integer combo array back to Card objects.
        
        Parameters
        ----------
        int_combo : np.ndarray
            Array of integer card values.
            
        Returns
        -------
        Tuple[Card, ...]
            Tuple of Card objects.
        """
        return tuple(self._int_to_card[int(c)] for c in int_combo)
