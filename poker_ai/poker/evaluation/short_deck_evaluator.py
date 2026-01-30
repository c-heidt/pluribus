"""Short Deck Poker hand evaluator with adjusted rankings.

In Short Deck (6+ Hold'em), the hand rankings differ from standard poker
due to the reduced deck size (36 cards: ranks 10-A only):

Standard Poker Rankings:
    Royal Flush > Straight Flush > Four of a Kind > Full House > 
    Flush > Straight > Three of a Kind > Two Pair > Pair > High Card

Short Deck Poker Rankings:
    Royal Flush > Straight Flush > Four of a Kind > Flush > 
    Full House > Three of a Kind > Straight > Two Pair > Pair > High Card

Key differences:
    - Flush > Full House (flushes are harder with only 4 cards per suit per rank range)
    - Three of a Kind > Straight (straights are more common with reduced ranks)
"""

import itertools

from poker_ai.poker.evaluation.evaluator import Evaluator
from poker_ai.poker.evaluation.lookup import LookupTable


class ShortDeckEvaluator(Evaluator):
    """
    Evaluator for Short Deck poker with adjusted hand rankings.
    
    Inherits from the standard Evaluator but remaps hand ranks to reflect
    Short Deck poker rules where:
    - Flush beats Full House
    - Three of a Kind beats Straight
    
    The approach is to remap the actual rank values so that comparisons work correctly.
    """
    
    def __init__(self):
        """Initialize with standard lookup table."""
        super().__init__()
    
    def evaluate(self, cards, board):
        """
        Evaluate hand and return remapped rank for Short Deck rules.
        
        This method calls the parent evaluate() to get the standard rank,
        then remaps it to reflect Short Deck hand rankings.
        
        Parameters
        ----------
        cards : list
            Player's hole cards as evaluation integers
        board : list
            Community cards as evaluation integers
            
        Returns
        -------
        rank : int
            Remapped hand rank where lower is better, adjusted for Short Deck
        """
        # Get standard rank from parent
        std_rank = super().evaluate(cards, board)
        
        # Remap to Short Deck ordering
        return self._remap_rank(std_rank)
    
    def _remap_rank(self, std_rank):
        """
        Remap standard poker rank to Short Deck rank.
        
        We need to swap the rank ranges for:
        - Full House (std: 167-322) <-> Flush (std: 323-1599)
        - Straight (std: 1600-1609) <-> Three of a Kind (std: 1610-2467)
        
        The remapping ensures lower ranks remain better.
        """
        std = LookupTable
        
        # Straight Flush and Four of a Kind stay the same
        if std_rank <= std.MAX_STRAIGHT_FLUSH:
            return std_rank  # 1-10: Straight Flush
        elif std_rank <= std.MAX_FOUR_OF_A_KIND:
            return std_rank  # 11-166: Four of a Kind
        
        # Swap Full House and Flush
        elif std_rank <= std.MAX_FULL_HOUSE:
            # Full House (167-322) -> move after Three of a Kind
            # Map to range 2302-2457 (after Three of a Kind)
            offset = std_rank - (std.MAX_FOUR_OF_A_KIND + 1)
            num_flushes = std.MAX_FLUSH - std.MAX_FULL_HOUSE  # 1277
            num_trips = std.MAX_THREE_OF_A_KIND - std.MAX_STRAIGHT  # 858
            return std.MAX_FOUR_OF_A_KIND + 1 + num_flushes + num_trips + offset
            
        elif std_rank <= std.MAX_FLUSH:
            # Flush (323-1599) -> move before Full House
            # Map to range starting after Four of a Kind (167-1443)
            offset = std_rank - (std.MAX_FULL_HOUSE + 1)
            return std.MAX_FOUR_OF_A_KIND + 1 + offset
        
        # Swap Straight and Three of a Kind
        elif std_rank <= std.MAX_STRAIGHT:
            # Straight (1600-1609) -> move after Full House
            # Map to range 2458-2467 (after Full House)
            offset = std_rank - (std.MAX_FLUSH + 1)
            num_flushes = std.MAX_FLUSH - std.MAX_FULL_HOUSE  # 1277
            num_trips = std.MAX_THREE_OF_A_KIND - std.MAX_STRAIGHT  # 858
            num_fh = std.MAX_FULL_HOUSE - std.MAX_FOUR_OF_A_KIND  # 156
            return std.MAX_FOUR_OF_A_KIND + 1 + num_flushes + num_trips + num_fh + offset
            
        elif std_rank <= std.MAX_THREE_OF_A_KIND:
            # Three of a Kind (1610-2467) -> move after Flush, before Full House
            # Map to range 1444-2301 (after Flush)
            offset = std_rank - (std.MAX_STRAIGHT + 1)
            num_flushes = std.MAX_FLUSH - std.MAX_FULL_HOUSE  # 1277
            return std.MAX_FOUR_OF_A_KIND + 1 + num_flushes + offset
        
        # Two Pair, Pair, High Card stay unchanged
        else:
            # These hands don't need remapping - they stay in the same order
            # Original: 2468-7462 -> No change: 2468-7462
            return std_rank
    
    def get_rank_class(self, hr):
        """
        Returns the class of hand with Short Deck rankings.
        
        Since we've remapped the ranks in evaluate(), we need to use
        the remapped boundaries here.
        """
        std = LookupTable
        
        # After remapping, the boundaries are:
        # 1-10: Straight Flush
        # 11-166: Four of a Kind  
        # 167-1443: Flush (moved up)
        # 1444-2301: Three of a Kind (moved up, before Full House)
        # 2302-2457: Full House (moved down, after Three of a Kind)
        # 2458-2467: Straight (moved down)
        # 2468-7462: Two Pair, Pair, High Card (unchanged from standard)
        
        if hr >= 0 and hr <= std.MAX_STRAIGHT_FLUSH:
            return 1  # Straight Flush
        elif hr <= std.MAX_FOUR_OF_A_KIND:
            return 2  # Four of a Kind
        elif hr <= 1443:  # 166 + 1277
            return 3  # Flush (promoted)
        elif hr <= 2301:  # 1443 + 858
            return 5  # Three of a Kind (promoted above Full House)
        elif hr <= 2457:  # 2301 + 156
            return 4  # Full House (demoted below Three of a Kind)
        elif hr <= 2467:  # 2457 + 10
            return 6  # Straight (demoted)
        elif hr <= std.MAX_TWO_PAIR:
            return 7  # Two Pair (unchanged boundaries)
        elif hr <= std.MAX_PAIR:
            return 8  # Pair (unchanged boundaries)
        else:
            return 9  # High Card
    
    def class_to_string(self, class_int):
        """
        Converts the integer class hand score into a human-readable string.
        Uses Short Deck ranking names.
        """
        rank_names = {
            1: "Straight Flush",
            2: "Four of a Kind",
            3: "Flush",
            4: "Full House",
            5: "Three of a Kind",
            6: "Straight",
            7: "Two Pair",
            8: "Pair",
            9: "High Card",
        }
        return rank_names[class_int]
