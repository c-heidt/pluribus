from typing import Dict, Tuple, List
import operator

from poker_ai.poker.card import Card


def make_starting_hand_bucket(starting_hand: List[Card], rank_to_index: Dict[int, int]) -> int:
    """
    Compute preflop abstraction bucket for any deck size.
    
    Uses a canonical representation:
    - Pairs: single bucket per rank (13 total for full deck)
    - Suited hands: one bucket per unique rank pair (78 total for full deck)
    - Offsuit hands: one bucket per unique rank pair (78 total for full deck)
    
    Total buckets: n_ranks + C(n_ranks, 2) * 2
    - 20-card (5 ranks): 5 + 10*2 = 25 buckets
    - 36-card (9 ranks): 9 + 36*2 = 81 buckets
    - 52-card (13 ranks): 13 + 78*2 = 169 buckets
    
    Parameters
    ----------
    starting_hand : List[Card]
        Two-card starting hand (Card objects).
    rank_to_index : Dict[int, int]
        Mapping from card rank to 0-indexed position (e.g., {2:0, 3:1, ..., 14:12})
        
    Returns
    -------
    int
        Bucket ID (0-indexed)
    """
    ranks = [card.rank_int for card in starting_hand]
    suits = [card.suit for card in starting_hand]
    
    # Normalize to higher rank first
    r1, r2 = sorted(ranks, reverse=True)
    suited = (len(set(suits)) == 1)
    
    # Convert to 0-indexed positions
    idx1 = rank_to_index[r1]
    idx2 = rank_to_index[r2]
    n_ranks = len(rank_to_index)
    
    if r1 == r2:
        # Pair: first n_ranks buckets
        return idx1
    else:
        # Non-pair: compute combinatorial index
        # Number of suited/offsuit combos with high rank > r1
        n_combos_above = idx1 * (idx1 - 1) // 2 if idx1 > 0 else 0
        # Number of combos with high rank == r1 and low rank > r2
        # (counts intermediate ranks between idx1 and idx2)
        n_combos_same_high = idx1 - idx2 - 1
        
        # Offset for this specific (r1, r2) pair
        combo_offset = n_combos_above + n_combos_same_high
        
        if suited:
            # Suited: buckets n_ranks to n_ranks + C(n_ranks, 2) - 1
            return n_ranks + combo_offset
        else:
            # Offsuit: buckets n_ranks + C(n_ranks, 2) to end
            n_unique_pairs = n_ranks * (n_ranks - 1) // 2
            return n_ranks + n_unique_pairs + combo_offset


def compute_preflop_lossless_abstraction(builder) -> Dict[Tuple[Card, Card], int]:
    """
    Compute the preflop abstraction dictionary.
    
    Supports 52-card (13 ranks), 36-card (9 ranks), and 20-card (5 ranks) decks.
    
    Parameters
    ----------
    builder : CardInfoLutBuilder
        Builder with _cards, starting_hands (int arrays), and int_to_cards() method
        
    Returns
    -------
    Dict[Tuple[Card, Card], int]
        Mapping from starting hand tuples to bucket IDs
    """
    # Get all ranks in the deck
    found_ranks = sorted(set([c.rank_int for c in builder._cards]))
    n_ranks = len(found_ranks)
    
    # Create rank-to-index mapping (e.g., {2:0, 3:1, ..., 14:12} for full deck)
    rank_to_index = {rank: idx for idx, rank in enumerate(found_ranks)}
    
    # Validate deck configuration
    if n_ranks not in [5, 9, 13]:
        raise ValueError(
            f"Preflop abstraction supports 20-card (5 ranks), 36-card (9 ranks), "
            f"or 52-card (13 ranks) decks. Found {n_ranks} ranks: {found_ranks}"
        )
    
    # Compute expected number of buckets
    expected_buckets = n_ranks + n_ranks * (n_ranks - 1)  # pairs + suited + offsuit
    
    # Getting combos and indexing with abstraction
    # starting_hands are now integer arrays, convert back to Card objects
    preflop_lossless: Dict[Tuple[Card, Card], int] = {}
    for starting_hand_ints in builder.starting_hands:
        # Convert integer array to Card objects
        starting_hand_cards = builder.int_to_cards(starting_hand_ints)
        # Sort by eval_card ascending to match combo ordering
        starting_hand_cards = sorted(
            starting_hand_cards,
            key=operator.attrgetter("eval_card"),
        )
        bucket = make_starting_hand_bucket(starting_hand_cards, rank_to_index)
        preflop_lossless[tuple(starting_hand_cards)] = bucket
    
    # Validate that all buckets are used (sanity check)
    unique_buckets = set(preflop_lossless.values())
    if len(unique_buckets) != expected_buckets:
        print(
            f"Warning: Expected {expected_buckets} buckets but found "
            f"{len(unique_buckets)} unique buckets"
        )
    
    return preflop_lossless
