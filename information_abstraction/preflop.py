"""Pre-flop lossless abstraction.

Maps every starting hand tuple to a bucket id.  Buckets are deck-size
agnostic: ``n_ranks`` pair buckets plus ``C(n_ranks, 2)`` suited and
``C(n_ranks, 2)`` offsuit buckets.

The abstraction is lossless because strategically-equivalent hands share
a bucket — the specific suits of an offsuit hand (e.g. ``AhKs`` vs.
``AsKh``) never affect preflop decisions, so collapsing them into a
single id is safe and keeps the preflop information set count low
enough that CFR can enumerate it without any sampling.
"""
from typing import Dict, List, Tuple

from environment.utils import card_rank_int, card_suit_str


def make_starting_hand_bucket(
    starting_hand: List[int],
    rank_to_index: Dict[int, int],
) -> int:
    """Compute the preflop bucket id for one starting hand.

    Buckets are laid out as ``[pairs | suited | offsuit]``:
    - **pairs**: first ``n_ranks`` ids (one per rank)
    - **suited**: next ``C(n_ranks, 2)`` ids
    - **offsuit**: final ``C(n_ranks, 2)`` ids

    The layout orders ranks from highest (index 0) to lowest.  Within the
    suited and offsuit regions, combos are enumerated in descending
    (high-rank, low-rank) order so the bucket id of e.g. ``AK`` is always
    lower than that of ``QJ`` regardless of deck size.

    Used by :func:`compute_preflop_lossless_abstraction` to populate the
    ``"pre_flop"`` entry of the
    :data:`~information_abstraction.lookup.InfoSetLut`.

    Parameters
    ----------
    starting_hand : List[int]
        Two-card starting hand as eval_card integers.
    rank_to_index : Dict[int, int]
        Mapping from card rank to 0-indexed position (highest rank → 0).

    Returns
    -------
    int
        Bucket id in the range ``[0, n_ranks + 2*C(n_ranks, 2))``.
    """
    ranks = [card_rank_int(c) for c in starting_hand]
    suits = [card_suit_str(c) for c in starting_hand]

    r1, r2 = sorted(ranks, reverse=True)
    suited = len(set(suits)) == 1

    idx1 = rank_to_index[r1]
    idx2 = rank_to_index[r2]
    n_ranks = len(rank_to_index)

    if r1 == r2:
        return idx1

    n_combos_above = idx1 * (idx1 - 1) // 2 if idx1 > 0 else 0
    n_combos_same_high = idx1 - idx2 - 1
    combo_offset = n_combos_above + n_combos_same_high

    if suited:
        return n_ranks + combo_offset
    n_unique_pairs = n_ranks * (n_ranks - 1) // 2
    return n_ranks + n_unique_pairs + combo_offset


def compute_preflop_lossless_abstraction(
    builder,
) -> Dict[Tuple[int, int], int]:
    """Build the ``{starting_hand_tuple: bucket_id}`` dictionary.

    Iterates over every starting hand exposed by ``builder`` and assigns
    it a lossless bucket via :func:`make_starting_hand_bucket`.  The
    builder is taken as a duck-typed handle rather than a concrete type
    so tests can pass a lightweight stand-in without instantiating the
    full combo generator.

    Parameters
    ----------
    builder
        Any object exposing ``_card_ints`` (iterable of eval_card ints) and
        ``starting_hands`` (iterable of 2-card tuples/arrays of ints).

    Returns
    -------
    Dict[Tuple[int, int], int]
        Mapping from sorted two-card tuples to bucket ids.  Tuples are
        sorted so callers can look up a hand without worrying about card
        order at the call site.

    Raises
    ------
    ValueError
        If the deck exposed by ``builder`` contains fewer than two
        ranks, which would make the pair-vs-non-pair split degenerate.
    """
    found_ranks = sorted({card_rank_int(c) for c in builder._card_ints})
    n_ranks = len(found_ranks)
    if n_ranks < 2:
        raise ValueError(
            f"Preflop abstraction requires at least 2 ranks. "
            f"Found {n_ranks} ranks: {found_ranks}"
        )
    rank_to_index = {rank: idx for idx, rank in enumerate(found_ranks)}
    expected_buckets = n_ranks + n_ranks * (n_ranks - 1)

    preflop_lossless: Dict[Tuple[int, int], int] = {}
    for starting_hand_ints in builder.starting_hands:
        hand_key = tuple(sorted(int(c) for c in starting_hand_ints))
        bucket = make_starting_hand_bucket(list(hand_key), rank_to_index)
        preflop_lossless[hand_key] = bucket

    unique_buckets = set(preflop_lossless.values())
    if len(unique_buckets) != expected_buckets:
        print(
            f"Warning: Expected {expected_buckets} buckets but found "
            f"{len(unique_buckets)} unique buckets"
        )
    return preflop_lossless
