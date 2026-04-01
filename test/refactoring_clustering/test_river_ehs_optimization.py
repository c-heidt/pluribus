"""
Test that the decomposed river EHS computation produces identical results
to the naive approach (calling evaluate() per opponent pair).
"""
import numpy as np
import time
from itertools import combinations

from poker_ai.clustering.card_combos import CardCombos
from poker_ai.poker.evaluation import Evaluator


def naive_river_ehs(evaluator, our_hand, board, card_ints):
    """Original naive implementation for comparison."""
    unavailable = set(our_hand.tolist() + board.tolist())
    available = [c for c in card_ints if c not in unavailable]

    our_rank = evaluator.evaluate(
        board=board.astype(np.int64).tolist(),
        cards=our_hand.astype(np.int64).tolist(),
    )

    wins = losses = ties = 0
    for opp_hand in combinations(available, 2):
        opp_rank = evaluator.evaluate(
            board=board.astype(np.int64).tolist(),
            cards=[int(c) for c in opp_hand],
        )
        if our_rank > opp_rank:
            wins += 1
        elif our_rank < opp_rank:
            losses += 1
        else:
            ties += 1

    total = wins + losses + ties
    if total == 0:
        return np.array([0.0, 0.0, 0.0])
    return np.array([wins / total, losses / total, ties / total])


def main():
    print("=" * 70)
    print("RIVER EHS OPTIMISATION CORRECTNESS + SPEED TEST")
    print("=" * 70)

    # Use 20-card deck (ranks 10-14 = T,J,Q,K,A)
    low, high = 10, 14
    combos = CardCombos(low, high, parallel=False)
    card_ints = combos._card_ints

    evaluator = Evaluator()

    # Test on all river combos
    river_combos = combos.river
    n = len(river_combos)
    print(f"\nDeck: {len(card_ints)} cards, {n} river combos to test")

    # Build a minimal ExactHandStrengthBuilder-like object for the optimized method
    from poker_ai.clustering.exact_lut_builder import ExactHandStrengthBuilder
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = ExactHandStrengthBuilder(
            low_card_rank=low,
            high_card_rank=high,
            save_dir=tmpdir,
        )

        # --- Correctness ---
        print("\n1. Correctness check (all river combos)...")
        mismatches = 0
        for i, combo in enumerate(river_combos):
            our_hand = combo[:2]
            board = combo[2:7]

            optimized = builder.compute_exact_river_ehs(our_hand, board)
            reference = naive_river_ehs(evaluator, our_hand, board, card_ints)

            if not np.allclose(optimized, reference, atol=1e-12):
                mismatches += 1
                if mismatches <= 3:
                    print(f"   MISMATCH at combo {i}: {combo}")
                    print(f"     optimized: {optimized}")
                    print(f"     reference: {reference}")

        if mismatches == 0:
            print(f"   All {n} combos match exactly!")
        else:
            print(f"   {mismatches}/{n} mismatches!")
            return

        # --- Speed comparison ---
        print("\n2. Speed comparison...")

        # Warm up
        for combo in river_combos[:5]:
            builder.compute_exact_river_ehs(combo[:2], combo[2:7])
            naive_river_ehs(evaluator, combo[:2], combo[2:7], card_ints)

        # Time optimized
        t0 = time.perf_counter()
        for combo in river_combos:
            builder.compute_exact_river_ehs(combo[:2], combo[2:7])
        t_opt = time.perf_counter() - t0

        # Time naive
        t0 = time.perf_counter()
        for combo in river_combos:
            naive_river_ehs(evaluator, combo[:2], combo[2:7], card_ints)
        t_naive = time.perf_counter() - t0

        speedup = t_naive / t_opt if t_opt > 0 else float("inf")
        print(f"   Naive:     {t_naive:.3f}s ({n/t_naive:.0f} combos/s)")
        print(f"   Optimized: {t_opt:.3f}s ({n/t_opt:.0f} combos/s)")
        print(f"   Speedup:   {speedup:.2f}x")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
