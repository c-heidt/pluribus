"""
Test to verify that optimized combo generation produces ascending-sorted combos
and that O(1) combinadic indexing via get_row_index() is correct.
"""
import numpy as np
import time
try:
    from math import comb
except ImportError:
    from scipy.special import comb as _comb
    def comb(n, k):
        return int(_comb(n, k, exact=True))
from poker_ai.clustering.card_combos import CardCombos


def test_ordering_and_indexing():
    """Verify ascending ordering and O(1) index correctness."""
    
    low_rank = 2
    high_rank = 3  # 2 ranks x 4 suits = 8 cards
    
    print(f"\n{'='*70}")
    print(f"COMBO ORDERING & INDEXING VERIFICATION TEST")
    print(f"{'='*70}")
    print(f"\nDeck: ranks {low_rank}-{high_rank} (8 cards)")
    
    combos = CardCombos(low_rank, high_rank, parallel=False)
    
    print(f"\nGenerated combos:")
    print(f"  Starting hands: {len(combos.starting_hands):,}")
    print(f"  Flop combos:    {len(combos.flop):,}")
    print(f"  Turn combos:    {len(combos.turn):,}")
    print(f"  River combos:   {len(combos.river):,}")
    
    # ----------------------------------------------------------
    # 1. Check ascending sort within each combo
    # ----------------------------------------------------------
    print(f"\n1. Checking ascending sort order:")
    for street, k_public in [("flop", 3), ("turn", 4), ("river", 5)]:
        data = getattr(combos, street)
        if len(data) == 0:
            print(f"   {street}: no combos (skipped)")
            continue
        
        for i, combo in enumerate(data):
            hole = combo[:2]
            public = combo[2:]
            assert hole[0] <= hole[1], (
                f"{street}[{i}] hole not ascending: {hole}"
            )
            for j in range(len(public) - 1):
                assert public[j] <= public[j + 1], (
                    f"{street}[{i}] public not ascending: {public}"
                )
        print(f"   {street}: ✓ all {len(data)} combos ascending")

    # ----------------------------------------------------------
    # 2. Hole-combo grouping (holes outer loop)
    # ----------------------------------------------------------
    print(f"\n2. Checking hole-combo grouping:")
    for street in ["flop", "turn", "river"]:
        data = getattr(combos, street)
        if len(data) == 0:
            continue
        prev_hole = tuple(data[0][:2])
        seen_holes = {prev_hole}
        for i in range(1, len(data)):
            hole = tuple(data[i][:2])
            if hole != prev_hole:
                assert hole not in seen_holes, (
                    f"{street}: hole {hole} re-appeared at index {i}"
                )
                seen_holes.add(hole)
                prev_hole = hole
        print(f"   {street}: ✓ {len(seen_holes)} hole groups, no re-appearances")

    # ----------------------------------------------------------
    # 3. Uniqueness
    # ----------------------------------------------------------
    print(f"\n3. Checking uniqueness:")
    for street in ["flop", "turn", "river"]:
        data = getattr(combos, street)
        unique = set(tuple(c) for c in data)
        assert len(data) == len(unique), (
            f"{street}: {len(data)} combos but only {len(unique)} unique"
        )
        print(f"   {street}: ✓ all {len(data)} combos unique")

    # ----------------------------------------------------------
    # 4. get_row_index() correctness — every row maps back exactly
    # ----------------------------------------------------------
    print(f"\n4. Checking get_row_index() correctness:")
    for street, k_public in [("flop", 3), ("turn", 4), ("river", 5)]:
        data = getattr(combos, street)
        if len(data) == 0:
            continue
        n = combos._n_cards
        expected_rows = comb(n, 2) * comb(n - 2, k_public)
        assert len(data) == expected_rows, (
            f"{street}: expected {expected_rows} rows, got {len(data)}"
        )
        for expected_idx, combo in enumerate(data):
            hole = np.array(combo[:2])
            public = np.array(combo[2:])
            computed_idx = combos.get_row_index(hole, public)
            assert computed_idx == expected_idx, (
                f"{street}[{expected_idx}]: get_row_index returned {computed_idx} "
                f"for combo {combo}"
            )
        print(f"   {street}: ✓ all {len(data)} indices match "
              f"(expected C({n},2)*C({n}-2,{k_public})={expected_rows})")

    print(f"\n{'='*70}")
    print(f"✓ ALL ORDERING & INDEXING CHECKS PASSED")
    print(f"{'='*70}")


def compare_sequential_parallel():
    """Compare sequential and parallel generation to ensure same ordering."""
    
    low_rank = 2
    high_rank = 4  # 3 ranks x 4 suits = 12 cards
    
    print(f"\n{'='*70}")
    print(f"SEQUENTIAL VS PARALLEL COMPARISON")
    print(f"{'='*70}")
    print(f"\nDeck: ranks {low_rank}-{high_rank} (12 cards)")
    
    print("\n1. Generating with sequential mode...")
    start = time.time()
    combos_seq = CardCombos(low_rank, high_rank, parallel=False)
    seq_time = time.time() - start
    print(f"   Completed in {seq_time:.2f}s")
    
    print("\n2. Generating with parallel mode...")
    start = time.time()
    combos_par = CardCombos(low_rank, high_rank, parallel=True, n_workers=2)
    par_time = time.time() - start
    print(f"   Completed in {par_time:.2f}s")
    
    all_match = True
    for street in ['flop', 'turn', 'river']:
        seq_data = getattr(combos_seq, street)
        par_data = getattr(combos_par, street)
        
        print(f"\n{street.upper()}:")
        print(f"  Sequential: {len(seq_data):,} combos")
        print(f"  Parallel:   {len(par_data):,} combos")
        
        if np.array_equal(seq_data, par_data):
            print(f"  ✓ IDENTICAL")
        else:
            print(f"  ❌ MISMATCH")
            all_match = False
            for i in range(min(len(seq_data), len(par_data))):
                if not np.array_equal(seq_data[i], par_data[i]):
                    print(f"    First difference at index {i}:")
                    print(f"      Sequential: {seq_data[i]}")
                    print(f"      Parallel:   {par_data[i]}")
                    break
    
    if all_match:
        print(f"\n✓ PARALLEL TEST PASSED")
    else:
        print(f"\n❌ PARALLEL TEST FAILED")
    return all_match


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CARD COMBO TEST SUITE  (ascending order + O(1) indexing)")
    print("="*70)
    
    try:
        test_ordering_and_indexing()
        compare_sequential_parallel()
        
        print("\n" + "="*70)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*70)
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌❌❌ TEST FAILED ❌❌❌")
        print(f"{'='*70}")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
