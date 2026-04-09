"""
Fast unit tests for UnifiedLutBuilder.

Uses a small 12-card deck (ranks 2-4, 4 suits) to keep every test under a few
seconds.  The preflop step (which requires >= 5 ranks) is bypassed by
pre-populating ``card_info_lut["pre_flop"]`` with dummy data.
"""
import tempfile
import shutil
import sys
import time

import numpy as np

from poker_ai.clustering.unified_lut_builder import (
    UnifiedLutBuilder,
    _PROCESS_CACHE,
)
from poker_ai.clustering.card_combos import _lex_rank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_builder(method="exact", tmpdir=None, n_sim=6):
    """Create a 12-card builder (ranks 2-4, 4 suits).

    12 cards is the smallest deck that supports river EHS evaluation:
    2 hole + 5 board = 7, leaving 5 available -> C(5,2) = 10 opponent pairs.
    """
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="test_ub_")
    return UnifiedLutBuilder(
        method=method,
        n_simulations_river=n_sim,
        low_card_rank=2,
        high_card_rank=4,
        save_dir=tmpdir,
        workers=1,
        chunk_size=5000,
        use_mini_batch=False,
        parallel_combos=False,
    )


def _inject_preflop(builder):
    """Inject a dummy pre_flop so compute() skips preflop."""
    builder.card_info_lut["pre_flop"] = {
        tuple(h): 0 for h in builder.starting_hands
    }


# ===========================================================================
# 1. Combo counts
# ===========================================================================

def test_combo_counts():
    """12 cards -> C(12,2)=66 holes, known combo counts for each street."""
    b = _make_builder()
    assert len(b.starting_hands) == 66, f"Expected 66, got {len(b.starting_hands)}"
    # river: C(12,2)*C(10,5) = 66*252 = 16632
    assert len(b.river) == 16632, f"River: expected 16632, got {len(b.river)}"
    # turn: C(12,2)*C(10,4) = 66*210 = 13860
    assert len(b.turn) == 13860, f"Turn: expected 13860, got {len(b.turn)}"
    # flop: C(12,2)*C(10,3) = 66*120 = 7920
    assert len(b.flop) == 7920, f"Flop: expected 7920, got {len(b.flop)}"
    shutil.rmtree(b.save_dir, ignore_errors=True)
    print("  PASS test_combo_counts")


# ===========================================================================
# 2. Lexicographic ranking
# ===========================================================================

def test_lex_rank():
    """Verify _lex_rank returns correct 0-based ranks."""
    # The first combo of k=2 from {0..7} in ascending lex order is (0,1)->0
    assert _lex_rank((0, 1), 8) == 0
    # (0,2)->1
    assert _lex_rank((0, 2), 8) == 1
    # Last: (6,7) = C(8,2)-1 = 27
    assert _lex_rank((6, 7), 8) == 27
    # k=5: first combo (0,1,2,3,4)->0
    assert _lex_rank((0, 1, 2, 3, 4), 8) == 0
    # k=5: last combo (3,4,5,6,7) = C(8,5)-1 = 55
    assert _lex_rank((3, 4, 5, 6, 7), 8) == 55
    print("  PASS test_lex_rank")


# ===========================================================================
# 3. get_row_index() round-trip
# ===========================================================================

def test_row_index_roundtrip():
    """Every river combo's row index should be unique and in range."""
    b = _make_builder()
    n = len(b.river)
    indices = set()
    for combo in b.river:
        hole = combo[:2]
        public = combo[2:]
        idx = b.get_row_index(hole, public)
        assert 0 <= idx < n, f"Index {idx} out of range [0, {n})"
        indices.add(idx)
    assert len(indices) == n, f"Expected {n} unique indices, got {len(indices)}"
    shutil.rmtree(b.save_dir, ignore_errors=True)
    print("  PASS test_row_index_roundtrip")


# ===========================================================================
# 4. Exact river EHS — basic invariants
# ===========================================================================

def test_exact_river_ehs():
    """Win/loss/tie rates should sum to 1.0 and be non-negative."""
    b = _make_builder(method="exact")
    combo = b.river[0]
    ehs = b.process_river_ehs(combo)
    assert ehs.shape == (3,), f"Expected shape (3,), got {ehs.shape}"
    assert abs(ehs.sum() - 1.0) < 1e-9, f"Rates sum to {ehs.sum()}, not 1.0"
    assert np.all(ehs >= 0), f"Negative rate found: {ehs}"
    shutil.rmtree(b.save_dir, ignore_errors=True)
    print("  PASS test_exact_river_ehs")


# ===========================================================================
# 5. MC river EHS — basic invariants
# ===========================================================================

def test_mc_river_ehs():
    """MC EHS should be a valid 3-d probability vector."""
    b = _make_builder(method="monte_carlo", n_sim=50)
    combo = b.river[0]
    ehs = b.process_river_ehs(combo)
    assert ehs.shape == (3,), f"Expected shape (3,), got {ehs.shape}"
    assert abs(ehs.sum() - 1.0) < 1e-9, f"Rates sum to {ehs.sum()}"
    assert np.all(ehs >= 0), f"Negative rate found: {ehs}"
    shutil.rmtree(b.save_dir, ignore_errors=True)
    print("  PASS test_mc_river_ehs")


# ===========================================================================
# 6. Exact vs MC river EHS should be close (with enough samples)
# ===========================================================================

def test_exact_mc_agreement():
    """With many MC samples the result should be close to exact."""
    b_exact = _make_builder(method="exact")
    b_mc = _make_builder(method="monte_carlo", n_sim=2000)

    # Average over several combos to reduce variance
    n_test = min(20, len(b_exact.river))
    exact_avg = np.zeros(3)
    mc_avg = np.zeros(3)
    for combo in b_exact.river[:n_test]:
        exact_avg += b_exact.process_river_ehs(combo)
        mc_avg += b_mc.process_river_ehs(combo)
    exact_avg /= n_test
    mc_avg /= n_test

    diff = np.abs(exact_avg - mc_avg).max()
    assert diff < 0.05, (
        f"Exact/MC average differs by {diff:.4f} (threshold 0.05)\n"
        f"  exact: {exact_avg}\n  mc:    {mc_avg}"
    )
    shutil.rmtree(b_exact.save_dir, ignore_errors=True)
    shutil.rmtree(b_mc.save_dir, ignore_errors=True)
    print(f"  PASS test_exact_mc_agreement (max_diff={diff:.4f})")


# ===========================================================================
# 7. Clustering smoke test
# ===========================================================================

def test_clustering():
    """Verify KMeans wrapper returns correct shapes."""
    b = _make_builder()
    rng = np.random.RandomState(42)
    X = rng.randn(50, 3)
    centroids, labels = b._cluster(num_clusters=3, X=X, street="test")
    assert centroids.shape == (3, 3), f"Centroids shape: {centroids.shape}"
    assert labels.shape == (50,), f"Labels shape: {labels.shape}"
    assert set(labels).issubset({0, 1, 2}), f"Bad labels: {set(labels)}"
    shutil.rmtree(b.save_dir, ignore_errors=True)
    print("  PASS test_clustering")


# ===========================================================================
# 8. create_card_lookup
# ===========================================================================

def test_create_card_lookup():
    combos = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    clusters = np.array([0, 2])
    lut = UnifiedLutBuilder.create_card_lookup(clusters, combos)
    assert lut[(1, 2, 3, 4, 5)] == 0
    assert lut[(6, 7, 8, 9, 10)] == 2
    print("  PASS test_create_card_lookup")


# ===========================================================================
# 9. Full pipeline integration — exact (preflop bypassed)
# ===========================================================================

def test_full_pipeline_exact():
    """Run compute() end-to-end in exact mode with 8 cards.
    Pre-fills preflop to bypass the rank-count restriction."""
    tmpdir = tempfile.mkdtemp(prefix="test_pipeline_exact_")
    try:
        b = _make_builder(method="exact", tmpdir=tmpdir)
        _inject_preflop(b)
        b.compute(n_river_clusters=3, n_turn_clusters=3, n_flop_clusters=3)

        for street in ["pre_flop", "river", "turn", "flop"]:
            assert street in b.card_info_lut, f"Missing {street}"
            assert len(b.card_info_lut[street]) > 0, f"{street} is empty"

        for street in ["river", "turn", "flop"]:
            cids = set(b.card_info_lut[street].values())
            assert max(cids) < 3, f"{street}: cluster id {max(cids)} >= 3"
            assert min(cids) >= 0
            assert street in b.centroids
            assert len(b.centroids[street]) == 3

        print(f"  PASS test_full_pipeline_exact ({len(b.river)} river combos)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# 10. Full pipeline integration — monte_carlo (preflop bypassed)
# ===========================================================================

def test_full_pipeline_mc():
    """Run compute() end-to-end in MC mode with 8 cards."""
    tmpdir = tempfile.mkdtemp(prefix="test_pipeline_mc_")
    try:
        b = _make_builder(method="monte_carlo", tmpdir=tmpdir, n_sim=10)
        _inject_preflop(b)
        b.compute(n_river_clusters=3, n_turn_clusters=3, n_flop_clusters=3)

        for street in ["pre_flop", "river", "turn", "flop"]:
            assert street in b.card_info_lut, f"Missing {street}"
            assert len(b.card_info_lut[street]) > 0, f"{street} is empty"

        for street in ["river", "turn", "flop"]:
            cids = set(b.card_info_lut[street].values())
            assert max(cids) < 3
            assert min(cids) >= 0

        print(f"  PASS test_full_pipeline_mc ({len(b.river)} river combos)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# 11. Key consistency — both modes produce the same combo keys
# ===========================================================================

def test_key_consistency():
    """Exact and MC should cluster the same set of combo keys."""
    tmpdir_e = tempfile.mkdtemp(prefix="test_keys_exact_")
    tmpdir_m = tempfile.mkdtemp(prefix="test_keys_mc_")
    try:
        be = _make_builder(method="exact", tmpdir=tmpdir_e)
        bm = _make_builder(method="monte_carlo", tmpdir=tmpdir_m, n_sim=10)
        _inject_preflop(be)
        _inject_preflop(bm)

        be.compute(3, 3, 3)
        bm.compute(3, 3, 3)

        for street in ["river", "turn", "flop"]:
            ke = set(be.card_info_lut[street].keys())
            km = set(bm.card_info_lut[street].keys())
            assert ke == km, (
                f"{street}: key mismatch ({len(ke)} vs {len(km)})"
            )
        print("  PASS test_key_consistency")
    finally:
        shutil.rmtree(tmpdir_e, ignore_errors=True)
        shutil.rmtree(tmpdir_m, ignore_errors=True)


# ===========================================================================
# Runner
# ===========================================================================

FAST_TESTS = [
    ("Combo counts",        test_combo_counts),
    ("Lex rank",            test_lex_rank),
    ("Row index roundtrip", test_row_index_roundtrip),
    ("Exact river EHS",     test_exact_river_ehs),
    ("MC river EHS",        test_mc_river_ehs),
    ("Exact/MC agreement",  test_exact_mc_agreement),
    ("Clustering",          test_clustering),
    ("create_card_lookup",  test_create_card_lookup),
]

SLOW_TESTS = [
    ("Pipeline (exact)",    test_full_pipeline_exact),
    ("Pipeline (MC)",       test_full_pipeline_mc),
    ("Key consistency",     test_key_consistency),
]


if __name__ == "__main__":
    fast_only = "--fast" in sys.argv
    tests = FAST_TESTS if fast_only else FAST_TESTS + SLOW_TESTS
    label = "FAST" if fast_only else "ALL"
    print(f"\nRunning {len(tests)} {label} tests for UnifiedLutBuilder\n")
    t0 = time.time()
    passed = failed = 0

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL {name}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  {passed} passed, {failed} failed  ({elapsed:.1f}s)")
    if fast_only:
        print(f"  (--fast mode: skipped {len(SLOW_TESTS)} pipeline tests)")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)
