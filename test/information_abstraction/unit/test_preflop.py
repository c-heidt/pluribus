"""Unit tests for compute_preflop_lossless_abstraction."""
from information_abstraction.preflop import (
    compute_preflop_lossless_abstraction,
)


class TestPreflopBuckets:
    def test_expected_bucket_count(self, tiny_combos):
        lut = compute_preflop_lossless_abstraction(builder=tiny_combos)
        # 3 ranks: 3 pairs + C(3,2) * 2 = 3 + 6 = 9 buckets used.
        assert len(set(lut.values())) == 9

    def test_every_starting_hand_is_covered(self, tiny_combos):
        lut = compute_preflop_lossless_abstraction(builder=tiny_combos)
        for combo in tiny_combos.starting_hands:
            key = tuple(sorted(int(c) for c in combo))
            assert key in lut

    def test_key_is_sorted_tuple(self, tiny_combos):
        lut = compute_preflop_lossless_abstraction(builder=tiny_combos)
        for key in lut:
            assert list(key) == sorted(key)

    def test_bucket_ids_occupy_prefix_range(self, tiny_combos):
        """Buckets are densely numbered from 0 up to #buckets-1."""
        lut = compute_preflop_lossless_abstraction(builder=tiny_combos)
        buckets = set(lut.values())
        assert buckets == set(range(len(buckets)))
