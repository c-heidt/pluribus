"""Unit tests for the per-street EHS feature extractors."""
import numpy as np
import pytest

from information_abstraction._combinatorics import comb
from information_abstraction.build.ehs import (
    FlopEHS,
    RiverEHS,
    TurnEHS,
    clear_process_cache_for_street,
)


@pytest.fixture(autouse=True)
def _reset_process_cache():
    """Ensure a clean global cache between tests."""
    clear_process_cache_for_street("river")
    clear_process_cache_for_street("turn")
    yield
    clear_process_cache_for_street("river")
    clear_process_cache_for_street("turn")


class TestRiverEHS:
    def test_output_is_win_loss_tie_distribution(
        self, evaluator, tiny_combos,
    ):
        ehs = RiverEHS(
            evaluator=evaluator,
            card_ints=tiny_combos._card_ints,
            method="exact",
        )
        combo = tiny_combos.river[0]
        out = ehs(combo)
        assert out.shape == (3,)
        assert np.isclose(out.sum(), 1.0)
        assert (out >= 0).all() and (out <= 1).all()

    def test_exact_and_mc_agree_in_the_limit(
        self, evaluator, tiny_combos,
    ):
        """With a huge MC sample the two paths converge (sanity check)."""
        combo = tiny_combos.river[0]
        exact = RiverEHS(
            evaluator=evaluator,
            card_ints=tiny_combos._card_ints,
            method="exact",
        )(combo)
        np.random.seed(0)
        mc_samples = 400
        mc = RiverEHS(
            evaluator=evaluator,
            card_ints=tiny_combos._card_ints,
            method="monte_carlo",
            n_simulations=mc_samples,
        )(combo)
        # Loose tolerance: 3-card deck has few opponents; still should agree
        # within a few percent with 400 samples.
        np.testing.assert_allclose(mc, exact, atol=0.15)


class TestTurnEHS:
    def test_distribution_sums_to_one(self, tmp_path, tiny_combos):
        # Write a fake river cluster_ids.dat: everyone maps to cluster 0.
        n = tiny_combos._n_cards
        n_river_rows = comb(n, 2) * comb(n - 2, 5)
        river_dir = tmp_path / "river"
        river_dir.mkdir()
        ids = np.zeros(n_river_rows, dtype=np.uint16)
        mm = np.memmap(
            river_dir / "cluster_ids.dat", dtype=np.uint16,
            mode="w+", shape=(n_river_rows,),
        )
        mm[:] = ids
        mm.flush()
        del mm

        n_river_clusters = 5
        turn_ehs = TurnEHS(
            card_ints=tiny_combos._card_ints,
            card_to_idx=tiny_combos._card_to_idx,
            n_cards=tiny_combos._n_cards,
            save_dir=str(tmp_path),
            n_river_clusters=n_river_clusters,
        )
        combo = tiny_combos.turn[0]
        out = turn_ehs(combo)
        assert out.shape == (n_river_clusters,)
        assert np.isclose(out.sum(), 1.0)
        # All upstream clusters are 0, so all mass sits on bucket 0.
        assert out[0] == pytest.approx(1.0)
        assert (out[1:] == 0).all()


class TestFlopEHS:
    def test_distribution_sums_to_one(self, tmp_path, tiny_combos):
        n = tiny_combos._n_cards
        n_turn_rows = comb(n, 2) * comb(n - 2, 4)
        turn_dir = tmp_path / "turn"
        turn_dir.mkdir()
        ids = np.arange(n_turn_rows, dtype=np.uint16) % 3  # 3 buckets
        mm = np.memmap(
            turn_dir / "cluster_ids.dat", dtype=np.uint16,
            mode="w+", shape=(n_turn_rows,),
        )
        mm[:] = ids
        mm.flush()
        del mm

        n_turn_clusters = 4  # include an empty bucket on purpose
        flop_ehs = FlopEHS(
            card_ints=tiny_combos._card_ints,
            card_to_idx=tiny_combos._card_to_idx,
            n_cards=tiny_combos._n_cards,
            save_dir=str(tmp_path),
            n_turn_clusters=n_turn_clusters,
        )
        combo = tiny_combos.flop[0]
        out = flop_ehs(combo)
        assert out.shape == (n_turn_clusters,)
        assert np.isclose(out.sum(), 1.0)
        # Bucket 3 is never populated by `ids % 3` so it stays zero.
        assert out[3] == 0.0
