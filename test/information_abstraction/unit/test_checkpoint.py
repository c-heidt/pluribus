"""Unit tests for CheckpointManager."""
import json

from information_abstraction.build.checkpoint import (
    CheckpointManager,
)


def _mk(tmp_path):
    return CheckpointManager(tmp_path / "checkpoint.json")


class TestEmpty:
    def test_fresh_init_has_zero_chunks(self, tmp_path):
        cp = _mk(tmp_path)
        for street in ("river", "turn", "flop"):
            assert cp.get_total_chunks(street) == 0
            assert cp.get_completed_chunks(street) == []
            assert cp.is_merge_done(street) is False
            assert cp.is_clustering_done(street) is False
            assert cp.get_feature_dim(street) is None

    def test_save_is_round_tripped_by_a_new_manager(self, tmp_path):
        cp = _mk(tmp_path)
        cp.reset_street("river", total_chunks=5, total_combos=50)
        cp.save()
        reopened = _mk(tmp_path)
        assert reopened.get_total_chunks("river") == 5
        assert reopened.get_total_combos("river") == 50


class TestStateTransitions:
    def test_mark_chunk_complete_appends_and_persists(self, tmp_path):
        cp = _mk(tmp_path)
        cp.reset_street("river", total_chunks=3, total_combos=30)
        cp.save(merge_with_disk=False)
        cp.mark_chunk_complete("river", 2)
        cp.mark_chunk_complete("river", 0)
        assert cp.get_completed_chunks("river") == [0, 2]
        # Idempotent
        cp.mark_chunk_complete("river", 0)
        assert cp.get_completed_chunks("river") == [0, 2]

    def test_incomplete_is_complement(self, tmp_path):
        cp = _mk(tmp_path)
        cp.reset_street("flop", total_chunks=4, total_combos=40)
        cp.save(merge_with_disk=False)
        cp.mark_chunk_complete("flop", 1)
        cp.mark_chunk_complete("flop", 3)
        assert cp.get_incomplete_chunks("flop") == [0, 2]

    def test_merge_and_clustering_flags(self, tmp_path):
        cp = _mk(tmp_path)
        cp.mark_merge_done("turn")
        assert cp.is_merge_done("turn")
        cp.unmark_merge_done("turn")
        assert not cp.is_merge_done("turn")
        cp.mark_clustering_done("river")
        assert cp.is_clustering_done("river")

    def test_feature_dim_persists(self, tmp_path):
        cp = _mk(tmp_path)
        cp.set_feature_dim("turn", 50)
        assert cp.get_feature_dim("turn") == 50
        assert _mk(tmp_path).get_feature_dim("turn") == 50

    def test_reset_street_clears_chunk_progress(self, tmp_path):
        cp = _mk(tmp_path)
        cp.reset_street("river", total_chunks=2, total_combos=20)
        cp.save(merge_with_disk=False)
        cp.mark_chunk_complete("river", 0)
        cp.mark_merge_done("river")
        cp.mark_clustering_done("river")
        cp.reset_street("river", total_chunks=5, total_combos=50)
        assert cp.get_completed_chunks("river") == []
        assert not cp.is_merge_done("river")
        assert not cp.is_clustering_done("river")
        assert cp.get_total_chunks("river") == 5


class TestBulkMerge:
    def test_update_completed_chunks_replaces(self, tmp_path):
        cp = _mk(tmp_path)
        cp.reset_street("river", total_chunks=6, total_combos=60)
        cp.save(merge_with_disk=False)
        cp.mark_chunk_complete("river", 5)
        cp.update_completed_chunks(
            "river", [1, 3, 5], merge_with_disk=False,
        )
        assert cp.get_completed_chunks("river") == [1, 3, 5]

    def test_merge_with_disk_unions_concurrent_writes(self, tmp_path):
        cp_a = _mk(tmp_path)
        cp_a.reset_street("river", total_chunks=6, total_combos=60)
        cp_a.save(merge_with_disk=False)

        # Second manager (mimics another process) writes chunk 0.
        cp_b = _mk(tmp_path)
        cp_b.mark_chunk_complete("river", 0)

        # First manager has chunk 2 in-memory only, then saves with merge.
        cp_a.mark_chunk_complete("river", 2)
        assert cp_a.get_completed_chunks("river") == [0, 2]

        # The merged file on disk must now contain both.
        with open(tmp_path / "checkpoint.json") as f:
            disk = json.load(f)
        assert disk["streets"]["river"]["completed_chunks"] == [0, 2]
