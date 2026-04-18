"""Unit tests for Clusterer."""
import numpy as np

from poker_ai.information_abstraction.build.clusterer import Clusterer


class TestDispatch:
    def test_small_data_returns_kmeans_shape(self):
        c = Clusterer(use_mini_batch=True, seed=0)
        X = np.random.RandomState(0).rand(200, 3).astype(np.float32)
        centroids, labels = c.fit_predict(X, num_clusters=4)
        assert centroids.shape == (4, 3)
        assert labels.shape == (200,)
        assert set(labels.tolist()).issubset({0, 1, 2, 3})

    def test_minibatch_threshold_triggers_on_large_data(self, tmp_path):
        c = Clusterer(use_mini_batch=True, seed=0)
        X = np.random.RandomState(0).rand(60_000, 2).astype(np.float32)
        centroids, labels = c.fit_predict(
            X, num_clusters=5, street="river", work_dir=tmp_path,
        )
        assert centroids.shape == (5, 2)
        assert labels.shape == (60_000,)
        # Partial checkpoints are cleaned up after success.
        assert not (tmp_path / "partial_kmeans.joblib").exists()
        assert not (tmp_path / "kmeans_progress.json").exists()

    def test_mini_batch_off_uses_full_batch_even_on_large_data(self):
        c = Clusterer(use_mini_batch=False, seed=0)
        X = np.random.RandomState(0).rand(1_000, 2).astype(np.float32)
        centroids, labels = c.fit_predict(X, num_clusters=3)
        assert centroids.shape == (3, 2)

    def test_clusters_exceed_samples_is_clamped(self):
        c = Clusterer(use_mini_batch=True, seed=0)
        X = np.random.RandomState(0).rand(3, 2).astype(np.float32)
        centroids, labels = c.fit_predict(X, num_clusters=10)
        assert centroids.shape[0] == 3  # reduced to n_samples

    def test_single_sample_edge_case(self):
        c = Clusterer()
        X = np.array([[1.0, 2.0]], dtype=np.float32)
        centroids, labels = c.fit_predict(X, num_clusters=3)
        np.testing.assert_array_equal(centroids, X)
        np.testing.assert_array_equal(labels, np.array([0]))


class TestDeterminism:
    def test_same_seed_same_labels_full_batch(self):
        X = np.random.RandomState(0).rand(500, 4).astype(np.float32)
        a = Clusterer(use_mini_batch=False, seed=7)
        b = Clusterer(use_mini_batch=False, seed=7)
        _, labels_a = a.fit_predict(X, num_clusters=5)
        _, labels_b = b.fit_predict(X, num_clusters=5)
        np.testing.assert_array_equal(labels_a, labels_b)
