"""KMeans wrapper with resumable minibatch path.

A single :class:`Clusterer` encapsulates the KMeans-vs-MiniBatchKMeans
decision, the batch-size heuristic, and the resume-from-checkpoint logic
that used to live directly on the builder.  Given a feature matrix
``X`` and a target cluster count, ``fit_predict`` returns
``(centroids, labels)``.
"""
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

log = logging.getLogger(
    "poker_ai.information_abstraction.build.clusterer",
)


_MINIBATCH_THRESHOLD = 50_000
_DEFAULT_BATCH_SIZE = 10_000


class Clusterer:
    """Fit a KMeans (or MiniBatchKMeans) model and return centroids + labels.

    Parameters
    ----------
    use_mini_batch : bool
        When ``True`` and ``n_samples > 50 000``, use
        :class:`sklearn.cluster.MiniBatchKMeans`.  Otherwise use full-batch
        :class:`sklearn.cluster.KMeans`.
    seed : int
        Passed to sklearn's ``random_state``.
    """

    def __init__(self, use_mini_batch: bool = True, seed: int = 0):
        self.use_mini_batch = use_mini_batch
        self.seed = seed

    def fit_predict(
        self,
        X: np.ndarray,
        num_clusters: int,
        street: Optional[str] = None,
        work_dir: Optional[Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster ``X`` and return ``(centroids, labels)``.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix; may be a memmap.
        num_clusters : int
            Target cluster count.  Clamped down to ``n_samples`` if larger.
        street : str, optional
            Label used in log messages only.
        work_dir : Path, optional
            Directory for the resumable minibatch checkpoint files.  Required
            only when ``use_mini_batch`` triggers.
        """
        label = street or "data"
        n_samples = X.shape[0]
        log.info(
            "Clustering %d samples into %d clusters (%s)",
            n_samples, num_clusters, label,
        )

        if n_samples < num_clusters:
            log.warning(
                "Samples (%d) < clusters (%d). Reducing to %d.",
                n_samples, num_clusters, n_samples,
            )
            num_clusters = n_samples
        if n_samples == 1:
            return np.asarray(X).copy(), np.array([0])

        if self.use_mini_batch and n_samples > _MINIBATCH_THRESHOLD:
            return self._fit_predict_minibatch(
                X, num_clusters, label, work_dir,
            )
        return self._fit_predict_full_batch(X, num_clusters, label)

    # ------------------------------------------------------------------

    def _fit_predict_full_batch(
        self, X: np.ndarray, num_clusters: int, label: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        log.info("Using standard KMeans for %s", label)
        km = KMeans(
            n_clusters=num_clusters,
            init="random",
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=self.seed,
        )
        y_km = km.fit_predict(X)
        return km.cluster_centers_, y_km

    def _fit_predict_minibatch(
        self,
        X: np.ndarray,
        num_clusters: int,
        label: str,
        work_dir: Optional[Path],
    ) -> Tuple[np.ndarray, np.ndarray]:
        log.info("Using MiniBatchKMeans for %s (large dataset)", label)
        n_samples = X.shape[0]
        batch_size = min(_DEFAULT_BATCH_SIZE, n_samples)

        partial_km_path = None
        progress_path = None
        if work_dir is not None:
            work_dir = Path(work_dir)
            partial_km_path = work_dir / "partial_kmeans.joblib"
            progress_path = work_dir / "kmeans_progress.json"

        start_batch, km = self._resume_or_init_minibatch(
            num_clusters, batch_size, partial_km_path, progress_path, label,
        )

        n_batches = (n_samples + batch_size - 1) // batch_size
        checkpoint_interval = max(1, n_batches // 10)

        for i in tqdm(
            range(start_batch, n_batches),
            desc=f"Clustering {label}",
            initial=start_batch,
            total=n_batches,
        ):
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, n_samples)
            batch_data = np.array(X[batch_start:batch_end])
            km.partial_fit(batch_data)

            if (i + 1) % checkpoint_interval == 0 or (i + 1) == n_batches:
                log.info(
                    "Clustering checkpoint: %d/%d (%d%%)",
                    i + 1, n_batches, 100 * (i + 1) // n_batches,
                )
                if partial_km_path is not None and progress_path is not None:
                    self._save_minibatch_checkpoint(
                        km, i + 1, n_batches,
                        partial_km_path, progress_path,
                    )

        log.info(
            "Predicting cluster assignments for %d samples...", n_samples,
        )
        y_km = km.predict(X)
        centroids = km.cluster_centers_

        if partial_km_path is not None and progress_path is not None:
            for p in (partial_km_path, progress_path):
                if p.exists():
                    p.unlink()
            log.info(
                "Cleaned up partial clustering checkpoints for %s", label,
            )
        return centroids, y_km

    def _resume_or_init_minibatch(
        self,
        num_clusters: int,
        batch_size: int,
        partial_km_path: Optional[Path],
        progress_path: Optional[Path],
        label: str,
    ) -> Tuple[int, MiniBatchKMeans]:
        if (
            partial_km_path is not None
            and progress_path is not None
            and partial_km_path.exists()
            and progress_path.exists()
        ):
            try:
                log.info("Resuming partial clustering for %s", label)
                km = joblib.load(partial_km_path)
                with open(progress_path, "r") as f:
                    progress = json.load(f)
                start_batch = int(progress.get("completed_batches", 0))
                log.info("Resuming from batch %d", start_batch)
                return start_batch, km
            except Exception as e:
                log.warning(
                    "Failed to load checkpoint, starting fresh: %s", e,
                )
        return 0, self._make_minibatch_km(num_clusters, batch_size)

    def _make_minibatch_km(
        self, num_clusters: int, batch_size: int,
    ) -> MiniBatchKMeans:
        return MiniBatchKMeans(
            n_clusters=num_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            batch_size=batch_size,
            random_state=self.seed,
            verbose=0,
        )

    @staticmethod
    def _save_minibatch_checkpoint(
        km: MiniBatchKMeans,
        completed: int,
        total: int,
        partial_km_path: Path,
        progress_path: Path,
    ) -> None:
        temp_km = partial_km_path.with_suffix(".tmp.joblib")
        temp_pg = progress_path.with_suffix(".tmp.json")
        try:
            joblib.dump(km, temp_km)
            with open(temp_pg, "w") as f:
                json.dump(
                    {
                        "completed_batches": completed,
                        "total_batches": total,
                    },
                    f,
                )
            shutil.move(str(temp_km), str(partial_km_path))
            shutil.move(str(temp_pg), str(progress_path))
        except Exception as e:
            log.warning("Checkpoint save failed: %s", e)
            for p in (temp_km, temp_pg):
                if p.exists():
                    p.unlink()
