"""On-disk state for the chunked build pipeline.

:class:`ChunkStore` owns every file inside a street's build directory: per-chunk
feature arrays, the merged memmap, the all-combos index, centroids, and cluster
labels.  It composes a :class:`CheckpointManager` for resume state.

Directory layout ``{save_dir}/{street}/``::

    chunks/chunk_000000.npy[.gz]   # EHS features per chunk, float16 on disk
    combos/                        # (legacy — new runs skip this)
    merged_data.dat                # memmap after merge
    all_combos.npy                 # shape (total_combos, k)
    centroids.npy
    clusters.npy
    cluster_ids.dat                # written by the builder, consumed by MemmapLookup
    partial_kmeans.joblib          # transient; cleared after clustering
    kmeans_progress.json           # transient; cleared after clustering
"""
import concurrent.futures
import gzip
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from information_abstraction.build.checkpoint import (
    CheckpointManager,
)

log = logging.getLogger("information_abstraction.build.chunk_store")


# ---------------------------------------------------------------------------
# Errors + worker
# ---------------------------------------------------------------------------


class CorruptChunkError(Exception):
    """Raised when corrupt chunk files are detected during merge.

    Attributes
    ----------
    street : str
        Street whose chunks are corrupt.
    corrupt_indices : List[int]
        Chunk indices that were unmarked and deleted, ready for reprocessing.
    """

    def __init__(self, street: str, corrupt_indices: List[int]):
        self.street = street
        self.corrupt_indices = corrupt_indices
        preview = str(corrupt_indices[:20])
        if len(corrupt_indices) > 20:
            preview = preview[:-1] + ", ...]"
        super().__init__(
            f"{len(corrupt_indices)} corrupt chunk(s) detected for {street} "
            f"(truncated/killed write). Indices: {preview}"
        )


def _process_single_chunk_worker(
    chunk_idx: int,
    chunk_combos: np.ndarray,
    item_processor: Callable[[np.ndarray], np.ndarray],
) -> Tuple[int, np.ndarray]:
    """Per-worker loop that evaluates ``item_processor`` over a chunk.

    Module-level so ``ProcessPoolExecutor`` can pickle it.  Returns
    ``(chunk_idx, results_array)``; combos are reconstructed upstream
    from the full ``all_combos`` array.
    """
    first = item_processor(chunk_combos[0])
    results = np.empty((len(chunk_combos), len(first)), dtype=np.float32)
    results[0] = first
    for i in range(1, len(chunk_combos)):
        results[i] = item_processor(chunk_combos[i])
    return chunk_idx, results


# ---------------------------------------------------------------------------
# ChunkStore
# ---------------------------------------------------------------------------


class ChunkStore:
    """On-disk chunk + centroid + cluster storage for a build.

    Parameters
    ----------
    save_dir : Path
        Root directory; each street lives in a subdirectory.
    chunk_size : int
        Items per chunk.
    use_compression : bool
        Gzip chunk files (trades ~10-20% I/O for 60-80% size reduction).
    storage_dtype : np.dtype
        On-disk dtype for chunk feature arrays.  ``float16`` is plenty for
        equity histograms; loads re-cast to ``float32`` for KMeans.
    """

    def __init__(
        self,
        save_dir: Path,
        chunk_size: int = 10000,
        use_compression: bool = True,
        storage_dtype: np.dtype = np.float16,
    ):
        self.save_dir = Path(save_dir)
        self.chunk_size = chunk_size
        self.use_compression = use_compression
        self.storage_dtype = storage_dtype
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = CheckpointManager(
            self.save_dir / "checkpoint.json"
        )

    # ------------------------------------------------------------------
    # Directory / path helpers
    # ------------------------------------------------------------------

    def get_street_dir(self, street: str) -> Path:
        street_dir = self.save_dir / street
        street_dir.mkdir(parents=True, exist_ok=True)
        return street_dir

    def get_chunks_dir(self, street: str) -> Path:
        d = self.get_street_dir(street) / "chunks"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_combos_dir(self, street: str) -> Path:
        d = self.get_street_dir(street) / "combos"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_chunk_path(self, street: str, chunk_idx: int) -> Path:
        ext = ".npy.gz" if self.use_compression else ".npy"
        return self.get_chunks_dir(street) / f"chunk_{chunk_idx:06d}{ext}"

    def get_combos_path(self, street: str, chunk_idx: int) -> Path:
        return self.get_combos_dir(street) / f"combos_{chunk_idx:06d}.npy"

    def get_chunk_indices(
        self, total_combos: int,
    ) -> List[Tuple[int, int, int]]:
        n_chunks = (total_combos + self.chunk_size - 1) // self.chunk_size
        return [
            (i, i * self.chunk_size, min((i + 1) * self.chunk_size, total_combos))
            for i in range(n_chunks)
        ]

    # ------------------------------------------------------------------
    # Street lifecycle
    # ------------------------------------------------------------------

    def initialize_street(
        self,
        street: str,
        total_combos: int,
        config: Optional[dict] = None,
    ) -> None:
        """Prepare a street for processing.

        Resets on-disk state and the checkpoint if the chunk count has
        changed (different configuration).  Otherwise leaves existing
        progress intact.
        """
        n_chunks = (total_combos + self.chunk_size - 1) // self.chunk_size
        if self.checkpoint.get_total_chunks(street) != n_chunks:
            log.info(
                "Initializing %s with %d chunks (%d combos)",
                street, n_chunks, total_combos,
            )
            self.checkpoint.reset_street(street, n_chunks, total_combos)
            self._clear_street_data(street)
        else:
            self.checkpoint.update_total_combos(street, total_combos)

        if config:
            self.checkpoint.set_config(config)
        self.checkpoint.save(merge_with_disk=False)

    def _clear_street_data(self, street: str) -> None:
        street_dir = self.get_street_dir(street)
        if not street_dir.exists():
            return
        for subdir in ("chunks", "combos"):
            subdir_path = street_dir / subdir
            if subdir_path.exists():
                shutil.rmtree(subdir_path)
        for f in street_dir.glob("*.npy"):
            f.unlink()
        for f in street_dir.glob("*.dat"):
            f.unlink()

    # ------------------------------------------------------------------
    # Chunk I/O
    # ------------------------------------------------------------------

    def save_chunk(
        self,
        street: str,
        chunk_idx: int,
        data: np.ndarray,
    ) -> None:
        """Atomically persist ``data`` as chunk ``chunk_idx`` for ``street``.

        Writes to a temp file, fsyncs, verifies the round-trip, then renames.
        """
        chunk_path = self.get_chunk_path(street, chunk_idx)
        temp_ext = ".tmp.npy.gz" if self.use_compression else ".tmp.npy"
        temp_chunk = chunk_path.parent / f"chunk_{chunk_idx:06d}{temp_ext}"
        try:
            data_to_save = data.astype(self.storage_dtype)
            expected_elements = data_to_save.size

            if self.use_compression:
                with gzip.open(temp_chunk, "wb", compresslevel=4) as f:
                    np.save(f, data_to_save)
            else:
                np.save(temp_chunk, data_to_save)

            # Parallel filesystems buffer writes past close(); fsync guards
            # against silent truncation on node crash or quota exhaustion.
            fd = os.open(str(temp_chunk), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

            # Round-trip verify catches FS-level corruption that fsync misses.
            if self.use_compression:
                with gzip.open(temp_chunk, "rb") as f:
                    verify = np.load(f, allow_pickle=True)
            else:
                verify = np.load(temp_chunk, allow_pickle=True)
            if verify.size != expected_elements:
                raise IOError(
                    f"Verification failed: wrote {expected_elements} elements "
                    f"but read back {verify.size}"
                )
            del verify

            shutil.move(str(temp_chunk), str(chunk_path))
        except Exception as e:
            if temp_chunk.exists():
                temp_chunk.unlink()
            raise RuntimeError(f"Failed to save chunk {chunk_idx}: {e}")

    def load_chunk(
        self, street: str, chunk_idx: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load ``(data_float32, combos_or_None)`` for a chunk."""
        chunk_path = self.get_chunk_path(street, chunk_idx)
        combos_path = self.get_combos_path(street, chunk_idx)

        if self.use_compression:
            with gzip.open(chunk_path, "rb") as f:
                data = np.load(f, allow_pickle=True)
        else:
            data = np.load(chunk_path, allow_pickle=True)
        data = data.astype(np.float32)

        combos = (
            np.load(combos_path, allow_pickle=True)
            if combos_path.exists() else None
        )
        return data, combos

    # ------------------------------------------------------------------
    # Parallel dispatch
    # ------------------------------------------------------------------

    def process_chunks_parallel(
        self,
        street: str,
        chunk_indices: List[int],
        all_combos: np.ndarray,
        item_processor: Callable[[np.ndarray], np.ndarray],
        workers: int,
    ) -> None:
        """Run ``item_processor`` over ``chunk_indices`` in parallel.

        Each worker receives one full chunk; results stream back and are
        persisted + checkpointed incrementally.  Failed chunks stay absent
        from the checkpoint so the next run retries them automatically.
        """
        if not chunk_indices:
            log.info("No chunks to process for %s", street)
            return

        chunk_specs = self.get_chunk_indices(len(all_combos))
        total_chunks = len(chunk_specs)
        log.info(
            "Processing %d chunks with %d workers (1 chunk per worker)...",
            len(chunk_indices), workers,
        )
        start_time = time.time()

        completed_set = set(self.checkpoint.get_completed_chunks(street))
        completed_count = len(completed_set)
        pending_flush = 0
        flush_interval = max(100, workers * 2)
        failed_chunks: List[int] = []

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
        ) as executor:
            futures = {}
            for chunk_idx in chunk_indices:
                _, start_idx, end_idx = chunk_specs[chunk_idx]
                chunk_combos = all_combos[start_idx:end_idx]
                fut = executor.submit(
                    _process_single_chunk_worker,
                    chunk_idx, chunk_combos, item_processor,
                )
                futures[fut] = chunk_idx

            for future in concurrent.futures.as_completed(futures):
                chunk_idx = futures[future]
                try:
                    result_idx, results = future.result()
                    self.save_chunk(street, result_idx, results)

                    if result_idx not in completed_set:
                        completed_set.add(result_idx)
                        pending_flush += 1
                    completed_count += 1

                    if pending_flush >= flush_interval:
                        self.checkpoint.update_completed_chunks(
                            street, completed_set, merge_with_disk=False,
                        )
                        pending_flush = 0

                    is_last = completed_count == total_chunks
                    if (
                        completed_count % 60 == 0
                        or completed_count == 1
                        or is_last
                    ):
                        elapsed = time.time() - start_time
                        rate = completed_count / elapsed if elapsed > 0 else 0
                        remaining = (
                            (total_chunks - completed_count) / rate
                            if rate > 0 else 0
                        )
                        log.info(
                            "[%s] Chunk %d/%d | Elapsed: %.1fm | "
                            "ETA: %.1fm | Rate: %.1f chunks/min",
                            street.upper(), completed_count, total_chunks,
                            elapsed / 60, remaining / 60, rate * 60,
                        )
                except Exception as e:
                    log.error("Chunk %d failed: %s", chunk_idx, e)
                    failed_chunks.append(chunk_idx)

        if pending_flush > 0:
            self.checkpoint.update_completed_chunks(
                street, completed_set, merge_with_disk=False,
            )

        if failed_chunks:
            preview = failed_chunks[:20]
            suffix = " ..." if len(failed_chunks) > 20 else ""
            log.warning(
                "%d chunk(s) failed for %s and will be retried on the "
                "next run: %s%s",
                len(failed_chunks), street, preview, suffix,
            )

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_chunks_to_memmap(
        self,
        street: str,
        dtype: np.dtype = np.float32,
        all_combos_full: Optional[np.ndarray] = None,
    ) -> Tuple[np.memmap, np.ndarray]:
        """Concatenate every chunk into a single memmap + combos array.

        Raises :class:`CorruptChunkError` if any chunk file is truncated;
        the bad chunks are removed and unmarked so the next dispatch retries
        only those.
        """
        log.info("Merging chunks for %s...", street)

        completed_chunks = sorted(self.checkpoint.get_completed_chunks(street))
        if not completed_chunks:
            raise ValueError(f"No completed chunks found for {street}")

        total_chunks = self.checkpoint.get_total_chunks(street)
        if len(completed_chunks) != total_chunks:
            missing = sorted(
                set(range(total_chunks)) - set(completed_chunks)
            )
            preview = missing[:20]
            suffix = " ..." if len(missing) > 20 else ""
            raise ValueError(
                f"Cannot merge {street}: {len(missing)} chunk(s) still "
                f"incomplete (indices {preview}{suffix}). "
                "Re-run to process missing chunks."
            )

        total_combos = self.checkpoint.get_total_combos(street)
        if total_combos is not None:
            total_rows = sum(
                min((idx + 1) * self.chunk_size, total_combos)
                - idx * self.chunk_size
                for idx in completed_chunks
            )
            first_data, _ = self.load_chunk(street, completed_chunks[0])
            feature_dim = (
                first_data.shape[1] if first_data.ndim > 1 else 1
            )
        else:
            log.warning(
                "total_combos not in checkpoint for %s; "
                "falling back to full-scan for row count (slower).",
                street,
            )
            first_data, _ = self.load_chunk(street, completed_chunks[0])
            feature_dim = (
                first_data.shape[1] if first_data.ndim > 1 else 1
            )
            total_rows = first_data.shape[0]
            for chunk_idx in completed_chunks[1:]:
                chunk_data, _ = self.load_chunk(street, chunk_idx)
                total_rows += chunk_data.shape[0]

        log.info(
            "Total rows for %s: %d, feature_dim: %d",
            street, total_rows, feature_dim,
        )
        self.checkpoint.set_feature_dim(street, feature_dim)

        merged_path = self.get_street_dir(street) / "merged_data.dat"
        merged_data = np.memmap(
            merged_path, dtype=dtype, mode="w+",
            shape=(total_rows, feature_dim),
        )

        all_combos: List[np.ndarray] = []
        current_row = 0
        flush_interval = 10
        corrupt_chunks: List[int] = []

        for i, chunk_idx in enumerate(completed_chunks):
            try:
                chunk_data, chunk_combos_file = self.load_chunk(
                    street, chunk_idx,
                )
            except Exception as e:
                log.warning(
                    "Chunk %d for %s is corrupt: %s",
                    chunk_idx, street, e,
                )
                corrupt_chunks.append(chunk_idx)
                continue
            n_rows = chunk_data.shape[0]

            if chunk_data.ndim == 1:
                chunk_data = chunk_data.reshape(-1, 1)

            if corrupt_chunks:
                # Can't write contiguously once the sequence is broken;
                # keep scanning to collect every corrupt index for the raise.
                continue

            merged_data[current_row:current_row + n_rows] = chunk_data

            if all_combos_full is not None:
                start_idx = chunk_idx * self.chunk_size
                end_idx = min(
                    start_idx + self.chunk_size,
                    total_combos
                    if total_combos is not None
                    else start_idx + n_rows,
                )
                all_combos.append(all_combos_full[start_idx:end_idx])
            elif chunk_combos_file is not None:
                all_combos.append(chunk_combos_file)

            current_row += n_rows

            if (i + 1) % flush_interval == 0:
                merged_data.flush()

        if corrupt_chunks:
            valid_set = set(completed_chunks) - set(corrupt_chunks)
            self.checkpoint.update_completed_chunks(
                street, valid_set, merge_with_disk=False,
            )
            self.checkpoint.unmark_merge_done(street)
            for cidx in corrupt_chunks:
                chunk_path = self.get_chunk_path(street, cidx)
                if chunk_path.exists():
                    chunk_path.unlink()
            del merged_data
            if merged_path.exists():
                merged_path.unlink()
            raise CorruptChunkError(
                street=street, corrupt_indices=corrupt_chunks,
            )

        merged_data.flush()

        all_combos_arr = np.concatenate(all_combos, axis=0)
        combos_path = self.get_street_dir(street) / "all_combos.npy"
        temp_combos = combos_path.with_suffix(".tmp.npy")
        try:
            np.save(temp_combos, all_combos_arr)
            shutil.move(str(temp_combos), str(combos_path))
        except Exception as e:
            if temp_combos.exists():
                temp_combos.unlink()
            raise RuntimeError(f"Failed to save combos: {e}")

        log.info(
            "Merged %d chunks into memory-mapped file",
            len(completed_chunks),
        )
        return merged_data, all_combos_arr

    def load_merged_data(
        self, street: str, dtype: np.dtype = np.float32,
    ) -> Tuple[np.memmap, np.ndarray]:
        merged_path = self.get_street_dir(street) / "merged_data.dat"
        combos_path = self.get_street_dir(street) / "all_combos.npy"
        if not merged_path.exists() or not combos_path.exists():
            raise FileNotFoundError(
                f"Merged data not found for {street}"
            )

        all_combos = np.load(combos_path, allow_pickle=True)

        feature_dim = self.checkpoint.get_feature_dim(street)
        if feature_dim is None:
            completed_chunks = sorted(
                self.checkpoint.get_completed_chunks(street)
            )
            if not completed_chunks:
                raise ValueError(f"No completed chunks found for {street}")
            try:
                first_data, _ = self.load_chunk(street, completed_chunks[0])
                feature_dim = (
                    first_data.shape[1] if first_data.ndim > 1 else 1
                )
                self.checkpoint.set_feature_dim(street, feature_dim)
            except FileNotFoundError:
                raise ValueError(
                    f"Cannot determine feature dimension for {street}: "
                    "chunks cleaned up and feature_dim not in checkpoint"
                )

        total_rows = len(all_combos)
        merged_data = np.memmap(
            merged_path, dtype=dtype, mode="r",
            shape=(total_rows, feature_dim),
        )
        log.info(
            "Loaded merged data for %s: %d rows", street, total_rows,
        )
        return merged_data, all_combos

    def get_or_merge_data(
        self,
        street: str,
        dtype: np.dtype = np.float32,
        all_combos_full: Optional[np.ndarray] = None,
    ) -> Tuple[np.memmap, np.ndarray]:
        if self.checkpoint.is_merge_done(street):
            try:
                log.info(
                    "Merge already complete for %s, loading from disk...",
                    street,
                )
                return self.load_merged_data(street, dtype)
            except (FileNotFoundError, ValueError) as e:
                log.warning(
                    "Failed to load merged data: %s. Re-merging from chunks.",
                    e,
                )
        merged_data, all_combos = self.merge_chunks_to_memmap(
            street, dtype, all_combos_full=all_combos_full,
        )
        self.checkpoint.mark_merge_done(street)
        return merged_data, all_combos

    # ------------------------------------------------------------------
    # Centroid / cluster persistence
    # ------------------------------------------------------------------

    def save_centroids(self, street: str, centroids: np.ndarray) -> None:
        path = self.get_street_dir(street) / "centroids.npy"
        self._atomic_np_save(centroids, path)

    def load_centroids(self, street: str) -> np.ndarray:
        return np.load(
            self.get_street_dir(street) / "centroids.npy",
            allow_pickle=True,
        )

    def save_clusters(self, street: str, clusters: np.ndarray) -> None:
        path = self.get_street_dir(street) / "clusters.npy"
        self._atomic_np_save(clusters, path)

    def load_clusters(self, street: str) -> np.ndarray:
        return np.load(
            self.get_street_dir(street) / "clusters.npy",
            allow_pickle=True,
        )

    @staticmethod
    def _atomic_np_save(arr: np.ndarray, path: Path) -> None:
        temp_path = path.with_suffix(".tmp.npy")
        try:
            np.save(temp_path, arr)
            shutil.move(str(temp_path), str(path))
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to save {path}: {e}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_chunks(self, street: str) -> None:
        """Remove per-chunk + per-combo files after a successful merge+cluster."""
        log.info("Cleaning up chunk files for %s...", street)
        chunks_dir = self.get_chunks_dir(street)
        combos_dir = self.get_combos_dir(street)

        deleted_size = 0
        for d in (chunks_dir, combos_dir):
            if d.exists():
                for file in d.rglob("*"):
                    if file.is_file():
                        deleted_size += file.stat().st_size
                shutil.rmtree(d)
        if deleted_size > 0:
            log.info(
                "Freed %.1f MB by removing %s chunk files",
                deleted_size / (1024 ** 2), street,
            )

    def cleanup_partial_clustering(self, street: str) -> None:
        """Remove partial KMeans checkpoint files after success."""
        street_dir = self.get_street_dir(street)
        deleted = []
        for name in ("partial_kmeans.joblib", "kmeans_progress.json"):
            p = street_dir / name
            if p.exists():
                p.unlink()
                deleted.append(p.name)
        if deleted:
            log.info(
                "Cleaned up partial clustering files for %s: %s",
                street, ", ".join(deleted),
            )

    def cleanup_all_intermediate_files(self) -> None:
        log.info("Cleaning up all intermediate files...")
        total_freed = 0
        for street in ("river", "turn", "flop"):
            if not self.checkpoint.is_clustering_done(street):
                log.warning(
                    "Skipping cleanup for %s - clustering not complete",
                    street,
                )
                continue
            street_dir = self.get_street_dir(street)
            if not street_dir.exists():
                continue
            initial = sum(
                f.stat().st_size
                for f in street_dir.rglob("*")
                if f.is_file()
            )
            self.cleanup_chunks(street)
            self.cleanup_partial_clustering(street)
            final = sum(
                f.stat().st_size
                for f in street_dir.rglob("*")
                if f.is_file()
            )
            total_freed += initial - final
        if total_freed > 0:
            log.info(
                "Total disk space freed: %.1f MB",
                total_freed / (1024 ** 2),
            )
