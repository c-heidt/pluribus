"""
Unified card information lookup table builder.

Merges exact and Monte Carlo computation into a single class, applying all
optimisation techniques to both modes:

- O(1) combinadic indexing (``get_row_index``) — no binary search
- Intermediate lookups: turn and flop use pre-computed data from the previous
  street instead of re-sampling from scratch
- Decomposed 7-card evaluation with early termination (exact-mode river)
- Module-level process cache for efficient multiprocessing
- Chunked processing with checkpointing for memory efficiency
- MiniBatchKMeans for large-scale clustering

Only difference between modes is river EHS computation:
  exact       — enumerates ALL opponent hole-card pairs (decomposed evaluation)
  monte_carlo — samples ``n_simulations_river`` random opponent pairs (same
                decomposed evaluation with early termination)

Turn and flop enumerate all future cards and look up pre-computed data
via O(1) index — identical for both modes.
"""
import json
import logging
import os
import shutil
import threading
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from math import comb
except ImportError:
    from scipy.special import comb as _comb

    def comb(n, k):
        return int(_comb(n, k, exact=True))

import joblib
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

from poker_ai.clustering.card_combos import CardCombos, _lex_rank
from poker_ai.clustering.chunked_processor import ChunkedProcessor, CorruptChunkError
from poker_ai.clustering.preflop import compute_preflop_lossless_abstraction
from poker_ai.utils.io import atomic_joblib_dump

log = logging.getLogger("poker_ai.clustering.unified_lut_builder")


# ---------------------------------------------------------------------------
# Memory-efficient street lookup (replaces the per-street Python dict)
# ---------------------------------------------------------------------------

class MemmapLookup:
    """
    Drop-in replacement for the ``{tuple: cluster_id}`` dict that
    ``create_card_lookup`` used to build for each street.

    For a 52-card deck the river has ~2.8 billion combos.  A Python dict for
    that many entries requires ~840 GB of RAM and makes ``joblib.dump`` hang
    for hours.  This class reads cluster IDs directly from the compact uint16
    ``cluster_ids.dat`` memmap (already written by
    ``_on_street_clustering_complete``) using the same O(1) combinadic
    ``get_row_index`` logic — consuming only a few KB when pickled.

    Call ``rebind(new_ids_path)`` after moving ``cluster_ids.dat`` to a new
    location (e.g. when copying results from a scratch workspace to permanent
    storage).
    """

    def __init__(
        self,
        ids_path: str,
        card_to_idx: Dict[int, int],
        n_cards: int,
        n_rows: int,
    ):
        self._ids_path = str(ids_path)
        self._card_to_idx = card_to_idx
        self._n_cards = n_cards
        self._n_rows = n_rows
        self._mm: Optional[np.memmap] = None

    # ------------------------------------------------------------------
    # Lazy memmap loading
    # ------------------------------------------------------------------

    def _load(self):
        if self._mm is None:
            self._mm = np.memmap(
                self._ids_path, dtype=np.uint16, mode="r",
                shape=(self._n_rows,)
            )

    # ------------------------------------------------------------------
    # Dict-like interface expected by the game state
    # ------------------------------------------------------------------

    def __getitem__(self, combo):
        """Return the cluster ID for *combo* (tuple of Card objects or ints)."""
        self._load()
        ints = [int(c) for c in combo]
        row = self._get_row_index(ints[:2], ints[2:])
        return int(self._mm[row])

    # ------------------------------------------------------------------
    # O(1) combinadic row index (mirrors CardCombos.get_row_index)
    # ------------------------------------------------------------------

    def _get_row_index(self, hole_ints, public_ints) -> int:
        card_to_idx = self._card_to_idx
        n = self._n_cards
        h_idx = sorted(card_to_idx[int(c)] for c in hole_ints)
        p_idx = sorted(card_to_idx[int(c)] for c in public_ints)
        hole_rank = _lex_rank(tuple(h_idx), n)
        h0, h1 = h_idx[0], h_idx[1]
        p_reindexed = tuple(p - (h0 < p) - (h1 < p) for p in p_idx)
        n_remaining = n - 2
        k_public = len(p_idx)
        public_rank = _lex_rank(p_reindexed, n_remaining)
        return hole_rank * comb(n_remaining, k_public) + public_rank

    # ------------------------------------------------------------------
    # Path update after copying files to a new location
    # ------------------------------------------------------------------

    def rebind(self, new_ids_path: str):
        """Point this lookup at a new ``cluster_ids.dat`` path and reset the memmap."""
        self._ids_path = str(new_ids_path)
        self._mm = None

    # ------------------------------------------------------------------
    # Pickle support — do not serialise the memmap handle
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# ---------------------------------------------------------------------------
# Module-level process cache for multiprocessing
# ---------------------------------------------------------------------------
# Worker processes maintain their own cache (lazy loaded on first use).
# With O(1) combinadic indexing, workers only need the memmap and centroids —
# no combo arrays or hole-card index needed.

_PROCESS_CACHE = {
    # Pre-computed cluster IDs — used by turn/flop workers for O(1) lookup.
    # ``cluster_ids[row_idx]`` is a uint16 giving the cluster for that combo.
    "river_cluster_ids": None,
    "turn_cluster_ids": None,
    "save_dir": None,
}

_CACHE_LOAD_LOCK = threading.Lock()



def _clear_process_cache_for_street(street: str):
    """Clear cached cluster IDs for a street to free memory."""
    _PROCESS_CACHE[f"{street}_cluster_ids"] = None
    log.debug(f"Cleared process cache for {street}")


def _get_cluster_id_cache(street: str, save_dir: str, n_rows: int) -> np.memmap:
    """
    Get or load pre-computed cluster IDs for a street.

    Returns a uint16 memory-mapped array of shape ``(n_rows,)`` where
    ``array[row_idx]`` is the cluster ID for that combo.

    Parameters
    ----------
    street : str
        ``"river"`` or ``"turn"``.
    save_dir : str
        Path to the save directory.
    n_rows : int
        Total number of combos for this street.
    """
    # Invalidate if save_dir changed (different run in same worker process)
    if _PROCESS_CACHE["save_dir"] != save_dir:
        _PROCESS_CACHE["save_dir"] = save_dir
        _PROCESS_CACHE["river_cluster_ids"] = None
        _PROCESS_CACHE["turn_cluster_ids"] = None

    cache_key = f"{street}_cluster_ids"
    if _PROCESS_CACHE[cache_key] is not None:
        return _PROCESS_CACHE[cache_key]

    with _CACHE_LOAD_LOCK:
        if _PROCESS_CACHE[cache_key] is not None:
            return _PROCESS_CACHE[cache_key]

        ids_path = Path(save_dir) / street / "cluster_ids.dat"
        if not ids_path.exists():
            raise FileNotFoundError(
                f"cluster_ids.dat not found for {street} at {ids_path}. "
                "Re-run clustering from scratch."
            )

        cluster_ids = np.memmap(
            ids_path, dtype=np.uint16, mode="r", shape=(n_rows,)
        )
        _PROCESS_CACHE[cache_key] = cluster_ids
        log.debug(
            f"Worker {os.getpid()} loaded {street} cluster_ids "
            f"({n_rows:,} entries)"
        )
        return cluster_ids


# ---------------------------------------------------------------------------
# Unified builder
# ---------------------------------------------------------------------------

class UnifiedLutBuilder(CardCombos):
    """
    Build card information lookup tables for poker AI clustering.

    Supports two computation methods selected via *method*:

    - ``"exact"``       — exhaustive evaluation (precise, slower)
    - ``"monte_carlo"`` — opponent sampling at the river (faster, configurable)

    In both modes, turn and flop use intermediate lookups into pre-computed
    street data via O(1) combinadic indexing.  The only branching point is
    the river, where *exact* enumerates all opponent pairs and *MC* samples.

    Attributes
    ----------
    card_info_lut : Dict[str, Any]
        ``{street: {combo_tuple: cluster_id, ...}}``.
    centroids : Dict[str, np.ndarray]
        ``{street: centroids_array}``.
    """

    def __init__(
        self,
        method: str = "monte_carlo",
        n_simulations_river: int = 6,
        low_card_rank: int = 2,
        high_card_rank: int = 14,
        save_dir: str = "",
        workers: Optional[int] = None,
        chunk_size: int = 10000,
        use_mini_batch: bool = True,
        parallel_combos: bool = True,
    ):
        """
        Parameters
        ----------
        method : str
            ``"exact"`` or ``"monte_carlo"``.
        n_simulations_river : int
            Opponent samples per river combo (MC mode only; ignored for exact).
        low_card_rank : int
            Lowest card rank (2-14).
        high_card_rank : int
            Highest card rank (2-14).
        save_dir : str
            Directory for results and checkpoints.
        workers : Optional[int]
            Worker processes.  Defaults to ``cpu_count()``.
        chunk_size : int
            Combos per processing chunk.
        use_mini_batch : bool
            Use MiniBatchKMeans for datasets with >50 000 samples.
        parallel_combos : bool
            Parallel combo generation (recommended for >15 cards).
        """
        self.method = method.lower()
        if self.method not in ("exact", "monte_carlo"):
            raise ValueError(
                f"method must be 'exact' or 'monte_carlo', got '{method}'"
            )

        self.n_simulations_river = n_simulations_river
        self.workers = workers
        self.chunk_size = chunk_size
        self.use_mini_batch = use_mini_batch

        super().__init__(
            low_card_rank, high_card_rank,
            parallel=parallel_combos, n_workers=workers,
        )

        # Evaluator ----------------------------------------------------------
        from poker_ai.environment.evaluation import Evaluator
        self._evaluator = Evaluator()

        # File paths ----------------------------------------------------------
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.card_info_lut_path: Path = self.save_dir / "card_info_lut.joblib"
        self.centroid_path: Path = self.save_dir / "centroids.joblib"

        # Chunked processing --------------------------------------------------
        self.chunked_processor = ChunkedProcessor(
            save_dir=self.save_dir,
            chunk_size=chunk_size,
            use_compression=False,
        )

        # Load existing results -----------------------------------------------
        try:
            self.card_info_lut: Dict[str, Any] = joblib.load(
                self.card_info_lut_path
            )
            self.centroids: Dict[str, Any] = joblib.load(self.centroid_path)
        except FileNotFoundError:
            self.centroids: Dict[str, Any] = {}
            self.card_info_lut: Dict[str, Any] = {}

        # Config for validation / serialisation --------------------------------
        self._config: Dict[str, Any] = {
            "method": self.method,
            "low_card_rank": low_card_rank,
            "high_card_rank": high_card_rank,
            "chunk_size": chunk_size,
            "save_dir": str(self.save_dir),
        }
        if self.method == "monte_carlo":
            self._config["n_simulations_river"] = n_simulations_river

    # ------------------------------------------------------------------
    # Serialisation helpers (multiprocessing)
    # ------------------------------------------------------------------

    def __getstate__(self):
        """Reduce pickle size — workers use module-level cache instead.

        Large combo arrays (river, turn, flop, starting_hands) are excluded
        because worker functions only use _card_ints, _card_to_idx, _n_cards,
        _evaluator, _config, method, and n_simulations_river.  For a 52-card
        deck self.river alone is ~78 GB, which exceeds the pickle 4 GiB limit
        and causes every chunk submission to fail.
        """
        state = self.__dict__.copy()
        state["card_info_lut"] = {}
        state["centroids"] = {}
        state["chunked_processor"] = None
        # Exclude large CardCombos arrays — workers don't need them.
        # These are the backing attributes for the lazy river/turn/flop properties.
        state["_river"] = None
        state["_turn"] = None
        state["_flop"] = None
        state["starting_hands"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Worker count
    # ------------------------------------------------------------------

    def _get_worker_count(self) -> int:
        if self.workers and int(self.workers) > 0:
            return int(self.workers)
        return os.cpu_count() or 1

    # ==================================================================
    # Main compute pipeline
    # ==================================================================

    def compute(
        self,
        n_river_clusters: int,
        n_turn_clusters: int,
        n_flop_clusters: int,
    ):
        """
        Run the full clustering pipeline: preflop -> river -> turn -> flop.

        Saves after each street for resumability.

        Parameters
        ----------
        n_river_clusters : int
            Number of river clusters.
        n_turn_clusters : int
            Number of turn clusters.
        n_flop_clusters : int
            Number of flop clusters.
        """
        log.info(
            f"Starting clustering using {self.method.upper()} method "
            f"({self._n_cards} cards)."
        )
        start = time.time()

        self._config.update({
            "n_river_clusters": n_river_clusters,
            "n_turn_clusters": n_turn_clusters,
            "n_flop_clusters": n_flop_clusters,
        })

        n = self._n_cards

        # -- Preflop (lossless) -----------------------------------------------
        if "pre_flop" not in self.card_info_lut:
            log.info("Computing pre-flop abstraction...")
            self.card_info_lut["pre_flop"] = (
                compute_preflop_lossless_abstraction(builder=self)
            )
            atomic_joblib_dump(self.card_info_lut, self.card_info_lut_path)

        # -- River -------------------------------------------------------------
        n_river_rows = comb(n, 2) * comb(n - 2, 5)
        if "river" not in self.card_info_lut:
            self.card_info_lut["river"] = self._compute_street_clusters(
                "river", n_river_clusters, self.river,  # lazy-built here
                self._process_river_chunks,
            )
            self.river = None  # free memory before next stage
            atomic_joblib_dump(self.card_info_lut, self.card_info_lut_path)
            atomic_joblib_dump(self.centroids, self.centroid_path)
        else:
            # Resume: pre-populate cache so turn workers inherit it via fork
            # and never need to load from disk themselves.
            _get_cluster_id_cache(
                "river", self._config["save_dir"], n_river_rows,
            )

        # -- Turn --------------------------------------------------------------
        n_turn_rows = comb(n, 2) * comb(n - 2, 4)
        if "turn" not in self.card_info_lut:
            self.card_info_lut["turn"] = self._compute_street_clusters(
                "turn", n_turn_clusters, self.turn,  # lazy-built here
                self._process_turn_chunks,
            )
            self.turn = None  # free memory before next stage
            atomic_joblib_dump(self.card_info_lut, self.card_info_lut_path)
            atomic_joblib_dump(self.centroids, self.centroid_path)
        else:
            # Resume: pre-populate cache so flop workers inherit it via fork.
            _get_cluster_id_cache(
                "turn", self._config["save_dir"], n_turn_rows,
            )

        # -- Flop --------------------------------------------------------------
        if "flop" not in self.card_info_lut:
            self.card_info_lut["flop"] = self._compute_street_clusters(
                "flop", n_flop_clusters, self.flop,  # lazy-built here
                self._process_flop_chunks,
            )
            self.flop = None  # free memory
            atomic_joblib_dump(self.card_info_lut, self.card_info_lut_path)
            atomic_joblib_dump(self.centroids, self.centroid_path)

        end = time.time()
        log.info(f"Finished clustering — {end - start:.2f}s total.")
        log.info("Cleaning up intermediate files...")
        self.chunked_processor.cleanup_all_intermediate_files()

    # ------------------------------------------------------------------
    # Generic street clustering (eliminates per-street code duplication)
    # ------------------------------------------------------------------

    def _compute_street_clusters(
        self,
        street: str,
        n_clusters: int,
        combos: np.ndarray,
        chunk_processor_fn,
    ) -> Dict:
        """
        Compute clusters for a single street.

        Handles chunked EHS computation, merging, KMeans clustering,
        saving results, and process-cache pre-population.

        Parameters
        ----------
        street : str
            ``"river"``, ``"turn"``, or ``"flop"``.
        n_clusters : int
            Number of clusters to create.
        combos : np.ndarray
            All card combinations for this street.
        chunk_processor_fn : callable
            Function that processes incomplete chunks for this street.

        Returns
        -------
        Dict
            ``{combo_tuple: cluster_id}`` lookup table.
        """
        stage_map = {"river": "1/3", "turn": "2/3", "flop": "3/3"}
        log.info(f"\n{'=' * 80}")
        log.info(f"STAGE {stage_map[street]}: {street.upper()} CLUSTERING")
        log.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"Clusters: {n_clusters} | Total combos: {len(combos):,}")
        log.info(f"{'=' * 80}\n")
        start = time.time()

        total_combos = len(combos)
        self.chunked_processor.initialize_street(
            street, total_combos, self._config,
        )

        # Process incomplete chunks
        incomplete_chunks = self.chunked_processor.get_incomplete_chunks(street)
        if incomplete_chunks:
            log.info(
                f"Processing {len(incomplete_chunks)} incomplete chunks "
                f"for {street}"
            )
            chunk_processor_fn(incomplete_chunks)

        # Guard: refuse to merge if any chunks are still missing.  Merging
        # partial data would silently produce a card_info_lut with gaps.
        still_incomplete = self.chunked_processor.get_incomplete_chunks(street)
        if still_incomplete:
            raise RuntimeError(
                f"{len(still_incomplete)} chunk(s) failed for {street} and must be "
                f"retried before clustering can proceed. "
                f"Re-run the script to retry them automatically."
            )

        # Merge and cluster, with automatic recovery for corrupt chunks.
        # If a killed HPC job left truncated chunk files on disk, the merge
        # detects them, deletes the bad files, and raises CorruptChunkError.
        # We then reprocess just those chunks and retry the merge — no need
        # to resubmit the whole job.
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                merged_data, all_combos, clusters = self._merge_and_cluster(
                    street, n_clusters, combos, chunk_processor_fn,
                )
                break
            except CorruptChunkError as e:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Still have corrupt chunks for {street} after "
                        f"{max_retries} recovery attempts. "
                        f"Last error: {e}"
                    ) from e
                log.warning(
                    f"Attempt {attempt}/{max_retries}: {e}. "
                    f"Reprocessing {len(e.corrupt_indices)} corrupt chunk(s)..."
                )
                chunk_processor_fn(e.corrupt_indices)

        # Pre-populate process cache for the next street's workers
        self._on_street_clustering_complete(
            street, merged_data, all_combos, clusters,
        )

        end = time.time()
        log.info(f"Finished {street} clusters — {end - start:.2f}s.")

        # Return a MemmapLookup instead of a full Python dict.
        # For a 52-card deck the river has ~2.8 billion combos; building a
        # Python dict for that requires ~840 GB RAM and makes joblib.dump
        # hang for hours.  MemmapLookup reads directly from the compact
        # cluster_ids.dat memmap that _on_street_clustering_complete already
        # wrote, serialising to only a few KB.
        ids_path = (
            Path(self._config["save_dir"]) / street / "cluster_ids.dat"
        )
        return MemmapLookup(
            ids_path=ids_path,
            card_to_idx=self._card_to_idx,
            n_cards=self._n_cards,
            n_rows=len(all_combos),
        )

    def _merge_and_cluster(
        self,
        street: str,
        n_clusters: int,
        combos: np.ndarray,
        chunk_processor_fn,
    ) -> Tuple[np.memmap, np.ndarray, np.ndarray]:
        """Merge chunks and cluster. Raises CorruptChunkError on bad files."""
        if self.chunked_processor.is_clustering_done(street):
            log.info(f"Loading existing clustering results for {street}")
            self.centroids[street] = self.chunked_processor.load_centroids(
                street
            )
            clusters = self.chunked_processor.load_clusters(street)
            merged_data, all_combos = (
                self.chunked_processor.get_or_merge_data(
                    street, all_combos_full=combos,
                )
            )
        else:
            merged_data, all_combos = (
                self.chunked_processor.get_or_merge_data(
                    street, all_combos_full=combos,
                )
            )
            self.centroids[street], clusters = self._cluster(
                num_clusters=n_clusters, X=merged_data, street=street,
            )
            self.chunked_processor.save_centroids(
                street, self.centroids[street],
            )
            self.chunked_processor.save_clusters(street, clusters)
            self.chunked_processor.mark_clustering_done(street)
            log.info(
                f"Cleaning up intermediate chunk files for {street}..."
            )
            self.chunked_processor.cleanup_chunks(street)
            self.chunked_processor.cleanup_partial_clustering(street)
        return merged_data, all_combos, clusters

    # ------------------------------------------------------------------
    # Chunk dispatch
    # ------------------------------------------------------------------

    def _process_river_chunks(self, chunk_indices: List[int]):
        workers = self._get_worker_count()
        self.chunked_processor.process_chunks_parallel(
            street="river",
            chunk_indices=chunk_indices,
            all_combos=self.river,
            item_processor=self.process_river_ehs,
            workers=workers,
        )

    def _process_turn_chunks(self, chunk_indices: List[int]):
        workers = self._get_worker_count()
        self.chunked_processor.process_chunks_parallel(
            street="turn",
            chunk_indices=chunk_indices,
            all_combos=self.turn,
            item_processor=self.process_turn_ehs_distributions,
            workers=workers,
        )

    def _process_flop_chunks(self, chunk_indices: List[int]):
        workers = self._get_worker_count()
        self.chunked_processor.process_chunks_parallel(
            street="flop",
            chunk_indices=chunk_indices,
            all_combos=self.flop,
            item_processor=self.process_flop_potential_aware_distributions,
            workers=workers,
        )

    # ------------------------------------------------------------------
    # Process cache management
    # ------------------------------------------------------------------

    def _on_street_clustering_complete(
        self,
        street: str,
        merged_data: np.ndarray,
        all_combos: np.ndarray,
        clusters: np.ndarray,
    ):
        """
        Save cluster IDs as a uint16 memmap and pre-populate the process cache
        so the next street's workers can do a direct O(1) integer lookup.
        """
        save_dir = self._config["save_dir"]
        ids_path = Path(save_dir) / street / "cluster_ids.dat"

        if not ids_path.exists():
            cluster_ids_mm = np.memmap(
                ids_path, dtype=np.uint16, mode="w+", shape=(len(clusters),)
            )
            cluster_ids_mm[:] = clusters.astype(np.uint16)
            cluster_ids_mm.flush()
        else:
            cluster_ids_mm = np.memmap(
                ids_path, dtype=np.uint16, mode="r", shape=(len(clusters),)
            )

        _PROCESS_CACHE["save_dir"] = save_dir
        _PROCESS_CACHE[f"{street}_cluster_ids"] = cluster_ids_mm
        log.info(
            f"Saved cluster_ids for {street} "
            f"({len(clusters):,} entries → {ids_path.name})"
        )

    # ==================================================================
    # Clustering
    # ==================================================================

    def _cluster(
        self,
        num_clusters: int,
        X: np.ndarray,
        street: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster data using KMeans or MiniBatchKMeans.

        Uses MiniBatchKMeans for large datasets (>50 000 samples) with
        resumable checkpointing.

        Parameters
        ----------
        num_clusters : int
            Number of clusters.
        X : np.ndarray
            Data to cluster (may be memory-mapped).
        street : str
            Street name for logging.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(centroids, cluster_assignments)``.
        """
        n_samples = X.shape[0]
        log.info(
            f"Clustering {n_samples:,} samples into "
            f"{num_clusters} clusters ({street})"
        )

        if n_samples < num_clusters:
            log.warning(
                f"Samples ({n_samples}) < clusters ({num_clusters}). "
                f"Reducing to {n_samples}."
            )
            num_clusters = n_samples

        if n_samples == 1:
            return X.copy(), np.array([0])

        use_minibatch = self.use_mini_batch and n_samples > 50000

        if use_minibatch:
            return self._cluster_minibatch(
                num_clusters, X, street, n_samples,
            )

        log.info(f"Using standard KMeans for {street}")
        km = KMeans(
            n_clusters=num_clusters,
            init="random",
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0,
        )
        y_km = km.fit_predict(X)
        return km.cluster_centers_, y_km

    def _cluster_minibatch(
        self,
        num_clusters: int,
        X: np.ndarray,
        street: str,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """MiniBatchKMeans with resumable checkpointing."""
        log.info(f"Using MiniBatchKMeans for {street} (large dataset)")
        batch_size = min(10000, n_samples)

        partial_km_path = (
            self.chunked_processor.get_street_dir(street)
            / "partial_kmeans.joblib"
        )
        progress_path = (
            self.chunked_processor.get_street_dir(street)
            / "kmeans_progress.json"
        )

        start_batch = 0
        if partial_km_path.exists() and progress_path.exists():
            try:
                log.info(f"Resuming partial clustering for {street}")
                km = joblib.load(partial_km_path)
                with open(progress_path, "r") as f:
                    progress = json.load(f)
                start_batch = progress.get("completed_batches", 0)
                log.info(f"Resuming from batch {start_batch}")
            except Exception as e:
                log.warning(
                    f"Failed to load checkpoint, starting fresh: {e}"
                )
                start_batch = 0
                km = self._make_minibatch_km(num_clusters, batch_size)
        else:
            km = self._make_minibatch_km(num_clusters, batch_size)

        n_batches = (n_samples + batch_size - 1) // batch_size
        checkpoint_interval = max(1, n_batches // 10)

        for i in tqdm(
            range(start_batch, n_batches),
            desc=f"Clustering {street}",
            initial=start_batch,
            total=n_batches,
        ):
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, n_samples)
            batch_data = np.array(X[batch_start:batch_end])
            km.partial_fit(batch_data)

            if (i + 1) % checkpoint_interval == 0 or (i + 1) == n_batches:
                log.info(
                    f"Clustering checkpoint: {i + 1}/{n_batches} "
                    f"({100 * (i + 1) // n_batches}%)"
                )
                temp_km = partial_km_path.with_suffix(".tmp.joblib")
                temp_pg = progress_path.with_suffix(".tmp.json")
                try:
                    joblib.dump(km, temp_km)
                    with open(temp_pg, "w") as f:
                        json.dump(
                            {
                                "completed_batches": i + 1,
                                "total_batches": n_batches,
                            },
                            f,
                        )
                    shutil.move(str(temp_km), str(partial_km_path))
                    shutil.move(str(temp_pg), str(progress_path))
                except Exception as e:
                    log.warning(f"Checkpoint save failed: {e}")
                    for p in [temp_km, temp_pg]:
                        if p.exists():
                            p.unlink()

        log.info(
            f"Predicting cluster assignments for {n_samples:,} samples..."
        )
        y_km = km.predict(X)
        centroids = km.cluster_centers_

        for p in [partial_km_path, progress_path]:
            if p.exists():
                p.unlink()
        log.info(f"Cleaned up partial clustering checkpoints for {street}")

        return centroids, y_km

    @staticmethod
    def _make_minibatch_km(num_clusters: int, batch_size: int):
        return MiniBatchKMeans(
            n_clusters=num_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            batch_size=batch_size,
            random_state=0,
            verbose=0,
        )

    # ==================================================================
    # RIVER — the only branching point between exact and MC
    # ==================================================================

    def process_river_ehs(self, public: np.ndarray) -> np.ndarray:
        """
        Compute expected hand strength for a river combo.

        Dispatches to exact or MC based on ``self.method``.

        Parameters
        ----------
        public : np.ndarray
            ``[hole1, hole2, flop1, flop2, flop3, turn, river]``

        Returns
        -------
        np.ndarray
            ``[win_rate, loss_rate, tie_rate]``
        """
        our_hand = public[:2]
        board = public[2:7]
        if self.method == "exact":
            return self._compute_exact_river_ehs(our_hand, board)
        return self._compute_mc_river_ehs(our_hand, board)

    # -- exact river -------------------------------------------------------

    def _compute_exact_river_ehs(
        self, our_hand: np.ndarray, board: np.ndarray,
    ) -> np.ndarray:
        """
        Exact river EHS via decomposed 7-card evaluation.

        A 7-card hand has C(7,5) = 21 five-card subsets in three categories:

        - Category 0: 5 board cards alone (1 subset, constant)
        - Category 1: 4 board + 1 hole card (5 per card, pre-computable)
        - Category 2: 3 board + 2 hole cards (10 per pair, per-opponent)

        Pre-computing categories 0 and 1 cuts per-opponent ``_five()`` calls
        from 21 down to at most 10, and early termination often skips them
        entirely.

        Parameters
        ----------
        our_hand : np.ndarray
            Our two hole cards.
        board : np.ndarray
            Five community cards on the river.

        Returns
        -------
        np.ndarray
            ``[win_rate, loss_rate, tie_rate]``
        """
        unavailable = set(our_hand.tolist() + board.tolist())
        available = [c for c in self._card_ints if c not in unavailable]

        board_ints = [int(c) for c in board]
        our_ints = [int(c) for c in our_hand]

        _five = self._evaluator._five
        our_rank = self._evaluator._seven(our_ints + board_ints)

        # Category 0: board-only rank
        board_only_rank = _five(board_ints)

        # Pre-compute board subsets
        board_4 = list(combinations(board_ints, 4))
        board_3 = list(combinations(board_ints, 3))

        # Category 1: best rank per single available card
        single_best = {}
        for c in available:
            ci = int(c)
            best = board_only_rank
            for b4 in board_4:
                s = _five(b4 + (ci,))
                if s < best:
                    best = s
            single_best[c] = best

        # Enumerate all opponent pairs
        wins = losses = ties = 0
        for o1, o2 in combinations(available, 2):
            opp_rank = min(single_best[o1], single_best[o2])

            # Category 2 only if outcome not yet determined
            if opp_rank >= our_rank:
                o1i, o2i = int(o1), int(o2)
                pair = (o1i, o2i)
                for b3 in board_3:
                    s = _five(b3 + pair)
                    if s < opp_rank:
                        opp_rank = s
                        if opp_rank < our_rank:
                            break  # outcome determined

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

    # -- MC river ----------------------------------------------------------

    def _compute_mc_river_ehs(
        self, our_hand: np.ndarray, board: np.ndarray,
    ) -> np.ndarray:
        """
        MC river EHS: sample ``n_simulations_river`` random opponent pairs.

        Uses the same decomposed evaluation as exact mode — precomputes
        Category 0 (board-only) and Category 1 (board+1 hole) ranks once,
        then applies Category 2 with early termination only when needed.

        Parameters
        ----------
        our_hand : np.ndarray
            Our two hole cards.
        board : np.ndarray
            Five community cards on the river.

        Returns
        -------
        np.ndarray
            ``[win_rate, loss_rate, tie_rate]``
        """
        unavailable = set(our_hand.tolist() + board.tolist())
        available = [c for c in self._card_ints if c not in unavailable]

        board_ints = [int(c) for c in board]
        our_ints = [int(c) for c in our_hand]

        _five = self._evaluator._five
        our_rank = self._evaluator._seven(our_ints + board_ints)

        board_only_rank = _five(board_ints)
        board_4 = list(combinations(board_ints, 4))
        board_3 = list(combinations(board_ints, 3))

        # Category 1: best rank per single available card (precomputed once)
        single_best = {}
        for c in available:
            ci = int(c)
            best = board_only_rank
            for b4 in board_4:
                s = _five(b4 + (ci,))
                if s < best:
                    best = s
            single_best[c] = best

        available_arr = np.array(available)
        n = self.n_simulations_river
        wins = losses = ties = 0

        for _ in range(n):
            o1, o2 = np.random.choice(available_arr, 2, replace=False)
            opp_rank = min(single_best[o1], single_best[o2])

            # Category 2 only when outcome not yet determined
            if opp_rank >= our_rank:
                pair = (int(o1), int(o2))
                for b3 in board_3:
                    s = _five(b3 + pair)
                    if s < opp_rank:
                        opp_rank = s
                        if opp_rank < our_rank:
                            break

            if our_rank > opp_rank:
                wins += 1
            elif our_rank < opp_rank:
                losses += 1
            else:
                ties += 1

        return np.array([wins / n, losses / n, ties / n])

    # ==================================================================
    # TURN — identical for both modes (intermediate lookup)
    # ==================================================================

    def process_turn_ehs_distributions(
        self, public: np.ndarray,
    ) -> np.ndarray:
        """
        Compute turn distribution over river clusters.

        Enumerates ALL remaining river cards and looks up pre-computed
        river EHS via O(1) combinadic index.

        Parameters
        ----------
        public : np.ndarray
            ``[hole1, hole2, flop1, flop2, flop3, turn]``

        Returns
        -------
        np.ndarray
            Distribution over river clusters (normalised frequencies).
        """
        our_hand = public[:2]
        board = public[2:6]
        return self._compute_turn_distribution(our_hand, board)

    def _compute_turn_distribution(
        self, our_hand: np.ndarray, board: np.ndarray,
    ) -> np.ndarray:
        """
        Enumerate all river cards and build a histogram over river clusters.

        Each future river card maps to its pre-computed cluster via a direct
        uint16 read: ``river_cluster_ids[get_row_index(hand, board)]``.
        """
        n = self._n_cards
        n_river_rows = comb(n, 2) * comb(n - 2, 5)
        river_cluster_ids = _get_cluster_id_cache(
            "river", self._config["save_dir"], n_river_rows,
        )

        n_river_clusters = self._config["n_river_clusters"]
        unavailable = set(our_hand.tolist() + board.tolist())
        available = [c for c in self._card_ints if c not in unavailable]
        dist = np.zeros(n_river_clusters)

        river_board = np.empty(5, dtype=np.int32)
        river_board[:4] = board
        for river_card in available:
            river_board[4] = river_card
            dist[int(river_cluster_ids[self.get_row_index(our_hand, river_board)])] += 1

        total = dist.sum()
        if total > 0:
            dist /= total
        return dist

    # ==================================================================
    # FLOP — identical for both modes (intermediate lookup)
    # ==================================================================

    def process_flop_potential_aware_distributions(
        self, public: np.ndarray,
    ) -> np.ndarray:
        """
        Compute flop distribution over turn clusters.

        Enumerates ALL remaining turn cards and looks up pre-computed
        turn distribution via O(1) combinadic index.

        Parameters
        ----------
        public : np.ndarray
            ``[hole1, hole2, flop1, flop2, flop3]``

        Returns
        -------
        np.ndarray
            Distribution over turn clusters (normalised frequencies).
        """
        our_hand = public[:2]
        board = public[2:5]
        return self._compute_flop_distribution(our_hand, board)

    def _compute_flop_distribution(
        self, our_hand: np.ndarray, board: np.ndarray,
    ) -> np.ndarray:
        """
        Enumerate all turn cards and build a histogram over turn clusters.

        Each future turn card maps to its pre-computed cluster via a direct
        uint16 read: ``turn_cluster_ids[get_row_index(hand, board)]``.
        """
        # Free river cluster IDs from cache — flop stage doesn't need them
        _clear_process_cache_for_street("river")

        n = self._n_cards
        n_turn_rows = comb(n, 2) * comb(n - 2, 4)
        turn_cluster_ids = _get_cluster_id_cache(
            "turn", self._config["save_dir"], n_turn_rows,
        )

        n_turn_clusters = self._config["n_turn_clusters"]
        unavailable = set(our_hand.tolist() + board.tolist())
        available = [c for c in self._card_ints if c not in unavailable]
        dist = np.zeros(n_turn_clusters)

        turn_board = np.empty(4, dtype=np.int32)
        turn_board[:3] = board
        for turn_card in available:
            turn_board[3] = turn_card
            dist[int(turn_cluster_ids[self.get_row_index(our_hand, turn_board)])] += 1

        total = dist.sum()
        if total > 0:
            dist /= total
        return dist

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def get_available_cards(
        cards: np.ndarray, unavailable_cards: np.ndarray,
    ) -> np.ndarray:
        """Return cards not in *unavailable_cards*."""
        unavailable = set(unavailable_cards.tolist())
        return np.array([c for c in cards if c not in unavailable])

    @staticmethod
    def create_card_lookup(
        clusters: np.ndarray, card_combos: np.ndarray,
    ) -> Dict:
        """Create ``{combo_tuple: cluster_id}`` lookup table."""
        log.info("Creating lookup table.")
        return {
            tuple(card_combo): clusters[i]
            for i, card_combo in enumerate(tqdm(card_combos))
        }
