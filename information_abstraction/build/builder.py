"""Orchestrator for the card-information abstraction build pipeline.

:class:`AbstractionBuilder` composes
:class:`information_abstraction.build.card_combos.CardCombos`,
:class:`information_abstraction.build.chunk_store.ChunkStore`,
:class:`information_abstraction.build.clusterer.Clusterer`, and the
per-street feature extractors in
:mod:`information_abstraction.build.ehs`.  Its :meth:`compute`
method drives the preflop → river → turn → flop pipeline and writes
``card_info_lut.joblib`` + ``centroids.joblib`` to ``save_dir``.
"""
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np

from information_abstraction.build.card_combos import CardCombos
from information_abstraction.build.chunk_store import (
    ChunkStore,
    CorruptChunkError,
)
from information_abstraction.build.clusterer import Clusterer
from information_abstraction.build.ehs import (
    FlopEHS,
    RiverEHS,
    TurnEHS,
    _PROCESS_CACHE,
    get_cluster_id_cache,
)
from information_abstraction.lookup import MemmapLookup, comb
from information_abstraction.preflop import (
    compute_preflop_lossless_abstraction,
)
from utils.io import atomic_joblib_dump

log = logging.getLogger("information_abstraction.build.builder")


_STAGE_LABELS = {"river": "1/3", "turn": "2/3", "flop": "3/3"}
_MAX_MERGE_RETRIES = 3


class AbstractionBuilder:
    """Build the card-information abstraction and persist it to disk.

    Parameters
    ----------
    method : str
        ``"exact"`` enumerates all opponent pairs at the river;
        ``"monte_carlo"`` samples ``n_simulations_river`` pairs.  Turn and
        flop are identical in both modes (intermediate lookups).
    n_simulations_river : int
        River opponent samples (MC mode only).
    low_card_rank, high_card_rank : int
        Deck rank bounds (2-14).
    save_dir : str
        Output directory for ``card_info_lut.joblib``, ``centroids.joblib``,
        and the per-street chunk + memmap state.
    workers : int, optional
        Worker-process count.  Defaults to ``cpu_count()``.
    chunk_size : int
        Combos per chunk.
    use_mini_batch : bool
        Use MiniBatchKMeans for datasets with more than 50 000 samples.
    parallel_combos : bool
        Parallel combo generation (recommended for large decks).
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
        self.method = method.lower()
        if self.method not in ("exact", "monte_carlo"):
            raise ValueError(
                f"method must be 'exact' or 'monte_carlo', got '{method}'"
            )
        self.n_simulations_river = n_simulations_river
        self.workers = workers
        self.chunk_size = chunk_size
        self.use_mini_batch = use_mini_batch

        # Combo generation (composition, not inheritance).
        self.combos = CardCombos(
            low_card_rank, high_card_rank,
            parallel=parallel_combos, n_workers=workers,
        )

        # Evaluator.
        from environment.evaluator import Evaluator
        self._evaluator = Evaluator()

        # Paths + persistence.
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.card_info_lut_path: Path = self.save_dir / "card_info_lut.joblib"
        self.centroid_path: Path = self.save_dir / "centroids.joblib"

        # Chunk + cluster persistence.
        self.chunk_store = ChunkStore(
            save_dir=self.save_dir,
            chunk_size=chunk_size,
            use_compression=False,
        )
        self.clusterer = Clusterer(use_mini_batch=use_mini_batch)

        # Previously-computed results (for resume).
        try:
            self.card_info_lut: Dict[str, Any] = joblib.load(
                self.card_info_lut_path
            )
            self.centroids: Dict[str, Any] = joblib.load(self.centroid_path)
        except FileNotFoundError:
            self.card_info_lut = {}
            self.centroids = {}

        # Config used by workers and persisted to the checkpoint.
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
    # Convenience proxies onto the combo generator
    # ------------------------------------------------------------------

    @property
    def _card_ints(self) -> np.ndarray:
        """Exposed so ``compute_preflop_lossless_abstraction`` can read it."""
        return self.combos._card_ints

    @property
    def starting_hands(self) -> np.ndarray:
        """Exposed so ``compute_preflop_lossless_abstraction`` can read it."""
        return self.combos.starting_hands

    @property
    def _card_to_idx(self) -> Dict[int, int]:
        return self.combos._card_to_idx

    @property
    def _n_cards(self) -> int:
        return self.combos._n_cards

    def _get_worker_count(self) -> int:
        if self.workers and int(self.workers) > 0:
            return int(self.workers)
        return os.cpu_count() or 1

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def compute(
        self,
        n_river_clusters: int,
        n_turn_clusters: int,
        n_flop_clusters: int,
    ) -> None:
        """Run preflop → river → turn → flop and persist after each street."""
        log.info(
            f"Starting abstraction build using {self.method.upper()} method "
            f"({self._n_cards} cards).",
        )
        start = time.time()

        self._config.update({
            "n_river_clusters": n_river_clusters,
            "n_turn_clusters": n_turn_clusters,
            "n_flop_clusters": n_flop_clusters,
        })
        n = self._n_cards

        self._run_preflop()

        n_river_rows = comb(n, 2) * comb(n - 2, 5)
        if "river" not in self.card_info_lut:
            self.card_info_lut["river"] = self._run_street(
                street="river",
                n_clusters=n_river_clusters,
                combos=self.combos.river,
                feature_extractor=self._make_river_ehs(),
                free_combos=lambda: setattr(self.combos, "river", None),
            )
            self._persist()
        else:
            get_cluster_id_cache("river", self._config["save_dir"], n_river_rows)

        n_turn_rows = comb(n, 2) * comb(n - 2, 4)
        if "turn" not in self.card_info_lut:
            self.card_info_lut["turn"] = self._run_street(
                street="turn",
                n_clusters=n_turn_clusters,
                combos=self.combos.turn,
                feature_extractor=self._make_turn_ehs(n_river_clusters),
                free_combos=lambda: setattr(self.combos, "turn", None),
            )
            self._persist()
        else:
            get_cluster_id_cache("turn", self._config["save_dir"], n_turn_rows)

        if "flop" not in self.card_info_lut:
            self.card_info_lut["flop"] = self._run_street(
                street="flop",
                n_clusters=n_flop_clusters,
                combos=self.combos.flop,
                feature_extractor=self._make_flop_ehs(n_turn_clusters),
                free_combos=lambda: setattr(self.combos, "flop", None),
            )
            self._persist()

        log.info(f"Finished abstraction build — {time.time() - start:.2f}s total.")
        log.info("Cleaning up intermediate files...")
        self.chunk_store.cleanup_all_intermediate_files()

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    def _run_preflop(self) -> None:
        if "pre_flop" not in self.card_info_lut:
            log.info("Computing pre-flop abstraction...")
            self.card_info_lut["pre_flop"] = (
                compute_preflop_lossless_abstraction(builder=self)
            )
            atomic_joblib_dump(self.card_info_lut, self.card_info_lut_path)

    def _persist(self) -> None:
        atomic_joblib_dump(self.card_info_lut, self.card_info_lut_path)
        atomic_joblib_dump(self.centroids, self.centroid_path)

    def _make_river_ehs(self) -> RiverEHS:
        return RiverEHS(
            evaluator=self._evaluator,
            card_ints=self._card_ints,
            method=self.method,
            n_simulations=self.n_simulations_river,
        )

    def _make_turn_ehs(self, n_river_clusters: int) -> TurnEHS:
        return TurnEHS(
            card_ints=self._card_ints,
            card_to_idx=self._card_to_idx,
            n_cards=self._n_cards,
            save_dir=self._config["save_dir"],
            n_river_clusters=n_river_clusters,
        )

    def _make_flop_ehs(self, n_turn_clusters: int) -> FlopEHS:
        return FlopEHS(
            card_ints=self._card_ints,
            card_to_idx=self._card_to_idx,
            n_cards=self._n_cards,
            save_dir=self._config["save_dir"],
            n_turn_clusters=n_turn_clusters,
        )

    def _run_street(
        self,
        street: str,
        n_clusters: int,
        combos: np.ndarray,
        feature_extractor: Callable[[np.ndarray], np.ndarray],
        free_combos: Callable[[], None],
    ) -> MemmapLookup:
        """Process one street end-to-end; return the consumer-facing lookup.

        Each street moves through: chunk-dispatch → merge → cluster →
        persist → write ``cluster_ids.dat`` → drop combo memory.
        """
        log.info("\n" + "=" * 80)
        log.info(
            f"STAGE {_STAGE_LABELS[street]}: {street.upper()} CLUSTERING",
        )
        log.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(
            f"Clusters: {n_clusters} | Total combos: {len(combos):,}",
        )
        log.info("=" * 80 + "\n")
        start = time.time()

        self.chunk_store.initialize_street(street, len(combos), self._config)

        self._process_incomplete_chunks(street, combos, feature_extractor)

        still_incomplete = self.chunk_store.checkpoint.get_incomplete_chunks(
            street,
        )
        if still_incomplete:
            raise RuntimeError(
                f"{len(still_incomplete)} chunk(s) failed for {street} and "
                f"must be retried before clustering can proceed. "
                f"Re-run the script to retry them automatically."
            )

        _, all_combos, clusters = self._merge_and_cluster(
            street, n_clusters, combos, feature_extractor,
        )

        self._write_cluster_ids(street, clusters)

        log.info(
            f"Finished {street} clusters — {time.time() - start:.2f}s.",
        )

        free_combos()

        ids_path = Path(self._config["save_dir"]) / street / "cluster_ids.dat"
        return MemmapLookup(
            ids_path=ids_path,
            card_to_idx=self._card_to_idx,
            n_cards=self._n_cards,
            n_rows=len(all_combos),
        )

    def _process_incomplete_chunks(
        self,
        street: str,
        combos: np.ndarray,
        feature_extractor: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        incomplete = self.chunk_store.checkpoint.get_incomplete_chunks(street)
        if not incomplete:
            return
        log.info(
            f"Processing {len(incomplete)} incomplete chunks for {street}",
        )
        self.chunk_store.process_chunks_parallel(
            street=street,
            chunk_indices=incomplete,
            all_combos=combos,
            item_processor=feature_extractor,
            workers=self._get_worker_count(),
        )

    def _merge_and_cluster(
        self,
        street: str,
        n_clusters: int,
        combos: np.ndarray,
        feature_extractor: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[np.memmap, np.ndarray, np.ndarray]:
        """Merge chunks and KMeans-cluster them.

        Retries the merge a few times if corrupt chunk files are detected
        mid-merge (e.g. a truncated write from a killed HPC job) — the
        merge raises :class:`CorruptChunkError` after unmarking the bad
        chunks, so a simple re-dispatch recovers.
        """
        for attempt in range(1, _MAX_MERGE_RETRIES + 1):
            try:
                return self._merge_and_cluster_once(
                    street, n_clusters, combos,
                )
            except CorruptChunkError as e:
                if attempt == _MAX_MERGE_RETRIES:
                    raise RuntimeError(
                        f"Still have corrupt chunks for {street} after "
                        f"{_MAX_MERGE_RETRIES} recovery attempts. "
                        f"Last error: {e}"
                    ) from e
                log.warning(
                    f"Attempt {attempt}/{_MAX_MERGE_RETRIES}: {e}. "
                    f"Reprocessing {len(e.corrupt_indices)} corrupt chunk(s)...",
                )
                self.chunk_store.process_chunks_parallel(
                    street=street,
                    chunk_indices=e.corrupt_indices,
                    all_combos=combos,
                    item_processor=feature_extractor,
                    workers=self._get_worker_count(),
                )

    def _merge_and_cluster_once(
        self,
        street: str,
        n_clusters: int,
        combos: np.ndarray,
    ) -> Tuple[np.memmap, np.ndarray, np.ndarray]:
        if self.chunk_store.checkpoint.is_clustering_done(street):
            log.info(f"Loading existing clustering results for {street}")
            self.centroids[street] = self.chunk_store.load_centroids(street)
            clusters = self.chunk_store.load_clusters(street)
            merged_data, all_combos = self.chunk_store.get_or_merge_data(
                street, all_combos_full=combos,
            )
            return merged_data, all_combos, clusters

        merged_data, all_combos = self.chunk_store.get_or_merge_data(
            street, all_combos_full=combos,
        )
        self.centroids[street], clusters = self.clusterer.fit_predict(
            X=merged_data,
            num_clusters=n_clusters,
            street=street,
            work_dir=self.chunk_store.get_street_dir(street),
        )
        self.chunk_store.save_centroids(street, self.centroids[street])
        self.chunk_store.save_clusters(street, clusters)
        self.chunk_store.checkpoint.mark_clustering_done(street)
        log.info(f"Cleaning up intermediate chunk files for {street}...")
        self.chunk_store.cleanup_chunks(street)
        self.chunk_store.cleanup_partial_clustering(street)
        return merged_data, all_combos, clusters

    def _write_cluster_ids(
        self, street: str, clusters: np.ndarray,
    ) -> None:
        """Persist cluster ids as a uint16 memmap and prime the worker cache."""
        save_dir = self._config["save_dir"]
        ids_path = Path(save_dir) / street / "cluster_ids.dat"
        if not ids_path.exists():
            mm = np.memmap(
                ids_path, dtype=np.uint16, mode="w+",
                shape=(len(clusters),),
            )
            mm[:] = clusters.astype(np.uint16)
            mm.flush()
        else:
            mm = np.memmap(
                ids_path, dtype=np.uint16, mode="r",
                shape=(len(clusters),),
            )
        _PROCESS_CACHE["save_dir"] = save_dir
        _PROCESS_CACHE[f"{street}_cluster_ids"] = mm
        log.info(
            f"Saved cluster_ids for {street} "
            f"({len(clusters):,} entries → {ids_path.name})",
        )
