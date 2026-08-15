"""Worker process for multi-process CFR training.

Each :class:`Worker` is a long-lived
:class:`multiprocessing.Process` that consumes jobs from a queue
dispatched by the :class:`~poker_ai.blueprint.multiprocess.server.Server`.
All per-traversal training logic lives in
:mod:`poker_ai.blueprint.training`; this class is a thin dispatch loop
around those primitives.

Job protocol
------------
The worker understands three job names:

- ``"cfr"`` — run ``kwargs["batch"]`` CFR traversals for player
  ``kwargs["i"]`` at iteration ``kwargs["t"]`` (defaults to ``1``
  for backward compatibility with callers that do not batch).  Regret
  updates are written to the worker's persistent :attr:`_local_delta`
  buffer across the full batch.  The same job then folds in
  ``kwargs["strat_batch"]`` (default ``0``) average-strategy
  playthroughs for the same player, accumulating visit counts into
  :attr:`_local_strategy_delta` — so the strategy pass rides along with
  CFR instead of running as its own sync barrier.
- ``"sync"`` — flush both :attr:`_local_delta` (regret) and
  :attr:`_local_strategy_delta` (average strategy) into the shared
  tables via :meth:`_flush_delta`.
- ``"terminate"`` — flush any remaining deltas, then exit the dispatch
  loop.  Sent by :meth:`Server.terminate` during shutdown.

Anything the server dispatches that is not one of these names raises
:class:`ValueError`, which is caught as a fatal exception and
surfaces as a :class:`WorkerError` on the server side.

Persistent local state
----------------------
The worker keeps two accumulator dicts for the lifetime of the process:
:attr:`_local_delta` (regret) and :attr:`_local_strategy_delta` (average
strategy).  CFR jobs accumulate into both and sync/terminate jobs flush
both.  Batching many traversals into one merge is the whole point of the
sync-barrier architecture — per-traversal merges would serialise every
worker behind the stripe locks of the shared tables.
"""

import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from poker_ai.blueprint.bias import BiasClass
from poker_ai.blueprint.cfr import merge_local_delta, merge_local_strategy_delta
from poker_ai.blueprint.core_runner import CoreDriver, core_enabled
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.blueprint.training import (
    cfr_step,
    pin_blas_threads,
    seed,
    strategy_step,
)
from information_abstraction import load_info_set_lut
from environment import poker_env as state

log = logging.getLogger("sync.worker")


class Worker(mp.Process):
    """Long-lived worker process running CFR jobs dispatched by the server.

    Workers are constructed in the parent process and started with
    :meth:`multiprocessing.Process.start`, which forks a new
    interpreter that inherits the shared tables, queues, and locks
    via copy-on-write.  :meth:`run` is the entry point of the child
    process.
    """

    def __init__(
        self,
        job_queue: mp.Queue,
        logging_queue: mp.Queue,
        locks: Dict[str, mp.synchronize.Lock],
        tables: CFRTables,
        lut_path: Union[str, Path],
        pickle_dir: bool,
        n_players: int,
        prune_threshold: int,
        c: int,
        save_path: Path,
        info_set_lut=None,
        error_event: Optional[mp.Event] = None,  # type: ignore
        bias: BiasClass = "none",
        bias_magnitude: float = 0.0,
    ):
        """Initialise the worker's fields in the parent process.

        All heavy post-fork work — LMDB reopen, LUT loading, RNG
        seeding — is deferred to :meth:`run`, which executes in the
        child process.

        Parameters
        ----------
        job_queue : multiprocessing.Queue
            Queue from which this worker consumes jobs.  Shared with
            every other worker in the pool and with the server.
        logging_queue : multiprocessing.Queue
            Queue on which this worker pushes status strings.  Drained
            by the server's main loop.
        locks : dict[str, multiprocessing.Lock]
            Named locks reserved for future cross-worker coordination.
            Currently unused by the CFR code path itself; retained
            so the construction signature stays stable.
        tables : CFRTables
            Shared regret and strategy tables.
        lut_path : str or Path
            Directory containing the card-info LUT.
        pickle_dir : bool
            Use the legacy pickle-directory LUT layout.
        n_players : int
            Number of players in the game.
        prune_threshold : int
            Raw iteration at which CFR-P becomes eligible.
        c : int
            CFR-P regret threshold.
        save_path : Path
            Root save directory (retained for symmetry with the
            server; unused by the worker itself).
        info_set_lut : object, optional
            Pre-loaded LUT object inherited via fork copy-on-write.
            When ``None`` the worker loads it from disk in
            :meth:`run`.
        error_event : multiprocessing.Event, optional
            Shared event the worker sets on unhandled exceptions so
            the server's :meth:`_join_queue` can detect the failure
            without blocking on the dead worker.
        """
        super().__init__(group=None, name=None, args=(), kwargs={}, daemon=None)
        self._job_queue: mp.Queue = job_queue
        self._logging_queue: mp.Queue = logging_queue
        self._locks = locks
        self._n_players = n_players
        self._tables = tables
        self._c = c
        self._prune_threshold = prune_threshold
        self._save_path = Path(save_path)
        self._lut_path = str(lut_path)
        self._pickle_dir = pickle_dir
        self._error_event: Optional[mp.Event] = error_event  # type: ignore
        self._bias = bias
        self._bias_magnitude = bias_magnitude
        if info_set_lut is not None:
            self._info_set_lut = info_set_lut
        # Persistent regret accumulator keyed by (betting_round,
        # info_set).  Batched across many CFR calls and flushed on
        # explicit "sync" jobs dispatched by the server.
        self._local_delta: Dict[Tuple[int, str], np.ndarray] = {}
        # Persistent average-strategy visit-count accumulator, symmetric with
        # ``_local_delta``.  CFR jobs fold ``strat_batch`` strategy playthroughs
        # into it (see the "cfr" branch); it is flushed on the same "sync" jobs,
        # so the strategy pass overlaps CFR instead of stalling at its own
        # barrier.
        self._local_strategy_delta: Dict[Tuple[int, str], np.ndarray] = {}

    def run(self):
        """Child-process entry point: set up state, then process jobs.

        Post-fork setup:

        1. Reopen every LMDB index so this process has its own reader
           lock-table slots — reusing the inherited handles would
           produce ``MDB_BAD_RSLOT`` errors on the first transaction.
        2. Attach the card-info LUT.  If the object was inherited via
           fork copy-on-write in :meth:`__init__` it is already
           present; otherwise load it from disk.
        3. Seed the RNG with a process-unique stream so sampled
           actions differ between workers.

        Dispatch loop:

        The worker blocks on :meth:`Queue.get` and dispatches each
        incoming job.  Exactly one ``task_done`` is emitted per
        message via the ``finally`` clause so the server's joinable
        queue can never leak a pending message, even if the job body
        raises or the worker is told to terminate.  Any unhandled
        exception sets the shared error event and causes the worker
        to break out of the loop — the server will notice the event
        from :meth:`Server._join_queue` and raise
        :class:`WorkerError`.
        """
        # Pin BLAS/OpenMP threads before any other post-fork setup — every
        # subsequent numpy-touching step (LUT load, CoreDriver, every CFR
        # traversal) must run under the pin, not just the traversal loop.
        pin_blas_threads(1)
        # Reopen LMDB indexes so this process has its own reader
        # lock-table slots; reusing the inherited handles triggers
        # MDB_BAD_RSLOT on the first transaction.
        self._tables.reopen_after_fork()
        if not hasattr(self, "_info_set_lut"):
            self._info_set_lut = load_info_set_lut(self._lut_path, self._pickle_dir)
        self._set_seed()
        # Build the compiled-core driver (PLURIBUS_CFR_CORE=1) *after* the fork
        # + LMDB reopen so its CoreTables holds this process's own shm cache
        # arrays and regret chunk mmaps.  The parent prewarmed the caches
        # before forking, so the pure-shm read path already sees a complete
        # mirror; the driver's own guardrail re-checks that per street.  Stays
        # None (Python path) when unset or biased.
        self._core = CoreDriver(self._tables) if core_enabled(self._bias) else None

        while True:
            name, kwargs = self._job_queue.get(block=True)
            should_break = False
            try:
                if name == "terminate":
                    self._flush_delta()
                    should_break = True
                elif name == "cfr":
                    # A single "cfr" queue item runs ``batch`` traversals
                    # back-to-back so the queue IPC cost (pickle +
                    # cross-process put/get) is amortised over many
                    # CFR calls.  Each traversal starts from a fresh
                    # game state; regret updates accumulate into the
                    # persistent :attr:`_local_delta` across the batch.
                    batch = kwargs.get("batch", 1)
                    player_i = kwargs["i"]
                    for _ in range(batch):
                        game_state = state.new_game(
                            self._n_players, self._info_set_lut,
                        )
                        cfr_step(
                            self._tables,
                            game_state,
                            player_i,
                            kwargs["t"],
                            self._prune_threshold,
                            self._c,
                            self._local_delta,
                            bias=self._bias,
                            bias_magnitude=self._bias_magnitude,
                            core=self._core,
                        )
                    # Fold the average-strategy pass into the same job: run
                    # ``strat_batch`` strategy playthroughs for the same player,
                    # accumulating visit counts into the persistent
                    # :attr:`_local_strategy_delta` (flushed with the regret
                    # delta at the next sync).  The server gates ``strat_batch``
                    # (0 during warm-up), so the strategy pass no longer needs a
                    # separate barrier — it overlaps CFR across the whole pool.
                    strat_batch = kwargs.get("strat_batch", 0)
                    for _ in range(strat_batch):
                        game_state = state.new_game(
                            self._n_players, self._info_set_lut,
                        )
                        strategy_step(
                            self._tables,
                            game_state,
                            player_i,
                            local_delta=self._local_strategy_delta,
                            core=self._core,
                        )
                elif name == "sync":
                    self._flush_delta()
                else:
                    raise ValueError(f"Unrecognised job name: {name}")
            except Exception:
                log.exception(
                    f"[worker={self.name}] Unhandled exception in job '{name}' — "
                    f"signaling shutdown"
                )
                if self._error_event is not None:
                    self._error_event.set()
                should_break = True
            finally:
                # Exactly one task_done per message, regardless of which
                # branch ran or whether an exception fired.
                self._job_queue.task_done()
            if should_break:
                break

    def _set_seed(self):
        """Seed the RNG from :func:`os.urandom` for process-independence.

        Each worker needs an independent random stream or they would
        draw identical pruning coins and sampled opponent actions,
        reducing the variance-reduction benefit of running many
        workers in parallel.  NumPy's default seeding strategy is
        known to reuse seeds across fork (see
        https://github.com/numpy/numpy/issues/9650), so we seed
        explicitly from :func:`os.urandom`.
        """
        random_seed: int = int.from_bytes(os.urandom(4), byteorder="little")
        seed(random_seed)

    def _flush_delta(self) -> None:
        """Flush the regret and average-strategy accumulators into the tables.

        Routes each ``(betting_round, info_set)`` regret delta into the
        corresponding per-street regret table via
        :func:`poker_ai.blueprint.cfr.merge_local_delta`, then flushes the
        folded strategy visit-count accumulator via
        :func:`poker_ai.blueprint.cfr.merge_local_strategy_delta` into
        ``tables.strategy`` — both use the same stripe-locked
        ``merge_delta_rows``, so a single sync barrier durably lands both.  Each
        accumulator is cleared after its flush.

        No-op for whichever accumulator is empty — the common case for
        ``terminate`` jobs that arrive right after a sync barrier.  The regret
        flush count is emitted at ``DEBUG`` level; at ``INFO`` or above only the
        server's periodic progress line is visible, keeping HPC cluster logs
        readable.
        """
        if self._local_delta:
            n_infosets = len(self._local_delta)
            merge_local_delta(self._tables, self._local_delta)
            self._local_delta.clear()
            log.debug(
                f"[worker={self.name}] Flushed {n_infosets:,} infosets to shared tables"
            )
        if self._local_strategy_delta:
            merge_local_strategy_delta(self._tables, self._local_strategy_delta)
            self._local_strategy_delta.clear()
