import logging
import mmap as _mmap
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Union

import joblib
import numpy as np

from poker_ai.ai import ai
from poker_ai.ai.agent import Agent
from poker_ai import utils
from poker_ai.games.short_deck import state

log = logging.getLogger("sync.worker")


class Worker(mp.Process):
    """Subclass of multiprocessing Process to handle agent optimisation."""

    def __init__(
        self,
        job_queue: mp.Queue,
        status_queue: mp.Queue,
        logging_queue: mp.Queue,
        locks: Dict[str, mp.synchronize.Lock],
        agent: Agent,
        lut_path: Union[str, Path],
        pickle_dir: bool,
        n_players: int,
        prune_threshold: int,
        c: int,
        discount_interval: int,
        save_path: Path,
        info_set_lut=None,
    ):
        """Construct the process, setup the state."""
        super().__init__(group=None, name=None, args=(), kwargs={}, daemon=None)
        self._job_queue: mp.Queue = job_queue
        self._status_queue: mp.Queue = status_queue
        self._logging_queue: mp.Queue = logging_queue
        self._locks = locks
        self._n_players = n_players
        self._prune_threshold = prune_threshold
        self._agent = agent
        self._c = c
        self._discount_interval = discount_interval
        self._save_path = Path(save_path)
        self._lut_path = str(lut_path)
        self._pickle_dir = pickle_dir
        # If the caller provides the pre-loaded LUT (e.g. the Server, which
        # already loaded it in __init__), store it here so that run() can skip
        # the expensive ~24s disk reload after fork.  When None, run() falls
        # back to loading from disk via the mmap path.
        self._info_set_lut = info_set_lut
        # Per-traversal regret accumulator.  Populated by cfr()/cfrp() and
        # flushed into agent.regret under lock by _sync_to_master().
        self._local_delta: Dict[str, Dict[str, float]] = {}
        # Total cfr traversals completed by this worker.
        self._local_iteration_count: int = 0

    def run(self):
        """Load the LUT, seed RNG, then process jobs dispatched by the server.

        The card info LUT is loaded here — after fork — so that the parent
        process does not hold the large dict and workers share the underlying
        OS file-cache pages via a read-only mmap (MAP_SHARED / ACCESS_READ).
        """
        # 1. Load the card info LUT.
        #    Fast path: the Server passes its already-loaded LUT via
        #    info_set_lut= so workers skip reloading 100+ MB from disk.
        #    Fallback: when no pre-loaded LUT was provided (e.g. in tests),
        #    load from disk after fork using a read-only shared mmap so that
        #    all workers can share the underlying OS file-cache pages.
        if self._info_set_lut is None:
            if not self._pickle_dir:
                lut_file_path = os.path.join(self._lut_path, "card_info_lut.joblib")
                lut_file = open(lut_file_path, "rb")
                lut_mmap = _mmap.mmap(lut_file.fileno(), 0, access=_mmap.ACCESS_READ)
                # mmap.madvise(mmap.MADV_RANDOM) is only available from Python 3.8;
                # on Python 3.7 this hint is silently skipped.
                self._info_set_lut = joblib.load(lut_mmap)
                lut_mmap.close()
                lut_file.close()
            else:
                self._info_set_lut = utils.io.load_info_set_lut(
                    self._lut_path, self._pickle_dir
                )
        # 2. Pin this worker to a NUMA node (best-effort; no-op if NUMA is
        #    unavailable or the OS does not support sched_setaffinity).
        self._try_numa_pin()
        # 3. Seed the process-local RNG before any game setup so that card
        #    dealing in _setup_new_game() draws from the seeded stream.
        self._set_seed()
        # 4. Deal the first hand now that the LUT and seed are ready.
        self._setup_new_game()
        # 5. Main dispatch loop — block on the job queue and execute each job.
        while True:
            # Get the name of the method and the key word arguments needed for
            # the method.
            self._update_status("idle")
            name, kwargs = self._job_queue.get(block=True)
            if name == "terminate":
                # Flush any remaining local delta before exiting so that no
                # regret increments are lost when the server shuts down.
                self._sync_to_master()
                self._job_queue.task_done()
                break
            elif name == "cfr":
                function = self._cfr
            elif name == "sync":
                function = self._sync_to_master
            elif name == "discount":
                function = self._discount
            elif name == "update_strategy":
                function = self._update_strategy
            elif name == "serialise":
                function = self._serialise
            else:
                raise ValueError(f"Unrecognised function name: {name}")
            self._update_status(name)
            function(**kwargs)
            # Notify the job queue that the task is done.
            self._job_queue.task_done()

    def _set_seed(self):
        """Lose all reproducability as we need unique streams per worker."""
        # NOTE(fedden): NumPy in particular has a problem with processes and
        #               seeds: https://github.com/numpy/numpy/issues/9650
        random_seed: int = int.from_bytes(os.urandom(4), byteorder="little")
        utils.random.seed(random_seed)

    def _sync_to_master(self) -> None:
        """Flush ``_local_delta`` into the shared ``agent.regret`` table.

        Acquires the regret lock once for the entire batch rather than once per
        infoset.  After the flush ``_local_delta`` is cleared.

        This method is safe to call with an empty delta — it returns
        immediately without acquiring the lock.

        Note: once ``Agent`` is refactored to use ``SparseRegretTable``
        (Phase 5.1) this method will be updated to use stripe locks and
        ``np.add`` instead of the dict-based merge.
        """
        if not self._local_delta:
            return
        n_infosets = len(self._local_delta)
        self._locks["regret"].acquire()
        ai.merge_local_delta(self._agent, self._local_delta)
        self._locks["regret"].release()
        self._local_delta.clear()
        self._logging_queue.put(
            f"[worker={self.name}] Synced {n_infosets:,} infosets to master",
            block=True,
        )

    def _cfr(self, t, i):
        """Search over random game and calculate the strategy."""
        self._setup_new_game()
        use_pruning: bool = np.random.uniform() < 0.95
        pruning_allowed: bool = t > self._prune_threshold
        if pruning_allowed and use_pruning:
            ai.cfrp(self._agent, self._state, i, t, self._c, self._local_delta)
        else:
            ai.cfr(self._agent, self._state, i, t, self._local_delta)
        self._local_iteration_count += 1
        # Flush immediately so that regret updates are visible to other workers
        # on the next iteration.  Phase 5 will decouple flush frequency from
        # traversal frequency via explicit server-dispatched "sync" jobs.
        self._sync_to_master()

    def _discount(self, t):
        """Discount previous regrets and strategy."""
        # TODO(fedden): Is discount_interval actually set/managed in
        #               minutes here? In Algorithm 1 this should be managed
        #               in minutes using perhaps the time module, but here
        #               it appears to be being managed by the iterations
        #               count.
        discount_factor = (t / self._discount_interval) / (
            (t / self._discount_interval) + 1
        )
        self._logging_queue.put(
            f"[t={t}] Discounting regrets and strategy (factor={discount_factor:.4f})",
            block=True,
        )
        # Per Pluribus paper only regret is discounted (Bug 2 fix).
        # Discounting strategy degrades convergence and is incorrect.
        #
        # IMPORTANT: agent.regret is a manager.dict() proxy.  Accessing
        # agent.regret[info_set] returns a *copy* of the nested dict, not a
        # proxy into it.  Mutating that copy in-place (e.g.
        # regret[info_set][action] *= d) is a silent lost update — the change
        # is never written back to the shared manager process.  We must
        # explicitly read the row, mutate a local copy, then write it back.
        self._locks["regret"].acquire()
        for info_set in list(self._agent.regret.keys()):
            row = dict(self._agent.regret[info_set])
            for action in row:
                row[action] *= discount_factor
            self._agent.regret[info_set] = row
        self._locks["regret"].release()

    def _update_strategy(self, t, i):
        """Update the strategy."""
        # Deal a fresh hand so each strategy update traverses a different game
        # path, giving more diverse strategy samples.
        self._setup_new_game()
        # Acquire the pre_flop_strategy lock for the full traversal so that
        # concurrent workers performing strategy updates don't race on the
        # same infoset's read-modify-write.  The lock is *not* acquired inside
        # ai.update_strategy() (that function is recursive and must not try to
        # re-acquire a non-reentrant lock); it is acquired here at the single
        # entry point for the whole traversal.
        self._locks["pre_flop_strategy"].acquire()
        ai.update_strategy(self._agent, self._state, i, t)
        self._locks["pre_flop_strategy"].release()

    def _serialise(self, t: int, server_state: Dict[str, Union[str, float, int, None]]):
        """Write progress of optimising agent (and server state) to file."""
        self._logging_queue.put(
            f"[t={t}] Saving checkpoint to {self._save_path}", block=True
        )
        ai.serialise(
            agent=self._agent,
            save_path=self._save_path,
            t=t,
            server_state=server_state,
            locks=self._locks,
        )
        n_info_sets = len(self._agent.regret)
        self._logging_queue.put(
            f"[t={t}] Checkpoint saved — {n_info_sets:,} info sets in regret table",
            block=True,
        )

    def _update_status(self, status):
        """Update the status of this worker by posting it to the server."""
        self._status_queue.put((self.name, status), block=True)

    def _try_numa_pin(self) -> None:
        """Pin this worker to a NUMA node (best-effort).

        Reads NUMA topology from ``/sys/devices/system/node/`` and uses
        ``os.sched_setaffinity`` to restrict this process to the cores of one
        node.  Workers are assigned to nodes in round-robin order derived from
        the process name (e.g. ``Process-3`` → worker index 2).

        Silently does nothing if NUMA information is unavailable, if
        ``os.sched_setaffinity`` is not supported, or if the process name
        cannot be parsed.
        """
        nodes = self._get_numa_nodes()
        if not nodes:
            return
        try:
            worker_idx = int(self.name.split("-")[-1]) - 1
        except (ValueError, IndexError):
            return
        node = nodes[worker_idx % len(nodes)]
        cores = self._get_cores_for_numa_node(node)
        if not cores:
            return
        try:
            os.sched_setaffinity(0, cores)
            log.debug(
                f"Worker {self.name} pinned to NUMA node {node} "
                f"({len(cores)} cores)"
            )
        except OSError:
            pass

    def _get_numa_nodes(self):
        """Return sorted list of NUMA node IDs from /sys/devices/system/node/."""
        node_dir = "/sys/devices/system/node"
        if not os.path.isdir(node_dir):
            return []
        nodes = []
        for entry in os.listdir(node_dir):
            if entry.startswith("node") and entry[4:].isdigit():
                nodes.append(int(entry[4:]))
        return sorted(nodes)

    def _get_cores_for_numa_node(self, node: int):
        """Return CPU core IDs for *node* by reading the kernel cpulist file."""
        cpulist_path = f"/sys/devices/system/node/node{node}/cpulist"
        try:
            with open(cpulist_path) as f:
                return self._parse_cpulist(f.read().strip())
        except OSError:
            return []

    @staticmethod
    def _parse_cpulist(cpulist: str):
        """Parse a Linux cpulist string (e.g. ``'0-3,8-11'``) into a list of ints."""
        cores = []
        for part in cpulist.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                cores.extend(range(int(start), int(end) + 1))
            elif part:
                cores.append(int(part))
        return cores

    def _setup_new_game(self):
        """Setup up new poker game."""
        self._state: state.ShortDeckPokerState = state.new_game(
            self._n_players, self._info_set_lut,
        )
