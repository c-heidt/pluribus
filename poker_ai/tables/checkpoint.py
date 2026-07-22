"""Full-sweep checkpointing with atomic writes and auto-resume.

A :class:`CheckpointManager` lives inside the training server and
handles three responsibilities:

1. **Periodic checkpoint writes.**  Invoked from the server loop at
   sync barriers that satisfy the checkpoint schedule.  Each
   checkpoint is a self-contained directory on disk that holds the
   dirty chunk files plus a ``server_state.pkl`` snapshot of the
   server's configuration and iteration counter.
2. **Signal-driven emergency checkpoints.**  ``SIGTERM`` and
   ``SIGINT`` are intercepted in the parent (server) process.  The
   handler merely sets a :class:`threading.Event` so the main loop
   can finish the current iteration cleanly and then write one final
   checkpoint before exiting — critical on SLURM where the job
   scheduler gives only a brief window between ``SIGTERM`` and
   ``SIGKILL``.
3. **Auto-resume on construction.**  If the save directory already
   contains a valid checkpoint the manager loads it back into shared
   memory before workers are spawned, updates the server's start
   iteration, and verifies that structural hyperparameters have not
   changed — a resume with a different action set or sync interval
   would silently corrupt training, so we refuse it.

Checkpoint write flow
---------------------
Split into two phases so workers only pay the sync-barrier cost
and not the full disk-I/O cost on every checkpoint.

*Phase 1 — inline, holds the barrier (RAM-only):*

1. Broadcast a sync job to every worker and wait until all pending
   regret deltas have been merged into the shared tables (skipped
   for emergency checkpoints that cannot afford the wait).
2. Memcpy every dirty chunk into a private buffer via
   :meth:`CFRTables.snapshot_dirty_chunks
   <poker_ai.tables.cfr_tables.CFRTables.snapshot_dirty_chunks>` and
   clear the dirty flags.  Any write after this point will be
   captured by the next snapshot.
3. Flush the per-street LMDB indexes so the on-disk index state is
   consistent with the chunk watermarks.
4. Build the state dict and hand the captured snapshot off to the
   writer thread (via an internal ``queue.Queue``).

*Phase 2 — background writer thread (disk-bound):*

5. Create a temporary directory next to the save root and write
   every buffered chunk as an atomic ``.npy`` file.
6. Hardlink the unchanged chunk files from the previous checkpoint
   into the temp directory so every checkpoint remains
   self-contained with no extra I/O (same-inode hardlinks are
   near-free on every POSIX filesystem).
7. Persist the server state dict.
8. Atomically rename the temp directory to its final name and
   delete the previous checkpoint (we keep exactly one generation
   on disk).

Emergency checkpoints (SIGTERM mid-run) and the final
end-of-training checkpoint drain the writer's queue and run the
write inline so the process cannot exit before the write reaches
disk.

Shutdown window
---------------
When running under SLURM, submit jobs with
``--signal=SIGTERM@300`` so the manager has five minutes between
``SIGTERM`` and ``SIGKILL`` to write the final checkpoint.
"""

import logging
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import joblib
import numpy as np

from poker_ai.tables.cfr_tables import CFRTables
from utils.io import atomic_joblib_dump

if TYPE_CHECKING:
    from poker_ai.blueprint.multiprocess.server import Server

log = logging.getLogger("poker_ai.tables.checkpoint")


@dataclass
class _PendingWrite:
    """Snapshot handed from the main thread to the background writer.

    Captured inside the sync barrier so the arrays are guaranteed to
    be a consistent view of the shared tables.  Owned entirely by the
    writer thread once enqueued — the main thread must not mutate any
    field after :meth:`CheckpointManager._enqueue_snapshot` returns.
    """

    t: int
    label: str
    buffers: List[Tuple[str, np.ndarray]]
    state_dict: dict
    snapshot_ms: float = 0.0
    created_wall_time: int = field(default_factory=lambda: int(time.time()))


def _extract_n_chunks_per_street(state_dict: dict) -> Dict[int, int]:
    """Extract per-street chunk counts from a checkpoint state dict.

    The current format stores a single ``n_chunks_per_street`` dict.
    A legacy format split the count into separate
    ``n_chunks_per_street`` (regret) and
    ``n_strategy_chunks_per_street`` dicts; for backward
    compatibility we take the maximum of the two on each street so
    the restore path never misses a chunk.

    Parameters
    ----------
    state_dict : dict
        Deserialised ``server_state.pkl`` contents.

    Returns
    -------
    dict[int, int]
        Street index → number of chunks expected in the checkpoint.
    """
    ncs = state_dict.get("n_chunks_per_street", {})
    old_strategy = state_dict.get("n_strategy_chunks_per_street", {})
    if old_strategy:
        return {
            r: max(ncs.get(r, 0), old_strategy.get(r, 0))
            for r in range(4)
        }
    if ncs:
        return {int(k): int(v) for k, v in ncs.items()}
    return {r: 0 for r in range(4)}


class CheckpointManager:
    """Periodic and emergency checkpointing for the training server.

    Instantiated in the **parent process before workers are spawned**
    for two reasons: signal handlers must bind to the correct PID
    (the server's, not a worker's), and
    :meth:`_load_checkpoint_if_exists` must run before workers attach
    to the shared-memory tables so they observe the restored state.

    Attributes
    ----------
    sigterm_event : threading.Event
        Set by the ``SIGTERM``/``SIGINT`` handler.  The server loop
        checks this event at every iteration and breaks out cleanly
        when it is set, giving the manager a chance to write one
        final checkpoint before the process exits.
    """

    def __init__(
        self,
        server: "Server",
        save_path: Path,
        lmdb_runtime_dir: Optional[Path] = None,
        lmdb_persistent_dir: Optional[Path] = None,
    ) -> None:
        """Register signal handlers and auto-resume from *save_path* if possible.

        Parameters
        ----------
        server : Server
            Parent training server.  The manager reaches into the
            server for its tables, configuration, and state dict.
        save_path : Path
            Root directory for checkpoint subdirectories.  Created
            implicitly when the first checkpoint is written.
        lmdb_runtime_dir : Path, optional
            Directory where the live LMDB indexes currently reside.
            When this differs from *lmdb_persistent_dir* (e.g. the
            indexes are staged on node-local fast scratch), every
            checkpoint mirrors a consistent snapshot back to
            *lmdb_persistent_dir* on the background writer thread.
        lmdb_persistent_dir : Path, optional
            Persistent destination on shared storage for the LMDB
            mirror.  Defaults to ``save_path/lmdb_index``.
        """
        self._server = server
        self._save_path = Path(save_path)
        self._lmdb_runtime_dir = (
            Path(lmdb_runtime_dir) if lmdb_runtime_dir is not None else None
        )
        self._lmdb_persistent_dir = (
            Path(lmdb_persistent_dir)
            if lmdb_persistent_dir is not None
            else self._save_path / "lmdb_index"
        )
        self._writeback_lmdb = (
            self._lmdb_runtime_dir is not None
            and self._lmdb_runtime_dir.resolve() != self._lmdb_persistent_dir.resolve()
        )
        if self._writeback_lmdb:
            log.info(
                f"CheckpointManager: LMDB writeback enabled "
                f"({self._lmdb_runtime_dir} → {self._lmdb_persistent_dir})"
            )
        self._last_checkpoint_path: Optional[Path] = None
        # Wall-clock seconds used to name the most recent checkpoint dir.
        # Checkpoints are now retained (never deleted), so two writes in the
        # same second would collide on ``checkpoint_<seconds>`` and the rename
        # onto a non-empty dir would fail.  We force the naming counter to
        # strictly increase so every retained generation gets a unique,
        # lexically-ordered name (resume picks the lexically-greatest one).
        self._last_final_seconds: Optional[int] = None
        self._sigterm_event: threading.Event = threading.Event()

        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)
        log.info("CheckpointManager: SIGTERM/SIGINT handlers registered")

        # Background writer infrastructure.  One daemon thread owns
        # all async disk I/O.  ``maxsize=1`` back-pressures the main
        # loop if checkpoints are queued faster than they can be
        # written — a warning is logged but we never drop a snapshot.
        self._write_queue: "queue.Queue[Optional[_PendingWrite]]" = queue.Queue(maxsize=1)
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="CheckpointWriter",
            daemon=True,
        )
        self._writer_thread.start()

        self._load_checkpoint_if_exists(self._save_path)

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    @property
    def sigterm_event(self) -> threading.Event:
        """Event set by the ``SIGTERM``/``SIGINT`` handler."""
        return self._sigterm_event

    def checkpoint(
        self, t: int, emergency: bool = False, wait: bool = False
    ) -> None:
        """Capture a consistent snapshot and hand it to the background writer.

        Phase 1 (inline, holds the sync barrier) is fast: it flushes
        pending worker deltas, memcpies every dirty chunk into a
        private buffer via
        :meth:`CFRTables.snapshot_dirty_chunks`, flushes the per-street
        LMDB indexes so the on-disk state is consistent with the
        snapshotted row watermarks, and builds the state dict.  Phase
        2 (the actual disk writes — ``atomic_numpy_save`` for every
        buffer, hardlink carry-forward, ``server_state.pkl``, atomic
        rename, delete-previous) runs on the background writer
        thread while workers resume training.

        Parameters
        ----------
        t : int
            Current training iteration.  Embedded in the saved state
            dict so the next run can resume at ``t + 1``.
        emergency : bool, optional
            When ``True`` skip the worker-flush step.  Used in the
            error-handling path when the server believes at least one
            worker has crashed — waiting on such a worker would
            deadlock the shutdown.  Implies ``wait=True``.
        wait : bool, optional
            When ``True`` block until the write has actually hit disk
            before returning.  Used for the final end-of-training
            checkpoint so ``terminate`` cannot race the writer.
        """
        label = "emergency" if emergency else "scheduled"

        # Non-blocking guard for scheduled checkpoints.  The background
        # writer owns all disk I/O.  If it has not yet even dequeued the
        # previously-handed-off snapshot, it is behind, and enqueuing
        # another would block the main loop on the ``maxsize=1`` queue —
        # but the main loop is the *only* thread that dispatches CFR
        # jobs, so blocking it idles every worker (the <1% CPU stall).
        # Skip this checkpoint instead and let the next one catch up.
        #
        # We must bail *before* ``snapshot_dirty_chunks`` runs, because
        # that call clears the chunk dirty flags.  Bailing here leaves
        # the flags set, so the chunks dirtied since the last successful
        # checkpoint stay dirty and are captured by the next one —
        # checkpoints coalesce under disk pressure, nothing is dropped.
        if not (emergency or wait) and self._write_queue.full():
            log.warning(
                f"[t={t}] Checkpoint skipped — background writer still busy "
                f"with the previous snapshot (disk slower than the checkpoint "
                f"cadence); dirty state retained for the next checkpoint"
            )
            return

        log.info(f"[t={t}] Checkpoint ({label}) starting")
        snapshot_start = time.monotonic()

        if not emergency:
            self._server.flush_all_workers()

        # Phase 1 — snapshot under the barrier.  Cheap: memcpy + LMDB sync.
        # In deferred-allocation mode, first bulk-write the rows the shm cache
        # has allocated since the last checkpoint into LMDB, so the on-disk index
        # matches the chunk snapshot taken immediately after (both cover
        # [0, occupancy)).  No-op when deferred allocation is off.
        self._server._tables.persist_indexes()
        buffers = self._server._tables.snapshot_dirty_chunks()
        self._server._tables.flush_indexes()
        state_dict = self._server.to_dict(t=t)
        snapshot_ms = (time.monotonic() - snapshot_start) * 1000.0

        pending = _PendingWrite(
            t=t,
            label=label,
            buffers=buffers,
            state_dict=state_dict,
            snapshot_ms=snapshot_ms,
        )

        log.info(
            f"[t={t}] Snapshot captured in {snapshot_ms:.0f} ms "
            f"({len(buffers)} dirty chunks) — handing off to writer"
        )

        if emergency or wait:
            # Drain any previously-queued write first so the previous
            # generation is on disk before we start writing this one.
            self._write_queue.join()
            self._write_snapshot(pending)
        else:
            # The full() guard at the top returned early if a snapshot
            # was still queued, and the main loop is the only producer,
            # so the queue has a free slot and this put() cannot block.
            self._write_queue.put(pending)

    def shutdown(self) -> None:
        """Flush the background writer and stop its thread.

        Called by the training server during orderly shutdown so the
        process does not exit with an unfinished disk write.  Safe to
        call multiple times.
        """
        if not self._writer_thread.is_alive():
            return
        self._write_queue.join()
        self._write_queue.put(None)
        self._writer_thread.join(timeout=30.0)
        if self._writer_thread.is_alive():
            log.warning("CheckpointWriter failed to stop within 30s")

    # -----------------------------------------------------------------------
    # Background writer
    # -----------------------------------------------------------------------

    def _writer_loop(self) -> None:
        """Daemon-thread main loop: drain snapshots, write them to disk.

        Exits when a ``None`` sentinel is received on the queue
        (enqueued by :meth:`shutdown`).  Errors from an individual
        write are logged but do not terminate the loop — the next
        queued snapshot is still attempted.
        """
        while True:
            item = self._write_queue.get()
            try:
                if item is None:
                    return
                self._write_snapshot(item)
            except Exception:
                log.exception(
                    "Background checkpoint write failed — continuing"
                )
            finally:
                self._write_queue.task_done()

    def _write_snapshot(self, pending: _PendingWrite) -> None:
        """Serialise a captured snapshot to disk atomically.

        Runs on the background writer thread for scheduled
        checkpoints and inline for emergency / end-of-training
        writes.  The full on-disk layout of a checkpoint directory is
        built inside a temp directory next to the save root and
        atomically renamed on success; on failure the temp directory
        is removed and the previous checkpoint is left untouched.
        """
        t = pending.t
        write_start = time.monotonic()
        tmp_path = self._save_path / f"checkpoint_tmp_{pending.created_wall_time}"
        tmp_path.mkdir(parents=True, exist_ok=True)

        lmdb_ms = 0.0
        try:
            CFRTables.write_buffers(pending.buffers, tmp_path)

            # Carry forward unchanged chunk files from the previous
            # checkpoint via hardlinks (instant, no I/O — same
            # filesystem).  This keeps every checkpoint directory
            # self-contained even though only dirty chunks are written.
            if self._last_checkpoint_path and self._last_checkpoint_path.exists():
                for old_file in self._last_checkpoint_path.glob("*.npy"):
                    new_file = tmp_path / old_file.name
                    if not new_file.exists():
                        os.link(str(old_file), str(new_file))

            atomic_joblib_dump(pending.state_dict, tmp_path / "server_state.pkl")

            # Mirror the runtime LMDB indexes back to persistent storage
            # BEFORE we publish the new chunks.  Ordering matters: if
            # the process is killed between publishing chunks and
            # mirroring LMDB, the persistent side would end up with
            # newer chunks than LMDB.  On resume, workers would
            # reassign row IDs that the chunks already hold data for,
            # silently mixing the new run's regrets with stale data
            # from whichever infoset previously occupied that row.
            # By mirroring LMDB first, persistent LMDB is always at
            # least as fresh as persistent chunks; the worst case is
            # the benign "LMDB knows about rows whose chunks are
            # zero" pattern (workers re-accumulate regret from zero
            # for those rows on resume).
            if self._writeback_lmdb:
                lmdb_start = time.monotonic()
                self._mirror_lmdb_to_persistent()
                lmdb_ms = (time.monotonic() - lmdb_start) * 1000.0

            seconds = int(time.time())
            if self._last_final_seconds is not None and seconds <= self._last_final_seconds:
                # Same-second (or clock-skew) collision: bump past the last
                # name so retained generations stay unique and monotonic.
                seconds = self._last_final_seconds + 1
            self._last_final_seconds = seconds
            final_path = self._save_path / f"checkpoint_{seconds}"
            tmp_path.rename(final_path)
            tmp_path_to_cleanup = None
        except Exception:
            log.exception("Checkpoint write failed — temp dir preserved for inspection")
            tmp_path_to_cleanup = tmp_path
            raise
        finally:
            if tmp_path_to_cleanup is not None and tmp_path_to_cleanup.exists():
                shutil.rmtree(tmp_path_to_cleanup, ignore_errors=True)

        # Every checkpoint is retained as a training snapshot for offline
        # average-strategy reconstruction — the previous generation is NOT
        # deleted.  ``_last_checkpoint_path`` still tracks the newest one so
        # the hardlink carry-forward above sources unchanged chunks from it
        # (chunks unchanged since the last checkpoint share an inode across
        # retained generations, so the on-disk cost is only the dirty chunks
        # each time).
        self._last_checkpoint_path = final_path

        writeback_ms = (time.monotonic() - write_start) * 1000.0
        log.info(
            f"[t={t}] Checkpoint complete — "
            f"snapshot={pending.snapshot_ms:.0f}ms "
            f"writeback={writeback_ms:.0f}ms "
            f"lmdb={lmdb_ms:.0f}ms "
            f"chunks={len(pending.buffers)} "
            f"path={final_path}"
        )

    def _mirror_lmdb_to_persistent(self) -> None:
        """Snapshot the live LMDB and rsync it to persistent storage.

        Two-step pattern: first :meth:`CFRTables.copy_indexes_to`
        writes a compacted, transactionally consistent image to a
        local temp directory under the runtime LMDB root (fast,
        same-filesystem); then ``rsync`` ships that snapshot to the
        persistent location.  Doing the snapshot locally first means
        ``env.copy`` (which streams the whole DB sequentially) hits
        node-local I/O speeds, and the network transfer can use
        rsync's delta algorithm to send only changed pages between
        checkpoints.

        ``--delete`` is passed to rsync so streets removed from the
        runtime env (shouldn't happen, but safe to defend against)
        don't linger in the persistent mirror.
        """
        snapshot_root = self._lmdb_runtime_dir.parent / (
            self._lmdb_runtime_dir.name + f".snapshot_tmp_{int(time.time())}"
        )
        try:
            self._server._tables.copy_indexes_to(snapshot_root)
            self._lmdb_persistent_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--delete",
                    f"{snapshot_root}/",
                    f"{self._lmdb_persistent_dir}/",
                ],
                check=True,
            )
        finally:
            if snapshot_root.exists():
                shutil.rmtree(snapshot_root, ignore_errors=True)

    # -----------------------------------------------------------------------
    # SIGTERM handler
    # -----------------------------------------------------------------------

    def _handle_sigterm(self, signum: int, frame) -> None:
        """Signal handler: set the sigterm event and return.

        The server's main loop polls :attr:`sigterm_event` at every
        iteration and breaks out cleanly, so the handler itself does
        no I/O.  Running I/O inside a signal handler would be unsafe
        — any lock held by the interrupted code would deadlock on
        re-entry.
        """
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        log.warning(f"{sig_name} received — signalling server loop to checkpoint and exit")
        self._sigterm_event.set()

    # -----------------------------------------------------------------------
    # Resume logic
    # -----------------------------------------------------------------------

    def _load_checkpoint_if_exists(self, save_path: Path) -> None:
        """Find and restore the most recent valid checkpoint, if any.

        Called once at construction time.  Walks the checkpoint
        directories in reverse time order so that a partial write
        (e.g. from a crash mid-checkpoint) is skipped in favour of
        the previous, known-good checkpoint.
        """
        checkpoints = sorted(save_path.glob("checkpoint_[0-9]*"))
        for cp in reversed(checkpoints):
            if self._checkpoint_is_valid(cp):
                log.info(f"Valid checkpoint found: {cp} — restoring")
                self._restore_from_checkpoint(cp)
                self._last_checkpoint_path = cp
                return
        log.info("No valid checkpoint found — starting fresh")

    def _checkpoint_is_valid(self, path: Path) -> bool:
        """Return ``True`` iff *path* holds a structurally complete checkpoint.

        A checkpoint is valid when the ``server_state.pkl`` file
        exists, the shared LMDB index directory exists, the state
        dict is loadable, and every chunk file it lists is present
        on disk.  We do not verify content checksums — a partial
        write would have left the temp directory under a different
        name and been cleaned up.
        """
        if not (path / "server_state.pkl").exists():
            return False
        if not (self._save_path / "lmdb_index").exists():
            return False
        try:
            state_dict = joblib.load(path / "server_state.pkl")
        except Exception:
            log.warning(f"Could not load server_state.pkl from {path}")
            return False
        ncs = _extract_n_chunks_per_street(state_dict)
        return self._server._tables.validate_chunks(path, ncs)

    _STRUCTURAL_KEYS = (
        "n_players",
        "sync_interval",
        "discount_interval",
        "discount_duration_cycles",
        "strategy_interval",
        "update_threshold",
        "prune_threshold",
        "c",
        "chunk_size",
        "info_set_encoding",
    )
    """Hyperparameters whose values must match between the saved state
    and the current :class:`Server` for a resume to be safe.  Changing
    any of these would silently corrupt training, so we refuse the
    resume and ask the user to either revert the change or start a
    fresh run.  Operational parameters (runtime budget, checkpoint
    interval, worker count, nickname) are intentionally *not*
    included — they are allowed to differ between runs.
    """

    def _validate_config_compatibility(self, state_dict: dict) -> None:
        """Raise if any structural hyperparameter has changed since the save.

        Parameters
        ----------
        state_dict : dict
            Deserialised ``server_state.pkl`` contents from the
            candidate checkpoint.

        Raises
        ------
        RuntimeError
            If the saved value of any key in :attr:`_STRUCTURAL_KEYS`
            differs from the current server's value.
        """
        def _current(key: str):
            # discount_duration_cycles lives on Server._discount_state;
            # chunk_size is a module-level constant in chunk_store; the
            # rest are plain Server attributes.
            if key == "discount_duration_cycles":
                return self._server._discount_state.duration_cycles
            if key == "chunk_size":
                from poker_ai.tables.chunk_store import CHUNK_SIZE
                return CHUNK_SIZE
            if key == "info_set_encoding":
                from environment.poker_env import INFO_SET_ENCODING
                return INFO_SET_ENCODING
            return getattr(self._server, "_" + key)

        mismatches = []
        for key in self._STRUCTURAL_KEYS:
            if key not in state_dict:
                continue  # old checkpoint format — skip silently
            saved = state_dict[key]
            current = _current(key)
            if saved != current:
                mismatches.append(f"{key}: saved={saved!r} current={current!r}")
        if mismatches:
            details = "\n  - ".join(mismatches)
            raise RuntimeError(
                "Cannot resume: structural config has changed since the "
                f"checkpoint was written:\n  - {details}\n"
                "Either revert these options or start a fresh run in a new "
                "save_path."
            )

    def _restore_from_checkpoint(self, path: Path) -> None:
        """Reload shared-memory tables and server state from *path*.

        Runs before workers are spawned so the restored state is
        visible to every worker as soon as it starts.  Advances
        ``server._start_t`` past the checkpointed iteration so the
        resumed run does not replay work.
        """
        state_dict = joblib.load(path / "server_state.pkl")

        self._validate_config_compatibility(state_dict)

        self._server._start_t = state_dict["t"]
        self._server._discount_state.active = state_dict.get("discount_active", True)

        ncs = _extract_n_chunks_per_street(state_dict)
        self._server._tables.restore_chunks(path, ncs)

        log.info(
            f"Restored from checkpoint — resuming at t={self._server._start_t} "
            f"(checkpoint was t={state_dict['t']})"
        )
