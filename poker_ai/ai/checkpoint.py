"""CheckpointManager — incremental write-on-dirty checkpointing.

Phase 6 of the training pipeline refactor.

Architecture
------------
Signal handling
    SIGTERM and SIGINT are intercepted in the *parent (server) process* before
    any workers are spawned.  The handler only sets a ``threading.Event``; all
    actual I/O happens from the server's main thread after the loop notices the
    event and breaks cleanly.  This prevents dead-locks that would occur if
    complex work were done inside a signal handler.

Checkpoint write flow (Phase 6.2)
    1. Flush all workers (broadcast sync + barrier join).
    2. Write dirty chunks to a temporary directory on the same filesystem as
       ``save_path`` so that the final ``rename`` is atomic (POSIX guarantee).
    3. Flush the LMDB index to disk.
    4. Persist the server state dict (includes current iteration ``t`` and the
       chunk counts needed for resume validation).
    5. Atomically rename the temp directory to the final checkpoint directory.
    6. Remove the previous checkpoint (only one generation is kept).
    7. Clear per-table dirty flags.

Resume flow (Phase 6.5)
    On construction, the most recent valid checkpoint is found and restored
    before any workers are spawned.  ``server._start_t`` is advanced past the
    last checkpointed iteration so training resumes without replaying work.

Shutdown window
    SLURM jobs should be submitted with ``--signal=SIGTERM@300`` to give five
    minutes for the checkpoint to complete.  The signal handler itself is
    instant (sets an event); all the time belongs to the flush + write.
"""

import logging
import shutil
import signal
import threading
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import joblib
import numpy as np

from poker_ai.ai.index import CHUNK_SIZE
from poker_ai.utils.io import atomic_joblib_dump, atomic_numpy_save

if TYPE_CHECKING:
    from poker_ai.ai.multiprocess.server import Server

log = logging.getLogger("poker_ai.ai.checkpoint")


class CheckpointManager:
    """Manages periodic and emergency checkpointing for the training server.

    Must be instantiated in the **parent process before workers are spawned**
    so that SIGTERM/SIGINT handlers are registered with the correct PID and
    that ``_load_checkpoint_if_exists`` runs before any shared-memory tables
    are populated by workers.

    Parameters
    ----------
    server:
        The ``Server`` instance that owns the agent and job queue.
    save_path:
        Root directory for checkpoints.  Each checkpoint is written to a
        timestamped subdirectory (``checkpoint_<unix_ts>``).
    """

    def __init__(self, server: "Server", save_path: Path) -> None:
        self._server = server
        self._save_path = Path(save_path)
        self._last_checkpoint_path: Optional[Path] = None
        self._sigterm_event: threading.Event = threading.Event()

        # Register signal handlers before workers are spawned.  Workers
        # inherit the parent's signal disposition via fork; we want only the
        # parent to react so workers can die cleanly without trying to
        # checkpoint themselves.
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)
        log.info("CheckpointManager: SIGTERM/SIGINT handlers registered")

        # Restore state from the most recent valid checkpoint (if any).
        self._load_checkpoint_if_exists(self._save_path)

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    @property
    def sigterm_event(self) -> threading.Event:
        """Event set by the SIGTERM/SIGINT handler.

        The server loop checks this after every ``job_queue.join()`` and breaks
        to trigger an emergency checkpoint from the main thread.
        """
        return self._sigterm_event

    def checkpoint(self, t: int, emergency: bool = False) -> None:
        """Write a full incremental checkpoint at iteration *t*.

        Parameters
        ----------
        t:
            Current training iteration.  Stored in ``server_state.pkl`` so
            that resumed runs start from ``t + 1``.
        emergency:
            When ``True`` skip the worker-flush step.  Use only when workers
            are known to be idle (e.g. the server loop has already broken out
            and no new jobs are being dispatched).  Normal scheduled
            checkpoints and SIGTERM-triggered checkpoints should leave this
            ``False`` so all in-flight local deltas are incorporated before
            the snapshot is taken.
        """
        label = "emergency" if emergency else "scheduled"
        log.info(f"[t={t}] Checkpoint ({label}) starting")
        t_start = time.monotonic()

        # Step 1: Flush all workers so no in-flight local deltas remain.
        if not emergency:
            self._server.flush_all_workers()

        # Step 2: Write to a temp directory on the same filesystem so that the
        # final rename is atomic (POSIX: rename(2) is atomic on same device).
        tmp_path = self._save_path / f"checkpoint_tmp_{int(time.time())}"
        tmp_path.mkdir(parents=True, exist_ok=True)

        chunks_written = 0
        chunks_skipped = 0

        try:
            # Step 3: Write dirty chunks.
            for r in range(4):
                for table, prefix in [
                    (self._server._agent.regret_tables[r],   f"regret_{r}"),
                    (self._server._agent.strategy_tables[r], f"strategy_{r}"),
                ]:
                    for chunk_id in table.get_dirty_chunks():
                        # Attach the server process to this chunk if it has
                        # not opened it yet (workers create chunks; the server
                        # never touches them on the hot path).
                        table._ensure_chunk(chunk_id)
                        # Use per-table n_allocated, not _index.n_entries which
                        # is shared across all eight tables and gives wrong counts.
                        valid_rows = min(
                            table.n_allocated - chunk_id * CHUNK_SIZE,
                            CHUNK_SIZE,
                        )
                        if valid_rows <= 0:
                            chunks_skipped += 1
                            continue
                        # .copy() so the snapshot is not affected by concurrent
                        # worker writes after we release the stripe lock.
                        atomic_numpy_save(
                            table._chunks[chunk_id][:valid_rows].copy(),
                            tmp_path / f"{prefix}_chunk_{chunk_id:06d}.npy",
                        )
                        chunks_written += 1

            # Step 4: Flush the LMDB index.
            self._server._agent._index.flush()

            # Step 5: Write the server state.
            state_dict = self._server.to_dict(t=t)
            atomic_joblib_dump(state_dict, tmp_path / "server_state.pkl")

            # Step 6: Atomic rename.
            final_path = self._save_path / f"checkpoint_{int(time.time())}"
            tmp_path.rename(final_path)
            tmp_path = None  # sentinel — rename succeeded; don't clean up

        except Exception:
            log.exception("Checkpoint write failed — temp dir preserved for inspection")
            if tmp_path is not None and tmp_path.exists():
                shutil.rmtree(tmp_path, ignore_errors=True)
            raise

        # Step 7: Delete the previous checkpoint (keep only one generation).
        if self._last_checkpoint_path and self._last_checkpoint_path.exists():
            shutil.rmtree(self._last_checkpoint_path)
            log.info(f"Deleted previous checkpoint: {self._last_checkpoint_path}")
        self._last_checkpoint_path = final_path

        # Step 8: Clear dirty flags so the next incremental write only covers
        # chunks that were modified after this checkpoint.
        for r in range(4):
            self._server._agent.regret_tables[r].clear_all_dirty()
            self._server._agent.strategy_tables[r].clear_all_dirty()

        elapsed = time.monotonic() - t_start
        log.info(
            f"[t={t}] Checkpoint complete in {elapsed:.1f}s — "
            f"{chunks_written} chunks written, {chunks_skipped} skipped. "
            f"Path: {final_path}"
        )

    # -----------------------------------------------------------------------
    # SIGTERM handler (Phase 6.3)
    # -----------------------------------------------------------------------

    def _handle_sigterm(self, signum: int, frame) -> None:
        """Set the SIGTERM event so the server loop can exit and checkpoint.

        This handler does *nothing* except set an event.  All I/O is deferred
        to the main thread.  This avoids re-entrancy issues, and means the
        full five-minute SLURM shutdown window (``--signal=SIGTERM@300``) is
        available for flushing workers and writing the checkpoint.
        """
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        log.warning(f"{sig_name} received — signalling server loop to checkpoint and exit")
        self._sigterm_event.set()

    # -----------------------------------------------------------------------
    # Resume logic (Phase 6.5)
    # -----------------------------------------------------------------------

    def _load_checkpoint_if_exists(self, save_path: Path) -> None:
        """Find the most recent valid checkpoint and restore from it."""
        checkpoints = sorted(save_path.glob("checkpoint_[0-9]*"))
        for cp in reversed(checkpoints):
            if self._checkpoint_is_valid(cp):
                log.info(f"Valid checkpoint found: {cp} — restoring")
                self._restore_from_checkpoint(cp)
                self._last_checkpoint_path = cp
                return
        log.info("No valid checkpoint found — starting fresh")

    def _checkpoint_is_valid(self, path: Path) -> bool:
        """Return ``True`` only if the checkpoint directory is structurally complete.

        Checks for the presence of ``server_state.pkl`` and all regret chunk
        files declared in that state.  The LMDB index lives at
        ``save_path / "lmdb_index"`` and is checked there (not inside the
        checkpoint directory).
        """
        if not (path / "server_state.pkl").exists():
            return False
        # LMDB lives beside the checkpoints, not inside them.
        if not (self._save_path / "lmdb_index").exists():
            return False
        try:
            state_dict = joblib.load(path / "server_state.pkl")
        except Exception:
            log.warning(f"Could not load server_state.pkl from {path}")
            return False
        n_chunks = state_dict.get("n_chunks_per_street", {})
        for r in range(4):
            for chunk_id in range(n_chunks.get(r, 0)):
                if not (path / f"regret_{r}_chunk_{chunk_id:06d}.npy").exists():
                    return False
        return True

    def _restore_from_checkpoint(self, path: Path) -> None:
        """Reload shared-memory tables from a checkpoint directory.

        Sets ``server._start_t`` to the iteration *after* the one that was
        checkpointed so training resumes without replaying completed work.
        """
        state_dict = joblib.load(path / "server_state.pkl")

        # Advance past the checkpointed iteration.
        self._server._start_t = state_dict["t"] + 1

        for r in range(4):
            n_regret = state_dict.get("n_chunks_per_street", {}).get(r, 0)
            for chunk_id in range(n_regret):
                arr = np.load(path / f"regret_{r}_chunk_{chunk_id:06d}.npy")
                self._server._agent.regret_tables[r]._restore_chunk(chunk_id, arr)

            n_strategy = state_dict.get("n_strategy_chunks_per_street", {}).get(r, 0)
            for chunk_id in range(n_strategy):
                arr = np.load(path / f"strategy_{r}_chunk_{chunk_id:06d}.npy")
                self._server._agent.strategy_tables[r]._restore_chunk(chunk_id, arr)

        log.info(
            f"Restored from checkpoint — resuming at t={self._server._start_t} "
            f"(checkpoint was t={state_dict['t']})"
        )
