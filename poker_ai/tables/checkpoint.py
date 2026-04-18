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
1. Broadcast a sync job to every worker and wait until all pending
   regret deltas have been merged into the shared tables
   (skipped for emergency checkpoints that cannot afford the wait).
2. Create a temporary directory next to the final target and hand
   control to :meth:`CFRTables.save_chunks
   <poker_ai.tables.cfr_tables.CFRTables.save_chunks>`, which writes
   every dirty chunk as an atomic ``.npy`` file.
3. Hardlink the unchanged chunk files from the previous checkpoint
   into the temp directory so every checkpoint remains
   self-contained with no extra I/O (same-inode hardlinks are
   near-free on every POSIX filesystem).
4. Flush the per-street LMDB indexes and persist the server state.
5. Atomically rename the temp directory to its final name and
   delete the previous checkpoint (we keep exactly one generation
   on disk).

Shutdown window
---------------
When running under SLURM, submit jobs with
``--signal=SIGTERM@300`` so the manager has five minutes between
``SIGTERM`` and ``SIGKILL`` to write the final checkpoint.
"""

import logging
import os
import shutil
import signal
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING, Union

import joblib

if TYPE_CHECKING:
    from poker_ai.blueprint.multiprocess.server import Server

log = logging.getLogger("poker_ai.tables.checkpoint")


def atomic_joblib_dump(obj: Any, path: Union[str, Path]) -> None:
    """Save *obj* with joblib atomically via a temp file then rename.

    Using a temp file in the same directory guarantees the rename is
    atomic on POSIX (rename syscall) even across NFS when the tmp and
    target are on the same mount point.  On failure the temp file is
    cleaned up and the original path is left untouched.
    """
    path = Path(path)
    tmp_fd, tmp_str = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp.joblib", prefix=path.stem + "_"
    )
    tmp_path = Path(tmp_str)
    try:
        os.close(tmp_fd)
        joblib.dump(obj, tmp_path)
        shutil.move(str(tmp_path), str(path))
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"atomic_joblib_dump failed for {path}: {exc}") from exc


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

    def __init__(self, server: "Server", save_path: Path) -> None:
        """Register signal handlers and auto-resume from *save_path* if possible.

        Parameters
        ----------
        server : Server
            Parent training server.  The manager reaches into the
            server for its tables, configuration, and state dict.
        save_path : Path
            Root directory for checkpoint subdirectories.  Created
            implicitly when the first checkpoint is written.
        """
        self._server = server
        self._save_path = Path(save_path)
        self._last_checkpoint_path: Optional[Path] = None
        self._sigterm_event: threading.Event = threading.Event()

        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)
        log.info("CheckpointManager: SIGTERM/SIGINT handlers registered")

        self._load_checkpoint_if_exists(self._save_path)

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    @property
    def sigterm_event(self) -> threading.Event:
        """Event set by the ``SIGTERM``/``SIGINT`` handler."""
        return self._sigterm_event

    def checkpoint(self, t: int, emergency: bool = False) -> None:
        """Write a full checkpoint at iteration *t*.

        The write is performed atomically by building the checkpoint
        in a temporary directory next to the save root and then
        renaming it to its final name.  If any step fails the temp
        directory is removed and the previous checkpoint is left
        untouched.

        Parameters
        ----------
        t : int
            Current training iteration.  Embedded in the saved state
            dict so the next run can resume at ``t + 1``.
        emergency : bool, optional
            When ``True`` skip the worker-flush step.  Used in the
            error-handling path when the server believes at least one
            worker has crashed — waiting on such a worker would
            deadlock the shutdown.
        """
        label = "emergency" if emergency else "scheduled"
        log.info(f"[t={t}] Checkpoint ({label}) starting")
        t_start = time.monotonic()

        if not emergency:
            self._server.flush_all_workers()

        tmp_path = self._save_path / f"checkpoint_tmp_{int(time.time())}"
        tmp_path.mkdir(parents=True, exist_ok=True)

        try:
            self._server._tables.save_chunks(tmp_path)

            # Carry forward unchanged chunk files from the previous checkpoint
            # via hardlinks (instant, no I/O — same filesystem).  This ensures
            # every checkpoint directory is self-contained even when only
            # dirty chunks were written above.
            if self._last_checkpoint_path and self._last_checkpoint_path.exists():
                for old_file in self._last_checkpoint_path.glob("*.npy"):
                    new_file = tmp_path / old_file.name
                    if not new_file.exists():
                        os.link(str(old_file), str(new_file))

            self._server._tables.flush_indexes()

            state_dict = self._server.to_dict(t=t)
            atomic_joblib_dump(state_dict, tmp_path / "server_state.pkl")

            final_path = self._save_path / f"checkpoint_{int(time.time())}"
            tmp_path.rename(final_path)
            tmp_path = None

        except Exception:
            log.exception("Checkpoint write failed — temp dir preserved for inspection")
            if tmp_path is not None and tmp_path.exists():
                shutil.rmtree(tmp_path, ignore_errors=True)
            raise

        if self._last_checkpoint_path and self._last_checkpoint_path.exists():
            shutil.rmtree(self._last_checkpoint_path)
            log.info(f"Deleted previous checkpoint: {self._last_checkpoint_path}")
        self._last_checkpoint_path = final_path

        elapsed = time.monotonic() - t_start
        log.info(
            f"[t={t}] Checkpoint complete in {elapsed:.1f}s — "
            f"Path: {final_path}"
        )

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
            # discount_duration_cycles lives on Server._discount_state; the
            # rest are plain Server attributes.
            if key == "discount_duration_cycles":
                return self._server._discount_state.duration_cycles
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

        self._server._start_t = state_dict["t"] + 1
        self._server._discount_state.active = state_dict.get("discount_active", True)

        ncs = _extract_n_chunks_per_street(state_dict)
        self._server._tables.restore_chunks(path, ncs)

        log.info(
            f"Restored from checkpoint — resuming at t={self._server._start_t} "
            f"(checkpoint was t={state_dict['t']})"
        )
