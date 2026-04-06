"""CheckpointManager — full-sweep checkpointing with atomic writes.

Architecture
------------
Signal handling
    SIGTERM and SIGINT are intercepted in the *parent (server) process* before
    any workers are spawned.  The handler only sets a ``threading.Event``; all
    actual I/O happens from the server's main thread after the loop notices the
    event and breaks cleanly.

Checkpoint write flow
    1. Flush all workers (broadcast sync + barrier join).
    2. Delegate chunk writing to ``CFRTables.save_chunks()`` which writes all
       allocated chunks for all 8 tables to a temporary directory.
    3. Flush per-street LMDB indexes to disk.
    4. Persist the server state dict (includes current iteration ``t`` and
       per-street chunk counts for resume validation).
    5. Atomically rename the temp directory to the final checkpoint directory.
    6. Remove the previous checkpoint (only one generation is kept).

Resume flow
    On construction, the most recent valid checkpoint is found and restored
    before any workers are spawned.  ``server._start_t`` is advanced past the
    last checkpointed iteration so training resumes without replaying work.

Shutdown window
    SLURM jobs should be submitted with ``--signal=SIGTERM@300`` to give five
    minutes for the checkpoint to complete.
"""

import logging
import os
import shutil
import signal
import threading
import time
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import joblib

from poker_ai.utils.io import atomic_joblib_dump

if TYPE_CHECKING:
    from poker_ai.ai.multiprocess.server import Server

log = logging.getLogger("poker_ai.ai.checkpoint")


def _extract_n_chunks_per_street(state_dict: dict) -> Dict[int, int]:
    """Extract per-street chunk counts from new or old format state dicts.

    New format stores a single ``n_chunks_per_street`` dict.
    Old format stored separate ``n_chunks_per_street`` (regret) and
    ``n_strategy_chunks_per_street`` dicts — we take the max per street.
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
    """Manages periodic and emergency checkpointing for the training server.

    Must be instantiated in the **parent process before workers are spawned**
    so that SIGTERM/SIGINT handlers are registered with the correct PID and
    that ``_load_checkpoint_if_exists`` runs before any shared-memory tables
    are populated by workers.

    Parameters
    ----------
    server:
        The ``Server`` instance that owns the tables and job queue.
    save_path:
        Root directory for checkpoints.  Each checkpoint is written to a
        timestamped subdirectory (``checkpoint_<unix_ts>``).
    """

    def __init__(self, server: "Server", save_path: Path) -> None:
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
        """Event set by the SIGTERM/SIGINT handler."""
        return self._sigterm_event

    def checkpoint(self, t: int, emergency: bool = False) -> None:
        """Write a full checkpoint at iteration *t*.

        Parameters
        ----------
        t:
            Current training iteration.
        emergency:
            When ``True`` skip the worker-flush step.
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
            # every checkpoint directory is self-contained even when only dirty
            # chunks were written above.
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
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        log.warning(f"{sig_name} received — signalling server loop to checkpoint and exit")
        self._sigterm_event.set()

    # -----------------------------------------------------------------------
    # Resume logic
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
        """Return ``True`` only if the checkpoint directory is structurally complete."""
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

    # Structural parameters that must match between the saved state and the
    # current Server for a resume to be safe.  Mismatches raise.
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

    def _validate_config_compatibility(self, state_dict: dict) -> None:
        """Refuse to resume if structural hyperparameters have changed.

        ``max_runtime_hours``, ``checkpoint_interval``, ``n_processes``,
        and ``nickname`` are intentionally *not* checked — they are
        allowed to differ between runs.
        """
        attr_map = {
            "n_players": "_n_players",
            "sync_interval": "_sync_interval",
            "discount_interval": "_discount_interval",
            "discount_duration_cycles": "_discount_duration_cycles",
            "strategy_interval": "_strategy_interval",
            "update_threshold": "_update_threshold",
            "prune_threshold": "_prune_threshold",
            "c": "_c",
        }
        mismatches = []
        for key in self._STRUCTURAL_KEYS:
            if key not in state_dict:
                continue  # old checkpoint format — skip silently
            saved = state_dict[key]
            current = getattr(self._server, attr_map[key])
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
        """Reload shared-memory tables from a checkpoint directory."""
        state_dict = joblib.load(path / "server_state.pkl")

        self._validate_config_compatibility(state_dict)

        self._server._start_t = state_dict["t"] + 1
        self._server._discounting_active = state_dict.get("discount_active", True)

        ncs = _extract_n_chunks_per_street(state_dict)
        self._server._tables.restore_chunks(path, ncs)

        log.info(
            f"Restored from checkpoint — resuming at t={self._server._start_t} "
            f"(checkpoint was t={state_dict['t']})"
        )
