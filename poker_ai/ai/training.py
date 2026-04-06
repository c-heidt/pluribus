"""Shared training primitives for single-process and multi-process CFR.

Both training modes (``singleprocess.train.simple_search`` and the
multi-process ``Server``/``Worker`` pair) drive exactly the same
schedule: one CFR traversal per player, sync-cycle-based strategy
updates, sync-cycle-based LCFR discounting, and periodic checkpoints.

This module factors out the logic that was previously duplicated across
those files:

- :func:`cfr_step` — one CFR traversal for one player, with the
  stochastic cfr/cfrp pruning decision.
- :func:`strategy_step` — one strategy-update traversal for one player.
- Schedule predicates (:func:`at_sync_barrier`,
  :func:`should_update_strategy`, :func:`should_discount`,
  :func:`should_checkpoint`) that check sync-cycle-based conditions.
- :class:`DiscountState` — owns the LCFR discount window and the
  standard ``discount_step / (discount_step + 1)`` factor formula.
- :func:`load_info_set_lut` — shared post-fork LUT loader.

Callers (single-process loop, worker dispatch branches, server main
loop) remain responsible for their own process/lifecycle concerns and
for owning their ``local_delta`` buffer.  The worker keeps a persistent
buffer across many CFR calls; single-process creates a fresh buffer per
iteration.  Neither ownership model is baked into the primitives.
"""

import logging
import mmap as _mmap
import os
from pathlib import Path
from typing import Dict, Tuple, Union

import joblib
import numpy as np

from poker_ai import utils
from poker_ai.ai.cfr import cfr, cfrp
from poker_ai.ai.cfr_tables import CFRTables
from poker_ai.ai.strategy import update_strategy
from poker_ai.environment.poker_env import PokerEnv as PokerState

log = logging.getLogger("poker_ai.ai.training")

PRUNE_PROBABILITY: float = 0.95
"""Probability of selecting CFR-P over standard CFR when ``t > prune_threshold``."""


# ---------------------------------------------------------------------------
# Per-player training steps
# ---------------------------------------------------------------------------


def cfr_step(
    tables: CFRTables,
    state: PokerState,
    i: int,
    t: int,
    prune_threshold: int,
    c: int,
    local_delta: Dict[Tuple[int, str], np.ndarray],
) -> None:
    """Execute one CFR traversal for player *i*.

    With probability :data:`PRUNE_PROBABILITY` and when ``t > prune_threshold``,
    uses CFR with pruning; otherwise standard CFR with external sampling.

    Writes regret deltas into the caller-owned ``local_delta``.  The
    caller decides when to flush via :func:`poker_ai.ai.cfr.merge_local_delta`.
    """
    use_pruning = np.random.uniform() < PRUNE_PROBABILITY
    if use_pruning and t > prune_threshold:
        cfrp(tables, state, i, t, c, local_delta)
    else:
        cfr(tables, state, i, t, local_delta)


def strategy_step(
    tables: CFRTables,
    state: PokerState,
    i: int,
) -> None:
    """Execute one strategy-update traversal for player *i*.

    Thin wrapper around :func:`poker_ai.ai.strategy.update_strategy` to
    keep the single- and multi-process loops symmetric with
    :func:`cfr_step`.
    """
    update_strategy(tables, state, i)


# ---------------------------------------------------------------------------
# Sync-cycle-based schedule predicates
# ---------------------------------------------------------------------------


def at_sync_barrier(t: int, sync_interval: int) -> bool:
    """Return ``True`` on iterations that are a multiple of ``sync_interval``."""
    return t % sync_interval == 0


def should_update_strategy(
    sync_step: int,
    strategy_interval: int,
    update_threshold: int,
) -> bool:
    """Return ``True`` if a strategy-update pass is due this sync cycle.

    Fires when we are past the warm-up threshold and the sync-step
    counter is divisible by the configured interval.
    """
    return (
        sync_step > update_threshold
        and sync_step % strategy_interval == 0
    )


def should_discount(sync_step: int, discount_interval: int) -> bool:
    """Return ``True`` if a discount is due this sync cycle."""
    return sync_step % discount_interval == 0


def should_checkpoint(sync_step: int, checkpoint_interval: int) -> bool:
    """Return ``True`` if a checkpoint is due this sync cycle."""
    return sync_step % checkpoint_interval == 0


# ---------------------------------------------------------------------------
# LCFR discount state
# ---------------------------------------------------------------------------


class DiscountState:
    """Owns the LCFR discount window and applies per-step discount factors.

    The discount factor at sync step *s* is ``s / (s + 1)`` computed on
    the number of discount applications that have occurred so far.
    After ``sync_step >= duration_cycles`` the window closes and
    further calls to :meth:`apply` are no-ops.

    Parameters
    ----------
    duration_cycles:
        Number of sync cycles over which discounting remains active.
    discount_interval:
        Discount every N sync cycles (``sync_step % discount_interval == 0``).
    """

    def __init__(self, duration_cycles: int, discount_interval: int):
        self._duration_cycles = duration_cycles
        self._discount_interval = discount_interval
        self.active: bool = True

    @property
    def duration_cycles(self) -> int:
        return self._duration_cycles

    def apply(self, tables: CFRTables, sync_step: int) -> None:
        """Apply the LCFR discount for the current sync step, if the window is open.

        No-op if the window has already closed or if ``sync_step`` is
        past ``duration_cycles`` — in the latter case the window is
        closed as a side-effect so future calls are cheap.
        """
        if not self.active:
            return
        if sync_step >= self._duration_cycles:
            self.active = False
            log.info(f"Discount window closed after {sync_step} sync cycles")
            return
        discount_step = sync_step // self._discount_interval
        factor = discount_step / (discount_step + 1)
        log.info(
            f"[sync_step={sync_step}] Discounting regrets and strategy "
            f"(step={discount_step}, factor={factor:.4f})"
        )
        tables.apply_discount(factor)


# ---------------------------------------------------------------------------
# LUT loading
# ---------------------------------------------------------------------------


def load_info_set_lut(
    lut_path: Union[str, Path],
    pickle_dir: bool,
):
    """Load the card-info LUT from disk.

    Two code paths are supported (shared by the server's parent-process
    pre-fork load and each worker's post-fork load):

    - **Joblib + mmap** (``pickle_dir=False``): the LUT is a single
      ``card_info_lut.joblib`` file; we mmap it read-only so multiple
      workers can share pages.
    - **Pickle directory** (``pickle_dir=True``): the deprecated
      per-street pickle layout, delegated to
      :func:`poker_ai.utils.io.load_info_set_lut`.
    """
    if pickle_dir:
        return utils.io.load_info_set_lut(lut_path, pickle_dir)
    lut_file_path = os.path.join(str(lut_path), "card_info_lut.joblib")
    with open(lut_file_path, "rb") as lut_file:
        with _mmap.mmap(lut_file.fileno(), 0, access=_mmap.ACCESS_READ) as lut_mmap:
            return joblib.load(lut_mmap)
