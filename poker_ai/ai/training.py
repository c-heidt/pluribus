"""Shared training primitives used by single- and multi-process CFR.

Both training modes — the single-process
:func:`~poker_ai.ai.singleprocess.train.simple_search` loop and the
multi-process :class:`~poker_ai.ai.multiprocess.server.Server` +
:class:`~poker_ai.ai.multiprocess.worker.Worker` pair — run the same
CFR schedule.  One CFR traversal per player per iteration, sync
barriers at regular intervals, sync-cycle-based strategy updates and
sync-cycle-based Linear CFR discounting.  The logic that drives that
schedule lives here so both modes call into a single, tested source
of truth.

Contents
--------
``cfr_step``
    One per-player CFR traversal with the stochastic cfr/cfrp choice.
``strategy_step``
    One per-player strategy-sampling traversal.
``at_sync_barrier``, ``should_update_strategy``, ``should_discount``,
``should_checkpoint``
    Schedule predicates that check sync-cycle-based conditions
    against the current iteration counter.
:class:`DiscountState`
    Owns the LCFR discount window and the standard
    ``discount_step / (discount_step + 1)`` factor formula.
``load_info_set_lut``
    Post-fork card-info LUT loader shared by workers and the server.

Invariants preserved by the primitives
--------------------------------------
- The caller owns the ``local_delta`` accumulator passed to
  :func:`cfr_step`.  The worker keeps a persistent accumulator across
  many calls and flushes on sync jobs; the single-process loop
  allocates a fresh accumulator per iteration and merges immediately.
  Neither ownership model is baked into the primitive.
- The discount formula and the pruning decision are implemented in
  exactly one place, guaranteeing single- and multi-process runs
  produce identical behaviour for the same hyperparameters and
  random seed.
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
"""Probability of selecting CFR-P over standard CFR once past ``prune_threshold``.

The remaining ``1 - PRUNE_PROBABILITY`` of traversals always use
standard CFR.  Keeping a non-zero fraction of unpruned traversals
lets previously-pruned actions recover if their regret climbs back
into the positive range.
"""


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
    """Execute one CFR traversal for player *i* with stochastic pruning.

    Draws a pruning coin: with probability :data:`PRUNE_PROBABILITY`
    and only once the iteration counter is past ``prune_threshold``,
    the traversal uses CFR-P; otherwise it uses standard external-
    sampling CFR.  The 5% unpruned fallback is what keeps pruned
    actions from being permanently stuck — their regrets still get
    updated on those traversals.

    Regret updates are written into the caller-owned ``local_delta``
    buffer; the caller decides when to flush via
    :func:`poker_ai.ai.cfr.merge_local_delta`.

    Parameters
    ----------
    tables : CFRTables
        Shared regret and strategy tables.
    state : PokerState
        Root game state for this traversal.
    i : int
        Traversing player index.
    t : int
        Current training iteration.
    prune_threshold : int
        Raw iteration count at which CFR-P becomes eligible.  Below
        this threshold the step always uses standard CFR.
    c : int
        Regret threshold for CFR-P: actions with cumulative regret at
        or below ``c`` are pruned (unless on the river).
    local_delta : dict[tuple[int, str], np.ndarray]
        Caller-owned regret accumulator.
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

    Thin wrapper around :func:`poker_ai.ai.strategy.update_strategy`
    that keeps the single- and multi-process loops structurally
    symmetric with :func:`cfr_step`.

    Parameters
    ----------
    tables : CFRTables
        Shared tables (only ``tables.strategy`` is written).
    state : PokerState
        Root game state for this traversal.
    i : int
        Traversing player whose average strategy is being updated.
    """
    update_strategy(tables, state, i)


# ---------------------------------------------------------------------------
# Sync-cycle-based schedule predicates
# ---------------------------------------------------------------------------


def at_sync_barrier(t: int, sync_interval: int) -> bool:
    """Return ``True`` when *t* is a sync-barrier iteration.

    A sync barrier is any iteration where ``t % sync_interval == 0``.
    Sync barriers are the points at which workers flush their
    accumulated regret deltas into the shared tables, and they are
    also the only iterations at which strategy updates, discounting,
    and checkpointing may fire.

    Parameters
    ----------
    t : int
        Current iteration counter.
    sync_interval : int
        Number of iterations between sync barriers.
    """
    return t % sync_interval == 0


def should_update_strategy(
    sync_step: int,
    strategy_interval: int,
    update_threshold: int,
) -> bool:
    """Return ``True`` if a strategy-update pass is due this sync cycle.

    Fires once two conditions are met: the warm-up period has
    elapsed (``sync_step > update_threshold``) and the sync-step
    counter is divisible by the configured interval.  Both values are
    counted in **sync cycles** — not raw iterations — so strategy
    updates cannot misalign with the sync barrier.

    Parameters
    ----------
    sync_step : int
        Current sync-step counter, ``t // sync_interval``.
    strategy_interval : int
        Period (in sync cycles) between strategy updates.
    update_threshold : int
        Number of sync cycles to run before strategy updates may begin.
    """
    return (
        sync_step > update_threshold
        and sync_step % strategy_interval == 0
    )


def should_discount(sync_step: int, discount_interval: int) -> bool:
    """Return ``True`` if an LCFR discount is due this sync cycle.

    Discounts fire whenever ``sync_step`` is divisible by
    ``discount_interval``.  Whether the discount is actually applied
    depends on :class:`DiscountState.active`, which closes once the
    window duration is reached.

    Parameters
    ----------
    sync_step : int
        Current sync-step counter.
    discount_interval : int
        Period (in sync cycles) between LCFR discount applications.
    """
    return sync_step % discount_interval == 0


def should_checkpoint(sync_step: int, checkpoint_interval: int) -> bool:
    """Return ``True`` if a checkpoint is due this sync cycle.

    Parameters
    ----------
    sync_step : int
        Current sync-step counter.
    checkpoint_interval : int
        Period (in sync cycles) between checkpoint writes.
    """
    return sync_step % checkpoint_interval == 0


# ---------------------------------------------------------------------------
# LCFR discount state
# ---------------------------------------------------------------------------


class DiscountState:
    """LCFR discount-window controller.

    Owns two pieces of state: whether the discount window is still
    active, and the total number of sync cycles over which it stays
    active.  The discount factor at the *k*-th discount application is
    ``k / (k + 1)``, which yields the standard Linear CFR weighting
    scheme when applied periodically.  Once ``sync_step`` reaches
    ``duration_cycles`` the window closes and subsequent calls to
    :meth:`apply` become no-ops.

    The class is intentionally minimal — it does not track the
    iteration counter, does not touch the tables except via
    :meth:`apply`, and holds no mutable references to shared state.
    This makes it safe to serialise the single ``active`` flag into a
    checkpoint and restore it later.

    Attributes
    ----------
    active : bool
        ``True`` while the discount window is open.  Cleared to
        ``False`` the first time :meth:`apply` sees a
        ``sync_step >= duration_cycles``.  Directly writable by
        :mod:`poker_ai.ai.checkpoint` during a resume.
    """

    def __init__(self, duration_cycles: int, discount_interval: int):
        """Create a new discount-window controller.

        Parameters
        ----------
        duration_cycles : int
            Number of sync cycles over which the LCFR discount remains
            active.  The window closes at the first call to
            :meth:`apply` with ``sync_step >= duration_cycles``.
        discount_interval : int
            Period (in sync cycles) between discount applications.
            The *k*-th application is computed at sync step
            ``k * discount_interval``.
        """
        self._duration_cycles = duration_cycles
        self._discount_interval = discount_interval
        self.active: bool = True

    @property
    def duration_cycles(self) -> int:
        """Length of the discount window in sync cycles."""
        return self._duration_cycles

    def apply(self, tables: CFRTables, sync_step: int) -> None:
        """Apply the LCFR discount for the current sync step.

        Called at every sync cycle where
        :func:`should_discount` returns ``True``.  No-op if the
        discount window has already closed or if ``sync_step`` is at
        or past ``duration_cycles`` (in which case the window is
        closed as a side-effect so future calls are cheap).

        The discount step is ``sync_step // discount_interval`` and
        the factor is ``discount_step / (discount_step + 1)``.  Both
        regret and strategy tables are scaled by this factor inside
        :meth:`CFRTables.apply_discount
        <poker_ai.ai.cfr_tables.CFRTables.apply_discount>`.

        Parameters
        ----------
        tables : CFRTables
            Tables to discount in place.
        sync_step : int
            Current sync-step counter.
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

    The card-info LUT maps hole-card + board combinations to abstract
    cluster identifiers and is the largest single object consumed by
    training (hundreds of megabytes for a six-player game).  Two
    on-disk layouts are supported:

    - **Joblib + mmap** (``pickle_dir=False``, the default): a single
      ``card_info_lut.joblib`` file is opened read-only and mmapped.
      Multiple worker processes that load the same file share its
      physical pages via the kernel page cache, so memory usage is
      roughly constant in the number of workers.
    - **Pickle directory** (``pickle_dir=True``, deprecated): a
      directory of per-street pickle files, delegated to
      :func:`poker_ai.utils.io.load_info_set_lut`.  Retained for
      backward compatibility with older datasets.

    Both server (parent process, pre-fork) and workers (child
    processes, post-fork) call this function so the loading logic
    lives in a single place.

    Parameters
    ----------
    lut_path : str or Path
        Directory containing ``card_info_lut.joblib`` (or the legacy
        per-street pickle layout).
    pickle_dir : bool
        Select the legacy pickle-directory code path when ``True``.

    Returns
    -------
    object
        The deserialised LUT object (typically a nested dict).
    """
    if pickle_dir:
        return utils.io.load_info_set_lut(lut_path, pickle_dir)
    lut_file_path = os.path.join(str(lut_path), "card_info_lut.joblib")
    with open(lut_file_path, "rb") as lut_file:
        with _mmap.mmap(lut_file.fileno(), 0, access=_mmap.ACCESS_READ) as lut_mmap:
            return joblib.load(lut_mmap)
