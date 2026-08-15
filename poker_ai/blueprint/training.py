"""Shared training primitives used by single- and multi-process CFR.

Both training modes — the single-process
:func:`~poker_ai.blueprint.singleprocess.train.simple_search` loop and the
multi-process :class:`~poker_ai.blueprint.multiprocess.server.Server` +
:class:`~poker_ai.blueprint.multiprocess.worker.Worker` pair — run the same
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
import random as _py_random
from typing import Dict, Tuple

import numpy as np

from poker_ai.blueprint.bias import BiasClass
from poker_ai.blueprint.cfr import cfr, cfrp
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.blueprint.strategy import update_strategy
from environment.poker_env import PokerEnv as PokerState

log = logging.getLogger("poker_ai.blueprint.training")


def seed(seed: int = 42) -> None:
    """Seed numpy and Python RNGs so CFR runs are reproducible."""
    np.random.seed(seed)
    _py_random.seed(seed)


def pin_blas_threads(n_threads: int = 1) -> None:
    """Pin this process's BLAS/OpenMP thread pools to ``n_threads``.

    Each CFR traversal is a single fine-grained walk over tiny per-node
    arrays that never benefits from intra-op BLAS threads — the same
    reasoning as ``poker_ai.search.parallel._limit_worker_threads``,
    duplicated here (not imported) since training has no other dependency
    on the search package. Without this, ``W`` forked training workers on a
    ``C``-core box each spin up OpenBLAS's default ~``C``-thread pool,
    oversubscribing to ``W*C`` threads; the search-side function's docstring
    documents a concrete measured regression this exact fix produced there
    (a 22-core/8-worker box where an unpinned pool ran *slower* than serial
    and inverted the compiled core's per-iteration win into a net loss).

    Best-effort and never raises (pinning is an optimisation, not
    correctness): sets the standard env vars AND calls OpenBLAS's runtime
    setter directly on numpy's bundled library, since a forked child's BLAS
    pool is already initialised and may ignore the env var alone.
    """
    import os as _os

    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ[_var] = str(n_threads)
    try:
        import ctypes
        import glob

        libdir = _os.path.join(_os.path.dirname(np.__file__), ".libs")
        for _so in glob.glob(_os.path.join(libdir, "libopenblas*.so")):
            try:
                _lib = ctypes.CDLL(_so)
            except OSError:
                continue
            if hasattr(_lib, "openblas_set_num_threads"):
                _lib.openblas_set_num_threads(int(n_threads))
    except Exception:
        pass  # best-effort; a missing/renamed BLAS must never break training


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
    bias: BiasClass = "none",
    bias_magnitude: float = 0.0,
    core=None,
) -> None:
    """Execute one CFR traversal for player *i* with stochastic pruning.

    Draws a pruning coin: with probability :data:`PRUNE_PROBABILITY`
    and only once the iteration counter is past ``prune_threshold``,
    the traversal uses CFR-P; otherwise it uses standard external-
    sampling CFR.  The 5% unpruned fallback is what keeps pruned
    actions from being permanently stuck — their regrets still get
    updated on those traversals.

    The pruning coin is drawn identically whether the traversal runs
    on the Python path or the compiled core, so the cfr/cfrp mix (and
    the global RNG stream advance) is the same either way; only the
    recursion body differs.

    Regret updates are written into the caller-owned ``local_delta``
    buffer; the caller decides when to flush via
    :func:`poker_ai.blueprint.cfr.merge_local_delta`.

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
    bias : BiasClass, optional
        Action class for biased-blueprint training.  Forwarded to
        :func:`cfr` / :func:`cfrp`; ``"none"`` (default) selects the
        legacy unbiased path.
    bias_magnitude : float, optional
        Per-occurrence terminal-payoff bonus applied when
        ``bias != "none"``.
    core : poker_ai.blueprint.core_runner.CoreDriver, optional
        When supplied (``PLURIBUS_CFR_CORE=1``), the traversal runs
        through the compiled Cython core instead of the Python
        :func:`cfr` / :func:`cfrp` recursion.  The core accumulates
        into the same ``local_delta`` in place, so the caller's flush
        path is unchanged.  ``None`` (default) keeps the Python path,
        which stays the live oracle.
    """
    use_pruning = np.random.uniform() < PRUNE_PROBABILITY
    prune_on = use_pruning and t > prune_threshold
    if core is not None:
        core.run_cfr(state, i, t, c, prune_on, local_delta)
        return
    if prune_on:
        cfrp(tables, state, i, t, c, local_delta,
             bias=bias, bias_magnitude=bias_magnitude)
    else:
        cfr(tables, state, i, t, local_delta,
            bias=bias, bias_magnitude=bias_magnitude)


def strategy_step(
    tables: CFRTables,
    state: PokerState,
    i: int,
    local_delta: Dict[Tuple[int, str], np.ndarray] = None,
    core=None,
) -> None:
    """Execute one strategy-update traversal for player *i*.

    Wrapper around :func:`poker_ai.blueprint.strategy.update_strategy` that keeps
    the single- and multi-process loops structurally symmetric with
    :func:`cfr_step` — same ``core`` / ``local_delta`` dispatch shape.

    Parameters
    ----------
    tables : CFRTables
        Shared tables (only ``tables.strategy`` is written, and only when
        ``local_delta is None``).
    state : PokerState
        Root game state for this traversal.
    i : int
        Traversing player whose average strategy is being updated.
    local_delta : dict, optional
        Caller-owned visit-count accumulator keyed by ``(betting_round,
        info_set)``.  When supplied, the sampled visit counts are written here
        (to be flushed later via
        :func:`poker_ai.blueprint.cfr.merge_local_strategy_delta`) instead of
        directly into the shared ``tables.strategy``.  ``None`` (default) keeps
        the legacy direct-write behaviour used by the single-process loop.
    core : poker_ai.blueprint.core_runner.CoreDriver, optional
        When supplied (``PLURIBUS_CFR_CORE=1``), the playthrough runs through the
        compiled core, accumulating into ``local_delta`` in place.  ``None``
        (default) keeps the Python path, which stays the live oracle.
    """
    if core is not None:
        core.run_strategy(state, i, local_delta)
        return
    update_strategy(tables, state, i, local_delta=local_delta)


# ---------------------------------------------------------------------------
# Sync-cycle-based schedule predicates
# ---------------------------------------------------------------------------


def at_sync_barrier(t: int, sync_interval: int, step: int = 1) -> bool:
    """Return ``True`` iff iteration *t* crossed a sync-barrier boundary.

    Uses crossing detection so the barrier fires exactly once per
    ``sync_interval`` traversals-per-player, even when ``t`` advances
    by more than one per loop pass (as happens in the multi-process
    server where each loop dispatches ``workers_per_player``
    traversals per player).

    For ``step == 1`` this is equivalent to the classical
    ``t % sync_interval == 0`` predicate — so legacy single-process
    callers see no behavioural change.

    Parameters
    ----------
    t : int
        Current traversals-per-player counter (post-increment).
    sync_interval : int
        Number of traversals-per-player between sync barriers.
    step : int, optional
        Size of the increment applied to ``t`` on the loop pass that
        produced this value.  Defaults to ``1``.
    """
    return (t - step) // sync_interval < t // sync_interval


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
        :mod:`poker_ai.tables.checkpoint` during a resume.
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
        <poker_ai.tables.cfr_tables.CFRTables.apply_discount>`.

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


