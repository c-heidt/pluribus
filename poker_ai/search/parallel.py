"""Serial search loop + the shared fork/worker helpers (§6.7).

Within-search parallelism (running ``W`` independent replicas of a *single* search and
merging them once) was retired: production runs one hand per core with the search at
``workers=1`` (per-hand parallelism, :func:`evaluation.runner.run_evaluation_parallel`),
so a nested replica pool is never used.  What remains here is:

- :func:`run_loop` — the orchestrator's serial iteration loop (Linear-CFR discount on the
  ``discount_interval`` cadence, stop on ``max_iterations`` **or** ``max_wall_seconds``),
  driven by :func:`poker_ai.search.solver.solve`;
- :func:`resolve_workers` — resolve a ``None`` worker count to a cpu-based default,
  reused by the **per-hand** pool (:mod:`evaluation.hand_pool`) and the calibration sweep;
- :func:`_reopen_leaf_fleet_lmdb` / :func:`_limit_worker_threads` — fork-safety helpers
  (LMDB reader-slot repair after a fork; BLAS thread pinning) used by any process that
  forks a worker that runs a search, i.e. the per-hand pool.
"""

from __future__ import annotations

import os
import time
from typing import Tuple

from poker_ai.search.context import SubgameContext
from poker_ai.search.solver_state import SolverConfig, SolverState


# ----------------------------------------------------------------------
# Worker count
# ----------------------------------------------------------------------


def resolve_workers(workers: "int | None") -> int:
    """Resolve a worker count to a concrete value.

    ``None`` → a cpu-based default (SLURM-aware, leaving one core free, mirroring
    :mod:`poker_ai.blueprint.multiprocess`).  Any explicit value is clamped to ≥ 1.
    Used by the per-hand pool and the calibration sweep to size their core count.
    """
    if workers is not None:
        return max(1, int(workers))
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        n = int(slurm) if slurm else (os.cpu_count() or 1)
    except (TypeError, ValueError):
        n = os.cpu_count() or 1
    return max(1, n - 1)


# ----------------------------------------------------------------------
# Serial iteration loop (the one search loop)
# ----------------------------------------------------------------------


def run_loop(solver, state: SolverState, cfg: SolverConfig,
             on_iteration=None) -> Tuple[int, str]:
    """Drive ``solver`` under the dual stop; return ``(iterations, stop_reason)``.

    This *is* the orchestrator's search loop (Linear-CFR discount on the
    ``discount_interval`` cadence, stop on ``max_iterations`` **or**
    ``max_wall_seconds``); :func:`solve` calls it.  The binding ``max_iterations`` is the
    **structural iteration budget** :func:`solve` computed for this subgame
    (:mod:`poker_ai.search.budget`) — a real subgame is too large for a single replica to
    reach a tight equilibrium online, so the stop is a machine-independent per-subgame
    iteration count, not an online convergence test.  ``stop_reason`` is which cap ended
    the loop (eval doc §6 ``decisions.stop_reason``): ``'wall_cap'`` if the wall-clock
    backstop broke early, else ``'iteration_cap'`` (the loop ran the full budget,
    including the degenerate 0-iteration case).
    """
    start = time.perf_counter()
    delta = cfg.discount_interval
    iterations = 0
    stop_reason = "iteration_cap"
    for t in range(1, cfg.max_iterations + 1):
        solver.iterate()
        iterations = t
        if delta > 0 and t % delta == 0:
            k = t / delta
            state.discount(k / (k + 1.0))
        # Optional per-iteration hook, fired AFTER the discount step — so what it observes
        # at ``t`` is byte-identical to what a solve with ``max_iterations == t`` returns.
        # Used by the calibration to snapshot the strategy at every ladder rung from ONE
        # search instead of re-running the same trajectory per rung.  ``None`` ⇒ zero cost.
        if on_iteration is not None:
            on_iteration(t, time.perf_counter() - start)
        if time.perf_counter() - start >= cfg.max_wall_seconds:
            stop_reason = "wall_cap"
            break
    return iterations, stop_reason


# ----------------------------------------------------------------------
# Fork-safety helpers (used by the per-hand pool that forks search workers)
# ----------------------------------------------------------------------


def _limit_worker_threads(n_threads: int = 1) -> None:
    """Pin this forked worker's BLAS/OpenMP thread pools to ``n_threads``.

    The per-hand pool forks one worker per core; numpy's OpenBLAS defaults to one thread
    per core, so ``W`` workers on a ``C``-core box spin up ~``W*C`` threads that busy-wait
    and thrash the scheduler.  Measured impact on this box (22 cores, 8 workers): the
    vector solve ran *slower* than serial and the compiled core's per-iteration win
    inverted into a net loss (core-on became slower than core-off) purely from the
    oversubscription.  Each search is a single fine-grained CFR walk over tiny
    (n_combos,) arrays that never benefits from intra-op BLAS threads, so **one thread
    per worker is optimal**.

    Best-effort and never raises (pinning is an optimisation, not correctness): sets the
    standard env vars (for any pool that re-reads them) AND calls OpenBLAS's runtime
    setter on numpy's bundled library — the reliable path in a forked child, whose BLAS
    pool is already initialised so the env var alone may be ignored.
    """
    import os as _os

    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ[_var] = str(n_threads)
    try:
        import ctypes
        import glob
        import numpy as _np

        libdir = _os.path.join(_os.path.dirname(_np.__file__), ".libs")
        for _so in glob.glob(_os.path.join(libdir, "libopenblas*.so")):
            try:
                _lib = ctypes.CDLL(_so)
            except OSError:
                continue
            if hasattr(_lib, "openblas_set_num_threads"):
                _lib.openblas_set_num_threads(int(n_threads))
    except Exception:
        pass  # best-effort; a missing/renamed BLAS must never break the solve


def _reopen_leaf_fleet_lmdb(ctx: SubgameContext) -> None:
    """Reopen the leaf fleet's LMDB-backed blueprint envs after a ``fork`` (§6.7).

    LMDB's reader-lock table is a **process-shared mmap** (``lock.mdb``), so a ``fork``
    is not reader-safe: a child that touches the inherited env clobbers the reader slot
    that belongs to the *parent's* thread, and the next read on that slot — in *either*
    process — trips ``mdb_txn_renew: MDB_BAD_RSLOT``.  Reopening gets a fresh env handle
    bound to a clean slot.  Called by each forked worker before its first leaf query (the
    per-hand pool) and by the calibration sweep's parent after its pool joins.

    Policies without an LMDB backend (the in-memory ``UniformPolicy`` in tests, a
    ``SearchPolicy``) expose no ``reopen_after_fork`` and are skipped; the four §4 bias
    variants share one blueprint object, so it is reopened once (deduped by id).

    **Covers ``ctx.models`` too** (opponent_modeling §5.1): a modeled seat's ``σ̂`` is
    typically a *blueprint-backed* policy, and the clamp queries it inside the walk — i.e.
    inside the forked worker.  An opponent model left out of this sweep would trip the
    very same ``MDB_BAD_RSLOT`` on its first query.  The dedup set is shared, so a
    blueprint reached through both the leaf fleet and a model is reopened exactly once.
    """
    seen: set = set()

    def _reopen(obj) -> None:
        if obj is None or id(obj) in seen:
            return
        reopen = getattr(obj, "reopen_after_fork", None)
        if reopen is not None:
            seen.add(id(obj))
            reopen()

    for policy in ctx.leaf.policies.values():
        _reopen(policy)
    for model in getattr(ctx, "models", {}).values():
        _reopen(model)
