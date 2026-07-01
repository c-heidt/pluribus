"""Parallel search: independent MCCFR/vector replicas merged once (§6.7 row 11).

The real-time budget (10⁴ iterations **and** 15 s) is a wall-clock budget, so the
way to spend more compute inside it is to run several searches at once and pool
their results.  This module implements the **independent-replica** scheme:

- ``W`` worker *processes* (not threads — pure-Python traversal does not scale on
  threads under the GIL; the ``numba nogil`` route is the deferred row 10, and the
  doc calls for ``multiprocessing`` first) each run the **full** CFR loop on their
  own :class:`~poker_ai.search.solver_state.SolverState` replica, seeded from an
  independent ``SeedSequence`` substream.
- For MCCFR every replica rotates the traversing **player** (acting-order position,
  not seat identity) through all live players each iteration; the workers differ
  only in their seeded substream and a staggered starting offset (so coverage is
  balanced for short budgets).  For the vector regime each replica samples its own
  river substream (one-board-per-worker, §6.7).
- At the **end** of the budget the replicas sync once and
  :meth:`SolverState.accumulate` sums their regrets and strategy sums into one
  merged state.

Determinism (§6.7): ``solve`` keeps the serial loop for ``workers == 1`` so that
path is bit-for-bit unchanged; a multi-worker run fixes per-worker substreams from
``SeedSequence`` so a given ``(seed, n_workers)`` is reproducible (it is *not*
identical to the serial result — the sampling trajectory differs).

Heavy shared inputs (the root env and its card-info LUT, the context) are handed to
workers by **fork inheritance** (copy-on-write) via a module global set just before
the pool is created, so only the tiny per-worker seed/offset is pickled.
"""

from __future__ import annotations

import copy
import dataclasses
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from poker_ai.search.context import SubgameContext
from poker_ai.search.mccfr import _MCCFRSolver
from poker_ai.search.solver_state import SearchStats, SolverConfig, SolverState
from poker_ai.search.vector import _VectorSolver


# ----------------------------------------------------------------------
# Worker budget + plan
# ----------------------------------------------------------------------


def resolve_workers(workers: "int | None") -> int:
    """Resolve ``SolverConfig.workers`` to a concrete count.

    ``None`` → a cpu-based default (SLURM-aware, leaving one core free, mirroring
    :mod:`poker_ai.blueprint.multiprocess`).  Any explicit value is clamped to ≥ 1.
    """
    if workers is not None:
        return max(1, int(workers))
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        n = int(slurm) if slurm else (os.cpu_count() or 1)
    except (TypeError, ValueError):
        n = os.cpu_count() or 1
    return max(1, n - 1)


@dataclass(frozen=True)
class WorkerPlan:
    """How ``n_workers`` replicas split the work (§6.7 row 11).

    ``seeds[k]`` is replica ``k``'s independent ``SeedSequence`` substream;
    ``offsets[k]`` is its starting traverser offset (``k mod n_live_players``), so
    the ensemble balances the traverser rotation across the live players even when
    the per-replica budget is short.  Every replica still rotates through **all**
    players — the offset only staggers the phase; pinning a replica to one
    traverser would leave the other players' regrets at uniform σ.
    """

    n_workers: int
    seeds: Tuple[np.random.SeedSequence, ...]
    offsets: Tuple[int, ...]


def plan_workers(
    total_workers: int, n_live_players: int, *, base_seed: int
) -> WorkerPlan:
    """Build the :class:`WorkerPlan` for ``total_workers`` replicas.

    ``base_seed`` seeds the root ``SeedSequence`` whose ``spawn`` gives the
    per-replica substreams, so ``(base_seed, total_workers)`` is reproducible.
    """
    n = max(1, int(total_workers))
    n_live = max(1, int(n_live_players))
    seeds = tuple(np.random.SeedSequence(base_seed).spawn(n))
    offsets = tuple(k % n_live for k in range(n))
    return WorkerPlan(n_workers=n, seeds=seeds, offsets=offsets)


# ----------------------------------------------------------------------
# Shared iteration loop (used by the serial path and every replica)
# ----------------------------------------------------------------------


def run_loop(solver, state: SolverState, cfg: SolverConfig) -> Tuple[int, str]:
    """Drive ``solver`` under the dual stop; return ``(iterations, stop_reason)``.

    This *is* the orchestrator's serial loop (Linear-CFR discount on the
    ``discount_interval`` cadence, stop on ``max_iterations`` **or**
    ``max_wall_seconds``); :func:`solve` and every replica share it so the
    single-worker path stays bit-for-bit identical.  ``stop_reason`` is which of
    the two budget caps ended the loop (eval doc §6 ``decisions.stop_reason``):
    ``'wall_cap'`` if the wall-clock check broke early, else ``'iteration_cap'``
    (the loop ran the full ``max_iterations``, including the degenerate 0-iteration
    case).
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
        if time.perf_counter() - start >= cfg.max_wall_seconds:
            stop_reason = "wall_cap"
            break
    return iterations, stop_reason


# ----------------------------------------------------------------------
# Worker entry (fork-inherited shared inputs)
# ----------------------------------------------------------------------

# Set by run_parallel in the parent *before* the pool is forked; the children
# inherit it copy-on-write, so the root env + LUT and the context are never pickled
# per task.  Keyed: root_env, ctx, cfg, warm, regime.
_SHARED: dict = {}


def _build_solver(root_env, state, ctx, cfg, rng, regime):
    if regime == "vector":
        return _VectorSolver(root_env, state, ctx, cfg, rng)
    return _MCCFRSolver(root_env, state, ctx, cfg, rng)


def _run_replica(payload: Tuple[int, np.random.SeedSequence, int]):
    """Run one replica from the fork-inherited :data:`_SHARED` inputs."""
    _idx, seed_seq, start_offset = payload
    root_env = _SHARED["root_env"]
    ctx: SubgameContext = _SHARED["ctx"]
    cfg: SolverConfig = _SHARED["cfg"]
    warm: Optional[SolverState] = _SHARED["warm"]
    regime: str = _SHARED["regime"]

    rng = np.random.default_rng(seed_seq)
    wctx = dataclasses.replace(ctx, rng=rng)
    state = copy.deepcopy(warm) if warm is not None else SolverState.empty()
    solver = _build_solver(root_env, state, wctx, cfg, rng, regime)
    # Stagger the MCCFR traverser rotation; the vector regime has no per-iteration
    # traverser (it samples a river instead), so the offset is inert there.
    if regime != "vector":
        solver._iter = int(start_offset)
    iterations, stop_reason = run_loop(solver, state, cfg)
    return state, iterations, stop_reason


def run_parallel(
    root_env,
    ctx: SubgameContext,
    cfg: SolverConfig,
    warm_start: Optional[SolverState],
    plan: WorkerPlan,
    regime: str,
) -> Tuple[SolverState, int, float, str, SearchStats]:
    """Run ``plan.n_workers`` replicas and merge them once.

    Returns ``(merged_state, iterations_total, wall_seconds, stop_reason, stats)``.
    ``stop_reason`` is ``'wall_cap'`` if *any* replica hit the wall budget (they
    share one wall budget and run concurrently, so they broadly agree), else
    ``'iteration_cap'``.  ``stats`` sums the per-replica walk/cache counters (eval
    doc §9.1); ``unique_pubkeys`` is taken from the *merged* ``legal_at`` (the true
    distinct-key count — replicas walk the same tree) rather than summed.
    """
    global _SHARED
    _SHARED = {
        "root_env": root_env,
        "ctx": ctx,
        "cfg": cfg,
        "warm": warm_start,
        "regime": regime,
    }
    try:
        mp_ctx = mp.get_context("fork")
    except ValueError:  # platform without fork — fall back (pickles shared inputs)
        mp_ctx = mp.get_context()

    payloads = [
        (k, plan.seeds[k], plan.offsets[k]) for k in range(plan.n_workers)
    ]
    start = time.perf_counter()
    try:
        with mp_ctx.Pool(processes=plan.n_workers) as pool:
            results: List[Tuple[SolverState, int, str]] = pool.map(
                _run_replica, payloads
            )
    finally:
        _SHARED = {}
    wall = time.perf_counter() - start

    states = [st for st, _, _ in results]
    iterations_total = sum(n for _, n, _ in results)
    stop_reason = (
        "wall_cap"
        if any(sr == "wall_cap" for _, _, sr in results)
        else "iteration_cap"
    )
    merged = SolverState.accumulate(states, baseline=warm_start)
    # Aggregate the per-replica counters; take distinct-key count from the merged
    # tree (summing replicas would double-count the shared nodes).
    stats = SearchStats()
    for st in states:
        stats = stats.combined_with(st.stats_snapshot())
    stats.unique_pubkeys = len(merged.legal_at)
    return merged, iterations_total, wall, stop_reason, stats
