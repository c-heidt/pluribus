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
    single-worker path stays bit-for-bit identical.  The binding ``max_iterations``
    is the **structural iteration budget** :func:`solve` computed for this subgame
    (:mod:`poker_ai.search.budget`) — a real subgame is too large for any single
    replica to reach a tight equilibrium online, so the stop is a machine-independent
    per-subgame iteration count, not an online convergence test.  ``stop_reason`` is
    which cap ended the loop (eval doc §6 ``decisions.stop_reason``): ``'wall_cap'``
    if the wall-clock backstop broke early, else ``'iteration_cap'`` (the loop ran the
    full budget, including the degenerate 0-iteration case).
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


def _limit_worker_threads(n_threads: int = 1) -> None:
    """Pin this forked replica's BLAS/OpenMP thread pools to ``n_threads``.

    Parallel search forks ``W`` replicas; numpy's OpenBLAS defaults to one thread
    per core, so ``W`` replicas on a ``C``-core box spin up ~``W*C`` threads that
    busy-wait and thrash the scheduler.  Measured impact on this box (22 cores, 8
    workers): the vector solve ran *slower* than serial and the compiled core's
    per-iteration win inverted into a net loss (core-on became slower than
    core-off) purely from the oversubscription.  Each replica is a single
    fine-grained CFR walk over tiny (n_combos,) arrays that never benefits from
    intra-op BLAS threads, so **one thread per replica is optimal**.

    Best-effort and never raises (pinning is an optimisation, not correctness):
    sets the standard env vars (for any pool that re-reads them) AND calls
    OpenBLAS's runtime setter on numpy's bundled library — the reliable path in a
    forked child, whose BLAS pool is already initialised so the env var alone may
    be ignored.
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


def _build_solver(root_env, state, ctx, cfg, rng, regime):
    if regime == "vector":
        return _VectorSolver(root_env, state, ctx, cfg, rng)
    return _MCCFRSolver(root_env, state, ctx, cfg, rng)


def _reopen_leaf_fleet_lmdb(ctx: SubgameContext) -> None:
    """Reopen the leaf fleet's LMDB-backed blueprint envs after a ``fork`` (§6.7).

    LMDB's reader-lock table is a **process-shared mmap** (``lock.mdb``), so a
    ``fork`` is not reader-safe: a child that touches the inherited env clobbers the
    reader slot that belongs to the *parent's* thread, and the next read on that slot
    — in *either* process — trips ``mdb_txn_renew: MDB_BAD_RSLOT``.  Reopening gets a
    fresh env handle bound to a clean slot.

    Called in **two** places, because both sides need repair:

    - each forked **replica**, before its first leaf query (:func:`_run_replica`);
    - the **parent**, right after the pool joins (:func:`run_parallel`), so its own
      post-search reads — the eval's opponent/hero blueprint + belief lookups all go
      through this same shared blueprint — don't hit ``MDB_BAD_RSLOT``.  This was the
      long-standing bug: the child repair existed, the parent repair did not.

    Policies without an LMDB backend (the in-memory ``UniformPolicy`` in tests, a
    ``SearchPolicy``) expose no ``reopen_after_fork`` and are skipped; the four §4
    bias variants share one blueprint object, so it is reopened once (deduped by id).
    """
    seen: set = set()
    for policy in ctx.leaf.policies.values():
        reopen = getattr(policy, "reopen_after_fork", None)
        if reopen is not None and id(policy) not in seen:
            seen.add(id(policy))
            reopen()


def _run_replica(payload: Tuple[int, np.random.SeedSequence, int]):
    """Run one replica from the fork-inherited :data:`_SHARED` inputs."""
    _idx, seed_seq, start_offset = payload
    root_env = _SHARED["root_env"]
    ctx: SubgameContext = _SHARED["ctx"]
    cfg: SolverConfig = _SHARED["cfg"]
    warm: Optional[SolverState] = _SHARED["warm"]
    regime: str = _SHARED["regime"]

    # Pin this replica to a single BLAS thread — W replicas each spinning a
    # per-core OpenBLAS pool oversubscribe the box and thrash (see the docstring).
    _limit_worker_threads(1)
    # Reopen fork-inherited blueprint LMDB envs before any leaf query (MDB_BAD_RSLOT).
    _reopen_leaf_fleet_lmdb(ctx)

    rng = np.random.default_rng(seed_seq)
    wctx = dataclasses.replace(ctx, rng=rng)
    state = copy.deepcopy(warm) if warm is not None else SolverState.empty()
    if warm is not None:
        # Each replica deep-copies the warm baseline, so it also inherits the
        # baseline's cumulative walk/cache counters; zero them so the per-replica
        # snapshot (summed in ``run_parallel``) counts only this re-search's work —
        # otherwise the baseline counters are multiplied by the replica count (§9.1).
        state.reset_counters()
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
        # Parent-side reader-slot repair: the forked pool clobbered this process's
        # slot in the shared LMDB reader table, so the parent's own subsequent reads
        # (the eval opponents / hero blueprint / belief lookups, all on this same
        # shared blueprint) would trip MDB_BAD_RSLOT.  Reopen unconditionally — it is
        # idempotent and cheap (read-only mmap remap), and must run even if a replica
        # raised so a failed search never leaves the parent env poisoned.
        _reopen_leaf_fleet_lmdb(ctx)
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
