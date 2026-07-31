"""Single-core work pool for evaluation + calibration (per-hand parallelism).

The eval and the calibration sweep are both "run many **independent single-core**
jobs across the box": a job is a *hand* (eval) or one *sweep solve* (calibration).
Because each job is fully determined by an integer index (``hand_index`` for eval, a
list position for the sweep) and reads only shared, read-only state (the blueprint),
they fan out trivially — and because the search inside runs ``workers=1`` there is
**no nested pool** (Pool workers are daemonic and cannot fork children; W=1 sidesteps
that, and gives *deeper* undivided MCCFR as a bonus).

:func:`run_index_pool` is the one primitive.  It hands out a **monotonic index** under
a lock (dynamic pull — effort per job varies wildly, e.g. a folded hand vs a minutes-
long turn solve, so static sharding would leave cores idle), skipping any index in a
``skip`` set (completed-set **resume**), until an optional ``target`` count, a shared
**wall budget**, or a **stop** event ends it.  Heavy shared state is passed by
**fork inheritance** (a module global set just before the pool is created), exactly as
:mod:`poker_ai.search.parallel` does, so only a tiny worker id is pickled.

The caller supplies three hooks (module-level or fork-inherited closures — never
pickled, since we fork):

- ``setup(worker_id, shared) -> worker_state`` — run once per worker *after* the fork.
  Reopen any fork-inherited LMDB handles here (the ``MDB_BAD_RSLOT`` pitfall) and open
  this worker's node-local artifacts (e.g. its own DB with a disjoint id block).
- ``process(index, worker_state, shared) -> None`` — do job ``index`` (mutating
  ``worker_state``: append a row, write a DB, bump a counter).
- ``teardown(worker_state) -> payload`` — return this worker's picklable result (row
  list, counts, DB path); the pool returns the list of the H payloads.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from typing import Any, Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Fork-inherited shared context (set by run_index_pool in the parent BEFORE the pool is
# created; children inherit it copy-on-write).  Mirrors search.parallel._SHARED so the
# heavy blueprint/session is never pickled per task.
_POOL_SHARED: dict = {}


def _worker(worker_id: int):
    """Run one worker: setup → pull-and-process loop → teardown; from _POOL_SHARED."""
    S = _POOL_SHARED
    # Pin BLAS to one thread — H single-core workers each spinning a per-core pool would
    # oversubscribe exactly as the search replicas do (search.parallel._limit_worker_threads).
    try:
        from poker_ai.search.parallel import _limit_worker_threads
        _limit_worker_threads(1)
    except Exception:  # pragma: no cover - pinning is an optimisation, never correctness
        pass

    setup: Callable = S["setup"]
    process: Callable = S["process"]
    teardown: Optional[Callable] = S["teardown"]
    shared: dict = S["shared"]
    counter = S["counter"]
    lock = S["lock"]
    target: Optional[int] = S["target"]
    skip = S["skip"]
    wall_budget_s: float = S["wall_budget_s"]
    run_start: float = S["run_start"]
    stop_event = S["stop_event"]

    state = setup(worker_id, shared)
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        if wall_budget_s > 0.0 and (time.monotonic() - run_start) >= wall_budget_s:
            break
        with lock:                      # atomic hand-out of the next index
            idx = counter.value
            counter.value = idx + 1
        if target is not None and idx >= target:
            break
        if skip and idx in skip:
            continue                    # already completed in a prior attempt (resume)
        try:
            process(idx, state, shared)
        except Exception:               # a bad job must not kill the worker (job logs itself)
            logger.exception("pool worker %d: job index %d raised", worker_id, idx)
    return teardown(state) if teardown is not None else None


def run_index_pool(
    *,
    n_workers: int,
    setup: Callable[[int, dict], Any],
    process: Callable[[int, Any, dict], None],
    teardown: Optional[Callable[[Any], Any]] = None,
    shared: Optional[dict] = None,
    target: Optional[int] = None,
    skip: Optional[Sequence[int]] = None,
    wall_budget_s: float = 0.0,
    stop_event=None,
) -> List[Any]:
    """Fan an index-keyed job over ``n_workers`` single-core forked workers.

    Dynamic pull of a monotonic index (lock-guarded), skipping ``skip`` (resume),
    until ``target`` jobs (``None`` ⇒ unbounded, wall/stop-bound), ``wall_budget_s``
    (``0`` ⇒ none), or ``stop_event`` set.  Returns the list of per-worker ``teardown``
    payloads.  ``n_workers == 1`` runs inline (no fork) for tests / trivial runs.
    """
    n = max(1, int(n_workers))
    shared = shared or {}
    skip_set = frozenset(skip or ())

    if n == 1:
        # Inline path — deterministic, fork-free (unit tests, tiny runs).
        counter = {"v": 0}
        state = setup(0, shared)
        start = time.monotonic()
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if wall_budget_s > 0.0 and (time.monotonic() - start) >= wall_budget_s:
                break
            idx = counter["v"]; counter["v"] = idx + 1
            if target is not None and idx >= target:
                break
            if idx in skip_set:
                continue
            try:
                process(idx, state, shared)
            except Exception:
                logger.exception("inline pool: job index %d raised", idx)
        return [teardown(state) if teardown is not None else None]

    try:
        ctx = mp.get_context("fork")
    except ValueError:  # platform without fork
        ctx = mp.get_context()
    global _POOL_SHARED
    _POOL_SHARED = {
        "setup": setup, "process": process, "teardown": teardown, "shared": shared,
        "counter": ctx.Value("q", 0), "lock": ctx.Lock(),
        "target": target, "skip": skip_set, "wall_budget_s": float(wall_budget_s),
        "run_start": time.monotonic(), "stop_event": stop_event,
    }
    try:
        with ctx.Pool(processes=n) as pool:
            payloads = pool.map(_worker, range(n))
    finally:
        _POOL_SHARED = {}
    return payloads
