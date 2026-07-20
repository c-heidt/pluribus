"""Reusable differential harness for the search compiled core.

The search-side counterpart of :mod:`test.training.core_diff` (the blueprint
harness).  The search-core plan must make the compiled walk produce
``SolverState`` tables **byte-identical** to the pure-Python ``_VectorSolver``.
As in the blueprint core, byte-equality is proven by *removing the RNG from the
comparison* rather than trying to match numpy's byte-stream.

The **vector** regime (``vector.py``) samples **nothing** inside the walk —
``_walk`` is full-width (every action expanded).  The only draw is the
per-iteration board ``_completion`` chosen in ``iterate()``.  So a vector pass is
*fully deterministic* given the completion: :func:`run_vector_pass` pins it and
the byte-exact gate is simply ``core(completion) == python(completion)`` (no
sampler needed).

(The scalar MCCFR record/replay harness retired with the traverser-vectorized
walk — there is no longer a separate regret/strategy two-pass to record; the
vectorized MCCFR walk is validated by the equilibrium oracle and the golden
digest.)

This module is import-only (no ``test_`` prefix) so pytest does not collect it;
the self-tests in ``functional/test_core_diff_harness.py`` and later phases import
its helpers.
"""

from typing import Dict

import numpy as np

Snapshot = Dict[object, np.ndarray]


def snapshot(table: Dict) -> Snapshot:
    """Deep-copy a ``SolverState`` table dict (``vregret`` / ``vstrat`` / etc.).

    Returns ``{key: float64 ndarray copy}`` so the caller holds an immutable
    fingerprint of the table after a pass, decoupled from further mutation.
    """
    return {k: np.array(v, copy=True) for k, v in table.items()}


def assert_tables_equal(a: Snapshot, b: Snapshot) -> None:
    """Assert two snapshotted ``SolverState`` tables are byte-identical.

    Same keys, and per key the float64 arrays equal element-for-element.  This is
    the acceptance predicate used to certify the compiled walk against the Python
    reference.
    """
    ka, kb = set(a), set(b)
    assert ka == kb, (
        f"table key sets differ: only-in-A={sorted(map(repr, ka - kb))!r}, "
        f"only-in-B={sorted(map(repr, kb - ka))!r}"
    )
    for key in a:
        va, vb = a[key], b[key]
        assert va.dtype == vb.dtype == np.float64, (
            f"table dtype for {key!r}: {va.dtype} vs {vb.dtype} (expected float64)"
        )
        assert np.array_equal(va, vb), (
            f"table mismatch for {key!r}: {va.tolist()} vs {vb.tolist()}"
        )


def run_vector_pass(solver, completion=None):
    """Run one vector pass with a pinned board completion; return ``(vregret, vstrat)``.

    ``_walk`` samples nothing, so a pass is fully deterministic given the sampled
    runout ``completion`` (a tuple of cards: ``()`` for a river subgame, one card
    for a turn root, two for a flop root).  ``None`` defaults to the solver's first
    available card(s), a stable deterministic pin.  Reproduces
    ``_VectorSolver.iterate`` with the completion fixed instead of drawn, so the
    env-vs-FastState walk can be compared ``core(completion) == python(completion)``.
    """
    if completion is None:
        completion = tuple(
            int(solver._cmaps.avail[i]) for i in range(solver._n_completion)
        )
    solver._completion = tuple(completion)
    if solver._n_completion:
        solver._cmaps.refresh(solver._completion)
    s0, s1 = solver._seats
    # Walk whatever env the solver was configured with — the ``PokerEnv`` root by
    # default, or the compiled ``FastEnvAdapter`` under PLURIBUS_SEARCH_CORE
    # (Phase 3c); the two must produce byte-identical tables.
    walk_env = getattr(solver, "_walk_env", solver.root_env)
    solver._walk(walk_env, s0, solver._reach[s0], solver._reach[s1])
    solver._walk(walk_env, s1, solver._reach[s1], solver._reach[s0])
    return snapshot(solver.state.vregret), snapshot(solver.state.vstrat)
