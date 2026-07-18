"""Reusable record/replay differential harness for the search compiled core.

The search-side counterpart of :mod:`test.training.core_diff` (the blueprint
harness).  Phases 3-4 of the search-core plan must make the compiled walk produce
``SolverState`` tables **byte-identical** to the pure-Python ``_MCCFRSolver`` /
``_VectorSolver``.  As in the blueprint core, byte-equality is proven by *removing
the RNG from the comparison* — recording the sampled decisions of a Python walk and
replaying that exact sequence into a second walk (Python now; the compiled core in
Phases 3-4) — never by trying to match numpy's RNG byte-stream (the core runs its
own sampler, exactly as ``_traverse.pyx`` declined RNG parity for blueprint).

Two regimes, two nondeterminism profiles:

- **MCCFR** (``mccfr.py``) samples opponents in the regret pass and every node in
  the strategy pass via :func:`poker_ai.blueprint.tree_utils.sample_index`.  The
  root-hole draw (``_sample_root_holes``) is *not* replayed — the harness pins a
  fixed ``holes`` dict (the "deal", the analogue of blueprint's fixed ``state``) and
  drives ``_traverse`` / ``_update_strategy`` directly, so only the per-node
  ``sample_index`` draws vary and are what we record.  :class:`RecordingSampler` /
  :class:`ReplaySampler` substitute ``sample_index`` in the ``mccfr`` **and** ``leaf``
  module namespaces.
- **Vector** (``vector.py``) samples **nothing** inside the walk — ``_walk`` is
  full-width (every action expanded).  The only draw is the per-iteration board
  ``_completion`` chosen in ``iterate()``.  So a vector pass is *fully deterministic*
  given the completion: :func:`run_vector_pass` pins it and the byte-exact gate is
  simply ``core(completion) == python(completion)`` (no sampler needed).

This module is import-only (no ``test_`` prefix) so pytest does not collect it;
the self-tests in ``functional/test_core_diff_harness.py`` and later phases import
its helpers.
"""

from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

import poker_ai.search.leaf as leaf_mod
import poker_ai.search.mccfr as mccfr_mod

# ``sample_index(rng, weights) -> int``: an inverse-CDF index draw over a strategy
# row.  The harness substitutes it with the recorders below; the real one lives in
# ``poker_ai.blueprint.tree_utils`` and is imported into both search modules.
Snapshot = Dict[object, np.ndarray]


class RecordingSampler:
    """Drop-in for ``sample_index`` that draws from a private RNG and records.

    Reproduces ``sample_index``'s inverse-CDF draw but forces it through a
    caller-owned :class:`numpy.random.RandomState`, so the recording is
    reproducible and never perturbs global numpy state.  Records ``(index, width)``
    per call in DFS call order; the ``width`` (the strategy row length at the node)
    is recorded so replay can *detect divergence* — if the driven walk reaches a
    node of a different width the two walks took different trees.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.RandomState(seed)
        self.choices: List[Tuple[int, int]] = []

    def __call__(self, rng, weights) -> int:
        w = np.asarray(weights, dtype=np.float64)
        total = w.sum()
        if total > 0:
            probs = w / total
        else:
            probs = np.full(len(w), 1.0 / len(w))
        threshold = self._rng.random_sample()
        cumulative = 0.0
        idx = len(w) - 1
        for i in range(len(w)):
            cumulative += probs[i]
            if threshold < cumulative:
                idx = i
                break
        self.choices.append((idx, len(w)))
        return idx


class ReplaySampler:
    """Drop-in for ``sample_index`` that replays a recorded decision sequence.

    Pops the next recorded ``(index, width)`` per node.  Asserts the node width
    matches (a mismatch means the driven walk diverged — the whole point of the
    harness) and that the sequence is neither over- nor under-consumed.
    """

    def __init__(self, choices: List[Tuple[int, int]]) -> None:
        self._choices = list(choices)
        self._i = 0

    def __call__(self, rng, weights) -> int:
        assert self._i < len(self._choices), (
            "replay exhausted: the driven walk visited more sampled nodes than "
            "were recorded — the two walks diverged"
        )
        idx, width = self._choices[self._i]
        self._i += 1
        assert len(weights) == width, (
            f"replayed node width {len(weights)} != recorded {width} — the two "
            "walks diverged"
        )
        return idx

    def exhausted(self) -> bool:
        """True iff every recorded choice was consumed (no under-consumption)."""
        return self._i == len(self._choices)


def snapshot(table: Dict) -> Snapshot:
    """Deep-copy a ``SolverState`` table dict (``regret`` / ``strat_sum`` / etc.).

    Returns ``{key: float64 ndarray copy}`` so the caller holds an immutable
    fingerprint of the table after a pass, decoupled from further mutation.
    """
    return {k: np.array(v, copy=True) for k, v in table.items()}


def assert_tables_equal(a: Snapshot, b: Snapshot) -> None:
    """Assert two snapshotted ``SolverState`` tables are byte-identical.

    Same keys, and per key the float64 arrays equal element-for-element.  This is
    the acceptance predicate Phases 3-4 use to certify the compiled walk against
    the Python reference.
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


def _run_with_sampler(solver, env, i, holes, sampler, *, strategy: bool):
    """Patch ``sample_index`` with ``sampler`` and run one MCCFR pass on ``solver``.

    ``strategy`` selects the strategy pass (``_update_strategy`` → ``strat_sum``)
    vs the regret pass (``_traverse`` → ``regret``).  ``env`` is deep-copied so the
    caller's root is untouched; the pass mutates ``solver.state`` in place.
    """
    orig_m, orig_l = mccfr_mod.sample_index, leaf_mod.sample_index
    mccfr_mod.sample_index = sampler
    leaf_mod.sample_index = sampler
    try:
        if strategy:
            solver._update_strategy(deepcopy(env), i, holes)
        else:
            solver._traverse(deepcopy(env), i, holes)
    finally:
        mccfr_mod.sample_index = orig_m
        leaf_mod.sample_index = orig_l


def run_traverse_recording(solver, env, i, holes, *, seed: int = 0):
    """Record one regret-pass (``_traverse``) walk with a :class:`RecordingSampler`.

    Returns ``(regret_snapshot, choices)``.  ``solver`` is built by the caller on a
    (copied) baseline state; the snapshot is of ``solver.state.regret`` after the
    pass.
    """
    sampler = RecordingSampler(seed=seed)
    _run_with_sampler(solver, env, i, holes, sampler, strategy=False)
    return snapshot(solver.state.regret), sampler.choices


def run_traverse_replay(solver, env, i, holes, choices):
    """Replay a recorded regret-pass sequence; assert full consumption.

    Returns the ``regret`` snapshot after the driven pass.
    """
    sampler = ReplaySampler(choices)
    _run_with_sampler(solver, env, i, holes, sampler, strategy=False)
    assert sampler.exhausted(), (
        "replay under-consumed: the driven regret walk visited fewer sampled "
        "nodes than were recorded — the two walks diverged"
    )
    return snapshot(solver.state.regret)


def run_strategy_recording(solver, env, i, holes, *, seed: int = 0):
    """Record one strategy-pass (``_update_strategy``) walk.

    Returns ``(strat_sum_snapshot, choices)``.  Unlike the regret pass (opponent
    nodes only), the strategy walk samples at **every** node, so ``choices`` spans
    the traverser's and opponents' nodes alike.
    """
    sampler = RecordingSampler(seed=seed)
    _run_with_sampler(solver, env, i, holes, sampler, strategy=True)
    return snapshot(solver.state.strat_sum), sampler.choices


def run_strategy_replay(solver, env, i, holes, choices):
    """Replay a recorded strategy-pass sequence; assert full consumption."""
    sampler = ReplaySampler(choices)
    _run_with_sampler(solver, env, i, holes, sampler, strategy=True)
    assert sampler.exhausted(), (
        "replay under-consumed: the driven strategy walk visited fewer sampled "
        "nodes than were recorded — the two walks diverged"
    )
    return snapshot(solver.state.strat_sum)


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
        completion = tuple(int(solver._avail[i]) for i in range(solver._n_completion))
    solver._completion = tuple(completion)
    if solver._n_completion:
        solver._refresh_cluster_maps(solver._completion)
    else:
        solver._feas_full = None
    s0, s1 = solver._seats
    # Walk whatever env the solver was configured with — the ``PokerEnv`` root by
    # default, or the compiled ``FastEnvAdapter`` under PLURIBUS_SEARCH_CORE
    # (Phase 3c); the two must produce byte-identical tables.
    walk_env = getattr(solver, "_walk_env", solver.root_env)
    solver._walk(walk_env, s0, solver._reach[s0], solver._reach[s1])
    solver._walk(walk_env, s1, solver._reach[s1], solver._reach[s0])
    return snapshot(solver.state.vregret), snapshot(solver.state.vstrat)
