"""Reusable record/replay differential harness for the compiled-core rewrite.

The Tier-1 rewrite must make the compiled ``_traverse`` produce a ``local_delta``
**byte-identical** to the pure-Python ``poker_ai.blueprint.cfr._traverse``.  The
only nondeterminism in a traversal is opponent (external-sampling) action
selection via ``sample_action``; everything else — the traverser's full-width
branching, regret matching, env step/undo, and terminal settlement — is a
deterministic function of the dealt hand and the (read-only) regret snapshot.

This harness removes the RNG from the comparison instead of trying to match it:

1. :class:`RecordingSampler` runs a Python traversal with its own seeded RNG and
   **records** every opponent action it chooses, in depth-first traversal order.
2. :class:`ReplaySampler` feeds that exact sequence back into a second traversal,
   so both walks explore the identical tree with zero RNG involved.

Because the traverser explores every legal action deterministically and the
opponent choices are replayed verbatim in the same DFS order, two traversals
driven by the same recorded sequence must yield identical ``local_delta`` dicts.
Phase 3 reuses this by replaying a Python-recorded sequence into the compiled
core and asserting :func:`assert_local_delta_equal`.

This module is import-only (no ``test_`` prefix) so pytest does not collect it;
tests and later phases import its helpers.
"""

from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

import poker_ai.blueprint.cfr as cfr_mod


class RecordingSampler:
    """Drop-in for ``sample_action`` that samples with a private RNG and records.

    Reimplements :func:`poker_ai.blueprint.tree_utils.sample_action`'s inverse-CDF
    draw (it does **not** call it) but forces it through a caller-owned
    :class:`numpy.random.RandomState` — so the recording is reproducible and never
    perturbs global ``numpy`` state — and appends each chosen action string to
    :attr:`choices` in call order, which for a depth-first traversal is the order
    a replay consumes them.  The reimplementation is only for *reproducibility*:
    replay reproduces ``local_delta`` from the recorded choices regardless of how
    they were drawn, so it need not track ``sample_action``'s exact distribution.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.RandomState(seed)
        self.choices: List[str] = []

    def __call__(self, legal_actions, sigma, a_to_i) -> str:
        # Reproduce sample_action's inverse-CDF draw, but from our own RNG.
        probs = np.array([sigma[a_to_i[a]] for a in legal_actions], dtype=np.float64)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs[:] = 1.0 / len(legal_actions)
        threshold = self._rng.random_sample()
        cumulative = 0.0
        chosen = legal_actions[-1]
        for i in range(len(legal_actions)):
            cumulative += probs[i]
            if threshold < cumulative:
                chosen = legal_actions[i]
                break
        self.choices.append(chosen)
        return chosen


class ReplaySampler:
    """Drop-in for ``sample_action`` that replays a recorded opponent sequence.

    Pops the next recorded action per opponent node.  Asserts the replayed
    action is legal at the node (a mismatch means the two walks diverged — the
    whole point of the harness is to catch exactly that) and that the sequence
    is neither over- nor under-consumed.
    """

    def __init__(self, choices: List[str]) -> None:
        self._choices = list(choices)
        self._i = 0

    def __call__(self, legal_actions, sigma, a_to_i) -> str:
        assert self._i < len(self._choices), (
            "replay exhausted: the driven traversal visited more opponent nodes "
            "than were recorded — the two walks diverged"
        )
        action = self._choices[self._i]
        self._i += 1
        assert action in legal_actions, (
            f"replayed action {action!r} illegal at node with legal actions "
            f"{legal_actions!r} — the two walks diverged"
        )
        return action

    def exhausted(self) -> bool:
        """True iff every recorded choice was consumed (no under-consumption)."""
        return self._i == len(self._choices)


def build_trained_tables(save_path, card_info_lut, *, n_players: int = 2,
                         n_iterations: int = 40, seed: int = 123):
    """Build a fresh ``CFRTables`` lightly pre-trained for non-uniform regrets.

    A degenerate all-uniform fresh table would not exercise regret matching, so
    a short seeded run populates realistic regrets; the seed makes the resulting
    snapshot identical every run (kept read-only afterward via explicit
    ``local_delta`` in the record/replay runs).  The caller owns closing the
    returned tables.  Shared by the harness self-tests and the Phase-1 kernel
    end-to-end tests so both train against the same distribution.
    """
    from pathlib import Path

    from environment.action_space import MAX_ACTIONS_PER_STREET
    from environment.poker_env import new_game
    from poker_ai.tables.cfr_tables import CFRTables
    from poker_ai.tables.index import lmdb_map_size_for_players

    shm_dir = Path(save_path) / "shm"
    shm_dir.mkdir(parents=True, exist_ok=True)
    tables = CFRTables(
        index_path=Path(save_path) / "lmdb_index",
        shm_dir=str(shm_dir),
        actions_per_street=MAX_ACTIONS_PER_STREET,
        lmdb_map_size=lmdb_map_size_for_players(n_players),
    )
    np.random.seed(seed)
    for t in range(1, n_iterations + 1):
        for i in range(n_players):
            cfr_mod.cfr(tables, new_game(n_players, card_info_lut), i, t)
    return tables


def run_recording(tables, state, i: int, t: int, *, seed: int = 0):
    """Run one Python traversal with a :class:`RecordingSampler`.

    Returns ``(local_delta, choices)``.  ``state`` is deep-copied so the caller's
    root is untouched; ``local_delta`` is passed in explicitly so ``cfr`` does
    **not** merge into ``tables`` (the tables stay a read-only snapshot shared by
    the recording and replay runs).
    """
    sampler = RecordingSampler(seed=seed)
    local_delta: Dict[Tuple[int, bytes], np.ndarray] = {}
    original = cfr_mod.sample_action
    cfr_mod.sample_action = sampler
    try:
        cfr_mod.cfr(tables, deepcopy(state), i, t, local_delta=local_delta)
    finally:
        cfr_mod.sample_action = original
    return local_delta, sampler.choices


def run_replay(tables, state, i: int, t: int, choices: List[str]):
    """Run one Python traversal driven by a recorded opponent sequence.

    Returns ``local_delta``.  Asserts the replay is fully consumed (the driven
    walk visited exactly the recorded opponent nodes).
    """
    sampler = ReplaySampler(choices)
    local_delta: Dict[Tuple[int, bytes], np.ndarray] = {}
    original = cfr_mod.sample_action
    cfr_mod.sample_action = sampler
    try:
        cfr_mod.cfr(tables, deepcopy(state), i, t, local_delta=local_delta)
    finally:
        cfr_mod.sample_action = original
    assert sampler.exhausted(), (
        "replay under-consumed: the driven traversal visited fewer opponent "
        "nodes than were recorded — the two walks diverged"
    )
    return local_delta


def run_recording_strategy(tables, state, i: int, *, seed: int = 0):
    """Run one Python ``update_strategy`` walk with a :class:`RecordingSampler`.

    The strategy-walk counterpart of :func:`run_recording`.  Returns
    ``(local_delta, choices)`` where ``local_delta`` is the visit-count
    accumulator ``update_strategy`` fills (``(0, info_set_bytes) -> int64``)
    and ``choices`` is every sampled action in DFS order.  The pre-flop
    UPDATE-STRATEGY pass samples at **traverser** nodes only (opponent nodes
    branch deterministically over every legal action), so ``choices`` spans
    player nodes only — the mirror of the cfr recording (opponent nodes only).
    ``state`` is deep-copied so the caller's root is untouched; passing an
    explicit ``local_delta`` keeps ``tables.strategy`` a read-only snapshot.
    """
    import poker_ai.blueprint.strategy as strat_mod

    sampler = RecordingSampler(seed=seed)
    local_delta: Dict[Tuple[int, bytes], np.ndarray] = {}
    original = strat_mod.sample_action
    strat_mod.sample_action = sampler
    try:
        strat_mod.update_strategy(tables, deepcopy(state), i, local_delta=local_delta)
    finally:
        strat_mod.sample_action = original
    return local_delta, sampler.choices


def run_replay_strategy(tables, state, i: int, choices: List[str]):
    """Run one Python ``update_strategy`` walk driven by a recorded sequence.

    The strategy-walk counterpart of :func:`run_replay`.  Returns the visit-count
    ``local_delta``.  Asserts the replay is fully consumed (the driven walk
    visited exactly the recorded nodes).
    """
    import poker_ai.blueprint.strategy as strat_mod

    sampler = ReplaySampler(choices)
    local_delta: Dict[Tuple[int, bytes], np.ndarray] = {}
    original = strat_mod.sample_action
    strat_mod.sample_action = sampler
    try:
        strat_mod.update_strategy(tables, deepcopy(state), i, local_delta=local_delta)
    finally:
        strat_mod.sample_action = original
    assert sampler.exhausted(), (
        "replay under-consumed: the driven strategy walk visited fewer nodes "
        "than were recorded — the two walks diverged"
    )
    return local_delta


def assert_local_delta_equal(a: Dict, b: Dict) -> None:
    """Assert two ``local_delta`` dicts are byte-identical.

    Same keys ``(betting_round, info_set_bytes)``, and per key the ``int64``
    delta arrays equal element-for-element.  This is the acceptance predicate
    Phase 3 uses to certify the compiled core against the Python reference.
    """
    ka, kb = set(a), set(b)
    assert ka == kb, (
        f"local_delta key sets differ: only-in-A={sorted(ka - kb)!r}, "
        f"only-in-B={sorted(kb - ka)!r}"
    )
    for key in a:
        va, vb = a[key], b[key]
        assert va.dtype == vb.dtype == np.int64, (
            f"delta dtype for {key!r}: {va.dtype} vs {vb.dtype} (expected int64)"
        )
        assert np.array_equal(va, vb), (
            f"delta mismatch for {key!r}: {va.tolist()} vs {vb.tolist()}"
        )
