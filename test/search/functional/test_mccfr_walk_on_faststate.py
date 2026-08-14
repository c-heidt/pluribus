"""Phase-3 gate: the MCCFR walk on the compiled FastState engine is byte-identical
to the PokerEnv walk.

The traverser-vectorized MCCFR walk moved onto ``FastState`` (make/undo + concrete
terminal settlement in-core, the depth-limit leaf handed a cloned FastState
frontier).  Every RNG-consuming step — hole sampling, opponent action sampling, the
frozen board runout (read off the PokerEnv root), and the FastState leaf rollout —
is identical regardless of the betting engine; only the make/undo engine and the
concrete settlement differ, and both are separately proven byte-identical
(``test_faststate_vector_payout_concrete`` + the vector walk differential).  So the
whole walk must produce **byte-for-byte identical** ``vregret`` / ``vstrat`` tables
whether it walks a ``PokerEnv`` or the ``FastMCCFRAdapter``.

To isolate the *walk engine* (not the leaf), both arms run with the compiled leaf
bound (``PLURIBUS_SEARCH_CORE=1``): arm A forces the PokerEnv walk
(``_use_core=False``), arm B the FastState walk (``_use_core=True``); everything
else — seeds, tables, leaf — is identical.  Covers leaf-containing roots (pre-flop,
multiway flop) and leaf-free roots (turn, river), equal and unequal stacks.
"""

import collections

import numpy as np
import pytest

from poker_ai import _core

pytestmark = pytest.mark.skipif(
    not _core.CORE_AVAILABLE, reason="compiled core extension not built"
)

if _core.CORE_AVAILABLE:
    import poker_ai.search.mccfr as _mccfr_mod
    from environment.player import Player
    from environment.poker_env import PokerEnv
    from poker_ai.search.context import SubgameContext
    from poker_ai.search.leaf import LeafConfig
    from poker_ai.search.leaf_fast import continuation_value_vector_fast
    from poker_ai.search.mccfr import _MCCFRSolver, _BIAS_CLASSES
    from poker_ai.search.solver_state import SolverConfig, SolverState
    from test.search._helpers import UniformPolicy
    from test.search.core_diff import assert_tables_equal, snapshot


@pytest.fixture(autouse=True)
def _bind_fast_leaf(monkeypatch):
    """Isolate the WALK engine: bind the compiled leaf in BOTH arms regardless of
    the ambient ``PLURIBUS_SEARCH_CORE`` (which controls the import-time rebind), so
    the only difference between the arms is PokerEnv-walk vs FastState-walk.  In
    production ``_use_core`` and this rebind are the same flag, so the forced
    ``_use_core=True`` + Python-leaf combination the default env would otherwise
    produce cannot occur there."""
    monkeypatch.setattr(
        _mccfr_mod, "continuation_value_vector", continuation_value_vector_fast
    )


def _stub_lut(env):
    env.card_info_lut = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0)
    )


def _root(target_round, stacks, seed):
    np.random.seed(seed)
    env = PokerEnv(players=[Player(i, s) for i, s in enumerate(stacks)],
                   low_card_rank=11, high_card_rank=14)
    _stub_lut(env)
    g = 0
    while env.betting_round < target_round and not env.is_terminal and g < 60:
        env.step_in_place("call" if "call" in env.legal_actions else "check")
        g += 1
    return env


def _solve(target_round, stacks, seed, use_core, iters=80):
    """Run ``iters`` MCCFR iterations from an identically-seeded solver; return a
    snapshot of ``(vregret, vstrat)``.  ``use_core`` selects the walk engine."""
    n = len(stacks)
    env = _root(target_round, stacks, seed)
    ranges = {s: np.ones(env.n_combos, np.float32) / env.n_combos for s in range(n)}
    leaf = LeafConfig(policies={c: UniformPolicy() for c in _BIAS_CLASSES})
    ctx = SubgameContext.from_runtime(
        env=env, my_seat=0, my_hole=tuple(int(c) for c in env.players[0].cards),
        ranges=ranges, folded_ranges={}, leaf=leaf, rng=np.random.default_rng(7),
    )
    st = SolverState.empty()
    solver = _MCCFRSolver(env, st, ctx, SolverConfig(leaf=leaf), np.random.default_rng(123))
    solver._use_core = use_core
    for _ in range(iters):
        solver.iterate()
    solver.restore_root()
    return snapshot(st.vregret), snapshot(st.vstrat)


@pytest.mark.parametrize("stacks", [(200, 200, 200), (120, 300, 300)])
@pytest.mark.parametrize("target_round,label", [
    (0, "preflop-leaf"),
    (2, "turn-leaf-free"),
    (3, "river-leaf-free"),
])
def test_mccfr_walk_byte_identical(target_round, label, stacks):
    """FastState MCCFR walk == PokerEnv MCCFR walk, byte-for-byte."""
    for seed in range(4):
        reg_py, strat_py = _solve(target_round, stacks, seed, use_core=False)
        reg_core, strat_core = _solve(target_round, stacks, seed, use_core=True)
        assert reg_py, f"{label}: no nodes trained (vacuous)"
        assert_tables_equal(reg_py, reg_core)
        assert_tables_equal(strat_py, strat_core)


def test_multiway_flop_leaf_byte_identical():
    """A multiway FLOP root exercises the after-2nd-raise depth-limit leaf on the
    FastState frontier clone — the hardest leaf handoff — byte-identically."""
    for seed in range(4):
        reg_py, strat_py = _solve(1, (200, 200, 200), seed, use_core=False)
        reg_core, strat_core = _solve(1, (200, 200, 200), seed, use_core=True)
        assert_tables_equal(reg_py, reg_core)
        assert_tables_equal(strat_py, strat_core)
