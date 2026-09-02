"""Golden-trace regression for the real-time subgame solver (Phase 0 safety net).

The search-side analogue of
``test/training/functional/test_core_golden_trace.py``.  A serial (``workers=1``)
``solve`` on a fixed small-deck subgame with a fixed ``ctx.rng`` seed is fully
deterministic, so it produces byte-identical ``SolverState`` tables every run.
This module freezes a digest of those tables — one per regime — as the **forward
regression oracle** for the search compiled-core rewrite:

- every phase that touches a walk must reproduce these exact digests through the
  pure-Python path (a byte-identical leaf kernel or walk port leaves them
  unchanged — that is the whole gate), and
- Phases 3-4 additionally assert the compiled walk reproduces them
  (``digest(core) == digest(python)``).

Two regimes are pinned independently: an MCCFR heads-up **preflop** subgame
(``regret`` + ``strat_sum``, keyed ``(public_key, hand_row)``) and a vector
heads-up **turn** subgame (``vregret`` + ``vstrat``, keyed ``public_key``).  A
stub LUT (every hand → bucket 0) keeps the fixture self-contained (no ``data/``
dependency); the digest still fingerprints the full betting walk + CFR maths.

The digest sorts keys by ``repr`` (hash-seed portable) and folds the float64
rows in, so it is a faithful fingerprint of the solved values independent of dict
iteration order.

Regeneration
------------
A *deliberate* change to the search maths, the env betting tree, or the small-deck
fixture changes a digest; that is the point of a regression anchor.  When intended,
regenerate the literals with::

    python -m pytest test/search/functional/test_search_core_golden.py \
        -k regenerate -s -q

and paste the printed values.  An *unexpected* mismatch means the walk / CFR maths
drifted — investigate, do not blindly re-freeze.
"""

import hashlib

import numpy as np
import pytest

from poker_ai.search.solver import solve
from poker_ai.search.solver_state import SolverConfig

from test.search._helpers import _ctx, _preflop_env, _late_env

# Fixed seeds for the two reference solves (fixture deal seed / ctx.rng seed).
_ENV_SEED = 0
_RNG_SEED = 7
_N_ITERS = 50

# Frozen fingerprints — see the module docstring for when and how to regenerate.
# MCCFR regenerated 2026-08-14: the depth-limit leaf rollout (continuation_value /
# continuation_value_vector / continuation_value_vector_fast) now always takes
# EXACTLY one action-line/board sample — the n_rollouts averaging loop (and the
# LeafConfig field driving it) was removed.  This fixture's shared _ctx() helper
# previously defaulted to n_rollouts=2, so the preflop root's depth-limit leaf
# (the vectorized continuation meta-game) now scores a genuinely different (single-
# sample) value — a deliberate maths change, not a regression.
#
# MCCFR regenerated again 2026-08-24 (RNG stream ownership, poker_ai/search/rng.py).
# No maths, env-tree or fixture change: the walk draws the SAME quantities from
# DIFFERENT streams.  Three deliberate re-plumbings contribute, and reverting all
# three reproduces the 2026-08-14 literal
# (a5521b63d4b0f443b22ac19370f78faebb45431e6842de8f902bc14867d11ee4) byte-for-byte,
# which is what establishes there is no accidental drift hiding in here:
#   1. the leaf board runout moved off the GLOBAL np.random onto ``ctx.board_rng``;
#   2. ``_MCCFRSolver._board_rng`` became a genuine child — on numpy 1.17.4 the bare
#      ``_seed_seq.spawn(1)`` it used returns a stream bit-identical to its parent,
#      so the separation its own comment describes had never taken effect;
#   3. ``SubgameContext.from_runtime`` now derives a board child, which advances the
#      parent's spawn counter and so shifts WHICH child (2) receives.
# The vector digest is unchanged: that fixture roots on the turn, which is leaf-free.
#
# BOTH regenerated 2026-08-31 for the per-raise-level action abstraction (the
# situation-indexed grid in ``environment/poker_env.py``: pre-flop open/3-bet/
# 4-bet, post-flop first-in vs facing a bet, no pre-flop limp).  Every node's
# legal set, and so the subgame tree the solver builds and the regret rows it
# writes, is different by construction — the digests MUST move.  Previous:
#   MCCFR  fee362b119ec094ea1e3e9c82dccf37e63b6a196138ff73a68840d9a5613e3ec
#   vector 7e6e3da6534d92c5e9a808a48b9da06054558cbb4a7ee8727fab0efa23113db2
#
# VECTOR ONLY, regenerated 2026-09-01: the fold terminal now returns 0 on a combo
# holding a card of the iteration's sampled completion, matching the showdown path
# (which gets it from the env's complete-board mask) and the root conditioning in
# ``_VectorSolver.iterate``.  Those rows were NOT inert: ``traverser_update`` weights
# the *strategy* sum by the traverser's own reach but adds the regret delta
# unweighted (the counterfactual weight lives in the child values), so a zero-reach
# combo still accrued regret from a fold priced at the full pot against a correctly
# zeroed showdown -- shoving looked free on exactly those iterations.  Fixing it is
# what closed the turn/flop oracle residual.  MCCFR is untouched: its terminals settle
# against concretely dealt opponents, so it never carried the range-vs-range mask.
# Previous vector: 2ad7afdd79a3d50cbf3f982dd73cef029990b6b38f3df122e4f5ce1b69b3937c
#
# MCCFR ONLY, regenerated 2026-09-01: the root-street AVERAGE strategy moved off the
# externally-sampled walk onto an enumerated pass (``_MCCFRSolver.accumulate_strategy``,
# driven by ``SolverConfig.strategy_interval``).  External sampling samples the
# OPPONENT's action, so the fused ``strat += pi_p*sigma`` accrued only on the sampled
# trajectory and carried an extra ``pi_{-i}(I)`` factor: measured on a real-LUT HU flop
# root, the MEDIAN root-street node accrued on ~4% of its traverser's iterations.  The
# pass branches every opponent action instead -- Pluribus's Algorithm 1
# ``UPDATE-STRATEGY`` fix, already applied to the blueprint trainer.  ``vregret`` is
# untouched (the regret update needs no such correction, and ``write_strat=False`` skips
# only the strategy accrual), so this digest moves purely through ``vstrat``.
# The pass fires on a periodic + power-of-two schedule owned by ``_MCCFRSolver.iterate``
# (not the orchestrator), so every driver of ``iterate()`` gets the average and the
# schedule stays a pure function of the iteration count -- which is what keeps the
# snapshot-ladder invariant intact.  The cadence is ``DEFAULT_STRATEGY_INTERVAL`` (100),
# NOT this fixture's ``discount_interval=20``: the two knobs are deliberately
# independent, so this digest is insensitive to the discount cadence.
# Previous MCCFR: 47f077d65e9f70e00492ae6f2c637dd6ed246718e56f56a475c0337805a3c585
GOLDEN_DIGEST_MCCFR = "1c15db3d4b50c16beb8c8269ff3ad1beb7f5e745aa8cdbeb6df111abca742c33"
#
# VECTOR regenerated again 2026-09-01: ``_late_env`` now installs a MULTI-CLUSTER LUT.
# It used to install a single-cluster stub (every hand → cluster 0), so this fixture's
# future streets collapsed to ``n_rows == 1`` and the cluster gather/scatter seam was
# never exercised here.  The solve is genuinely different, and genuinely more
# representative — production always runs a bucketed future-street LUT.  MCCFR's digest
# is UNCHANGED by that switch: its fixture roots pre-flop, where the subgame ends at the
# round boundary, so it has no future-street cluster rows to differ over.
# Previous vector: 1719868eb53769c09b830aed66499efa60a535e7702cb192cb57c0e14b29632d
GOLDEN_DIGEST_VECTOR = "b501561f629d25be532a2da7b5cd52e2ec45c69e82821342119a40a4e4645f03"


def _digest_tables(*tables) -> str:
    """SHA-256 over table dicts: each key's ``repr`` then its float64 row bytes.

    Keys are sorted by ``repr`` so the digest is independent of dict insertion
    order and portable across ``PYTHONHASHSEED`` (the key space is tuples of ints
    / action strings).  Including the key ``repr`` makes a key-set drift (a node
    appearing/disappearing) change the digest, not just a value drift.
    """
    h = hashlib.sha256()
    for table in tables:
        for key in sorted(table.keys(), key=repr):
            h.update(repr(key).encode("utf-8"))
            h.update(np.ascontiguousarray(table[key], dtype=np.float64).tobytes())
    return h.hexdigest()


def _solve_mccfr():
    """Deterministic MCCFR heads-up preflop solve → its ``SolverState``.

    A heads-up *flop* subgame now routes to the vector regime (§6.5 cluster-keyed
    future streets), so the MCCFR golden trace roots at the preflop instead — one
    street earlier, still ``_select_regime`` → MCCFR, and heads-up play extends to
    real showdowns, exercising the full external-sampling walk.

    The digest anchors the **pure-Python** MCCFR path, so pin the Python leaf +
    PokerEnv walk here: the ambient ``PLURIBUS_SEARCH_CORE`` import-rebinds the leaf
    to the compiled rollout and enables the FastState walk, and while the walk is
    byte-identical, the FastState *leaf* is equilibrium-gated (its board draw
    diverges the RNG stream), so an un-pinned solve under the flag would not match
    the frozen digest.  The compiled paths have their own gates (walk byte-identity
    in ``test_mccfr_walk_on_faststate``; leaf unbiasedness in ``test_leaf_fast``).
    """
    import poker_ai.search.mccfr as _mccfr
    from poker_ai.search.leaf import continuation_value_vector as _py_leaf

    saved_leaf = _mccfr.continuation_value_vector
    saved_build = _mccfr.build_fast_mccfr_env
    _mccfr.continuation_value_vector = _py_leaf         # pin the Python leaf
    _mccfr.build_fast_mccfr_env = lambda env: None      # force the PokerEnv walk
    try:
        env = _preflop_env(seed=_ENV_SEED)
        assert env.betting_round == 0 and not env.is_terminal
        ctx = _ctx(env, seed=_RNG_SEED)
        cfg = SolverConfig(
            leaf=ctx.leaf, max_iterations=_N_ITERS, max_wall_seconds=60.0,
            discount_interval=20,
        )
        res = solve(env, ctx, cfg)
        assert res.regime == "mccfr", res.regime
        return res.state
    finally:
        _mccfr.continuation_value_vector = saved_leaf
        _mccfr.build_fast_mccfr_env = saved_build


def _solve_vector():
    """Deterministic vector heads-up turn solve → its ``SolverState``."""
    env = _late_env(2, seed=_ENV_SEED)
    assert env.betting_round == 2 and not env.is_terminal
    ctx = _ctx(env, seed=_RNG_SEED)
    cfg = SolverConfig(
        leaf=ctx.leaf, max_iterations=_N_ITERS, max_wall_seconds=60.0,
        discount_interval=20,
    )
    res = solve(env, ctx, cfg)
    assert res.regime == "vector", res.regime
    return res.state


def _mccfr_digest() -> str:
    st = _solve_mccfr()
    # The MCCFR regime is now traverser-vectorized: it writes the shared
    # ``vregret``/``vstrat`` matrices (per public_key), not the scalar
    # ``regret``/``strat_sum`` dicts.
    return _digest_tables(st.vregret, st.vstrat)


def _vector_digest() -> str:
    st = _solve_vector()
    return _digest_tables(st.vregret, st.vstrat)


class TestGoldenTrace:
    def test_mccfr_matches_frozen_digest(self):
        """The MCCFR flop solve reproduces its frozen golden digest byte-for-byte."""
        assert _mccfr_digest() == GOLDEN_DIGEST_MCCFR, (
            "MCCFR search output drifted from the golden trace. If this was an "
            "intentional maths / env-tree / fixture change, regenerate the digest "
            "(see module docstring); otherwise investigate a regression."
        )

    def test_vector_matches_frozen_digest(self):
        """The vector turn solve reproduces its frozen golden digest byte-for-byte."""
        assert _vector_digest() == GOLDEN_DIGEST_VECTOR, (
            "vector search output drifted from the golden trace. If this was an "
            "intentional maths / env-tree / fixture change, regenerate the digest "
            "(see module docstring); otherwise investigate a regression."
        )

    def test_mccfr_deterministic_across_runs(self):
        """Two independent MCCFR solves produce the identical digest.

        Abstraction-independent: this must hold regardless of the frozen literal,
        and is the determinism property the whole make/undo + seeded-RNG design
        relies on (float64 CFR maths on one platform is reproducible).
        """
        assert _mccfr_digest() == _mccfr_digest()

    def test_vector_deterministic_across_runs(self):
        """Two independent vector solves produce the identical digest."""
        assert _vector_digest() == _vector_digest()

    def test_tables_nonempty(self):
        """Both regimes actually allocate rows, so the traces are non-vacuous."""
        st_m = _solve_mccfr()
        assert st_m.vregret and st_m.vstrat
        st_v = _solve_vector()
        assert st_v.vregret and st_v.vstrat


@pytest.mark.skip(reason="regeneration helper — run explicitly with -k regenerate -s")
def test_regenerate_golden_digests():
    """Print the current digests for pasting into the ``GOLDEN_DIGEST_*`` literals.

    Skipped by default; run with ``-k regenerate -s`` after a deliberate change.
    """
    print(f'\nGOLDEN_DIGEST_MCCFR = "{_mccfr_digest()}"')
    print(f'GOLDEN_DIGEST_VECTOR = "{_vector_digest()}"')
