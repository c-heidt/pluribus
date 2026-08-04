"""The cluster-keyed model seam (P1b) — one info-set key across both walk engines.

The opponent-model clamp must ask the model what it does at *hypothetical* holdings.
That query used to go through ``PokerEnv.policy_state_for(combo)``, a method the
compiled ``FastState`` adapters cannot serve — so a modeled solve fell off the Cython
core, and (because ``SearchAgent._solve_and_store`` swallows solve exceptions) an
attempt to force it produced a **silent blueprint fallback**: zero exploitation,
reported as DBR ≈ vanilla Pluribus.

The fix keys the model by **cluster** instead of by combo, which is exact because
``info_set = (cluster, canonicalised history)`` and the history is public.  This file
pins the three claims that makes the fix correct:

1. **Combo-keyed ≡ cluster-keyed** on a ``PokerEnv`` — the cluster really is the only
   card-dependent term (``test_cluster_keyed_matches_combo_keyed``).
2. **FastState ≡ PokerEnv** for the same cluster, byte-for-byte, at every node of a
   walk (``test_fast_state_info_set_for_matches_poker_env``).
3. Therefore a **modeled solve is byte-identical with the core on and off**
   (``test_modeled_solve_is_byte_identical_across_engines``) — the headline gate.

Plus the row↔cluster inverse (``ClusterMapper.universe``), whose misuse would key the
model at the wrong info-set without changing any shape or raising anything.
"""

import dataclasses

import numpy as np
import pytest

from poker_ai.search.cluster_maps import ClusterMapper
from poker_ai.search.solver import solve

from test.search._helpers import _ctx, _real_lut_env
from test.search.test_budget import _cfg
from test.search.test_belief_swap import _RecordingModel
from test.search.test_model_clamp import _digest

pytestmark = pytest.mark.requires_lut


def _modeled(ctx, models):
    return dataclasses.replace(ctx, models=models)


def _fast(env):
    """A configured ``FastState`` over ``env`` (skips if the core is unavailable)."""
    pytest.importorskip("poker_ai._core._state")
    from poker_ai.search.fast_env import build_fast_walk_env
    adapter = build_fast_walk_env(env)
    if adapter is None:
        pytest.skip("compiled core unavailable")
    return adapter


# --------------------------------------------------------------------------- #
# 1. Combo-keyed ≡ cluster-keyed (on PokerEnv)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("street", [0, 1, 2, 3])
def test_cluster_keyed_matches_combo_keyed(street):
    """``policy_state_for_cluster(cluster_of(combo))`` == ``policy_state_for(combo)``.

    If this ever fails, the clamp is querying the model at a different info-set than
    the blueprint and the belief swap use — the three would silently describe
    *different* opponents.
    """
    env = _real_lut_env(street)
    cmaps = ClusterMapper(env.card_info_lut, env.combo_cards,
                          env.community_cards, street)
    rcof = cmaps.root_cluster_of()
    public = env.policy_public_fields()

    checked = 0
    for ci in range(0, env.combo_cards.shape[0], 7):     # sample across the range
        if rcof[ci] < 0:
            continue
        by_combo = env.policy_state_for(env.combo_cards[ci], for_blueprint=True,
                                        public=public)
        by_cluster = env.policy_state_for_cluster(int(rcof[ci]), public=public)
        assert by_cluster.info_set == by_combo.info_set
        assert by_cluster.legal_actions == by_combo.legal_actions
        assert by_cluster.player_i == by_combo.player_i
        assert by_cluster.betting_round == by_combo.betting_round
        np.testing.assert_array_equal(by_cluster.valid_mask, by_combo.valid_mask)
        checked += 1
    assert checked > 1, "fixture produced no comparable combos — test is vacuous"


def test_distinct_clusters_give_distinct_info_sets():
    """The dedup in ``_fill_model_rows`` collapses rows by cluster; that is only
    sound if distinct clusters do *not* collide onto one info-set."""
    env = _real_lut_env(3)
    cmaps = ClusterMapper(env.card_info_lut, env.combo_cards,
                          env.community_cards, 3)
    clusters = sorted({int(c) for c in cmaps.root_cluster_of() if c >= 0})
    assert len(clusters) > 1, "single-cluster fixture — test is vacuous"
    keys = {env.policy_state_for_cluster(c).info_set for c in clusters}
    assert len(keys) == len(clusters)


# --------------------------------------------------------------------------- #
# 2. FastState ≡ PokerEnv, at every node
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("street", [0, 1, 2, 3])
def test_fast_state_info_set_for_matches_poker_env(street):
    """Byte-identical keys at the root, for every cluster."""
    env = _real_lut_env(street)
    adapter = _fast(env)
    cmaps = ClusterMapper(env.card_info_lut, env.combo_cards,
                          env.community_cards, street)
    clusters = sorted({int(c) for c in cmaps.root_cluster_of() if c >= 0})
    assert clusters, "no clusters at the root — test is vacuous"

    env_public = env.policy_public_fields()
    fast_public = adapter.policy_public_fields()
    assert fast_public.player_i == env_public.player_i
    assert fast_public.betting_round == env_public.betting_round
    assert fast_public.legal_actions == env_public.legal_actions
    np.testing.assert_array_equal(fast_public.valid_mask, env_public.valid_mask)

    for c in clusters:
        assert (adapter.policy_state_for_cluster(c).info_set
                == env.policy_state_for_cluster(c).info_set)


def test_fast_state_info_set_for_matches_after_actions():
    """The key is ``varint(cluster) ++ history`` — so it must keep matching as the
    history grows.  A root-only check would miss a history-encoding divergence."""
    env = _real_lut_env(1)
    adapter = _fast(env)
    cmaps = ClusterMapper(env.card_info_lut, env.combo_cards,
                          env.community_cards, 1)
    clusters = sorted({int(c) for c in cmaps.root_cluster_of() if c >= 0})[:5]

    depth = 0
    while not env.is_terminal and depth < 4:
        legal = [a for a in env.legal_actions if a is not None]
        if not legal:
            break
        action = "call" if "call" in legal else legal[0]
        env.step_in_place(action, settle_winners=False)
        adapter.step_in_place(action, settle_winners=False)
        depth += 1
        if env.is_terminal or env.betting_round != 1:
            break
        for c in clusters:
            assert (adapter.policy_state_for_cluster(c).info_set
                    == env.policy_state_for_cluster(c).info_set), (
                f"info-set diverged after {depth} action(s)"
            )
    assert depth > 0, "no actions taken — test is vacuous"


# --------------------------------------------------------------------------- #
# The row↔cluster inverse
# --------------------------------------------------------------------------- #

def test_universe_inverts_the_dense_row_map():
    """``universe(street)[cluster_of(street)[combo]]`` is the combo's raw cluster.

    The clamp keys the model by ``universe(street)[row]``.  Dense rows are a local
    relabelling — using a row *as* a cluster would query a wrong-but-existing
    info-set, changing no shape and raising nothing.
    """
    env = _real_lut_env(1)
    cmaps = ClusterMapper(env.card_info_lut, env.combo_cards,
                          env.community_cards, 1)
    avail = cmaps.avail
    cmaps.refresh((int(avail[0]), int(avail[1])))
    from information_abstraction.lookup import clusters_for_board

    for street, comp in ((2, 1), (3, 2)):
        board = np.array(list(env.community_cards) + [int(avail[i]) for i in range(comp)],
                         dtype=np.int64)
        raw = clusters_for_board(env.card_info_lut[{2: "turn", 3: "river"}[street]],
                                 env.combo_cards, board)
        dense = cmaps.cluster_of(street)
        feasible = dense >= 0
        assert feasible.sum() > 0
        np.testing.assert_array_equal(
            cmaps.universe(street)[dense[feasible]], raw[feasible]
        )
        # And the relabelling is non-trivial, or the check above proves nothing.
        assert not np.array_equal(cmaps.universe(street),
                                  np.arange(cmaps.n_rows(street)))


# --------------------------------------------------------------------------- #
# 3. The headline gate: a modeled solve is engine-independent
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("env_fn,regime", [
    (lambda: _real_lut_env(3), "vector"),
    (lambda: _real_lut_env(2), "vector"),
])
def test_modeled_solve_is_byte_identical_across_engines(monkeypatch, env_fn, regime):
    """Core on vs core off ⇒ identical solver state, with models attached.

    This is the claim that lets DBR run on the core at all: the engine is an
    implementation detail of the *walk*, never of the result.

    **Vector regime only** — and not because the model seam is weaker in MCCFR, but
    because the MCCFR core walk was never byte-identical *for vanilla either*: its
    depth-limit leaf draws the rollout board off the ``FastState``, consuming
    ``ctx.rng`` differently from the env shuffle.  That regime is gated statistically
    (the slow equilibrium oracle) by long-standing design.  The seam itself is gated
    engine-exactly for both regimes by the ``info_set_for`` tests above, which are
    what the clamp actually depends on.
    """
    pytest.importorskip("poker_ai._core._state")

    def run(core: bool):
        monkeypatch.setenv("PLURIBUS_SEARCH_CORE", "1" if core else "0")
        env = env_fn()
        ctx = _modeled(_ctx(env, seed=7), {1: _RecordingModel(c=0.6)})
        res = solve(env, ctx, _cfg(auto_budget=False, max_iterations=25,
                                   max_wall_seconds=1e9))
        assert res.regime == regime
        return _digest(res.state)

    assert run(True) == run(False)


def test_modeled_vector_solve_keeps_the_compiled_walk(monkeypatch):
    """The point of P1b: models no longer knock the solve off the core."""
    from poker_ai.search.solver_state import SolverState
    from poker_ai.search.vector import _VectorSolver

    pytest.importorskip("poker_ai._core._state")
    monkeypatch.setenv("PLURIBUS_SEARCH_CORE", "1")
    env = _real_lut_env(3)
    solver = _VectorSolver(
        env, SolverState(), _modeled(_ctx(env, seed=7), {1: _RecordingModel()}),
        _cfg(auto_budget=False, max_iterations=2, max_wall_seconds=1e9),
        np.random.default_rng(0),
    )
    assert solver._walk_env is not env, "modeled solve fell off the compiled walk"


def test_modeled_mccfr_solve_keeps_the_compiled_walk(monkeypatch):
    from poker_ai.search.mccfr import _MCCFRSolver
    from poker_ai.search.solver_state import SolverState

    pytest.importorskip("poker_ai._core._state")
    monkeypatch.setenv("PLURIBUS_SEARCH_CORE", "1")
    env = _real_lut_env(0)
    solver = _MCCFRSolver(
        env, SolverState(), _modeled(_ctx(env, seed=7), {1: _RecordingModel()}),
        _cfg(auto_budget=False, max_iterations=2, max_wall_seconds=1e9),
        np.random.default_rng(0),
    )
    assert solver._use_core, "modeled MCCFR solve fell off the compiled walk"
    # ``_cmaps`` is None at a pre-flop root, so the clamp's cluster map must come
    # from the LUT-derived fallback — without it the core walk cannot key the model.
    assert solver._cmaps is None
    assert solver._root_cluster is not None
    assert (solver._root_cluster >= 0).any()


def test_modeled_mccfr_solve_completes_and_clamps_under_the_core(monkeypatch):
    """MCCFR's engine gate (byte-identity does not apply — see above).

    The failure this guards is silent: ``SearchAgent._solve_and_store`` swallows solve
    exceptions, so a modeled core solve that raises is reported as a solve that simply
    found nothing to exploit.  Assert it both *completes* and *clamped*.
    """
    pytest.importorskip("poker_ai._core._state")
    monkeypatch.setenv("PLURIBUS_SEARCH_CORE", "1")
    env = _real_lut_env(0)
    model = _RecordingModel(c=0.6)
    res = solve(env, _modeled(_ctx(env, seed=7), {1: model}),
                _cfg(auto_budget=False, max_iterations=15,
                     max_wall_seconds=1e9))
    assert res.regime == "mccfr"
    assert model.seen, "the clamp never queried the model under the core"
    assert len(res.state.model_sigma_cache) > 0


def test_core_walk_gets_the_core_leaf(monkeypatch):
    """A compiled frontier must be handed the compiled leaf rollout.

    The leaf binding is resolved at *import* time, but whether the walk runs on the
    core is decided per solver and per iteration.  Setting the flag after import made
    those disagree — the Python rollout received a ``FastMCCFRAdapter`` and raised on
    ``with_hole_cards``.  Latent before modeled solves were allowed on the core (they
    were forced off it), and silent when it fired.
    """
    pytest.importorskip("poker_ai._core._state")
    from poker_ai.search.fast_env import build_fast_mccfr_env
    from poker_ai.search.leaf import continuation_value_vector as py_leaf
    from poker_ai.search.leaf_fast import continuation_value_vector_fast
    from poker_ai.search.mccfr import _leaf_fn

    env = _real_lut_env(0)
    monkeypatch.setenv("PLURIBUS_SEARCH_CORE", "1")
    adapter = build_fast_mccfr_env(env)
    if adapter is None:
        pytest.skip("compiled core unavailable")
    assert _leaf_fn(adapter) is continuation_value_vector_fast
    # ...and a PokerEnv frontier is never downgraded away from whatever the module
    # global selected (that would move GOLDEN_DIGEST_MCCFR).
    assert _leaf_fn(env) in (py_leaf, continuation_value_vector_fast)


def test_root_cluster_is_not_built_without_models():
    """Vanilla pays nothing for the model seam."""
    from poker_ai.search.mccfr import _MCCFRSolver
    from poker_ai.search.solver_state import SolverState

    env = _real_lut_env(0)
    solver = _MCCFRSolver(
        env, SolverState(), _ctx(env, seed=7),
        _cfg(auto_budget=False, max_iterations=2, max_wall_seconds=1e9),
        np.random.default_rng(0),
    )
    assert solver._root_cluster is None
