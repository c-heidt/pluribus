"""Phase-4c wiring gate: the in-core blueprint policy path inside the FastState
leaf rollout (``leaf_fast``).

Two things the isolated ``test_core_blueprint_sigma`` cannot check:

* ``_resolve_core_policy`` picks the in-core path for a cache-backed
  ``BlueprintPolicy`` fleet and rejects everything else (UniformPolicy, a blueprint
  opened without the shm cache) → Python fallback.
* On **real FastState rollout nodes**, leaf_fast's argument construction
  (``legal_cols = [ACTION_TO_IDX[br][a] for a in legal]``, ``fs.info_set()``,
  ``fs.betting_round``, the seat's bias) drives ``core_sigma`` to exactly what the
  Python ``_policy_state`` + ``BlueprintPolicy.strategy`` would produce at that node.

LUT-free: a stub LUT + synthetic strategy rows written at the exact info-set keys the
(deterministic) rollout line produces, so the reads HIT.  Skips without the core.
"""

import collections
import copy

import numpy as np
import pytest

from poker_ai import _core

pytestmark = pytest.mark.skipif(
    not _core.CORE_AVAILABLE, reason="compiled core extension not built"
)

if _core.CORE_AVAILABLE:
    from environment.action_space import (
        ACTION_TO_IDX,
        CANONICAL_ACTIONS,
        MAX_ACTIONS_PER_STREET,
    )
    from environment.player import Player
    from environment.poker_env import (
        PokerEnv, PolicyState,
        _ACTION_BYTE, _STAGE_ID, RAISE_SIZES_BY_STAGE, MAX_RAISES_PER_ROUND,
    )
    from poker_ai._core import _state as _cystate
    from poker_ai.search.context import SubgameContext
    from poker_ai.search.leaf import LeafConfig, continuation_value_vector
    from poker_ai.search.leaf_fast import (
        _resolve_core_policy, continuation_value_vector_fast,
    )
    from poker_ai.search.policy import BlueprintPolicy
    from poker_ai.tables.cfr_tables import CFRTables
    from poker_ai.tables.index import lmdb_map_size_for_players
    from test.search._helpers import UniformPolicy

    _cystate.configure(_STAGE_ID, _ACTION_BYTE, RAISE_SIZES_BY_STAGE, MAX_RAISES_PER_ROUND)
    FastState = _cystate.FastState
    _CAPS = {r: 1 << 14 for r in range(4)}
    _BIASES = ("none", "fold", "call", "raise")


def _stub_lut(env):
    env.card_info_lut = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0))


def _flop_frontier(seed, n=3):
    np.random.seed(seed)
    env = PokerEnv(players=[Player(i, 200) for i in range(n)],
                   low_card_rank=9, high_card_rank=14, small_blind=25, big_blind=50)
    _stub_lut(env)
    g = 0
    while env.betting_round < 1 and not env.is_terminal and g < 20:
        env.step_in_place("call" if "call" in env.legal_actions else "check")
        g += 1
    return env


def _tables(tmp_path, cache=True):
    shm = tmp_path / "shm"
    shm.mkdir(parents=True, exist_ok=True)
    return CFRTables(
        index_path=tmp_path / "lmdb_index",
        shm_dir=str(shm),
        actions_per_street=MAX_ACTIONS_PER_STREET,
        lmdb_map_size=lmdb_map_size_for_players(2),
        enable_index_cache=cache,
        index_capacities=_CAPS if cache else None,
    )


def _fixed_board(env):
    prefix = [int(c) for c in env.community_cards]
    used = set(prefix)
    for p in env.players:
        used.update(int(c) for c in p.cards)
    deck = [int(c) for c in np.unique(env.combo_cards)]
    undealt = [c for c in deck if c not in used]
    return prefix + undealt[:5 - len(prefix)]


def _drive_line(env, board):
    """Deterministic check/call line on a FastState over ``board``; returns the list
    of ``(betting_round, info_set, legal_tuple)`` decision nodes visited."""
    fs = FastState.from_poker_env(env)
    fs.set_board(board)
    fs.refresh_clusters(env.card_info_lut)
    nodes = []
    g = 0
    while not fs.is_terminal and g < 60:
        legal = [a for a in fs.legal_actions() if a is not None]
        nodes.append((fs.betting_round, fs.info_set(), tuple(legal)))
        fs.step_in_place("check" if "check" in legal else "call")
        g += 1
    return nodes


def test_resolve_core_policy_selects_and_rejects(tmp_path):
    """Resolves a cache-backed BlueprintPolicy fleet; rejects UniformPolicy and a
    blueprint opened WITHOUT the shm cache."""
    biases = ("none", "fold", "call", "raise")

    uni = UniformPolicy()
    assert _resolve_core_policy({b: uni for b in biases}, {0: "none", 1: "call"}) is None

    nocache = _tables(tmp_path / "nc", cache=False)
    try:
        bp0 = BlueprintPolicy(nocache)
        assert _resolve_core_policy({b: bp0 for b in biases}, {0: "none"}) is None
    finally:
        nocache.close()

    cached = _tables(tmp_path / "c", cache=True)
    try:
        cached.prewarm_caches()
        bp = BlueprintPolicy(cached)
        got = _resolve_core_policy({b: bp for b in biases}, {0: "none", 1: "raise"})
        assert got is bp                      # in-core path taken
        # Heterogeneous fleet (two distinct instances) → reject (one CoreTables
        # cannot be guaranteed to back both).
        bp2 = BlueprintPolicy(cached)
        mixed = {"none": bp, "fold": bp2, "call": bp, "raise": bp}
        assert _resolve_core_policy(mixed, {0: "none", 1: "fold"}) is None
    finally:
        cached.close()


@pytest.mark.parametrize("bias", ["none", "fold", "call", "raise"])
def test_core_sigma_matches_python_on_rollout_nodes(tmp_path, bias):
    """On the real FastState nodes of a rollout line, leaf_fast's in-core argument
    construction reproduces ``BlueprintPolicy.strategy`` within 1e-6 — the wiring
    proof (info_set / betting_round / legal_cols all correct)."""
    env = _flop_frontier(3)
    board = _fixed_board(env)
    nodes = _drive_line(env, board)
    assert nodes, "no decision nodes on the line — vacuous"

    tables = _tables(tmp_path)
    try:
        rng = np.random.RandomState(7)
        for br, iset, legal in nodes:
            w = MAX_ACTIONS_PER_STREET[br]
            # Populate a non-uniform strategy row (mass well above min) + a regret
            # row, so both the avg-strategy and (for other keys) fallback paths are
            # real rather than trivially uniform.
            tables.strategy[br].merge_delta_row(
                iset, rng.randint(1, 300, size=w).astype(np.int32))
            tables.regret[br].merge_delta_row(
                iset, rng.randint(-200, 400, size=w).astype(np.int32))
        tables.prewarm_caches()
        policy = BlueprintPolicy(tables, bias_multiplier=5.0, min_strategy_mass=10)
        core = policy._ensure_core()
        assert core is not None, "core reader not built (cache missing?)"

        for br, iset, legal in nodes:
            legal_cols = np.array([ACTION_TO_IDX[br][a] for a in legal], dtype=np.int64)
            got = np.asarray(policy.core_sigma(br, iset, legal_cols, bias))
            valid_mask = np.array(
                [a in set(legal) for a in CANONICAL_ACTIONS[br]], dtype=bool)
            ref = policy.strategy(
                PolicyState(player_i=0, betting_round=br, info_set=iset,
                            valid_mask=valid_mask, legal_actions=tuple(legal)),
                bias=bias,
            )
            np.testing.assert_allclose(got, ref, atol=1e-6,
                                       err_msg=f"bias={bias} br={br} legal={legal}")
    finally:
        tables.close()


def _ctx(env, policy, seed):
    n = env.n_players
    ranges = {s: np.ones(env.n_combos, np.float32) / env.n_combos for s in range(n)}
    leaf = LeafConfig(policies={c: policy for c in _BIASES})
    return SubgameContext.from_runtime(
        env=env, my_seat=0, my_hole=tuple(int(c) for c in env.players[0].cards),
        ranges=ranges, folded_ranges={}, leaf=leaf, rng=np.random.default_rng(seed))


def test_rollout_in_core_policy_unbiased_vs_python(tmp_path):
    """End-to-end: with a cache-backed BlueprintPolicy fleet the fast per-combo
    rollout takes the IN-CORE policy path and stays an unbiased estimator of the
    Python-callback rollout using the SAME blueprint.  They differ only in the
    (same-distribution) board draw + rng and in core_sigma-vs-strategy (proven
    <1e-6), so the per-combo grand means must agree within Monte-Carlo error.
    Confirms the full integration (in-core read + fallback + bias + remap + per-combo
    settlement) end to end.
    """
    env = _flop_frontier(7)
    active = [s for s in range(env.n_players) if env.players[s].is_active]
    profile = {s: "none" for s in active}
    traverser = active[0]

    # Populate the deterministic reference line's keys so some decisions HIT a
    # non-uniform blueprint (others miss → uniform); correctness holds either way.
    board = _fixed_board(env)
    nodes = _drive_line(env, board)
    tables = _tables(tmp_path)
    try:
        rng0 = np.random.RandomState(11)
        for br, iset, _legal in nodes:
            w = MAX_ACTIONS_PER_STREET[br]
            tables.strategy[br].merge_delta_row(
                iset, rng0.randint(1, 250, size=w).astype(np.int32))
        tables.prewarm_caches()
        policy = BlueprintPolicy(tables, bias_multiplier=5.0, min_strategy_mass=10)

        # The in-core path must actually be selected for this fleet.
        assert _resolve_core_policy({c: policy for c in _BIASES}, profile) is policy

        N = 800  # independent single-rollout calls averaged per arm

        def estimate(fast, base_seed):
            vals = []
            for i in range(N):
                ctx = _ctx(env, policy, seed=base_seed * 100_000 + i)
                fn = continuation_value_vector_fast if fast else continuation_value_vector
                vals.append(fn(copy.deepcopy(env), profile, ctx, traverser))
            arr = np.array(vals)                       # (N, n_combos)
            return arr.mean(0), arr.std(0, ddof=1) / np.sqrt(N)

        m_fast, se_fast = estimate(True, 1)
        m_py, se_py = estimate(False, 2)
        sem = np.sqrt(se_fast ** 2 + se_py ** 2)
        gap = np.abs(m_fast - m_py)
        assert np.all(gap < 4.0 * sem + 0.5), (
            f"in-core-policy rollout biased vs python: max gap={float(gap.max()):.3f} "
            f"at combo {int(gap.argmax())} (sem there={float(sem[gap.argmax()]):.3f})")
    finally:
        tables.close()


def test_reopen_after_fork_rebuilds_core_and_reads_match(tmp_path):
    """Fork-safety: a forked child that ``reopen_after_fork()`` + ``_ensure_core()``
    reads the in-core blueprint sigma **identically** to the parent.

    ``reopen_after_fork`` nulls the inherited CoreTables so the child rebuilds it
    against its own reopened index / fork-shared shm arrays; the ShmIndexCache mmaps
    are MAP_SHARED so the digest→row probe and the chunk reads resolve the same bytes
    the parent sees.  This is the real ``run_parallel`` → ``_reopen_leaf_fleet_lmdb``
    path in miniature.
    """
    import os
    import pickle

    env = _flop_frontier(5)
    board = _fixed_board(env)
    nodes = _drive_line(env, board)
    tables = _tables(tmp_path)
    rng = np.random.RandomState(21)
    for br, iset, _legal in nodes:
        w = MAX_ACTIONS_PER_STREET[br]
        tables.strategy[br].merge_delta_row(
            iset, rng.randint(1, 300, size=w).astype(np.int32))
    tables.prewarm_caches()
    policy = BlueprintPolicy(tables, bias_multiplier=5.0, min_strategy_mass=10)

    cases = []  # (br, iset, legal_cols, bias)
    for i, (br, iset, legal) in enumerate(nodes):
        legal_cols = np.array([ACTION_TO_IDX[br][a] for a in legal], dtype=np.int64)
        cases.append((br, iset, legal_cols, _BIASES[i % 4]))

    def sigmas(pol):
        return [np.asarray(pol.core_sigma(br, iset, lc, bias))
                for (br, iset, lc, bias) in cases]

    assert policy._ensure_core() is not None
    parent = sigmas(policy)

    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:                                   # child
        payload = None
        try:
            os.close(r_fd)
            policy.reopen_after_fork()             # nulls core + reopens LMDB
            assert policy._core_tables is False     # rebuilt lazily on next use
            payload = ("ok", sigmas(policy))         # forces the rebuild
        except Exception:                            # pragma: no cover - child crash
            import traceback
            payload = ("err", traceback.format_exc())
        try:
            with os.fdopen(w_fd, "wb") as f:
                pickle.dump(payload, f)
        finally:
            os._exit(0)
    os.close(w_fd)                                 # parent
    with os.fdopen(r_fd, "rb") as f:
        status, child = pickle.load(f)
    os.waitpid(pid, 0)
    tables.close()

    assert status == "ok", f"child failed:\n{child}"
    assert len(child) == len(parent)
    for i, (a, b) in enumerate(zip(child, parent)):
        np.testing.assert_array_equal(a, b, err_msg=f"fork read mismatch at case {i}")
