"""Tests for parallel search — independent replicas merged once (§6.7 row 11).

Three levels:

- **Merge (exact units)** — :meth:`SolverState.accumulate` sums replica tables over
  the key union, adds a warm-start ``baseline`` exactly once, and carries the
  baseline's frozen rows.  Pure, no multiprocessing.
- **Planner (exact units)** — :func:`plan_workers` staggers the traverser offset by
  ``k mod n_live`` and is reproducible from ``base_seed``; :func:`resolve_workers`
  resolves the budget.
- **End-to-end (fork)** — ``solve(..., workers>1)`` runs W replicas, is reproducible
  per ``(seed, W)``, pools ~W× the iterations, yields a valid policy, and works for
  both regimes and a warm-started re-search.  Skipped where ``fork`` is unavailable.
"""

import collections
import copy
import multiprocessing as mp

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv
from poker_ai.search.context import SubgameContext
from poker_ai.search.leaf import LeafConfig
from poker_ai.search.mccfr import _BIAS_CLASSES
from poker_ai.search.policy import Policy
from poker_ai.search.parallel import (
    WorkerPlan,
    _reopen_leaf_fleet_lmdb,
    plan_workers,
    resolve_workers,
)
from poker_ai.search.solver import solve, SolverConfig, SolverState


# --------------------------------------------------------------------------- #
# Helpers (mirrors test_solver.py)
# --------------------------------------------------------------------------- #

class UniformPolicy(Policy):
    def strategy(self, state, bias="none"):
        n = len(state.legal_actions)
        return np.full(n, 1.0 / n, dtype=np.float32) if n else np.array([], np.float32)


def _policies():
    return {c: UniformPolicy() for c in _BIAS_CLASSES}


# --------------------------------------------------------------------------- #
# Fork-safety: reopen LMDB-backed leaf policies in each worker (MDB_BAD_RSLOT)
# --------------------------------------------------------------------------- #

class _RecordingPolicy(Policy):
    """A leaf policy exposing ``reopen_after_fork`` (stands in for a blueprint)."""

    def __init__(self):
        self.reopens = 0

    def strategy(self, state, bias="none"):
        n = len(state.legal_actions)
        return np.full(n, 1.0 / n, dtype=np.float32) if n else np.array([], np.float32)

    def reopen_after_fork(self):
        self.reopens += 1


class TestReopenForkedLmdb:
    """`_reopen_leaf_fleet_lmdb` reopens each unique LMDB policy once, skips the rest."""

    def _ctx(self, policies):
        from types import SimpleNamespace
        return SimpleNamespace(leaf=SimpleNamespace(policies=policies))

    def test_shared_blueprint_reopened_once(self):
        # The four §4 bias variants share ONE blueprint object → reopen exactly once.
        bp = _RecordingPolicy()
        _reopen_leaf_fleet_lmdb(self._ctx({c: bp for c in _BIAS_CLASSES}))
        assert bp.reopens == 1

    def test_distinct_policies_each_reopened(self):
        a, b = _RecordingPolicy(), _RecordingPolicy()
        _reopen_leaf_fleet_lmdb(self._ctx({"none": a, "fold": b}))
        assert a.reopens == 1 and b.reopens == 1

    def test_policies_without_reopen_are_skipped(self):
        # A mix: the in-memory UniformPolicy has no reopen hook → skipped, no error.
        bp = _RecordingPolicy()
        _reopen_leaf_fleet_lmdb(self._ctx({"none": UniformPolicy(), "fold": bp}))
        assert bp.reopens == 1

    def test_no_reopenable_policies_is_a_noop(self):
        _reopen_leaf_fleet_lmdb(self._ctx(_policies()))   # all UniformPolicy → no crash


def _stub_lut(env: PokerEnv) -> None:
    env.card_info_lut = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0)
    )


def _preflop_env(low=11, high=14, stacks=(200, 200), seed=0) -> PokerEnv:
    """Heads-up small-deck env at the preflop root (the MCCFR regime).

    A heads-up *flop* subgame now routes to the vector regime (§6.5), so the
    MCCFR fork-parallel tests root at the preflop instead — still ``_select_regime``
    → MCCFR, heads-up play extending to real showdowns.
    """
    np.random.seed(seed)
    env = PokerEnv(
        players=[Player(i, s) for i, s in enumerate(stacks)],
        low_card_rank=low,
        high_card_rank=high,
    )
    _stub_lut(env)
    return env


def _turn_env(low=11, high=14, stacks=(200, 200), seed=0) -> PokerEnv:
    """Heads-up small-deck env advanced to the turn (vector regime)."""
    np.random.seed(seed)
    env = PokerEnv(
        players=[Player(i, s) for i, s in enumerate(stacks)],
        low_card_rank=low,
        high_card_rank=high,
    )
    _stub_lut(env)
    guard = 0
    while env.betting_round < 2 and not env.is_terminal and guard < 60:
        env.step_in_place("call" if "call" in env.legal_actions else "check")
        guard += 1
    return env


def _ctx(env, *, n_rollouts=2, seed=0) -> SubgameContext:
    ranges = {s: np.ones(env.n_combos, np.float32) / env.n_combos for s in range(2)}
    leaf = LeafConfig(policies=_policies(), n_rollouts=n_rollouts)
    return SubgameContext.from_runtime(
        env=env,
        my_seat=0,
        my_hole=tuple(int(c) for c in env.players[0].cards),
        ranges=ranges,
        folded_ranges={},
        leaf=leaf,
        rng=np.random.default_rng(seed),
    )


def _cfg(*, iters=40, discount=20, workers=1) -> SolverConfig:
    return SolverConfig(
        leaf=LeafConfig(policies=_policies(), n_rollouts=2),
        max_iterations=iters,
        max_wall_seconds=30.0,
        discount_interval=discount,
        workers=workers,
    )


def _node_state(pk, legal, actor, *, regret=None, strat=None, frozen=None, n_rows=10):
    # Build a combo-keyed vector node (the storage both regimes now write) so the
    # merge semantics are exercised on ``vregret``/``vstrat``.
    st = SolverState.empty()
    st.ensure_vnode(pk, legal, actor, n_rows, "combo")
    for row, vec in (regret or {}).items():
        st.vregret[pk][row] = np.array(vec, dtype=float)
    for row, vec in (strat or {}).items():
        st.vstrat[pk][row] = np.array(vec, dtype=float)
    for row, vec in (frozen or {}).items():
        st.frozen[(pk, row)] = np.array(vec, dtype=float)
    return st


def _regret_tables_equal(a: SolverState, b: SolverState) -> bool:
    # The MCCFR regime is traverser-vectorized → writes ``vregret`` (per public_key),
    # not the scalar ``regret`` dict.
    if set(a.vregret) != set(b.vregret):
        return False
    return all(np.array_equal(a.vregret[k], b.vregret[k]) for k in a.vregret)


_FORK = "fork" in mp.get_all_start_methods()
needs_fork = pytest.mark.skipif(not _FORK, reason="requires the fork start method")


# --------------------------------------------------------------------------- #
# Merge — SolverState.accumulate
# --------------------------------------------------------------------------- #

class TestAccumulate:

    def test_sums_regret_and_strat_over_union(self):
        pk = ("flop", ())
        a = _node_state(pk, ("f", "c", "r"), 0, regret={7: [1, 2, 3]}, strat={7: [1, 0, 0]})
        b = _node_state(
            pk, ("f", "c", "r"), 0,
            regret={7: [0, 1, 1], 8: [2, 2, 2]}, strat={8: [0, 1, 0]},
        )
        m = SolverState.accumulate([a, b])
        np.testing.assert_allclose(m.vregret[pk][7], [1, 3, 4])
        np.testing.assert_allclose(m.vregret[pk][8], [2, 2, 2])   # only b saw row 8
        np.testing.assert_allclose(m.vstrat[pk][7], [1, 0, 0])
        np.testing.assert_allclose(m.vstrat[pk][8], [0, 1, 0])
        assert m.legal_at[pk] == ("f", "c", "r")
        assert m.actor_at[pk] == 0

    def test_does_not_alias_replica_rows(self):
        # The merged row must be a fresh array, not a view onto a replica's row.
        pk = ("flop", ())
        a = _node_state(pk, ("f", "c"), 0, regret={7: [1.0, 1.0]})
        m = SolverState.accumulate([a])
        m.vregret[pk][7] += 5.0
        np.testing.assert_allclose(a.vregret[pk][7], [1.0, 1.0])

    def test_sums_vector_tables(self):
        pk = ("turn", ())
        a = SolverState.empty(); a.ensure_vnode(pk, ("c", "r"), 0, 3, "cluster")
        a.vregret[pk][:] = 1.0; a.vstrat[pk][:] = 2.0
        b = SolverState.empty(); b.ensure_vnode(pk, ("c", "r"), 0, 3, "cluster")
        b.vregret[pk][:] = 3.0; b.vstrat[pk][:] = 4.0
        m = SolverState.accumulate([a, b])
        np.testing.assert_allclose(m.vregret[pk], 4.0)
        np.testing.assert_allclose(m.vstrat[pk], 6.0)
        assert m.vrow_space[pk] == "cluster"  # row space carries through the merge

    def test_baseline_added_exactly_once(self):
        # Each replica is seeded from `baseline`; merged must be base + Σ deltas,
        # NOT W×base + ... (the warm-start regrets counted once).
        pk = ("flop", ())
        base = _node_state(pk, ("f", "c", "r"), 0, regret={7: [10, 10, 10]})
        r1 = _node_state(pk, ("f", "c", "r"), 0, regret={7: [11, 12, 13]})  # base + [1,2,3]
        r2 = _node_state(pk, ("f", "c", "r"), 0, regret={7: [10, 15, 10]})  # base + [0,5,0]
        m = SolverState.accumulate([r1, r2], baseline=base)
        np.testing.assert_allclose(m.vregret[pk][7], [11, 17, 13])  # 10 + [1,2,3] + [0,5,0]

    def test_baseline_new_node_summed_without_double_count(self):
        # A node only the replicas discovered (not in baseline) is a pure sum.
        pk = ("flop", ())
        base = _node_state(pk, ("f", "c"), 0, regret={7: [4, 4]})
        r1 = _node_state(pk, ("f", "c"), 0, regret={7: [5, 4], 8: [1, 1]})
        r2 = _node_state(pk, ("f", "c"), 0, regret={7: [4, 6], 8: [2, 2]})
        m = SolverState.accumulate([r1, r2], baseline=base)
        np.testing.assert_allclose(m.vregret[pk][7], [5, 6])   # 4 + [1,0] + [0,2]
        np.testing.assert_allclose(m.vregret[pk][8], [3, 3])   # pure sum (no baseline)

    def test_carries_baseline_frozen(self):
        pk = ("flop", ())
        base = _node_state(pk, ("f", "c"), 0, frozen={7: [0.3, 0.7]})
        r1 = _node_state(pk, ("f", "c"), 0, regret={7: [1, 1]})
        m = SolverState.accumulate([r1], baseline=base)
        np.testing.assert_allclose(m.frozen[(pk, 7)], [0.3, 0.7])

    def test_no_baseline_is_pure_sum(self):
        pk = ("flop", ())
        a = _node_state(pk, ("f", "c"), 0, regret={7: [1, 2]})
        b = _node_state(pk, ("f", "c"), 0, regret={7: [3, 4]})
        m = SolverState.accumulate([a, b])
        np.testing.assert_allclose(m.vregret[pk][7], [4, 6])
        assert m.frozen == {}


# --------------------------------------------------------------------------- #
# Planner — plan_workers / resolve_workers
# --------------------------------------------------------------------------- #

class TestPlanner:

    def test_staggers_offsets_mod_n_live(self):
        p = plan_workers(5, 3, base_seed=1)
        assert isinstance(p, WorkerPlan)
        assert p.n_workers == 5
        assert p.offsets == (0, 1, 2, 0, 1)
        assert len(p.seeds) == 5

    def test_reproducible_from_base_seed(self):
        draw = lambda s: int(np.random.default_rng(s).integers(0, 10 ** 9))
        a = plan_workers(4, 2, base_seed=7)
        b = plan_workers(4, 2, base_seed=7)
        c = plan_workers(4, 2, base_seed=8)
        assert [draw(s) for s in a.seeds] == [draw(s) for s in b.seeds]
        assert [draw(s) for s in a.seeds] != [draw(s) for s in c.seeds]

    def test_substreams_are_distinct(self):
        draw = lambda s: int(np.random.default_rng(s).integers(0, 10 ** 9))
        p = plan_workers(4, 2, base_seed=3)
        assert len({draw(s) for s in p.seeds}) == 4

    @pytest.mark.parametrize("w,n,expected", [
        (1, 3, (0,)),            # single worker
        (3, 1, (0, 0, 0)),       # N == 1 → all offset 0
        (2, 5, (0, 1)),          # W < N
        (4, 4, (0, 1, 2, 3)),    # W == N
        (6, 2, (0, 1, 0, 1, 0, 1)),  # W >> N
    ])
    def test_offset_edges(self, w, n, expected):
        assert plan_workers(w, n, base_seed=0).offsets == expected

    @pytest.mark.parametrize("val,expected", [(1, 1), (4, 4), (0, 1), (-2, 1)])
    def test_resolve_explicit(self, val, expected):
        assert resolve_workers(val) == expected

    def test_resolve_none_is_cpu_based(self):
        assert resolve_workers(None) >= 1


# --------------------------------------------------------------------------- #
# End-to-end — multiprocessing (fork)
# --------------------------------------------------------------------------- #

@needs_fork
class TestParallelSolve:

    def _run(self, *, workers, iters=40, seed=0):
        env = _preflop_env(seed=0)  # HU preflop → MCCFR (flop now routes to vector)
        ctx = _ctx(env, seed=seed)
        return solve(env, ctx, _cfg(iters=iters, workers=workers))

    def test_serial_path_reproducible(self):
        # The refactored serial loop (workers=1) is still deterministic per seed.
        a = self._run(workers=1)
        b = self._run(workers=1)
        assert _regret_tables_equal(a.state, b.state)

    def test_parallel_reproducible_per_seed_and_workers(self):
        a = self._run(workers=3)
        b = self._run(workers=3)
        assert set(a.state.vregret) == set(b.state.vregret)
        assert _regret_tables_equal(a.state, b.state)

    def test_parallel_differs_from_serial(self):
        # Same seed, different worker count → different sampling trajectory
        # (documented: parallel is not bit-identical to serial).
        serial = self._run(workers=1)
        par = self._run(workers=3)
        assert not _regret_tables_equal(serial.state, par.state)

    def test_parallel_pools_iterations(self):
        # iterations_run is the sum across replicas (~W× the per-replica cap).
        res = self._run(workers=3, iters=30)
        assert res.iterations_run == 3 * 30

    def test_parallel_repairs_parent_lmdb_slots(self, monkeypatch):
        # Regression: the fork pool clobbers the PARENT's slot in LMDB's shared
        # reader table, so run_parallel must repair the parent env after the pool
        # joins or the parent's next blueprint read trips MDB_BAD_RSLOT (this bit the
        # eval — opponent lookups on the shared blueprint failed post-search).  The
        # spy counter lives in parent memory; forked children increment their own
        # copy, so the parent sees exactly one repair per parallel solve, none serial.
        import poker_ai.search.parallel as par

        calls = []
        real = par._reopen_leaf_fleet_lmdb
        monkeypatch.setattr(
            par, "_reopen_leaf_fleet_lmdb",
            lambda ctx: (calls.append(1), real(ctx))[1],
        )
        self._run(workers=1)
        assert calls == []                    # serial never forks → no parent repair
        self._run(workers=3)
        assert len(calls) == 1                 # exactly one PARENT repair after the pool

    def test_parallel_policy_is_valid(self):
        res = self._run(workers=3)
        env = _preflop_env(seed=0)  # must match ``_run``'s root (MCCFR)
        pk = env.public_key
        hr = int(env.combo_index[tuple(sorted(int(c) for c in env.players[0].cards))])
        legal = [a for a in env.legal_actions if a is not None]
        prob = np.asarray(res.average_policy.strategy_for(pk, hr, legal), dtype=float)
        assert prob.shape == (len(legal),)
        np.testing.assert_allclose(prob.sum(), 1.0, atol=1e-5)
        assert (prob >= 0).all()
        assert len(res.state.vregret) > 0

    def test_vector_regime_parallel_runs(self):
        # The vector regime parallelizes like MCCFR (§6.7 row 11): W replicas each
        # sample their own river substream, iterations pool, and the merged 3-D
        # vregret/vstrat yield a valid policy.
        env = _turn_env(seed=1)
        assert env.betting_round == 2
        ctx = _ctx(env, seed=0)
        res = solve(env, ctx, _cfg(iters=6, workers=2))
        assert len(res.state.vregret) > 0
        assert res.iterations_run == 2 * 6           # pooled across replicas

        pk = env.public_key
        hr = int(env.combo_index[tuple(sorted(int(c) for c in env.players[0].cards))])
        legal = [a for a in env.legal_actions if a is not None]
        prob = np.asarray(res.average_policy.strategy_for(pk, hr, legal), dtype=float)
        np.testing.assert_allclose(prob.sum(), 1.0, atol=1e-5)
        assert (prob >= 0).all()

    def test_vector_regime_parallel_reproducible_per_seed(self):
        # Chance-sampled → stochastic, but (seed, workers) is reproducible.
        env = _turn_env(seed=1)
        a = solve(env, _ctx(_turn_env(seed=1), seed=0), _cfg(iters=8, workers=2))
        b = solve(env, _ctx(_turn_env(seed=1), seed=0), _cfg(iters=8, workers=2))
        assert set(a.state.vregret) == set(b.state.vregret)
        for k in a.state.vregret:
            np.testing.assert_array_equal(a.state.vregret[k], b.state.vregret[k])

    def test_warm_start_re_search_parallel(self):
        # A warm-started parallel re-search runs, keeps the baseline's frozen rows,
        # and produces a non-empty merged state.  MCCFR path → preflop root (a HU
        # flop now routes to the vector regime, §6.5).
        env = _preflop_env(seed=0)
        ctx = _ctx(env, seed=0)
        first = solve(env, ctx, _cfg(iters=30, workers=1))
        pk = env.public_key
        hr = int(env.combo_index[tuple(sorted(int(c) for c in env.players[0].cards))])
        first.state.frozen[(pk, hr)] = np.full(
            first.state.width(pk), 1.0 / first.state.width(pk)
        )
        env2 = _preflop_env(seed=0)
        ctx2 = _ctx(env2, seed=1)
        again = solve(env2, ctx2, _cfg(iters=30, workers=2), warm_start=first.state)
        assert (pk, hr) in again.state.frozen
        assert len(again.state.vregret) > 0
        assert again.iterations_run == 2 * 30
