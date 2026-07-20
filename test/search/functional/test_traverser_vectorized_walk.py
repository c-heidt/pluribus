"""Phase 2 gate: the traverser-vectorized MCCFR walk on a leaf-free subgame.

A multiway (3-player) **turn** root plays to real showdowns (the depth limit never
returns a "leaf" for a turn root), so the solver takes the new single-pass
vectorized walk.  This gate pins the properties the change exists for:

- **Anti-starvation** — every board-feasible combo row at the root accrues average
  strategy each iteration, so after a handful of iterations essentially *all* of
  them carry mass (vs the scalar walk's ~one row per iteration).
- **Determinism** — same seed ⇒ identical ``vregret``/``vstrat``.
- **Validity** — finite tables, normalisable strategy rows, writes the shared
  ``vregret``/``vstrat`` matrices (the ones ``SearchPolicy`` reads).

The equilibrium-correctness proof (the exploitability floor dropping) is the slow
oracle in ``test_equilibrium_oracle`` — out of this fast gate's scope.
"""

import collections

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv
from poker_ai.search.context import SubgameContext
from poker_ai.search.leaf import LeafConfig
from poker_ai.search.mccfr import _BIAS_CLASSES, _MCCFRSolver
from poker_ai.search.policy import Policy
from poker_ai.search.solver import _select_regime
from poker_ai.search.solver_state import SolverConfig, SolverState
from poker_ai.search.vector import _regret_match_matrix


class _UniformPolicy(Policy):
    def strategy(self, state, bias="none"):
        n = len(state.legal_actions)
        return np.full(n, 1.0 / n, dtype=np.float32) if n else np.array([], np.float32)


def _stub_lut(env):
    env.card_info_lut = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0)
    )


def _turn_env_3p(seed, low=9, high=14, stacks=(200, 200, 200)):
    """3-player small-deck env advanced to the turn root (all calls/checks)."""
    np.random.seed(seed)
    env = PokerEnv(players=[Player(i, s) for i, s in enumerate(stacks)],
                   low_card_rank=low, high_card_rank=high)
    _stub_lut(env)
    g = 0
    while env.betting_round < 2 and not env.is_terminal and g < 80:
        env.step_in_place("call" if "call" in env.legal_actions else "check")
        g += 1
    return env


def _ctx(env, seed):
    ranges = {s: np.ones(env.n_combos, np.float32) / env.n_combos for s in range(3)}
    leaf = LeafConfig(policies={c: _UniformPolicy() for c in _BIAS_CLASSES}, n_rollouts=1)
    return SubgameContext.from_runtime(
        env=env, my_seat=0,
        my_hole=tuple(int(c) for c in env.players[0].cards),
        ranges=ranges, folded_ranges={}, leaf=leaf,
        rng=np.random.default_rng(seed),
    )


def _cfg(ctx, iters):
    return SolverConfig(leaf=ctx.leaf, max_iterations=iters, max_wall_seconds=60.0,
                        discount_interval=0, workers=1)


def _run(env, ctx, iters):
    state = SolverState.empty()
    solver = _MCCFRSolver(env, state, ctx, _cfg(ctx, iters), ctx.rng)
    for _ in range(iters):
        solver.iterate()
    solver.restore_root()
    return solver, state


def _board_feasible(env):
    board = set(int(c) for c in env.community_cards)
    cc = env.combo_cards
    return np.array([int(cc[k, 0]) not in board and int(cc[k, 1]) not in board
                     for k in range(cc.shape[0])])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_turn_root_takes_vectorized_walk(seed):
    env = _turn_env_3p(seed)
    assert env.betting_round == 2 and not env.is_terminal
    ctx = _ctx(env, seed)
    assert _select_regime(ctx) == "mccfr"          # multiway → MCCFR regime
    root_pk = env.public_key
    solver, state = _run(env, ctx, iters=40)
    assert solver._cmaps is not None               # turn root ⇒ cluster machinery
    # The vectorized walk writes the shared vregret/vstrat matrices.
    assert state.vregret and state.vstrat
    assert root_pk in state.vregret


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_full_width_strategy_no_starvation(seed):
    """Every board-feasible combo row at the root accrues average strategy — the
    anti-starvation property (the scalar walk touches ~one row per iteration)."""
    env = _turn_env_3p(seed)
    ctx = _ctx(env, seed)
    root_pk = env.public_key
    _, state = _run(env, ctx, iters=60)

    vstrat = state.vstrat[root_pk]                  # (n_combos, width)
    assert vstrat.shape[0] == env.n_combos
    feasible = _board_feasible(env)
    n_feas = int(feasible.sum())
    populated = vstrat.sum(axis=1) > 0
    # Nearly every feasible combo row has mass after a handful of iterations.
    covered = int((populated & feasible).sum())
    assert covered >= 0.9 * n_feas, f"only {covered}/{n_feas} feasible rows populated"
    # No mass leaks onto board-infeasible rows.
    assert not (populated & ~feasible).any()


@pytest.mark.parametrize("seed", [0, 1])
def test_deterministic_per_seed(seed):
    env_a = _turn_env_3p(seed)
    ctx_a = _ctx(env_a, seed)
    root_pk = env_a.public_key
    _, state_a = _run(env_a, ctx_a, iters=30)

    env_b = _turn_env_3p(seed)
    ctx_b = _ctx(env_b, seed)
    _, state_b = _run(env_b, ctx_b, iters=30)

    assert state_a.vregret.keys() == state_b.vregret.keys()
    for pk in state_a.vregret:
        assert np.array_equal(state_a.vregret[pk], state_b.vregret[pk])
        assert np.array_equal(state_a.vstrat[pk], state_b.vstrat[pk])
    # And the root regret is finite + yields a valid (row-normalised) strategy.
    reg = state_a.vregret[root_pk]
    assert np.isfinite(reg).all()
    sigma = _regret_match_matrix(reg)
    assert np.allclose(sigma.sum(axis=1), 1.0)


class _MultiClusterLUT:
    """A dict-street stand-in mapping combos to several clusters (hole-based)."""

    def __init__(self, n=4):
        self.n = n

    def __getitem__(self, key):
        return (int(key[0]) + int(key[1])) % self.n


@pytest.mark.parametrize("seed", [0, 1])
def test_walk_exercises_multi_cluster_river_nodes(seed):
    """With a real (multi-cluster) river LUT the walk must gather/scatter over
    >1 cluster row at the river and still produce full-width root strategy."""
    env = _turn_env_3p(seed)
    # Swap the stub LUT for a multi-cluster one on the future (river) street.
    env.card_info_lut = collections.defaultdict(lambda: _MultiClusterLUT(4))
    ctx = _ctx(env, seed)
    root_pk = env.public_key
    solver, state = _run(env, ctx, iters=40)

    # A river (future-street) node exists and is stored per cluster: >1 row.
    river_nodes = [pk for pk in state.vregret
                   if state.vrow_space.get(pk) == "cluster"]
    assert river_nodes, "no clustered river node created"
    assert any(state.vregret[pk].shape[0] > 1 for pk in river_nodes), \
        "river nodes never had >1 cluster row (multi-cluster path untested)"
    # Root strategy still full-width, tables finite.
    vstrat = state.vstrat[root_pk]
    feasible = _board_feasible(env)
    covered = int(((vstrat.sum(axis=1) > 0) & feasible).sum())
    assert covered >= 0.9 * int(feasible.sum())
    for pk in state.vregret:
        assert np.isfinite(state.vregret[pk]).all()
        assert np.isfinite(state.vstrat[pk]).all()


def test_phantom_traverser_hole_is_uniform_not_range_weighted():
    """The traverser's reseat hole is a phantom: it must be drawn UNIFORMLY, not
    from the (possibly skewed) range — otherwise the frozen board runout's marginal
    is biased (range-frequent cards under-sampled as future board cards).  Opponents
    must still follow their belief."""
    env = _turn_env_3p(0)
    # Heavily skew seat 0's (the traverser's) range onto its first two feasible combos.
    feasible = np.flatnonzero(_board_feasible(env))
    trav_range = np.full(env.n_combos, 1e-6, np.float32)
    trav_range[feasible[0]] = 1.0
    trav_range[feasible[1]] = 1.0
    ranges = {0: trav_range,
              1: np.ones(env.n_combos, np.float32) / env.n_combos,
              2: np.ones(env.n_combos, np.float32) / env.n_combos}
    leaf = LeafConfig(policies={c: _UniformPolicy() for c in _BIAS_CLASSES}, n_rollouts=1)
    ctx = SubgameContext.from_runtime(
        env=env, my_seat=0, my_hole=tuple(int(c) for c in env.players[0].cards),
        ranges=ranges, folded_ranges={}, leaf=leaf, rng=np.random.default_rng(0))
    solver = _MCCFRSolver(env, SolverState.empty(), ctx, _cfg(ctx, 1), ctx.rng)

    counts = collections.Counter()
    N = 4000
    for _ in range(N):
        holes = solver._sample_root_holes_vectorized(traverser=0)
        counts[tuple(sorted(holes[0]))] += 1

    # The two range-favoured combos must NOT dominate — a range-weighted draw would
    # put ~all mass on them; a uniform phantom spreads across ~all feasible combos.
    cc = env.combo_cards
    fav = {tuple(sorted((int(cc[feasible[0], 0]), int(cc[feasible[0], 1])))),
           tuple(sorted((int(cc[feasible[1], 0]), int(cc[feasible[1], 1]))))}
    fav_frac = sum(counts[h] for h in fav) / N
    assert fav_frac < 0.10, f"phantom hole looks range-weighted (fav_frac={fav_frac:.3f})"
    # And coverage is broad (uniform-ish over the feasible combos).
    assert len(counts) >= 0.5 * feasible.size


def test_restore_root_leaves_env_pristine():
    """The in-place vectorized walk must leave root_env byte-identical (the
    'never mutates root_env' contract)."""
    env = _turn_env_3p(0)
    ctx = _ctx(env, 0)
    before_cards = env.deck._cards.copy()
    before_holes = [tuple(int(c) for c in p._cards) for p in env.players]
    _run(env, ctx, iters=25)
    assert np.array_equal(env.deck._cards, before_cards)
    assert [tuple(int(c) for c in p._cards) for p in env.players] == before_holes
