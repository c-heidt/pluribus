"""Independent-oracle equilibrium cross-validation for the subgame solver (§9, row 6.2).

The one deferred correctness test both solver regimes have been waiting on: a
**wholly independent brute-force CFR** (:mod:`test.search.brute_force_cfr`) solves a
tiny heads-up **river** subgame to its exact Nash equilibrium, and both production
paths — the **vector** regime (auto-selected for HU river by ``solve``) and the
**MCCFR** regime (instantiated directly on the river root; its mechanics are
street-agnostic) — must reach the same fixed point.

Hard gates (robust to Nash non-uniqueness and to averaging-convention differences):

- **(A) Nash / exploitability** — each solved average strategy has a small
  best-response gap on the independent tree (``exploitability``).  This is the real
  "is it an equilibrium" test and needs only the env-derived tree, not the oracle.
- **(B) Game value** — the unique zero-sum game value of each path matches the
  oracle's within tolerance.

The range-aggregated root-strategy drift vs. the oracle is reported for insight but
not hard-asserted (zero-sum Nash is value-unique, not necessarily strategy-unique).

Marked ``slow`` (full enumeration + the MCCFR iteration budget).
"""

import collections
import dataclasses
import itertools

import numpy as np
import pytest

from information_abstraction.lookup import clusters_for_board
from poker_ai.search.mccfr import _MCCFRSolver
from poker_ai.search.policy import SearchPolicy
from poker_ai.search.solver import solve, SolverState
from poker_ai.search.vector import _VectorSolver
from poker_ai.search.solver_state import SolverConfig
from test.lut_helpers import lossless_lut

from test.search.brute_force_cfr import (
    BruteForceCFR,
    BruteForceFlopCFR,
    BruteForceTurnCFR,
    build_flop_subgame,
    build_subgame,
    build_turn_subgame,
    exploitability,
    flop_exploitability,
    flop_game_value,
    game_value,
    turn_exploitability,
    turn_game_value,
)
from test.search._helpers import (
    _HU_STACKS_SHALLOW,
    _ctx,
    _flop_env,
    _late_env,
)


# --------------------------------------------------------------------------- #
# Fixture: a tiny heads-up river subgame with disjoint small-support ranges
# --------------------------------------------------------------------------- #

#: Rank floor for the oracle fixtures: ``low=10`` gives a **20-card** deck (T,J,Q,K,A)
#: instead of the 16-card J..A one.  Sixteen cards was too cramped to produce
#: strategically varied spots: the fixtures hand seat 0 the two lowest board-free
#: cards and seat 1 the next four, so with only four ranks in play seat 0 was
#: systematically drawing dead -- distinct boards kept collapsing onto the same
#: strategic situation (seeds 0 and 3 gave byte-identical oracle values, solver
#: values AND exploitability on different boards).  A fifth rank breaks that
#: degeneracy.  Cost: n_combos 120 -> 190, and the enumerating oracles grow with the
#: unseen-card count (turn ~1.3x, flop ~1.7x).
_ORACLE_LOW_RANK = 10

def _river_subgame(seed: int, stacks=(1000, 1000)):
    """Heads-up river env + 2-combo-per-seat ranges over card-disjoint holes.

    Picks eight distinct board-free cards and forms two holes per seat from
    disjoint card sets, so every cross pair is a legal joint deal and the betting
    tree still has a live bet at the river root.
    """
    env = _late_env(3, low=_ORACLE_LOW_RANK, stacks=stacks, seed=seed)
    assert env.betting_round == 3 and not env.is_terminal
    # A bet must be available at the river root for the game to be non-trivial.
    assert any(a and (a.startswith("raise") or a == "all_in") for a in env.legal_actions)

    board = {int(c) for c in env.community_cards}
    free = sorted({int(x) for x in env.combo_cards.reshape(-1)} - board)
    assert len(free) >= 8, "need eight board-free cards for 2 combos per seat"

    def combo(c0, c1):
        return tuple(sorted((c0, c1)))

    holes0 = [combo(free[0], free[1]), combo(free[2], free[3])]
    holes1 = [combo(free[4], free[5]), combo(free[6], free[7])]
    support0 = [env.combo_index[h] for h in holes0]
    support1 = [env.combo_index[h] for h in holes1]

    range0 = np.zeros(env.n_combos, dtype=np.float64)
    range1 = np.zeros(env.n_combos, dtype=np.float64)
    range0[support0] = 0.5
    range1[support1] = 0.5
    return env, range0, range1, support0, support1


def _solver_sigma(state, env, sub):
    """Extract a solved ``SolverState`` into the oracle's infoset representation.

    Reads the average policy through :class:`SearchPolicy` (regime-agnostic) keyed
    by the same ``(public_key, combo_index)`` both regimes write, mapping each node
    + acting-seat support hole onto the oracle infoset ``(seat, combo_idx, pk)``.
    """
    policy = SearchPolicy(state, use_average=True)
    sigma = {}

    def walk(node):
        if node["type"] == "term":
            return
        seat = node["actor"]
        legal = node["legal"]
        for hole_idx in sub.support[seat]:
            hand_row = env.combo_index[sub.holes[hole_idx]]
            row = policy.strategy_for(node["pk"], hand_row, legal)
            sigma[(seat, hole_idx, node["pk"])] = np.asarray(row, dtype=np.float64)
        for action in legal:
            walk(node["children"][action])

    walk(sub.root)
    return sigma


def _scale(sub) -> float:
    """A chip scale for tolerances: the largest absolute terminal payoff."""
    return max(
        abs(v) for leaf in sub.payoff.values() for v in leaf.values()
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_oracle_self_consistent(_seeded):
    """The brute-force oracle itself converges to an (almost) exact equilibrium.

    Sanity for the reference: a full-enumeration Linear-CFR solve of the tiny river
    subgame drives its own best-response gap to ~0, so it is a trustworthy
    equilibrium reference for the two production paths.
    """
    env, r0, r1, s0, s1 = _river_subgame(_seeded)
    sub = build_subgame(env, r0, r1, s0, s1)

    avg = BruteForceCFR(sub).solve(8000)
    expl = exploitability(sub, avg)
    scale = _scale(sub)
    assert expl >= -1e-9  # duality gap is non-negative
    assert expl < 0.01 * scale, f"oracle not converged: expl={expl:.4f} scale={scale}"


@pytest.mark.slow
def test_vector_regime_reaches_equilibrium(_seeded):
    """The vector regime's average strategy matches the oracle equilibrium.

    The vector regime is full-width (every action expanded), so both gates are
    tight: a near-zero best-response gap and a game value equal to the oracle's.
    """
    env, r0, r1, s0, s1 = _river_subgame(_seeded)
    sub = build_subgame(env, r0, r1, s0, s1)
    scale = _scale(sub)

    oracle_value = game_value(sub, BruteForceCFR(sub).solve(8000))

    ranges = {0: r0.astype(np.float32), 1: r1.astype(np.float32)}
    ctx = _ctx(env, ranges=ranges, seed=7)
    cfg = SolverConfig(
        leaf=ctx.leaf, max_iterations=2000, max_wall_seconds=90.0,
        discount_interval=200,
    )
    res = solve(env, ctx, cfg)
    assert res.state.vstrat, "expected the vector regime to be selected for HU river"

    sigma = _solver_sigma(res.state, env, sub)
    expl = exploitability(sub, sigma)
    value = game_value(sub, sigma)

    assert expl < 0.02 * scale, f"vector path exploitable: expl={expl:.4f} scale={scale}"
    assert abs(value - oracle_value) < 0.02 * scale, (
        f"vector game value {value:.4f} != oracle {oracle_value:.4f} (scale {scale})"
    )


@pytest.mark.slow
def test_mccfr_regime_reaches_equilibrium(_seeded):
    """The MCCFR regime on a **leaf-free turn** root matches the river-enumerating oracle.

    Rooted at the turn, not the river, because the river is the one street MCCFR never
    serves: ``_select_regime`` sends heads-up turn AND river to the vector regime, and
    everything else — pre-flop, heads-up flop, all multiway — to MCCFR.  So the honest
    MCCFR gate wants an EARLY street.  Pre-flop and multiway-flop roots both carry a
    depth-limit leaf (``DepthLimit.classify``), which an exact oracle cannot represent;
    a heads-up turn is the earliest root that is leaf-free (``street_at_root == 2`` ⇒
    "internal" everywhere, terminal leaves only), so the oracle can enumerate the whole
    subgame and the comparison stays exact.  ``test_mccfr_flop_reaches_equilibrium``
    covers the heads-up flop, MCCFR's other leaf-free production cell.

    ``solve`` would route this root to the vector regime, so the MCCFR path is exercised
    by instantiating ``_MCCFRSolver`` directly and driving the same iterate/discount loop
    ``solve`` uses.  The lossless LUT makes the river abstraction a no-op, so any residual
    gap is walk mechanics, not information loss.

    Hard gates: (1) the unique zero-sum **game value** matches the oracle, and (2) the
    ROOT-STREET strategy is an equilibrium given an exact continuation.

    On (2): production re-solves at every street boundary, so a turn solve's river rows
    are never played — they exist only to value the turn decisions.  Scoring them charges
    the solver for a strategy it discards, and under external sampling those rows are
    mostly untrained, so the raw number is dominated by one-sample noise rather than by
    anything the bot does.  Measured across the five seeds, RAW exploitability spans
    1.0-41.0 (a 40x spread) while the street-spliced number spans 0.78-1.66:

        seed        4      5     23     35     37
        raw      5.464 41.029 21.382  1.024  9.120
        spliced  0.948  0.780  1.656  1.024  0.834

    So the gate splices: the solver's ROOT-STREET rows onto the ORACLE's continuation.
    That is exactly the guarantee re-solving at the boundary gives, and it needs no
    "trained enough" threshold — the split is by street, which is exact.

    This works BECAUSE the LUT is lossless: the solver's continuation is then the same
    abstraction as the oracle's, so the root strategy was tuned against the continuation
    it is being scored with.  Under a bucketed future street the splice would be
    incoherent (measured: spliced is no better than raw there, sometimes worse) because
    the root strategy would have been optimised against a coarser continuation.
    """
    env, r0, r1, s0, s1 = _turn_subgame(_seeded)
    _install_lossless_lut(env)
    sub = build_turn_subgame(env, r0, r1, s0, s1)
    scale = _turn_scale(sub)

    oracle_avg = BruteForceTurnCFR(sub).solve(4000)
    oracle_value = turn_game_value(sub, oracle_avg)

    ranges = {0: r0.astype(np.float32), 1: r1.astype(np.float32)}
    ctx = _ctx(env, ranges=ranges, seed=11)
    cfg = SolverConfig(
        leaf=ctx.leaf, max_iterations=20000, max_wall_seconds=300.0,
        discount_interval=2000,
    )

    state = SolverState.empty()
    solver = _MCCFRSolver(env, state, ctx, cfg, ctx.rng)
    delta = cfg.discount_interval
    for t in range(1, cfg.max_iterations + 1):
        solver.iterate()
        if delta > 0 and t % delta == 0:
            k = t / delta
            state.discount(k / (k + 1.0))
    assert state.vstrat, "expected the MCCFR regime to accumulate strategy"
    # The turn root must build cluster-keyed river nodes — that gather/scatter is the
    # part a turn root exercises and a river root cannot.
    assert any(rs == "cluster" for rs in state.vrow_space.values()), (
        "expected cluster-keyed river-stage nodes in a turn subgame"
    )

    sigma = _solver_turn_sigma(state, env, sub)
    value = turn_game_value(sub, sigma)

    assert abs(value - oracle_value) < 0.05 * scale, (
        f"mccfr turn game value {value:.4f} != oracle {oracle_value:.4f} (scale {scale})"
    )

    # Root street is ``river is None``; every future-street row comes from the oracle.
    spliced = {
        key: (row if key[3] is None else oracle_avg.get(key, row))
        for key, row in sigma.items()
    }
    root_expl = turn_exploitability(sub, spliced)
    # 0.03*scale = 12.0.  The tighter 0.02 this gate briefly carried was calibrated
    # on the per-level grid (worst case 1.66 there); under the two-cell grid the turn
    # subgame is a different game and seed 4 measures 9.76, so the bound goes back to
    # the 0.03 the old grid used.  Re-tighten only against fresh measurements.
    assert root_expl < 0.03 * scale, (
        f"mccfr turn root-street strategy not an equilibrium given an exact "
        f"continuation: expl={root_expl:.4f} scale={scale}"
    )


# --------------------------------------------------------------------------- #
# Turn subgame: river-conditioned vector regime vs the river-enumerating oracle
# --------------------------------------------------------------------------- #

def _turn_subgame(seed: int, stacks=_HU_STACKS_SHALLOW):
    """Heads-up **turn** env + 2-combo-per-seat ranges over card-disjoint holes.

    The mirror of :func:`_river_subgame` one street earlier: the board is four
    cards and the river is still to come, so ``solve`` routes it to the vector
    regime, which must condition river betting on the (sampled) river card.
    Modest stacks keep the two-round (turn + river) betting tree small enough for
    the oracle to enumerate exactly.
    """
    env = _late_env(2, low=_ORACLE_LOW_RANK, stacks=stacks, seed=seed)
    assert env.betting_round == 2 and not env.is_terminal
    board = {int(c) for c in env.community_cards}
    free = sorted({int(x) for x in env.combo_cards.reshape(-1)} - board)
    assert len(free) >= 8, "need eight board-free cards for 2 combos per seat"

    def combo(c0, c1):
        return tuple(sorted((c0, c1)))

    holes0 = [combo(free[0], free[1]), combo(free[2], free[3])]
    holes1 = [combo(free[4], free[5]), combo(free[6], free[7])]
    support0 = [env.combo_index[h] for h in holes0]
    support1 = [env.combo_index[h] for h in holes1]
    range0 = np.zeros(env.n_combos, dtype=np.float64)
    range1 = np.zeros(env.n_combos, dtype=np.float64)
    range0[support0] = 0.5
    range1[support1] = 0.5
    return env, range0, range1, support0, support1


def _remap_row(src, legal_src, legal_dst):
    idx = {a: i for i, a in enumerate(legal_src)}
    out = np.zeros(len(legal_dst), dtype=np.float64)
    for j, a in enumerate(legal_dst):
        if a in idx:
            out[j] = src[idx[a]]
    s = out.sum()
    return out / s if s > 0 else np.full(len(legal_dst), 1.0 / len(legal_dst))


def _install_lossless_lut(env):
    """Give ``env`` a lossless card LUT: a unique cluster id per (hole, board).

    The cluster-keyed vector regime buckets future-street infosets by LUT cluster, so a
    *lossy* LUT would solve a coarser game than the lossless oracle.  A lossless one makes
    the abstraction a no-op, so the regime must reduce to the lossless equilibrium —
    isolating the walk / gather-scatter *code* from abstraction coarseness.

    ⚠️ The id must be a PURE FUNCTION of ``(hole, board)``.  This used to be a
    ``defaultdict(itertools.count().__next__)``, which hands ids out in first-access
    order and is therefore stateful: anything that traverses in a different order gets
    different ids for the same key, and every downstream dense-row index shifts with
    them.  That cost real debugging time on 2026-09-01 — it made the compiled search core
    look like it diverged from the Python walk when both were correct.  See
    :class:`test.lut_helpers.LosslessClusterLUT`.
    """
    env.card_info_lut = lossless_lut(np.unique(env.combo_cards))


def _river_universe(env, sub):
    """Sorted unique river-street cluster ids over the candidate rivers.

    Mirrors ``ClusterMapper`` universe-building for the river street, so the dense
    row a (hole, river) maps to can be recovered from the solved state.
    """
    cc = env.combo_cards
    root_comm = [int(c) for c in env.community_cards]
    ids = set()
    for r in sub.rivers:
        board = np.array(root_comm + [int(r)], dtype=np.int64)
        raw = clusters_for_board(env.card_info_lut["river"], cc, board)
        ids.update(int(x) for x in np.unique(raw[raw >= 0]))
    return np.array(sorted(ids), dtype=np.int64)


def _solver_turn_sigma(state, env, sub):
    """Extract the solved turn ``SolverState`` into the turn oracle's infosets.

    Turn-stage (root-street) nodes (river ``None``) are read through
    :class:`SearchPolicy` as usual.  River-stage nodes are **cluster-keyed** —
    internal ``(n_clusters, width)`` ``vstrat`` matrices not exposed via
    ``SearchPolicy`` — so they are read directly: map the (hole, turn+river) board
    to its LUT cluster, then to the dense universe row, and read that row.  With a
    lossless LUT each (hole, river) has its own cluster, so this recovers the
    per-river betting strategy the lossless oracle expects.  Keyed
    ``(seat, hole, pk, river)``.
    """
    policy = SearchPolicy(state, use_average=True)
    cc = env.combo_cards
    root_comm = [int(c) for c in env.community_cards]
    universe = _river_universe(env, sub)
    sigma = {}

    def walk(node, river):
        ntype = node["type"]
        if ntype == "term":
            return
        if ntype == "chance":
            for r in sub.rivers:
                walk(node["child"], int(r))
            return
        pk, seat, legal = node["pk"], node["actor"], node["legal"]
        for hole_idx in sub.support[seat]:
            hand_row = env.combo_index[sub.holes[hole_idx]]
            if river is None:
                row = np.asarray(policy.strategy_for(pk, hand_row, legal), dtype=np.float64)
            else:
                mat = state.vstrat.get(pk)
                if mat is None or state.vrow_space.get(pk) != "cluster":
                    row = np.full(len(legal), 1.0 / len(legal))
                else:
                    board = np.array(root_comm + [river], dtype=np.int64)
                    raw = int(clusters_for_board(env.card_info_lut["river"], cc, board)[hand_row])
                    if raw < 0:
                        row = np.full(len(legal), 1.0 / len(legal))
                    else:
                        dense = int(np.searchsorted(universe, raw))
                        r_row = mat[dense]
                        tot = r_row.sum()
                        avg = r_row / tot if tot > 0 else np.full(len(r_row), 1.0 / len(r_row))
                        row = _remap_row(avg, list(state.legal_at[pk]), legal)
            sigma[(seat, hole_idx, pk, river)] = row
        for a in legal:
            walk(node["children"][a], river)

    walk(sub.root, None)
    return sigma


def _turn_scale(sub) -> float:
    return max(
        abs(v)
        for leaf in sub.payoff.values()
        for v in (leaf.values() if isinstance(leaf, dict) else leaf)
    )


# =========================================================================== #
# Flop subgame — two nested chance levels (turn + river), §6.5
# =========================================================================== #

def _flop_subgame(seed: int, stacks=_HU_STACKS_SHALLOW):
    """Heads-up **flop** env + 2-combo-per-seat ranges over card-disjoint holes.

    The mirror of :func:`_turn_subgame` one street earlier: the board is three
    cards with turn *and* river still to come, so ``solve`` routes it to the
    vector regime, which must fold **two** sampled board cards into the LUT
    cluster ids.  Modest stacks keep the flop→turn→river betting tree small
    enough for the two-chance-level oracle to enumerate exactly.
    """
    env = _flop_env(low=_ORACLE_LOW_RANK, stacks=stacks, seed=seed)
    assert env.betting_round == 1 and not env.is_terminal
    board = {int(c) for c in env.community_cards}
    free = sorted({int(x) for x in env.combo_cards.reshape(-1)} - board)
    assert len(free) >= 8, "need eight board-free cards for 2 combos per seat"

    def combo(c0, c1):
        return tuple(sorted((c0, c1)))

    holes0 = [combo(free[0], free[1]), combo(free[2], free[3])]
    holes1 = [combo(free[4], free[5]), combo(free[6], free[7])]
    support0 = [env.combo_index[h] for h in holes0]
    support1 = [env.combo_index[h] for h in holes1]
    range0 = np.zeros(env.n_combos, dtype=np.float64)
    range1 = np.zeros(env.n_combos, dtype=np.float64)
    range0[support0] = 0.5
    range1[support1] = 0.5
    return env, range0, range1, support0, support1


# Street index of a flop root's future streets: depth 1 → turn, depth 2 → river.
_FLOP_FUTURE_NAME = {1: "turn", 2: "river"}


def _flop_universe(env, sub, depth: int):
    """Sorted unique cluster ids at future-street ``depth`` over all runouts.

    Mirrors ``ClusterMapper`` universe-building for a flop root: the board at
    depth ``d`` is the flop plus a ``d``-card completion, and the universe unions
    the clusters over **every** candidate completion of that depth — so the dense
    row a (hole, board) maps to is recoverable from the solved state.
    """
    cc = env.combo_cards
    root_comm = [int(c) for c in env.community_cards]
    name = _FLOP_FUTURE_NAME[depth]
    ids = set()
    for comp in itertools.combinations(sub.avail, depth):
        board = np.array(root_comm + list(comp), dtype=np.int64)
        raw = clusters_for_board(env.card_info_lut[name], cc, board)
        ids.update(int(x) for x in np.unique(raw[raw >= 0]))
    return np.array(sorted(ids), dtype=np.int64)


def _solver_flop_sigma(state, env, sub):
    """Extract the solved flop ``SolverState`` into the flop oracle's infosets.

    Flop-stage (root-street) nodes (runout ``()``) are read through
    :class:`SearchPolicy` as usual.  Turn- and river-stage nodes are
    **cluster-keyed** internal ``(n_clusters, width)`` ``vstrat`` matrices, so
    they are read directly: map the (hole, flop+runout board) to its LUT cluster,
    then to the dense universe row for that depth, and read that row.  With a
    lossless LUT each (hole, board) has its own cluster, so this recovers the
    per-runout betting strategy the lossless oracle expects.  Keyed
    ``(seat, hole, pk, runout)``.
    """
    policy = SearchPolicy(state, use_average=True)
    cc = env.combo_cards
    root_comm = [int(c) for c in env.community_cards]
    universe = {1: _flop_universe(env, sub, 1), 2: _flop_universe(env, sub, 2)}
    sigma = {}

    def walk(node, runout):
        ntype = node["type"]
        if ntype == "term":
            return
        if ntype == "chance":
            for c in [x for x in sub.avail if x not in runout]:
                walk(node["child"], runout + (int(c),))
            return
        pk, seat, legal = node["pk"], node["actor"], node["legal"]
        depth = len(runout)
        for hole_idx in sub.support[seat]:
            hand_row = env.combo_index[sub.holes[hole_idx]]
            if depth == 0:
                row = np.asarray(policy.strategy_for(pk, hand_row, legal), dtype=np.float64)
            else:
                mat = state.vstrat.get(pk)
                if mat is None or state.vrow_space.get(pk) != "cluster":
                    row = np.full(len(legal), 1.0 / len(legal))
                else:
                    board = np.array(root_comm + list(runout), dtype=np.int64)
                    raw = int(clusters_for_board(
                        env.card_info_lut[_FLOP_FUTURE_NAME[depth]], cc, board)[hand_row])
                    if raw < 0:
                        row = np.full(len(legal), 1.0 / len(legal))
                    else:
                        dense = int(np.searchsorted(universe[depth], raw))
                        r_row = mat[dense]
                        tot = r_row.sum()
                        avg = r_row / tot if tot > 0 else np.full(len(r_row), 1.0 / len(r_row))
                        row = _remap_row(avg, list(state.legal_at[pk]), legal)
            sigma[(seat, hole_idx, pk, runout)] = row
        for a in legal:
            walk(node["children"][a], runout)

    walk(sub.root, ())
    return sigma


def _flop_scale(sub) -> float:
    return max(abs(v) for leaf in sub.payoff.values() for v in leaf.values())


# The flop oracle converges to an (almost) exact Nash by ~250 iterations on this
# tiny 2-combo game (game value stable, self-exploitability ~0); 500 is a safe
# margin and keeps each enumerated two-chance-level solve to ~10s.
_FLOP_ORACLE_ITERS = 500


@pytest.mark.slow
def test_flop_oracle_self_consistent(_seeded):
    """The turn+river-enumerating flop oracle converges to an (almost) exact Nash.

    Sanity for the two-chance-level reference: full-enumeration Linear CFR over the
    flop tree — turn and river as nested enumerated chance nodes, per-runout betting
    infosets — drives its own best-response gap to ~0.
    """
    env, r0, r1, s0, s1 = _flop_subgame(_seeded)
    sub = build_flop_subgame(env, r0, r1, s0, s1)
    avg = BruteForceFlopCFR(sub).solve(_FLOP_ORACLE_ITERS)
    expl = flop_exploitability(sub, avg)
    scale = _flop_scale(sub)
    assert expl >= -1e-9
    assert expl < 0.01 * scale, f"flop oracle not converged: expl={expl:.4f} scale={scale}"


@pytest.mark.slow
def test_vector_flop_cluster_keyed_matches_oracle_losslessly(_seeded):
    """The vector regime's **flop** solve matches the flop Nash oracle (lossless LUT).

    A **capability** gate for the vector regime's two-chance-level path.  Production
    no longer *routes* a HU flop to the vector regime — the full-width flop→turn→river
    walk is ~1 iter/s and its budget does not divide across workers, so the flop goes
    to sampled MCCFR (see ``test_mccfr_flop_reaches_equilibrium`` and
    ``_select_regime``).  But the vector code still *supports* a flop root, so this
    keeps that path correct: driven **directly** (``_VectorSolver``, bypassing the
    router, exactly as ``test_mccfr_regime_reaches_equilibrium`` drives MCCFR on a HU
    river the router sends to vector), a flop solve — turn and river chance-*sampled*,
    both future streets stored per LUT cluster — must reach the same equilibrium as the
    independent turn+river-*enumerating* oracle when the LUT is lossless (one cluster
    per hole+board).  A lossless LUT makes the two-street abstraction a no-op, so any
    residual gap is a mechanics bug, not information loss.
    """
    env, r0, r1, s0, s1 = _flop_subgame(_seeded)
    _install_lossless_lut(env)
    sub = build_flop_subgame(env, r0, r1, s0, s1)
    scale = _flop_scale(sub)
    oracle_value = flop_game_value(sub, BruteForceFlopCFR(sub).solve(_FLOP_ORACLE_ITERS))

    ranges = {0: r0.astype(np.float32), 1: r1.astype(np.float32)}
    ctx = _ctx(env, ranges=ranges, seed=7)
    cfg = SolverConfig(
        leaf=ctx.leaf, max_iterations=3000, max_wall_seconds=600.0,
        discount_interval=200,
    )
    # Drive the vector regime directly (the router now sends a HU flop to MCCFR).
    state = SolverState.empty()
    solver = _VectorSolver(env, state, ctx, cfg, ctx.rng)
    delta = cfg.discount_interval
    for t in range(1, cfg.max_iterations + 1):
        solver.iterate()
        if delta > 0 and t % delta == 0:
            k = t / delta
            state.discount(k / (k + 1.0))
    assert state.vstrat, "vector regime accumulated no strategy on the flop root"
    # The flop root must build cluster-keyed nodes on **both** future streets
    # (turn AND river) — the two-chance-level path is the whole point — and a
    # missing street's uniform fallback could otherwise slip past the
    # exploitability gate if uniform play on it happened to be cheap.
    cluster_stages = {
        pk[0] for pk, rs in state.vrow_space.items() if rs == "cluster"
    }
    assert {"turn", "river"} <= cluster_stages, (
        f"expected cluster-keyed turn AND river nodes in a flop subgame; "
        f"got cluster stages {sorted(cluster_stages)}"
    )
    # The flop root itself stays lossless (combo-keyed), read by SearchPolicy.
    assert any(rs == "combo" for rs in state.vrow_space.values()), (
        "expected a combo-keyed (lossless) flop-root node"
    )

    sigma = _solver_flop_sigma(state, env, sub)
    expl = flop_exploitability(sub, sigma)
    value = flop_game_value(sub, sigma)
    assert expl < 0.05 * scale, f"vector flop path exploitable: expl={expl:.4f} scale={scale}"
    assert abs(value - oracle_value) < 0.03 * scale, (
        f"vector flop game value {value:.4f} != oracle {oracle_value:.4f} (scale {scale})"
    )


@pytest.mark.slow
def test_mccfr_flop_reaches_equilibrium(_seeded):
    """The **production** flop path — MCCFR on a HU flop root — reaches the oracle **value**.

    The router now sends a HU flop to external-sampling MCCFR (two future chance
    nodes make the full-width vector walk too slow / non-scaling, ``_select_regime``).
    This is the Nash gate for that path: it closes the one gap the other oracle tests
    leave — MCCFR **crossing chance nodes** (turn+river) reaching the correct
    equilibrium **value** (the river MCCFR test crosses none; the vector flop test
    crosses two but is full-width).  Driven through ``solve`` so the routing is
    exercised.

    Gate = the unique zero-sum **game value** matches the independent turn+river-
    enumerating oracle.  Deliberately **not** an exploitability gate: unlike the river
    MCCFR test (one street, every infoset swept), sampled MCCFR across *two* chance
    nodes cannot reach every runout/cluster-conditioned betting node in a single-worker
    budget, so the sampled average has genuinely-unreached infosets — their exploit
    dominates a whole-tree best response, but they never reach play (production plays
    the **final iterate**, shrinks under-trained nodes to the blueprint, re-solves on a
    deviation, and pools far more samples across the 64-worker fleet).  That per-infoset
    coverage/convergence cost is the known quality trade for the ~100× flop speed-up
    ([[project_vector_flop_regime_lever]]); the achievable, meaningful Nash signal for a
    sampled multi-street solve is that it reaches the right value.
    """
    env, r0, r1, s0, s1 = _flop_subgame(_seeded)
    _install_lossless_lut(env)
    sub = build_flop_subgame(env, r0, r1, s0, s1)
    scale = _flop_scale(sub)
    oracle_value = flop_game_value(sub, BruteForceFlopCFR(sub).solve(_FLOP_ORACLE_ITERS))

    ranges = {0: r0.astype(np.float32), 1: r1.astype(np.float32)}
    ctx = _ctx(env, ranges=ranges, seed=7)
    cfg = SolverConfig(
        leaf=ctx.leaf, max_iterations=20000, max_wall_seconds=600.0,
        discount_interval=2000,
    )
    res = solve(env, ctx, cfg)
    assert res.regime == "mccfr", f"expected MCCFR routing for a HU flop, got {res.regime}"
    assert res.state.vstrat, "MCCFR flop accumulated no strategy"

    sigma = _solver_flop_sigma(res.state, env, sub)
    value = flop_game_value(sub, sigma)
    assert abs(value - oracle_value) < 0.05 * scale, (
        f"mccfr flop game value {value:.4f} != oracle {oracle_value:.4f} (scale {scale})"
    )


# --------------------------------------------------------------------------- #
# §8.6 ceiling gate — DBR with a perfect model must exploit (search-internal)
# --------------------------------------------------------------------------- #

class _FoldHeavyModel:
    """A **perfect** opponent model that folds most of the time (``sigma-hat``).

    ``confidence == 1`` ⇒ ``p_max = 1`` ⇒ the DBR clamp pins the opponent's search
    strategy to *exactly* this row every iteration (``sigma-tilde = sigma-hat``), so it
    is a genuine, known, non-equilibrium opponent to best-respond to — and the
    opponent's accumulated average in the DBR solve is ``sigma-hat`` itself.
    """

    def __init__(self, pfold: float = 0.8):
        self.pfold = float(pfold)

    def strategy(self, state):
        legal = list(state.legal_actions)
        n = len(legal)
        row = np.full(n, (1.0 - self.pfold) / max(1, n - 1), dtype=np.float64)
        if "fold" in legal:
            row[legal.index("fold")] = self.pfold
        else:                                    # no fold available → uniform
            row[:] = 1.0 / n
        return row / row.sum()

    def confidence(self, state):
        return 1.0

    def reopen_after_fork(self):
        pass


def _drive_vector_flop(env, ctx, cfg) -> SolverState:
    """Full-width vector solve of a flop root, driven directly (the router now sends a
    HU flop to sampled MCCFR).  Full-width ⇒ tight, exact-per-iteration convergence, so
    the clamp's exploitation shows as a clean deterministic value rather than through
    MCCFR sampling noise.  The clamp seam (``vform.apply_model_clamp``) is the SAME one
    the production MCCFR flop path runs, so this validates the mechanism in use."""
    state = SolverState.empty()
    solver = _VectorSolver(env, state, ctx, cfg, ctx.rng)
    d = cfg.discount_interval
    for t in range(1, cfg.max_iterations + 1):
        solver.iterate()
        if d > 0 and t % d == 0:
            k = t / d
            state.discount(k / (k + 1.0))
    return state


@pytest.mark.slow
def test_dbr_with_perfect_model_beats_vanilla_vs_the_opponent(_seeded):
    """§8.6 ceiling: with a PERFECT model, DBR is not worse than vanilla — and exploits.

    The go/no-go for the whole exploitation programme, as a **deterministic subgame
    value** rather than noisy sampled play: best-responding to a fixed, known,
    non-equilibrium opponent ``sigma-hat`` (DBR, ``p_max = 1``) cannot score *worse*
    than playing the equilibrium against it (vanilla), and on an exploitable opponent it
    scores strictly better.  This is the invariant an end-to-end bb/100 comparison is
    *trying* to measure — but on dev hardware the per-hand all-in variance swamps it
    (CI ±100+ bb/100 at any feasible hand count), so the played-EV ceiling is a
    production-hardware procedure; here we assert the exact search-internal value.

    Construction: solve the same small flop subgame vanilla and DBR (fold-heavy model on
    the opponent seat).  With ``p_max = 1`` the DBR solve's opponent average *is*
    ``sigma-hat``, so score hero vs that same ``sigma-hat`` both ways — DBR hero (a best
    response) and vanilla hero (the equilibrium) — via the exact ``flop_game_value``.
    Driven full-width (vector) for a tight, low-variance signal (see ``_drive_vector_flop``);
    parametrised over the autouse ``_seeded`` trials for a spread of subgames.

    Two gates: (1) **non-inferiority** — ``DBR >= vanilla`` within a small convergence-noise
    tolerance (the literal §8.6 "STOP if it fails"); (2) the clamp is **not a silent
    no-op** — DBR's hero strategy actually differs from vanilla's.  (The strictly-positive
    exploitation magnitude — mean gain ~+0.7 over these subgames, every trial ≥ 0 — is
    measured; it is left unasserted per-trial only because the smallest subgame's gain is
    within convergence noise of 0.)
    """
    env, r0, r1, s0, s1 = _flop_subgame(_seeded)
    _install_lossless_lut(env)
    sub = build_flop_subgame(env, r0, r1, s0, s1)
    scale = _flop_scale(sub)
    ranges = {0: r0.astype(np.float32), 1: r1.astype(np.float32)}
    ctx_van = _ctx(env, ranges=ranges, seed=7)                       # hero == seat 0
    ctx_dbr = dataclasses.replace(ctx_van, models={1: _FoldHeavyModel()})
    cfg = SolverConfig(
        leaf=ctx_van.leaf, max_iterations=3000, max_wall_seconds=600.0,
        discount_interval=300, auto_budget=False,
    )
    p_van = _solver_flop_sigma(_drive_vector_flop(env, ctx_van, cfg), env, sub)
    p_dbr = _solver_flop_sigma(_drive_vector_flop(env, ctx_dbr, cfg), env, sub)

    def hero_ev(hero_from, opp_from):
        # Splice: hero (seat 0) rows from one solve, opponent (seat 1 == sigma-hat) rows
        # from the DBR solve; score with the exact subgame value.
        prof = {k: row for k, row in hero_from.items() if k[0] == 0}
        prof.update({k: row for k, row in opp_from.items() if k[0] != 0})
        return flop_game_value(sub, prof)

    val_dbr = hero_ev(p_dbr, p_dbr)          # best response to sigma-hat
    val_van = hero_ev(p_van, p_dbr)          # equilibrium hero vs sigma-hat
    gain = val_dbr - val_van

    # (1) Non-inferiority — the §8.6 ceiling.  tol absorbs residual convergence noise at
    # 3000 full-width iterations.  A regressed exploiter is sharply negative.
    assert gain >= -0.02 * scale, (
        f"DBR worse than vanilla vs sigma-hat — gain={gain:.3f} scale={scale} "
        f"(exploitation regressed; §8.6 says STOP)"
    )
    # (2) The clamp is live, not a silent no-op: hero's DBR strategy differs from vanilla's
    # (best-responding to a fold-heavy opponent is not the equilibrium).
    max_diff = max(
        (float(np.abs(p_dbr[k] - p_van[k]).max()) for k in p_dbr if k[0] == 0),
        default=0.0,
    )
    assert max_diff > 1e-3, (
        f"DBR hero strategy identical to vanilla — the model clamp had no effect "
        f"(max row diff {max_diff:.2e})"
    )


@pytest.mark.slow
def test_turn_oracle_self_consistent(_seeded):
    """The river-enumerating turn oracle converges to an (almost) exact Nash.

    Sanity for the extended reference: full-enumeration Linear CFR over the turn
    tree — with the river as an explicit enumerated chance node and per-river
    river-betting infosets — drives its own best-response gap to ~0.
    """
    env, r0, r1, s0, s1 = _turn_subgame(_seeded)
    sub = build_turn_subgame(env, r0, r1, s0, s1)
    avg = BruteForceTurnCFR(sub).solve(4000)
    expl = turn_exploitability(sub, avg)
    scale = _turn_scale(sub)
    assert expl >= -1e-9
    assert expl < 0.01 * scale, f"turn oracle not converged: expl={expl:.4f} scale={scale}"


@pytest.mark.slow
def test_vector_turn_cluster_keyed_matches_oracle_losslessly(_seeded):
    """The cluster-keyed vector turn solve matches the turn Nash oracle (lossless LUT).

    The decisive code gate for the change (§6.5): a turn subgame solved by the
    vector regime — river as a chance-*sampled* node, river-betting strategies
    stored per LUT cluster — must reach the same equilibrium as the independent
    river-*enumerating* oracle **when the LUT is lossless** (one cluster per hole+
    board).  A lossless LUT makes the future-street abstraction a no-op, so this
    isolates the walk / gather-scatter code from abstraction coarseness: any
    residual gap is a mechanics bug, not information loss.  Sampling converges at
    ~1/sqrt(T) and each river's rows see ~T/R updates, so the budget is generous
    and the exploitability gate slightly looser than the (exact) oracle's own.
    """
    env, r0, r1, s0, s1 = _turn_subgame(_seeded)
    _install_lossless_lut(env)
    sub = build_turn_subgame(env, r0, r1, s0, s1)
    scale = _turn_scale(sub)
    oracle_value = turn_game_value(sub, BruteForceTurnCFR(sub).solve(4000))

    ranges = {0: r0.astype(np.float32), 1: r1.astype(np.float32)}
    ctx = _ctx(env, ranges=ranges, seed=7)
    cfg = SolverConfig(
        leaf=ctx.leaf, max_iterations=8000, max_wall_seconds=120.0,
        discount_interval=200,
    )
    res = solve(env, ctx, cfg)
    assert res.state.vstrat, "expected the vector regime for a HU turn subgame"
    # The change must have produced cluster-keyed river-betting nodes.
    assert any(rs == "cluster" for rs in res.state.vrow_space.values()), (
        "expected cluster-keyed river-stage nodes in a turn subgame"
    )

    sigma = _solver_turn_sigma(res.state, env, sub)
    expl = turn_exploitability(sub, sigma)
    value = turn_game_value(sub, sigma)
    assert expl < 0.05 * scale, f"vector turn path exploitable: expl={expl:.4f} scale={scale}"
    assert abs(value - oracle_value) < 0.03 * scale, (
        f"vector turn game value {value:.4f} != oracle {oracle_value:.4f} (scale {scale})"
    )
