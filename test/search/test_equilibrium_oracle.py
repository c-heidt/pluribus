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

import numpy as np
import pytest

from poker_ai.search.mccfr import _MCCFRSolver
from poker_ai.search.policy import SearchPolicy
from poker_ai.search.solver import solve, SolverState
from poker_ai.search.solver_state import SolverConfig

from test.search.brute_force_cfr import (
    BruteForceCFR,
    BruteForceTurnCFR,
    build_subgame,
    build_turn_subgame,
    exploitability,
    game_value,
    turn_exploitability,
    turn_game_value,
)
from test.search.test_solver import _ctx, _late_env


# --------------------------------------------------------------------------- #
# Fixture: a tiny heads-up river subgame with disjoint small-support ranges
# --------------------------------------------------------------------------- #

def _river_subgame(seed: int, stacks=(1000, 1000)):
    """Heads-up river env + 2-combo-per-seat ranges over card-disjoint holes.

    Picks eight distinct board-free cards and forms two holes per seat from
    disjoint card sets, so every cross pair is a legal joint deal and the betting
    tree still has a live bet at the river root.
    """
    env = _late_env(3, stacks=stacks, seed=seed)
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
        discount_interval=200, workers=1,
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
    """The MCCFR regime, run directly on the river root, matches the oracle.

    ``solve`` routes HU river to the vector regime, so the MCCFR path is exercised
    by instantiating ``_MCCFRSolver`` directly and driving the same iterate/discount
    loop ``solve`` uses (solver.py:110-118).

    External-sampling MCCFR trains *regrets* at every infoset (the regret pass
    explores all the traverser's actions), but accumulates the *average strategy*
    only along the single sampled trajectory of the strategy pass — exactly like the
    production blueprint trainer's ``update_strategy``.  So an infoset reachable only
    via an action that is ~never sampled at equilibrium keeps ``strat_sum == 0`` and
    falls back to **uniform**.  A whole-tree best response deviates into those
    off-equilibrium-path branches and exploits the uniform default, so the *raw*
    exploitability of the sampled average has a residual floor that barely shrinks
    with ``T`` (verified: 17.7 → 15.4 → 14.6 chips at 20k/80k/320k on a mixed-
    equilibrium board).  The **game value is exact** regardless (those infosets are
    off-path), and the vector regime — full-width, no sampling — trains the whole
    average and has no such floor.

    Hard gates therefore: (1) the unique zero-sum **game value** matches the oracle,
    and (2) the **trained** part of the average is a genuine equilibrium — filling
    only the untrained (``strat_sum == 0``) infosets from the oracle drives
    exploitability to ~0.  Raw exploitability is kept as a generous sanity bound.
    """
    env, r0, r1, s0, s1 = _river_subgame(_seeded)
    sub = build_subgame(env, r0, r1, s0, s1)
    scale = _scale(sub)

    oracle_avg = BruteForceCFR(sub).solve(8000)
    oracle_value = game_value(sub, oracle_avg)

    ranges = {0: r0.astype(np.float32), 1: r1.astype(np.float32)}
    ctx = _ctx(env, ranges=ranges, seed=11)
    cfg = SolverConfig(
        leaf=ctx.leaf, max_iterations=20000, max_wall_seconds=120.0,
        discount_interval=2000, workers=1,
    )

    state = SolverState.empty()
    solver = _MCCFRSolver(env, state, ctx, cfg, ctx.rng)
    delta = cfg.discount_interval
    for t in range(1, cfg.max_iterations + 1):
        solver.iterate()
        if delta > 0 and t % delta == 0:
            k = t / delta
            state.discount(k / (k + 1.0))
    assert state.strat_sum, "expected the MCCFR regime to accumulate strategy"

    sigma = _solver_sigma(state, env, sub)
    value = game_value(sub, sigma)

    # Hard gate 1: the unique zero-sum game value matches the independent oracle.
    assert abs(value - oracle_value) < 0.05 * scale, (
        f"mccfr game value {value:.4f} != oracle {oracle_value:.4f} (scale {scale})"
    )

    # Hard gate 2: where MCCFR actually trained an average (strat_sum > 0), it is a
    # genuine equilibrium — untrained off-path infosets (the uniform-default branches
    # a best response exploits) are filled from the oracle to isolate the claim.
    trained = {}
    for key, row in sigma.items():
        seat, hole, pk = key
        ss = state.strat_sum.get((pk, env.combo_index[sub.holes[hole]]))
        is_trained = ss is not None and ss.sum() > 0.0
        trained[key] = row if is_trained else oracle_avg.get(key, row)
    trained_expl = exploitability(sub, trained)
    assert trained_expl < 0.03 * scale, (
        f"mccfr trained strategy not an equilibrium: expl={trained_expl:.4f} scale={scale}"
    )

    # Generous sanity: the sampled average is not grossly exploitable overall.
    assert exploitability(sub, sigma) < 0.10 * scale


# --------------------------------------------------------------------------- #
# Turn subgame: river-conditioned vector regime vs the river-enumerating oracle
# --------------------------------------------------------------------------- #

def _turn_subgame(seed: int, stacks=(200, 200)):
    """Heads-up **turn** env + 2-combo-per-seat ranges over card-disjoint holes.

    The mirror of :func:`_river_subgame` one street earlier: the board is four
    cards and the river is still to come, so ``solve`` routes it to the vector
    regime, which must condition river betting on the (sampled) river card.
    Modest stacks keep the two-round (turn + river) betting tree small enough for
    the oracle to enumerate exactly.
    """
    env = _late_env(2, stacks=stacks, seed=seed)
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


def _solver_turn_sigma(state, env, sub):
    """Extract the solved turn ``SolverState`` into the turn oracle's infosets.

    Turn-stage nodes (river ``None``) are read through :class:`SearchPolicy` as
    usual.  River-stage nodes are **river-conditioned** — internal 3-D
    ``(n_combos, n_rivers, width)`` ``vstrat`` tensors that are *not* exposed via
    ``SearchPolicy`` (they are never played; the real river is re-solved) — so they
    are read directly here, the solver's river axis aligning with the oracle's
    sorted candidate-river order.  Keyed ``(seat, hole, pk, river)``.
    """
    policy = SearchPolicy(state, use_average=True)
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
                if mat is None or mat.ndim != 3:
                    row = np.full(len(legal), 1.0 / len(legal))
                else:
                    k = sub.rivers.index(river)
                    r_row = mat[hand_row, k]
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
def test_vector_turn_conditions_on_river_and_matches_oracle(_seeded):
    """The river-conditioned vector turn solve matches the turn Nash oracle.

    The decisive gate for the change (§6.5): a turn subgame solved by the vector
    regime — river as a chance-*sampled* node (one river per iteration), per-river
    conditioned river-betting strategies — must reach the same equilibrium as the
    independent river-*enumerating* oracle.  Both the best-response gap and the
    unique zero-sum game value are asserted.  Sampling converges at ~1/sqrt(T) and
    each river's infoset sees only ~T/R updates, so the budget is generous and the
    exploitability gate slightly looser than the (exact) oracle's own.
    """
    env, r0, r1, s0, s1 = _turn_subgame(_seeded)
    sub = build_turn_subgame(env, r0, r1, s0, s1)
    scale = _turn_scale(sub)
    oracle_value = turn_game_value(sub, BruteForceTurnCFR(sub).solve(4000))

    ranges = {0: r0.astype(np.float32), 1: r1.astype(np.float32)}
    ctx = _ctx(env, ranges=ranges, seed=7)
    cfg = SolverConfig(
        leaf=ctx.leaf, max_iterations=8000, max_wall_seconds=120.0,
        discount_interval=200, workers=1,
    )
    res = solve(env, ctx, cfg)
    assert res.state.vstrat, "expected the vector regime for a HU turn subgame"
    # The change must have produced river-conditioned (3-D) river-betting nodes.
    assert any(m.ndim == 3 for m in res.state.vstrat.values()), (
        "expected river-conditioned 3-D river-betting nodes in a turn subgame"
    )

    sigma = _solver_turn_sigma(res.state, env, sub)
    expl = turn_exploitability(sub, sigma)
    value = turn_game_value(sub, sigma)
    assert expl < 0.04 * scale, f"vector turn path exploitable: expl={expl:.4f} scale={scale}"
    assert abs(value - oracle_value) < 0.025 * scale, (
        f"vector turn game value {value:.4f} != oracle {oracle_value:.4f} (scale {scale})"
    )
