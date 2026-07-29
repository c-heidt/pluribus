"""Gates for the OX-Search gadget root in the vector regime (§11.3 step 11).

OX-Search (Approach B, PO-CES-HU; Ge et al. ICML 2024) refines the bot's strategy
to **exploit** a believed opponent range ``p̂`` while staying **adaptation-safe** —
no more exploitable than the blueprint, ``exp(σ') ≤ exp(σ) + Δ/β`` (Thm 4.6).  The
solver realises the paper's gadget without touching the vector ``_walk``: the
opponent's *root entry reach* becomes the mix ``c_expl·p̂ + c_safe·q_enter`` (with
``c_expl = 1/(kβ+1)``, ``c_safe = β/(kβ+1)``), and a per-combo opt-out regret row —
enter the subgame (value = the opponent's per-combo root CFV) vs take ``CBV_ref``
(the opponent's BR value vs the blueprint) — drives ``q_enter``.

Gates:

* **β = 0** (no safety) — the gadget collapses to a plain best response to the belief
  ``p̂``: the bot's tables must be byte-identical to a vanilla solve with the opponent
  range set to ``p̂``.
* **β → ∞** (safety dominates) — the exploitation coefficient ``c_expl → 0``, so the
  refined strategy stops depending on ``p̂`` (belief-independent, resolving-like).
* **no hidden shift** — the bot's regret update under the gadget equals a plain
  ``_walk`` fed the identical gadget entry reach (the ``CBV_ref`` shift cancels in
  every interior/bot regret, appearing only at the opt-out node).
* **safety margin** — the refined strategy's subgame exploitability exceeds the
  blueprint's by at most ``Δ/β`` (the Thm 4.6 bound), and larger β is safer, while
  a small β genuinely exploits (σ' ≠ blueprint).

All gates use the tiny lossless-LUT heads-up river subgame (no future streets, so the
bot's whole strategy is combo-keyed and readable) and the independent brute-force
oracle for best-response values.
"""

import numpy as np
import pytest

from poker_ai.search.parallel import run_loop
from poker_ai.search.solver import solve
from poker_ai.search.solver_state import SolverConfig, SolverState
from poker_ai.search.vector import _OX_ENTER, _VectorSolver
from poker_ai.search.vform import regret_match_matrix
from test.search._helpers import _ctx
from test.search.brute_force_cfr import br_value, build_subgame
from test.search.test_equilibrium_oracle import (
    _install_lossless_lut,
    _river_subgame,
    _solver_sigma,
    _turn_subgame,
)


def _cfg(leaf, *, beta=None, iters=400, discount=10):
    return SolverConfig(
        leaf=leaf, beta=beta, max_iterations=iters, max_wall_seconds=1e9,
        discount_interval=discount, workers=1, auto_budget=False,
    )


def _normalize(vec):
    v = np.asarray(vec, dtype=np.float64).copy()
    s = v.sum()
    return v / s if s > 0 else v


def _delta(sub):
    """Utility swing Δ = max_z u − min_z u over the subgame (seat-0 payoffs)."""
    vals = [v for leaf in sub.payoff.values() for v in leaf.values()]
    return max(vals) - min(vals)


# --------------------------------------------------------------------------- #
# Gate: β = 0 is a plain best response to the belief p̂.
# --------------------------------------------------------------------------- #
def test_ox_beta_zero_matches_belief_best_response(_seeded):
    """β = 0 ⇒ the gadget entry reach is exactly ``p̂`` (c_safe = 0), so the BOT's
    solved tables are byte-identical to a vanilla solve whose opponent range is ``p̂``.

    (``my_seat == 0`` makes the OX pass order — bot then opp — coincide with the
    vanilla seat order, and a river root samples no chance, so the two solves share
    every floating-point op.)
    """
    env, r0, r1, s0, s1 = _river_subgame(_seeded)
    _install_lossless_lut(env)
    phat = _normalize(r1)                                   # the belief, normalised
    ranges = {0: _normalize(r0), 1: phat}

    ctx_v = _ctx(env, ranges=dict(ranges), seed=7)
    ctx_ox = _ctx(env, ranges=dict(ranges), seed=7)
    res_v = solve(env, ctx_v, _cfg(ctx_v.leaf, beta=None))
    res_ox = solve(env, ctx_ox, _cfg(ctx_ox.leaf, beta=0.0))

    shared = set(res_v.state.vregret) & set(res_ox.state.vregret)
    assert shared, "expected shared vector nodes between the two solves"
    for pk in shared:
        assert np.array_equal(res_v.state.vregret[pk], res_ox.state.vregret[pk]), (
            f"β=0 vregret diverged from vanilla at {pk!r}"
        )
        assert np.array_equal(res_v.state.vstrat[pk], res_ox.state.vstrat[pk]), (
            f"β=0 vstrat diverged from vanilla at {pk!r}"
        )


# --------------------------------------------------------------------------- #
# Gate: β → ∞ makes the refined strategy belief-independent.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_ox_large_beta_ignores_belief(_seeded):
    """As β → ∞ the exploitation coefficient ``1/(kβ+1) → 0``, so the belief cannot
    influence the solve: two very different beliefs must yield the same bot **value**.

    We compare *exploitability* (a value — identical across every equilibrium of the
    subgame) rather than the raw strategy mixture: at large β the safety resolve is
    belief-independent, but an infoset can be *indifferent*, and there the mixture is
    not unique (a ~1e-8 belief perturbation can tip a near-tie and swing the mixture
    while leaving the value untouched).  The value is the belief-independence claim
    that actually holds.
    """
    env, r0, r1, s0, s1 = _river_subgame(_seeded)
    _install_lossless_lut(env)
    sub = build_subgame(env, r0, r1, s0, s1)
    opp = 1

    belief_a = np.zeros(env.n_combos); belief_a[s1[0]] = 0.9; belief_a[s1[1]] = 0.1
    belief_b = np.zeros(env.n_combos); belief_b[s1[0]] = 0.1; belief_b[s1[1]] = 0.9

    def exploitability(belief):
        ctx = _ctx(env, ranges={0: _normalize(r0), 1: _normalize(belief)}, seed=7)
        res = solve(env, ctx, _cfg(ctx.leaf, beta=1e6, iters=1200, discount=50))
        return br_value(sub, opp, _solver_sigma(res.state, env, sub))

    ea, eb = exploitability(belief_a), exploitability(belief_b)
    assert abs(ea - eb) < 0.02 * _delta(sub), (
        f"large-β exploitability still belief-dependent: {ea:.3f} vs {eb:.3f}"
    )


# --------------------------------------------------------------------------- #
# Gate: the bot walk carries no hidden CBV_ref shift.
# --------------------------------------------------------------------------- #
def test_ox_bot_walk_has_no_hidden_shift(_seeded):
    """One OX iteration's BOT pass must equal a plain ``_walk`` fed the identical
    gadget entry reach — i.e. the ``CBV_ref`` shift never enters the walk's regrets.
    """
    env, r0, r1, s0, s1 = _river_subgame(_seeded)
    _install_lossless_lut(env)
    ranges = {0: _normalize(r0), 1: _normalize(r1)}

    ctx = _ctx(env, ranges=dict(ranges), seed=7)
    st_ox = SolverState.empty()
    sox = _VectorSolver(env, st_ox, ctx, _cfg(ctx.leaf, beta=3.0), ctx.rng)

    # Pre-seed the opt-out row so q_enter is non-uniform (exercises the real mix).
    optr = st_ox.vregret[sox._ox_optout_key]
    feas = sox._ox_bc > 0
    optr[feas, _OX_ENTER] = 1.0                            # push enter over out
    q_enter = regret_match_matrix(optr)[:, _OX_ENTER]
    opp_entry = (sox._ox_c_expl * sox._ox_phat
                 + sox._ox_c_safe * q_enter) * sox._ox_bc

    sox._completion = ()                                   # river root: no runout
    sox._walk(sox._walk_env, sox._ox_bot, sox._reach[sox._ox_bot], opp_entry)

    # Plain solver (β=None) fed the SAME opponent entry reach.
    ctx_p = _ctx(env, ranges=dict(ranges), seed=7)
    st_p = SolverState.empty()
    sp = _VectorSolver(env, st_p, ctx_p, _cfg(ctx_p.leaf, beta=None), ctx_p.rng)
    sp._completion = ()
    sp._walk(sp._walk_env, sox._ox_bot, sp._reach[sox._ox_bot], opp_entry.copy())

    for pk in st_p.vregret:
        assert np.array_equal(st_ox.vregret[pk], st_p.vregret[pk]), (
            f"OX bot walk diverged from the plain walk at {pk!r} (a hidden shift?)"
        )


# --------------------------------------------------------------------------- #
# Gate: the Thm 4.6 safety margin.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_ox_safety_margin_bound(_seeded):
    """The refined strategy's subgame exploitability exceeds the blueprint's by at
    most ``Δ/β`` (Thm 4.6); a smaller β deviates further from the blueprint (more
    exploitation); and a small β genuinely exploits.

    Note the exploitability itself is **not** monotone in β here: the blueprint is the
    *uniform* leaf policy (deliberately weak), so exploiting the belief at a small β can
    make σ' both more exploitative AND *less* exploitable than uniform.  The safety
    guarantee is a one-sided upper bound (Thm 4.6), not a monotone ordering — the
    directional signal is the deviation from the blueprint, not the exploitability.
    """
    env, r0, r1, s0, s1 = _river_subgame(_seeded)
    _install_lossless_lut(env)
    sub = build_subgame(env, r0, r1, s0, s1)
    bot, opp = 0, 1
    delta = _delta(sub)

    # Blueprint (the leaf fleet's uniform policy) exploitability: the opponent's BR
    # value vs a uniform bot (an empty sigma dict → uniform in the oracle).
    exp_bp = br_value(sub, opp, {})

    # A skewed belief so exploitation has something to bite on.
    belief = np.zeros(env.n_combos)
    for j, c in enumerate(s1):
        belief[c] = 1.0 + j                                # non-uniform over the support

    def refined(beta, iters=1500):
        ctx = _ctx(env, ranges={0: _normalize(r0), 1: _normalize(belief)}, seed=7)
        res = solve(env, ctx, _cfg(ctx.leaf, beta=beta, iters=iters, discount=50))
        sigma = _solver_sigma(res.state, env, sub)
        exp_prime = br_value(sub, opp, sigma)              # opp BR vs the refined bot
        dev = max(np.abs(row - 1.0 / len(row)).max()
                  for k, row in sigma.items() if k[0] == bot)
        return sigma, exp_prime, dev

    tol = 0.03 * delta
    _, exp_big, dev_big = refined(50.0)
    _, exp_small, dev_small = refined(0.5)

    # (1) Safety (Thm 4.6): the refined exploitability stays within Δ/β of the blueprint.
    assert exp_big - exp_bp <= delta / 50.0 + tol, (
        f"β=50 unsafe: exp'−exp_bp={exp_big - exp_bp:.4f} > Δ/β={delta / 50.0:.4f}"
    )
    assert exp_small - exp_bp <= delta / 0.5 + tol
    # (2) Exploitation/safety tradeoff: a smaller β deviates further from the blueprint.
    assert dev_small >= dev_big - 1e-3, (
        f"smaller β did not exploit more: dev(β=0.5)={dev_small:.3f} "
        f"< dev(β=50)={dev_big:.3f}"
    )
    # (3) Small β genuinely exploits — the refined bot deviates from the uniform blueprint.
    assert dev_small > 0.05, (
        f"small β did not exploit (max deviation from uniform {dev_small:.3f})"
    )


# --------------------------------------------------------------------------- #
# Turn-root gates — exercise the CHANCE-SAMPLING path (v_enter sampled per river
# vs CBV_ref enumerated).  A turn root has a river chance node, so this covers the
# machinery the river-root gates above cannot reach.
# --------------------------------------------------------------------------- #
def test_ox_turn_root_beta_zero_matches_belief(_seeded):
    """β = 0 on a TURN root is byte-identical to a vanilla solve with opp range = ``p̂``.

    The turn root samples one river per iteration; this confirms ``_iterate_ox``
    drives that chance-sampled dual-pass walk exactly as the vanilla loop does (same
    completion draw, same pass order), so the OX plumbing is correct across the
    chance node — not just on a river root.
    """
    env, r0, r1, s0, s1 = _turn_subgame(_seeded)
    _install_lossless_lut(env)
    ranges = {0: _normalize(r0), 1: _normalize(r1)}

    ctx_v = _ctx(env, ranges=dict(ranges), seed=7)
    ctx_ox = _ctx(env, ranges=dict(ranges), seed=7)
    res_v = solve(env, ctx_v, _cfg(ctx_v.leaf, beta=None, iters=200, discount=50))
    res_ox = solve(env, ctx_ox, _cfg(ctx_ox.leaf, beta=0.0, iters=200, discount=50))

    shared = set(res_v.state.vregret) & set(res_ox.state.vregret)
    assert shared, "expected shared vector nodes (incl. clustered river nodes)"
    for pk in shared:
        assert np.array_equal(res_v.state.vregret[pk], res_ox.state.vregret[pk]), (
            f"β=0 turn vregret diverged from vanilla at {pk!r}"
        )
        assert np.array_equal(res_v.state.vstrat[pk], res_ox.state.vstrat[pk]), (
            f"β=0 turn vstrat diverged from vanilla at {pk!r}"
        )


@pytest.mark.slow
def test_ox_turn_root_opt_out_active_and_finite(_seeded):
    """A moderate-β TURN solve runs the opt-out path — sampled ``v_enter`` (one river)
    vs enumerated ``CBV_ref`` — end to end: everything stays finite, the saturation
    metric is a valid probability, and the opt-out row genuinely moves (so a
    feasibility/measure mismatch between the sampled and enumerated river paths, which
    would leave the row dead or produce NaN/inf, is caught).

    (A full turn-root safety-margin gate is blocked by the cluster-node external-read
    guard — the bot's river strategy is stored per LUT cluster and not externally
    readable — so this asserts finiteness + activity rather than the Δ/β bound, which
    the river-root gate covers.)
    """
    env, r0, r1, s0, s1 = _turn_subgame(_seeded)
    _install_lossless_lut(env)
    belief = np.zeros(env.n_combos)
    for j, c in enumerate(s1):
        belief[c] = 1.0 + j
    ctx = _ctx(env, ranges={0: _normalize(r0), 1: _normalize(belief)}, seed=7)

    st = SolverState.empty()
    cfg = _cfg(ctx.leaf, beta=3.0, iters=600, discount=50)
    solver = _VectorSolver(env, st, ctx, cfg, ctx.rng)
    run_loop(solver, st, cfg)

    # Everything finite (a sampled/enumerated river mismatch tends to surface as NaN).
    for pk, mat in st.vregret.items():
        assert np.isfinite(mat).all(), f"non-finite regret at {pk!r}"
    optr = st.vregret[(env.public_key, "OX_OPTOUT")]
    assert np.isfinite(optr).all()
    # Saturation metric is a valid probability.
    assert 0.0 <= solver.ox_enter_prob <= 1.0, solver.ox_enter_prob
    # The opt-out actually moved for some feasible combo (v_enter ≠ CBV_ref somewhere).
    feas = solver._ox_bc > 0
    assert np.abs(optr[feas]).max() > 0.0, "opt-out row never moved (dead sampled path)"
    q = regret_match_matrix(optr)
    assert np.all(q[feas] >= 0.0) and np.allclose(q[feas].sum(axis=1), 1.0)
