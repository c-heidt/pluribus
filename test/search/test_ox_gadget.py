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

from poker_ai.search.solver import solve
from poker_ai.search.solver_state import SolverConfig, SolverState
from poker_ai.search.vector import _OX_ENTER, _OX_OUT, _VectorSolver
from poker_ai.search.vform import regret_match_matrix
from test.search._helpers import _ctx
from test.search.brute_force_cfr import br_value, build_subgame
from test.search.test_equilibrium_oracle import (
    _install_lossless_lut,
    _river_subgame,
    _solver_sigma,
)


def _cfg(leaf, *, beta=None, iters=400, discount=10):
    return SolverConfig(
        leaf=leaf, beta=beta, max_iterations=iters, max_wall_seconds=1e9,
        discount_interval=discount, auto_budget=False,
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
# Gate: the OX iterate path records the calibration root-value signal.
# --------------------------------------------------------------------------- #
def test_ox_solve_records_root_value(_seeded):
    """Regression: the gadget iterate path must accumulate the hero root-value signal
    the calibration reads (``SearchResult.root_value``).  ``_iterate_ox`` previously
    returned before ``_record_root_value`` ran, so every OX solve reported
    ``root_value=None`` → the OX calibration's value_gap / replica_spread came back NaN
    for every rung (``data/calibration_summary_ox.json``)."""
    env, r0, r1, s0, s1 = _river_subgame(_seeded)
    _install_lossless_lut(env)
    ranges = {0: _normalize(r0), 1: _normalize(r1)}
    ctx = _ctx(env, ranges=ranges, seed=7)
    res = solve(env, ctx, _cfg(ctx.leaf, beta=0.5))
    assert res.regime == "vector"
    assert res.root_value is not None, "OX solve did not record a root value (still None)"
    assert np.isfinite(res.root_value), f"OX root_value not finite: {res.root_value}"


# --------------------------------------------------------------------------- #
# Gate: β → ∞ makes the refined strategy belief-independent.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_ox_large_beta_ignores_belief(_seeded):
    """As β → ∞ the exploitation coefficient ``1/(kβ+1) → 0``, so two very different
    beliefs over the same support must yield (nearly) the same refined bot strategy.
    """
    env, r0, r1, s0, s1 = _river_subgame(_seeded)
    _install_lossless_lut(env)
    sub = build_subgame(env, r0, r1, s0, s1)

    opp = [i for i in s1]                                   # the two opp support combos
    belief_a = np.zeros(env.n_combos); belief_a[opp[0]] = 0.9; belief_a[opp[1]] = 0.1
    belief_b = np.zeros(env.n_combos); belief_b[opp[0]] = 0.1; belief_b[opp[1]] = 0.9

    def bot_sigma(belief):
        ctx = _ctx(env, ranges={0: _normalize(r0), 1: _normalize(belief)}, seed=7)
        res = solve(env, ctx, _cfg(ctx.leaf, beta=1e6, iters=800, discount=50))
        return _solver_sigma(res.state, env, sub)

    sa, sb = bot_sigma(belief_a), bot_sigma(belief_b)
    bot = 0
    diffs = [np.abs(sa[k] - sb[k]).max() for k in sa if k[0] == bot and k in sb]
    assert diffs, "expected bot decision rows to compare"
    assert max(diffs) < 1e-2, (
        f"large-β strategy still belief-dependent (max row diff {max(diffs):.4f})"
    )


# --------------------------------------------------------------------------- #
# Gate: the safety-branch OPT-OUT node is the paper's shifted gadget node.
# --------------------------------------------------------------------------- #
def test_ox_opt_out_equals_shifted_gadget(_seeded):
    """The opt-out row update `_iterate_ox` performs is byte-identical to the paper's
    SHIFTED safety-branch node.

    The paper shifts every subgame utility by ``−CBV_ref(I₁)``, so at the opt-out node
    ENTER pays ``v_enter − CBV_ref`` and OUT pays ``0``.  This code compares the RAW
    values (ENTER = v_enter, OUT = CBV_ref) instead — which is the SAME node because a
    per-infoset constant cancels in a CFR regret.  This gate proves that equivalence on
    the REAL ``v_enter`` the walk produces, not just on paper: it drives one gadget
    iteration two ways from identical fresh solvers and checks the opt-out row deltas
    coincide with BOTH the raw and the shifted formulas.
    """
    env, r0, r1, s0, s1 = _river_subgame(_seeded)         # river root ⇒ no chance sampling
    _install_lossless_lut(env)
    ranges = {0: _normalize(r0), 1: _normalize(r1)}

    def fresh_solver():
        ctx = _ctx(env, ranges=dict(ranges), seed=7)
        st = SolverState.empty()
        sv = _VectorSolver(env, st, ctx, _cfg(ctx.leaf, beta=2.0), ctx.rng)
        # Pre-seed a NON-uniform opt-out so q ≠ [0.5, 0.5] (the general case).
        optr = st.vregret[sv._ox_optout_key]
        optr[sv._ox_bc > 0, _OX_ENTER] = 0.7
        return sv, st, optr

    # (A) run the real `_iterate_ox` and record the opt-out row delta.
    sa, sta, optr_a = fresh_solver()
    sa._completion = ()
    before = optr_a.copy()
    sa._iterate_ox()
    actual = optr_a - before

    # (B) reproduce the SAME walk manually on an identical solver to recover v_enter,
    # then form both the raw (implemented) and shifted (paper) opt-out deltas.
    sb, stb, optr_b = fresh_solver()
    sb._completion = ()
    q = regret_match_matrix(optr_b)
    qe, qo = q[:, _OX_ENTER], q[:, _OX_OUT]
    opp_entry = (sb._ox_c_expl * sb._ox_phat + sb._ox_c_safe * qe) * sb._ox_bc
    sb._walk(sb._walk_env, sb._ox_bot, sb._reach[sb._ox_bot], opp_entry)     # bot pass
    v_enter = sb._walk(sb._walk_env, sb._ox_opp, opp_entry, sb._reach[sb._ox_bot])
    cbv, cs = sb._ox_cbv, sb._ox_c_safe

    node_raw = qe * v_enter + qo * cbv                    # implemented (raw) formula
    d_raw = np.stack([cs * (v_enter - node_raw), cs * (cbv - node_raw)], axis=1)
    ve_s, vo_s = v_enter - cbv, np.zeros_like(cbv)        # paper: shift by −CBV_ref
    node_s = qe * ve_s + qo * vo_s
    d_shift = np.stack([cs * (ve_s - node_s), cs * (vo_s - node_s)], axis=1)

    assert np.allclose(actual, d_raw), "real _iterate_ox opt-out delta != raw formula"
    assert np.allclose(d_raw, d_shift), "raw opt-out != paper's shifted gadget node"
    # OUT must never regret positively where the bot is already ≤ blueprint (v_enter ≤ CBV
    # ⇒ opponent prefers OUT ⇒ ENTER regret ≤ 0), a direct check of the maximizer sign.
    safe = v_enter <= cbv
    assert np.all(actual[safe, _OX_ENTER] <= 1e-12), "ENTER regret grew where bot is safe"


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
