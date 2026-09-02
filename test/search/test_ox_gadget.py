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
from environment.poker_env import PokerEnv
from information_abstraction.lookup import clusters_for_board
from poker_ai.search.context import SubgameContext
from poker_ai.search.leaf import LeafConfig
from poker_ai.search.mccfr import _BIAS_CLASSES
from poker_ai.search.policy import Policy
from test.search._helpers import _ctx
from test.search.brute_force_cfr import BruteForceCFR, br_value, build_subgame
from test.search.test_equilibrium_oracle import (
    _install_lossless_lut,
    _remap_row,
    _river_subgame,
    _solver_sigma,
)


def _cfg(leaf, *, beta=None, kbeta=None, iters=400, discount=10):
    return SolverConfig(
        leaf=leaf, beta=beta, ox_kbeta=kbeta, max_iterations=iters,
        max_wall_seconds=1e9, discount_interval=discount, auto_budget=False,
    )


#: Exploitation/safety bracket for the β tradeoff, expressed as **kβ** — the only
#: deck-agnostic form.  β never acts alone: the gadget mixes by ``kβ``
#: (``c_expl = 1/(kβ+1)``), and ``k`` is the count of board-compatible opponent root
#: combos, so it moves with the deck — ``C(47,2) = 1081`` on a 52-card river vs
#: ``C(15,2) = 105`` on this 20-card test deck.
#:
#: The paper specifies ``1/(kβ+1)`` directly: 1/16 (Leduc), 1/51 (Flop Hold'em).
#: The eval's ``DEFAULT_OX_KBETA = 50`` is exactly the paper's FHP setting on any deck
#: — so ``_KB_SAFE = 50`` sits at production's operating point and
#: ``_KB_EXPLOIT = 1`` is well to the exploitative side of it.
#:
#: Raw β would NOT survive a deck change: at β=0.5/50 on this deck, kβ is 52.5/5250, i.e.
#: BOTH arms are at-or-beyond production's conservative point, the gadget saturates
#: (``q_enter`` pinned) and the tradeoff being asserted is unobservable.
_KB_EXPLOIT = 1.0
_KB_SAFE = 50.0


def _normalize(vec):
    v = np.asarray(vec, dtype=np.float64).copy()
    s = v.sum()
    return v / s if s > 0 else v


def _delta(sub):
    """Utility swing Δ = max_z u − min_z u over the subgame (seat-0 payoffs)."""
    vals = [v for leaf in sub.payoff.values() for v in leaf.values()]
    return max(vals) - min(vals)



# --------------------------------------------------------------------------- #
# A blueprint of CONTROLLED strength: (1-ε)·NE + ε·uniform
# --------------------------------------------------------------------------- #
#: How far below Nash the OX fixture's blueprint sits.  Both extremes are degenerate:
#:
#: * a UNIFORM blueprint (what these tests used to use) is ~500-exploitable on a ~1000
#:   scale, so ``CBV_ref`` — the value the opponent banks by opting OUT — is enormous.
#:   Safety guarantees the refined strategy is no more exploitable than the blueprint,
#:   so entering can never compete and the opponent opts out ~96% of the time.  Measured:
#:   ``ox_enter_prob`` 0.036-0.058, and IDENTICAL at β=0.5 and β=50 on some seeds — β has
#:   no lever, so a β-tradeoff assertion is testing nothing.
#: * an EXACT NE blueprint leaves the opponent nothing to gain by opting out either, and
#:   σ' cannot be safer than Nash, so there is no exploitation room to trade against.
#:
#: ε=0.05 puts Δ_bp (the blueprint's own exploitability) at ~20 on a ~1000 scale, about
#: 4-5x the oracle's own residual (~1-4.5 at 8000 iterations), so the blueprint is
#: measurably-but-slightly weaker than Nash — the regime the gadget is designed for.
_BLUEPRINT_EPS = 0.05


class _MixedNashBlueprint(Policy):
    """Serves ``(1-ε)·NE + ε·uniform`` rows, keyed by ``PolicyState.info_set``.

    ``reference.py::_blueprint_sigma`` queries the blueprint as
    ``strategy(env.policy_state_for_cluster(cluster, public=...))``, and ``info_set``
    encodes exactly ``(cluster, history)`` — so a table built by walking the oracle tree
    in lockstep with the env is queryable with no inversion.  Anything not in the table
    (an off-support combo, whose reach is zero anyway) falls back to uniform.
    """

    def __init__(self, rows):
        self._rows = rows

    def strategy(self, state, bias="none"):
        n = len(state.legal_actions)
        if n == 0:
            return np.array([], dtype=np.float32)
        row = self._rows.get(state.info_set)
        if row is None or len(row) != n:
            return np.full(n, 1.0 / n, dtype=np.float32)
        return np.asarray(row, dtype=np.float32)


def _mixed_nash_sigma(sub, ne, eps=_BLUEPRINT_EPS):
    """Oracle-keyed blueprint rows: ``(1-ε)·NE + ε·uniform``."""
    return {
        k: (1.0 - eps) * np.asarray(v, dtype=np.float64)
        + eps * np.full(len(v), 1.0 / len(v))
        for k, v in ne.items()
    }


def _blueprint_from_sigma(env: PokerEnv, sub, bp_sigma, bot_seat):
    """Build a :class:`_MixedNashBlueprint` serving ``bp_sigma`` at the bot's nodes.

    Walks the oracle tree and the env together (the oracle's ``pk`` IS
    ``env.public_key``), so each bot node's ``info_set`` can be materialised for the
    cluster each support hole maps to.  River subgame ⇒ the board is complete and the
    cluster map is fixed, and the lossless LUT makes hole↔cluster a bijection.
    """
    cc = env.combo_cards
    board = np.array([int(c) for c in env.community_cards], dtype=np.int64)
    cluster_of = clusters_for_board(env.card_info_lut["river"], cc, board)
    rows = {}

    def walk(node, e):
        if node["type"] != "node":          # "term" / "chance" carry no bot decision
            return
        pk, seat, legal = node["pk"], node["actor"], node["legal"]
        if seat == bot_seat:
            public = e.policy_public_fields()
            for hole_idx in sub.support[bot_seat]:
                base = bp_sigma.get((seat, hole_idx, pk))
                if base is None:
                    continue
                combo = env.combo_index[sub.holes[hole_idx]]
                cluster = int(cluster_of[combo])
                if cluster < 0:
                    continue
                st = e.policy_state_for_cluster(cluster, public=public)
                rows[st.info_set] = _remap_row(
                    base, list(legal), list(st.legal_actions)
                )
        for a in legal:
            token = e.step_in_place(a, settle_winners=False)
            walk(node["children"][a], e)
            e.undo(token)

    walk(sub.root, env)
    return _MixedNashBlueprint(rows)


def _ctx_with_blueprint(env, ranges, blueprint, seed=7):
    """``_ctx`` but with the whole leaf fleet served by ``blueprint``.

    A river subgame is leaf-free, so only ``policies["none"]`` (the CBV_ref anchor) is
    ever consulted; the other bias classes are wired to the same object so the fleet is
    well-formed rather than half-uniform.
    """
    return SubgameContext.from_runtime(
        env=env,
        my_seat=0,
        my_hole=tuple(int(c) for c in env.players[0].cards),
        ranges=ranges,
        folded_ranges={},
        leaf=LeafConfig(policies={c: blueprint for c in _BIAS_CLASSES}),
        rng=np.random.default_rng(seed),
    )


# --------------------------------------------------------------------------- #
# Gate: β = 0 is a plain best response to the belief p̂.
# --------------------------------------------------------------------------- #
def test_ox_kbeta_zero_matches_belief_best_response(_seeded):
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
    diffs = np.array(
        [np.abs(sa[k] - sb[k]).max() for k in sa if k[0] == bot and k in sb]
    )
    assert diffs.size, "expected bot decision rows to compare"

    # Aggregate over rows, do NOT take the max.  ``c_expl = 1/(kβ+1) ≈ 1e-6`` here, so the
    # belief's influence is ~nil and nearly every row matches to ~1e-7 — but regret
    # matching has ties, and at a tie the two runs can settle on different (equally
    # valued) actions.  That is an arbitrary tie-break, not belief dependence, and a
    # ``max`` over rows turns one of them into a failure: measured across seeds, the
    # median row diff is 1e-7 while 1-2 of 16 rows sit at 1e-2..8e-2.
    #
    # Detection power is kept by the shape of a REAL failure: if the refined strategy
    # still tracked the belief, it would move MOST rows, not one — so the median moves
    # off the floor and the outlier count blows past the cap.  Verified by construction
    # at β=0.01, where the strategy IS supposed to track the belief (seeds 4/5/37):
    #
    #     β=1e6   median 1.2e-07 / 2.6e-06 / 6.0e-08   rows>1e-2:  1 /  2 /  1   (cap 4)
    #     β=0.01  median 4.3e-01 / 5.8e-01 / 3.7e-02   rows>1e-2: 16 / 16 /  9
    #
    # Six orders of magnitude between "holds" and "fails", with the 1e-4 bound in the
    # middle — so this is a sharper gate than the old ``max``, not a looser one.
    median, n_over = float(np.median(diffs)), int((diffs > 1e-2).sum())
    assert median < 1e-4, (
        f"large-β strategy still belief-dependent: median row diff {median:.2e} "
        f"(expected ~1e-7; c_expl≈1e-6)"
    )
    assert n_over <= max(1, diffs.size // 4), (
        f"large-β belief dependence is not confined to tie-breaks: "
        f"{n_over}/{diffs.size} rows differ by > 1e-2 (max {diffs.max():.4f})"
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
    """Thm 4.6 safety: the refined strategy's exploitability exceeds the blueprint's by
    at most ``Δ/β``; and the gadget is not inert (a small kβ genuinely exploits).

    **This gate deliberately does NOT assert that a smaller kβ exploits more.**  That
    ordering is Ge et al.'s result about the METHOD, and it needs opponent-model error to
    exist at all: safety insures against a wrong belief, so where the belief is right,
    safety is pure cost and there is no tradeoff to observe.  This fixture's ``p̂`` sits
    on the opponent's exact true support (weights 1/3, 2/3 against a true 1/2, 1/2), i.e.
    essentially zero model error — and measured, the belief-weighted exploitation (Eq 1
    of the paper) is FLAT in kβ until the safety entry mechanically overtakes the
    exploitation entry, then steps down:

        seed 4, Eq-1 exploitation vs kβ ∈ {1e-3 … 1e3}:
            230.712  230.719  230.566  230.711  230.674  100.019  96.953
        seed 37: 292.219 → 291.763 across the whole range (0.16%) — no headroom at all.

    The step sits where the two entry masses cross, ``kβ·q̄ ≈ 1``, and it is a step rather
    than a ramp because the entry distributions are maximally unlike (``p̂`` on 2 combos
    vs uniform over ``k``): there is no intermediate strategy to interpolate toward.

    The mixture WIRING is covered without that assertion, at both ends:
    :func:`test_ox_kbeta_zero_matches_belief_best_response` pins kβ→0 (byte-identical to a
    vanilla solve against ``p̂``) and :func:`test_ox_large_beta_ignores_belief` pins kβ→∞
    (belief-independent).  A swapped ``c_expl``/``c_safe``, or a bad β-from-kβ
    derivation, fails one of those immediately.  What a monotonicity assertion would add
    on top is the shape of the interpolation between them — the paper's theory, re-proved
    empirically in the one regime where it does not apply.

    Observing the real tradeoff needs a deliberately WRONG ``p̂``, swept over an error
    axis; that is the evaluation design (every OX figure in the paper puts estimation
    error on the x-axis), not a unit test.
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

    def refined(kbeta, iters=1500):
        ctx = _ctx(env, ranges={0: _normalize(r0), 1: _normalize(belief)}, seed=7)
        res = solve(env, ctx, _cfg(ctx.leaf, kbeta=kbeta, iters=iters, discount=50))
        sigma = _solver_sigma(res.state, env, sub)
        exp_prime = br_value(sub, opp, sigma)              # opp BR vs the refined bot
        # Deviation from the blueprint, AGGREGATED over the bot's rows: the
        # mean total-variation distance from uniform.  Not the max over rows
        # of the max element — that reports whichever single row happens to
        # be the most concentrated, which is noise rather than a measure of
        # how far the strategy moved, and it saturates (one fully-committed
        # width-2 row pins it at 0.5 no matter what the rest of σ' does).
        # Measured on all five seeds: the mean-TV metric orders β=0.5 above
        # β=50 every time, while the max metric inverts on two of them.
        dev = float(np.mean([
            0.5 * np.abs(row - 1.0 / len(row)).sum()
            for k, row in sigma.items() if k[0] == bot
        ]))
        return sigma, exp_prime, dev

    tol = 0.03 * delta
    # Thm 4.6 bounds by Δ/β, so recover the β each kβ target actually resolves to.
    k = float(np.asarray(_ctx(env, seed=7).board_compatible, dtype=np.float64).sum())
    beta_safe, beta_expl = _KB_SAFE / k, _KB_EXPLOIT / k
    _, exp_big, _ = refined(_KB_SAFE)          # safety arm: only its exploitability is read
    _, exp_small, dev_small = refined(_KB_EXPLOIT)

    # (1) Safety (Thm 4.6): the refined exploitability stays within Δ/β of the blueprint.
    assert exp_big - exp_bp <= delta / beta_safe + tol, (
        f"kβ={_KB_SAFE} (β={beta_safe:.4g}) unsafe: exp'−exp_bp={exp_big - exp_bp:.4f} "
        f"> Δ/β={delta / beta_safe:.4f}"
    )
    assert exp_small - exp_bp <= delta / beta_expl + tol
    # (2) A small kβ genuinely exploits — the bot deviates from the uniform blueprint.
    assert dev_small > 0.05, (
        f"small kβ did not exploit (mean TV distance from uniform {dev_small:.3f})"
    )
