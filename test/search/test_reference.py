"""Tests for the OX-Search reference CBV pass (poker_ai/search/reference.py, §11.3 step 10).

The reference computes ``CBV₁^σ`` — the opponent's exact best-response counterfactual
value against the bot playing the blueprint — per opponent root combo.  Gate (a): it
must equal an **independent brute-force best response** over the explicit game tree
(river, and turn with the river enumerated).  Gate (b): the ``F(σ)`` self-check — the
reference is deterministic and, fed the blueprint as the bot, reproduces itself
(zero margin), i.e. the feasibility boundary the gadget's safety constraint sits on.

The bot's blueprint here is :class:`UniformPolicy` (uniform over legal), so the
independent BR can use a uniform bot at bot nodes.  ``vector_payout`` weights by the
**raw** bot range (not the normalised chance measure), so the brute-force uses the raw
range too, keeping the two in identical units.
"""

import numpy as np
import pytest

from information_abstraction.lookup import clusters_for_board
from poker_ai.search.context import SubgameContext
from poker_ai.search.leaf import LeafConfig
from poker_ai.search.mccfr import _BIAS_CLASSES
from poker_ai.search.policy import Policy
from poker_ai.search.reference import compute_cbv_ref
from poker_ai.search.solver_state import SolverConfig
from test.search._helpers import _ctx, _real_lut_env
from test.search.brute_force_cfr import build_subgame, build_turn_subgame
from test.search.test_equilibrium_oracle import (
    _install_lossless_lut,
    _river_subgame,
    _turn_subgame,
)

_STREET_NAME = {0: "pre_flop", 1: "flop", 2: "turn", 3: "river"}


def _cfg(ctx):
    return SolverConfig(leaf=ctx.leaf, max_iterations=1, max_wall_seconds=1e9)


def _scale(sub) -> float:
    vals = [abs(v) for leaf in sub.payoff.values()
            for v in (leaf.values() if isinstance(leaf, dict) else [])]
    return max(vals) if vals else 1.0


# --------------------------------------------------------------------------- #
# Independent brute-force CBV of the opponent vs a uniform bot (raw reach).
# --------------------------------------------------------------------------- #
def _opp_cbv_river(sub, bot_seat, opp_seat, bot_range):
    """Per opp-hole ``CBV`` vs the bot playing uniform, weighted by raw ``bot_range``.

    Backward induction on the static river tree: max at opponent nodes (best
    response), uniform average at bot nodes, and at a terminal a sum over
    card-disjoint bot holes weighted by the raw bot range (``vector_payout`` units).
    """
    def rec(node, oh, bot_reach):
        if node["type"] == "term":
            v = 0.0
            for bh, r in bot_reach.items():
                pair = (oh, bh) if opp_seat == 0 else (bh, oh)
                p0 = sub.payoff[node["leaf"]].get(pair)
                if p0 is not None:
                    v += r * (p0 if opp_seat == 0 else -p0)
            return v
        legal = node["legal"]
        if node["actor"] == opp_seat:
            return max(rec(node["children"][a], oh, bot_reach) for a in legal)
        w = 1.0 / len(legal)                       # uniform bot
        return sum(rec(node["children"][a], oh, {bh: r * w for bh, r in bot_reach.items()})
                   for a in legal)

    out = {}
    for oh in sub.support[opp_seat]:
        reach = {bh: float(bot_range[bh]) for bh in sub.support[bot_seat]
                 if not (set(sub.holes[oh]) & set(sub.holes[bh]))}
        out[oh] = rec(sub.root, oh, reach) if reach else 0.0
    return out


def _opp_cbv_turn(sub, bot_seat, opp_seat, bot_range):
    """Per opp-hole ``CBV`` for a turn subgame (river chance enumerated, ``1/R``)."""
    rivers = sub.rivers

    def rec(node, oh, bot_reach, river):
        ntype = node["type"]
        if ntype == "term":
            row = sub.payoff[node["leaf"]]
            v = 0.0
            for bh, r in bot_reach.items():
                if opp_seat == 0:
                    key = (oh, bh) if river is None else (oh, bh, river)
                else:
                    key = (bh, oh) if river is None else (bh, oh, river)
                p0 = row.get(key)
                if p0 is not None:
                    v += r * (p0 if opp_seat == 0 else -p0)
            return v
        if ntype == "chance":
            inv = 1.0 / len(rivers)
            v = 0.0
            for rr in rivers:
                if rr in sub.holes[oh]:
                    continue                       # opp cannot see its own card
                sub_reach = {bh: rv for bh, rv in bot_reach.items() if rr not in sub.holes[bh]}
                if sub_reach:
                    v += inv * rec(node["child"], oh, sub_reach, int(rr))
            return v
        legal = node["legal"]
        if node["actor"] == opp_seat:
            return max(rec(node["children"][a], oh, bot_reach, river) for a in legal)
        w = 1.0 / len(legal)                       # uniform bot
        return sum(rec(node["children"][a], oh, {bh: rv * w for bh, rv in bot_reach.items()}, river)
                   for a in legal)

    out = {}
    for oh in sub.support[opp_seat]:
        reach = {bh: float(bot_range[bh]) for bh in sub.support[bot_seat]
                 if not (set(sub.holes[oh]) & set(sub.holes[bh]))}
        out[oh] = rec(sub.root, oh, reach, None) if reach else 0.0
    return out


# --------------------------------------------------------------------------- #
# Gate (a) — brute-force best response
# --------------------------------------------------------------------------- #
def test_cbv_ref_river_matches_bruteforce(_seeded):
    """River root: per-combo CBV_ref == the independent brute-force BR (vs uniform bot)."""
    env, r0, r1, s0, s1 = _river_subgame(_seeded)
    _install_lossless_lut(env)
    ctx = _ctx(env, ranges={0: r0, 1: r1}, seed=7)      # my_seat=0 (bot), opp=1
    cbv = compute_cbv_ref(env, ctx, _cfg(ctx))

    sub = build_subgame(env, r0, r1, s0, s1)
    ref = _opp_cbv_river(sub, bot_seat=0, opp_seat=1, bot_range=r0)
    scale = _scale(sub)
    for oh, val in ref.items():
        assert np.isclose(cbv[oh], val, atol=1e-6 * scale + 1e-9), (
            f"river CBV_ref[{oh}]={cbv[oh]:.6f} != brute {val:.6f} (scale {scale})"
        )


def test_cbv_ref_turn_matches_bruteforce(_seeded):
    """Turn root: CBV_ref (river integrated mid-tree) == brute-force BR (river enumerated).

    This is the gate on the exact chance integration — the opponent's turn best
    response must max over the river *expectation*, which only holds if the river is
    integrated below each turn action (not chosen clairvoyantly).
    """
    env, r0, r1, s0, s1 = _turn_subgame(_seeded)
    _install_lossless_lut(env)
    ctx = _ctx(env, ranges={0: r0, 1: r1}, seed=7)
    cbv = compute_cbv_ref(env, ctx, _cfg(ctx))

    sub = build_turn_subgame(env, r0, r1, s0, s1)
    ref = _opp_cbv_turn(sub, bot_seat=0, opp_seat=1, bot_range=r0)
    scale = max(
        abs(v) for leaf in sub.payoff.values() for v in leaf.values()
    )
    for oh, val in ref.items():
        assert np.isclose(cbv[oh], val, atol=1e-6 * scale + 1e-9), (
            f"turn CBV_ref[{oh}]={cbv[oh]:.6f} != brute {val:.6f} (scale {scale})"
        )


# --------------------------------------------------------------------------- #
# Gate (b) — F(σ) self-check + basic invariants
# --------------------------------------------------------------------------- #
def test_cbv_ref_is_deterministic(_seeded):
    """No RNG: two passes on the same subgame return identical CBV_ref (F(σ) stability)."""
    env, r0, r1, s0, s1 = _river_subgame(_seeded)
    _install_lossless_lut(env)
    ctx = _ctx(env, ranges={0: r0, 1: r1}, seed=7)
    a = compute_cbv_ref(env, ctx, _cfg(ctx))
    b = compute_cbv_ref(env, ctx, _cfg(ctx))
    assert np.array_equal(a, b)


def test_cbv_ref_zero_on_board_infeasible_combos(_seeded):
    """A combo sharing a card with the community has no reachable terminal → CBV_ref 0."""
    env, r0, r1, s0, s1 = _river_subgame(_seeded)
    _install_lossless_lut(env)
    ctx = _ctx(env, ranges={0: r0, 1: r1}, seed=7)
    cbv = compute_cbv_ref(env, ctx, _cfg(ctx))

    board = {int(c) for c in env.community_cards}
    infeasible = [i for i in range(env.n_combos)
                  if set(int(x) for x in env.combo_cards[i]) & board]
    assert infeasible, "expected at least one board-conflicting combo"
    assert np.allclose(cbv[infeasible], 0.0)


# =========================================================================== #
# Gate (c) — cluster-ABSTRACTED reference (a real lossy LUT + a non-uniform bot).
# =========================================================================== #
#
# The gates above use a LOSSLESS LUT and a UNIFORM policy, so they cannot see two
# whole classes of bug in ``_blueprint_sigma``: (1) the future-street cluster gather
# (``cluster_of``/``universe``/``gather_of`` → ``rows[gof]``), which is identity when
# cluster == combo, and (2) any per-combo cluster mis-key, which a uniform policy
# returns the same row for regardless.  These gates drive the reference under
# ``data/20cards_exact`` (25/50/50/45 clusters, so ``n_rows >> 1``) with a synthetic
# blueprint whose row VARIES by cluster, and check it against an oracle that recovers
# each combo's cluster straight from ``clusters_for_board`` — bypassing the
# ``ClusterMapper`` dense/universe/gather layer the reference relies on.  A real
# on-disk blueprint is deliberately NOT used (the synthetic one is self-contained and
# needs no artifact); the LUT is gated with ``requires_lut``.


def _leading_uvarint(buf: bytes) -> int:
    """Decode the leading unsigned-LEB128 varint of an ``info_set`` — the cluster id.

    ``encode_info_set`` writes the cluster first via ``_put_uvarint`` (poker_env.py),
    so this recovers exactly the cluster the reference fed the policy, letting the
    synthetic blueprint key on it without decoding history.
    """
    val = shift = 0
    for byte in buf:
        val |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return val
        shift += 7
    raise ValueError("truncated uvarint in info_set")


def _cluster_dist(cluster: int, width: int) -> np.ndarray:
    """Deterministic, strictly-positive, non-uniform row keyed by ``(cluster, width)``.

    Distinct clusters get distinct rows, so a wrong cluster (or a broken cluster→combo
    gather) produces a wrong row and the oracle catches it — which a uniform policy
    cannot.  A pure function of its inputs: the reference (cluster recovered from the
    info_set the ClusterMapper built) and the oracle (cluster from ``clusters_for_board``)
    must land on the identical row.
    """
    if width <= 0:
        return np.zeros(0, dtype=np.float64)
    k = np.arange(1, width + 1, dtype=np.int64)
    w = ((int(cluster) * 2654435761 + k * 40503) % 251 + 1).astype(np.float64)
    return w / w.sum()


class _ClusterKeyedPolicy(Policy):
    """Blueprint stand-in: a non-uniform action row keyed by the info_set's cluster."""

    def strategy(self, state, bias="none"):
        row = _cluster_dist(_leading_uvarint(state.info_set), len(state.legal_actions))
        return row.astype(np.float32)


def _raw_clusters(env, street: int, board) -> np.ndarray:
    """Independent per-combo raw LUT cluster ids for ``board`` (−1 on card conflict).

    Calls ``clusters_for_board`` directly — the ground-truth abstraction lookup that
    ``ClusterMapper`` wraps — so the oracle never touches the dense-row/universe/gather
    machinery the reference is being tested on.
    """
    sub = env.card_info_lut[_STREET_NAME[street]]
    return clusters_for_board(sub, env.combo_cards, np.asarray(board, dtype=np.int64))


def _cluster_ctx(env, ranges, seed=0):
    """A :class:`SubgameContext` whose leaf fleet is the cluster-keyed blueprint."""
    leaf = LeafConfig(policies={c: _ClusterKeyedPolicy() for c in _BIAS_CLASSES},
                      n_rollouts=2)
    return SubgameContext.from_runtime(
        env=env, my_seat=0,
        my_hole=tuple(int(c) for c in env.players[0].cards),
        ranges=ranges, folded_ranges={}, leaf=leaf,
        rng=np.random.default_rng(seed),
    )


def _spread_support(env, n_each=6):
    """Two disjoint sets of spread-out, board-compatible combo indices (bot, opp)."""
    board = {int(c) for c in env.community_cards}
    compat = [i for i in range(env.n_combos)
              if not (set(int(x) for x in env.combo_cards[i]) & board)]
    n_each = min(n_each, len(compat) // 2)
    step = max(1, len(compat) // (2 * n_each))
    sel = compat[::step][:2 * n_each]
    return sel[0::2], sel[1::2]


def _cluster_ranges(env, support0, support1):
    """Non-uniform bot range (stresses reach-weighting) + a flat opponent range."""
    r0 = np.zeros(env.n_combos, dtype=np.float64)
    r1 = np.zeros(env.n_combos, dtype=np.float64)
    for j, b in enumerate(support0):
        r0[b] = float(j + 1)
    for oh in support1:
        r1[oh] = 1.0
    return r0, r1


# --- Independent CBV oracles: the uniform-bot walk, but the bot plays the ------ #
# --- cluster-keyed blueprint (cluster from clusters_for_board, per street). ----- #
def _opp_cbv_river_bp(sub, env, bot_seat, opp_seat, bot_range):
    rc = _raw_clusters(env, 3, [int(c) for c in env.community_cards])

    def rec(node, oh, bot_reach):
        if node["type"] == "term":
            v = 0.0
            for bh, r in bot_reach.items():
                pair = (oh, bh) if opp_seat == 0 else (bh, oh)
                p0 = sub.payoff[node["leaf"]].get(pair)
                if p0 is not None:
                    v += r * (p0 if opp_seat == 0 else -p0)
            return v
        legal = node["legal"]
        if node["actor"] == opp_seat:
            return max(rec(node["children"][a], oh, bot_reach) for a in legal)
        width = len(legal)
        total = 0.0
        for k, a in enumerate(legal):
            nr = {bh: r * _cluster_dist(int(rc[bh]), width)[k]
                  for bh, r in bot_reach.items()}
            total += rec(node["children"][a], oh, nr)
        return total

    out = {}
    for oh in sub.support[opp_seat]:
        reach = {bh: float(bot_range[bh]) for bh in sub.support[bot_seat]
                 if not (set(sub.holes[oh]) & set(sub.holes[bh]))}
        out[oh] = rec(sub.root, oh, reach) if reach else 0.0
    return out


def _opp_cbv_turn_bp(sub, env, bot_seat, opp_seat, bot_range):
    rivers = sub.rivers
    tboard = [int(c) for c in env.community_cards]
    tc = _raw_clusters(env, 2, tboard)                  # 4-card turn clusters
    rc_cache = {}

    def river_clusters(rr):
        if rr not in rc_cache:
            rc_cache[rr] = _raw_clusters(env, 3, tboard + [int(rr)])
        return rc_cache[rr]

    def rec(node, oh, bot_reach, river):
        ntype = node["type"]
        if ntype == "term":
            row = sub.payoff[node["leaf"]]
            v = 0.0
            for bh, r in bot_reach.items():
                if opp_seat == 0:
                    key = (oh, bh) if river is None else (oh, bh, river)
                else:
                    key = (bh, oh) if river is None else (bh, oh, river)
                p0 = row.get(key)
                if p0 is not None:
                    v += r * (p0 if opp_seat == 0 else -p0)
            return v
        if ntype == "chance":
            inv = 1.0 / len(rivers)
            v = 0.0
            for rr in rivers:
                if rr in sub.holes[oh]:
                    continue
                sub_reach = {bh: rv for bh, rv in bot_reach.items()
                             if rr not in sub.holes[bh]}
                if sub_reach:
                    v += inv * rec(node["child"], oh, sub_reach, int(rr))
            return v
        legal = node["legal"]
        if node["actor"] == opp_seat:
            return max(rec(node["children"][a], oh, bot_reach, river) for a in legal)
        width = len(legal)
        clv = tc if river is None else river_clusters(river)
        total = 0.0
        for k, a in enumerate(legal):
            nr = {bh: rv * _cluster_dist(int(clv[bh]), width)[k]
                  for bh, rv in bot_reach.items()}
            total += rec(node["children"][a], oh, nr, river)
        return total

    out = {}
    for oh in sub.support[opp_seat]:
        reach = {bh: float(bot_range[bh]) for bh in sub.support[bot_seat]
                 if not (set(sub.holes[oh]) & set(sub.holes[bh]))}
        out[oh] = rec(sub.root, oh, reach, None) if reach else 0.0
    return out


@pytest.mark.requires_lut
def test_cbv_ref_river_cluster_abstracted():
    """River root, real lossy LUT + non-uniform bot: CBV_ref == the independent oracle.

    Exercises the ``is_root`` cluster query/broadcast in ``_blueprint_sigma`` with real
    multi-combo-per-cluster grouping — invisible under the lossless-LUT gates.
    """
    env = _real_lut_env(3, stacks=(200, 200), seed=1)
    support0, support1 = _spread_support(env)
    rc = _raw_clusters(env, 3, [int(c) for c in env.community_cards])
    assert len({int(rc[b]) for b in support0}) > 1, "support not cluster-diverse"

    r0, r1 = _cluster_ranges(env, support0, support1)
    ctx = _cluster_ctx(env, {0: r0, 1: r1})
    cbv = compute_cbv_ref(env, ctx, _cfg(ctx))

    sub = build_subgame(env, r0, r1, support0, support1)
    ref = _opp_cbv_river_bp(sub, env, 0, 1, r0)
    scale = _scale(sub)
    for oh, val in ref.items():
        assert np.isclose(cbv[oh], val, atol=1e-6 * scale + 1e-9), (
            f"river cluster CBV_ref[{oh}]={cbv[oh]:.6f} != oracle {val:.6f}"
        )


@pytest.mark.requires_lut
def test_cbv_ref_turn_cluster_abstracted():
    """Turn root, real lossy LUT + non-uniform bot: CBV_ref == the independent oracle.

    The critical gate: a turn root makes the river a FUTURE street, so
    ``_blueprint_sigma`` takes its cluster-gather path (``cluster_of``/``universe`` →
    ``rows[gof]``).  The oracle recovers the same clusters straight from
    ``clusters_for_board``, so a dense-row ↔ raw-cluster mismatch would diverge here.
    """
    env = _real_lut_env(2, stacks=(200, 200), seed=1)
    support0, support1 = _spread_support(env)
    tboard = [int(c) for c in env.community_cards]
    assert len({int(c) for c in _raw_clusters(env, 2, tboard)[support0]}) > 1, (
        "turn support not cluster-diverse"
    )
    # And the river (future) street must key >1 cluster over the bot support for at
    # least one river, or the gather path is trivially single-row.
    rivers = sorted({int(x) for x in np.unique(env.combo_cards)} - set(tboard))
    assert any(
        len({int(c) for c in _raw_clusters(env, 3, tboard + [rr])[support0]}) > 1
        for rr in rivers
    ), "no river yields cluster-diverse bot support"

    r0, r1 = _cluster_ranges(env, support0, support1)
    ctx = _cluster_ctx(env, {0: r0, 1: r1})
    cbv = compute_cbv_ref(env, ctx, _cfg(ctx))

    sub = build_turn_subgame(env, r0, r1, support0, support1)
    ref = _opp_cbv_turn_bp(sub, env, 0, 1, r0)
    scale = max(abs(v) for leaf in sub.payoff.values() for v in leaf.values())
    for oh, val in ref.items():
        assert np.isclose(cbv[oh], val, atol=1e-6 * scale + 1e-9), (
            f"turn cluster CBV_ref[{oh}]={cbv[oh]:.6f} != oracle {val:.6f}"
        )
