"""Learn an opponent model: P(action | situation, hand strength).  No blueprint.

Two phases over one hand budget:

1. **Population** — play every archetype, fit what "a player" does.
2. **Refinement** — play one opponent, let its parameters move off the population fit by
   as much as its own data supports.

**The model.**  Per situation ``C = (street, facing-a-bet)``, a multinomial logit that is
continuous in hand strength::

    P(k | C, x) = exp(a_k + b_k x) / sum over LEGAL j of exp(a_j + b_j x)

Hand strength enters as TWO features, both centred on zero and both public given the
holding:

* ``x1`` — the holding's percentile among the holdings still LIVE on this board at this
  node.  Relative position: how many of the hands the opponent could hold beat this one.
  Computed per node, not from a global per-street table — a cluster's rank among the 50
  clusters is not a hand's rank among hands, and cluster occupancy differs by street, so
  the old axis put the flop at mean 0.41 and the turn at 0.62.  The same holding scored
  differently depending only on which street asked, which inverted the fitted flop
  coefficients (fold 93% with STRONG hands).
* ``x3`` — the SPREAD of the holding's EHS histogram: drawiness.  The mean alone cannot
  describe a flop hand — a made hand and a draw with the same average equity share a mean
  and are played very differently, and the blueprint plays the whole cluster.  Measured:
  ranking by the mean alone, the opponent appears to fold STRONGER hands than it calls
  with on the flop (0.58 vs 0.32), is flat on the turn, and is ordered correctly on the
  river.  That gradient tracks how much future each street has left, i.e. exactly where
  the mean stops being sufficient.
* ``x4`` — POT ODDS, ``to_call / (pot + to_call)``: the price being offered.  The
  situation label only says "facing a bet", so without this a min-bet and an all-in are
  the same spot — yet folding correctly at 4-to-1 and at 1-to-2 are different decisions.
  It is public, known at prediction time, and was simply missing; it is the only variable
  here that the model previously could not see at all rather than merely see coarsely.
* ``x2`` — the holding's absolute equity.  Percentile alone hides how far below the top
  you are: sixtieth percentile on a dry board and sixtieth on a four-flush board are
  different spots, and only the second is a fold.  Aces with no flush on a monotone board
  rank mid-pack and are worth almost nothing; ``x1`` says mid-pack, ``x2`` says worthless,
  and the pair says fold.

Board texture is therefore captured without a texture feature: it shows up as the gap
between where you rank and what you are worth.

Normalising over the **legal set at the node** is what makes this well-posed.  An earlier
attempt estimated ``P(take k | k available)`` per class and renormalised afterwards; that
is only valid if availability is independent of choice, and it collapsed to the
most-common-class predictor whenever the slope vanished.

**The likelihood is per HAND, not per decision** — this is the part that uses all the
information::

    L(hand) = sum_c  pi_0(c) * prod_t P(a_t | C_t, x_t(c))

``pi_0`` is the prior over the opponent's holdings (board- and hero-masked uniform, all
public).  The product runs over that opponent's decisions in the hand, and because every
factor shares the same ``c``, the mixture IS the belief: calling the flop re-weights which
holdings are plausible on the turn, with no separate update step and nothing to bootstrap.

Three kinds of evidence fall out of this one expression rather than needing special cases:

* **showdown** — the hole is revealed, so ``pi_0`` collapses to a point and every decision
  in the hand carries an exact ``x``;
* **folding** — a fold is an ordinary factor, and the hands that could have produced it
  are exactly the ones the mixture keeps weight on;
* **reaching the end without showdown** — still informative, because the whole action
  sequence had to come from one holding in the range.

Folds never reach showdown, so no fold is ever seen with a known strength.  What
identifies folding behaviour is the contrast: showdowns reveal the strengths of hands that
CONTINUED, and the known prior reveals by subtraction what must have folded.  That only
works if both are terms in the same likelihood.

**Refinement** is penalised maximum likelihood: the player's parameters maximise their own
log-likelihood minus ``lambda * ||theta - theta_pop||^2``.  With no data the penalty wins
and the player *is* the population; with plenty, the likelihood wins.

``lambda`` is MEASURED, not chosen.  Under a Gaussian prior ``theta ~ N(theta_pop, tau^2)``
the penalty is exactly ``lambda = 1 / (2 tau^2)``, so the question is how far archetypes
genuinely sit from the population — which the population phase can answer, since it plays
every archetype.  Fitting each archetype separately gives their spread, but that spread is
inflated by each fit's own noise, so it is corrected by a SPLIT-HALF estimate: refit each
archetype on each half of its hands, and ``(theta_A - theta_B)^2 / 4`` estimates the
sampling variance of a full fit without needing a Hessian.  ``tau^2 = between - within``,
pooled across parameters because per-parameter estimates from six archetypes are too
noisy to invert.

**The belief is the search's own range tracker.**  ``poker_ai.search.ranges.RangeTracker``
already maintains a dense per-combo range and Bayes-updates it against an observed action
via ``sigma_for_combo`` — the same machinery the solver uses, and the same machinery DBR
will later exploit.  Learning over it rather than over a hand-rolled belief means the
model is estimated on exactly the representation it will be consumed through; nothing is
added here beyond a lighter parameterisation to make the fit feasible.  It also brings
card removal, board re-masking, fold-time range retention and a counted degeneracy signal
(``fallback_count``) that a hand-rolled version kept getting wrong.

The likelihood keeps its own trajectory representation, because there the belief has to be
a FUNCTION of the parameters being fitted — it changes at every optimiser step — whereas
the tracker maintains a point belief under a fixed policy.

The blueprint appears nowhere in the model: not as a prior, not as a base, not in the
belief.  It drives the hero and defines the archetypes; the LUT supplies each holding's
strength rank, which is public.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def _load_shared():
    from importlib.machinery import SourceFileLoader

    here = Path(__file__).resolve().parent
    return SourceFileLoader(
        "_type_profiles", str(here / "measure_type_separability.py")
    ).load_module()


_P = _load_shared()
PROFILES = _P.PROFILES
_BoardView = _P._BoardView
_class_of = _P._class_of
_entry = _P._entry
_kernel_weight = _P._kernel_weight
cluster_strength = _P.cluster_strength
cluster_spread = _P.cluster_spread


def cluster_potential(lut_path) -> dict:
    """Per-cluster ``(up, dn, top, bot)`` per street, from the EHS centroids.

    ``cluster_spread`` is a standard deviation and therefore symmetric, so it cannot tell
    a hand that can only improve from one that can only decay.  Billings et al. separate
    those as PPot/NPot; the centroids are full EHS histograms, so both sides are already
    on disk.  ``up``/``dn`` are the upside/downside semideviations; ``top``/``bot`` are the
    tail masses above 0.8 and below 0.2 — the polarisation an equity mean cannot express.
    """
    import joblib

    cents = joblib.load(Path(lut_path) / "centroids.joblib")
    out = {}
    for street, arr in cents.items():
        a = np.asarray(arr, dtype=np.float64)
        b = (np.arange(a.shape[1]) + 0.5) / a.shape[1]
        m = a @ b
        d = b[None, :] - m[:, None]
        out[street] = ((a * np.maximum(d, 0.0)).sum(axis=1),
                       (a * np.maximum(-d, 0.0)).sum(axis=1),
                       a[:, b > 0.8].sum(axis=1), a[:, b < 0.2].sum(axis=1))
    return out
cluster_rank = _P.cluster_rank

_POSTFLOP = ("flop", "turn", "river")
_ACTION_CLASSES = ("fold", "call", "raise")
# Parameters per situation = 2 non-reference classes x (DEG + 1) coefficients.  DEG = 1
# is a straight logit in strength; DEG = 2 adds curvature, which is exactly the model you
# get from per-class Gaussian strength distributions with class-specific variance.
DEG = 1

# ---------------------------------------------------------------------------
# Deployed feature set (measured in scripts/probe_feature_ceiling.py)
# ---------------------------------------------------------------------------
# The ceiling probe scored candidate blocks by total variation against the opponent's
# TRUE action distribution, on 183,922 decisions across six archetypes.  Two blocks paid
# and are deployed here; the rest were measured and dropped.
#
#   tokens (bag)            +0.028 TV   the line, counted per (stage, actor, token)
#   potential + polarisation +0.020 TV   PPot/NPot split and tail mass
#   together                +0.052 TV   0.265 -> 0.213
#
# Dropped after measuring: cross-street strength flow (+0.003), position (+0.003),
# x1*odds (+0.001), board texture (+0.006), strength bins (worse than the RBF), and the
# 26-column token ORDER block, which tied the 68-column bag exactly — the counts already
# carry the sequence, so ordering is dead weight.
#
# Split by what the feature depends on, which is a memory requirement rather than taste:
# the hand block varies per candidate holding and lives in ``Hand.X`` (J, T, .); the
# context block is holding-INDEPENDENT and lives in ``Hand.Z`` (T, .).  Tiling 60 token
# columns across ~190 trajectories would inflate the design ~50x for no information.
_BAG_KEYS = [('pre_flop', 0, 'all_in'), ('pre_flop', 0, 'call'), ('pre_flop', 0, 'raise:1.0'), ('pre_flop', 0, 'raise:1.33'), ('pre_flop', 0, 'raise:1.5'), ('pre_flop', 0, 'raise:1.7'), ('pre_flop', 0, 'raise:2.25'), ('pre_flop', 0, 'skip'), ('pre_flop', 1, 'all_in'), ('pre_flop', 1, 'call'), ('pre_flop', 1, 'raise:1.0'), ('pre_flop', 1, 'raise:1.33'), ('pre_flop', 1, 'raise:1.5'), ('pre_flop', 1, 'raise:1.7'), ('pre_flop', 1, 'raise:2.25'), ('pre_flop', 1, 'skip'), ('flop', 0, 'all_in'), ('flop', 0, 'call'), ('flop', 0, 'raise:0.33'), ('flop', 0, 'raise:0.75'), ('flop', 0, 'raise:1.0'), ('flop', 0, 'raise:1.5'), ('flop', 0, 'raise:2.0'), ('flop', 0, 'skip'), ('flop', 1, 'all_in'), ('flop', 1, 'call'), ('flop', 1, 'raise:0.33'), ('flop', 1, 'raise:0.75'), ('flop', 1, 'raise:1.0'), ('flop', 1, 'raise:1.5'), ('flop', 1, 'raise:2.0'), ('flop', 1, 'skip'), ('turn', 0, 'all_in'), ('turn', 0, 'call'), ('turn', 0, 'raise:0.33'), ('turn', 0, 'raise:0.75'), ('turn', 0, 'raise:1.5'), ('turn', 0, 'skip'), ('turn', 1, 'all_in'), ('turn', 1, 'call'), ('turn', 1, 'raise:0.33'), ('turn', 1, 'raise:0.75'), ('turn', 1, 'raise:1.5'), ('turn', 1, 'skip'), ('river', 0, 'all_in'), ('river', 0, 'call'), ('river', 0, 'raise:0.33'), ('river', 0, 'raise:0.75'), ('river', 0, 'raise:1.0'), ('river', 0, 'raise:1.5'), ('river', 0, 'raise:2.0'), ('river', 0, 'skip'), ('river', 1, 'all_in'), ('river', 1, 'call'), ('river', 1, 'raise:0.33'), ('river', 1, 'raise:0.75'), ('river', 1, 'raise:1.0'), ('river', 1, 'raise:1.5'), ('river', 1, 'raise:2.0'), ('river', 1, 'skip')]
_CTX_NAMES = ("odds",) + tuple("tb_%s_%d_%s" % (st, a, t) for st, a, t in _BAG_KEYS)
# Set once from the population phase: token columns that never fire on this deck/grid
# carry a parameter each and no signal.
Z_KEEP = np.arange(len(_CTX_NAMES))
# "base" reproduces the pre-probe feature set (x1, x2, x3, pot odds) so the deployed set
# can be A/B'd against it under identical scoring; "deployed" adds the two blocks the
# ceiling probe showed paying.
FEATURE_SET = "deployed"
# Whether the CONTEXT coefficients are shared across situations.  They are the whole
# parameter problem: giving each situation its own copy of every token coefficient makes
# the context block multiply by n_sit, which is the axis that grows (6 postflop cells here,
# ~18 once multiway adds a live-player dimension).  Sharing costs little information
# because the token columns are ALREADY stage- and actor-tagged — `tb_fl_0_r075` is "bet
# 0.75 pot on the flop", so a separate per-situation coefficient largely re-encodes a
# distinction the feature name already carries.  Hand coefficients and intercepts stay
# per-situation, where the street genuinely changes their meaning.
#   6 sit:  6*2*48 = 576  ->  6*2*8 + 2*40 = 176
#  18 sit: 18*2*48 = 1728 -> 18*2*8 + 2*40 = 368
SHARE_CTX = False
_BAG_INDEX = {k: i for i, k in enumerate(_BAG_KEYS)}


def set_ctx_keep(hands, min_rate: float = 0.005) -> int:
    """Keep pot odds plus token columns that actually fire; drop the rest.

    Most of the 60 (stage, actor, token) cells are structurally unreachable on a given
    grid, and a column that is always zero still costs one parameter per situation per
    action class while carrying no signal — which is pure variance in the per-player
    refinement, where data is scarcest.  Derived from the POPULATION phase only, then held
    fixed, so the two phases share one parameter vector.
    """
    global Z_KEEP
    if FEATURE_SET != "deployed":
        Z_KEEP = np.array([0])          # pot odds only
        return 1
    if not hands:
        return int(Z_KEEP.size)
    Z = np.concatenate([h.Z for h in hands if h.Z is not None and h.Z.size], axis=0)
    rate = (np.abs(Z) > 1e-12).mean(axis=0)
    keep = np.flatnonzero(rate >= min_rate)
    if keep.size == 0 or 0 not in keep:
        keep = np.unique(np.concatenate([[0], keep]))
    Z_KEEP = keep
    return int(Z_KEEP.size)


def _action_class(a: str) -> str:
    if a == "fold":
        return "fold"
    if a.startswith("raise") or a == "all_in":
        return "raise"
    return "call"


BASIS = "poly"
RBF_C = np.linspace(-0.5, 0.5, 5)
RBF_H = 0.18


def _feats(raw: np.ndarray, z=None) -> np.ndarray:
    """Design matrix from the raw strength features.  ``raw`` is ``(n, 2)``: ``(x1, x2)``.

    ``poly``: ``[1, x, x^2, ...]`` — smooth but monotone-or-single-turning, so it cannot
    put mass at both ends of the strength range with a trough between.

    ``rbf``: fixed Gaussian bumps at spread centres.  Same expressiveness as a Gaussian
    mixture over strength for our purposes, but LINEAR IN THE PARAMETERS, so the fit
    stays convex and the closed-form gradient is unchanged — a mixture with free means
    would be neither.  This is what can represent a POLARISED range (raise with the nuts
    and with air, check everything between), which the showdown-labelled data shows on
    the later streets and which is the shape the target error needs.
    """
    raw = np.atleast_2d(raw)
    x1, x2, x3 = raw[:, 0], raw[:, 1], raw[:, 2]
    up, dn, top, bot = raw[:, 3], raw[:, 4], raw[:, 5], raw[:, 6]
    if BASIS == "rbf":
        B = np.exp(-0.5 * ((x1[:, None] - RBF_C[None, :]) / RBF_H) ** 2)
    else:
        B = np.stack([x1 ** d for d in range(DEG + 1)], axis=1)
    if FEATURE_SET == "deployed":
        F = np.concatenate([B, x2[:, None], x3[:, None], up[:, None], dn[:, None],
                            top[:, None], bot[:, None]], axis=1)
    else:
        F = np.concatenate([B, x2[:, None], x3[:, None]], axis=1)
    if z is None:
        return F
    # Holding-independent columns are identical for every candidate, so broadcast rather
    # than store them per trajectory.
    zk = np.asarray(z, dtype=np.float64).ravel()[Z_KEEP]
    return np.concatenate([F, np.tile(zk, (F.shape[0], 1))], axis=1)


_T0 = time.time()


def _log(msg: str) -> None:
    """Timestamped progress line, flushed.

    Long runs were previously silent between "phase 2 begins" and every trial finishing at
    once, so a run that had quietly become 3x slower looked identical to one about to
    finish.  Worth the two lines: it also reports the collect-vs-fit split, which is the
    number needed to size the next run rather than guess at it.
    """
    el = time.time() - _T0
    print("[%3d:%02d] %s" % (int(el // 60), int(el % 60), msg), flush=True)


def tables_of(type_list, n_opp):
    """Chunk archetypes into tables of ``n_opp``, cycling to fill the last one."""
    if n_opp <= 1:
        return [(t,) for t in type_list]
    out_t, i = [], 0
    while i < len(type_list):
        chunk = list(type_list[i:i + n_opp])
        while len(chunk) < n_opp:                 # pad from the front, never empty
            chunk.append(type_list[len(chunk) % len(type_list)])
        out_t.append(tuple(chunk))
        i += n_opp
    return out_t


def seat_map(hand: int, n_p: int, n_opp: int):
    """``(hero_seat, {identity: seat}, {seat: identity})`` for one hand.

    Both the hero seat and the identity-to-seat map rotate, so an archetype is observed
    from every position.  Without this, position is confounded with type — a nit that
    always sits on the button produces a positional effect the model would attribute to
    the archetype.
    """
    hero = hand % n_p
    opp_seats = [s for s in range(n_p) if s != hero]
    seat_of = {i: opp_seats[(i + hand) % n_opp] for i in range(n_opp)}
    return hero, seat_of, {v: k for k, v in seat_of.items()}


def _token_bag(acts, opp) -> list:
    """Counts over the repo's OWN action vocabulary, per (stage, actor, token).

    ``encode_info_set`` keys the blueprint on exactly these tokens, so counting them is a
    mechanical projection of the real key rather than a hand-picked summary.  That matters
    empirically: a nine-scalar hand-rolled summary of the same line scored +0.018 TV, this
    bag +0.028, and the difference was information the hand-rolling silently discarded
    (which street each action fell on, and the size used there).
    """
    idx = _BAG_INDEX
    bag = [0.0] * len(_BAG_KEYS)
    for st, evs in acts.items():
        for pl, t in evs:
            j = idx.get((st, 0 if pl == opp else 1, t))
            if j is not None:
                bag[j] += 1.0
    return bag


def _n_hand() -> int:
    """Holding-dependent columns: strength basis, equity, spread, and (deployed) PPot/NPot."""
    b = RBF_C.size if BASIS == "rbf" else DEG + 1
    return b + (6 if FEATURE_SET == "deployed" else 2)


def _n_ctx() -> int:
    return int(Z_KEEP.size)


def _n_feats() -> int:
    return _n_hand() + _n_ctx()


def n_params(n_sit: int) -> int:
    """Length of the flat parameter vector under the active layout."""
    if SHARE_CTX:
        return n_sit * 2 * _n_hand() + 2 * _n_ctx()
    return n_sit * 2 * _n_feats()


def unpack(theta_flat, n_sit):
    """``(H, C)`` — per-situation blocks and the shared context block (``None`` if not shared)."""
    th = np.asarray(theta_flat, dtype=np.float64)
    nh, nc = _n_hand(), _n_ctx()
    if not SHARE_CTX:
        return th.reshape(n_sit, 2, nh + nc), None
    cut = n_sit * 2 * nh
    return th[:cut].reshape(n_sit, 2, nh), th[cut:].reshape(2, nc)


def theta_sit(H, C, s_ix):
    """The full ``(2, D)`` coefficient matrix for one situation."""
    return H[s_ix] if C is None else np.concatenate([H[s_ix], C], axis=1)


def act_probs(theta_c: np.ndarray, mask: np.ndarray, raw, z=None) -> np.ndarray:
    """``P(k | features)`` over legal classes for ONE situation."""
    raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
    F = _feats(raw, z)                               # (n, D)
    th = theta_c.reshape(2, F.shape[1])              # call, raise
    z = np.zeros((F.shape[0], 3))
    z[:, 1] = F @ th[0]
    z[:, 2] = F @ th[1]
    z = np.where(mask[None, :], z, -1e9)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Hand:
    """One hand's worth of evidence about the opponent.

    ``sit`` / ``mask`` / ``took``  the opponent's decisions: situation index, legal action
                                   classes, and the class it chose.
    ``X``                          strength features of each candidate holding at each
                                   decision, ``(J, T, 2)`` — one row per cluster
                                   TRAJECTORY,
                                   since holdings that follow the same trajectory are
                                   indistinguishable to the model.
    ``w``                          prior weight of each trajectory (how many holdings take
                                   it).  A showdown collapses this to one-hot on the truth.
    """

    __slots__ = ("sit", "mask", "took", "X", "Z", "w")

    def __init__(self, sit, mask, took, X, w, Z=None):
        self.sit = np.asarray(sit, dtype=int)
        self.mask = np.asarray(mask, dtype=bool)
        self.took = np.asarray(took, dtype=int)
        self.X = np.asarray(X, dtype=np.float64)
        self.Z = (None if Z is None else np.asarray(Z, dtype=np.float64))
        w = np.asarray(w, dtype=np.float64)
        self.w = w / w.sum() if w.sum() > 0 else np.full(w.size, 1.0 / max(w.size, 1))


def neg_loglik(theta_flat: np.ndarray, hands, n_sit: int,
               prior: np.ndarray = None, lam: float = 0.0,
               ridge: float = 1e-2) -> float:
    """``-log L`` over hands, optionally penalised toward ``prior``.

    ``ridge`` is not cosmetic.  Fold is the reference class, but fold is NOT legal when
    the opponent is unopened — there the softmax runs over {call, raise} only, and adding
    a constant to both intercepts leaves the likelihood untouched.  That direction is
    unidentified and the optimiser walks off along it (intercepts reached ~64 before this
    was added).  A small ridge pins it without materially moving the directions the data
    does identify.
    """
    H, C = unpack(theta_flat, n_sit)
    total = 0.0
    for h in hands:
        # P(action sequence | trajectory j) = product over the hand's decisions.
        logp = np.zeros(h.X.shape[0])
        for t in range(h.sit.size):
            p = act_probs(th[h.sit[t]], h.mask[t], h.X[:, t, :])[:, h.took[t]]
            logp += np.log(np.clip(p, 1e-12, None))
        m = logp.max()
        total += m + np.log(max(float(h.w @ np.exp(logp - m)), 1e-300))
    nll = -total + ridge * float((theta_flat ** 2).sum())
    if prior is not None and lam > 0:
        nll += lam * float(((theta_flat - prior) ** 2).sum())
    return nll


def nll_and_grad(theta_flat: np.ndarray, hands, n_sit: int,
                 prior: np.ndarray = None, lam: float = 0.0,
                 ridge: float = 1e-2) -> tuple:
    """``(-log L, gradient)``.  Closed form, so the fit costs one evaluation per step.

    For the mixture ``L = sum_j w_j exp(S_j)`` with ``S_j`` the log-probability of the
    action sequence under trajectory ``j``, the gradient is the posterior-weighted sum of
    each trajectory's own gradient, ``sum_j r_j dS_j/dtheta`` with
    ``r_j = w_j exp(S_j) / sum w exp(S)``.  ``r`` is the posterior over the opponent's
    holding given everything it did this hand — the belief, appearing as a by-product of
    differentiating rather than as a separate update.

    Each softmax term contributes the textbook ``d log p_k / d a_m = [k = m] - p_m`` and
    ``d log p_k / d b_m = x ([k = m] - p_m)``.
    """
    H, C = unpack(theta_flat, n_sit)
    total = 0.0
    gH = np.zeros_like(H)
    gC = None if C is None else np.zeros_like(C)
    nh = _n_hand()
    for h in hands:
        T = h.sit.size
        P = []                                  # per decision: (J, 3) action probs
        logp = np.zeros(h.X.shape[0])
        for t in range(T):
            zt = None if h.Z is None else h.Z[t]
            pt = act_probs(theta_sit(H, C, h.sit[t]), h.mask[t], h.X[:, t, :], zt)
            P.append(pt)
            logp += np.log(np.clip(pt[:, h.took[t]], 1e-12, None))
        m = logp.max()
        ex = np.exp(logp - m)
        denom = float(h.w @ ex)
        total += m + np.log(max(denom, 1e-300))
        r = (h.w * ex) / max(denom, 1e-300)     # posterior over trajectories
        for t in range(T):
            s_ix = h.sit[t]
            k = h.took[t]
            pt = P[t]
            F = _feats(h.X[:, t, :], None if h.Z is None else h.Z[t])   # (J, D)
            for m_ix in (1, 2):
                if not h.mask[t][m_ix]:
                    continue
                d = ((1.0 if k == m_ix else 0.0) - pt[:, m_ix])
                contrib = (r * d) @ F
                if gC is None:
                    gH[s_ix, m_ix - 1] += contrib
                else:
                    # Hand part is situation-local; context part accumulates globally.
                    gH[s_ix, m_ix - 1] += contrib[:nh]
                    gC[m_ix - 1] += contrib[nh:]
    g_flat = (gH.ravel() if gC is None
              else np.concatenate([gH.ravel(), gC.ravel()]))
    nll = -total + ridge * float((theta_flat ** 2).sum())
    grad = -g_flat + 2.0 * ridge * theta_flat
    if prior is not None and lam > 0:
        diff = theta_flat - prior
        nll += lam * float((diff ** 2).sum())
        grad += 2.0 * lam * diff
    return nll, grad


def fit(hands, n_sit: int, theta0=None, prior=None, lam: float = 0.0,
        ridge: float = 1e-2, maxiter: int = 150) -> np.ndarray:
    """Maximum (penalised) likelihood over the flattened per-situation parameters."""
    p0 = (np.zeros(n_params(n_sit)) if theta0 is None
          else np.asarray(theta0).ravel().copy())
    if not hands:
        return p0 if prior is None else np.asarray(prior).ravel().copy()
    r = minimize(nll_and_grad, p0, args=(hands, n_sit, prior, lam, ridge),
                 jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return np.asarray(r.x, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blueprint-path", required=True)
    ap.add_argument("--lut-path", required=True)
    ap.add_argument("--population-hands", type=int, default=5000)
    ap.add_argument("--hands", type=int, default=5000)
    ap.add_argument("--trials-per-type", type=int, default=2)
    ap.add_argument("--bias-multiplier", type=float, default=5.0)
    ap.add_argument("--lam", type=float, default=-1.0,
                    help="Penalty pulling a player toward the population fit. Negative "
                         "(default) = estimate it from the population phase by empirical "
                         "Bayes; a positive value overrides with a fixed constant.")
    ap.add_argument("--basis", choices=("poly", "rbf"), default="poly",
                    help="Strength basis. 'rbf' = fixed Gaussian bumps: can represent a "
                         "polarised (bimodal) range, still convex to fit.")
    ap.add_argument("--n-basis", type=int, default=5,
                    help="Number of RBF centres across the strength range.")
    ap.add_argument("--degree", type=int, default=1, choices=(1, 2),
                    help="Strength terms: 1 = linear logit, 2 = adds curvature (the "
                         "per-class-Gaussian model).")
    ap.add_argument("--split", choices=("none", "raises"), default="none",
                    help="Situation granularity. 'raises' splits on the RAW number of "
                         "raises this round (0/1/2+) instead of collapsing everything "
                         "past the first into one level — so 2-bet and 3-bet pots stop "
                         "sharing parameters.")
    ap.add_argument("--ridge", type=float, default=1e-2,
                    help="L2 on the parameters. Required, not optional: with fold "
                         "illegal when unopened, one direction of the logit is "
                         "unidentified and the fit runs away without it.")
    ap.add_argument("--refit-every", type=int, default=250,
                    help="Refit the player model every N hands.  Scoring is prequential: "
                         "a hand is predicted by the model fitted before it.")
    ap.add_argument("--checkpoints", default="1250,2500,5000")
    ap.add_argument("--low-card-rank", type=int, default=10)
    ap.add_argument("--high-card-rank", type=int, default=14)
    ap.add_argument("--starting-stack", type=int, default=10000)
    ap.add_argument("--small-blind", type=int, default=50)
    ap.add_argument("--big-blind", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1,
                    help="Run trials in parallel, one core each.  Trials are fully "
                         "independent once the population fit exists, so this is a pure "
                         "wall-clock win; WITHIN a trial the scoring is prequential and "
                         "must stay serial.")
    ap.add_argument("--n-players", type=int, default=2,
                    help="Table size. >2 models EVERY non-hero seat at once (one hand is "
                         "evidence on all of them) and adds a live-player axis to the "
                         "model's situations. Use 4 with --low-card-rank 2 for production.")
    ap.add_argument("--shm-dir", default="/dev/shm",
                    help="Shared-memory dir for the blueprint's chunk tables. Give each "
                         "concurrent job its OWN dir on a shared node: block names are "
                         "machine-global and persist, so two jobs sharing this silently "
                         "share storage.")
    ap.add_argument("--progress-every", type=int, default=500,
                    help="Log a line every N hands collected (0 = off). Refit and "
                         "per-arm fit totals are always logged.")
    ap.add_argument("--arms", default="deployed",
                    help="Comma-separated arms fitted on ONE shared collection, e.g. "
                         "'base,deployed+shared,deployed'. Suffix '+shared' shares the "
                         "context coefficients across situations.")
    ap.add_argument("--share-ctx", action="store_true",
                    help="Share the context (token) coefficients across situations, "
                         "keeping intercepts and hand coefficients per-situation. Cuts "
                         "params ~3x now and ~5x once multiway grows n_sit.")
    ap.add_argument("--features", choices=("deployed", "base"), default="deployed",
                    help="'base' = the pre-probe set (x1,x2,x3,odds); 'deployed' adds the "
                         "PPot/NPot + polarisation block and the token bag.")
    ap.add_argument("--min-token-rate", type=float, default=0.005,
                    help="Drop token columns firing on fewer than this fraction of "
                         "population decisions; they cost a parameter and carry no signal.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from environment.player import Player
    from environment.poker_env import PokerEnv
    from environment.action_space import MAX_ACTIONS_PER_STREET
    from information_abstraction.lookup import load_info_set_lut
    from poker_ai.search.policy import BlueprintPolicy
    from poker_ai.search.ranges import RangeTracker
    from poker_ai.tables.cfr_tables import CFRTables
    from poker_ai.tables.warm_start import apply_warm_start_to_tables

    lut = load_info_set_lut(args.lut_path, pickle_dir=False)
    root = Path(args.blueprint_path)
    index_root = root / "lmdb_index" if (root / "lmdb_index").exists() else root
    # ChunkedTable's shm blocks are named machine-globally (`pluribus_strategy_{r}_...`)
    # with no per-blueprint qualifier, and they persist between runs -- so two jobs on one
    # node with the default dir silently SHARE storage and the second overwrites the
    # first.  Harmless on a private laptop, a correctness hazard on a shared cluster node.
    tables = CFRTables(index_path=index_root, shm_dir=args.shm_dir,
                       actions_per_street=MAX_ACTIONS_PER_STREET)
    apply_warm_start_to_tables(tables, args.blueprint_path, int(args.n_players))
    strength = cluster_strength(args.lut_path)
    spread = cluster_spread(args.lut_path)
    potential = cluster_potential(args.lut_path)
    ranks = cluster_rank(strength)
    mult = float(args.bias_multiplier)

    pol_cache: dict = {}

    def policy_at(m: float):
        k = round(float(m), 4)
        p = pol_cache.get(k)
        if p is None:
            p = pol_cache[k] = BlueprintPolicy(tables, bias_multiplier=k)
        return p

    global DEG, _NPAR, BASIS, RBF_C, RBF_H, FEATURE_SET, SHARE_CTX
    DEG = int(args.degree)
    BASIS = args.basis
    RBF_C = np.linspace(-0.5, 0.5, max(2, int(args.n_basis)))
    RBF_H = float(1.0 / max(2, int(args.n_basis)))
    FEATURE_SET = args.features
    SHARE_CTX = bool(args.share_ctx)
    _NPAR = 2 * _n_feats()

    n_lv = 3 if args.split == "raises" else 2
    n_p = int(args.n_players)
    # The model's partition gains a live-player axis multiway: a bet into three players is
    # not the node a bet heads-up is, and pooling them asks one coefficient to describe
    # both.  Heads-up this collapses to the old 2-tuple, so HU situation counts and every
    # heads-up measurement already taken are unchanged.
    multiway = n_p > 2
    live_axis = tuple(range(2, n_p + 1)) if multiway else (2,)

    def sit_of(env):
        """The situation key: street, aggression so far, and (multiway) the field size."""
        if args.split == "raises":
            base = (env.betting_stage, min(int(env._n_raises), 2))
        else:
            base = _class_of(env)
        return base + (_P.n_live(env),) if multiway else base

    walker = policy_at(1.0)
    types = list(PROFILES)

    sits = ([(s, lv, k) for s in _POSTFLOP for lv in range(n_lv) for k in live_axis]
            if multiway else [(s, lv) for s in _POSTFLOP for lv in range(n_lv)])
    sit_ix = {c: i for i, c in enumerate(sits)}
    n_sit = len(sits)
    checkpoints = sorted({int(x) for x in args.checkpoints.split(",") if x.strip()})
    board_cache: dict = {}

    def play(table_types, n_hands, seed_base, record_by_id=None):
        """Play ``n_hands`` at a table of ``len(table_types)`` opponents at once.

        Every non-hero seat is an opponent we learn about, so one hand yields evidence on
        all of them rather than on one — the whole point of a multiway table, and free
        data: the simulation cost is per hand, not per opponent.

        Opponents are IDENTITIES, not seats.  Each hand rotates both the hero seat and the
        identity-to-seat map, so an archetype is observed from every position instead of
        having its position confounded with its type (a nit that always sits on the button
        looks like a positional effect).

        Returns ``{identity: [Hand, ...]}``; ``record_by_id`` collects the per-decision
        scoring inputs the same way, one list per identity.
        """
        n_opp = len(table_types)
        out = {i: [] for i in range(n_opp)}
        for hand in range(1, n_hands + 1):
            hero_seat, seat_of, id_of = seat_map(hand, n_p, n_opp)
            np.random.seed(seed_base + hand)
            players = [Player(i, args.starting_stack) for i in range(n_p)]
            env = PokerEnv(players=players, small_blind=args.small_blind,
                           big_blind=args.big_blind, low_card_rank=args.low_card_rank,
                           high_card_rank=args.high_card_rank)
            env.card_info_lut = lut
            rng = np.random.RandomState(seed_base * 3 + hand)
            hero_cards = [int(c) for c in env.players[hero_seat].cards]
            # pi_0: every holding an opponent could hold, board- and hero-masked.  The
            # OTHER opponents' cards stay in: we do not observe them, so removing them
            # would be information the modeller does not have.
            live0 = (~np.isin(env.combo_cards[:, 0], hero_cards)
                     & ~np.isin(env.combo_cards[:, 1], hero_cards))
            tracker = RangeTracker(env, hero_seat, tuple(hero_cards),
                                   sorted(seat_of.values()))
            seen_board = set(int(c) for c in env.community_cards)
            D = {i: {"sit": [], "mask": [], "took": [], "cl": [], "ft": [], "cx": [],
                     "rec": []} for i in range(n_opp)}
            acts = {}  # stage -> [(seat, token)]

            guard = 0
            while not env.is_terminal and guard < 200:
                guard += 1
                ps = env.policy_state
                legal = ps.legal_actions
                if not legal:
                    break
                stage = env.betting_stage
                seat = int(env.player_i)
                if seat == hero_seat or stage not in _POSTFLOP:
                    row = np.asarray(walker.strategy(ps, "none"), np.float64)
                    c = np.cumsum(np.clip(row, 0, None))
                    i = (int(np.searchsorted(c, rng.random_sample() * c[-1]))
                         if c[-1] > 0 else int(rng.randint(len(legal))))
                    tok = legal[min(i, len(legal) - 1)]
                    acts.setdefault(stage, []).append((seat, tok))
                    env.step_in_place(tok)
                    continue

                oid = id_of[seat]
                true_type = table_types[oid]
                now_board = set(int(c) for c in env.community_cards)
                new_cards = sorted(now_board - seen_board)
                if new_cards:
                    tracker.on_board_update(new_cards)
                    seen_board = now_board
                public = env.policy_public_fields()
                bkey = (stage, tuple(sorted(int(c) for c in env.community_cards)))
                view = board_cache.get(bkey)
                if view is None:
                    view = board_cache[bkey] = _BoardView(env, stage)
                cl = view.cluster
                eqs = strength[stage]
                sps = spread[stage]
                sit = sit_of(env)
                k_live = _P.n_live(env)
                live_now = live0 & (cl >= 0)
                le = np.sort(eqs[cl[live_now]]) if live_now.any() else np.array([0.5])
                nlive = max(le.size, 1)

                _bets = [int(pl.n_bet_chips) for pl in env.players]
                _to_call = max(0, max(_bets) - _bets[seat])
                _pot = float(env.pot_size) + float(sum(_bets))
                _odds = (_to_call / (_pot + _to_call)) if _to_call > 0 else 0.0
                p_up, p_dn, p_top, p_bot = potential[stage]

                def _x(cid):
                    e = float(eqs[cid])
                    lo = float(np.searchsorted(le, e, "left"))
                    hi = float(np.searchsorted(le, e, "right"))
                    return (0.5 * (lo + hi) / nlive - 0.5, e - 0.5,
                            float(sps[cid]) - 0.15,
                            float(p_up[cid]) - 0.1, float(p_dn[cid]) - 0.1,
                            float(p_top[cid]) - 0.1, float(p_bot[cid]) - 0.1)
                mask = np.zeros(3, dtype=bool)
                for a in legal:
                    mask[_ACTION_CLASSES.index(_action_class(a))] = True

                actual = int(cl[env.combo_index[
                    tuple(sorted(int(c) for c in env.players[seat].cards))]])
                live = live_now
                present = [int(c) for c in np.unique(cl[live])]
                if actual not in present:
                    present.append(actual)
                true_by_cl = {}
                bias, centre, width = _entry(PROFILES[true_type], sit)
                # Field size scales the LEAK, not the base policy: the blueprint already
                # adjusts multiway on its own, and what separates archetypes is how far
                # each over- or under-does that.  Exactly 1.0 heads-up.
                ms = _P.multiway_scale(true_type, k_live)
                for cid in present:
                    ps_c = env.policy_state_for(view.rep[cid], public=public)
                    b = np.asarray(walker.strategy(ps_c, "none"), np.float64)
                    kw = (0.0 if bias == "none"
                          else _kernel_weight(ranks, stage, cid, centre, width) * ms)
                    true_by_cl[cid] = (b if (bias == "none" or kw <= 0.0)
                                       else np.asarray(policy_at(
                                           1.0 + (mult - 1.0) * kw).strategy(ps_c, bias),
                                           np.float64))

                rec = None
                if record_by_id is not None:
                    try:
                        ob = np.asarray(tracker.range_of(seat), dtype=np.float64)
                    except KeyError:                 # seat already folded out
                        ob = live.astype(np.float64)
                    cnt = np.bincount(cl[live], weights=ob[live],
                                      minlength=int(cl.max()) + 1).astype(float)
                    bw = np.array([cnt[c] if c < cnt.size else 0.0 for c in present])
                    tm = np.zeros(len(legal))
                    for i, cid in enumerate(present):
                        if bw[i] > 0:
                            tm += bw[i] * true_by_cl[cid][:len(legal)]
                    zc = np.asarray([_odds - 0.25] + _token_bag(acts, seat),
                                    dtype=np.float64)
                    tw3, tmc3 = np.zeros(3), np.zeros(3)
                    trow = true_by_cl[actual][:len(legal)]
                    for i, a in enumerate(legal):
                        kk = _ACTION_CLASSES.index(_action_class(a))
                        if i < trow.size:
                            tw3[kk] += trow[i]
                        if i < tm.size:
                            tmc3[kk] += tm[i]
                    rec = [sit, mask.copy(), np.array(_x(actual)), zc, tw3, tmc3, -1]

                row = true_by_cl[actual]
                c = np.cumsum(np.clip(row, 0, None))
                a_idx = (int(np.searchsorted(c, rng.random_sample() * c[-1]))
                         if c[-1] > 0 else int(rng.randint(len(legal))))
                a_idx = min(a_idx, len(legal) - 1)
                d = D[oid]
                d["sit"].append(sit_ix[sit])
                d["mask"].append(mask)
                d["took"].append(_ACTION_CLASSES.index(_action_class(legal[a_idx])))
                d["cl"].append(cl.copy())
                d["ft"].append({c2: _x(c2) for c2 in np.unique(cl[live_now])})
                d["cx"].append([_odds - 0.25] + _token_bag(acts, seat))
                if rec is not None:
                    rec[6] = d["took"][-1]
                    d["rec"].append(rec)

                def _sigma_for_combo(hidx, _cl=cl, _tb=true_by_cl, _n=len(legal)):
                    cid = int(_cl[hidx])
                    r0 = _tb.get(cid)
                    if r0 is None:
                        return np.full(_n, 1.0 / max(_n, 1))
                    r = np.asarray(r0[:_n], dtype=np.float64)
                    t = r.sum()
                    return r / t if t > 0 else np.full(_n, 1.0 / max(_n, 1))

                acts.setdefault(stage, []).append((seat, legal[a_idx]))
                tracker.on_action(seat, env, legal[a_idx], _sigma_for_combo)
                if _action_class(legal[a_idx]) == "fold":
                    tracker.on_seat_folded(seat)
                env.step_in_place(legal[a_idx])

            n_show = sum(1 for p in env.players if p.is_active)
            for oid in range(n_opp):
                d = D[oid]
                if not d["sit"]:
                    continue
                C = np.stack(d["cl"], axis=1)                 # (n_combos, T)
                keep = live0 & (C >= 0).all(axis=1)
                if not keep.any():
                    continue
                traj, inv = np.unique(C[keep], axis=0, return_inverse=True)
                w = np.bincount(inv).astype(np.float64)
                X = np.zeros(traj.shape + (7,), dtype=np.float64)
                _zero7 = (0.0,) * 7
                for t in range(traj.shape[1]):
                    fmap = d["ft"][t]
                    for j in range(traj.shape[0]):
                        X[j, t] = fmap.get(int(traj[j, t]), _zero7)
                # A showdown reveals THIS opponent's holding, collapsing its prior onto
                # the one trajectory it actually took.
                s_seat = seat_of[oid]
                if n_show >= 2 and env.players[s_seat].is_active:
                    ai = env.combo_index[
                        tuple(sorted(int(c) for c in env.players[s_seat].cards))]
                    if keep[ai]:
                        w2 = np.zeros_like(w)
                        w2[inv[np.flatnonzero(keep) == ai]] = 1.0
                        if w2.sum() > 0:
                            w = w2
                # Tagged with the HAND NUMBER: checkpoints count hands played, not hands
                # that happened to give this opponent a postflop decision, and those
                # differ by a large factor multiway.
                out[oid].append((hand, Hand(d["sit"], d["mask"], d["took"], X, w,
                                            np.asarray(d["cx"], dtype=np.float64))))
                if record_by_id is not None:
                    record_by_id.setdefault(oid, []).append((hand, d["rec"]))
        return out



    def parallel_map(n_items, work, n_workers):
        """Run ``work(i)`` for i in range(n_items) across forked workers.

        LMDB's reader-lock table is a process-shared mmap, so an inherited env is NOT
        reader-safe: a child touching it clobbers the parent's slot and the next read in
        either process trips MDB_BAD_RSLOT.  BOTH halves are required — close before the
        fork, reopen on both sides after.  Reopening only in the child is not enough
        (observed: every worker died and the pool returned nothing).
        """
        if n_workers <= 1 or n_items <= 1:
            return [work(i) for i in range(n_items)]
        from evaluation.hand_pool import run_index_pool

        def _setup(worker_id, shared):
            try:
                tables.open_envs()
            except Exception:
                try:
                    tables.reopen_after_fork()
                except Exception:
                    pass
            return {"out": []}

        def _process(idx, state, shared):
            state["out"].append((idx, work(idx)))

        try:
            tables.close_envs()
        except Exception:
            pass
        try:
            payloads = run_index_pool(
                n_workers=min(int(n_workers), n_items), setup=_setup, process=_process,
                teardown=lambda st: st["out"], target=n_items)
        finally:
            try:
                tables.open_envs()
            except Exception:
                pass
        got = {}
        for pl in payloads:
            for idx, val in (pl or []):
                got[idx] = val
        return [got.get(i) for i in range(n_items)]

    # ---------------- phase 1: population ----------------
    print("phase 1: population over %d archetypes, %d hands" % (len(types), args.population_hands))
    per_type = max(1, args.population_hands // len(types))
    # Hands are independent given their seed, so the population phase forks by archetype
    # instead of running ~5,000 simulations serially in the parent ahead of every trial.
    pop_tables = tables_of(types, n_p - 1)
    per_table = max(1, args.population_hands // max(len(pop_tables), 1))

    def _pop_one(ti):
        t0 = time.time()
        got = play(pop_tables[ti], per_table, args.seed * 7919 + 1000 * ti + 1)
        got = {oid: [h for _n, h in v] for oid, v in got.items()}
        _log("  pop table %d %s: %d hands -> %s with decisions (%.0fs)"
             % (ti + 1, "/".join(pop_tables[ti]), per_table,
                "/".join(str(len(v)) for v in got.values()), time.time() - t0))
        return got

    _pop = parallel_map(len(pop_tables), _pop_one, args.workers)
    pop_hands = []
    per_arch_hands = defaultdict(list)
    for ti, tbl in enumerate(pop_tables):
        got = _pop[ti] or {}
        for oid, hs in got.items():
            per_arch_hands[tbl[oid]] += hs
            pop_hands += hs
    def _lambda_for(theta_pop_a):
        """Empirical-Bayes shrinkage strength for one arm.

        tau^2 = between-archetype variance minus the within-archetype noise, estimated by
        splitting each archetype's hands in half; lambda = 1/(2 tau^2).  The per-archetype
        fits are independent, so they fork rather than running serially.
        """
        if args.lam >= 0:
            return args.lam
        names = [t for t in types if len(per_arch_hands.get(t, [])) >= 20]
        if not names:
            return 2.0

        def one(i):
            hs = per_arch_hands[names[i]]
            half = len(hs) // 2
            return (fit(hs, n_sit, theta0=theta_pop_a, ridge=args.ridge),
                    fit(hs[:half], n_sit, theta0=theta_pop_a, ridge=args.ridge),
                    fit(hs[half:], n_sit, theta0=theta_pop_a, ridge=args.ridge))

        got = parallel_map(len(names), one, args.workers)
        between = [g[0] - theta_pop_a for g in got if g is not None]
        within = [(g[1] - g[2]) ** 2 / 4.0 for g in got if g is not None]
        if not between:
            return 2.0
        b = float(np.mean(np.var(np.array(between), axis=0, ddof=1)))
        w = float(np.mean(np.array(within)))
        tau2 = max(b - w, 1e-4)
        return 1.0 / (2.0 * tau2)

    # ---------------- arms ----------------
    # Every arm sees the SAME hands.  The opponent samples from its own archetype policy
    # and the other seat plays the blueprint, so a learned model never influences play --
    # it is only ever scored.  The whole simulation is therefore arm-independent, and
    # simulating it per arm was pure waste: building `true_by_cl` queries the blueprint for
    # EVERY cluster present (up to 50 per decision, against one in the ceiling probe),
    # which dominates the run.  Collect once, fit many.
    print("   %d hands with opponent decisions" % len(pop_hands))
    arms = []
    for spec in [a.strip() for a in args.arms.split(",") if a.strip()]:
        feats, share = (spec, False)
        if spec.endswith("+shared"):
            feats, share = spec[:-len("+shared")], True
        FEATURE_SET, SHARE_CTX = feats, share
        _NPAR = 2 * _n_feats()
        # Pruned from the POPULATION phase only, then frozen: the per-player phase never
        # gets to choose its own feature set.
        nc = set_ctx_keep(pop_hands, args.min_token_rate)
        zk = Z_KEEP.copy()
        t0 = time.time()
        th = fit(pop_hands, n_sit, ridge=args.ridge)
        t1 = time.time()
        lm = _lambda_for(th)
        _log("  arm %-18s population fit %.0fs, lambda est %.0fs"
             % (spec, t1 - t0, time.time() - t1))
        arms.append({"name": spec, "features": feats, "share": share, "z_keep": zk,
                     "theta_pop": th, "lam": lm, "n_ctx": nc,
                     "n_par": n_params(n_sit)})
        print("   arm %-18s %2d hand + %2d ctx = %4d params   lambda=%.2f"
              % (spec, _n_hand(), nc, n_params(n_sit), lm))

    def use_arm(a):
        global FEATURE_SET, SHARE_CTX, Z_KEEP, _NPAR
        FEATURE_SET, SHARE_CTX = a["features"], a["share"]
        Z_KEEP = a["z_keep"]
        _NPAR = 2 * _n_feats()

    # ---------------- phase 2: refinement ----------------
    print("\nphase 2: refinement, %d hands per opponent, %d trials per archetype"
          % (args.hands, args.trials_per_type))
    res = {a["name"]: {c: defaultdict(list) for c in checkpoints} for a in arms}
    # A trial is a TABLE of opponents, not a single archetype: multiway, one table
    # produces a refined model per seat from the same hands.
    trials = tables_of(types, n_p - 1) * max(1, args.trials_per_type)

    def run_trial(trial_i):
        """Simulate one TABLE once, then fit and score every arm on every opponent.

        A multiway hand is evidence about all ``n_opp`` opponents at once, so the
        simulation — which is the expensive half — is shared not just across arms but
        across opponents.  Each opponent still gets its OWN refined model: that is what
        DBR deploys, and pooling them would measure a different thing.
        """
        table = trials[trial_i]
        base = args.seed * 104729 + trial_i * 7919 + 77
        recs = {}
        t_col = time.time()
        sim = play(table, args.hands, base, recs)
        col_s = time.time() - t_col
        _log("  t%d %-24s collected %d hands -> %s decisions (%.0fs)"
             % (trial_i + 1, "/".join(table), args.hands,
                "/".join(str(len(v)) for v in sim.values()) or "0", col_s))

        out = {}
        for arm in arms:
            use_arm(arm)
            theta_pop_a = arm["theta_pop"]
            per_arm = {}
            for oid, tname in enumerate(table):
                hand_list = sim.get(oid, [])
                rec_list = recs.get(oid, [])
                if not hand_list:
                    continue
                theta = theta_pop_a.copy()
                seen, done, fit_s, ptr = [], 0, 0.0, 0
                sc = {c: defaultdict(float) for c in checkpoints}
                while done < args.hands:
                    chunk = min(args.refit_every, args.hands - done)
                    hi = done + chunk
                    Hh, Ch = unpack(theta, n_sit)
                    Hp, Cp = unpack(theta_pop_a, n_sit)
                    while ptr < len(hand_list) and hand_list[ptr][0] <= hi:
                        hno, hobj = hand_list[ptr]
                        cp = next((c for c in checkpoints if hno <= c), None)
                        if cp is not None and ptr < len(rec_list):
                            for sit, mask, x_true, zc, tw3, tmc3, took in rec_list[ptr][1]:
                                si = sit_ix[sit]
                                pm = act_probs(theta_sit(Hh, Ch, si), mask, x_true, zc)[0]
                                pp = act_probs(theta_sit(Hp, Cp, si), mask, x_true, zc)[0]
                                tws = tw3.sum()
                                twn = tw3 / tws if tws > 0 else np.full(3, 1.0 / 3.0)
                                sc[cp]["n"] += 1
                                sc[cp]["hit"] += 1.0 if int(np.argmax(pm)) == took else 0.0
                                sc[cp]["conf"] += float(np.max(pm))
                                sc[cp]["pop"] += 1.0 if int(np.argmax(pp)) == took else 0.0
                                sc[cp]["oh"] += 1.0 if int(np.argmax(tw3)) == took else 0.0
                                sc[cp]["ob"] += 1.0 if int(np.argmax(tmc3)) == took else 0.0
                                sc[cp]["tv"] += float(0.5 * np.abs(pm - twn).sum())
                                sc[cp]["tvpop"] += float(0.5 * np.abs(pp - twn).sum())
                        seen.append(hobj)
                        ptr += 1
                    done = hi
                    t_f = time.time()
                    theta = fit(seen, n_sit, theta0=theta, prior=theta_pop_a,
                                lam=arm["lam"], ridge=args.ridge)
                    fit_s += time.time() - t_f
                res_a = {}
                for cp in checkpoints:
                    if sc[cp]["n"] > 0:
                        n = sc[cp]["n"]
                        res_a[cp] = {k: sc[cp][k] / n for k in
                                     ("hit", "conf", "pop", "oh", "ob", "tv", "tvpop")}
                per_arm[tname] = res_a
                _log("  t%d %-12s %-16s %d obs, fits %.0fs"
                     % (trial_i + 1, tname, arm["name"], len(seen), fit_s))
            out[arm["name"]] = per_arm
        _log("trial %2d/%2d %-24s DONE (collect %.0fs, %d arms x %d opponents)"
             % (trial_i + 1, len(trials), "/".join(table), col_s, len(arms), len(table)))
        return out

    def _collect(out):
        for arm_name, per_type_res in out.items():
            for _tname, per_cp in per_type_res.items():
                for cp, d in per_cp.items():
                    for k, v in d.items():
                        res[arm_name][cp][k].append(v)

    for out in parallel_map(len(trials), run_trial, args.workers):
        if out:
            _collect(out)

    out = {}
    for arm in arms:
        r = res[arm["name"]]
        print("\n%s  (%d params, lambda=%.2f)"
              % (arm["name"], arm["n_par"], arm["lam"]))
        print("%-8s %8s %8s %8s %11s %8s %10s %11s %10s %9s"
              % ("hands", "TV", "TVpop", "dTV", "error", "conf", "pop-only", "ORACLE-bl",
                 "orac-hand", "lift"))
        per = {}
        for cp in checkpoints:
            if not r[cp]["hit"]:
                continue
            acc = float(np.mean(r[cp]["hit"]))
            cf = float(np.mean(r[cp]["conf"]))
            pp = float(np.mean(r[cp]["pop"]))
            oh = float(np.mean(r[cp]["oh"]))
            ob = float(np.mean(r[cp]["ob"]))
            tv = float(np.mean(r[cp]["tv"]))
            tvp = float(np.mean(r[cp]["tvpop"]))
            print("%-8d %8.3f %8.3f %+8.3f %11.3f %8.3f %10.3f %11.3f %10.3f %+9.3f"
                  % (cp, tv, tvp, tvp - tv, 1 - acc, cf, pp, ob, oh, acc - pp))
            per[str(cp)] = {"tv": tv, "tv_pop": tvp, "error": 1 - acc, "confidence": cf,
                            "pop_only": pp, "oracle_blind": ob, "oracle_hand": oh,
                            "lift": acc - pp}
        out[arm["name"]] = {"lam": arm["lam"], "n_ctx": arm["n_ctx"],
                            "n_params": arm["n_par"], "by_checkpoint": per}

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"situations": ["%s L%d" % c for c in sits], "arms": out}, indent=2))
        print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
