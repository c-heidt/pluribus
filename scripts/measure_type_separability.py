"""Measure opponent-type separability ``d_C`` per strategic class, from PUBLIC info only.

The exploitation eval models an opponent as one of a few **stylized types**, each a
per-class bias *profile* over the blueprint (see the design note in
``project_exploitation_eval_design``).  The bot's belief over types is held per
strategic class ``C = (street, facing-a-bet)`` — the granularity every poker HUD
statistic uses — and its confidence there is driven by

    q_C(true) = 1 / (1 + (|T|-1) * exp(-d_C * n(C)))

so ``d_C`` — how distinguishable two types are from ONE observation in class ``C`` —
decides whether the design works at all:

* ``d_C ~ 0`` in a class  =>  nothing to learn there (and nothing to exploit, since the
  types play alike).  Confidence correctly stays at the prior.
* ``d_C`` large            =>  a few hundred hands resolve the type.

What you WANT is the mixed picture: zero where the profiles agree, large where they
disagree.  ``d_C`` large everywhere means the types are trivially learnable (the
``bias_multiplier`` is too big); ``d_C ~ 0`` everywhere means nothing is learnable (too
small).  So this probe also *calibrates* ``--bias-multiplier``: pick the smallest one for
which the target hand count resolves the type in the classes where the profiles differ.

⚠**The observer never sees hole cards it has not been shown.**  An earlier version of
this probe scored each node with the KL between the two types' policies *conditioned on
the hand actually dealt*.  Nothing observes that: a player's action is all that leaves
the table, and the holding behind it stays hidden unless the hand reaches showdown.  So
what a type costs to identify is the divergence between the two types' **observable
action distributions**, marginalised over the observer's belief about the holding:

    P_t(a | public) = sum_h  pi(h | public) * sigma_t(a | h, public)

and ``d_C`` is the symmetric KL between those.  The correction only ever moves ``d_C``
DOWN — mixing over ``h`` is an averaging step, so by convexity it destroys information —
and it moves it down hardest exactly where the leak is narrow in hand strength, which is
the whole design.  Reading the hole-conditioned number as if it were learnable would
overstate how fast a type resolves, i.e. under-estimate the hands the experiment needs.

**Showdown is the exception**, and it is scored as one.  A hand that gets there reveals
the holding, so an observer replaying it afterwards *can* condition on it — that is how a
real HUD learns "he called the turn with third pair".  Each node is therefore scored
twice, and which one counts depends on how its hand ended:

* ``hidden``   — the marginal divergence, what the observer had at the time;
* ``revealed`` — the hole-conditioned divergence, available only after a showdown;
* ``effective`` = ``(1 - p_showdown) * hidden + p_showdown * revealed``, the expected
  information per observation in that class, and the number to put into ``q_C``.

**The belief over holdings** starts board-masked uniform on the flop and is updated by
Bayes under the *baseline blueprint* — the model the observer has before it knows the
type — at every action that seat takes.  Pre-flop narrowing is not modelled, so the
range entering the flop is wider than a real observer's and ``d_C`` is, if anything,
conservative.  (Pre-flop is also where every profile agrees, so its ``d_C`` is 0 by
construction either way.)

⚠**What transfers between decks.**  The zero/nonzero partition is EXACT on any deck —
where neither profile biases a class the two policies are literally the same object, so
``d_C = 0`` by construction, and *which* classes those are comes from the profile table
rather than the cards.  The MAGNITUDES do not transfer: ``d_C`` under a multiplicative
bias depends on the base sigma, and a 20-card blueprint is far more deterministic than a
52-card one.  So validate the profile STRUCTURE locally on 20 cards, and calibrate
``--bias-multiplier`` / the hand count on the cluster against the real blueprint.

Usage (identical locally and on the cluster)::

    python scripts/measure_type_separability.py \\
        --blueprint-path data/blueprint_2p_20cards_6h_avg_linear \\
        --lut-path data/20cards_exact --low-card-rank 10 \\
        --hands 20000 --bias-multiplier 1.5,2.0,3.0,5.0
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# The stylized types: SIX archetypes, one bias entry per (street, facing-a-bet) cell.
#
# The taxonomy is the standard one, not invented here.  Every classification scheme in
# use is built on the same two axes — how many hands a player enters (tight/loose) and
# how they play them (passive/aggressive) — whose four quadrants are the nit, the TAG,
# the calling station and the LAG.  The six-type systems extend that with the MANIAC (the
# documented extreme of the LAG corner) and a near-equilibrium reference regular.
#
# ``solid_reg`` IS that reference, and it is unbiased: the blueprint is a near-equilibrium
# 2-player bot, i.e. exactly the 19/17-22/20 TAG the sources describe as a solid regular.
# That also means "solid regular" and "elite regular" — separate entries in some six-type
# lists — collapse onto the same object here and could never be told apart; using both
# would put two identical rows in the type set and make the posterior unidentifiable.
# So the sixth slot goes to FIT-OR-FOLD, which the quadrant scheme misses because it is
# street-dependent (loose entering, tight once the flop misses) and which carries its own
# well-defined HUD signature.
#
# Each archetype is pinned to the statistic that DEFINES it:
#
#   solid_reg    VPIP/PFR 19/17-22/20, AGG 32-40%      -> the unbiased blueprint
#   nit          VPIP<15, PFR<12, low AF                -> folds to aggression, never
#                                                          aggresses without a hand
#   station      fold-to-cbet ~40%, WTSD ~37%           -> calls down, never raises
#   fit_or_fold  fold-to-cbet 60-75%, WTSD 20-23%,      -> folds when it misses, bets
#                AF>5                                      hard when it hits
#   lag          VPIP 24-30, PFR 20-26, steal 50%+,     -> bets/raises a wide band
#                AGG 40%+
#   maniac       VPIP>35, PFR>25, flop AF>50, low WTSD  -> bets regardless of strength,
#                                                          worst on the river
#
# ⚠**Post-flop only, deliberately.**  VPIP and PFR are pre-flop statistics, but (a) the
# LUT carries no pre-flop strength axis, so a pre-flop leak could only be a FLAT bias —
# the incoherence the kernel below exists to remove (it would fold aces more often too);
# and (b) the bot never SEARCHES pre-flop, it plays it from the blueprint, so a pre-flop
# difference is unexploitable and would inflate d_C with no exploitation upside.  Each
# archetype is therefore expressed by its post-flop signature.
#
# The STRUCTURE is the point: different PAIRS separate in different classes, which is what
# makes a per-class belief worth holding.  Where two archetypes genuinely agree the entry
# is identical and d_C is exactly 0 — nit and station are both passive when unopened, nit
# and fit_or_fold both fold to aggression — so the probe should show a MIXED picture, not
# a uniformly large one.
# --------------------------------------------------------------------------- #
# Each entry is ``(bias_class, kernel_centre, kernel_width)`` in EQUITY units.  The bias
# is applied with an effective multiplier ``1 + (m-1)*w(e)`` where ``w`` is a Gaussian in
# the hand's expected equity --- so the leak fades smoothly toward air and toward the nuts
# instead of switching off at a threshold.  That is what makes a type a *coherent player*
# rather than a uniform shift: a calling station calls too wide with MARGINAL hands, it
# does not also call more with the nuts (which a flat bias would, incoherently).
#
# ⚠Every centre sits INSIDE the reachable equity band on both decks.  An earlier table
# centred a band at 0.75, which is above the production LUT's maximum cluster equity on
# the turn (0.677) and river (0.729) — that archetype was inert there while looking fine
# on the 20-card deck.  Check the per-street equity spread the probe prints first.
#
# ⚠The leak conditions FINER than the belief.  The belief classes stay (street,
# facing-a-bet) --- the HUD partition the model can actually track --- while the leak also
# conditions on hand strength.  So the opponent is heterogeneous WITHIN a belief class and
# the type mixture can never reproduce sigma exactly, even at q=1: an irreducible model
# error floor, which is what real models have.  It is also why the marginalisation above
# bites: the observer cannot see WHICH side of the kernel the opponent is on.
# Bands are (centre, width) in equity PERCENTILE — see :func:`cluster_rank` for why rank
# and not raw equity.  Each was set against a measurement of where the kernel actually
# fires and where a multiplicative bias actually moves the policy:
#
# * a fold band centred on AIR amplifies folds the blueprint already makes — correct play,
#   not a leak.  Over-folding is an error in the DEFENSIBLE middle, so that is where the
#   fold archetypes are centred;
# * "the hands it hit" must be an upper-rank band, not an absolute equity: at 0.60 raw it
#   sat at the turn's 85th percentile on this deck and above the maximum on production;
# * a LAG that fires on 94% of clusters is not selective, it is a maniac with a smaller
#   number attached.  Its band is genuinely narrow now, which is what separates the two.
#
# The multiplicative operator is SELF-TARGETING, which is why the kernel does not need to
# hunt for the spots with room to move.  Measured, a 2x fold bias by base fold probability:
#
#     base p(fold)   0.00-0.02  0.02-0.10  0.10-0.30  0.30-0.60  0.60-0.90  0.90-1.00
#     fold freq x       1.98       1.90       1.64       1.38       1.13       1.01
#
# It delivers the full multiplier exactly where folding is UNLIKELY — the decision points
# a nit actually gets wrong — and nothing where the base already folds purely, which is
# most hands and where there is correctly nothing to bias.  (Judging this by the absolute
# L1 move is misleading: 0.046 -> 0.088 is 0.08 of L1 but a near-doubling of the fold
# frequency, which is what an observer sees and what costs EV.)  So the kernel below is
# NOT there to find room to move; its only job is archetype coherence.
_DEFEND = (50.0, 22.0)     # the defensible middle — where over-folding is a real error
_MARGINAL = (50.0, 18.0)   # genuinely marginal — the call-down band
_MADE = (78.0, 14.0)       # upper band: it actually hit something
_SELECTIVE = (55.0, 15.0)  # a LAG's band — wide-ish but really is selective
_ANY = (50.0, 60.0)        # near-flat: the maniac bets whatever it holds
PROFILES = {
    # The near-equilibrium regular: no leak, the reference every other type is measured
    # against (and the strategy the exploiting bot already plays well against).
    "solid_reg":   {},
    # Tight-passive.  Folds to aggression on every street; when unopened it checks rather
    # than bet a marginal hand (the low AF that defines the type).
    "nit":         {("flop", 1): ("fold", *_DEFEND),
                    ("turn", 1): ("fold", *_DEFEND),
                    ("river", 1): ("fold", *_DEFEND),
                    ("flop", 0): ("call", *_MARGINAL),
                    ("turn", 0): ("call", *_MARGINAL),
                    ("river", 0): ("call", *_MARGINAL)},
    # Loose-passive.  Fold-to-cbet ~40% and WTSD ~37%: calls down with marginal holdings
    # on every street and essentially never takes the lead.
    "station":     {("flop", 1): ("call", *_MARGINAL),
                    ("turn", 1): ("call", *_MARGINAL),
                    ("river", 1): ("call", *_MARGINAL),
                    ("flop", 0): ("call", *_MARGINAL),
                    ("turn", 0): ("call", *_MARGINAL),
                    ("river", 0): ("call", *_MARGINAL)},
    # Sees flops, gives up when it misses (fold-to-cbet 60-75%, WTSD ~20%), but AF>5 —
    # "few bluffs, either raises when one has a hand or folds".  So: folds weak to a bet,
    # bets hard when it connected.  Shares the nit's fold rows by construction; the two
    # part company when unopened.
    "fit_or_fold": {("flop", 1): ("fold", *_DEFEND),
                    ("turn", 1): ("fold", *_DEFEND),
                    ("river", 1): ("fold", *_DEFEND),
                    ("flop", 0): ("raise", *_MADE),
                    ("turn", 0): ("raise", *_MADE),
                    ("river", 0): ("raise", *_MADE)},
    # Loose-aggressive: AGG 40%+, steal 50%+.  Bets a wide, semi-bluff-inclusive band when
    # unopened and raises back on the earlier streets, but is not the maniac — it still
    # slows down facing river aggression.
    "lag":         {("flop", 0): ("raise", *_SELECTIVE),
                    ("turn", 0): ("raise", *_SELECTIVE),
                    ("river", 0): ("raise", *_SELECTIVE),
                    ("flop", 1): ("raise", *_SELECTIVE),
                    ("turn", 1): ("raise", *_SELECTIVE)},
    # The documented extreme of the LAG corner: flop AF>50, bets "with almost any hand",
    # "particularly aggressive on the river".  Near-flat kernel = strength-blind.
    "maniac":      {("flop", 0): ("raise", *_ANY),
                    ("turn", 0): ("raise", *_ANY),
                    ("river", 0): ("raise", *_ANY),
                    ("flop", 1): ("raise", *_ANY),
                    ("turn", 1): ("raise", *_ANY),
                    ("river", 1): ("raise", *_ANY)},
}

_STREETS = ("pre_flop", "flop", "turn", "river")
_POSTFLOP = ("flop", "turn", "river")

# How each archetype's leak responds to the SIZE OF THE FIELD.
#
# The profile table is keyed on (street, facing-a-bet), which is complete heads-up but not
# multiway: folding to a bet with three players still live is a different decision from
# folding heads-up, and an archetype that ignores that is not a coherent player.  The
# blueprint already tightens multiway on its own (it is trained 4-handed); what varies by
# type is how far each one OVER- or UNDER-does that adjustment, so the sensitivity is
# applied to the LEAK, not to the base policy.
#
# Scale is ``1 + sens * (n_live - 2)``, so every value is exactly 1.0 heads-up: the
# two-player behaviour — and therefore every heads-up measurement already taken — is
# unchanged by construction.
MULTIWAY_SENS = {
    "solid_reg":   0.00,   # no leak to scale
    "nit":        +0.25,   # the defining error: gives up even more against a big field
    "station":     0.00,   # calls down regardless of who else is in — that IS the type
    "fit_or_fold":+0.25,   # "gives up when it misses" bites harder multiway
    "lag":        -0.25,   # loose but not stupid: tones the steal down into a field
    "maniac":      0.00,   # strength- AND count-blind by definition
}


def n_live(env) -> int:
    """Players still contesting the pot (active and not folded)."""
    try:
        return max(2, int(sum(1 for p in env.players if p.is_active)))
    except Exception:
        return 2


def _class_of(env) -> tuple:
    """The strategic class of the node the env is at: ``(street, facing-a-bet)``.

    This is the ARCHETYPE key and stays two-dimensional on purpose — the profile table is
    written per street and per facing-a-bet, with the field size entering through
    :data:`MULTIWAY_SENS` instead of multiplying the number of table rows by three.
    """
    from environment.poker_env import raise_level

    stage = env.betting_stage
    return (stage, raise_level(stage, env.n_raises_this_round))


def sit_of(env, multiway: bool = False) -> tuple:
    """The MODEL's situation key: ``(street, facing-a-bet[, n_live])``.

    The model's partition is separate from the archetype's, and needs the field size: a
    bet into three players is not the node a bet heads-up is, and pooling them would ask
    one coefficient to describe both.  Heads-up (``multiway=False``) this is exactly the
    old two-tuple, so existing situation counts are unchanged.
    """
    cls = _class_of(env)
    return cls + (n_live(env),) if multiway else cls


def _entry(profile: dict, cls: tuple) -> tuple:
    """``(bias, centre, width)`` for ``cls`` — ``("none", ...)`` where unbiased."""
    return profile.get(cls[:2], ("none", 0.0, 1.0))


def multiway_scale(type_name: str, k: int) -> float:
    """Field-size scale on the leak strength; exactly 1.0 heads-up."""
    return max(0.0, 1.0 + MULTIWAY_SENS.get(type_name, 0.0) * (int(k) - 2))


def cluster_strength(lut_path) -> dict:
    """Per-cluster **expected equity** in [0, 1], per post-flop street.

    KMeans labels are arbitrary — sklearn does not order them by centroid — so a cluster
    id carries no strength information and any ``cluster // m`` banding would be
    meaningless.  The centroids do carry it: each is a normalised EHS *histogram*, so its
    mean under the bin centres is that cluster's expected equity.  That recovers the
    strength axis the leak needs without touching the LUT build.
    """
    import joblib

    cents = joblib.load(Path(lut_path) / "centroids.joblib")
    out = {}
    for street, arr in cents.items():
        a = np.asarray(arr, dtype=np.float64)
        out[street] = a @ ((np.arange(a.shape[1]) + 0.5) / a.shape[1])
    return out


def cluster_spread(lut_path) -> dict:
    """Per-cluster EHS-histogram standard deviation — "drawiness" — per post-flop street.

    The mean alone cannot describe a flop hand.  A made hand and a draw with the same
    average equity have the same mean and very different histograms, and the blueprint
    plays the whole cluster rather than its mean, so it treats them differently.  Measured
    consequence: ranking by the mean alone, the opponent appears to FOLD stronger hands
    than it calls with on the flop (mean rank 0.58 folding vs 0.32 calling) — correct on
    the river where there is no future left, flat on the turn, inverted on the flop.  That
    gradient is the mean losing sufficiency exactly where potential matters most.

    A wide histogram at a middling mean is a draw; a narrow one at a high mean is a made
    hand.  The spread is already in the centroids and was simply being discarded.
    """
    import joblib

    cents = joblib.load(Path(lut_path) / "centroids.joblib")
    out = {}
    for street, arr in cents.items():
        a = np.asarray(arr, dtype=np.float64)
        b = (np.arange(a.shape[1]) + 0.5) / a.shape[1]
        m = a @ b
        out[street] = np.sqrt(np.maximum(a @ (b ** 2) - m ** 2, 0.0))
    return out


def cluster_rank(strength: dict) -> dict:
    """Per-cluster percentile RANK (0-100) of expected equity, per post-flop street.

    The kernel is defined on rank rather than raw equity, and that choice is load-bearing.
    Raw equity is not comparable across streets or decks: the turn's distribution on the
    20-card LUT sits low (median 0.41, p75 0.49), so a band centred at 0.60 lands near the
    85th percentile there while landing mid-range on the flop — an archetype's leak then
    fires on one street and not another for no reason anyone chose.  It is worse across
    decks: a band centred at 0.75 is mid-strength on this deck and ABOVE the production
    LUT's maximum on the turn (0.677), i.e. silently inert exactly where the experiment
    runs.  Ranks make "marginal" and "the hands it hit" mean the same thing everywhere,
    so a profile written once behaves the same on every street and every deck.
    """
    out = {}
    for street, eq in strength.items():
        a = np.asarray(eq, dtype=np.float64)
        order = a.argsort()
        rank = np.empty_like(order, dtype=np.float64)
        rank[order] = np.arange(a.size, dtype=np.float64)
        out[street] = 100.0 * rank / max(a.size - 1, 1)
    return out


def _cluster_of(env):
    """The acting player's ACTUAL cluster id, or ``None`` (pre-flop / no LUT entry).

    This is the one place a real hole card is read, and it feeds only the ``revealed``
    score — the divergence an observer gets *after* a showdown has shown it the hand.
    Nothing on the ``hidden`` path may call this.
    """
    stage = env.betting_stage
    if stage not in _POSTFLOP:
        return None
    try:
        key = tuple(sorted(env.current_player.cards) + sorted(env.community_cards))
        return int(env.card_info_lut[stage][key])
    except Exception:
        return None


class _BoardView:
    """Cluster table + one representative combo per cluster, for the current board.

    A policy reads a holding only through its cluster, so the observer's marginal over
    holdings collapses from a sum over combos (1326 on a 52-card deck) to a sum over the
    clusters actually present.  Both tables are pure functions of the public board, so
    they are built once per board and shared by every node on it.
    """

    def __init__(self, env, stage: str) -> None:
        combos = env.combo_cards
        board = sorted(int(c) for c in env.community_cards)
        blocked = set(board)
        table = env.card_info_lut[stage]
        bkey = tuple(board)
        self.cluster = np.full(combos.shape[0], -1, dtype=np.int64)
        self.rep: dict = {}
        for i in range(combos.shape[0]):
            c0, c1 = int(combos[i, 0]), int(combos[i, 1])
            if c0 in blocked or c1 in blocked:
                continue                       # card removal: the board holds them
            try:
                cid = int(table[(c0, c1) + bkey])
            except KeyError:                   # pragma: no cover - partial LUT
                continue
            self.cluster[i] = cid
            self.rep.setdefault(cid, (c0, c1))

    @property
    def live(self) -> np.ndarray:
        return self.cluster >= 0


def _kernel_weight(ranks: dict, street: str, cluster, centre: float,
                   width: float) -> float:
    """Gaussian kernel in equity PERCENTILE: 1.0 at the centre, tapering both ways.

    ``ranks`` comes from :func:`cluster_rank`; ``centre`` and ``width`` are in percentile
    units (0-100), so a profile's coverage is identical on every street and every deck.
    """
    tbl = ranks.get(street)
    if tbl is None or cluster is None or not (0 <= cluster < len(tbl)):
        return 0.0
    z = (float(tbl[cluster]) - centre) / max(width, 1e-9)
    return float(np.exp(-0.5 * z * z))


def _kl(p: np.ndarray, q: np.ndarray, floor: float = 1e-12) -> float:
    """``KL(p || q)`` over a shared support, floored so a structural zero cannot blow up.

    The multiplicative bias rescales and renormalises, so it never introduces or removes
    a zero: both rows share the base policy's support and the floor is a numerical guard,
    not a modelling choice.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), floor, None)
    q = np.clip(np.asarray(q, dtype=np.float64), floor, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _kl_pair(p: np.ndarray, q: np.ndarray) -> tuple:
    """``(jeffreys, worst_case)`` divergence between two observable action rows.

    A Bayesian observer's log-odds between two types drifts at ``KL(true || other)`` per
    observation — a ONE-SIDED rate that depends on which type is actually at the table.
    Two numbers follow:

    * ``jeffreys = KL(p||q) + KL(q||p)`` is the direction-free summary, and is what a
      "how different are these two" question wants.  It is NOT a drift rate: it adds both
      directions, so using it as one overstates the rate by up to 2x and understates the
      hands needed by the same factor.
    * ``worst_case = min(KL(p||q), KL(q||p))`` is the rate when the harder of the two
      types is the true one.  Identification has to work whichever type turned up, so
      this is the number a hand budget must be sized on.

    Both are reported; the hand counts use ``worst_case``.
    """
    a, b = _kl(p, q), _kl(q, p)
    return a + b, min(a, b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blueprint-path", required=True)
    ap.add_argument("--lut-path", required=True)
    ap.add_argument("--hands", type=int, default=300)
    ap.add_argument("--bias-multiplier", default="1.5",
                    help="One value, or a comma list (e.g. '1.2,1.5,2.0,3.0'). A list is "
                         "swept in ONE pass: the walk follows the UNBIASED blueprint, "
                         "which is multiplier-independent, so every multiplier sees the "
                         "identical node sequence and the (slow) LUT + blueprint load is "
                         "paid once. That matters on the cluster, where the load dominates.")
    ap.add_argument("--n-players", type=int, default=2)
    ap.add_argument("--low-card-rank", type=int, default=10)
    ap.add_argument("--high-card-rank", type=int, default=14)
    ap.add_argument("--starting-stack", type=int, default=10000)
    ap.add_argument("--small-blind", type=int, default=50)
    ap.add_argument("--big-blind", type=int, default=100)
    ap.add_argument("--eval-hands", type=int, default=3000,
                    help="Hand budget the EVAL will run. The report shows the type "
                         "confidence q_C each pair actually reaches at this budget — the "
                         "margin left for model confidence/error lives in how far below "
                         "1.0 those land.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Write the full result as JSON here.")
    args = ap.parse_args()

    from environment.player import Player
    from environment.poker_env import PokerEnv
    from environment.action_space import MAX_ACTIONS_PER_STREET
    from information_abstraction.lookup import load_info_set_lut
    from poker_ai.search.policy import BlueprintPolicy
    from poker_ai.tables.cfr_tables import CFRTables
    from poker_ai.tables.warm_start import apply_warm_start_to_tables

    lut = load_info_set_lut(args.lut_path, pickle_dir=False)
    root = Path(args.blueprint_path)
    index_root = root / "lmdb_index" if (root / "lmdb_index").exists() else root
    tables = CFRTables(index_path=index_root, actions_per_street=MAX_ACTIONS_PER_STREET)
    apply_warm_start_to_tables(tables, args.blueprint_path, args.n_players)
    mults = [float(x) for x in str(args.bias_multiplier).split(",") if x.strip()]
    strength = cluster_strength(args.lut_path)
    ranks = cluster_rank(strength)
    for st in sorted(strength):
        e = strength[st]
        print("equity per cluster — %-5s n=%d  min=%.3f  median=%.3f  max=%.3f"
              % (st, len(e), e.min(), np.median(e), e.max()))
    # The kernel makes the effective multiplier continuous, so cache one policy per
    # distinct value (a BlueprintPolicy is a thin wrapper over the shared tables).  Going
    # through the real policy keeps the production bias path exact: applying the reweight
    # to the already-legal-filtered row would renormalise in the wrong order.
    pol_cache: dict = {}

    def policy_at(mult: float):
        key = round(float(mult), 4)
        p = pol_cache.get(key)
        if p is None:
            p = pol_cache[key] = BlueprintPolicy(tables, bias_multiplier=key)
        return p

    walker = policy_at(1.0)   # bias "none" ignores the multiplier

    names = list(PROFILES)
    pairs = list(itertools.combinations(names, 2))
    # multiplier -> class -> pair -> list of symmetric KLs, one list per visibility.
    def _acc():
        return {m: defaultdict(lambda: defaultdict(list)) for m in mults}

    kl_hidden, kl_shown = _acc(), _acc()          # Jeffreys
    lo_hidden, lo_shown = _acc(), _acc()          # worst-case one-sided drift rate
    n_nodes: dict = defaultdict(int)
    n_shown: dict = defaultdict(int)          # nodes whose hand reached showdown
    touch: dict = defaultdict(list)           # per class: how much of it the kernel reaches
    board_cache: dict = {}

    for hand in range(args.hands):
        np.random.seed(args.seed * 1_000_003 + hand)
        players = [Player(i, args.starting_stack) for i in range(args.n_players)]
        env = PokerEnv(players=players, small_blind=args.small_blind,
                       big_blind=args.big_blind, low_card_rank=args.low_card_rank,
                       high_card_rank=args.high_card_rank)
        env.card_info_lut = lut
        rng = np.random.RandomState(args.seed * 7919 + hand)

        n_combos = env.combo_cards.shape[0]
        # The observer's belief over each seat's holding.  Board-masked uniform at the
        # flop, then Bayes under the BASELINE blueprint — the model it has before it
        # knows the type.  Held over COMBOS, not clusters, because cluster ids are
        # per-street and would not survive the next board card.
        belief = {s: np.ones(n_combos, dtype=np.float64) for s in range(args.n_players)}
        # Buffered per hand: which score counts depends on how the hand ENDS.
        pending: list = []

        guard = 0
        while not env.is_terminal and guard < 200:
            guard += 1
            ps = env.policy_state
            legal = ps.legal_actions
            if not legal:
                break
            cls = _class_of(env)
            stage = env.betting_stage
            seat = int(env.player_i)
            n_nodes[cls] += 1

            base = np.asarray(walker.strategy(ps, "none"), dtype=np.float64)

            if stage not in _POSTFLOP:
                # Pre-flop: every profile is unbiased here, so both scores are 0 by
                # construction.  Skipping the marginalisation costs nothing and avoids
                # the one place clusters do not exist.
                touch[cls].append(0.0)
                probs = np.clip(base, 0.0, None)
                c = np.cumsum(probs)
                idx = (int(np.searchsorted(c, rng.random_sample() * c[-1]))
                       if c[-1] > 0 else int(rng.randint(len(legal))))
                env.step_in_place(legal[min(idx, len(legal) - 1)])
                continue

            bkey = (stage, tuple(sorted(int(c) for c in env.community_cards)))
            view = board_cache.get(bkey)
            if view is None:
                view = board_cache[bkey] = _BoardView(env, stage)
            live = view.live
            # Re-mask on a new board: combos containing a freshly dealt card are dead.
            w = belief[seat] * live
            if w.sum() <= 0:
                w = live.astype(np.float64)
            w = w / w.sum()
            belief[seat] = w

            # Range mass per cluster — the observer's view of "what could be there".
            mass = np.bincount(view.cluster[live], weights=w[live])
            present = [int(c) for c in np.nonzero(mass)[0]]

            public = env.policy_public_fields()
            # One policy row per (cluster, type, multiplier).  ``policy_state_for`` reads
            # only public state plus the hypothetical combo, so nothing here can leak the
            # hand actually dealt.
            base_by_cl: dict = {}
            rows_by_cl: dict = {m: defaultdict(dict) for m in mults}
            kernel_seen = []
            for cid in present:
                ps_c = env.policy_state_for(view.rep[cid], public=public)
                base_c = np.asarray(walker.strategy(ps_c, "none"), dtype=np.float64)
                base_by_cl[cid] = base_c
                for t in names:
                    bias, centre, width = _entry(PROFILES[t], cls)
                    kw = (0.0 if bias == "none"
                          else _kernel_weight(ranks, stage, cid, centre, width))
                    if t != "blueprint":
                        kernel_seen.append(kw)
                    for m in mults:
                        rows_by_cl[m][t][cid] = (
                            base_c if (bias == "none" or kw <= 0.0)
                            else np.asarray(
                                policy_at(1.0 + (m - 1.0) * kw).strategy(ps_c, bias),
                                dtype=np.float64)
                        )
            # Kernel touch, range-weighted: how much of the class's REACHABLE strength
            # the leak covers, as the observer sees it rather than on the dealt hand.
            touch[cls].append(float(np.max(kernel_seen)) if kernel_seen else 0.0)

            actual = _cluster_of(env)          # revealed path ONLY (see _cluster_of)
            node = {"cls": cls, "hidden": {}, "shown": {},
                    "hidden_lo": {}, "shown_lo": {}}
            for m in mults:
                marg = {}
                for t in names:
                    acc = None
                    for cid in present:
                        r = rows_by_cl[m][t][cid]
                        acc = (r * mass[cid]) if acc is None else acc + r * mass[cid]
                    marg[t] = acc if acc is not None else base
                for a, b in pairs:
                    ra, rb = marg[a], marg[b]
                    if ra.shape == rb.shape and ra.size:
                        sym, lo = _kl_pair(ra, rb)
                        node["hidden"][(m, a, b)] = sym
                        node["hidden_lo"][(m, a, b)] = lo
                if actual is not None and actual in base_by_cl:
                    for a, b in pairs:
                        ra = rows_by_cl[m][a][actual]
                        rb = rows_by_cl[m][b][actual]
                        if ra.shape == rb.shape and ra.size:
                            sym, lo = _kl_pair(ra, rb)
                            node["shown"][(m, a, b)] = sym
                            node["shown_lo"][(m, a, b)] = lo
            pending.append(node)

            # Advance under the UNBIASED blueprint: reach(C) is then a property of
            # baseline play, the common reference every type is measured against.
            probs = np.clip(base, 0.0, None)
            c = np.cumsum(probs)
            idx = (int(np.searchsorted(c, rng.random_sample() * c[-1]))
                   if c[-1] > 0 else int(rng.randint(len(legal))))
            taken = legal[min(idx, len(legal) - 1)]
            # Bayes: the observer saw this action, so holdings that would rarely take it
            # lose weight.  Under the BASELINE blueprint — it does not know the type yet.
            a_idx = list(legal).index(taken)
            upd = np.zeros(n_combos, dtype=np.float64)
            for cid in present:
                r = base_by_cl[cid]
                if a_idx < r.size:
                    upd[view.cluster == cid] = float(r[a_idx])
            nb = belief[seat] * upd
            belief[seat] = nb if nb.sum() > 0 else belief[seat]
            env.step_in_place(taken)

        # A showdown reveals the holding, so THIS hand's nodes are scored on the
        # hole-conditioned divergence; otherwise the observer only ever had the marginal.
        # ``betting_stage`` reads 'terminal' even for showdowns, so count live seats.
        showdown = sum(1 for p in env.players if p.is_active) >= 2
        for node in pending:
            cls = node["cls"]
            if showdown:
                n_shown[cls] += 1
            for (m, a, b), v in node["hidden"].items():
                kl_hidden[m][cls][(a, b)].append(v)
            for (m, a, b), v in node["hidden_lo"].items():
                lo_hidden[m][cls][(a, b)].append(v)
            if showdown:
                for (m, a, b), v in node["shown"].items():
                    kl_shown[m][cls][(a, b)].append(v)
                for (m, a, b), v in node["shown_lo"].items():
                    lo_shown[m][cls][(a, b)].append(v)

    total = sum(n_nodes.values()) or 1
    order = [(s, lv) for s in _STREETS for lv in (0, 1) if (s, lv) in n_nodes]

    result = {"hands": args.hands, "multipliers": mults,
              "observability": "public actions; hole cards only after a showdown",
              "profiles": {k: {"%s/%d" % c: v for c, v in p.items()}
                           for k, p in PROFILES.items()},
              "by_multiplier": {}}

    def _mean(d, m, cls, pair):
        v = d[m][cls].get(pair, [])
        return float(np.mean(v)) if v else 0.0

    for m in mults:
        print("\nd_C — symmetric KL between what the two types are OBSERVED to do")
        print("bias_multiplier=%.2f  hands=%d  nodes=%d" % (m, args.hands, total))
        print("hidden = marginal over the belief (no hole cards);"
              " shown = after a showdown reveals them")
        head = "%-16s %7s %7s %7s %6s  " % ("class", "nodes", "reach", "kernel", "sd%")
        head += "  ".join("%-30s" % ("%s|%s" % (a[:9], b[:9])) for a, b in pairs)
        print("\n" + head)
        print("%-16s %7s %7s %7s %6s  " % ("", "", "", "", "")
              + "  ".join("%-30s" % "hidden / shown / effective" for _ in pairs))
        print("-" * len(head))
        per_class = {}
        for cls in order:
            n = n_nodes[cls]
            tw = float(np.mean(touch[cls])) if touch[cls] else 0.0
            p_sd = (n_shown[cls] / n) if n else 0.0
            row = ("%-16s %7d %6.1f%% %7.3f %5.1f%%  "
                   % ("%s L%d" % cls, n, 100.0 * n / total, tw, 100.0 * p_sd))
            cell = {}
            for a, b in pairs:
                hid = _mean(kl_hidden, m, cls, (a, b))
                shn = _mean(kl_shown, m, cls, (a, b))
                eff = (1.0 - p_sd) * hid + p_sd * shn
                lo_h = _mean(lo_hidden, m, cls, (a, b))
                lo_s = _mean(lo_shown, m, cls, (a, b))
                cell["%s|%s" % (a, b)] = {
                    "hidden": hid, "shown": shn, "effective": eff,
                    # the rate a hand budget must be sized on — see _kl_pair
                    "worst_case": (1.0 - p_sd) * lo_h + p_sd * lo_s,
                }
                row += "%-30s" % ("%.4f / %.4f / %.4f" % (hid, shn, eff))
            print(row)
            per_class["%s L%d" % cls] = {
                "nodes": n, "reach": n / total, "kernel_touch": tw,
                "showdown_rate": p_sd, "d_C": cell,
            }
        result["by_multiplier"]["%g" % m] = per_class

        # Precision of the estimate, for the cell that actually drives the design (the
        # largest d_C in each class).  d_C is a MEAN over the nodes in a class, and the
        # kernel makes it heavily skewed — many nodes carry w~0 and contribute ~0 — so the
        # sample size that matters is the rarest class, not the total hand count.
        # rel_SE ~ 10% is plenty: hands-to-learn scales as 1/d_C, so 10% there is 10% there.
        print("\nprecision — sharpest HIDDEN cell per class (rel SE = SE/mean):")
        print("%-16s %7s  %-24s %10s %9s" % ("class", "nodes", "cell", "d_C", "rel SE"))
        worst = 0.0
        for cls in order:
            best_pair, best_mean, best_se = None, 0.0, 0.0
            for a, b in pairs:
                v = kl_hidden[m][cls].get((a, b), [])
                if len(v) < 2:
                    continue
                mean = float(np.mean(v))
                if mean > best_mean:
                    best_pair = (a, b)
                    best_mean = mean
                    best_se = float(np.std(v, ddof=1)) / np.sqrt(len(v))
            if best_pair is None or best_mean <= 0.0:
                continue
            rel = best_se / best_mean
            worst = max(worst, rel)
            print("%-16s %7d  %-24s %10.4f %8.1f%%"
                  % ("%s L%d" % cls, n_nodes[cls],
                     "%s|%s" % (best_pair[0][:9], best_pair[1][:9]), best_mean,
                     100.0 * rel))
        if worst > 0.0:
            need = args.hands * (worst / 0.10) ** 2
            print("worst rel SE %.1f%%  =>  ~%.0f hands would bring every class to 10%%"
                  % (100.0 * worst, need))

        # What the number is FOR: hands until the belief in a class resolves the type.
        # q_C = 1/(1 + (|T|-1) e^{-d n}) => n = ln((|T|-1)/(1/q - 1)) / d, and n(C) per
        # hand is that class's node rate.  Uses ``effective`` — the observer gets the
        # revealed score only on the hands that showed down.
        # What the number is FOR.  A pair is identified once the log-odds between the
        # two types clears the prior, and it accumulates at the WORST-CASE one-sided rate
        # (see _kl_pair) — whichever of the two is actually at the table has to be found.
        # A pair only needs ONE class to separate it, so each pair takes its best class.
        n_types = len(PROFILES)
        need_dn = np.log((n_types - 1) / (1.0 / 0.90 - 1.0))
        rates = {c: n_nodes[c] / float(args.hands) for c in order}
        best_for_pair = {}
        for a, b in pairs:
            for cls in order:
                n = n_nodes[cls]
                if not n or rates[cls] <= 0:
                    continue
                p_sd = n_shown[cls] / n
                d = ((1.0 - p_sd) * _mean(lo_hidden, m, cls, (a, b))
                     + p_sd * _mean(lo_shown, m, cls, (a, b)))
                if d <= 0:
                    continue
                h = need_dn / (d * rates[cls])
                if (a, b) not in best_for_pair or h < best_for_pair[(a, b)][0]:
                    best_for_pair[(a, b)] = (h, cls, d)
        print("\nidentification — worst-case rate, %d types, q=0.90 (needs d*n = %.2f)"
              % (n_types, need_dn))
        print("a pair needs only its EASIEST class; the budget is set by the hardest pair")
        print("%-24s %-12s %10s %10s   %10s" % ("pair", "via", "d_worst", "hands",
                                                "q@%d" % args.eval_hands))
        ranked = sorted(best_for_pair.items(), key=lambda kv: -kv[1][0])
        for (a, b), (h, cls, d) in ranked:
            n_obs = d * rates[cls] * args.eval_hands
            q = 1.0 / (1.0 + (n_types - 1) * np.exp(-n_obs))
            print("%-24s %-12s %10.4f %10.0f   %10.3f"
                  % ("%s|%s" % (a[:10], b[:10]), "%s L%d" % cls, d, h, q))
        if ranked:
            print("ALL %d pairs identified within %.0f hands; at %d hands the hardest "
                  "pair sits at q=%.2f"
                  % (len(ranked), ranked[0][1][0], args.eval_hands,
                     1.0 / (1.0 + (n_types - 1)
                            * np.exp(-ranked[0][1][2]
                                     * rates[ranked[0][1][1]] * args.eval_hands))))

    print("\nColumns are 'hidden / shown / effective' per type pair."
          "\n  hidden    = what the observer actually had: the two types' action"
          "\n              distributions MARGINALISED over its belief about the holding."
          "\n  shown     = the same divergence conditioned on the true hand — available"
          "\n              only for hands that reached showdown, hence the sd% column."
          "\n  effective = (1-sd)*hidden + sd*shown, the expected information per"
          "\n              observation.  This is the one to put into q_C."
          "\nhidden <= shown always (averaging destroys information); a large gap means"
          "\nthe leak is sharp in hand strength and mostly invisible until showdown.")
    print("\n'kernel' = mean weight the strength kernel gives the range at that class:"
          "\nhow much of the class the leak actually reaches (1.0 = all of it, 0.0 = none)."
          "\nA low value with a nonzero d_C is the intended shape — a leak confined to the"
          "\nhands it should apply to.  Near 0 everywhere means the kernel misses the"
          "\nreachable strength range: check the equity spread printed above.")
    print("\nReading it: 0.0 = the two profiles are the SAME object in that class"
          "\n(nothing to learn, nothing to exploit).  What you want is a MIXED picture —"
          "\nzero where the profiles agree, clearly nonzero where they disagree."
          "\nLarge everywhere ⇒ multiplier too big; ~0 everywhere ⇒ too small.")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
