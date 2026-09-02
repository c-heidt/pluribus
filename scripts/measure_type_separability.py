"""Measure opponent-type separability ``d_C`` per strategic class.

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

⚠**What transfers between decks.**  The zero/nonzero partition is EXACT on any deck —
where neither profile biases a class the two policies are literally the same object, so
``d_C = 0`` by construction, and *which* classes those are comes from the profile table
rather than the cards.  The MAGNITUDES do not transfer: ``d_C`` under a multiplicative
bias depends on the base sigma, and a 20-card blueprint is far more deterministic than a
52-card one.  So validate the profile STRUCTURE locally on 20 cards, and calibrate
``--bias-multiplier`` / the hand count on the cluster against the real blueprint.

Usage (identical locally and on the cluster)::

    python -m scripts.measure_type_separability \\
        --blueprint-path data/2player_20cards_v2_strategy \\
        --lut-path data/20cards_exact \\
        --hands 300 --bias-multiplier 1.5 --low-card-rank 10
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# The stylized types: a bias class per (street, facing-a-bet) cell.
#
# Grounded in the documented recreational-player leaks rather than invented: the
# tight/loose x passive/aggressive taxonomy gives the nit and the calling station, and
# the street-specific literature gives the rest --- "people don't like to fold when it
# comes to the river, they are almost always calling" (station, worst late); "enter pots
# loosely but give up if the flop misses ... overfold to pressure on turns and rivers"
# (fit_or_fold); "feel they have to call one bet and at least see the turn card" (both
# call the flop).
#
# The STRUCTURE is the point: all three agree pre-flop and on the flop, and different
# PAIRS separate in different classes --- station vs fit_or_fold only when facing a bet
# on turn/river, trap vs both only at turn/river first-in.  A globally-biased opponent
# (one bias everywhere) would make d_C identical in every class and the per-class
# machinery pointless, which is exactly what this table avoids.
# --------------------------------------------------------------------------- #
PROFILES = {
    "blueprint":   {},
    "station":     {("flop", 1): "call", ("turn", 1): "call", ("river", 1): "call"},
    "fit_or_fold": {("flop", 1): "call", ("turn", 1): "fold", ("river", 1): "fold"},
    "trap":        {("flop", 1): "call", ("turn", 0): "raise", ("river", 0): "raise"},
}

_STREETS = ("pre_flop", "flop", "turn", "river")


def _class_of(env) -> tuple:
    """The strategic class of the node the env is at: ``(street, facing-a-bet)``."""
    from environment.poker_env import raise_level

    stage = env.betting_stage
    return (stage, raise_level(stage, env.n_raises_this_round))


def _bias_for(profile: dict, cls: tuple) -> str:
    """The bias class ``profile`` applies at ``cls`` (``"none"`` where it is unbiased)."""
    return profile.get(cls, "none")


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
    policies = {m: BlueprintPolicy(tables, bias_multiplier=m) for m in mults}
    walker = policies[mults[0]]   # bias "none" ignores the multiplier — any will do

    names = list(PROFILES)
    pairs = list(itertools.combinations(names, 2))
    # multiplier -> class -> pair -> list of symmetric KLs; class -> node count.
    kls: dict = {m: defaultdict(lambda: defaultdict(list)) for m in mults}
    n_nodes: dict = defaultdict(int)

    for hand in range(args.hands):
        np.random.seed(args.seed * 1_000_003 + hand)
        players = [Player(i, args.starting_stack) for i in range(args.n_players)]
        env = PokerEnv(players=players, small_blind=args.small_blind,
                       big_blind=args.big_blind, low_card_rank=args.low_card_rank,
                       high_card_rank=args.high_card_rank)
        env.card_info_lut = lut
        rng = np.random.RandomState(args.seed * 7919 + hand)

        guard = 0
        while not env.is_terminal and guard < 200:
            guard += 1
            ps = env.policy_state
            legal = ps.legal_actions
            if not legal:
                break
            cls = _class_of(env)
            n_nodes[cls] += 1

            # One policy row per (multiplier, type) at this node, then every pairwise
            # divergence.  All multipliers share this node because the walk below does not
            # depend on them.
            for m in mults:
                rows_m = {t: np.asarray(
                    policies[m].strategy(ps, _bias_for(PROFILES[t], cls)),
                    dtype=np.float64) for t in names}
                for a, b in pairs:
                    ra, rb = rows_m[a], rows_m[b]
                    if ra.shape != rb.shape or ra.size == 0:
                        continue
                    # Symmetric (Jeffreys) divergence: separability is direction-free, and
                    # the asymmetric form would depend on which type we called "true".
                    kls[m][cls][(a, b)].append(_kl(ra, rb) + _kl(rb, ra))

            # Advance under the UNBIASED blueprint: reach(C) is then a property of
            # baseline play, the common reference every type is measured against.
            probs = np.clip(np.asarray(walker.strategy(ps, "none"), dtype=np.float64),
                            0.0, None)
            c = np.cumsum(probs)
            idx = (int(np.searchsorted(c, rng.random_sample() * c[-1]))
                   if c[-1] > 0 else int(rng.randint(len(legal))))
            env.step_in_place(legal[min(idx, len(legal) - 1)])

    total = sum(n_nodes.values()) or 1
    order = [(s, lv) for s in _STREETS for lv in (0, 1) if (s, lv) in n_nodes]

    result = {"hands": args.hands, "multipliers": mults,
              "profiles": {k: {"%s/%d" % c: v for c, v in p.items()}
                           for k, p in PROFILES.items()},
              "by_multiplier": {}}
    for m in mults:
        print("\nd_C — symmetric KL between type policies, per strategic class")
        print("bias_multiplier=%.2f  hands=%d  nodes=%d\n" % (m, args.hands, total))
        head = "%-16s %7s %7s  " % ("class", "nodes", "reach")
        head += "  ".join("%-22s" % ("%s|%s" % (a[:9], b[:9])) for a, b in pairs)
        print(head)
        print("-" * len(head))
        per_class = {}
        for cls in order:
            n = n_nodes[cls]
            row = "%-16s %7d %6.1f%%  " % ("%s L%d" % cls, n, 100.0 * n / total)
            cell = {}
            for a, b in pairs:
                v = kls[m][cls].get((a, b), [])
                mean = float(np.mean(v)) if v else 0.0
                cell["%s|%s" % (a, b)] = mean
                row += "%-22.4f" % mean
            print(row)
            per_class["%s L%d" % cls] = {"nodes": n, "reach": n / total, "d_C": cell}
        result["by_multiplier"]["%g" % m] = per_class

    print("\nReading it: 0.0 = the two profiles are the SAME object in that class"
          "\n(nothing to learn, nothing to exploit).  What you want is a MIXED picture —"
          "\nzero where the profiles agree, clearly nonzero where they disagree."
          "\nLarge everywhere ⇒ multiplier too big; ~0 everywhere ⇒ too small.")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
