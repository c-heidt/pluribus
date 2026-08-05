"""End-of-run experiment summary (docs/evaluation.md §8).

``python -m evaluation.summarize <snapshot.sqlite>`` reads the four §6 tables into
a compact human-readable sanity check — *is the bot winning, did search stay in
budget, is range tracking helping?* — prints it to the log and writes a
``summary.json`` next to the snapshot for programmatic cross-run comparison.

It is **read-only** and has **no search-package dependency** (it reads the schema,
nothing else), so it runs automatically at the end of a run (after the final
sync-back, against the permanent-FS snapshot) *and* standalone against any past
snapshot for a retrospective check.

Design points from the doc:

- Every headline is one grouped query (§8); percentiles and CIs that SQLite has no
  native function for are computed in Python from the pulled column.
- ``CASE``-based conditional aggregation (not the ``FILTER`` clause) so the queries
  run against older SQLite versions on whatever machine does the analysis.
- Thresholds live in one place (:data:`THRESHOLDS`); the summary prints the numbers
  regardless, and only the *flag* lines are threshold-gated (§8 "Automated flags").
- Grain is **one hand per row** (``game_id`` == a deal); bb/100 is a per-hand rate
  (§6), so strength is just ``AVG(100 * hero_chips_delta / big_blind)``.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Tunables — the §8 flag thresholds, in one place so they are easy to retune.
# --------------------------------------------------------------------------- #

THRESHOLDS = {
    "wallcap_budget_bound": 0.40,   # share of wall_cap stops → "budget-bound"
    "fire_rate_silent": 0.01,       # overall fire rate below → "search silent"
    "collapse_high": 0.05,          # resolved collapse rate above → "high collapse"
    "resolved_thin": 0.05,          # resolved fraction below → "thin resolution"
    "blueprint_heavy_high": 0.20,   # share of played searches >50% blueprint → "prior-bound"
}

_CI_Z = 1.96                        # normal-approx 95% CI multiplier (§8)

# Position name by offset-from-button for the table sizes we run (§6/§8 "by
# position").  Offset = (hero_seat - button_seat) mod n_players; 0 == button.
_POSITION_NAMES = {
    2: ["BTN", "BB"],                            # heads-up: button posts the SB
    3: ["BTN", "SB", "BB"],
    6: ["BTN", "SB", "BB", "UTG", "MP", "CO"],
}


# games.hu_from_street values → street names (betting_round indices).
_STREET_NAME = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}


def _position_name(hero_seat: int, button_seat: int, n_players: int) -> str:
    """Offset-from-button position label (``BTN``/``SB``/…); generic fallback."""
    offset = (int(hero_seat) - int(button_seat)) % int(n_players)
    names = _POSITION_NAMES.get(int(n_players))
    if names is not None and 0 <= offset < len(names):
        return names[offset]
    return f"POS{offset}"


# --------------------------------------------------------------------------- #
# Small aggregation helpers (kept pure for testability)
# --------------------------------------------------------------------------- #


def _mean_ci(values: Sequence[float]) -> Dict[str, float]:
    """Mean and normal-approx 95% CI half-width for ``values`` (§8 strength query).

    Uses the doc's population-variance formula (``AVG(x²) − AVG(x)²``) so the
    number matches the documented SQL; the CI half-width is ``z · std / √n``.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "ci95": None}
    mean = sum(values) / n
    var = sum(v * v for v in values) / n - mean * mean
    var = max(var, 0.0)                      # guard tiny negative from rounding
    ci = _CI_Z * math.sqrt(var) / math.sqrt(n) if n > 0 else None
    return {"n": n, "mean": mean, "ci95": ci}


def _bootstrap_ci(
    values: Sequence[float], *, n_resamples: int = 2000, seed: int = 0, alpha: float = 0.05
) -> Dict[str, Optional[float]]:
    """Deterministic percentile-bootstrap 95% CI on the mean of ``values``.

    Fixed-seed stdlib ``random`` (the standalone summary path avoids a numpy
    dependency, see :func:`_percentile`) so the reported interval is reproducible
    run-to-run.  Returns the ``(alpha/2, 1-alpha/2)`` percentiles of the resample
    means; ``None`` bounds for fewer than two points.  Used for the cross-condition
    paired difference (§10.1), where the bootstrap is preferred over the normal
    approximation because per-hand poker outcomes are heavy-tailed.
    """
    n = len(values)
    if n < 2:
        return {"lo": None, "hi": None, "n_resamples": 0}
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return {
        "lo": _percentile(means, 100.0 * (alpha / 2.0)),
        "hi": _percentile(means, 100.0 * (1.0 - alpha / 2.0)),
        "n_resamples": n_resamples,
    }


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated ``q``-percentile (0..100); ``None`` for no data.

    Small self-contained implementation (SQLite has no percentile function and we
    avoid a numpy dependency in the standalone summary path).
    """
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    rank = (q / 100.0) * (len(xs) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(xs[lo])
    frac = rank - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


def _weighted(cells: List[dict], value_key: str, weight_key: str) -> Optional[float]:
    """Weighted mean of ``value_key`` over ``cells`` (skips NULL value / 0 weight)."""
    num = 0.0
    den = 0.0
    for c in cells:
        v = c.get(value_key)
        w = c.get(weight_key)
        if v is None or not w:
            continue
        num += v * w
        den += w
    return (num / den) if den else None


def _rows(con: sqlite3.Connection, sql: str, params: Sequence = ()) -> List[dict]:
    """Run ``sql`` and return rows as dicts keyed by column name."""
    cur = con.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _scalar(con: sqlite3.Connection, sql: str, params: Sequence = ()):
    row = con.execute(sql, params).fetchone()
    return None if row is None else row[0]


# --------------------------------------------------------------------------- #
# Section queries (each returns a plain dict — the summary.json shape)
# --------------------------------------------------------------------------- #


def _query_meta(con: sqlite3.Connection) -> dict:
    """Run-level header: run_id(s), git_sha, hand count, table size (§8 header)."""
    n_hands = _scalar(con, "SELECT COUNT(*) FROM games") or 0
    run_ids = [r["run_id"] for r in _rows(con, "SELECT DISTINCT run_id FROM games")]
    head = _rows(con, "SELECT git_sha, n_players FROM games LIMIT 1")
    git_sha = head[0]["git_sha"] if head else None
    n_players = head[0]["n_players"] if head else None
    return {
        "run_id": run_ids[0] if len(run_ids) == 1 else run_ids,
        "git_sha": git_sha,
        "n_hands": int(n_hands),
        "n_players": n_players,
    }


def _query_strength(con: sqlite3.Connection) -> dict:
    """Hero bb/100 with 95% CI, per table + by position + overall (§8 strength).

    Prefers ``aivat_value`` (§10.2) when every game carries it (unbiased, tighter
    CI); falls back to raw ``hero_chips_delta`` otherwise — the current default,
    since AIVAT is not built yet.  bb/100 = ``100 · value / big_blind`` per hand.
    """
    games = _rows(
        con,
        "SELECT table_label, hero_chips_delta, aivat_value, big_blind, "
        "hero_seat, button_seat, n_players FROM games",
    )
    n_missing_aivat = sum(1 for g in games if g["aivat_value"] is None)
    used_aivat = bool(games) and n_missing_aivat == 0

    def bb100(g) -> Optional[float]:
        bb = g["big_blind"]
        val = g["aivat_value"] if used_aivat else g["hero_chips_delta"]
        if bb is None or bb == 0 or val is None:
            return None
        return 100.0 * val / bb

    by_table: Dict[str, List[float]] = {}
    by_pos: Dict[str, List[float]] = {}
    overall: List[float] = []
    for g in games:
        v = bb100(g)
        if v is None:
            continue
        overall.append(v)
        by_table.setdefault(g["table_label"], []).append(v)
        pos = _position_name(g["hero_seat"], g["button_seat"], g["n_players"])
        by_pos.setdefault(pos, []).append(v)

    return {
        "used_aivat": used_aivat,
        "tables": {t: _mean_ci(vs) for t, vs in sorted(by_table.items())},
        "by_position": {p: _mean_ci(vs) for p, vs in by_pos.items()},
        "overall": _mean_ci(overall),
    }


# Baseline preference for the paired difference: the first arm present becomes the
# reference the others are differenced against (treatment − baseline).  The
# no-exploitation baseline is **vanilla Pluribus** (search, no opponent model); the
# no-search ``blueprint_only`` arm is only a fallback reference (a pipeline test).
_PAIRED_BASELINE_PREFERENCE = ("vanilla", "blueprint_only")


def _query_paired(con: sqlite3.Connection) -> dict:
    """Cross-condition CRN paired differences (§10.1) — the multi-arm headline.

    When two or more ``condition`` arms are present, the comparison of interest is
    not each arm's absolute bb/100 but the **per-hand difference on the matched
    deal**: join arms on ``deck_seed`` (equal per hand when they share
    ``run_seed``/``table_policy``/table shape), difference the bb/100, and CI the
    mean.  The shared card-luck cancels, so the CI is far tighter than differencing
    two independent arm means.  ``None``/absent when fewer than two arms are logged.

    Prefers ``aivat_value`` when every joined game carries it (stacks with the CRN
    cancellation), else raw ``hero_chips_delta``.  Defensive against a rare 32-bit
    ``deck_seed`` collision within an arm: such deck_seeds are dropped from the join
    (ambiguous) and counted, rather than paired arbitrarily.

    Each comparison also carries a **coverage-restricted** delta (§9 A7): the same
    paired difference over only the deals where the *treatment* arm actually made a
    modeled decision (``decisions.modeled_decision = 1``).  A hand the model never
    touched — the hero folded pre-flop, or every play was a blueprint fallback —
    dilutes the full delta toward zero; the restricted delta is the exploitation
    signal where the model applied.  ``None`` on baseline/unmodeled arms.
    """
    n_cond = _scalar(
        con, "SELECT COUNT(DISTINCT condition) FROM games WHERE condition IS NOT NULL"
    )
    n_cond = int(n_cond or 0)
    if n_cond < 2:
        return {"available": False, "n_conditions": n_cond}

    games = _rows(
        con,
        "SELECT condition, deck_seed, hero_chips_delta, aivat_value, big_blind "
        "FROM games WHERE condition IS NOT NULL AND deck_seed IS NOT NULL",
    )
    used_aivat = bool(games) and all(g["aivat_value"] is not None for g in games)

    def bb100(g) -> Optional[float]:
        bb = g["big_blind"]
        val = g["aivat_value"] if used_aivat else g["hero_chips_delta"]
        if bb is None or bb == 0 or val is None:
            return None
        return 100.0 * val / bb

    per_cond: Dict[str, Dict[int, float]] = {}
    dupes: Dict[str, set] = {}
    for g in games:
        c, ds, v = g["condition"], g["deck_seed"], bb100(g)
        if v is None:
            continue
        seen = per_cond.setdefault(c, {})
        if ds in seen:
            dupes.setdefault(c, set()).add(ds)
        else:
            seen[ds] = v
    for c, seeds in dupes.items():                       # drop ambiguous deck_seeds
        for ds in seeds:
            per_cond[c].pop(ds, None)

    # Deck-seeds on which each condition actually applied its exploitation mechanism
    # — the coverage slice.  Keyed by condition so a comparison restricts to the
    # *treatment*'s covered deals.  A deal counts as covered when the treatment made
    # ≥1 genuine treatment decision: a modeled (DBR) decision ``modeled_decision = 1``,
    # OR an OX-Search decision ``ox_enter_prob IS NOT NULL`` (the gadget fired — HU
    # turn/river, Approach B; OX is reach-only so ``modeled_decision`` is always 0 for
    # it, and this per-decision signal is more precise than the per-hand
    # ``hu_from_street``).  Empty for baseline/unmodeled arms (vanilla, blueprint_only).
    covered: Dict[str, set] = {}
    for r in _rows(
        con,
        "SELECT g.condition AS condition, g.deck_seed AS deck_seed "
        "FROM games g JOIN decisions d ON d.game_id = g.game_id "
        "WHERE (d.modeled_decision = 1 OR d.ox_enter_prob IS NOT NULL) "
        "AND g.condition IS NOT NULL "
        "AND g.deck_seed IS NOT NULL GROUP BY g.condition, g.deck_seed",
    ):
        covered.setdefault(r["condition"], set()).add(r["deck_seed"])

    conditions = sorted(per_cond)
    baseline = next(
        (b for b in _PAIRED_BASELINE_PREFERENCE if b in per_cond), conditions[0]
    )
    base_map = per_cond[baseline]
    comparisons = []
    for c in conditions:
        if c == baseline:
            continue
        tmap = per_cond[c]
        shared = sorted(set(tmap) & set(base_map))
        deltas = [tmap[ds] - base_map[ds] for ds in shared]
        stat = _mean_ci(deltas)
        entry = {
            "treatment": c,
            "baseline": baseline,
            "n_paired": len(deltas),
            "mean_delta_bb100": stat["mean"],
            "ci95_normal": stat["ci95"],
            "ci95_bootstrap": _bootstrap_ci(deltas),
        }
        # Coverage-restricted delta: same pairs, but only deals the treatment arm
        # actually modeled.  Absent (None) when the arm has no modeled decisions.
        cov = covered.get(c)
        if cov:
            cshared = [ds for ds in shared if ds in cov]
            cdeltas = [tmap[ds] - base_map[ds] for ds in cshared]
            cstat = _mean_ci(cdeltas)
            entry["covered"] = {
                "n_paired": len(cdeltas),
                "mean_delta_bb100": cstat["mean"],
                "ci95_normal": cstat["ci95"],
                "ci95_bootstrap": _bootstrap_ci(cdeltas),
            }
        else:
            entry["covered"] = None
        comparisons.append(entry)
    return {
        "available": True,
        "metric": "bb100",
        "used_aivat": used_aivat,
        "baseline": baseline,
        "conditions": conditions,
        "dropped_ambiguous_deck_seeds": {c: len(s) for c, s in dupes.items()},
        "comparisons": comparisons,
    }


def _query_range_health(con: sqlite3.Connection) -> dict:
    """Range-tracking net info gain + collapse/fallback, by opponent & stage (§8).

    Joined to ``game_seats`` so quality attributes to the opponent *type* at the
    seat.  Conditional aggregates use ``CASE`` (portable) over resolved snapshots;
    Python rolls the (opponent, stage) grid up to by-opponent / by-stage / overall,
    weighting each cell by its resolved-snapshot count.
    """
    grid = _rows(
        con,
        """
        SELECT s.agent_label            AS opponent,
               rq.betting_stage         AS stage,
               COUNT(*)                 AS n_snap,
               SUM(rq.resolved)         AS n_resolved,
               AVG(CASE WHEN rq.resolved = 1 THEN rq.net_info_gain END)  AS net_gain,
               AVG(CASE WHEN rq.resolved = 1 THEN rq.collapsed_truth END) AS collapse,
               AVG(CAST(rq.uniform_fallback AS REAL))                     AS fallback
        FROM range_quality rq
        JOIN game_seats s ON s.game_id = rq.game_id AND s.seat = rq.seat
        GROUP BY s.agent_label, rq.betting_stage
        """,
    )

    def rollup(cells: List[dict]) -> dict:
        n_snap = sum(c["n_snap"] or 0 for c in cells)
        n_resolved = sum(c["n_resolved"] or 0 for c in cells)
        return {
            "snapshots": int(n_snap),
            "resolved_frac": (n_resolved / n_snap) if n_snap else None,
            "net_info_gain": _weighted(cells, "net_gain", "n_resolved"),
            "collapse_rate": _weighted(cells, "collapse", "n_resolved"),
            "fallback_rate": _weighted(cells, "fallback", "n_snap"),
        }

    by_opponent: Dict[str, List[dict]] = {}
    by_stage: Dict[str, List[dict]] = {}
    for c in grid:
        by_opponent.setdefault(c["opponent"], []).append(c)
        by_stage.setdefault(c["stage"], []).append(c)

    return {
        "overall": rollup(grid),
        "by_opponent": {k: rollup(v) for k, v in sorted(by_opponent.items())},
        "by_stage": {k: rollup(v) for k, v in sorted(by_stage.items())},
    }


def _query_approach(con: sqlite3.Connection) -> dict:
    """Solver-approach usage mix + budget health + a routing correctness check (§8).

    ``regime`` (``mccfr`` | ``vector``) is the approach.  Routing check: the vector
    regime must fire only heads-up (``num_live == 2``) on flop/turn/river (subgame
    §6.5 — a HU post-preflop subgame is "small/late", so all of it takes the vector
    path with the future streets keyed by LUT cluster) — a mis-routed solver is a
    correctness bug the usage mix alone would hide.  The check is scoped to that
    clearest envelope; ``num_live IS NULL`` rows are not counted as violations (older
    rows may not carry it).
    """
    total = _scalar(con, "SELECT COUNT(*) FROM decisions WHERE searched = 1") or 0
    dcols = {r[1] for r in con.execute("PRAGMA table_info(decisions)")}
    approaches = _rows(
        con,
        """
        SELECT regime,
               COUNT(*)                                             AS searches,
               AVG(wall_seconds)                                    AS mean_wall,
               AVG(iterations)                                      AS mean_iters,
               AVG(CASE WHEN stop_reason = 'wall_cap' THEN 1.0 ELSE 0.0 END)
                                                                    AS wallcap_rate
        FROM decisions
        WHERE searched = 1
        GROUP BY regime
        ORDER BY searches DESC
        """,
    )
    for a in approaches:
        a["share"] = (a["searches"] / total) if total else None

    routing_violations = _scalar(
        con,
        "SELECT COUNT(*) FROM decisions WHERE searched = 1 AND regime = 'vector' "
        "AND (betting_stage NOT IN ('flop', 'turn', 'river') "
        "     OR (num_live IS NOT NULL AND num_live != 2))",
    ) or 0
    # Blueprint-prior health (§8): over covered (``searched = 1``) decisions, how much
    # the played search read was shrunk toward the blueprint.  A high mean or heavy
    # rate means the search is adding little over the prior at the nodes it played —
    # under-trained rows, or ``kappa`` set too aggressively.  ``blueprint_weight`` is a
    # v5 column; legacy DBs lack it, so report ``None`` rather than crash.
    bp_mean = bp_heavy = None
    if "blueprint_weight" in dcols and total:
        bp_mean = _scalar(
            con,
            "SELECT AVG(blueprint_weight) FROM decisions "
            "WHERE searched = 1 AND blueprint_weight IS NOT NULL",
        )
        bp_heavy = _scalar(
            con,
            "SELECT AVG(CASE WHEN blueprint_weight > 0.5 THEN 1.0 ELSE 0.0 END) "
            "FROM decisions WHERE searched = 1 AND blueprint_weight IS NOT NULL",
        )
    return {
        "approaches": approaches,
        "total_searched": int(total),
        "routing_violations": int(routing_violations),
        "routing_ok": routing_violations == 0,
        "blueprint_weight_mean": bp_mean,
        "blueprint_heavy_rate": bp_heavy,
    }


def _query_hu_coverage(con: sqlite3.Connection) -> dict:
    """HU coverage: where OX-Search (HU) could have fired.

    ``games.hu_from_street`` is the earliest street at which a betting round began
    heads-up with the hero.  ``eligible`` (street ≥ 2, turn or later) is the
    OX-Search-HU activation predicate (design doc §4.2b); coverage-restricted
    DBR-vs-OX-Search comparisons slice on it.  Logged for every condition, so the
    same query serves
    them all.  Older snapshots (schema v1) lack the column → ``available: False``.
    """
    have = {r[1] for r in con.execute("PRAGMA table_info(games)")}
    if "hu_from_street" not in have:
        return {"available": False}
    n_hands = _scalar(con, "SELECT COUNT(*) FROM games") or 0
    by_street = {
        int(r["hu_from_street"]): r["n"]
        for r in _rows(
            con,
            "SELECT hu_from_street, COUNT(*) AS n FROM games "
            "WHERE hu_from_street IS NOT NULL GROUP BY hu_from_street",
        )
    }
    n_hu = sum(by_street.values())
    n_eligible = sum(n for s, n in by_street.items() if s >= 2)
    per_table = {
        r["table_label"]: (r["elig"] / r["n"]) if r["n"] else None
        for r in _rows(
            con,
            "SELECT table_label, COUNT(*) AS n, "
            "SUM(CASE WHEN hu_from_street >= 2 THEN 1 ELSE 0 END) AS elig "
            "FROM games GROUP BY table_label",
        )
    }
    return {
        "available": True,
        "n_hands": int(n_hands),
        "hu_frac": (n_hu / n_hands) if n_hands else None,
        "eligible_frac": (n_eligible / n_hands) if n_hands else None,
        "by_street": by_street,
        "eligible_frac_by_table": per_table,
    }


def _query_search_cost(con: sqlite3.Connection) -> dict:
    """Search fire rate, wall (mean + p95), stop-reason split, cache hit rate (§8)."""
    n_decisions = _scalar(con, "SELECT COUNT(*) FROM decisions") or 0
    n_searched = _scalar(con, "SELECT COUNT(*) FROM decisions WHERE searched = 1") or 0
    fire_rate = (n_searched / n_decisions) if n_decisions else None

    agg = _rows(
        con,
        """
        SELECT AVG(wall_seconds)  AS mean_wall,
               AVG(iterations)    AS mean_iters,
               AVG(iters_per_sec) AS mean_ips,
               SUM(cache_hits)    AS hits,
               SUM(cache_misses)  AS misses
        FROM decisions WHERE searched = 1
        """,
    )[0]
    hits = agg["hits"] or 0
    misses = agg["misses"] or 0
    cache_hit_rate = (hits / (hits + misses)) if (hits + misses) else None

    walls = [
        r["wall_seconds"]
        for r in _rows(
            con,
            "SELECT wall_seconds FROM decisions "
            "WHERE searched = 1 AND wall_seconds IS NOT NULL",
        )
    ]
    stop = {
        r["stop_reason"]: r["n"]
        for r in _rows(
            con,
            "SELECT stop_reason, COUNT(*) AS n FROM decisions "
            "WHERE searched = 1 GROUP BY stop_reason",
        )
    }
    n_stops = sum(stop.values()) or 0
    return {
        "n_decisions": int(n_decisions),
        "n_searched": int(n_searched),
        "fire_rate": fire_rate,
        "mean_wall": agg["mean_wall"],
        "p95_wall": _percentile(walls, 95.0),
        "mean_iters": agg["mean_iters"],
        "mean_iters_per_sec": agg["mean_ips"],
        "cache_hit_rate": cache_hit_rate,
        "stop_reason": stop,
        "wallcap_rate": (stop.get("wall_cap", 0) / n_stops) if n_stops else None,
    }


# --------------------------------------------------------------------------- #
# Flags (§8 "Automated flags") — the only threshold-gated output
# --------------------------------------------------------------------------- #


def _evaluate_flags(report: dict) -> List[dict]:
    """Turn the §8 thresholds into explicit warn/info lines (numbers print anyway)."""
    flags: List[dict] = []
    T = THRESHOLDS

    # losing — a table whose strength CI lies entirely below 0 (genuine loss).
    for label, s in report["strength"]["tables"].items():
        if s["mean"] is not None and s["ci95"] is not None and s["mean"] + s["ci95"] < 0:
            flags.append({
                "level": "warn", "key": "losing",
                "message": f"losing vs {label}: {s['mean']:.1f} ± {s['ci95']:.1f} bb/100",
            })

    # range net-harmful — tracking hurts (mean net info gain < 0) at some stage.
    for stage, r in report["range_quality"]["by_stage"].items():
        g = r["net_info_gain"]
        if g is not None and g < 0:
            flags.append({
                "level": "warn", "key": "range_net_harmful",
                "message": f"range tracking net-harmful on {stage}: {g:+.2f} nats",
            })

    # budget-bound — search mostly exhausts the wall budget, not the iteration cap.
    wc = report["search"]["wallcap_rate"]
    if wc is not None and wc > T["wallcap_budget_bound"]:
        flags.append({
            "level": "warn", "key": "budget_bound",
            "message": f"{wc:.0%} of searches hit the wall cap — search is budget-bound",
        })

    # search silent — search essentially never fired (likely a trigger/config error).
    fr = report["search"]["fire_rate"]
    if fr is not None and fr < T["fire_rate_silent"]:
        flags.append({
            "level": "warn", "key": "search_silent",
            "message": f"search fired {fr:.1%} of decisions — likely a trigger/config error",
        })

    # high collapse — belief routinely rules out reality (replay / card-removal bug).
    cr = report["range_quality"]["overall"]["collapse_rate"]
    if cr is not None and cr > T["collapse_high"]:
        flags.append({
            "level": "warn", "key": "high_collapse",
            "message": f"collapsed truth {cr:.1%} of resolved — belief rules out reality",
        })

    # routing — an approach fired outside its intended envelope (correctness bug).
    if not report["approach"]["routing_ok"]:
        flags.append({
            "level": "warn", "key": "routing",
            "message": f"{report['approach']['routing_violations']} vector searches "
                       "fired outside heads-up flop/turn/river",
        })

    # prior-bound — the search often plays a mostly-blueprint mix at the nodes it
    # covers (starved rows, or kappa too aggressive): search is adding little there.
    bh = report["approach"].get("blueprint_heavy_rate")
    if bh is not None and bh > T["blueprint_heavy_high"]:
        flags.append({
            "level": "warn", "key": "blueprint_prior_bound",
            "message": f"{bh:.0%} of played searches were >50% blueprint prior — "
                       "search adds little over the blueprint at those nodes",
        })

    # thin resolution — range metrics rest on very few showdowns (weak evidence).
    rf = report["range_quality"]["overall"]["resolved_frac"]
    if rf is not None and rf < T["resolved_thin"]:
        flags.append({
            "level": "info", "key": "thin_resolution",
            "message": f"resolved fraction {rf:.1%} — range aggregates are weak evidence",
        })
    return flags


# --------------------------------------------------------------------------- #
# Human-readable block (§8 layout)
# --------------------------------------------------------------------------- #


def _fmt(v, spec: str = ".2f", none: str = "n/a") -> str:
    return none if v is None else format(v, spec)


def _print_human(report: dict) -> str:
    """Render the §8 summary block; returns the string (also logged by the caller)."""
    m = report["meta"]
    st = report["strength"]
    rq = report["range_quality"]
    ap = report["approach"]
    sc = report["search"]
    L: List[str] = []
    run_id = m["run_id"] if isinstance(m["run_id"], str) else ",".join(m["run_id"] or [])
    L.append(
        f"experiment  run_id={run_id}   git={m['git_sha']}   "
        f"{m['n_hands']} hands   {m['n_players']}-max"
    )
    L.append("─" * 76)

    metric = "aivat bb/100" if st["used_aivat"] else "hero bb/100"
    L.append(f"STRENGTH ({metric}, 95% CI)")
    for label, s in st["tables"].items():
        L.append(f"  table={label:<22} {_fmt(s['mean'],'+.1f')} ± "
                 f"{_fmt(s['ci95'],'.1f')}   ({s['n']} hands)")
    ov = st["overall"]
    L.append(f"  overall{'':<22} {_fmt(ov['mean'],'+.1f')} ± "
             f"{_fmt(ov['ci95'],'.1f')}   ({ov['n']} hands)")
    if st["by_position"]:
        pos = "  ".join(
            f"{p} {_fmt(s['mean'],'+.0f')}" for p, s in st["by_position"].items()
        )
        L.append(f"  by position  {pos}  (bb/100)")

    pr = report.get("paired")
    if pr and pr.get("available"):
        metric = "aivat" if pr["used_aivat"] else "raw"
        L.append("")
        L.append(
            f"PAIRED Δ vs {pr['baseline']}  (CRN, {metric} bb/100, deck-matched; "
            "95% bootstrap CI)"
        )
        for cmp in pr["comparisons"]:
            b = cmp["ci95_bootstrap"]
            L.append(
                f"  {cmp['treatment']:<14} {_fmt(cmp['mean_delta_bb100'],'+.2f')}  "
                f"[{_fmt(b['lo'],'+.2f')}, {_fmt(b['hi'],'+.2f')}]   "
                f"({cmp['n_paired']} paired hands)"
            )
            cov = cmp.get("covered")
            if cov:
                cb = cov["ci95_bootstrap"]
                L.append(
                    f"  {'  └ modeled only':<14} "
                    f"{_fmt(cov['mean_delta_bb100'],'+.2f')}  "
                    f"[{_fmt(cb['lo'],'+.2f')}, {_fmt(cb['hi'],'+.2f')}]   "
                    f"({cov['n_paired']} paired hands)"
                )

    ov = rq["overall"]
    rf = ov["resolved_frac"]
    L.append("")
    L.append(f"RANGE TRACKING (resolved at showdown: {_fmt(rf,'.1%')} of seat-snapshots)")
    L.append(f"  net info gain    {_fmt(ov['net_info_gain'],'+.2f')} nats overall")
    if rq["by_opponent"]:
        opp = "   ".join(
            f"{k} {_fmt(v['net_info_gain'],'+.2f')}" for k, v in rq["by_opponent"].items()
        )
        L.append(f"    by opponent    {opp}")
    if rq["by_stage"]:
        stg = "   ".join(
            f"{k} {_fmt(v['net_info_gain'],'+.2f')}" for k, v in rq["by_stage"].items()
        )
        L.append(f"    by stage       {stg}")
    L.append(f"  collapsed truth  {_fmt(ov['collapse_rate'],'.1%')} of resolved   "
             f"uniform fallback {_fmt(ov['fallback_rate'],'.1%')} of seat-snapshots")

    hc = report["hu_coverage"]
    if hc.get("available"):
        L.append("")
        L.append("HU COVERAGE (OX-Search-HU eligibility: heads-up with hero from turn+)")
        streets = "  ".join(
            f"{_STREET_NAME.get(s, s)} {n}" for s, n in sorted(hc["by_street"].items())
        )
        L.append(
            f"  HU-with-hero     {_fmt(hc['hu_frac'],'.1%')} of hands   "
            f"eligible (turn+) {_fmt(hc['eligible_frac'],'.1%')}"
        )
        if streets:
            L.append(f"    first HU street  {streets}")
        if hc["eligible_frac_by_table"]:
            per = "   ".join(
                f"{k} {_fmt(v,'.0%')}" for k, v in hc["eligible_frac_by_table"].items()
            )
            L.append(f"    eligible by table  {per}")

    L.append("")
    L.append("SOLVER APPROACH  (share · mean wall · wall-cap rate · mean iters)")
    for a in ap["approaches"]:
        L.append(
            f"  {str(a['regime']):<7} "
            f"{_fmt(a['share'],'.0%'):>5}   {_fmt(a['mean_wall'],'.1f')}s   "
            f"{_fmt(a['wallcap_rate'],'.0%')} wall-cap   "
            f"{_fmt(a['mean_iters'],'.0f')} it"
        )
    L.append(f"  routing check: {'OK ✓' if ap['routing_ok'] else 'VIOLATIONS ✗'}")
    if ap.get("blueprint_weight_mean") is not None:
        L.append(
            f"  blueprint prior: mean weight {_fmt(ap['blueprint_weight_mean'],'.1%')} · "
            f"heavy (>50%) {_fmt(ap['blueprint_heavy_rate'],'.1%')} of played searches"
        )

    L.append("")
    L.append("SEARCH COST / BUDGET")
    L.append(f"  fired            {_fmt(sc['fire_rate'],'.1%')} of decisions "
             f"({sc['n_searched']} / {sc['n_decisions']})")
    L.append(f"  wall / search    mean {_fmt(sc['mean_wall'],'.1f')}s   "
             f"p95 {_fmt(sc['p95_wall'],'.1f')}s")
    L.append(f"  stop reason      wall_cap {_fmt(sc['wallcap_rate'],'.0%')}")
    L.append(f"  iterations       mean {_fmt(sc['mean_iters'],'.0f')}   "
             f"iters/s {_fmt(sc['mean_iters_per_sec'],'.0f')}   "
             f"cache hit rate {_fmt(sc['cache_hit_rate'],'.1%')}")

    if report["flags"]:
        L.append("")
        L.append("FLAGS")
        for f in report["flags"]:
            mark = "⚠" if f["level"] == "warn" else "ⓘ"
            L.append(f"  {mark} {f['message']}")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Orchestrator + entry point
# --------------------------------------------------------------------------- #


def build_report(con: sqlite3.Connection) -> dict:
    """Assemble the full report dict from an open (read-only) connection."""
    report = {
        "meta": _query_meta(con),
        "strength": _query_strength(con),
        "paired": _query_paired(con),
        "range_quality": _query_range_health(con),
        "hu_coverage": _query_hu_coverage(con),
        "approach": _query_approach(con),
        "search": _query_search_cost(con),
    }
    report["flags"] = _evaluate_flags(report)
    return report


def summarize(db_path, *, write_json: bool = True, echo: bool = True) -> dict:
    """Summarize the snapshot at ``db_path`` (§8); return the report dict.

    Opens the file **read-only** (never the live node-local WAL file — the runner
    points this at the permanent-FS snapshot).  Prints the human block and, unless
    disabled, writes ``summary.json`` next to the snapshot for cross-run comparison.
    """
    db_path = Path(db_path)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        report = build_report(con)
    finally:
        con.close()

    block = _print_human(report)
    if echo:
        print(block)
    if write_json:
        with open(db_path.parent / "summary.json", "w") as fh:
            json.dump(report, fh, indent=2, default=str)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m evaluation.summarize",
        description="Print the end-of-run experiment summary (docs/evaluation.md §8).",
    )
    parser.add_argument("snapshot", help="Path to an experiment .sqlite snapshot.")
    parser.add_argument(
        "--no-json", action="store_true", help="Do not write summary.json."
    )
    args = parser.parse_args(argv)
    summarize(args.snapshot, write_json=not args.no_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
