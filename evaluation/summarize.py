"""End-of-run experiment summary (docs/evaluation.md §8).

``python -m evaluation.summarize <snapshot.sqlite>`` reads the four §6 tables into
a compact human-readable sanity check — *is the bot winning, did search stay in
budget, is range tracking helping?* — prints it to the log and writes a
``summary.json`` next to the snapshot for programmatic cross-run comparison.

It is **read-only** and has **no search-package dependency** (it reads the schema,
nothing else), so it runs automatically at the end of a run (after the final
sync-back, against the permanent-FS snapshot) *and* standalone against any past
snapshot for a retrospective check.

Design points:

- **Grain: the arm.** A snapshot routinely holds several *arms* — a ``condition``
  (``vanilla`` / ``DBR(...)`` / ``OX(...)`` / ``blueprint_only``) crossed with a
  ``table_label`` (the opponent mix).  Under CRN the arms replay the *same deals*,
  so a snapshot with A arms has A rows per deal.  Every number here is therefore
  computed **within one arm**; nothing is averaged or counted across arms.  A
  bb/100 pooled over vanilla and DBR rows answers no question, and a pooled hand /
  decision / search count is just the deal count multiplied by A.
- **Cross-arm ⇒ paired difference only.** The single legitimate cross-arm number is
  the CRN paired Δ (§10.1) — matched on the *same deal*, so it is a difference, not
  a pool, and the shared card-luck cancels.  Deals are matched on
  ``(table_label, deck_seed)``: ``deck_seed`` is a pure function of
  ``(run_seed, hand_index)`` and is **not** table-dependent, so a snapshot holding
  two table policies repeats every ``deck_seed`` and matching on the seed alone
  collapses the whole comparison.
- **Rates, not shares.** "Search fired" is reported as a rate *within* a street
  (searched ÷ decisions at that street), never as a regime's share of a pooled
  search total — regimes run at disjoint streets (subgame §6.5), so a share only
  restates how often each street came up.  Costs (wall, iterations) are likewise
  per street: the budget is a per-street constant, so a mean across streets is a
  mean over a mixture nobody chose.
- **The one meaningful sum** is cost: total search seconds ÷ hands = wall per hand,
  the number an experiment budget is actually built from.
- Percentiles and CIs SQLite has no native function for are computed in Python from
  the pulled column; ``CASE``-based conditional aggregation (not the ``FILTER``
  clause) so the queries run against older SQLite versions.
- Thresholds live in one place (:data:`THRESHOLDS`); the summary prints the numbers
  regardless, and only the *flag* lines are threshold-gated (§8 "Automated flags").
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Tunables — the §8 flag thresholds, in one place so they are easy to retune.
# --------------------------------------------------------------------------- #

THRESHOLDS = {
    "wallcap_budget_bound": 0.40,   # share of wall_cap stops → "budget-bound"
    "fire_rate_silent": 0.01,       # street fire rate below → "search silent"
    "collapse_high": 0.05,          # resolved collapse rate above → "high collapse"
    "resolved_thin": 0.05,          # resolved fraction below → "thin resolution"
    "pairing_thin": 0.80,           # paired ÷ min(arm hands) below → "thin pairing"
    "arm_imbalance": 0.02,          # relative hand-count spread above → "unbalanced"
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

# Street print/sort order; anything unknown sorts last under its own name.
_STAGE_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

# Label for rows written before ``games.condition`` existed, or by a single-arm run
# that never set it.  Keeps such a run a first-class (single) arm rather than a hole.
_UNLABELLED = "(unlabelled)"

# Baseline preference for the paired difference: the first arm present becomes the
# reference the others are differenced against (treatment − baseline).  The
# no-exploitation baseline is **vanilla Pluribus** (search, no opponent model); the
# no-search ``blueprint_only`` arm is only a fallback reference (a pipeline test).
_PAIRED_BASELINE_PREFERENCE = ("vanilla", "blueprint_only")

# Conditions that are *supposed* to never search, so a zero fire rate is correct
# behaviour there and must not raise the "search silent" flag.
_NO_SEARCH_CONDITIONS = ("blueprint_only",)


def _position_name(hero_seat: int, button_seat: int, n_players: int) -> str:
    """Offset-from-button position label (``BTN``/``SB``/…); generic fallback."""
    offset = (int(hero_seat) - int(button_seat)) % int(n_players)
    names = _POSITION_NAMES.get(int(n_players))
    if names is not None and 0 <= offset < len(names):
        return names[offset]
    return f"POS{offset}"


def _condition_label(value) -> str:
    """``games.condition`` as a display/group key; NULL becomes :data:`_UNLABELLED`."""
    return value if value else _UNLABELLED


def _stage_sort_key(stage) -> Tuple[int, str]:
    return (_STAGE_ORDER.get(stage, len(_STAGE_ORDER)), str(stage))


def _condition_sort_key(condition) -> Tuple[int, str]:
    """Sort conditions with the paired baseline first, then alphabetically."""
    low = str(condition).strip().lower()
    try:
        rank = _PAIRED_BASELINE_PREFERENCE.index(low)
    except ValueError:
        rank = len(_PAIRED_BASELINE_PREFERENCE)
    return (rank, str(condition))


def _arm_sort_key(arm: Tuple[str, str]) -> Tuple[str, int, str]:
    """Arms print grouped by table, baseline condition first inside each table."""
    condition, table = arm
    rank, name = _condition_sort_key(condition)
    return (str(table), rank, name)


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


def _verdict(mean: Optional[float], ci: Optional[float]) -> str:
    """``winning`` / ``losing`` / ``inconclusive`` from a mean and its CI half-width.

    §8: a CI straddling zero is *inconclusive*, not *bad* — the distinction the old
    bare "±" line left the reader to make (and that the flags then contradicted).
    """
    if mean is None or ci is None:
        return "n/a"
    if mean - ci > 0:
        return "winning"
    if mean + ci < 0:
        return "losing"
    return "inconclusive"


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

    Each resample is drawn with ``Random.choices`` rather than an explicit
    ``randrange`` loop: the paired section now bootstraps once per (comparison,
    table) cell plus its coverage slice, so the inner loop runs often enough for the
    ~4× to matter (≈10s → ≈2s per cell at 10k pairs).
    """
    n = len(values)
    if n < 2:
        return {"lo": None, "hi": None, "n_resamples": 0}
    draw = random.Random(seed).choices
    xs = list(values)
    means = sorted(sum(draw(xs, k=n)) / n for _ in range(n_resamples))
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
    """Run-level header: run_id(s), git_sha, table size, and the **arm inventory**.

    The headline count is ``n_deals`` — distinct ``(table_label, deck_seed)`` pairs,
    i.e. how many *situations* were actually played — kept separate from ``n_rows``,
    the ``games`` row count, which under CRN is ``n_deals × arms``.  Reporting the
    row count as "hands" is the arithmetic that made the old header claim 2400 hands
    for a 400-deal, 3-arm, 2-table experiment.
    """
    n_rows = _scalar(con, "SELECT COUNT(*) FROM games") or 0
    n_deals = _scalar(
        con, "SELECT COUNT(*) FROM (SELECT DISTINCT table_label, deck_seed FROM games)"
    ) or 0
    run_ids = [r["run_id"] for r in _rows(con, "SELECT DISTINCT run_id FROM games")]
    head = _rows(con, "SELECT git_sha, n_players FROM games LIMIT 1")

    arm_rows = _rows(
        con,
        "SELECT condition, table_label, COUNT(*) AS n_hands FROM games "
        "GROUP BY condition, table_label",
    )
    arms = [
        {
            "condition": _condition_label(r["condition"]),
            "table": r["table_label"],
            "n_hands": int(r["n_hands"]),
        }
        for r in arm_rows
    ]
    arms.sort(key=lambda a: _arm_sort_key((a["condition"], a["table"])))
    counts = [a["n_hands"] for a in arms]
    # Relative spread of arm sizes: CRN pairing assumes every arm covered the same
    # hand_index range, so an imbalance means an arm died early or resumed short.
    imbalance = ((max(counts) - min(counts)) / max(counts)) if counts else None

    conditions = sorted({a["condition"] for a in arms}, key=_condition_sort_key)
    tables = sorted({a["table"] for a in arms}, key=str)
    return {
        "run_id": run_ids[0] if len(run_ids) == 1 else run_ids,
        "git_sha": head[0]["git_sha"] if head else None,
        "n_players": head[0]["n_players"] if head else None,
        "n_rows": int(n_rows),
        "n_deals": int(n_deals),
        "conditions": conditions,
        "tables": tables,
        "arms": arms,
        "n_arms": len(arms),
        "multi_arm": len(conditions) > 1,
        "arm_imbalance": imbalance,
    }


def _bb100_reader(games: List[dict]):
    """``(reader, used_aivat)`` for per-hand bb/100 over ``games``.

    Prefers ``aivat_value`` (§10.2) when **every** row carries it (unbiased, tighter
    CI); falls back to raw ``hero_chips_delta`` otherwise.  Shared by the strength
    and paired sections so the two headlines can never quietly use different metrics.
    """
    used_aivat = bool(games) and all(g["aivat_value"] is not None for g in games)

    def bb100(g) -> Optional[float]:
        bb = g["big_blind"]
        val = g["aivat_value"] if used_aivat else g["hero_chips_delta"]
        if bb is None or bb == 0 or val is None:
            return None
        return 100.0 * val / bb

    return bb100, used_aivat


def _query_strength(con: sqlite3.Connection) -> dict:
    """Hero bb/100 with 95% CI **per arm** — ``(condition, table_label)`` (§8).

    One row per arm, plus that arm's by-position split.  There is deliberately no
    cross-arm "overall": arms differ in the thing under test, so their pooled mean
    is a mixture whose weights are an accident of how many hands each arm got.  Use
    :func:`_query_paired` to compare arms.
    """
    games = _rows(
        con,
        "SELECT condition, table_label, hero_chips_delta, aivat_value, big_blind, "
        "hero_seat, button_seat, n_players FROM games",
    )
    bb100, used_aivat = _bb100_reader(games)

    by_arm: Dict[Tuple[str, str], List[float]] = {}
    by_arm_pos: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for g in games:
        v = bb100(g)
        if v is None:
            continue
        arm = (_condition_label(g["condition"]), g["table_label"])
        by_arm.setdefault(arm, []).append(v)
        pos = _position_name(g["hero_seat"], g["button_seat"], g["n_players"])
        by_arm_pos.setdefault(arm, {}).setdefault(pos, []).append(v)

    arms = []
    for arm in sorted(by_arm, key=_arm_sort_key):
        stat = _mean_ci(by_arm[arm])
        positions = by_arm_pos.get(arm, {})
        arms.append({
            "condition": arm[0],
            "table": arm[1],
            "n_hands": stat["n"],
            "mean_bb100": stat["mean"],
            "ci95": stat["ci95"],
            "verdict": _verdict(stat["mean"], stat["ci95"]),
            "by_position": {
                p: _mean_ci(vs)
                for p, vs in sorted(positions.items(), key=lambda kv: kv[0])
            },
        })
    return {
        "metric": "aivat_bb100" if used_aivat else "raw_bb100",
        "used_aivat": used_aivat,
        "grain": "condition x table_label",
        "arms": arms,
    }


def _query_paired(con: sqlite3.Connection) -> dict:
    """Cross-condition CRN paired differences (§10.1) — the multi-arm headline.

    When two or more ``condition`` arms are present, the comparison of interest is
    not each arm's absolute bb/100 but the **per-hand difference on the matched
    deal**: match arms on the deal, difference the bb/100, and CI the mean.  The
    shared card-luck cancels, so the CI is far tighter than differencing two
    independent arm means.  ``available: False`` when fewer than two arms are logged.

    The match key is ``(table_label, deck_seed)``, **not** ``deck_seed`` alone:
    ``deck_seed`` is derived from ``(run_seed, hand_index)`` only
    (:func:`evaluation.runner.derive_seeds`), so a snapshot holding two table
    policies repeats every seed.  Keyed on the seed alone every deal looked like a
    within-arm duplicate, got dropped as ambiguous, and the whole comparison
    silently reported zero pairs.  Genuine duplicates — the same arm re-run at the
    same ``run_seed`` — are still dropped and counted.

    Each comparison is reported per table (the opponent mix changes the size of the
    edge) plus, when more than one table is present, a pooled row over all matched
    deals.  Both arms' matched-sample means are carried alongside the Δ so the
    difference can be read against the levels it came from.

    Each comparison also carries a **coverage-restricted** delta (§9 A7): the same
    paired difference over only the deals where the *treatment* arm actually made a
    modeled decision (``decisions.modeled_decision = 1``) or fired the OX gadget
    (``ox_enter_prob IS NOT NULL``).  A hand the model never touched — the hero
    folded pre-flop, or every play was a blueprint fallback — dilutes the full delta
    toward zero; the restricted delta is the exploitation signal where the mechanism
    applied.  ``None`` on baseline/unmodeled arms.
    """
    conditions_present = [
        r["condition"]
        for r in _rows(
            con, "SELECT DISTINCT condition FROM games WHERE condition IS NOT NULL"
        )
    ]
    if len(conditions_present) < 2:
        return {"available": False, "n_conditions": len(conditions_present)}

    games = _rows(
        con,
        "SELECT condition, table_label, deck_seed, hero_chips_delta, aivat_value, "
        "big_blind FROM games WHERE condition IS NOT NULL AND deck_seed IS NOT NULL",
    )
    bb100, used_aivat = _bb100_reader(games)

    per_cond: Dict[str, Dict[Tuple[str, int], float]] = {}
    dupes: Dict[str, set] = {}
    for g in games:
        v = bb100(g)
        if v is None:
            continue
        key = (g["table_label"], g["deck_seed"])
        seen = per_cond.setdefault(g["condition"], {})
        if key in seen:                                  # same arm played it twice
            dupes.setdefault(g["condition"], set()).add(key)
        else:
            seen[key] = v
    for c, keys in dupes.items():                        # drop ambiguous deals
        for k in keys:
            per_cond[c].pop(k, None)

    # Deals on which each condition actually applied its exploitation mechanism — the
    # coverage slice, keyed the same way as the pairing so a comparison can restrict
    # to the *treatment*'s covered deals.  Empty for baseline/unmodeled arms.
    covered: Dict[str, set] = {}
    for r in _rows(
        con,
        "SELECT g.condition AS condition, g.table_label AS table_label, "
        "g.deck_seed AS deck_seed "
        "FROM games g JOIN decisions d ON d.game_id = g.game_id "
        "WHERE (d.modeled_decision = 1 OR d.ox_enter_prob IS NOT NULL) "
        "AND g.condition IS NOT NULL AND g.deck_seed IS NOT NULL "
        "GROUP BY g.condition, g.table_label, g.deck_seed",
    ):
        covered.setdefault(r["condition"], set()).add((r["table_label"], r["deck_seed"]))

    conditions = sorted(per_cond, key=_condition_sort_key)
    if len(conditions) < 2:
        # Two arms are labelled but fewer than two carry a usable outcome — every
        # hand missing ``hero_chips_delta``/``aivat_value``, or all of an arm's deals
        # dropped as ambiguous.  Nothing to difference; say so rather than raise.
        return {
            "available": False,
            "n_conditions": len(conditions_present),
            "n_conditions_with_values": len(conditions),
            "dropped_ambiguous_deals": {c: len(s) for c, s in dupes.items()},
        }
    baseline = conditions[0]                             # baseline preference first
    base_map = per_cond[baseline]

    def _stats(keys: Sequence[Tuple[str, int]], tmap) -> dict:
        """Matched-sample stats for one group of paired deals.

        The verdict reads the **bootstrap** interval, since that is the interval
        printed: ``better``/``worse`` only when it excludes zero.  Falls back to the
        normal CI when the bootstrap is undefined (fewer than two pairs).
        """
        deltas = [tmap[k] - base_map[k] for k in keys]
        stat = _mean_ci(deltas)
        boot = _bootstrap_ci(deltas)
        if boot["lo"] is not None:
            verdict = ("better" if boot["lo"] > 0 else
                       "worse" if boot["hi"] < 0 else "inconclusive")
        else:
            v = _verdict(stat["mean"], stat["ci95"])
            verdict = {"winning": "better", "losing": "worse"}.get(v, v)
        return {
            "n_paired": len(deltas),
            "baseline_mean_bb100": (
                sum(base_map[k] for k in keys) / len(keys) if keys else None
            ),
            "treatment_mean_bb100": (
                sum(tmap[k] for k in keys) / len(keys) if keys else None
            ),
            "mean_delta_bb100": stat["mean"],
            "ci95_normal": stat["ci95"],
            "ci95_bootstrap": boot,
            "verdict": verdict,
        }

    comparisons = []
    for c in conditions:
        if c == baseline:
            continue
        tmap = per_cond[c]
        shared = sorted(set(tmap) & set(base_map))
        tables = sorted({k[0] for k in shared}, key=str)
        groups: List[Tuple[Optional[str], List[Tuple[str, int]]]] = [
            (t, [k for k in shared if k[0] == t]) for t in tables
        ]
        if len(tables) > 1:                              # pooled row, clearly labelled
            groups.append((None, shared))
        cov = covered.get(c) or set()

        cells = []
        for table, keys in groups:
            cell = {"table": table}
            cell.update(_stats(keys, tmap))
            if cov:
                ckeys = [k for k in keys if k in cov]
                cell["covered"] = _stats(ckeys, tmap)
            else:
                cell["covered"] = None
            cells.append(cell)

        comparisons.append({
            "treatment": c,
            "baseline": baseline,
            "n_baseline_hands": len(base_map),
            "n_treatment_hands": len(tmap),
            "n_paired": len(shared),
            "pairing_rate": (
                len(shared) / min(len(base_map), len(tmap))
                if base_map and tmap else None
            ),
            "cells": cells,
        })

    return {
        "available": True,
        "metric": "aivat_bb100" if used_aivat else "raw_bb100",
        "used_aivat": used_aivat,
        "match_key": "table_label + deck_seed",
        "baseline": baseline,
        "conditions": conditions,
        "dropped_ambiguous_deals": {c: len(s) for c, s in dupes.items()},
        "comparisons": comparisons,
    }


def _query_range_health(con: sqlite3.Connection) -> dict:
    """Range-tracking net info gain + collapse/fallback, **per condition** (§8).

    The belief is produced by the arm that is playing, so quality is a property of
    the condition, not of the snapshot — pooling a DBR arm's tracker with vanilla's
    averages two different trackers.  Within a condition the grid is joined to
    ``game_seats`` so quality still attributes to the opponent *type* at the seat,
    and is rolled up to by-opponent / by-stage, each cell weighted by its
    resolved-snapshot count.
    """
    grid = _rows(
        con,
        """
        SELECT g.condition             AS condition,
               s.agent_label           AS opponent,
               rq.betting_stage        AS stage,
               COUNT(*)                AS n_snap,
               SUM(rq.resolved)        AS n_resolved,
               AVG(CASE WHEN rq.resolved = 1 THEN rq.net_info_gain END)   AS net_gain,
               AVG(CASE WHEN rq.resolved = 1 THEN rq.collapsed_truth END) AS collapse,
               AVG(CAST(rq.uniform_fallback AS REAL))                     AS fallback
        FROM range_quality rq
        JOIN game_seats s ON s.game_id = rq.game_id AND s.seat = rq.seat
        JOIN games g      ON g.game_id = rq.game_id
        GROUP BY g.condition, s.agent_label, rq.betting_stage
        """,
    )

    def rollup(cells: List[dict]) -> dict:
        n_snap = sum(c["n_snap"] or 0 for c in cells)
        n_resolved = sum(c["n_resolved"] or 0 for c in cells)
        return {
            "snapshots": int(n_snap),
            "resolved": int(n_resolved),
            "resolved_frac": (n_resolved / n_snap) if n_snap else None,
            "net_info_gain": _weighted(cells, "net_gain", "n_resolved"),
            "collapse_rate": _weighted(cells, "collapse", "n_resolved"),
            "fallback_rate": _weighted(cells, "fallback", "n_snap"),
        }

    by_cond: Dict[str, List[dict]] = {}
    for c in grid:
        by_cond.setdefault(_condition_label(c["condition"]), []).append(c)

    conditions = []
    for cond in sorted(by_cond, key=_condition_sort_key):
        cells = by_cond[cond]
        by_opponent: Dict[str, List[dict]] = {}
        by_stage: Dict[str, List[dict]] = {}
        for c in cells:
            by_opponent.setdefault(c["opponent"], []).append(c)
            by_stage.setdefault(c["stage"], []).append(c)
        conditions.append({
            "condition": cond,
            "overall": rollup(cells),
            "by_opponent": {k: rollup(v) for k, v in sorted(by_opponent.items())},
            "by_stage": {
                k: rollup(by_stage[k])
                for k in sorted(by_stage, key=_stage_sort_key)
            },
        })
    return {"grain": "condition", "conditions": conditions}


def _query_search(con: sqlite3.Connection) -> dict:
    """Search firing, cost and budget health **per condition × street** (§8).

    Replaces the old pair of sections (a regime "usage mix" plus a pooled cost
    block).  Three deliberate changes:

    - **Fire rate is a within-street rate** (``searched ÷ decisions`` at that
      street), not a regime's share of the pooled search count.  Regimes own
      disjoint streets (subgame §6.5), so the old share only restated how often each
      street arose while inviting a head-to-head reading the doc explicitly warns
      against.
    - **Cost is per street.** The iteration budget is a per-street constant
      (``search/budget.py``) and wall differs by an order of magnitude between
      pre-flop and flop, so a mean over all searches describes a mixture, not the
      solver.  ``regimes`` on each cell names which solver actually ran there.
    - **The roll-up sums only what is summable:** total search seconds ÷ hands, the
      wall-clock cost per hand an experiment budget is built from.

    ``routing_violations`` keeps the §8 correctness check: the vector regime must
    fire only heads-up (``num_live == 2``) on flop/turn/river.  ``num_live IS NULL``
    rows are not counted as violations (older rows may not carry it).
    """
    cells = _rows(
        con,
        """
        SELECT g.condition           AS condition,
               d.betting_stage       AS stage,
               COUNT(*)              AS n_decisions,
               SUM(d.searched)       AS n_searched,
               AVG(CASE WHEN d.searched = 1 THEN d.wall_seconds END)    AS mean_wall,
               SUM(CASE WHEN d.searched = 1 THEN d.wall_seconds END)    AS total_wall,
               AVG(CASE WHEN d.searched = 1 THEN d.iterations END)      AS mean_iters,
               AVG(CASE WHEN d.searched = 1 THEN d.iters_per_sec END)   AS mean_ips,
               AVG(CASE WHEN d.searched = 1 AND d.stop_reason = 'wall_cap' THEN 1.0
                        WHEN d.searched = 1 THEN 0.0 END)               AS wallcap_rate,
               SUM(CASE WHEN d.searched = 1 THEN d.cache_hits END)      AS hits,
               SUM(CASE WHEN d.searched = 1 THEN d.cache_misses END)    AS misses
        FROM decisions d JOIN games g ON g.game_id = d.game_id
        GROUP BY g.condition, d.betting_stage
        """,
    )
    # Which solver ran at each (condition, street) — the routing view that replaces
    # the pooled regime share.
    regimes: Dict[Tuple[str, str], Dict[str, int]] = {}
    for r in _rows(
        con,
        "SELECT g.condition AS condition, d.betting_stage AS stage, "
        "d.regime AS regime, COUNT(*) AS n "
        "FROM decisions d JOIN games g ON g.game_id = d.game_id "
        "WHERE d.searched = 1 GROUP BY g.condition, d.betting_stage, d.regime",
    ):
        key = (_condition_label(r["condition"]), r["stage"])
        regimes.setdefault(key, {})[r["regime"]] = int(r["n"])

    # p95 needs the raw column (no SQLite percentile); pulled once, bucketed here.
    walls: Dict[Tuple[str, str], List[float]] = {}
    for r in _rows(
        con,
        "SELECT g.condition AS condition, d.betting_stage AS stage, "
        "d.wall_seconds AS w FROM decisions d JOIN games g ON g.game_id = d.game_id "
        "WHERE d.searched = 1 AND d.wall_seconds IS NOT NULL",
    ):
        walls.setdefault((_condition_label(r["condition"]), r["stage"]), []).append(r["w"])

    hands = {
        _condition_label(r["condition"]): int(r["n"])
        for r in _rows(con, "SELECT condition, COUNT(*) AS n FROM games GROUP BY condition")
    }
    violations = {
        _condition_label(r["condition"]): int(r["n"])
        for r in _rows(
            con,
            "SELECT g.condition AS condition, COUNT(*) AS n "
            "FROM decisions d JOIN games g ON g.game_id = d.game_id "
            "WHERE d.searched = 1 AND d.regime = 'vector' "
            "AND (d.betting_stage NOT IN ('flop', 'turn', 'river') "
            "     OR (d.num_live IS NOT NULL AND d.num_live != 2)) "
            "GROUP BY g.condition",
        )
    }

    by_cond: Dict[str, List[dict]] = {}
    for c in cells:
        by_cond.setdefault(_condition_label(c["condition"]), []).append(c)

    conditions = []
    for cond in sorted(by_cond, key=_condition_sort_key):
        raw = sorted(by_cond[cond], key=lambda c: _stage_sort_key(c["stage"]))
        streets = []
        for c in raw:
            n_dec = int(c["n_decisions"] or 0)
            n_srch = int(c["n_searched"] or 0)
            hit, miss = c["hits"] or 0, c["misses"] or 0
            streets.append({
                "stage": c["stage"],
                "regimes": regimes.get((cond, c["stage"]), {}),
                "n_decisions": n_dec,
                "n_searched": n_srch,
                "fire_rate": (n_srch / n_dec) if n_dec else None,
                "mean_wall": c["mean_wall"],
                "p95_wall": _percentile(walls.get((cond, c["stage"]), []), 95.0),
                "wallcap_rate": c["wallcap_rate"],
                "mean_iters": c["mean_iters"],
                "mean_iters_per_sec": c["mean_ips"],
                "cache_hit_rate": (hit / (hit + miss)) if (hit + miss) else None,
            })
        n_dec = sum(s["n_decisions"] for s in streets)
        n_srch = sum(s["n_searched"] for s in streets)
        total_wall = sum(c["total_wall"] or 0.0 for c in raw)
        n_hands = hands.get(cond, 0)
        v = violations.get(cond, 0)
        conditions.append({
            "condition": cond,
            "n_hands": n_hands,
            "n_decisions": n_dec,
            "n_searched": n_srch,
            "fire_rate": (n_srch / n_dec) if n_dec else None,
            "decisions_per_hand": (n_dec / n_hands) if n_hands else None,
            # The one legitimate sum: what a hand costs in wall-clock.
            "search_wall_per_hand": (total_wall / n_hands) if n_hands else None,
            "total_search_wall": total_wall,
            "routing_violations": v,
            "routing_ok": v == 0,
            "streets": streets,
        })
    return {"grain": "condition x betting_stage", "conditions": conditions}


def _query_hu_coverage(con: sqlite3.Connection) -> dict:
    """HU coverage per condition: where OX-Search (HU) could have fired.

    ``games.hu_from_street`` is the earliest street at which a betting round began
    heads-up with the hero.  ``eligible`` (street ≥ 2, turn or later) is the
    OX-Search-HU activation predicate (design doc §4.2b).  Reported as **fractions
    of that condition's own hands** — the old raw street counts were summed over
    every arm, so a 400-deal experiment printed street counts near 1200 and no
    denominator to read them against.  Older snapshots (schema v1) lack the column →
    ``available: False``.
    """
    have = {r[1] for r in con.execute("PRAGMA table_info(games)")}
    if "hu_from_street" not in have:
        return {"available": False}

    per_cond: Dict[str, Dict[str, object]] = {}
    for r in _rows(
        con,
        "SELECT condition, hu_from_street, COUNT(*) AS n FROM games "
        "GROUP BY condition, hu_from_street",
    ):
        cond = _condition_label(r["condition"])
        entry = per_cond.setdefault(cond, {"n_hands": 0, "by_street": {}})
        entry["n_hands"] += int(r["n"])
        if r["hu_from_street"] is not None:
            entry["by_street"][int(r["hu_from_street"])] = int(r["n"])

    conditions = []
    for cond in sorted(per_cond, key=_condition_sort_key):
        e = per_cond[cond]
        n_hands = int(e["n_hands"])
        by_street = e["by_street"]
        n_hu = sum(by_street.values())
        n_eligible = sum(n for s, n in by_street.items() if s >= 2)
        conditions.append({
            "condition": cond,
            "n_hands": n_hands,
            "hu_frac": (n_hu / n_hands) if n_hands else None,
            "eligible_frac": (n_eligible / n_hands) if n_hands else None,
            "street_frac": {
                s: (n / n_hands) if n_hands else None
                for s, n in sorted(by_street.items())
            },
            "by_street": dict(sorted(by_street.items())),
        })
    return {"available": True, "grain": "condition", "conditions": conditions}


# --------------------------------------------------------------------------- #
# Flags (§8 "Automated flags") — the only threshold-gated output
# --------------------------------------------------------------------------- #


def _evaluate_flags(report: dict) -> List[dict]:
    """Turn the §8 thresholds into explicit warn/info lines (numbers print anyway).

    Every flag names the **arm** it fired for; a threshold crossed in one arm says
    nothing about another, and the old un-attributed messages were unreadable as
    soon as a snapshot held more than one.
    """
    flags: List[dict] = []
    T = THRESHOLDS

    def add(level, key, message):
        flags.append({"level": level, "key": key, "message": message})

    # losing — an arm whose strength CI lies entirely below 0 (genuine loss).
    for a in report["strength"]["arms"]:
        if a["verdict"] == "losing":
            add("warn", "losing",
                f"losing: {a['condition']} on table={a['table']} "
                f"{a['mean_bb100']:+.1f} ± {a['ci95']:.1f} bb/100 "
                f"({a['n_hands']} hands)")

    # pairing — the CRN comparison is the multi-arm headline, so a comparison that
    # matched few or no deals is a *louder* problem than any number it prints.
    pr = report.get("paired") or {}
    if pr.get("available"):
        for cmp in pr["comparisons"]:
            rate = cmp["pairing_rate"]
            if cmp["n_paired"] == 0:
                add("warn", "unpaired",
                    f"{cmp['treatment']} vs {cmp['baseline']}: 0 deals matched — "
                    "arms must share run_seed/table_policy/table shape to pair")
            elif rate is not None and rate < T["pairing_thin"]:
                add("warn", "thin_pairing",
                    f"{cmp['treatment']} vs {cmp['baseline']}: only {rate:.0%} of "
                    f"deals matched ({cmp['n_paired']} pairs) — arms cover "
                    "different hand ranges")
    elif report["meta"]["multi_arm"]:
        add("warn", "unpaired",
            f"{len(report['meta']['conditions'])} conditions present but no paired "
            "comparison was produced — check games.condition / deck_seed")

    # unbalanced arms — CRN assumes every arm covered the same hand_index range.
    imb = report["meta"]["arm_imbalance"]
    if imb is not None and imb > T["arm_imbalance"] and report["meta"]["n_arms"] > 1:
        add("info", "unbalanced_arms",
            f"arm hand counts differ by {imb:.0%} — an arm ran short "
            "(resume cursor / early failure)")

    # range net-harmful — tracking hurts (mean net info gain < 0) at some stage.
    for c in report["range_quality"]["conditions"]:
        for stage, r in c["by_stage"].items():
            g = r["net_info_gain"]
            if g is not None and g < 0:
                add("warn", "range_net_harmful",
                    f"range tracking net-harmful [{c['condition']}] on {stage}: "
                    f"{g:+.2f} nats")

    # high collapse — belief routinely rules out reality (replay / card-removal bug).
    for c in report["range_quality"]["conditions"]:
        cr = c["overall"]["collapse_rate"]
        if cr is not None and cr > T["collapse_high"]:
            add("warn", "high_collapse",
                f"collapsed truth {cr:.1%} of resolved [{c['condition']}] — "
                "belief rules out reality")

    # thin resolution — range metrics rest on very few showdowns (weak evidence).
    for c in report["range_quality"]["conditions"]:
        rf = c["overall"]["resolved_frac"]
        if rf is not None and rf < T["resolved_thin"]:
            add("info", "thin_resolution",
                f"resolved fraction {rf:.1%} [{c['condition']}] — range aggregates "
                "are weak evidence")

    for c in report["search"]["conditions"]:
        cond = c["condition"]
        # budget-bound — per street, since the budget itself is per street.  Named
        # streets, not one pooled rate that no single solver ever experienced.
        hot = [
            s for s in c["streets"]
            if s["n_searched"] and s["wallcap_rate"] is not None
            and s["wallcap_rate"] > T["wallcap_budget_bound"]
        ]
        if hot:
            detail = ", ".join(f"{s['stage']} {s['wallcap_rate']:.0%}" for s in hot)
            add("warn", "budget_bound",
                f"search is budget-bound [{cond}]: wall-cap stops on {detail}")

        # search silent — search essentially never fired where it should have.
        # blueprint_only is *supposed* to be silent, so it is exempt.
        if str(cond).strip().lower() not in _NO_SEARCH_CONDITIONS:
            fr = c["fire_rate"]
            if fr is not None and fr < T["fire_rate_silent"]:
                add("warn", "search_silent",
                    f"search fired {fr:.1%} of decisions [{cond}] — likely a "
                    "trigger/config error")

        # routing — a solver fired outside its intended envelope (correctness bug).
        if not c["routing_ok"]:
            add("warn", "routing",
                f"{c['routing_violations']} vector searches [{cond}] fired outside "
                "heads-up flop/turn/river")
    return flags


# --------------------------------------------------------------------------- #
# Human-readable block (§8 layout)
# --------------------------------------------------------------------------- #


def _fmt(v, spec: str = ".2f", none: str = "n/a") -> str:
    return none if v is None else format(v, spec)


def _secs(v) -> str:
    """Seconds with a unit suffix; ``n/a`` (not ``n/as``) when the value is NULL."""
    return "n/a" if v is None else f"{v:.1f}s"


_VERDICT_MARK = {
    "winning": "✓", "losing": "✗",          # absolute strength (vs zero)
    "better": "✓", "worse": "✗",            # paired difference (vs the baseline arm)
    "inconclusive": "~", "n/a": " ",
}


def _print_human(report: dict) -> str:
    """Render the §8 summary block; returns the string (also logged by the caller)."""
    m = report["meta"]
    L: List[str] = []
    run_id = m["run_id"] if isinstance(m["run_id"], str) else ",".join(m["run_id"] or [])
    L.append(f"experiment  run_id={run_id}   git={m['git_sha']}   {m['n_players']}-max")
    # Deals × conditions, never a single pooled "hands" number: under CRN the games
    # row count is the deal count times the number of arms replaying those deals, so
    # printing the row count as "hands" silently multiplies the experiment size.
    n_cond, n_tab = len(m["conditions"]), len(m["tables"])
    if m["n_arms"] <= 1:
        L.append(f"  {m['n_deals']} deals   1 arm")
    else:
        L.append(
            f"  {m['n_deals']} deals × {n_cond} conditions = {m['n_rows']} game rows"
            f"   ({m['n_arms']} arms = {n_cond} conditions × {n_tab} tables)"
        )
    if m["conditions"]:
        L.append(f"  conditions: {', '.join(map(str, m['conditions']))}")
    if m["tables"]:
        L.append(f"  tables:     {', '.join(map(str, m['tables']))}")
    L.append("─" * 78)

    _strength_block(L, report)
    _paired_block(L, report)
    _range_block(L, report)
    _search_block(L, report)
    _hu_block(L, report)

    if report["flags"]:
        L.append("")
        L.append("FLAGS")
        for f in report["flags"]:
            L.append(f"  {'⚠' if f['level'] == 'warn' else 'ⓘ'} {f['message']}")
    return "\n".join(L)


def _strength_block(L: List[str], report: dict) -> None:
    st = report["strength"]
    metric = "aivat bb/100" if st["used_aivat"] else "raw bb/100"
    L.append(f"STRENGTH — per arm ({metric} ± 95% CI).  Arms are never pooled.")
    if not st["arms"]:
        L.append("  (no hands logged)")
        return
    width = max(len(str(a["condition"])) for a in st["arms"])
    table = None
    for a in st["arms"]:
        if a["table"] != table:
            table = a["table"]
            L.append(f"  table={table}")
        L.append(
            f"    {str(a['condition']):<{width}}  {_fmt(a['mean_bb100'], '+8.1f')} ± "
            f"{_fmt(a['ci95'], '.1f'):<6} {a['n_hands']:>6} hands   "
            f"{_VERDICT_MARK[a['verdict']]} {a['verdict']}"
        )
        if len(a["by_position"]) > 1:
            pos = "  ".join(
                f"{p} {_fmt(s['mean'], '+.0f')}" for p, s in a["by_position"].items()
            )
            L.append(f"      by position  {pos}")


def _paired_block(L: List[str], report: dict) -> None:
    pr = report.get("paired")
    if not pr or not pr.get("available"):
        return
    metric = "aivat" if pr["used_aivat"] else "raw"
    L.append("")
    L.append(
        f"PAIRED Δ vs {pr['baseline']} — CRN, matched on {pr['match_key']} "
        f"({metric} bb/100, 95% bootstrap CI)"
    )
    for cmp in pr["comparisons"]:
        L.append(
            f"  {cmp['treatment']}   "
            f"({cmp['n_paired']} of {min(cmp['n_baseline_hands'], cmp['n_treatment_hands'])} "
            f"deals matched)"
        )
        for cell in cmp["cells"]:
            label = cell["table"] if cell["table"] is not None else "ALL TABLES pooled"
            b = cell["ci95_bootstrap"]
            L.append(
                f"    {str(label):<20} {_fmt(cell['mean_delta_bb100'], '+8.2f')}  "
                f"[{_fmt(b['lo'], '+.2f')}, {_fmt(b['hi'], '+.2f')}]  "
                f"{cell['n_paired']:>6} pairs   {_VERDICT_MARK[cell['verdict']]} "
                f"{cell['verdict']}"
            )
            cov = cell.get("covered")
            if cov and cov["n_paired"]:
                cb = cov["ci95_bootstrap"]
                L.append(
                    f"      {'└ where it fired':<18} "
                    f"{_fmt(cov['mean_delta_bb100'], '+8.2f')}  "
                    f"[{_fmt(cb['lo'], '+.2f')}, {_fmt(cb['hi'], '+.2f')}]  "
                    f"{cov['n_paired']:>6} pairs   {_VERDICT_MARK[cov['verdict']]} "
                    f"{cov['verdict']}"
                )
            L.append(
                f"      matched-sample means: {pr['baseline']} "
                f"{_fmt(cell['baseline_mean_bb100'], '+.1f')} → {cmp['treatment']} "
                f"{_fmt(cell['treatment_mean_bb100'], '+.1f')}"
            )


def _range_block(L: List[str], report: dict) -> None:
    conds = report["range_quality"]["conditions"]
    L.append("")
    L.append("RANGE TRACKING — per condition (net info gain vs the uniform prior, nats)")
    if not conds:
        # An explicit "none" beats a vanished section: a run that logged hands but no
        # belief snapshots is an anomaly, not an empty report.
        L.append("  (no range-quality snapshots logged)")
        return
    for c in conds:
        ov = c["overall"]
        L.append(
            f"  {c['condition']}   net gain {_fmt(ov['net_info_gain'], '+.2f')} nats   "
            f"resolved {_fmt(ov['resolved_frac'], '.1%')} of {ov['snapshots']} "
            f"seat-snapshots   collapsed {_fmt(ov['collapse_rate'], '.1%')}   "
            f"fallback {_fmt(ov['fallback_rate'], '.1%')}"
        )
        if len(c["by_opponent"]) > 1:
            opp = "   ".join(
                f"{k} {_fmt(v['net_info_gain'], '+.2f')}"
                for k, v in c["by_opponent"].items()
            )
            L.append(f"      by opponent  {opp}")
        if c["by_stage"]:
            stg = "   ".join(
                f"{k} {_fmt(v['net_info_gain'], '+.2f')}" for k, v in c["by_stage"].items()
            )
            L.append(f"      by street    {stg}")


def _search_block(L: List[str], report: dict) -> None:
    conds = report["search"]["conditions"]
    L.append("")
    L.append("SEARCH — per condition × street (fire rate is within the street)")
    if not conds:
        L.append("  (no hero decisions logged)")
        return
    for c in conds:
        L.append(
            f"  {c['condition']}   fired {_fmt(c['fire_rate'], '.1%')} of "
            f"{c['n_decisions']} hero decisions "
            f"({_fmt(c['decisions_per_hand'], '.1f')}/hand)   "
            f"{_secs(c['search_wall_per_hand'])} search per hand   "
            f"routing {'OK ✓' if c['routing_ok'] else 'VIOLATIONS ✗'}"
        )
        if not c["streets"]:
            continue
        L.append(
            f"      {'street':<8}{'solver':<10}{'fired':>7}{'n':>8}"
            f"{'wall':>8}{'p95':>8}{'wall-cap':>10}{'iters':>8}{'it/s':>8}"
        )
        for s in c["streets"]:
            solver = "·".join(sorted(s["regimes"])) if s["regimes"] else "—"
            L.append(
                f"      {str(s['stage']):<8}{solver:<10}"
                f"{_fmt(s['fire_rate'], '.0%'):>7}{s['n_decisions']:>8}"
                f"{_secs(s['mean_wall']):>8}"
                f"{_secs(s['p95_wall']):>8}"
                f"{_fmt(s['wallcap_rate'], '.0%'):>10}"
                f"{_fmt(s['mean_iters'], '.0f'):>8}"
                f"{_fmt(s['mean_iters_per_sec'], '.0f'):>8}"
            )


def _hu_block(L: List[str], report: dict) -> None:
    hc = report["hu_coverage"]
    if not hc.get("available") or not hc.get("conditions"):
        return
    L.append("")
    L.append("HU COVERAGE — heads-up with hero (OX-Search-HU fires from the turn on)")
    for c in hc["conditions"]:
        streets = "  ".join(
            f"{_STREET_NAME.get(s, s)} {_fmt(f, '.0%')}"
            for s, f in c["street_frac"].items()
        )
        L.append(
            f"  {c['condition']}   HU at some point {_fmt(c['hu_frac'], '.1%')}   "
            f"eligible (turn+) {_fmt(c['eligible_frac'], '.1%')}   "
            f"of {c['n_hands']} hands"
        )
        if streets:
            L.append(f"      first HU street  {streets}")


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
        "search": _query_search(con),
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
