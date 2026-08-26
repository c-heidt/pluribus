"""Tests for the end-of-run summary (evaluation/summarize.py, doc §8).

Two styles:

- **Hand-built DB** — rows inserted directly through :class:`ExperimentLog` with
  known values, so the strength CI, bb/100, net-info-gain roll-ups, routing check,
  and every flag are asserted against arithmetic we control.
- **Pure-function** unit tests for the stats helpers (``_mean_ci`` / ``_percentile``
  / ``_position_name`` / ``_verdict``) and edge cases (empty DB must not crash).

The class :class:`TestArmGrain` is the regression suite for the summary's central
rule: **every number is computed within one arm** (``condition`` × ``table_label``),
and the only cross-arm number is the CRN paired difference.
"""

import json
import math

import pytest

from evaluation.sqlite_logging import (
    DecisionRow,
    ExperimentLog,
    GameRow,
    RangeQualityRow,
    SeatRow,
)
from evaluation.summarize import (
    _bootstrap_ci,
    _mean_ci,
    _percentile,
    _position_name,
    _print_human,
    _query_paired,
    _verdict,
    build_report,
    summarize,
)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def _game(gid_hand, **kw):
    base = dict(
        run_id="R",
        hand_index=gid_hand,
        config_fingerprint="fp",
        table_label="all_blueprint",
        table_config="{}",
        hero_seat=0,
        button_seat=0,
        n_players=6,
        big_blind=100.0,
        starting_stack=10000.0,
        deck_seed=1,
        git_sha="abc1234",
    )
    base.update(kw)
    return GameRow(**base)


@pytest.fixture
def db(tmp_path):
    log = ExperimentLog.open(tmp_path / "run.sqlite")
    yield log, tmp_path
    log.close()


def _arm(report, condition=None, table=None):
    """The single strength arm matching ``condition``/``table`` (None == any)."""
    hits = [
        a for a in report["strength"]["arms"]
        if (condition is None or a["condition"] == condition)
        and (table is None or a["table"] == table)
    ]
    assert len(hits) == 1, f"expected exactly one arm, got {hits}"
    return hits[0]


def _search(report, condition="(unlabelled)"):
    """The search block for one condition."""
    (hit,) = [c for c in report["search"]["conditions"] if c["condition"] == condition]
    return hit


def _street(search_cond, stage):
    (hit,) = [s for s in search_cond["streets"] if s["stage"] == stage]
    return hit


def _range(report, condition="(unlabelled)"):
    (hit,) = [
        c for c in report["range_quality"]["conditions"] if c["condition"] == condition
    ]
    return hit


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

class TestHelpers:

    def test_mean_ci_basic(self):
        r = _mean_ci([1.0, 1.0, 1.0])
        assert r["n"] == 3 and r["mean"] == 1.0 and r["ci95"] == 0.0

    def test_mean_ci_matches_population_formula(self):
        vals = [0.0, 10.0]
        r = _mean_ci(vals)
        # population std = 5, se = 5/sqrt(2), ci = 1.96*se
        assert r["mean"] == 5.0
        assert math.isclose(r["ci95"], 1.96 * 5.0 / math.sqrt(2), rel_tol=1e-9)

    def test_mean_ci_empty(self):
        r = _mean_ci([])
        assert r == {"n": 0, "mean": None, "ci95": None}

    def test_percentile_interpolates(self):
        xs = list(range(1, 101))          # 1..100
        assert math.isclose(_percentile(xs, 95.0), 95.05, rel_tol=1e-6)
        assert _percentile([], 95.0) is None
        assert _percentile([7.0], 95.0) == 7.0

    def test_position_name(self):
        # offset 0 = BTN, then SB, BB, ... for 6-max.
        assert _position_name(0, 0, 6) == "BTN"
        assert _position_name(1, 0, 6) == "SB"
        assert _position_name(2, 0, 6) == "BB"
        assert _position_name(5, 0, 6) == "CO"
        assert _position_name(0, 2, 6) == "MP"     # offset (0-2)%6 = 4 → MP
        # 4-max, the shipped multiplayer table size.  PokerEnv seats SB at 0, BB at
        # 1 and the button last (`poker_env` `is_dealer` on players[-1]), so with
        # button=3: seat 3 BTN, 0 SB, 1 BB, and seat 2 — first to act pre-flop under
        # the `[2:]+[:2]` order — is UTG.
        assert [_position_name(s, 3, 4) for s in (3, 0, 1, 2)] == \
            ["BTN", "SB", "BB", "UTG"]
        assert [_position_name(s, 4, 5) for s in (4, 0, 1, 2, 3)] == \
            ["BTN", "SB", "BB", "UTG", "CO"]
        # unmapped table size still falls back to POSk (offset from button).
        assert _position_name(3, 0, 7) == "POS3"

    def test_verdict_treats_straddling_ci_as_inconclusive(self):
        # §8: a CI straddling zero is *inconclusive*, not *bad*.
        assert _verdict(10.0, 3.0) == "winning"
        assert _verdict(-10.0, 3.0) == "losing"
        assert _verdict(10.0, 30.0) == "inconclusive"
        assert _verdict(None, None) == "n/a"

    def test_bootstrap_ci_is_deterministic_and_brackets_mean(self):
        vals = [1.0, 2.0, 3.0, 4.0, 100.0]        # heavy tail → bootstrap over normal
        a = _bootstrap_ci(vals, n_resamples=500)
        b = _bootstrap_ci(vals, n_resamples=500)
        assert a == b                              # fixed-seed → reproducible
        assert a["lo"] <= sum(vals) / len(vals) <= a["hi"]
        assert _bootstrap_ci([1.0])["lo"] is None  # < 2 points → undefined


# --------------------------------------------------------------------------- #
# The arm-grain rule — the regressions this summary exists to prevent
# --------------------------------------------------------------------------- #

class TestArmGrain:
    """Nothing is pooled across arms, and counts are reported at the right grain."""

    def _two_arms_two_tables(self, log):
        """2 conditions × 2 tables × 2 deals = 8 game rows over only 4 deals."""
        with log.game():
            h = 0
            for ds in (1, 2):
                for table, base in (("all_blueprint", 0.0), ("random", -400.0)):
                    for cond, bump in (("vanilla", 0.0), ("DBR", 100.0)):
                        log.log_game(_game(
                            h, condition=cond, table_label=table, deck_seed=ds,
                            hero_chips_delta=base + bump,
                        ))
                        h += 1

    def test_meta_counts_deals_not_game_rows(self, db):
        log, _ = db
        self._two_arms_two_tables(log)
        m = build_report(log._con)["meta"]
        # The old header called the games row count "hands", so this experiment
        # would have announced 8 hands — twice the 4 deals actually played.
        assert m["n_rows"] == 8
        assert m["n_deals"] == 4                   # distinct (table_label, deck_seed)
        assert m["n_arms"] == 4                    # 2 conditions × 2 tables
        assert m["conditions"] == ["vanilla", "DBR"]      # baseline sorts first
        assert m["tables"] == ["all_blueprint", "random"]
        assert m["multi_arm"] is True

    def test_strength_is_per_arm_and_never_pooled(self, db):
        log, _ = db
        self._two_arms_two_tables(log)
        st = build_report(log._con)["strength"]
        # One row per (condition, table) — and no cross-arm aggregate at all.
        assert {(a["condition"], a["table"]) for a in st["arms"]} == {
            ("vanilla", "all_blueprint"), ("DBR", "all_blueprint"),
            ("vanilla", "random"), ("DBR", "random"),
        }
        assert "overall" not in st and "tables" not in st
        # Each arm keeps its own hands; nothing is summed across conditions.
        for a in st["arms"]:
            assert a["n_hands"] == 2
        rep = {"strength": st}
        assert math.isclose(_arm(rep, "vanilla", "all_blueprint")["mean_bb100"], 0.0)
        assert math.isclose(_arm(rep, "DBR", "all_blueprint")["mean_bb100"], 100.0)
        assert math.isclose(_arm(rep, "vanilla", "random")["mean_bb100"], -400.0)
        # The old pooled "overall" would have averaged these four into -150 bb/100,
        # a number no arm ever played for.

    def test_search_and_range_and_hu_are_per_condition(self, db):
        log, _ = db
        with log.game():
            for i, cond in enumerate(("vanilla", "DBR")):
                gid = log.log_game(_game(i, condition=cond, deck_seed=i,
                                         hu_from_street=2))
                log.log_seats(gid, [SeatRow(seat=1, is_hero=0, agent_label="bp")])
                log.log_decision(gid, DecisionRow(
                    betting_stage="flop", regime="mccfr", searched=1,
                    wall_seconds=1.0, iterations=10, stop_reason="iteration_cap",
                ))
                log.log_range_quality(gid, RangeQualityRow(
                    seat=1, betting_stage="flop", resolved=1, net_info_gain=0.5,
                    collapsed_truth=0, uniform_fallback=0,
                ))
        rep = build_report(log._con)
        for section in ("search", "range_quality"):
            assert [c["condition"] for c in rep[section]["conditions"]] == \
                ["vanilla", "DBR"]
        assert [c["condition"] for c in rep["hu_coverage"]["conditions"]] == \
            ["vanilla", "DBR"]
        # Per-condition counts, not the 2× pooled totals the old block printed.
        assert _search(rep, "vanilla")["n_decisions"] == 1
        assert _range(rep, "DBR")["overall"]["snapshots"] == 1
        assert rep["hu_coverage"]["conditions"][0]["n_hands"] == 1


# --------------------------------------------------------------------------- #
# Cross-condition paired difference (CRN, §10.1)
# --------------------------------------------------------------------------- #

class TestPairedDifference:

    def test_absent_with_fewer_than_two_conditions(self, db):
        log, _ = db
        with log.game():
            log.log_game(_game(0, condition="vanilla", deck_seed=1))
        pr = _query_paired(log._con)
        assert pr["available"] is False and pr["n_conditions"] == 1

    def test_pairs_on_deck_seed_and_differences(self, db):
        log, _ = db
        # Same three deals under vanilla and DBR; bb=100 so bb/100 == chips value.
        b0 = {100: 0.0, 101: 100.0, 102: 200.0}
        a = {100: 100.0, 101: 100.0, 102: 500.0}
        with log.game():
            for i, (ds, v) in enumerate(b0.items()):
                log.log_game(_game(i, condition="vanilla", deck_seed=ds, hero_chips_delta=v))
            for i, (ds, v) in enumerate(a.items()):
                log.log_game(_game(10 + i, condition="DBR", deck_seed=ds,
                                   hero_chips_delta=v))
            # A deal only A saw — must be excluded from the paired join.
            log.log_game(_game(99, condition="DBR", deck_seed=777, hero_chips_delta=9.0))
        pr = _query_paired(log._con)
        assert pr["available"] and pr["baseline"] == "vanilla"
        (cmp,) = pr["comparisons"]
        assert cmp["treatment"] == "DBR" and cmp["n_paired"] == 3   # 777 excluded
        (cell,) = cmp["cells"]                       # one table → no pooled row
        assert cell["table"] == "all_blueprint"
        # Δ = [100, 0, 300] → mean 133.33; deck-matched, not (mean(DBR)-mean(vanilla)).
        assert math.isclose(cell["mean_delta_bb100"], (100 + 0 + 300) / 3, rel_tol=1e-9)
        # Matched-sample means are carried so the Δ can be read against its levels,
        # and are exactly consistent with it.
        assert math.isclose(
            cell["treatment_mean_bb100"] - cell["baseline_mean_bb100"],
            cell["mean_delta_bb100"], rel_tol=1e-9,
        )

    def test_pairs_within_table_when_deck_seeds_repeat(self, db):
        """Two table policies share every ``deck_seed`` — must still pair.

        ``deck_seed`` is a pure function of ``(run_seed, hand_index)`` and carries no
        table component, so a snapshot holding two table policies repeats every seed.
        Matched on the seed alone, every deal looked like a within-arm duplicate, was
        dropped as ambiguous, and the whole comparison silently reported zero pairs.
        """
        log, _ = db
        with log.game():
            h = 0
            for ds in (1, 2):
                for table in ("all_blueprint", "random"):
                    log.log_game(_game(h, condition="vanilla", table_label=table,
                                       deck_seed=ds, hero_chips_delta=0.0))
                    log.log_game(_game(h + 1, condition="DBR", table_label=table,
                                       deck_seed=ds, hero_chips_delta=100.0))
                    h += 2
        pr = _query_paired(log._con)
        assert pr["dropped_ambiguous_deals"] == {}       # no false duplicates
        (cmp,) = pr["comparisons"]
        assert cmp["n_paired"] == 4                      # was 0 before the fix
        assert cmp["pairing_rate"] == 1.0
        cells = {c["table"]: c for c in cmp["cells"]}
        assert set(cells) == {"all_blueprint", "random", None}
        assert cells["all_blueprint"]["n_paired"] == 2
        assert cells[None]["n_paired"] == 4              # pooled row, clearly labelled
        assert math.isclose(cells[None]["mean_delta_bb100"], 100.0)

    def test_genuine_duplicate_deal_is_still_dropped(self, db):
        log, _ = db
        # The same arm replaying the same (table, deck_seed) is ambiguous — drop it.
        with log.game():
            log.log_game(_game(0, condition="vanilla", deck_seed=1, hero_chips_delta=0.0))
            log.log_game(_game(1, condition="vanilla", deck_seed=1, hero_chips_delta=5.0))
            log.log_game(_game(2, condition="DBR", deck_seed=1, hero_chips_delta=100.0))
        pr = _query_paired(log._con)
        assert pr["dropped_ambiguous_deals"] == {"vanilla": 1}
        assert pr["comparisons"][0]["n_paired"] == 0

    def test_prefers_aivat_and_bootstrap_present(self, db):
        log, _ = db
        with log.game():
            log.log_game(_game(0, condition="vanilla", deck_seed=1,
                               hero_chips_delta=999.0, aivat_value=0.0))
            log.log_game(_game(1, condition="vanilla", deck_seed=2,
                               hero_chips_delta=999.0, aivat_value=50.0))
            log.log_game(_game(2, condition="DBR", deck_seed=1,
                               hero_chips_delta=999.0, aivat_value=80.0))
            log.log_game(_game(3, condition="DBR", deck_seed=2,
                               hero_chips_delta=999.0, aivat_value=60.0))
        pr = _query_paired(log._con)
        assert pr["used_aivat"] is True            # differences use aivat, not raw
        (cell,) = pr["comparisons"][0]["cells"]
        # Δ = aivat: [80-0, 60-50] = [80, 10] → mean 45.
        assert math.isclose(cell["mean_delta_bb100"], 45.0, rel_tol=1e-9)
        assert cell["ci95_bootstrap"]["n_resamples"] > 0

    def test_coverage_restricted_delta_uses_modeled_deals_only(self, db):
        log, _ = db
        with log.game():
            for ds in (1, 2):
                log.log_game(_game(ds, condition="vanilla", deck_seed=ds,
                                   hero_chips_delta=0.0))
            for ds, v in ((1, 100.0), (2, 500.0)):
                gid = log.log_game(_game(10 + ds, condition="DBR", deck_seed=ds,
                                         hero_chips_delta=v))
                # Only deal 2 actually had a modeled decision.
                log.log_decision(gid, DecisionRow(
                    betting_stage="flop", regime="mccfr", searched=1,
                    modeled_decision=1 if ds == 2 else 0,
                ))
        (cell,) = _query_paired(log._con)["comparisons"][0]["cells"]
        assert math.isclose(cell["mean_delta_bb100"], 300.0)      # both deals
        assert cell["covered"]["n_paired"] == 1
        assert math.isclose(cell["covered"]["mean_delta_bb100"], 500.0)

    def test_unpaired_arms_are_flagged(self, db):
        log, _ = db
        # Arms that never met on a deal: the Δ section is the multi-arm headline, so
        # a comparison with nothing to compare must announce itself.
        with log.game():
            log.log_game(_game(0, condition="vanilla", deck_seed=1, hero_chips_delta=0.0))
            log.log_game(_game(1, condition="DBR", deck_seed=999, hero_chips_delta=0.0))
        rep = build_report(log._con)
        assert rep["paired"]["comparisons"][0]["n_paired"] == 0
        assert "unpaired" in {f["key"] for f in rep["flags"]}


# --------------------------------------------------------------------------- #
# Strength
# --------------------------------------------------------------------------- #

class TestStrength:

    def test_bb100_and_ci_and_position(self, db):
        log, _ = db
        # Two hands: +200 and -100 chips at bb=100 → bb/100 = +200 and -100.
        with log.game():
            log.log_game(_game(0, hero_chips_delta=200.0, hero_seat=0, button_seat=0))
        with log.game():
            log.log_game(_game(1, hero_chips_delta=-100.0, hero_seat=1, button_seat=0))
        rep = build_report(log._con)
        assert rep["strength"]["used_aivat"] is False
        a = _arm(rep)
        assert a["n_hands"] == 2
        assert math.isclose(a["mean_bb100"], 50.0)               # mean(200,-100)
        assert a["verdict"] == "inconclusive"                    # CI straddles zero
        # by position: hero_seat 0 (BTN) got +200, hero_seat 1 (SB) got -100.
        assert math.isclose(a["by_position"]["BTN"]["mean"], 200.0)
        assert math.isclose(a["by_position"]["SB"]["mean"], -100.0)

    def test_prefers_aivat_when_fully_populated(self, db):
        log, _ = db
        with log.game():
            log.log_game(_game(0, hero_chips_delta=999.0, aivat_value=100.0))
        with log.game():
            log.log_game(_game(1, hero_chips_delta=999.0, aivat_value=300.0))
        rep = build_report(log._con)
        assert rep["strength"]["used_aivat"] is True
        assert math.isclose(_arm(rep)["mean_bb100"], 200.0)       # from aivat, not 999

    def test_metric_raw_overrides_fully_populated_aivat(self, db):
        """``metric='raw'`` must win over the auto AIVAT preference.

        AIVAT is only worth using when it actually reduces variance, which is a
        property of a given run — so the choice has to be overridable rather than
        inferred from mere presence of the column.
        """
        log, _ = db
        for h, (raw, aiv) in enumerate(((999.0, 100.0), (555.0, 300.0))):
            with log.game():
                log.log_game(_game(h, condition="vanilla", deck_seed=h,
                                   hero_chips_delta=raw, aivat_value=aiv))
                log.log_game(_game(10 + h, condition="DBR", deck_seed=h,
                                   hero_chips_delta=raw + 40.0, aivat_value=aiv + 90.0))
        auto = build_report(log._con)
        raw = build_report(log._con, metric="raw")
        aivat = build_report(log._con, metric="aivat")

        assert auto["strength"]["used_aivat"] is True            # auto picks AIVAT
        assert auto["strength"]["metric_mode"] == "auto"
        assert math.isclose(_arm(auto, "vanilla")["mean_bb100"], 200.0)

        assert raw["strength"]["used_aivat"] is False
        assert raw["strength"]["metric"] == "raw_bb100"
        assert math.isclose(_arm(raw, "vanilla")["mean_bb100"], 777.0)   # (999+555)/2
        assert math.isclose(_arm(raw, "DBR")["mean_bb100"], 817.0)

        assert aivat["strength"]["used_aivat"] is True
        assert math.isclose(_arm(aivat, "vanilla")["mean_bb100"], 200.0)

        # The paired Δ must move onto the same column — the two headlines can never
        # disagree about which metric they are reporting.
        (rcell,) = raw["paired"]["comparisons"][0]["cells"]
        (acell,) = auto["paired"]["comparisons"][0]["cells"]
        assert raw["paired"]["used_aivat"] is False
        assert math.isclose(rcell["mean_delta_bb100"], 40.0)      # raw gap
        assert math.isclose(acell["mean_delta_bb100"], 90.0)      # aivat gap

    def test_metric_is_reported_and_validated(self, db):
        log, _ = db
        with log.game():
            log.log_game(_game(0, hero_chips_delta=100.0, aivat_value=50.0))
        # A forced metric is stated in the rendered block, so a saved summary can
        # never be misread as the automatic choice.
        assert "forced" in _print_human(build_report(log._con, metric="raw"))
        assert "forced" not in _print_human(build_report(log._con))
        with pytest.raises(ValueError):
            build_report(log._con, metric="bogus")

    def test_losing_flag_names_the_arm(self, db):
        log, _ = db
        # Three identical -500 bb/100 hands → CI is 0, mean+ci < 0 → losing flag.
        for h in range(3):
            with log.game():
                log.log_game(_game(h, condition="vanilla", table_label="tough",
                                   hero_chips_delta=-500.0))
        rep = build_report(log._con)
        assert _arm(rep)["verdict"] == "losing"
        (flag,) = [f for f in rep["flags"] if f["key"] == "losing"]
        # A threshold crossed in one arm says nothing about another, so the message
        # must identify which arm it fired for.
        assert "vanilla" in flag["message"] and "tough" in flag["message"]


# --------------------------------------------------------------------------- #
# Range tracking
# --------------------------------------------------------------------------- #

class TestRangeHealth:

    def _hand_with_rq(self, log, hand, rows, **game_kw):
        with log.game():
            gid = log.log_game(_game(hand, **game_kw))
            log.log_seats(gid, [
                SeatRow(seat=0, is_hero=1, agent_label="hero"),
                SeatRow(seat=1, is_hero=0, agent_label="bp"),
            ])
            for r in rows:
                log.log_range_quality(gid, r)

    def test_net_gain_and_resolved_fraction(self, db):
        log, _ = db
        # One resolved flop snapshot (gain +0.5) + one unresolved turn snapshot.
        self._hand_with_rq(log, 0, [
            RangeQualityRow(seat=1, betting_stage="flop", resolved=1,
                            net_info_gain=0.5, collapsed_truth=0, uniform_fallback=0),
            RangeQualityRow(seat=1, betting_stage="turn", resolved=0,
                            uniform_fallback=0),
        ])
        c = _range(build_report(log._con))
        assert c["overall"]["snapshots"] == 2
        assert math.isclose(c["overall"]["resolved_frac"], 0.5)
        assert math.isclose(c["overall"]["net_info_gain"], 0.5)   # resolved-only
        assert c["by_opponent"]["bp"]["snapshots"] == 2
        assert math.isclose(c["by_stage"]["flop"]["net_info_gain"], 0.5)
        assert list(c["by_stage"]) == ["flop", "turn"]            # street order

    def test_net_harmful_and_collapse_flags(self, db):
        log, _ = db
        # A resolved river snapshot with negative gain + a collapse → two flags.
        self._hand_with_rq(log, 0, [
            RangeQualityRow(seat=1, betting_stage="river", resolved=1,
                            net_info_gain=-0.3, collapsed_truth=1, uniform_fallback=0),
        ])
        keys = {f["key"] for f in build_report(log._con)["flags"]}
        assert "range_net_harmful" in keys
        assert "high_collapse" in keys        # 100% collapse > 5% threshold


# --------------------------------------------------------------------------- #
# Search: routing, within-street fire rate, per-street cost
# --------------------------------------------------------------------------- #

class TestSearch:

    def _dec(self, log, gid, **kw):
        base = dict(betting_stage="flop", regime="mccfr", searched=1)
        base.update(kw)
        log.log_decision(gid, DecisionRow(**base))

    def test_routing_violation_flagged(self, db):
        log, _ = db
        with log.game():
            gid = log.log_game(_game(0))
            # A vector search on the PREFLOP (the vector envelope is heads-up
            # flop/turn/river, §6.5 — preflop is always MCCFR) → routing violation.
            self._dec(log, gid, regime="vector",
                      betting_stage="preflop", num_live=2, stop_reason="iteration_cap",
                      wall_seconds=1.0, iterations=100, cache_hits=1, cache_misses=0)
        rep = build_report(log._con)
        c = _search(rep)
        assert c["routing_ok"] is False and c["routing_violations"] == 1
        assert "routing" in {f["key"] for f in rep["flags"]}

    def test_vector_multiway_flagged(self, db):
        log, _ = db
        with log.game():
            gid = log.log_game(_game(0))
            # Vector must be heads-up: a multiway (num_live != 2) vector search is a
            # routing violation even on an in-envelope street.
            self._dec(log, gid, regime="vector",
                      betting_stage="flop", num_live=3, stop_reason="wall_cap",
                      wall_seconds=2.0, iterations=9000, cache_hits=9, cache_misses=1)
        c = _search(build_report(log._con))
        assert c["routing_ok"] is False and c["routing_violations"] == 1

    def test_vector_headsup_flop_turn_river_is_clean(self, db):
        # §6.5: the vector regime fires heads-up on flop, turn, AND river — all clean.
        log, _ = db
        with log.game():
            gid = log.log_game(_game(0))
            for stage in ("flop", "turn", "river"):
                self._dec(log, gid, regime="vector",
                          betting_stage=stage, num_live=2, stop_reason="wall_cap",
                          wall_seconds=2.0, iterations=9000, cache_hits=9, cache_misses=1)
        c = _search(build_report(log._con))
        assert c["routing_ok"] is True and c["routing_violations"] == 0
        # The solver that actually ran is named per street — replacing the old
        # pooled "share of searches", which only restated how often each street came
        # up while inviting a head-to-head reading of two disjoint solvers.
        assert _street(c, "turn")["regimes"] == {"vector": 1}
        assert "share" not in _street(c, "turn")

    def test_fire_rate_is_within_the_street(self, db):
        log, _ = db
        with log.game():
            gid = log.log_game(_game(0))
            # Flop: 1 of 2 decisions searched.  River: 2 of 2.
            self._dec(log, gid, betting_stage="flop", wall_seconds=1.0,
                      iterations=10, stop_reason="iteration_cap")
            self._dec(log, gid, betting_stage="flop", regime="blueprint", searched=0)
            for _ in range(2):
                self._dec(log, gid, betting_stage="river", regime="vector",
                          num_live=2, wall_seconds=1.0, iterations=10,
                          stop_reason="iteration_cap")
        c = _search(build_report(log._con))
        assert math.isclose(_street(c, "flop")["fire_rate"], 0.5)
        assert math.isclose(_street(c, "river")["fire_rate"], 1.0)
        # The condition roll-up is a real rate too (3 of 4), not a sum of shares.
        assert math.isclose(c["fire_rate"], 0.75)
        assert [s["stage"] for s in c["streets"]] == ["flop", "river"]

    def test_cost_is_per_street_not_pooled(self, db):
        log, _ = db
        with log.game():
            gid = log.log_game(_game(0))
            # Flop is the expensive street; river is cheap.  A pooled mean (5.5s)
            # describes a mixture no solver ever ran at.
            self._dec(log, gid, betting_stage="flop", stop_reason="wall_cap",
                      wall_seconds=10.0, iterations=5000, iters_per_sec=500.0,
                      cache_hits=90, cache_misses=10)
            self._dec(log, gid, betting_stage="flop", stop_reason="wall_cap",
                      wall_seconds=12.0, iterations=6000, iters_per_sec=500.0,
                      cache_hits=90, cache_misses=10)
            self._dec(log, gid, betting_stage="river", regime="vector", num_live=2,
                      stop_reason="iteration_cap", wall_seconds=1.0, iterations=500,
                      iters_per_sec=500.0, cache_hits=10, cache_misses=0)
            self._dec(log, gid, regime="blueprint", searched=0)
        rep = build_report(log._con)
        c = _search(rep)
        flop, river = _street(c, "flop"), _street(c, "river")
        assert math.isclose(flop["mean_wall"], 11.0)
        assert math.isclose(flop["p95_wall"], 11.9, rel_tol=1e-6)  # interp of [10,12]
        assert math.isclose(flop["wallcap_rate"], 1.0)            # both wall_cap
        assert math.isclose(flop["cache_hit_rate"], 0.9)          # 180/200
        assert math.isclose(river["mean_wall"], 1.0)
        assert math.isclose(river["wallcap_rate"], 0.0)           # river is not bound
        # The one legitimate sum: wall-clock cost per hand (23s over 1 hand).
        assert math.isclose(c["search_wall_per_hand"], 23.0)
        assert math.isclose(c["decisions_per_hand"], 4.0)
        # …and the flag names the street that is actually budget-bound.
        (flag,) = [f for f in rep["flags"] if f["key"] == "budget_bound"]
        assert "flop" in flag["message"] and "river" not in flag["message"]

    def test_search_silent_flag_exempts_blueprint_only(self, db):
        log, _ = db
        with log.game():
            gid = log.log_game(_game(0, condition="blueprint_only"))
            self._dec(log, gid, regime="blueprint", searched=0)
        with log.game():
            gid = log.log_game(_game(1, condition="vanilla", deck_seed=2))
            self._dec(log, gid, regime="blueprint", searched=0)
        rep = build_report(log._con)
        silent = [f for f in rep["flags"] if f["key"] == "search_silent"]
        # blueprint_only is *supposed* never to search; vanilla is not.
        assert len(silent) == 1 and "vanilla" in silent[0]["message"]


# --------------------------------------------------------------------------- #
# Orchestration: summarize() end-to-end + edge cases
# --------------------------------------------------------------------------- #

class TestSummarizeEndToEnd:

    def test_writes_json_and_prints(self, db, capsys):
        log, tmp_path = db
        with log.game():
            gid = log.log_game(_game(0, hero_chips_delta=150.0))
            log.log_seats(gid, [SeatRow(seat=0, is_hero=1, agent_label="hero")])
        log.snapshot(tmp_path / "snap.sqlite")     # summarize reads a snapshot RO
        report = summarize(tmp_path / "snap.sqlite")
        out = capsys.readouterr().out
        assert "STRENGTH" in out and "SEARCH" in out
        # summary.json written next to the snapshot and round-trips.
        with open(tmp_path / "summary.json") as fh:
            loaded = json.load(fh)
        assert loaded["meta"]["n_deals"] == 1
        assert loaded["strength"]["arms"][0]["n_hands"] == 1
        assert report["meta"]["run_id"] == "R"

    def test_empty_db_does_not_crash(self, db):
        log, _ = db
        rep = build_report(log._con)           # no games at all
        assert rep["meta"]["n_deals"] == 0 and rep["meta"]["n_rows"] == 0
        assert rep["strength"]["arms"] == []
        assert rep["range_quality"]["conditions"] == []
        assert rep["search"]["conditions"] == []
        assert rep["paired"]["available"] is False
        _ = rep["flags"]                        # flag evaluation must tolerate NULLs
        assert "STRENGTH" in _print_human(rep)  # …and so must rendering

    def test_read_only_open_does_not_write(self, db):
        log, tmp_path = db
        with log.game():
            log.log_game(_game(0))
        log.snapshot(tmp_path / "snap.sqlite")
        # mode=ro: a summary run must not create -wal/-shm sidecars on the snapshot.
        summarize(tmp_path / "snap.sqlite", write_json=False, echo=False)
        sidecars = [p.name for p in tmp_path.iterdir()
                    if p.name.startswith("snap.sqlite-")]
        assert sidecars == []


# --------------------------------------------------------------------------- #
# AIVAT sanity check + dual-metric reporting
# --------------------------------------------------------------------------- #

def _aivat_arm(report, condition=None):
    hits = [
        a for a in report["aivat_health"]["arms"]
        if condition is None or a["condition"] == condition
    ]
    assert len(hits) == 1, f"expected exactly one aivat arm, got {hits}"
    return hits[0]


def _flag_keys(report):
    return {f["key"] for f in report["flags"]}


def _fill(log, raws, aivats, *, condition="vanilla", start=0):
    """One game per (raw, aivat) pair, all in one arm."""
    for i, (r, a) in enumerate(zip(raws, aivats)):
        with log.game():
            log.log_game(_game(start + i, condition=condition,
                               hero_chips_delta=r, aivat_value=a,
                               deck_seed=start + i))


class TestAivatVerdict:
    """``_aivat_verdict`` thresholds — the sign of the answer must not be fuzzy."""

    def test_below_one_is_harmful(self):
        from evaluation.summarize import _aivat_verdict
        assert _aivat_verdict(0.99) == "harmful"

    def test_marginal_gain_is_negligible(self):
        # AIVAT costs real per-hand compute; a 2% variance win does not earn it.
        from evaluation.summarize import _aivat_verdict
        assert _aivat_verdict(1.02) == "negligible"

    def test_clear_gain_is_helping(self):
        from evaluation.summarize import _aivat_verdict
        assert _aivat_verdict(1.4) == "helping"

    def test_missing_is_na(self):
        from evaluation.summarize import _aivat_verdict
        assert _aivat_verdict(None) == "n/a"


class TestAivatHealth:

    def test_var_x_is_the_variance_ratio(self, db):
        log, _ = db
        # aivat = raw/2 exactly -> var(raw)/var(aivat) == 4.
        raws = [100.0, -100.0, 300.0, -300.0, 500.0, -500.0]
        _fill(log, raws, [r / 2 for r in raws])
        a = _aivat_arm(build_report(log._con))
        assert math.isclose(a["var_x"], 4.0, rel_tol=1e-9)
        assert math.isclose(a["ci_shrink"], 2.0, rel_tol=1e-9)
        assert a["verdict"] == "helping"

    def test_detects_aivat_adding_variance(self, db):
        log, _ = db
        raws = [100.0, -100.0, 200.0, -200.0]
        _fill(log, raws, [r * 2 for r in raws])       # aivat is WORSE
        report = build_report(log._con)
        a = _aivat_arm(report)
        assert math.isclose(a["var_x"], 0.25, rel_tol=1e-9)
        assert a["verdict"] == "harmful"
        assert "aivat_harmful" in _flag_keys(report)

    def test_unbiased_shift_straddles_zero(self, db):
        log, _ = db
        raws = [100.0, -100.0, 300.0, -300.0]
        _fill(log, raws, [r / 2 for r in raws])       # mean shift exactly 0
        a = _aivat_arm(build_report(log._con))
        assert math.isclose(a["mean_shift"], 0.0, abs_tol=1e-9)
        assert a["shift_significant"] is False

    def test_flags_a_mean_shift_that_clears_its_ci(self, db):
        log, _ = db
        # A constant offset with no spread: the shift cannot be sampling noise, so
        # it is an implementation bug (the estimator is unbiased by construction).
        raws = [100.0, 100.0, 100.0, 100.0]
        _fill(log, raws, [r + 500.0 for r in raws])
        report = build_report(log._con)
        assert _aivat_arm(report)["shift_significant"] is True
        assert "aivat_biased" in _flag_keys(report)

    def test_absent_when_no_aivat_column(self, db):
        log, _ = db
        for i in range(4):
            with log.game():
                log.log_game(_game(i, condition="vanilla",
                                   hero_chips_delta=100.0, deck_seed=i))
        report = build_report(log._con)
        assert report["aivat_health"]["available"] is False
        assert not any(k.startswith("aivat_") for k in _flag_keys(report))

    def test_reported_even_when_headline_is_raw(self, db):
        """A run summarised on --metric raw still wants to know what AIVAT bought."""
        log, _ = db
        raws = [100.0, -100.0, 300.0, -300.0]
        _fill(log, raws, [r / 2 for r in raws])
        report = build_report(log._con, metric="raw")
        assert report["strength"]["used_aivat"] is False
        assert report["aivat_health"]["available"] is True
        assert math.isclose(_aivat_arm(report)["var_x"], 4.0, rel_tol=1e-9)


class TestBothMetricsReported:

    def test_alternate_metric_is_computed(self, db):
        log, _ = db
        # Deliberately asymmetric: with a zero-mean series both metrics report 0.0
        # and the "they differ" assertion below would pass vacuously.
        raws = [100.0, -100.0, 300.0, -100.0]
        _fill(log, raws, [r / 2 for r in raws])
        report = build_report(log._con)                    # auto -> aivat
        assert report["alt_metric"] == "raw"
        assert report["strength_alt"]["used_aivat"] is False
        # The two headlines must differ here, else the test proves nothing.
        assert (report["strength"]["arms"][0]["mean_bb100"]
                != report["strength_alt"]["arms"][0]["mean_bb100"])

    def test_alternate_is_none_when_aivat_incomplete(self, db):
        log, _ = db
        with log.game():
            log.log_game(_game(0, condition="v", hero_chips_delta=1.0,
                               aivat_value=1.0, deck_seed=0))
        with log.game():
            log.log_game(_game(1, condition="v", hero_chips_delta=1.0,
                               deck_seed=1))            # no aivat_value
        report = build_report(log._con)
        assert report["alt_metric"] is None
        assert report["strength_alt"] is None

    def test_human_block_shows_both_and_the_sanity_check(self, db):
        log, _ = db
        raws = [100.0, -100.0, 300.0, -300.0]
        _fill(log, raws, [r / 2 for r in raws])
        block = _print_human(build_report(log._con))
        assert "AIVAT SANITY" in block
        assert "var_x" in block
        assert "headline metric" in block
        # both column labels present on the strength line
        assert "*aivat" in block and "raw" in block
