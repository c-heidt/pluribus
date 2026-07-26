"""Tests for the end-of-run summary (evaluation/summarize.py, doc §8).

Two styles:

- **Hand-built DB** — rows inserted directly through :class:`ExperimentLog` with
  known values, so the strength CI, bb/100, net-info-gain roll-ups, routing check,
  and every flag are asserted against arithmetic we control.
- **Pure-function** unit tests for the stats helpers (``_mean_ci`` / ``_percentile``
  / ``_position_name``) and edge cases (empty DB must not crash).
"""

import json
import math
import sqlite3

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
    _query_paired,
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
        # unmapped table size falls back to POSk (offset from button).
        assert _position_name(3, 0, 4) == "POS3"

    def test_bootstrap_ci_is_deterministic_and_brackets_mean(self):
        vals = [1.0, 2.0, 3.0, 4.0, 100.0]        # heavy tail → bootstrap over normal
        a = _bootstrap_ci(vals, n_resamples=500)
        b = _bootstrap_ci(vals, n_resamples=500)
        assert a == b                              # fixed-seed → reproducible
        assert a["lo"] <= sum(vals) / len(vals) <= a["hi"]
        assert _bootstrap_ci([1.0])["lo"] is None  # < 2 points → undefined


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
        # Δ = [100, 0, 300] → mean 133.33; deck-matched, not (mean(DBR)-mean(vanilla)).
        assert math.isclose(cmp["mean_delta_bb100"], (100 + 0 + 300) / 3, rel_tol=1e-9)

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
        (cmp,) = pr["comparisons"]
        # Δ = aivat: [80-0, 60-50] = [80, 10] → mean 45.
        assert math.isclose(cmp["mean_delta_bb100"], 45.0, rel_tol=1e-9)
        assert cmp["ci95_bootstrap"]["n_resamples"] > 0


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
        s = rep["strength"]
        assert s["used_aivat"] is False
        assert s["overall"]["n"] == 2
        assert math.isclose(s["overall"]["mean"], 50.0)          # mean(200,-100)
        # by position: hero_seat 0 (BTN) got +200, hero_seat 1 (SB) got -100.
        assert math.isclose(s["by_position"]["BTN"]["mean"], 200.0)
        assert math.isclose(s["by_position"]["SB"]["mean"], -100.0)

    def test_prefers_aivat_when_fully_populated(self, db):
        log, _ = db
        with log.game():
            log.log_game(_game(0, hero_chips_delta=999.0, aivat_value=100.0))
        with log.game():
            log.log_game(_game(1, hero_chips_delta=999.0, aivat_value=300.0))
        s = build_report(log._con)["strength"]
        assert s["used_aivat"] is True
        assert math.isclose(s["overall"]["mean"], 200.0)         # from aivat, not 999

    def test_losing_flag_when_ci_below_zero(self, db):
        log, _ = db
        # Three identical -500 bb/100 hands → CI is 0, mean+ci < 0 → losing flag.
        for h in range(3):
            with log.game():
                log.log_game(_game(h, table_label="tough", hero_chips_delta=-500.0))
        rep = build_report(log._con)
        keys = {f["key"] for f in rep["flags"]}
        assert "losing" in keys


# --------------------------------------------------------------------------- #
# Range tracking
# --------------------------------------------------------------------------- #

class TestRangeHealth:

    def _hand_with_rq(self, log, hand, rows):
        with log.game():
            gid = log.log_game(_game(hand))
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
        rq = build_report(log._con)["range_quality"]
        assert rq["overall"]["snapshots"] == 2
        assert math.isclose(rq["overall"]["resolved_frac"], 0.5)
        assert math.isclose(rq["overall"]["net_info_gain"], 0.5)   # resolved-only
        assert rq["by_opponent"]["bp"]["snapshots"] == 2
        assert math.isclose(rq["by_stage"]["flop"]["net_info_gain"], 0.5)

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
# Approach / routing + search cost
# --------------------------------------------------------------------------- #

class TestApproachAndSearch:

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
            self._dec(log, gid, regime="vector", leaf_mode="exact_range",
                      betting_stage="preflop", num_live=2, stop_reason="iteration_cap",
                      wall_seconds=1.0, iterations=100, cache_hits=1, cache_misses=0)
        rep = build_report(log._con)
        assert rep["approach"]["routing_ok"] is False
        assert rep["approach"]["routing_violations"] == 1
        assert "routing" in {f["key"] for f in rep["flags"]}

    def test_vector_multiway_flagged(self, db):
        log, _ = db
        with log.game():
            gid = log.log_game(_game(0))
            # Vector must be heads-up: a multiway (num_live != 2) vector search is a
            # routing violation even on an in-envelope street.
            self._dec(log, gid, regime="vector", leaf_mode="exact_range",
                      betting_stage="flop", num_live=3, stop_reason="wall_cap",
                      wall_seconds=2.0, iterations=9000, cache_hits=9, cache_misses=1)
        rep = build_report(log._con)
        assert rep["approach"]["routing_ok"] is False
        assert rep["approach"]["routing_violations"] == 1

    def test_vector_headsup_flop_turn_river_is_clean(self, db):
        # §6.5: the vector regime fires heads-up on flop, turn, AND river — all clean.
        log, _ = db
        with log.game():
            gid = log.log_game(_game(0))
            for stage in ("flop", "turn", "river"):
                self._dec(log, gid, regime="vector", leaf_mode="exact_range",
                          betting_stage=stage, num_live=2, stop_reason="wall_cap",
                          wall_seconds=2.0, iterations=9000, cache_hits=9, cache_misses=1)
        rep = build_report(log._con)
        assert rep["approach"]["routing_ok"] is True
        assert rep["approach"]["routing_violations"] == 0

    def test_blueprint_prior_metrics_and_flag(self, db):
        # Over covered (searched) decisions, the summary reports how much the played
        # read was shrunk toward the blueprint, and flags an over-frequent fallback.
        log, _ = db
        with log.game():
            gid = log.log_game(_game(0))
            # Three heavily-blueprint plays (>50%) + one pure-search play.
            for w in (0.9, 0.8, 0.7, 0.0):
                self._dec(log, gid, regime="mccfr", leaf_mode="blueprint",
                          betting_stage="flop", num_live=2, stop_reason="wall_cap",
                          wall_seconds=1.0, iterations=100, cache_hits=1, cache_misses=0,
                          blueprint_weight=w)
        rep = build_report(log._con)
        ap = rep["approach"]
        assert ap["blueprint_weight_mean"] == pytest.approx((0.9 + 0.8 + 0.7) / 4)
        assert ap["blueprint_heavy_rate"] == pytest.approx(3 / 4)   # 3 of 4 > 0.5
        assert "blueprint_prior_bound" in {f["key"] for f in rep["flags"]}  # 75% > 20%

    def test_search_cost_and_budget_flag(self, db):
        log, _ = db
        with log.game():
            gid = log.log_game(_game(0))
            # Two searched (both wall_cap) + one blueprint (unsearched) decision.
            self._dec(log, gid, stop_reason="wall_cap", wall_seconds=10.0,
                      iterations=5000, iters_per_sec=500.0,
                      cache_hits=90, cache_misses=10, leaf_mode="sampled_runout",
                      num_live=3)
            self._dec(log, gid, stop_reason="wall_cap", wall_seconds=12.0,
                      iterations=6000, iters_per_sec=500.0,
                      cache_hits=90, cache_misses=10, leaf_mode="sampled_runout",
                      num_live=3)
            self._dec(log, gid, regime="blueprint", searched=0)
        rep = build_report(log._con)
        sc = rep["search"]
        assert sc["n_decisions"] == 3 and sc["n_searched"] == 2
        assert math.isclose(sc["fire_rate"], 2 / 3)
        assert math.isclose(sc["wallcap_rate"], 1.0)           # both wall_cap
        assert math.isclose(sc["cache_hit_rate"], 0.9)         # 180/200
        assert math.isclose(sc["p95_wall"], 11.9, rel_tol=1e-6)  # interp of [10,12]
        assert "budget_bound" in {f["key"] for f in rep["flags"]}


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
        assert "STRENGTH" in out and "SEARCH COST" in out
        # summary.json written next to the snapshot and round-trips.
        with open(tmp_path / "summary.json") as fh:
            loaded = json.load(fh)
        assert loaded["meta"]["n_hands"] == 1
        assert loaded["strength"]["overall"]["n"] == 1
        assert report["meta"]["run_id"] == "R"

    def test_empty_db_does_not_crash(self, db):
        log, _ = db
        rep = build_report(log._con)           # no games at all
        assert rep["meta"]["n_hands"] == 0
        assert rep["strength"]["overall"]["n"] == 0
        assert rep["range_quality"]["overall"]["resolved_frac"] is None
        assert rep["search"]["fire_rate"] is None
        _ = rep["flags"]                        # flag evaluation must tolerate NULLs

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
