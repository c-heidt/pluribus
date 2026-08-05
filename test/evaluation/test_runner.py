"""Tests for the evaluation runner core (evaluation/runner.py, doc §9.3, §10.1).

The core is driven against a **stub session** — a ``UniformPolicy`` blueprint, a
small deck, a stub LUT, and the real solver at a tiny iteration budget — so a full
run plays real hands and populates the real schema without a trained blueprint on
disk.  One ``requires_lut`` test exercises the same loop over the real 20-card deck
+ card LUT.  Covered: a full run populates games/game_seats/decisions, the resume
cursor continues cleanly, seeding is reproducible, and hero position rotates.
"""

import collections
import json
import sqlite3

import numpy as np
import pytest

from evaluation.runner import (
    EvalConfig,
    EvalSession,
    _hu_street,
    _sync_due,
    derive_seeds,
    run_evaluation,
)
from evaluation.sqlite_logging import ExperimentLog
from poker_ai.search.leaf import LeafConfig
from poker_ai.search.solver import SolverConfig
from test.search._helpers import UniformPolicy, _policies


# --------------------------------------------------------------------------- #
# Stub session + helpers
# --------------------------------------------------------------------------- #

def _stub_session(
    *,
    run_id="test",
    run_seed=7,
    table_policy="all_blueprint",
    fixed_seats=None,
    n_players=3,
    low=11,
    high=14,
    starting_stack=600,
    blueprint=None,
    card_info_lut=None,
) -> EvalSession:
    leaf = LeafConfig(policies=_policies(), n_rollouts=1)
    solver_cfg = SolverConfig(
        leaf=leaf, max_iterations=4, max_wall_seconds=30.0,
        discount_interval=20,   # serial → fast + deterministic
    )
    lut = card_info_lut
    if lut is None:
        lut = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    cfg = EvalConfig(
        run_id=run_id, run_seed=run_seed, table_policy=table_policy,
        fixed_seats=fixed_seats,
        n_players=n_players, time_budget_hours=0.0,   # unbounded → max_hands governs
        big_blind=100, small_blind=50, starting_stack=starting_stack,
        low_card_rank=low, high_card_rank=high,
    )
    return EvalSession(
        config=cfg, solver_cfg=solver_cfg,
        blueprint_policy=blueprint or UniformPolicy(), card_info_lut=lut,
    )


def _rows(con, sql, *args):
    return con.execute(sql, args).fetchall()


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #

class TestSeeding:

    def test_derive_seeds_is_pure_function_of_run_and_hand(self):
        a = derive_seeds(11, 3)
        b = derive_seeds(11, 3)
        assert a[0] == b[0] and a[1] == b[1]           # deck_seed, agent_seed stable

    def test_different_hands_differ(self):
        assert derive_seeds(11, 0)[0] != derive_seeds(11, 1)[0]

    def test_different_runs_differ(self):
        assert derive_seeds(1, 0)[0] != derive_seeds(2, 0)[0]


# --------------------------------------------------------------------------- #
# A full run
# --------------------------------------------------------------------------- #

class TestRun:

    def test_populates_all_grains(self, tmp_path):
        session = _stub_session(n_players=3)
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            n = run_evaluation(log=log, session=session, max_hands=4,
                               now_fn=lambda: "2026-07-01T00:00:00",
                               git_sha="abc1234", hostname="node01")
        finally:
            con = log._con
            games = _rows(con, "SELECT game_id, hand_index, hero_seat, n_players, "
                          "hero_chips_delta, config_fingerprint FROM games ORDER BY hand_index")
            seats = _rows(con, "SELECT COUNT(*) FROM game_seats")
            decisions = _rows(con, "SELECT COUNT(*) FROM decisions")
            log.close()

        assert n == 4
        assert [g[1] for g in games] == [0, 1, 2, 3]      # 0-based hand_index
        assert all(g[3] == 3 for g in games)              # n_players logged
        assert all(g[5] for g in games)                   # config_fingerprint set
        assert seats[0][0] == 4 * 3                        # one seat row per (game, seat)
        assert decisions[0][0] >= 1                        # at least one hero decision

    def test_hu_from_street_semantics(self):
        """``_hu_street``: street iff exactly two active seats incl. the hero.

        HU coverage for OX-Search-HU (opponent-modeling doc §11.4): the earliest round
        start that is heads-up *with the hero* counts; a table that is heads-up
        without the hero (hero folded) does not.
        """
        import collections

        from environment.player import Player
        from environment.poker_env import PokerEnv

        np.random.seed(0)
        env = PokerEnv(players=[Player(i, 600) for i in range(3)],
                       low_card_rank=11, high_card_rank=14)
        env.card_info_lut = collections.defaultdict(
            lambda: collections.defaultdict(lambda: 0)
        )
        # Three active seats: not heads-up for anyone.
        assert _hu_street(env, 0, env.betting_round) is None
        # Fold the current actor: exactly two remain.
        folder = env.player_i
        env.step_in_place("fold")
        survivors = [i for i in range(3) if i != folder]
        street = env.betting_round
        for hero in survivors:
            assert _hu_street(env, hero, street) == street
        assert _hu_street(env, folder, street) is None    # hero folded ⇒ no coverage

    def test_hu_from_street_logged_and_sane(self, tmp_path):
        """Every ``games.hu_from_street`` is NULL or a street a 3-max hand can
        first turn heads-up on (flop/turn/river — never preflop, which starts
        3-active), and the column is populated by the run loop."""
        session = _stub_session(n_players=3)
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            n = run_evaluation(log=log, session=session, max_hands=6,
                               now_fn=lambda: "2026-07-01T00:00:00",
                               git_sha="abc1234", hostname="node01")
            rows = _rows(log._con, "SELECT hu_from_street FROM games")
        finally:
            log.close()
        assert n == 6 and len(rows) == 6
        vals = [r[0] for r in rows]
        assert all(v is None or v in (1, 2, 3) for v in vals)

    def test_hero_seat_rotates_by_hand(self, tmp_path):
        session = _stub_session(n_players=3)
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            run_evaluation(log=log, session=session, max_hands=6)
            hero_seats = [
                r[0] for r in _rows(
                    log._con, "SELECT hero_seat FROM games ORDER BY hand_index"
                )
            ]
        finally:
            log.close()
        assert hero_seats == [0, 1, 2, 0, 1, 2]           # hand_index % n_players

    def test_exactly_one_hero_seat_per_game(self, tmp_path):
        session = _stub_session(n_players=3)
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            run_evaluation(log=log, session=session, max_hands=3)
            rows = _rows(log._con,
                         "SELECT game_id, SUM(is_hero) FROM game_seats GROUP BY game_id")
        finally:
            log.close()
        assert all(s == 1 for _, s in rows)

    def test_searched_decisions_carry_step1_metadata(self, tmp_path):
        session = _stub_session(n_players=2, starting_stack=1000)
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            run_evaluation(log=log, session=session, max_hands=6)
            rows = _rows(
                log._con,
                "SELECT regime, stop_reason, node_count, unique_pubkeys, "
                "cache_hits, iterations, action_dist FROM decisions WHERE searched=1"
            )
        finally:
            log.close()
        # Some searched decisions should exist (rounds 2-4 always solve HU).
        if rows:
            for regime, stop, nodes, pks, hits, iters, dist in rows:
                assert regime in ("mccfr", "vector")
                assert stop in ("iteration_cap", "wall_cap")
                assert nodes is not None and pks is not None
                assert json.loads(dist)                    # valid JSON action dist

    def test_showdown_is_detected_for_allin_hands(self, tmp_path):
        # Regression: the engine reports betting_stage='terminal' even at a showdown,
        # so went_to_showdown must key off the live-seat count, not the stage string.
        # An all-in/call HU policy guarantees showdowns — none must read as 0.
        class _AllIn(UniformPolicy):
            def strategy(self, state, bias="none"):
                legal = list(state.legal_actions)
                w = np.array(
                    [1.0 if a in ("all_in", "call", "check") else 0.0 for a in legal],
                    dtype=np.float32,
                )
                return w / w.sum() if w.sum() > 0 else np.full(
                    len(legal), 1.0 / len(legal), dtype=np.float32
                )

        session = _stub_session(n_players=2, starting_stack=1000, blueprint=_AllIn())
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            run_evaluation(log=log, session=session, max_hands=8)
            n_showdown = _rows(log._con, "SELECT SUM(went_to_showdown) FROM games")[0][0]
        finally:
            log.close()
        assert n_showdown and n_showdown >= 1     # would be 0 under the old check

    def test_range_quality_rows_populate_and_resolve(self, tmp_path):
        # Step 4: a run buffers opponent beliefs per street and resolves them at
        # showdown.  An all-in/call HU policy forces showdowns, so at least some
        # snapshots resolve with the full metric block (net_info_gain non-NULL);
        # every row carries a stage and an opponent (non-hero) seat.
        class _AllIn(UniformPolicy):
            def strategy(self, state, bias="none"):
                legal = list(state.legal_actions)
                w = np.array(
                    [1.0 if a in ("all_in", "call", "check") else 0.0 for a in legal],
                    dtype=np.float32,
                )
                return w / w.sum() if w.sum() > 0 else np.full(
                    len(legal), 1.0 / len(legal), dtype=np.float32
                )

        session = _stub_session(n_players=2, starting_stack=1000, blueprint=_AllIn())
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            run_evaluation(log=log, session=session, max_hands=12)
            rows = _rows(
                log._con,
                "SELECT rq.seat, rq.betting_stage, rq.resolved, rq.net_info_gain, "
                "g.hero_seat FROM range_quality rq JOIN games g ON g.game_id=rq.game_id"
            )
            resolved = _rows(
                log._con, "SELECT COUNT(*) FROM range_quality WHERE resolved=1"
            )[0][0]
            # Resolved rows must carry the full metric block; unresolved must not.
            bad = _rows(
                log._con,
                "SELECT COUNT(*) FROM range_quality "
                "WHERE (resolved=1) != (net_info_gain IS NOT NULL)"
            )[0][0]
        finally:
            log.close()
        assert rows, "no range_quality rows were logged"
        for seat, stage, res, gain, hero_seat in rows:
            assert seat != hero_seat                      # opponent seats only
            assert stage in ("flop", "turn", "river")
            assert res in (0, 1)
        assert resolved >= 1                              # showdowns did resolve
        assert bad == 0                                   # metrics ⇔ resolved

    def test_blueprint_round1_decisions_logged(self, tmp_path):
        session = _stub_session(n_players=2, starting_stack=1000)
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            run_evaluation(log=log, session=session, max_hands=6)
            regimes = {
                r[0] for r in _rows(
                    log._con, "SELECT DISTINCT regime FROM decisions WHERE searched=0"
                )
            }
        finally:
            log.close()
        # Round-1 hero plays are logged as blueprint (no search fired).
        assert regimes <= {"blueprint"}


# --------------------------------------------------------------------------- #
# Resume + determinism
# --------------------------------------------------------------------------- #

class TestResumeAndDeterminism:

    def test_resume_continues_from_cursor(self, tmp_path):
        path = tmp_path / "run.sqlite"
        session = _stub_session(run_id="R", n_players=3)
        log = ExperimentLog.open(path)
        run_evaluation(log=log, session=session, max_hands=3)
        log.close()

        # Reopen and resume: continues at hand_index 3, no replays.
        log2 = ExperimentLog.open(path)
        try:
            run_evaluation(log=log2, session=_stub_session(run_id="R", n_players=3),
                           max_hands=2)
            idx = [r[0] for r in _rows(log2._con,
                   "SELECT hand_index FROM games ORDER BY hand_index")]
        finally:
            log2.close()
        assert idx == [0, 1, 2, 3, 4]                     # contiguous, no gaps/dupes

    def test_same_seed_reproduces_outcomes(self, tmp_path):
        def _run(p):
            log = ExperimentLog.open(p)
            try:
                run_evaluation(log=log, session=_stub_session(run_seed=99, n_players=3),
                               max_hands=4)
                return _rows(log._con, "SELECT deck_seed, agent_seed, hero_chips_delta, "
                             "hero_hole FROM games ORDER BY hand_index")
            finally:
                log.close()
        assert _run(tmp_path / "a.sqlite") == _run(tmp_path / "b.sqlite")


# --------------------------------------------------------------------------- #
# Cross-condition CRN plumbing (§10.1) — condition label, paired-mode max_hands,
# and the load-bearing property: the deal is hero-independent.
# --------------------------------------------------------------------------- #

class TestCRN:

    def test_condition_is_logged_on_every_game(self, tmp_path):
        session = _stub_session(run_id="C", n_players=3)
        session.config.condition = "A(p0.8,t50)"
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            run_evaluation(log=log, session=session, max_hands=3)
            conds = [r[0] for r in _rows(log._con, "SELECT condition FROM games")]
        finally:
            log.close()
        assert conds == ["A(p0.8,t50)"] * 3

    def test_cfg_max_hands_is_total_based_and_ignores_budget(self, tmp_path):
        path = tmp_path / "run.sqlite"
        session = _stub_session(run_id="P", n_players=3)
        session.config.max_hands = 3
        session.config.time_budget_hours = 999.0     # would never stop if consulted
        log = ExperimentLog.open(path)
        try:
            n = run_evaluation(log=log, session=session)   # no param → cfg governs
            idx = [r[0] for r in _rows(log._con,
                   "SELECT hand_index FROM games ORDER BY hand_index")]
        finally:
            log.close()
        assert n == 3 and idx == [0, 1, 2]           # fixed count, budget ignored

        # Resume: total is already 3, so a re-run plays 0 more (total-based, not
        # per-call — the paired arms must not over-run on restart).
        log2 = ExperimentLog.open(path)
        s2 = _stub_session(run_id="P", n_players=3)
        s2.config.max_hands = 3
        try:
            n2 = run_evaluation(log=log2, session=s2)
            idx2 = [r[0] for r in _rows(log2._con,
                    "SELECT hand_index FROM games ORDER BY hand_index")]
        finally:
            log2.close()
        assert n2 == 0 and idx2 == [0, 1, 2]

    def test_deal_is_hero_independent(self, tmp_path, monkeypatch):
        """Same (run_seed, config) ⇒ identical cards per hand, regardless of what the
        hero does mid-hand — the load-bearing CRN property (§10.1).  Simulated by a
        hero that burns extra global-RNG each hand; the deal must not move."""
        import evaluation.runner as R
        real = R.play_hand

        def deal(p):
            log = ExperimentLog.open(p)
            try:
                run_evaluation(log=log, session=_stub_session(run_seed=5, n_players=3),
                               max_hands=5)
                return _rows(log._con, "SELECT hand_index, deck_seed, hero_hole "
                             "FROM games ORDER BY hand_index")
            finally:
                log.close()

        baseline = deal(tmp_path / "a.sqlite")

        def greedy(*a, **kw):
            np.random.rand(257)                      # a hero that behaves differently
            return real(*a, **kw)

        monkeypatch.setattr(R, "play_hand", greedy)
        varied = deal(tmp_path / "b.sqlite")

        assert len(baseline) == 5
        assert baseline == varied                    # deck_seed + hole cards unchanged


# --------------------------------------------------------------------------- #
# Config validation + table policies
# --------------------------------------------------------------------------- #

class TestConfigValidation:

    @pytest.mark.parametrize("kw", [
        {"n_players": 1},
        {"table_policy": "nonsense"},
        {"table_policy": "fixed"},                       # no fixed_seats
        {"table_policy": "fixed", "fixed_seats": {0: "bp"}},   # incomplete map
        {"table_policy": "fixed",
         "fixed_seats": {0: "bp", 1: "bp", 2: "who?"}},        # bad label
    ])
    def test_bad_config_fails_fast(self, tmp_path, kw):
        session = _stub_session(**kw)          # helper defaults n_players=3
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            with pytest.raises(ValueError):
                run_evaluation(log=log, session=session, max_hands=1)
            # Failed before playing anything — no rows, no wasted failure logging.
            assert _rows(log._con, "SELECT COUNT(*) FROM games")[0][0] == 0
            assert _rows(log._con, "SELECT COUNT(*) FROM hand_failures")[0][0] == 0
        finally:
            log.close()

    def test_nonpositive_max_hands_fails_fast(self, tmp_path):
        session = _stub_session(n_players=3)
        session.config.max_hands = 0
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            with pytest.raises(ValueError, match="max_hands"):
                run_evaluation(log=log, session=session)
        finally:
            log.close()

    def test_fixed_policy_assigns_and_logs_per_seat(self, tmp_path):
        fixed = {0: "bp", 1: "bp_raise", 2: "bp_call"}
        session = _stub_session(table_policy="fixed", fixed_seats=fixed, n_players=3)
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            run_evaluation(log=log, session=session, max_hands=3)
            # Non-hero seats carry their fixed label; the hero's seat is 'hero'.
            rows = _rows(log._con,
                "SELECT g.hero_seat, s.seat, s.agent_label FROM games g "
                "JOIN game_seats s ON s.game_id=g.game_id ORDER BY g.hand_index, s.seat")
        finally:
            log.close()
        for hero_seat, seat, label in rows:
            if seat == hero_seat:
                assert label == "hero"
            else:
                assert label == fixed[seat]              # fixed by seat, per doc §10.1


# --------------------------------------------------------------------------- #
# Per-hand fault tolerance (§9.3)
# --------------------------------------------------------------------------- #

class TestFailureHandling:

    def test_isolated_failure_is_logged_and_run_continues(self, tmp_path, monkeypatch):
        import evaluation.runner as R
        real = R.play_hand
        calls = {"n": 0}

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 2:                    # fail the 2nd hand (hand_index 1)
                raise RuntimeError("boom in play")
            return real(*a, **kw)

        monkeypatch.setattr(R, "play_hand", flaky)
        session = _stub_session(run_id="F", n_players=2, starting_stack=1000)
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            n = run_evaluation(log=log, session=session, max_hands=4,
                               now_fn=lambda: "2026-07-01T00:00:00",
                               git_sha="g", hostname="h")
            games = [r[0] for r in _rows(log._con,
                     "SELECT hand_index FROM games ORDER BY hand_index")]
            fails = _rows(log._con, "SELECT hand_index, error_type, error, deck_seed, "
                          "hero_seat FROM hand_failures")
            cursor = log.next_hand_index("F")
        finally:
            log.close()
        assert n == 4                              # all 4 attempted
        assert games == [0, 2, 3]                  # the failed hand rolled back
        assert len(fails) == 1
        idx, etype, err, deck, hero = fails[0]
        assert idx == 1 and etype == "RuntimeError" and "boom" in err
        assert deck is not None and hero == 1      # seeds/seat captured for replay
        assert cursor == 4                         # cursor counts the failure

    def test_circuit_breaker_aborts_on_systematic_failure(self, tmp_path, monkeypatch):
        import evaluation.runner as R

        def always_fail(*a, **kw):
            raise ValueError("systematic")

        monkeypatch.setattr(R, "play_hand", always_fail)
        session = _stub_session(run_id="B", n_players=2)
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            with pytest.raises(RuntimeError, match="consecutive"):
                run_evaluation(log=log, session=session, max_hands=100,
                               max_consecutive_failures=3)
            n_fail = _rows(log._con, "SELECT COUNT(*) FROM hand_failures")[0][0]
        finally:
            log.close()
        assert n_fail == 3                         # each logged before the abort


# --------------------------------------------------------------------------- #
# Cluster I/O sync-back (§5, step 5)
# --------------------------------------------------------------------------- #

class TestSyncBack:

    def test_periodic_and_final_sync_fire(self, tmp_path):
        # sync_interval_hands=2 over 5 hands → periodic at hands 2 and 4, plus the
        # final sync after the loop = 3 calls.
        session = _stub_session(n_players=3)
        session.config.sync_interval_hands = 2
        calls = {"n": 0}
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            run_evaluation(log=log, session=session, max_hands=5,
                           sync_fn=lambda: calls.__setitem__("n", calls["n"] + 1))
        finally:
            log.close()
        assert calls["n"] == 3                    # 2 periodic + 1 final

    def test_only_final_sync_when_interval_zero(self, tmp_path):
        session = _stub_session(n_players=3)
        session.config.sync_interval_hands = 0    # 0 → only the final sync
        session.config.sync_interval_minutes = 0.0
        calls = {"n": 0}
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            run_evaluation(log=log, session=session, max_hands=4,
                           sync_fn=lambda: calls.__setitem__("n", calls["n"] + 1))
        finally:
            log.close()
        assert calls["n"] == 1                    # final sync only

    def test_no_sync_fn_is_a_noop(self, tmp_path):
        # Backwards compatible: without a sync_fn the loop behaves exactly as before.
        session = _stub_session(n_players=3)
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            n = run_evaluation(log=log, session=session, max_hands=3)
        finally:
            log.close()
        assert n == 3

    def test_periodic_sync_failure_does_not_abort_run(self, tmp_path):
        # A transient permanent-FS hiccup on a periodic sync is best-effort: logged,
        # not fatal — the run finishes and the final (strict) sync still runs.
        session = _stub_session(n_players=3)
        session.config.sync_interval_hands = 1
        calls = {"n": 0}

        def flaky_sync():
            calls["n"] += 1
            if calls["n"] <= 2:                   # first two (periodic) calls fail
                raise OSError("permanent FS hiccup")

        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            n = run_evaluation(log=log, session=session, max_hands=3,
                               sync_fn=flaky_sync)
        finally:
            log.close()
        assert n == 3                             # run completed despite sync errors

    def test_final_sync_runs_on_should_stop(self, tmp_path, monkeypatch):
        # A SIGTERM (should_stop) breaks the loop at a hand boundary and still syncs.
        session = _stub_session(n_players=3)
        session.config.sync_interval_hands = 1000  # never periodic in this short run
        calls = {"n": 0}
        stop = {"go": False}

        # Stop after the first hand: flip the flag from a play_hand wrapper.
        import evaluation.runner as R
        real = R.play_hand

        def once(*a, **kw):
            out = real(*a, **kw)
            stop["go"] = True
            return out

        monkeypatch.setattr(R, "play_hand", once)
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            run_evaluation(log=log, session=session, max_hands=50,
                           should_stop=lambda: stop["go"],
                           sync_fn=lambda: calls.__setitem__("n", calls["n"] + 1))
        finally:
            log.close()
        assert calls["n"] == 1                    # final sync fired after the stop

    def test_final_sync_failure_propagates(self, tmp_path):
        # The final sync is STRICT: the permanent snapshot is the run's product, so
        # a failure there must propagate (not be silently swallowed like a periodic).
        session = _stub_session(n_players=3)
        session.config.sync_interval_hands = 0    # 0 → the only sync is the final one

        def boom():
            raise OSError("permanent FS full at final sync")

        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            with pytest.raises(OSError, match="final sync"):
                run_evaluation(log=log, session=session, max_hands=3, sync_fn=boom)
        finally:
            log.close()

    def test_circuit_breaker_syncs_before_aborting(self, tmp_path, monkeypatch):
        # A best-effort sync must fire before the circuit-breaker re-raises, so the
        # hands logged before a systematic failure are preserved on the permanent FS.
        import evaluation.runner as R

        def always_fail(*a, **kw):
            raise ValueError("systematic")

        monkeypatch.setattr(R, "play_hand", always_fail)
        session = _stub_session(run_id="CB", n_players=2)   # default interval → no periodic
        calls = {"n": 0}
        log = ExperimentLog.open(tmp_path / "run.sqlite")
        try:
            with pytest.raises(RuntimeError, match="consecutive"):
                run_evaluation(log=log, session=session, max_hands=100,
                               max_consecutive_failures=3,
                               sync_fn=lambda: calls.__setitem__("n", calls["n"] + 1))
        finally:
            log.close()
        assert calls["n"] == 1                    # one best-effort sync, before abort


class TestSyncDue:
    """Unit coverage for the §5 sync cadence (both the hands and minutes paths)."""

    def _cfg(self, **kw):
        return EvalConfig(run_id="x", **kw)

    def test_hands_cadence(self):
        cfg = self._cfg(sync_interval_hands=10, sync_interval_minutes=0.0)
        assert _sync_due(cfg, 9, 0.0) is False
        assert _sync_due(cfg, 10, 0.0) is True     # >= boundary
        assert _sync_due(cfg, 11, 0.0) is True

    def test_minutes_cadence(self):
        cfg = self._cfg(sync_interval_hands=0, sync_interval_minutes=5.0)
        assert _sync_due(cfg, 0, 4 * 60.0) is False
        assert _sync_due(cfg, 0, 5 * 60.0) is True  # 5 min reached
        assert _sync_due(cfg, 0, 9 * 60.0) is True

    def test_either_trigger_fires(self):
        cfg = self._cfg(sync_interval_hands=10, sync_interval_minutes=5.0)
        assert _sync_due(cfg, 10, 0.0) is True      # hands alone
        assert _sync_due(cfg, 0, 5 * 60.0) is True  # minutes alone

    def test_both_zero_disables(self):
        cfg = self._cfg(sync_interval_hands=0, sync_interval_minutes=0.0)
        assert _sync_due(cfg, 10_000, 10_000.0) is False


# --------------------------------------------------------------------------- #
# Real 20-card deck + LUT integration
# --------------------------------------------------------------------------- #

@pytest.mark.requires_lut
def test_run_over_real_lut(tmp_path, lut):
    """The full loop over the real 20-card deck + card LUT (real clustering)."""
    session = _stub_session(n_players=3, low=10, high=14, starting_stack=600,
                            card_info_lut=lut)
    log = ExperimentLog.open(tmp_path / "run.sqlite")
    try:
        n = run_evaluation(log=log, session=session, max_hands=2)
        games = _rows(log._con, "SELECT COUNT(*) FROM games")[0][0]
        # The join across grains the summary (§8) will use must work.
        joined = _rows(log._con,
            "SELECT g.hand_index, s.agent_label FROM games g "
            "JOIN game_seats s ON s.game_id = g.game_id WHERE s.is_hero=1")
    finally:
        log.close()
    assert n == 2 and games == 2
    assert len(joined) == 2
