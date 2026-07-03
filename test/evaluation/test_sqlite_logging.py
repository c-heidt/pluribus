"""Tests for the SQLite evaluation sink (evaluation/sqlite_logging.py, doc §4–§6, §9.2).

Covers the contract the later steps (runner §10.1, summary §8) lean on:

- ``open`` applies the §4 pragmas and creates the §6 schema (idempotent reopen).
- The four writers round-trip every column, including NULLs and JSON blobs, and
  children reference the parent ``game_id``.
- The per-hand :meth:`ExperimentLog.game` transaction is all-or-nothing (commit
  on success, rollback on error) — the property the resume cursor relies on.
- ``VACUUM INTO`` produces a readable, consistent snapshot while the DB is live.
- ``next_hand_index`` is an exact resume cursor per ``run_id``.
"""

import json
import sqlite3

import pytest

from evaluation.sqlite_logging import (
    SCHEMA_VERSION,
    _SCHEMA_DDL,
    DecisionRow,
    ExperimentLog,
    GameRow,
    HandFailureRow,
    RangeQualityRow,
    SeatRow,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _game(run_id="run", hand_index=0, **kw):
    base = dict(
        run_id=run_id,
        hand_index=hand_index,
        config_fingerprint="abc123",
        table_label="all_blueprint",
        table_config=json.dumps({"seats": ["bp"] * 6}),
        hero_seat=0,
        button_seat=1,
        n_players=6,
        big_blind=100.0,
        starting_stack=20000.0,
        deck_seed=42,
    )
    base.update(kw)
    return GameRow(**base)


@pytest.fixture
def log(tmp_path):
    lg = ExperimentLog.open(tmp_path / "run.sqlite")
    yield lg
    lg.close()


# --------------------------------------------------------------------------- #
# Open / schema / pragmas
# --------------------------------------------------------------------------- #

class TestOpen:

    def test_pragmas_applied(self, log):
        con = log._con
        assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        # synchronous NORMAL == 1
        assert con.execute("PRAGMA synchronous").fetchone()[0] == 1
        assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_all_four_tables_created(self, log):
        names = {
            r[0]
            for r in log._con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"games", "game_seats", "decisions", "range_quality"} <= names

    def test_reopen_existing_file_is_noop(self, tmp_path):
        path = tmp_path / "run.sqlite"
        lg = ExperimentLog.open(path)
        with lg.game():
            lg.log_game(_game())
        lg.close()
        # Reopen must not raise (CREATE TABLE IF NOT EXISTS) and must see the row.
        lg2 = ExperimentLog.open(path)
        assert lg2._con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
        lg2.close()

    def test_v1_db_migrates_hu_from_street(self, tmp_path):
        """Opening a schema-v1 DB adds ``games.hu_from_street`` (additive migration).

        A v1 file has the games table *without* the column; ``CREATE TABLE IF NOT
        EXISTS`` alone would leave it missing and the name-based insert would then
        fail.  ``open`` must ALTER it in, old rows read back NULL, and new rows
        round-trip the value.
        """
        path = tmp_path / "old.sqlite"
        v1_ddl = _SCHEMA_DDL.replace("    hu_from_street     INTEGER,\n", "")
        assert "hu_from_street" not in v1_ddl  # the replace actually removed it
        con = sqlite3.connect(str(path))
        con.executescript(v1_ddl)
        con.execute(
            "INSERT INTO games (run_id, hand_index, schema_version, "
            "config_fingerprint, table_label, table_config, hero_seat, "
            "button_seat, n_players, big_blind, starting_stack, deck_seed) "
            "VALUES ('old', 0, 1, 'f', 't', '{}', 0, 1, 3, 100.0, 600.0, 7)"
        )
        con.commit()
        con.close()

        lg = ExperimentLog.open(path)
        cols = {r[1] for r in lg._con.execute("PRAGMA table_info(games)")}
        assert "hu_from_street" in cols
        # Pre-migration row reads back NULL; a new row round-trips the value.
        assert lg._con.execute(
            "SELECT hu_from_street FROM games WHERE run_id = 'old'"
        ).fetchone()[0] is None
        with lg.game():
            lg.log_game(_game(run_id="new", hu_from_street=2))
        got = lg._con.execute(
            "SELECT hu_from_street FROM games WHERE run_id = 'new'"
        ).fetchone()[0]
        assert got == 2
        lg.close()


# --------------------------------------------------------------------------- #
# Writers / round-trip
# --------------------------------------------------------------------------- #

class TestWriters:

    def test_log_game_returns_incrementing_ids(self, log):
        with log.game():
            g1 = log.log_game(_game(hand_index=0))
            g2 = log.log_game(_game(hand_index=1))
        assert g2 == g1 + 1

    def test_game_row_round_trips(self, log):
        row = _game(
            hand_index=3,
            agent_seed=7,
            hero_chips_delta=-150.0,
            went_to_showdown=1,
            terminal_street="river",
            final_pot=800.0,
            hero_hole="Ah Kd",
            final_board="2c 7d 9s Ts Jh",
            git_sha="d64a49b",
            hostname="node01",
            started_at="2026-07-01T00:00:00",
        )
        with log.game():
            gid = log.log_game(row)
        got = log._con.execute(
            "SELECT * FROM games WHERE game_id = ?", (gid,)
        ).fetchone()
        cols = [d[0] for d in log._con.execute("SELECT * FROM games").description]
        rec = dict(zip(cols, got))
        assert rec["schema_version"] == SCHEMA_VERSION
        assert rec["hero_hole"] == "Ah Kd"
        assert rec["hero_chips_delta"] == -150.0
        assert rec["aivat_value"] is None          # nullable, unset
        assert json.loads(rec["table_config"])["seats"] == ["bp"] * 6

    def test_seats_decisions_range_quality_attach_to_game(self, log):
        with log.game():
            gid = log.log_game(_game())
            log.log_seats(
                gid,
                [
                    SeatRow(seat=0, is_hero=1, agent_label="hero"),
                    SeatRow(seat=1, is_hero=0, agent_label="bp_raise",
                            agent_config=json.dumps({"bias": "raise"})),
                ],
            )
            did = log.log_decision(
                gid,
                DecisionRow(
                    betting_stage="flop", regime="mccfr", searched=1,
                    leaf_mode="decision_free", num_live=2, iterations=5000,
                    wall_seconds=7.5, iters_per_sec=666.7, stop_reason="wall_cap",
                    node_count=1234, unique_pubkeys=42, cache_hits=900,
                    cache_misses=100, action_played="call",
                    action_dist=json.dumps({"fold": 0.1, "call": 0.9}),
                ),
            )
            rid = log.log_range_quality(
                gid,
                RangeQualityRow(seat=1, betting_stage="flop", resolved=0),
            )
        con = log._con
        assert con.execute(
            "SELECT COUNT(*) FROM game_seats WHERE game_id=?", (gid,)
        ).fetchone()[0] == 2
        dec = con.execute(
            "SELECT game_id, stop_reason, cache_hits, exploitability, action_dist "
            "FROM decisions WHERE decision_id=?", (did,)
        ).fetchone()
        assert dec[0] == gid and dec[1] == "wall_cap" and dec[2] == 900
        assert dec[3] is None                       # exploitability reserved, NULL
        assert json.loads(dec[4])["call"] == 0.9
        rq = con.execute(
            "SELECT game_id, resolved, net_info_gain FROM range_quality WHERE id=?",
            (rid,),
        ).fetchone()
        assert rq[0] == gid and rq[1] == 0 and rq[2] is None

    def test_foreign_key_rejects_orphan_child(self, log):
        # No games row with id 999 → FK enforcement must reject the decision.
        with pytest.raises(sqlite3.IntegrityError):
            with log.game():
                log.log_decision(
                    999, DecisionRow(betting_stage="flop", regime="mccfr", searched=0)
                )


# --------------------------------------------------------------------------- #
# Per-hand transaction (§4)
# --------------------------------------------------------------------------- #

class TestTransaction:

    def test_commit_persists_the_hand(self, log):
        with log.game():
            log.log_game(_game())
        assert log._con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1

    def test_rollback_leaves_no_partial_hand(self, log):
        with pytest.raises(RuntimeError):
            with log.game():
                gid = log.log_game(_game())
                log.log_seats(gid, [SeatRow(seat=0, is_hero=1, agent_label="hero")])
                raise RuntimeError("mid-hand failure")
        # Neither the game nor its seats survived — all-or-nothing.
        assert log._con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 0
        assert log._con.execute("SELECT COUNT(*) FROM game_seats").fetchone()[0] == 0


# --------------------------------------------------------------------------- #
# Snapshot (§5) + resume cursor (§10.1)
# --------------------------------------------------------------------------- #

class TestSnapshotAndResume:

    def test_snapshot_is_readable_and_consistent(self, log, tmp_path):
        with log.game():
            gid = log.log_game(_game(hero_chips_delta=250.0))
            log.log_seats(gid, [SeatRow(seat=0, is_hero=1, agent_label="hero")])
        dest = tmp_path / "permanent.sqlite"
        log.snapshot(dest)
        assert dest.exists()
        con = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
        try:
            assert con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
            assert con.execute(
                "SELECT hero_chips_delta FROM games"
            ).fetchone()[0] == 250.0
            assert con.execute("SELECT COUNT(*) FROM game_seats").fetchone()[0] == 1
        finally:
            con.close()

    def test_snapshot_keeps_taking_writes(self, log, tmp_path):
        with log.game():
            log.log_game(_game(hand_index=0))
        log.snapshot(tmp_path / "snap.sqlite")
        # DB still writable after a snapshot (autocommit, not inside a txn).
        with log.game():
            log.log_game(_game(hand_index=1))
        assert log._con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 2

    def test_sync_to_creates_and_overwrites_atomically(self, log, tmp_path):
        # First sync creates the permanent snapshot; a second sync to the SAME path
        # overwrites it (VACUUM INTO alone would raise "output file already exists").
        dest = tmp_path / "perm" / "experiment.sqlite"
        dest.parent.mkdir()
        with log.game():
            log.log_game(_game(hand_index=0))
        log.sync_to(dest)
        assert dest.exists()
        with log.game():
            log.log_game(_game(hand_index=1))
        log.sync_to(dest)                                   # must not raise
        con = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
        try:
            assert con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 2
        finally:
            con.close()

    def test_sync_to_leaves_no_temp_file(self, log, tmp_path):
        dest = tmp_path / "experiment.sqlite"
        with log.game():
            log.log_game(_game())
        log.sync_to(dest)
        # The snapshot-to-temp + os.replace must leave only the final file behind.
        leftovers = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
        assert leftovers == []

    def test_sync_to_preserves_dest_when_snapshot_fails(self, log, tmp_path, monkeypatch):
        # Atomicity: if VACUUM INTO the temp fails, os.replace is never reached, so
        # the previous good snapshot at dest survives intact (never half-written)
        # and no orphan temp is left in the destination dir.
        dest = tmp_path / "experiment.sqlite"
        with log.game():
            log.log_game(_game(hand_index=0))
        log.sync_to(dest)                              # good snapshot: 1 game
        with log.game():
            log.log_game(_game(hand_index=1))          # a second game, not yet synced

        def boom(_tmp):
            raise OSError("permanent FS full")

        monkeypatch.setattr(log, "snapshot", boom)
        with pytest.raises(OSError):
            log.sync_to(dest)
        con = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
        try:
            assert con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
        finally:
            con.close()
        assert [p.name for p in tmp_path.iterdir() if ".tmp." in p.name] == []

    def test_next_hand_index_empty_run_is_zero(self, log):
        assert log.next_hand_index("fresh") == 0

    def test_next_hand_index_is_one_past_max(self, log):
        for h in (0, 1, 2):
            with log.game():
                log.log_game(_game(run_id="A", hand_index=h))
        with log.game():
            log.log_game(_game(run_id="B", hand_index=9))
        assert log.next_hand_index("A") == 3        # 1 + max(0,1,2)
        assert log.next_hand_index("B") == 10       # per-run cursor, independent
        assert log.next_hand_index("C") == 0


class TestFailures:

    def test_log_failure_persists_outside_a_game_txn(self, log):
        # Autocommitted, so it survives even though the failed game rolled back.
        rid = log.log_failure(HandFailureRow(
            run_id="X", hand_index=4, deck_seed=99, agent_seed=7, hero_seat=2,
            error_type="RuntimeError", error="boom", traceback="Traceback...",
            git_sha="abc", hostname="node01", failed_at="2026-07-01T00:00:00",
        ))
        row = log._con.execute(
            "SELECT run_id, hand_index, deck_seed, error_type, error FROM hand_failures "
            "WHERE id=?", (rid,),
        ).fetchone()
        assert row == ("X", 4, 99, "RuntimeError", "boom")

    def test_cursor_counts_failed_hands(self, log):
        # A run whose only record is a failure still advances past it (no retry loop).
        log.log_failure(HandFailureRow(run_id="X", hand_index=0))
        assert log.next_hand_index("X") == 1
        # Max across BOTH games and failures.
        with log.game():
            log.log_game(_game(run_id="X", hand_index=1))
        log.log_failure(HandFailureRow(run_id="X", hand_index=2))
        assert log.next_hand_index("X") == 3        # max(games=1, failures=2) + 1
