"""The SQLite logging sink for evaluation runs (docs/evaluation.md §4–§6, §9.2).

A single :class:`ExperimentLog` owns one SQLite connection and is the only writer
(the run plays hands sequentially on one node, §1), so there is no concurrency to
manage — just a clean append path and a snapshot primitive.  The four grains of
§3 map to four tables (`games`, `game_seats`, `decisions`, `range_quality`); the
row shape of each is a dataclass here (:class:`GameRow` etc.) whose field names
match the DDL columns one-for-one, so the ``INSERT`` is generated from the
dataclass and the Python side stays in sync with the schema.

Usage (one commit per hand — the crash-safety + write-batching boundary of §4):

    log = ExperimentLog.open("run.sqlite")
    with log.game():                       # BEGIN … COMMIT (ROLLBACK on error)
        gid = log.log_game(GameRow(run_id="r", hand_index=0, ...))
        log.log_seats(gid, [SeatRow(seat=0, is_hero=1, agent_label="hero"), ...])
        log.log_decision(gid, DecisionRow(betting_stage="flop", regime="mccfr",
                                          searched=1, ...))
        log.log_range_quality(gid, RangeQualityRow(seat=1, betting_stage="flop",
                                                   resolved=0))
    log.snapshot("permanent/run.sqlite")   # VACUUM INTO (outside a transaction)
    log.close()

Design points carried straight from the doc:

- **WAL + ``synchronous=NORMAL``** (§4) — an analysis query can read while the run
  writes, and per-commit ``fsync`` is skipped (safe against corruption on
  node-local disk; only the last transaction is at risk on an OS/power crash).
- **One transaction per hand** — :meth:`game` is that boundary; every log call
  must run inside it.  The connection is in autocommit mode
  (``isolation_level=None``) so schema creation / snapshots run outside any
  transaction and :meth:`game` controls the hand transaction explicitly.
- **``CREATE TABLE IF NOT EXISTS``** — reopening an existing file (the resume
  path, §10.1) is a no-op, not an error.
- **No search-package dependency** — the caller (the future runner) unpacks a
  ``SearchResult`` into a :class:`DecisionRow`; this module never imports it.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional

# Bump on any schema change (games.schema_version); lets analysis span runs (§6).
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Schema (§6) — faithful to the doc's DDL, with IF NOT EXISTS for reopen/resume.
# ---------------------------------------------------------------------------

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS games (
    game_id            INTEGER PRIMARY KEY,
    run_id             TEXT    NOT NULL,
    hand_index         INTEGER NOT NULL,
    schema_version     INTEGER NOT NULL,
    config_fingerprint TEXT    NOT NULL,
    table_label        TEXT    NOT NULL,
    table_config       TEXT    NOT NULL,
    hero_seat          INTEGER NOT NULL,
    button_seat        INTEGER NOT NULL,
    n_players          INTEGER NOT NULL,
    big_blind          REAL    NOT NULL,
    starting_stack     REAL    NOT NULL,
    deck_seed          INTEGER NOT NULL,
    agent_seed         INTEGER,
    aivat_value        REAL,
    pairing_id         INTEGER,
    variant            TEXT,
    hero_chips_delta   REAL,
    went_to_showdown   INTEGER,
    terminal_street    TEXT,
    final_pot          REAL,
    hero_hole          TEXT,
    final_board        TEXT,
    git_sha            TEXT,
    hostname           TEXT,
    started_at         TEXT
);

CREATE TABLE IF NOT EXISTS game_seats (
    game_id      INTEGER NOT NULL REFERENCES games(game_id),
    seat         INTEGER NOT NULL,
    is_hero      INTEGER NOT NULL,
    agent_label  TEXT    NOT NULL,
    agent_config TEXT,
    PRIMARY KEY (game_id, seat)
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id     INTEGER PRIMARY KEY,
    game_id         INTEGER NOT NULL REFERENCES games(game_id),
    betting_stage   TEXT    NOT NULL,
    regime          TEXT    NOT NULL,
    leaf_mode       TEXT,
    searched        INTEGER NOT NULL,
    is_research     INTEGER,
    num_live        INTEGER,
    pot_before      REAL,
    to_call         REAL,
    hero_stack      REAL,
    iterations      INTEGER,
    wall_seconds    REAL,
    iters_per_sec   REAL,
    stop_reason     TEXT,
    node_count      INTEGER,
    unique_pubkeys  INTEGER,
    cache_hits      INTEGER,
    cache_misses    INTEGER,
    action_played   TEXT,
    action_dist     TEXT,
    exploitability  REAL,
    game_value      REAL
);

CREATE TABLE IF NOT EXISTS range_quality (
    id                 INTEGER PRIMARY KEY,
    game_id            INTEGER NOT NULL REFERENCES games(game_id),
    seat               INTEGER NOT NULL,
    betting_stage      TEXT    NOT NULL,
    n_actions_replayed INTEGER,
    true_combo         INTEGER,
    true_combo_mass    REAL,
    true_combo_rank    REAL,
    effective_support  INTEGER,
    log_loss           REAL,
    log_loss_uniform   REAL,
    net_info_gain      REAL,
    collapsed_truth    INTEGER,
    uniform_fallback   INTEGER,
    entropy            REAL,
    resolved           INTEGER NOT NULL
);

-- One row per HAND that raised while being played (§9.3).  Kept out of `games`
-- so that grain stays "complete hands only" — the per-game transaction rolls a
-- failed hand's partial writes back — while the failure is still recorded (not
-- silently skipped), with the seeds needed to reproduce it.  The resume cursor
-- counts these too, so a deterministically-failing hand is not retried forever.
CREATE TABLE IF NOT EXISTS hand_failures (
    id           INTEGER PRIMARY KEY,
    run_id       TEXT    NOT NULL,
    hand_index   INTEGER NOT NULL,
    deck_seed    INTEGER,
    agent_seed   INTEGER,
    hero_seat    INTEGER,
    error_type   TEXT,
    error        TEXT,
    traceback    TEXT,
    git_sha      TEXT,
    hostname     TEXT,
    failed_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_games_table     ON games(table_label);
CREATE INDEX IF NOT EXISTS idx_games_run       ON games(run_id, hand_index);
CREATE INDEX IF NOT EXISTS idx_games_pairing   ON games(pairing_id);
CREATE INDEX IF NOT EXISTS idx_seats_agent     ON game_seats(agent_label);
CREATE INDEX IF NOT EXISTS idx_seats_lookup    ON game_seats(game_id, seat);
CREATE INDEX IF NOT EXISTS idx_decisions_game  ON decisions(game_id);
CREATE INDEX IF NOT EXISTS idx_decisions_stage ON decisions(betting_stage);
CREATE INDEX IF NOT EXISTS idx_range_game      ON range_quality(game_id);
CREATE INDEX IF NOT EXISTS idx_failures_run    ON hand_failures(run_id, hand_index);
"""


# ---------------------------------------------------------------------------
# Row shapes (§6) — one dataclass per table; field names == column names.
# Required (NOT NULL) columns are positional; nullable ones default to None.
# ``game_id`` is omitted: it is the autoincrement key (games) or supplied by the
# log method from the parent insert (children).
# ---------------------------------------------------------------------------


@dataclass
class GameRow:
    """One deal (§6 ``games``).  bb/100 is per-hand — ``game`` == ``hand``."""

    run_id: str
    hand_index: int
    config_fingerprint: str
    table_label: str
    table_config: str            # JSON blob (all seats' agents), provenance
    hero_seat: int
    button_seat: int
    n_players: int
    big_blind: float
    starting_stack: float
    deck_seed: int
    # reproducibility / variance reduction / outcome / provenance (nullable) ----
    agent_seed: Optional[int] = None
    aivat_value: Optional[float] = None          # filled once AIVAT exists (§10.2)
    pairing_id: Optional[int] = None
    variant: Optional[str] = None
    hero_chips_delta: Optional[float] = None     # raw primary outcome (chips)
    went_to_showdown: Optional[int] = None
    terminal_street: Optional[str] = None
    final_pot: Optional[float] = None
    hero_hole: Optional[str] = None
    final_board: Optional[str] = None
    git_sha: Optional[str] = None
    hostname: Optional[str] = None
    started_at: Optional[str] = None             # ISO-8601, passed in (§6)
    schema_version: int = SCHEMA_VERSION


@dataclass
class SeatRow:
    """Who sat where (§6 ``game_seats``) — attributes results to an opponent type."""

    seat: int
    is_hero: int
    agent_label: str
    agent_config: Optional[str] = None           # per-seat config JSON, provenance


@dataclass
class DecisionRow:
    """One hero search invocation (§6 ``decisions``).

    ``(regime, leaf_mode)`` is the solver approach; the solver-run block is what
    :class:`~poker_ai.search.solver.SearchResult` / ``SearchStats`` surface (§9.1)
    — the caller unpacks them here so this module keeps no search dependency.
    ``exploitability`` / ``game_value`` stay NULL until that evaluator exists.
    """

    betting_stage: str
    regime: str                                  # 'mccfr' | 'vector' | 'blueprint'
    searched: int                                # 0/1: search fired, or blueprint
    leaf_mode: Optional[str] = None
    is_research: Optional[int] = None            # 1 if an off-tree re-search
    num_live: Optional[int] = None
    pot_before: Optional[float] = None
    to_call: Optional[float] = None
    hero_stack: Optional[float] = None
    iterations: Optional[int] = None
    wall_seconds: Optional[float] = None
    iters_per_sec: Optional[float] = None
    stop_reason: Optional[str] = None            # 'iteration_cap' | 'wall_cap'
    node_count: Optional[int] = None
    unique_pubkeys: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None
    action_played: Optional[str] = None
    action_dist: Optional[str] = None            # JSON: root action distribution
    exploitability: Optional[float] = None
    game_value: Optional[float] = None


@dataclass
class RangeQualityRow:
    """One (opponent seat, belief snapshot), resolved at showdown (§6, §7).

    Unresolved (folded-before-showdown) snapshots are logged with ``resolved=0``
    and the metric columns left NULL — kept for the resolved-fraction denominator
    (§8/§11), excluded from quality aggregates.
    """

    seat: int
    betting_stage: str
    resolved: int
    n_actions_replayed: Optional[int] = None
    true_combo: Optional[int] = None
    true_combo_mass: Optional[float] = None
    true_combo_rank: Optional[float] = None
    effective_support: Optional[int] = None
    log_loss: Optional[float] = None
    log_loss_uniform: Optional[float] = None
    net_info_gain: Optional[float] = None
    collapsed_truth: Optional[int] = None
    uniform_fallback: Optional[int] = None
    entropy: Optional[float] = None


@dataclass
class HandFailureRow:
    """One hand that raised while being played (§6 ``hand_failures``, §9.3).

    ``deck_seed`` / ``agent_seed`` are carried so the failing hand can be replayed
    exactly for debugging; ``error`` / ``traceback`` are the diagnostic.
    """

    run_id: str
    hand_index: int
    deck_seed: Optional[int] = None
    agent_seed: Optional[int] = None
    hero_seat: Optional[int] = None
    error_type: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    git_sha: Optional[str] = None
    hostname: Optional[str] = None
    failed_at: Optional[str] = None


def _insert(con: sqlite3.Connection, table: str, row, **extra) -> int:
    """Insert a row-dataclass into ``table``; return the new rowid.

    Columns and values come from the dataclass fields (names must match the DDL);
    ``extra`` injects columns not on the dataclass — ``game_id`` for the child
    tables, taken from the parent ``games`` insert.
    """
    data = dataclasses.asdict(row)
    data.update(extra)
    cols = list(data)
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    cur = con.execute(sql, [data[c] for c in cols])
    return int(cur.lastrowid)


class ExperimentLog:
    """A single-writer SQLite sink for one evaluation run (§4)."""

    def __init__(self, con: sqlite3.Connection) -> None:
        self._con = con

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, path) -> "ExperimentLog":
        """Open (or create) the DB at ``path`` with the §4 pragmas + §6 schema.

        Autocommit mode (``isolation_level=None``) so the schema DDL and
        :meth:`snapshot` run outside any transaction and :meth:`game` owns the
        per-hand transaction explicitly.  ``path`` is the **node-local** file
        during a run (§5); the permanent-FS copy is produced by :meth:`snapshot`.
        """
        con = sqlite3.connect(str(path), isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.executescript(_SCHEMA_DDL)
        return cls(con)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "ExperimentLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Per-hand transaction (§4 — the crash-safety + write-batching boundary)
    # ------------------------------------------------------------------

    @contextmanager
    def game(self) -> Iterator[None]:
        """Transaction context for one hand: ``BEGIN`` … ``COMMIT``.

        Rolls back on any exception, so a killed run never leaves a half-written
        hand (the resume cursor of §10.1 relies on this all-or-nothing property).
        All ``log_*`` calls for the hand must run inside this block.
        """
        self._con.execute("BEGIN")
        try:
            yield
        except BaseException:
            self._con.execute("ROLLBACK")
            raise
        else:
            self._con.execute("COMMIT")

    # ------------------------------------------------------------------
    # Writers (§6 grains)
    # ------------------------------------------------------------------

    def log_game(self, row: GameRow) -> int:
        """Insert the ``games`` row; return its ``game_id`` for the child rows."""
        return _insert(self._con, "games", row)

    def log_seats(self, game_id: int, rows: Iterable[SeatRow]) -> None:
        """Insert the per-seat rows for ``game_id`` (§6 ``game_seats``)."""
        for row in rows:
            _insert(self._con, "game_seats", row, game_id=game_id)

    def log_decision(self, game_id: int, row: DecisionRow) -> int:
        """Insert one ``decisions`` row; return its ``decision_id``."""
        return _insert(self._con, "decisions", row, game_id=game_id)

    def log_range_quality(self, game_id: int, row: RangeQualityRow) -> int:
        """Insert one ``range_quality`` row; return its ``id``."""
        return _insert(self._con, "range_quality", row, game_id=game_id)

    def log_failure(self, row: HandFailureRow) -> int:
        """Insert one ``hand_failures`` row; return its ``id`` (§9.3).

        Its own autocommitted insert — a failure is recorded **after** the game's
        transaction has already rolled back (§4), so it must not be inside one.
        """
        return _insert(self._con, "hand_failures", row)

    # ------------------------------------------------------------------
    # Snapshot (§5 — the checkpoint-copy primitive)
    # ------------------------------------------------------------------

    def snapshot(self, dest) -> None:
        """``VACUUM INTO dest`` — a clean single-file snapshot, WAL-state safe.

        Unlike ``cp`` of the live ``.db`` (which is inconsistent while WAL frames
        are uncheckpointed), this yields a defragmented copy safe to run while the
        experiment keeps writing (§5).  Must be called **outside** a :meth:`game`
        transaction (autocommit); the runner uses it for the periodic and final
        sync-back to the permanent filesystem.
        """
        self._con.execute("VACUUM INTO ?", (str(dest),))

    # ------------------------------------------------------------------
    # Resume cursor (§10.1) — a read on the schema, so it belongs to the sink.
    # ------------------------------------------------------------------

    def next_hand_index(self, run_id: str) -> int:
        """The ``hand_index`` to resume ``run_id`` at: ``1 + max``, else 0.

        The per-hand transaction makes each hand fully logged or absent, so the
        max completed ``hand_index`` is an exact cursor — a restarted run reads it
        and continues from the next (§10.1).  Failed hands (``hand_failures``) count
        too: a hand that raises is deterministic, so advancing past it avoids
        retrying the same failure on every restart.
        """
        cur = self._con.execute(
            "SELECT MAX(hand_index) FROM ("
            "  SELECT hand_index FROM games         WHERE run_id = :r "
            "  UNION ALL "
            "  SELECT hand_index FROM hand_failures WHERE run_id = :r"
            ")",
            {"r": run_id},
        )
        row = cur.fetchone()
        return 0 if row is None or row[0] is None else int(row[0]) + 1


def open(path) -> ExperimentLog:  # noqa: A001 — matches the doc's ``open(path)`` API
    """Module-level alias for :meth:`ExperimentLog.open` (doc §9.2 sketch)."""
    return ExperimentLog.open(path)


__all__ = [
    "SCHEMA_VERSION",
    "ExperimentLog",
    "GameRow",
    "SeatRow",
    "DecisionRow",
    "RangeQualityRow",
    "HandFailureRow",
    "open",
]
