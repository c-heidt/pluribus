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
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional

# Bump on any schema change (games.schema_version); lets analysis span runs (§6).
# v2: + games.hu_from_street (HU coverage for OX-Search-HU, opponent-modeling doc §11.4).
# v3: + games.condition (experiment arm for cross-condition CRN pairing, §10.1).
# v5: + games.opponent_models + decisions.modeled_decision (A7 model provenance +
#     coverage-restricted slicing, opponent-modeling doc §9).
# v6: + decisions.n_live (live-range count the solver sized its budget on — the
#     calibration axis for per-street/per-live-count throughput & budget).
# v7: + decisions.ox_enter_prob (OX-Search Approach B opt-out saturation; NULL for
#     vanilla/DBR and non-vector subgames, so a non-NULL row is a genuine OX decision).
# v8: + decisions.raise_level (the env action grid's second axis — 0 when no raise
#     has gone in this round, 1 once one has, i.e. `poker_env.raise_level`, which
#     is the `first_raise` / `subsequent_raise` split RAISE_SIZES_BY_STAGE is cut
#     on; lets the summary report the played action mix at exactly the grid's
#     (stage, level) granularity).
SCHEMA_VERSION = 8


# ---------------------------------------------------------------------------
# Schema (§6) — faithful to the doc's DDL, with IF NOT EXISTS for reopen/resume.
# ---------------------------------------------------------------------------

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS games (
    game_id            INTEGER PRIMARY KEY,
    run_id             TEXT    NOT NULL,
    condition          TEXT,
    opponent_models    TEXT,
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
    hu_from_street     INTEGER,
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
    searched        INTEGER NOT NULL,
    is_research     INTEGER,
    num_live        INTEGER,
    n_live          INTEGER,   -- live ranges the solver sized its budget on (calibration axis)
    pot_before      REAL,
    to_call         REAL,
    hero_stack      REAL,
    iterations      INTEGER,
    wall_seconds    REAL,
    iters_per_sec   REAL,
    stop_reason     TEXT,      -- 'iteration_cap' | 'wall_cap'
    node_count      INTEGER,
    unique_pubkeys  INTEGER,
    cache_hits      INTEGER,
    cache_misses    INTEGER,
    action_played   TEXT,
    action_dist     TEXT,
    raise_level     INTEGER,   -- action-grid level: 0 = first_raise, 1 = subsequent_raise
    exploitability  REAL,
    game_value      REAL,
    modeled_decision INTEGER,
    ox_enter_prob   REAL      -- OX-Search opt-out saturation; NULL off the gadget
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

"""

# Indexes are created *after* the additive column migrations in :meth:`open`, since
# an index may reference a column that an older DB lacks until the ALTER runs
# (e.g. ``idx_games_condition`` on the v3 ``condition`` column).
_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_games_table     ON games(table_label);
CREATE INDEX IF NOT EXISTS idx_games_run       ON games(run_id, hand_index);
CREATE INDEX IF NOT EXISTS idx_games_pairing   ON games(pairing_id);
CREATE INDEX IF NOT EXISTS idx_games_deck      ON games(deck_seed);      -- cross-condition CRN join (§10.1)
CREATE INDEX IF NOT EXISTS idx_games_condition ON games(condition);
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
    condition: Optional[str] = None              # experiment arm for CRN pairing (§10.1)
    opponent_models: Optional[str] = None        # JSON: seats modeled + spec, else NULL (A7)
    agent_seed: Optional[int] = None
    aivat_value: Optional[float] = None          # filled once AIVAT exists (§10.2)
    pairing_id: Optional[int] = None
    variant: Optional[str] = None
    hero_chips_delta: Optional[float] = None     # raw primary outcome (chips)
    went_to_showdown: Optional[int] = None
    terminal_street: Optional[str] = None
    hu_from_street: Optional[int] = None         # earliest round start HU-with-hero
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

    ``regime`` is the solver approach; the solver-run block is what
    :class:`~poker_ai.search.solver.SearchResult` / ``SearchStats`` surface (§9.1)
    — the caller unpacks them here so this module keeps no search dependency.
    ``exploitability`` / ``game_value`` stay NULL until that evaluator exists.
    """

    betting_stage: str
    regime: str                                  # 'mccfr' | 'vector' | 'blueprint'
    searched: int                                # 0/1: search fired, or blueprint
    is_research: Optional[int] = None            # 1 if an off-tree re-search
    num_live: Optional[int] = None               # table-active seats at the node
    n_live: Optional[int] = None                 # live ranges the solver sized on (searched rows)
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
    raise_level: Optional[int] = None            # action-grid level at the node (clamped)
    exploitability: Optional[float] = None
    game_value: Optional[float] = None
    modeled_decision: Optional[int] = None       # 1 iff a modeled solve produced this play (A7)
    ox_enter_prob: Optional[float] = None        # OX-Search opt-out saturation; None off the gadget


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
        # Additive migrations for DBs created before a column existed —
        # ``CREATE TABLE IF NOT EXISTS`` does not extend an existing table, and
        # the name-based ``_insert`` would then fail on the new column.  Rows
        # written before the migration keep NULL (schema_version tells them
        # apart, §6).
        have = {r[1] for r in con.execute("PRAGMA table_info(games)")}
        if "hu_from_street" not in have:  # v1 → v2
            con.execute("ALTER TABLE games ADD COLUMN hu_from_street INTEGER")
        if "condition" not in have:       # v2 → v3
            con.execute("ALTER TABLE games ADD COLUMN condition TEXT")
        if "opponent_models" not in have:  # v4 → v5 (A7 model provenance)
            con.execute("ALTER TABLE games ADD COLUMN opponent_models TEXT")
        dhave = {r[1] for r in con.execute("PRAGMA table_info(decisions)")}
        # (A retired v3 → v4 migration added ``term_runout``/``term_payout`` — a
        # per-terminal scalar-evaluator mix that the vectorized walk no longer
        # produces; the columns are gone from the DDL and left untouched where an
        # older DB already has them, since ``asdict``-driven inserts simply skip them.)
        if "modeled_decision" not in dhave:  # v4 → v5 (A7 coverage flag)
            con.execute("ALTER TABLE decisions ADD COLUMN modeled_decision INTEGER")
        if "n_live" not in dhave:  # v5 → v6 (calibration axis)
            con.execute("ALTER TABLE decisions ADD COLUMN n_live INTEGER")
        if "ox_enter_prob" not in dhave:  # v6 → v7 (OX-Search opt-out saturation)
            con.execute("ALTER TABLE decisions ADD COLUMN ox_enter_prob REAL")
        if "raise_level" not in dhave:  # v7 → v8 (action-grid level for the action mix)
            con.execute("ALTER TABLE decisions ADD COLUMN raise_level INTEGER")
        # Indexes last — after the ALTERs, so an index on a freshly-migrated column
        # (idx_games_condition) has its column to reference.
        con.executescript(_INDEX_DDL)
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

    def sync_to(self, dest) -> None:
        """Atomically refresh the permanent-FS snapshot at ``dest`` (§5 sync-back).

        The periodic + on-SIGTERM sync-back writes the **same** permanent path
        repeatedly, but ``VACUUM INTO`` refuses to overwrite an existing file, so
        it cannot target ``dest`` directly.  Snapshot to a sibling temp file and
        ``os.replace`` it over ``dest`` — atomic on one filesystem, so an analysis
        reader (or a crash mid-sync) never sees a half-written snapshot and the
        previous good snapshot survives until the new one is complete.  The temp is
        a sibling of ``dest`` so the replace stays on the destination filesystem
        (a cross-device ``os.replace`` would raise).
        """
        dest = os.fspath(dest)
        tmp = f"{dest}.tmp.{os.getpid()}"
        if os.path.exists(tmp):
            os.remove(tmp)  # a stale temp from a killed prior sync (pid reuse)
        self.snapshot(tmp)
        os.replace(tmp, dest)

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

    def completed_hand_indices(self, run_id: str) -> "set[int]":
        """The SET of ``hand_index`` already logged (games ∪ hand_failures) for ``run_id``.

        The **parallel** resume cursor.  Under the dynamic hand pool hands complete
        out of order, so the ``max + 1`` cursor of :meth:`next_hand_index` is unsafe
        (a hand still running when the process dies leaves a gap *below* the max that
        ``max + 1`` would skip).  The parallel runner instead enqueues the *complement*
        of this set up to the target count, so gaps are refilled and nothing is
        double-played.  The per-hand transaction keeps each hand all-or-nothing, so the
        set is exact.
        """
        cur = self._con.execute(
            "SELECT hand_index FROM games         WHERE run_id = :r "
            "UNION "
            "SELECT hand_index FROM hand_failures WHERE run_id = :r",
            {"r": run_id},
        )
        return {int(r[0]) for r in cur.fetchall()}


# Tables merged from per-worker DBs, in FK-dependency order, with the surrogate-key
# columns each one shifts by the worker's disjoint base (see :func:`merge_logs`).
_MERGE_SHIFT = {
    "games": {"game_id"},
    "game_seats": {"game_id"},                 # composite PK (game_id, seat)
    "decisions": {"decision_id", "game_id"},
    "range_quality": {"id", "game_id"},
    "hand_failures": {"id"},
}
_MERGE_ORDER = ("games", "game_seats", "decisions", "range_quality", "hand_failures")


def merge_logs(target_path, worker_paths, *, base_block: int = 1 << 40) -> None:
    """Merge per-worker node-local DBs into ``target_path`` (disjoint-range, resume-safe).

    Each worker writes a vanilla DB (autoincrement ids from 1) — the hot insert path is
    untouched.  The merge shifts worker k's surrogate ids (and the FKs that reference
    them) into a **disjoint block** ``free_base + k · base_block``, where ``free_base``
    is a block boundary **above the target's current max id**.  So (a) workers never
    collide with each other, (b) a resumed run never collides with a prior attempt
    already merged into the target, and (c) every ``game_id`` FK still points at its
    shifted parent (parent + child shift by the same base).  ``base_block`` (2^40) is
    far larger than any per-attempt-per-worker row count, so blocks never overlap.

    The target is opened with the normal schema/migrations, so it is created if absent.
    Foreign keys are disabled for the bulk copy (rows are inserted parent-first anyway;
    the shift preserves referential integrity) and re-enabled after.
    """
    tgt = open(target_path)
    con = tgt._con
    try:
        row = con.execute(
            "SELECT MAX(m) FROM ("
            "  SELECT MAX(game_id)     AS m FROM games "
            "  UNION ALL SELECT MAX(decision_id) FROM decisions "
            "  UNION ALL SELECT MAX(id)          FROM range_quality "
            "  UNION ALL SELECT MAX(id)          FROM hand_failures)"
        ).fetchone()
        max_id = int(row[0]) if row and row[0] is not None else 0
        free_base = ((max_id // base_block) + 1) * base_block
        con.execute("PRAGMA foreign_keys=OFF")
        try:
            for k, wpath in enumerate(worker_paths):
                base = free_base + k * base_block
                con.execute("ATTACH ? AS w", (str(wpath),))
                try:
                    con.execute("BEGIN")
                    for table in _MERGE_ORDER:
                        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
                        shift = _MERGE_SHIFT[table]
                        sel = ", ".join(
                            (f"({c} + {base})" if c in shift else c) for c in cols
                        )
                        con.execute(
                            f"INSERT INTO main.{table} ({', '.join(cols)}) "
                            f"SELECT {sel} FROM w.{table}"
                        )
                    con.execute("COMMIT")
                except BaseException:
                    con.execute("ROLLBACK")
                    raise
                finally:
                    con.execute("DETACH w")
        finally:
            con.execute("PRAGMA foreign_keys=ON")
    finally:
        tgt.close()


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
