# Evaluation & Experiment Logging

Design for the offline evaluation harness of the real-time search component
([docs/subgame_solving.md](subgame_solving.md)). Where the subgame document
specifies *how the bot plays*, this document specifies *how we measure whether
it plays well* and *how the measurements are stored for later analysis*.

This document is deliberately staged. The **logging backbone** (§4–§7) and the
**end-of-run summary** (§8) are the first things to build and are fully specified
here. The **evaluation runner** (§10.1, time-budgeted games vs. blueprint-derived
bots) and the **AIVAT** variance-reduced strength estimate (§10.2) are planned in
detail but built after the backbone; genuinely-future work (heterogeneous opponents,
exploitation solver, exploitability) is listed in §10.3.

---

## 1. Purpose and Scope

The goal is to run **thousands of games against varying opponent
configurations** on a single compute node and analyse the results in aggregate:
strength by opponent, cost of search, which solver approach ran where, and the quality
of the belief (range) tracking that feeds it.

The workload is **write-once, read-analytically-many**: a long sequential run
produces one record per game and many records per game (one per search
invocation and belief update), and analysis happens afterwards with grouped
queries ("mean chip delta by opponent config", "wall-clock by betting stage",
"how often does range tracking rule out the true hand?").

Games are played **sequentially on one node** so that all cores go to *solving*
each subgame (§6.7 of the subgame document) rather than to running many games in
parallel. This single-writer topology is what makes the storage decision in §4
straightforward.

## 2. Goals and Non-Goals

### Goals

- A structured, queryable record of every evaluation game and every search
  invocation within it, comparable across runs and opponent configurations.
- Capture of the solver-run metadata that is currently computed and discarded
  (iterations, wall-clock, cache behaviour) plus problem-shape metadata that is
  not computed today (tree size, node count) — see [the audit note](#appendix-a-current-logging-state).
- Capture of **range-tracking quality** (§7): whether the tracked belief over
  each opponent's hole cards resembles the hand that is actually revealed at
  showdown. A belief that diverges from reality hurts the solve more than it
  helps, and today nothing measures this.
- A storage format and cluster I/O strategy that survive long runs and
  preemption without corrupting data or dominating wall-clock.

### Non-Goals

- **Exploitability / best-response** is explicitly **deferred** (see subgame
  document §9). It is a *computation* to build (no best-response walker exists in
  the repo yet), not merely a metric to log, and it is expensive enough to belong
  behind its own eval flag rather than the real-time loop. The schema (§6)
  reserves nullable columns for it so it can be added without migration.
- No live/interactive dashboards or per-iteration console verbosity. The subgame
  runs already log human-readable status via the repository's `RichHandler`
  config ([poker_ai/__init__.py](../poker_ai/__init__.py)); this document adds a
  *machine-readable* sink alongside it, not a replacement for it.
- No opponent modelling. Varying "opponent configuration" here means varying the
  *opponent agents* we evaluate against — at the start, the blueprint and its
  fold/call/raise-biased variants (§10.1); stronger opponents drop in later (§10.3) —
  not adapting the bot to the opponent.
- No new heavy dependencies (no wandb / tensorboard / mlflow). The chosen sink is
  a single SQLite file from the standard library.

## 3. What We Measure

The game is **6-max**: one hero (the agent) and **five opponent seats**, which may
run *different* agents. That makes per-seat opponent identity a first-class grain,
not a heads-up afterthought.

There are **four natural grains**. Keeping them separate (rather than flattening
into one wide row per decision) avoids repeating table state on every row and keeps
"group by opponent" unambiguous.

| Grain | One row per | Carries |
|---|---|---|
| **Game / hand** | hand played | table-composition label, positions, seed, big blind, stack depth, provenance, hero chip outcome |
| **Seat** | (game, seat) | which agent sat there — attributes results/range-quality to an opponent *type* |
| **Decision / solve** | search invocation within a game | regime, iterations, wall-clock, stop reason, cache stats, node/tree size, `num_live`, action played |
| **Range quality** | (game, opponent seat, belief snapshot) resolved at showdown | true-combo mass, rank, log-loss vs. uniform, collapse flag, entropy |

All join on `game_id`; the seat and range-quality tables additionally carry `seat`.
Exploitability, when built, attaches at the decision grain as nullable columns.

## 4. Storage Format Decision

**Decision: a single SQLite database file, written node-locally.**

The single-writer, sequential-play topology (§1) removes the only reason to
prefer append-only text files: there is no concurrent-writer contention to avoid.
That makes SQLite the cleanest fit for a *read-analytically-many* workload —
real SQL joins across the four grains, indexes on the columns we group by, and
zero analysis tooling (pandas `read_sql`, DuckDB `ATTACH`, or the `sqlite3` CLI
all read the file directly).

Settings:

- **WAL mode** (`PRAGMA journal_mode=WAL`) — lets an analysis query read the
  file while a run is still writing.
- **`PRAGMA synchronous=NORMAL`** — skips the per-commit `fsync` (syncing only at
  checkpoint). On node-local disk this keeps logging cost negligible against the
  seconds-to-tens-of-seconds each search takes, while remaining safe against
  database *corruption*; the only exposure is losing the last transaction on an
  OS/power-level crash.
- **One transaction per game** — a game's `games` row and all of its `game_seats`,
  `decisions`, and `range_quality` rows commit together. This is both the crash-safety
  boundary (a killed run never leaves a half-written game) and the write-batching
  mechanism (one commit per game, not per decision), so no in-application
  buffering across games is needed or wanted — buffering would only move the
  durability boundary out and risk losing more on a crash.

### Rejected alternatives

| Option | Why not |
|---|---|
| **CSV** | Cannot hold the nested action distribution / opponent-config blobs; untyped. |
| **JSON-lines → Parquet** | The right choice *only* under parallel writers (append-only, contention-free). With a single writer it adds a compaction step and loses in-file SQL joins for no benefit. |
| **Parquet as the write sink** | Immutable/batch-oriented; appending one record per solve is awkward. It is an *analysis* format, not a *capture* format. |
| **SQLite on the network filesystem** | Network FS locking is unreliable and a known route to SQLite corruption, on top of per-transaction latency. Avoided by writing node-locally (§5). |

## 5. Cluster Execution & Durability

The database is written to **node-local scratch** (`$TMPDIR` / local SSD) during
the run and **synced back to the permanent filesystem** on an interval and at the
end — mirroring the blueprint training approach (LUTs staged local, checkpoints
written local, synced back; see [scripts/training.sh](../scripts/training.sh)).
Writing to the shared filesystem per hand would be slow and, for SQLite,
unsafe (§4). The SLURM wrapper that stages scratch and forwards SIGTERM is
[scripts/evaluation.sh](../scripts/evaluation.sh).

- **Snapshot with `VACUUM INTO`, not `cp`.** In WAL mode a plain copy of the main
  `.db` file while uncheckpointed WAL frames exist yields an inconsistent copy.
  `VACUUM INTO 'permanent/…/experiment.sqlite'` produces a clean, single-file,
  defragmented snapshot regardless of WAL state, and is safe to run while the
  experiment keeps writing. This is the checkpoint-copy primitive
  (`ExperimentLog.snapshot`).
- **The sync-back is atomic-replace, because `VACUUM INTO` refuses to overwrite.**
  The periodic sync-back rewrites the *same* permanent path every interval, but
  `VACUUM INTO` errors if its target file already exists. `ExperimentLog.sync_to`
  therefore snapshots to a sibling temp file (`dest.tmp.<pid>`) and `os.replace`s it
  over `dest` — atomic on one filesystem, so an analysis reader (or a crash
  mid-sync) never sees a half-written snapshot and the previous good snapshot
  survives until the new one is complete. A failed `VACUUM INTO` leaves `dest`
  untouched; the stale temp is cleaned on the next attempt.
- **Periodic sync-back**, not only at the end. A `VACUUM INTO` to the permanent
  filesystem every *N* games (`sync_interval_hands`, default 500) or *T* minutes
  (`sync_interval_minutes`) bounds how much a node failure or preemption can cost.
  Because it is one file, the copy is cheap — none of the many-small-files overhead
  of the chunked table checkpoints. Periodic syncs are **best-effort** (a transient
  FS hiccup is logged and retried next interval, never fatal — bounded loss).
- **Final sync on SIGTERM / preemption.** The runner polls a SIGTERM/SIGINT event
  at each hand boundary (the signal-handling pattern already in the blueprint
  runners, [poker_ai/blueprint/multiprocess/server.py](../poker_ai/blueprint/multiprocess/server.py),
  and `training.sh`); on stop it finishes the current hand's transaction and runs a
  final `VACUUM INTO` so nothing since the last periodic snapshot is lost. The final
  sync is **strict** — it *is* the run's product, so a failure there propagates
  rather than silently leaving stale permanent data.
- **The blueprint and LUT are both staged node-local, not just the DB.** Real-time
  search hits the blueprint on the hot path (its per-street LMDB info-set index is
  queried on every leaf-fleet / opponent / hero lookup), so `evaluation.sh` rsyncs
  both artifacts to `$TMPDIR`. This is a real resource footprint: the LUT (~250 GB)
  and blueprint (~150 GB) together need **~420 GB of node-local disk** (`--tmp`),
  and — because `CFRTables` restores the blueprint's regret/strategy chunks into
  `/dev/shm` (tmpfs) — **~150 GB of the blueprint is resident in RAM** (`--mem`),
  on top of the solver working set and LUT page cache. The wrapper runs a preflight
  free-space check and aborts before staging if the node cannot hold both.
- **Analysis reads the permanent-FS snapshot**, never the live node-local file.

## 6. Schema

Concrete DDL sketch (types are SQLite affinities; JSON blobs are stored as `TEXT`
and queried with `json_extract` / `->>` when needed).

```sql
-- One row per HAND (one deal). bb/100 is per-hand; "game" == "hand" here.
CREATE TABLE games (
    game_id            INTEGER PRIMARY KEY,
    run_id             TEXT    NOT NULL,   -- groups games of one experiment batch
    condition          TEXT,               -- experiment arm: 'vanilla' (baseline: search, no
                                           --   model) | 'DBR(p_max=...)' | 'blueprint_only' (no-search
                                           --   test) — the cross-condition GROUP BY; NULL for single-arm (§10.1)
    hand_index         INTEGER NOT NULL,   -- 0-based position within the run; the resume cursor (§10.1)
    schema_version     INTEGER NOT NULL,   -- bump on schema change; lets analysis span runs
    config_fingerprint TEXT    NOT NULL,   -- hash(solver + leaf + table composition)
    table_label        TEXT    NOT NULL,   -- table-composition label, the top-line GROUP BY key
    table_config       TEXT    NOT NULL,   -- full JSON blob (all seats' agents), for provenance
    hero_seat          INTEGER NOT NULL,
    button_seat        INTEGER NOT NULL,   -- position: hero_position derivable from the two
    n_players          INTEGER NOT NULL,   -- 6 here; kept for generality
    -- normalisation / slicing covariates -------------------------------------
    big_blind          REAL    NOT NULL,   -- BB in chips; normalises delta to bb/100
    starting_stack     REAL    NOT NULL,   -- effective stack in chips; /big_blind = depth in BB
    -- reproducibility --------------------------------------------------------
    deck_seed          INTEGER NOT NULL,   -- reproduces the deal; also the cross-condition CRN
                                           --   join key (shared run_seed ⇒ equal deck_seed per hand, §10.1)
    agent_seed         INTEGER,            -- reproduces the search sampling (MCCFR/vector RNG)
    -- variance reduction -----------------------------------------------------
    aivat_value        REAL,               -- AIVAT-adjusted hero outcome (§10.2); the low-variance
                                           --   estimator once built; unbiased, ~same mean as
                                           --   hero_chips_delta with far tighter CI
    pairing_id         INTEGER,            -- seat-rotation set (optional; AIVAT largely subsumes it)
    variant            TEXT,               -- rotation index within the set
    -- outcome ----------------------------------------------------------------
    hero_chips_delta   REAL,               -- raw primary outcome (chips); always logged
    went_to_showdown   INTEGER,            -- 1/0; gates unbiased range-quality resolution (§7)
    terminal_street    TEXT,               -- where the hand ended (preflop..river)
    final_pot          REAL,
    hero_hole          TEXT,               -- e.g. 'Ah Kd'; win rate by starting hand
    final_board        TEXT,               -- board-texture slicing
    -- provenance -------------------------------------------------------------
    git_sha            TEXT,
    hostname           TEXT,               -- which node produced this (cluster debugging)
    started_at         TEXT                -- ISO-8601, passed in (not Date.now-style)
);

-- One row per seat per game: who sat where. The 6-max change — lets results and
-- range quality be attributed to an opponent *type*, not just the table label.
CREATE TABLE game_seats (
    game_id     INTEGER NOT NULL REFERENCES games(game_id),
    seat        INTEGER NOT NULL,
    is_hero     INTEGER NOT NULL,          -- 1 for the agent's seat, else 0
    agent_label TEXT    NOT NULL,          -- opponent type at this seat, the GROUP BY key
    agent_config TEXT,                     -- per-seat config JSON, provenance
    PRIMARY KEY (game_id, seat)
);

-- One row per HERO search invocation within a game. Only the agent is logged;
-- opponents are simple bots that do not solve, so there is nothing to log for them.
CREATE TABLE decisions (
    decision_id     INTEGER PRIMARY KEY,
    game_id         INTEGER NOT NULL REFERENCES games(game_id),
    betting_stage   TEXT    NOT NULL,      -- preflop/flop/turn/river
    regime          TEXT    NOT NULL,      -- 'mccfr' | 'vector' | 'blueprint' — IS the solver approach
    searched        INTEGER NOT NULL,      -- 0/1: did search fire, or blueprint play?
    is_research     INTEGER,               -- 1 if a re-search triggered by an off-tree action
    num_live        INTEGER,               -- players still in the hand (multiway vs HU slicing)
    -- decision context (bet-size / spot analysis) ----------------------------
    pot_before      REAL,
    to_call         REAL,
    hero_stack      REAL,
    -- solver run -------------------------------------------------------------
    iterations      INTEGER,
    wall_seconds    REAL,
    iters_per_sec   REAL,
    stop_reason     TEXT,                  -- 'iteration_cap' | 'wall_cap' (the two budget caps)
    node_count      INTEGER,               -- decision nodes visited (new counter)
    unique_pubkeys  INTEGER,               -- distinct public_key (new counter)
    cache_hits      INTEGER,               -- leaf/runout/legal_actions caches
    cache_misses    INTEGER,
    action_played   TEXT,                  -- the sampled action
    action_dist     TEXT,                  -- JSON: root action distribution
    exploitability  REAL,                  -- NULL until the exploitability evaluator exists (§10.3)
    game_value      REAL                   -- NULL until then
);

-- One row per (opponent seat, belief snapshot), resolved at showdown (§7).
CREATE TABLE range_quality (
    id                INTEGER PRIMARY KEY,
    game_id           INTEGER NOT NULL REFERENCES games(game_id),
    seat              INTEGER NOT NULL,
    betting_stage     TEXT    NOT NULL,    -- stage at which the belief was held
    n_actions_replayed INTEGER,            -- belief updates folded in; error compounds with it
    true_combo        INTEGER,             -- revealed hand index; slice quality by hand class
    true_combo_mass   REAL,                -- belief weight on the revealed hand
    true_combo_rank   REAL,                -- percentile of the true combo (0..1)
    effective_support INTEGER,             -- nonzero combos; the uniform baseline's size
    log_loss          REAL,                -- -log(true_combo_mass)
    log_loss_uniform  REAL,                -- -log(1/effective_support): the baseline
    net_info_gain     REAL,                -- log_loss_uniform - log_loss (>0 = helped)
    collapsed_truth   INTEGER,             -- 1 if belief assigned ~0 to the truth
    uniform_fallback  INTEGER,             -- 1 if _uniform_fallback fired this hand/seat
    entropy           REAL,                -- concentration of the belief
    resolved          INTEGER NOT NULL     -- 1 if truth was observed (showdown), else 0
);

CREATE INDEX idx_games_table      ON games(table_label);
CREATE INDEX idx_games_run        ON games(run_id, hand_index);   -- resume cursor lookup
CREATE INDEX idx_games_pairing    ON games(pairing_id);   -- variance-reduced pairing joins
CREATE INDEX idx_seats_agent      ON game_seats(agent_label);  -- per-opponent-type slices
CREATE INDEX idx_seats_lookup     ON game_seats(game_id, seat);
CREATE INDEX idx_decisions_game   ON decisions(game_id);
CREATE INDEX idx_decisions_stage  ON decisions(betting_stage);
CREATE INDEX idx_range_game       ON range_quality(game_id);
```

Notes:

- **Store the table config both ways** — a short `table_label` to `GROUP BY` for
  the top line, and the full `table_config` JSON for provenance ("what exactly was
  table B?"). Per-seat detail lives in `game_seats` so results attribute to the
  opponent *type* at each seat, not just the table as a whole.
- **`config_fingerprint`** is a stable hash of the solver config
  (`SolverConfig` + `LeafConfig`) and the `table_policy`, so records group across
  runs even as settings are tweaked over time. This identity does not exist today
  and must be added.
- **Do not store full belief vectors** in `range_quality`. `n_combos` is ~1300;
  thousands of games × seats × streets × 1300 floats would bloat the file for no
  analytical gain. Store the derived scalars. (Optionally dump the full vector for
  a small *sampled* subset of hands to a side file for debugging.)
- Timestamps are **passed in** by the runner, not generated inside any workflow
  context that forbids wall-clock reads.

The covariate columns exist to make the common slices one query each:

- **Grain is one hand per row** (`game_id` == a single deal). bb/100 is a per-hand
  rate; a match is just a `run_id` group. This is stated so nobody double-counts.
- **Position** (`hero_seat`, `button_seat`) — poker results are strongly
  position-dependent; without it, per-opponent win rates blur across positions. Log
  both seats and derive `hero_position` in analysis.
- **Effective stack depth** (`starting_stack / big_blind`) — as fundamental a slice
  as the opponent itself; strategy and win rate shift sharply with depth. Logged in
  chips, normalised at query time (same pattern as `big_blind`).
- **Variance reduction is AIVAT** (`aivat_value`, §10.2) — the planned low-variance
  estimator. It is unbiased (same mean as `hero_chips_delta`) with a far tighter CI,
  and it works on *singly-played* hands, so it needs no deck replay. `hero_chips_delta`
  is always logged raw; `aivat_value` is filled once the estimator exists (nullable
  until then). **Seat rotation** (`pairing_id`, `variant`) is kept as an *optional*
  fallback reducer — a rotation set replayed from rotated seats, averaged as the
  sample unit — but AIVAT largely subsumes it, so the runner (§10.1) may skip
  implementing rotation entirely. Both columns nullable.
- **Cross-run key is `(run_id, game_id)`, not `game_id` alone.** `game_id` is an
  autoincrement surrogate, unique only *within* one snapshot file — every run's
  `games` restarts at 1. Merging snapshots (ATTACH/UNION across per-run files, §11)
  on `game_id` alone collides and silently cross-joins run A's `decisions` onto run
  B's game 1. Always carry `run_id` and treat `(run_id, game_id)` as the logical key;
  the merge step namespaces or remaps `game_id` on import. This is what makes pooling
  runs (and the future exploitation-solver comparison, §10.3) safe.
- **Reproducibility needs two seeds** — `deck_seed` fixes the deal, `agent_seed`
  fixes the search sampling (MCCFR/vector are stochastic). Both are required to
  replay a surprising hand exactly.
- **Per-seat opponent identity** is handled by `game_seats` (above): join it to
  `range_quality.seat` to attribute belief quality to the opponent type at that
  seat, and to `decisions`/positions to break strength down by *who* the hero faced
  where. This is what makes heterogeneous 6-max tables analysable rather than a
  single opaque "opponent".
- **Multiway is the default, not the exception.** With five opponents most hands
  see folds before showdown, so `range_quality.resolved` fires for fewer seats and
  the resolved fraction runs lower than heads-up — report it prominently (§8) and
  treat range aggregates as showdown-conditional (§11). Where the engine allows,
  **force-reveal all live hands at showdown** in self-play so every seat that
  reaches it is resolvable, not just the pot winner.

## 7. Range-Tracking Quality

**Motivation.** The solver conditions on a per-combo belief over every live
seat's hole cards ([RangeTracker](../poker_ai/search/ranges.py)). If that belief
does not resemble the hand actually held, the subgame is solved against a
fiction: card-removal, leaf equities, and continuation choices are all skewed.
Beyond a point this *hurts more than it helps* versus a plain uniform prior. The
tracker already self-reports its worst failure — a belief that collapses below
the numerical floor and resets to uniform emits a `RuntimeWarning`
([ranges.py:245](../poker_ai/search/ranges.py#L245)) — but nothing quantifies the
common, quieter case of a belief that is merely *wrong*.

**Ground truth.** Opponent hole cards are revealed at **showdown**. At each belief
snapshot (a round-boundary update, [ranges.py:160](../poker_ai/search/ranges.py#L160)),
buffer the seat's belief vector in memory; when the hand ends and holes are
revealed, compute the metrics below against the true combo and write the
`range_quality` rows inside the game's transaction (§4). Folded hands that never
reach showdown are unverifiable — record them with `resolved = 0` and exclude
them from quality aggregates (see the sampling-bias caveat in §11).

**Metrics** (all derived from the belief vector `w` and the true combo index `h*`):

- **True-combo mass** `w[h*]` — belief weight on reality. Higher is better.
- **Log-loss** `-log(w[h*])` — penalises assigning low mass to the truth; infinite
  when the truth was zeroed. The single most informative scalar.
- **Log-loss vs. uniform** `-log(1/|support|)` — the belief the tracker *would*
  have had with no updates (uniform over board-compatible combos), i.e. the
  baseline it must beat.
- **Net information gain** = `log_loss_uniform − log_loss`. **This is the headline
  metric that answers the user's question**: averaged over resolved hands, a
  positive value means tracking helps, negative means it hurts. Break it down by
  `betting_stage` to see whether error compounds as more actions are replayed.
- **True-combo rank / percentile** — calibration-free ordering check: where does
  the true combo fall in the belief's sorted order? Robust when masses are tiny.
- **Collapsed-truth indicator** — did the belief assign ~0 to the true combo
  (`w[h*] < floor`)? This is the catastrophic case: the solve ruled out reality.
  Track its **rate**, not just its occurrence.
- **Uniform-fallback count** — how often `_uniform_fallback` fired
  ([ranges.py:244](../poker_ai/search/ranges.py#L244)); today only a warning, here
  a counted signal that the belief self-destructed.
- **Entropy / effective support** — concentration of the belief. A confidently
  *wrong* belief (low entropy, low true-combo mass) is the most damaging; a diffuse
  belief is close to the uniform prior and relatively harmless. Entropy contextualises
  the log-loss.

**Interpretation.** The evaluation should be able to answer, per opponent type
and per street: *is the mean net information gain positive?* If it is negative for
a given opponent/street, range tracking is net-harmful there and the mitigation
levers (belief flooring, earlier fallback, cluster-bucketed ranges — subgame doc
§10) come into play. This is the concrete decision the metric exists to inform.

The **end-of-run summary** (§8) surfaces the headline of this section — mean net
information gain per stage — so a net-harmful belief announces itself without
anyone querying the table.

## 8. Experiment Summary (end-of-run sanity check)

A single command — `python -m evaluation.summarize <snapshot.sqlite>` — runs
**automatically when the experiment ends** (as the last step of the runner and of
the cluster script, *after* the final `VACUUM INTO` sync-back in §5, reading the
permanent-filesystem snapshot, never the live node-local file). It prints a
compact human summary to the log and writes a `summary.json` next to the snapshot
for programmatic comparison across runs. Its job is a **quick sanity check** — "is
the bot winning, did search stay in budget, is range tracking helping?" — not deep
analysis.

It reads only the four tables (§6); every number below is one grouped query.

### Grain: the arm

A snapshot routinely holds several **arms** — a `condition` (`vanilla` /
`DBR(...)` / `OX(...)` / `blueprint_only`) crossed with a `table_label` (the
opponent mix). Under CRN the arms replay the *same deals*, so a snapshot with A
arms holds A `games` rows per deal.

Every number is therefore computed **within one arm**, and nothing is averaged or
counted across arms:

- A bb/100 pooled over `vanilla` and `DBR` rows answers no question — it is a
  mixture whose weights are an accident of how many hands each arm got.
- A pooled hand / decision / search count is just the deal count multiplied by A;
  reporting it as "hands" silently inflates the apparent experiment size.
- The one legitimate cross-arm number is the **CRN paired Δ** below. It is a
  *difference on the matched deal*, not a pool.

Two consequences worth stating outright, because both were once got wrong:

- **Deals are matched on `(table_label, deck_seed)`, not `deck_seed`.**
  `deck_seed` is a pure function of `(run_seed, hand_index)` and carries no table
  component (`runner.derive_seeds`), so a snapshot holding two table policies
  repeats every seed. Matched on the seed alone, every deal looks like a
  within-arm duplicate, is dropped as ambiguous, and the whole comparison silently
  reports zero pairs.
- **Firing and cost are per street, as rates, never as shares of a pooled total.**
  The regimes own disjoint streets (subgame §6.5), the iteration budget is a
  per-street constant, and wall differs by an order of magnitude between pre-flop
  and flop — so a regime's "share of searches" only restates how often each street
  arose, and a mean wall over all searches describes a mixture no solver ever ran
  at. The one sum that *is* meaningful is cost: total search seconds ÷ hands.

### What it prints

```
experiment  run_id=2026-07-01_6max_mix   git=d64a49b   6-max
  3000 deals × 2 conditions = 6000 game rows   (4 arms = 2 conditions × 2 tables)
  conditions: vanilla, DBR(p_max=0.6)
  tables:     all_blueprint, random_bias
──────────────────────────────────────────────────────────────────────────────
STRENGTH — per arm (aivat bb/100 ± 95% CI).  Arms are never pooled.
  table=all_blueprint
    vanilla             +14.8 ± 5.1     1500 hands   ✓ winning
      by position  BTN +31  CO +18  MP +6  UTG -9  SB -22  BB -14
    DBR(p_max=0.6)      +21.0 ± 5.2     1500 hands   ✓ winning
      by position  BTN +38  CO +24  MP +9  UTG -6  SB -19  BB -11
  table=random_bias
    vanilla              -1.9 ± 7.2     1500 hands   ~ inconclusive
    DBR(p_max=0.6)      +12.4 ± 7.1     1500 hands   ✓ winning

PAIRED Δ vs vanilla — CRN, matched on table_label + deck_seed (aivat bb/100, 95% bootstrap CI)
  DBR(p_max=0.6)   (3000 of 3000 deals matched)
    all_blueprint            +6.20  [+2.85, +9.61]     1500 pairs   ✓ better
      └ where it fired       +8.90  [+4.71, +13.02]     980 pairs   ✓ better
      matched-sample means: vanilla +14.8 → DBR(p_max=0.6) +21.0
    random_bias             +14.30  [+9.92, +18.71]    1500 pairs   ✓ better
      └ where it fired      +18.44  [+12.90, +23.85]   1121 pairs   ✓ better
      matched-sample means: vanilla -1.9 → DBR(p_max=0.6) +12.4
    ALL TABLES pooled       +10.25  [+7.48, +13.05]    3000 pairs   ✓ better
      matched-sample means: vanilla +6.5 → DBR(p_max=0.6) +16.7

RANGE TRACKING — per condition (net info gain vs the uniform prior, nats)
  vanilla   net gain +0.31 nats   resolved 18.4% of 22400 seat-snapshots   collapsed 3.1%   fallback 1.2%
      by opponent  bp +0.44   bp_call +0.05   bp_raise -0.12
      by street    flop +0.52   turn +0.21   river -0.06
  DBR(p_max=0.6)   net gain +0.29 nats   resolved 18.1% of 22610 seat-snapshots   collapsed 3.4%   fallback 1.3%
      by opponent  bp +0.41   bp_call +0.04   bp_raise -0.10
      by street    flop +0.49   turn +0.20   river -0.05

SEARCH — per condition × street (fire rate is within the street)
  vanilla   fired 68.0% of 8965 hero decisions (3.0/hand)   14.2s search per hand   routing OK ✓
      street  solver      fired       n    wall     p95  wall-cap   iters    it/s
      preflop mccfr          31%    2410    1.4s    2.1s        4%    3120    2230
      flop    mccfr          74%    2280    9.6s   14.7s       63%    5010     520
      turn    vector         86%    2160    3.1s    5.2s       12%    1000     320
      river   vector         89%    2115    1.2s    2.0s        3%     500     420

HU COVERAGE — heads-up with hero (OX-Search-HU fires from the turn on)
  vanilla   HU at some point 61.2%   eligible (turn+) 38.9%   of 3000 hands
      first HU street  preflop 8%  flop 14%  turn 22%  river 17%

FLAGS
  ⚠ range tracking net-harmful [vanilla] on river: -0.06 nats
  ⚠ search is budget-bound [vanilla]: wall-cap stops on flop 63%
  ⓘ resolved fraction 18.4% [vanilla] — range aggregates are weak evidence
```

The shape is the point. Every arm's absolute bb/100 here is a ±5–7 interval, but
the DBR edge itself is `+6.20 [+2.85, +9.61]` — a tighter interval than either arm's
own, because the paired difference cancels the shared card-luck. That is the number
the experiment is run to produce, and it exists only at the arm grain.

### The queries

**Strength — the headline.** Hero win rate in **bb/100** (chip delta normalised by
the big blind, ×100) with a 95% CI, **one row per arm**. Sign + CI answers "am I
winning, and is it significant?"; the arm grouping is what makes the answer belong
to something.

```sql
WITH g AS (                                  -- per-hand win rate in bb/100
    SELECT condition, table_label,           -- the arm: never grouped away
           100.0 * hero_chips_delta / big_blind AS bb100
    FROM games
)
SELECT condition, table_label,
       COUNT(*)                               AS hands,
       AVG(bb100)                             AS mean_bb100,
       1.96 * (  -- normal-approx 95% CI half-width
         SQRT( AVG(bb100*bb100) - AVG(bb100)*AVG(bb100) )
         / SQRT(COUNT(*)) )                    AS ci95
FROM g
GROUP BY condition, table_label;
```

`big_blind` is stored per hand (§6), so this holds even if the blind level varies.
A CI straddling zero → *inconclusive*, not *bad*; the summary labels each arm
`winning` / `losing` / `inconclusive` rather than leaving the reader to compare a
mean against its own `±`. There is deliberately **no cross-arm "overall" row** —
see "Grain: the arm" above. Two 6-max-specific breakdowns matter alongside the top
line:

- **By position** — group the same `bb100` by `hero_position` (derived from
  `hero_seat` − `button_seat` mod `n_players`), *within the arm*. In 6-max,
  aggregate strength hides large per-position swings; a positive arm with a
  bleeding blind defence is a real finding, not noise.
- **Variance reduction** — when `aivat_value` is populated (§10.2), report the CI on
  it instead of `hero_chips_delta`: same mean (AIVAT is unbiased), far tighter
  interval, so small edges become detectable. The summary prefers `aivat_value` when
  present and falls back to the raw query above when it is not. (Seat-rotation
  pairing is an optional secondary fallback, §6.)
- **Cross-condition paired difference (CRN)** — when comparing arms
  (`condition` = vanilla / DBR, §10.1), the headline is not each arm's absolute
  `bb100` but the **per-hand difference** between arms on the matched deal: join on
  `(table_label, deck_seed)` (see "Grain: the arm" — the seed alone is not unique
  across table policies), compute `Δ = aivat_value(DBR) − aivat_value(vanilla)` per
  hand, and bootstrap the CI on `mean(Δ)`. Because the shared card-luck cancels,
  this CI is dramatically tighter than differencing the two arms' independent means
  — it is what makes a small `DBR − vanilla` edge significant at ~10k hands. The Δ
  is reported per table (the opponent mix changes the size of the edge) plus a
  clearly-labelled pooled row, and carries both arms' **matched-sample** means so
  the difference can be read against the levels it came from. `n_paired` is printed
  against the arm sizes, and a comparison that matched few or no deals raises a
  flag: a silent "0 pairs" is the failure mode this section must never have.
  (The model-error *sweep* is the same Δ
  computed at each injected `SyntheticOpponentModel` error level — see
  [opponent_modeling.md](opponent_modeling.md) scope note; the old cumulative-count
  "learning curve" was removed with the online learner.)

**Search: firing, cost and budget health — per condition × street.** Are searches
firing where they should, and do they fit the budget? One grouped query, at the
grain the budget itself is defined on.

```sql
SELECT g.condition, d.betting_stage,
       COUNT(*)                                   AS n_decisions,
       CAST(SUM(d.searched) AS REAL) / COUNT(*)   AS fire_rate,   -- WITHIN the street
       AVG(CASE WHEN d.searched=1 THEN d.wall_seconds END)  AS mean_wall,
       MAX(CASE WHEN d.searched=1 THEN d.wall_seconds END)  AS max_wall,  -- p95 in the script
       SUM(CASE WHEN d.searched=1 THEN d.wall_seconds END)  AS total_wall,
       AVG(CASE WHEN d.searched=1 THEN d.iterations END)    AS mean_iters,
       AVG(CASE WHEN d.searched=1 THEN d.iters_per_sec END) AS mean_ips,
       AVG(CASE WHEN d.searched=1 AND d.stop_reason='wall_cap' THEN 1.0
                WHEN d.searched=1 THEN 0.0 END)             AS wallcap_rate
FROM decisions d JOIN games g ON g.game_id = d.game_id
GROUP BY g.condition, d.betting_stage;
```

(Percentiles like p95 are computed in the script from the pulled column, since
SQLite has no native percentile function.)

`regime` (`mccfr` | `vector`) — the solver approach (§6) — is reported *alongside*
each street as the solver that actually ran there, not as a share of a pooled search
total. The regimes own disjoint streets (subgame §6.5), so a share only restates how
often each street came up while inviting the head-to-head reading this section
explicitly cannot support (a true quality comparison would need the deferred
exploitability oracle; not in scope here).

Reading it — "is search correct, and is it doing what it should":

- **Routing (the correctness signal)** — each approach should fire only in its
  intended spots (vector → heads-up flop/turn/river; mccfr → the rest; subgame doc
  §6.4.1 / §6.7). A one-line check that `betting_stage` / `num_live` match the
  expected envelope catches a mis-routed solver — a correctness bug the usage
  numbers alone would hide.
- **Budget** — `wallcap_rate` near 1 (or `mean_iters` pinned at the cap) means the
  solver almost always exhausts the wall budget rather than the iteration budget on
  that street: it is the expensive one there. A cost signal, not a correctness one.
  Per street, because the iteration budget is a per-street constant
  (`search/budget.py`) and pre-flop and flop wall differ by an order of magnitude —
  one pooled rate describes a mixture no solver ever experienced, and names nothing
  actionable. (Whether it has actually *converged* by then is a separate story — no
  convergence test exists in the solve loop today, so it is not measured here.)
- **Cost** — `total_wall / hands` is the summary's one deliberate sum: the
  wall-clock a hand costs, which is what an experiment budget is built from.

**Range-tracking health — the second headline.** Is the belief helping versus a
uniform prior, and where does it break down? Joined to `game_seats` so it breaks
down **by the opponent type at the seat** — the 6-max question of *which* opponent
the tracker models well.

```sql
SELECT g.condition,                        -- the arm whose tracker this is
       s.agent_label      AS opponent,     -- who sat at this seat
       rq.betting_stage,
       AVG(rq.resolved)                          AS resolved_frac,
       AVG(rq.net_info_gain)  FILTER (WHERE rq.resolved) AS mean_net_gain,
       AVG(rq.collapsed_truth) FILTER (WHERE rq.resolved) AS collapse_rate,
       AVG(rq.uniform_fallback)                  AS fallback_rate,
       COUNT(*)                                  AS snapshots
FROM range_quality rq
JOIN game_seats s ON s.game_id = rq.game_id AND s.seat = rq.seat
JOIN games g      ON g.game_id = rq.game_id
GROUP BY g.condition, s.agent_label, rq.betting_stage;
```

`mean_net_gain > 0` means tracking beats the uniform prior for that opponent/stage;
`< 0` means it *hurts more than it helps* there (§7) and is a flag. The
`resolved_frac` runs lower than heads-up — with five opponents most seats fold
before showdown — so report it alongside every quality number (§11). Grouped by
`condition` because the belief is produced by the arm that is playing: pooling a DBR
arm's tracker with vanilla's averages two different trackers into a number that
describes neither.

### Automated flags

The script turns a few thresholds into explicit warnings so a bad run announces
itself without anyone reading the tables:

Every flag **names the arm it fired for**: a threshold crossed in one arm says
nothing about another, and an un-attributed message is unreadable the moment a
snapshot holds more than one.

| Flag | Condition | Reading |
|---|---|---|
| losing | an arm's strength CI entirely below 0 | genuine loss for that arm, not noise |
| unpaired | a CRN comparison matched 0 deals (or none was produced despite ≥2 conditions) | the multi-arm headline is missing — arms must share `run_seed` / `table_policy` / table shape to pair |
| thin pairing | matched deals well under the smaller arm's hand count | arms cover different hand ranges; the Δ rests on a subset |
| unbalanced arms | arm hand counts differ materially | an arm ran short (resume cursor / early failure) — CRN assumes equal coverage |
| range net-harmful | `mean_net_gain < 0` for any (condition, stage) | tracking hurts there; revisit flooring / fallback / bucketing (subgame §10) |
| budget-bound | high share of `stop_reason = 'wall_cap'` **on a named street** | search exhausts the wall budget rather than the iteration budget there — raise the cap or shrink the tree |
| search silent | an arm's `fire_rate ≈ 0` | search never triggered — likely a trigger/config error. `blueprint_only` is exempt: it is *supposed* to be silent |
| high collapse | `collapse_rate` above a set threshold | belief routinely rules out reality — card-removal / replay bug or too-aggressive updates |
| thin resolution | `resolved_frac` very low | range metrics rest on few showdowns — treat as weak evidence (§11 sampling bias) |

Thresholds live in one place in the script so they are easy to tune; the summary
prints the numbers regardless, and only the flag lines are threshold-gated.

### Entry point (sketch)

```python
# evaluation/summarize.py  —  python -m evaluation.summarize <snapshot.sqlite>
def summarize(db_path: str) -> dict:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)  # read-only
    report = {
        "meta":          _query_meta(con),          # arm inventory: deals vs game rows
        "strength":      _query_strength(con),      # per arm + position, never pooled
        "paired":        _query_paired(con),        # CRN Δ, matched (table, deck_seed)
        "range_quality": _query_range_health(con),  # per condition, per opponent/street
        "hu_coverage":   _query_hu_coverage(con),   # per condition, as fractions
        "search":        _query_search(con),        # per condition x street; p95 in python
    }
    report["flags"] = _evaluate_flags(report)       # the table above
    _print_human(report)                            # the block shown above
    return report                                   # also dumped to summary.json
```

It has no search-package dependency — it reads the schema, nothing else — so it
also runs standalone against any past snapshot for a retrospective check.

As built ([evaluation/summarize.py](../evaluation/summarize.py)), two portability
choices differ from the sketch above: the queries use **`CASE`-based conditional
aggregation, not the `FILTER` clause** shown in §8 (so the summary runs against
older SQLite on whatever box does the analysis), and the CI / percentile helpers
are **self-contained Python** (no numpy dependency in the standalone path). The CLI
is `argparse` (`python -m evaluation.summarize <snapshot> [--no-json] [--metric …]`).
Strength and the paired Δ share one metric chooser, so they can never quietly
disagree: both prefer `aivat_value` when every game carries it and fall back to raw
`hero_chips_delta` otherwise. `--metric raw` overrides that preference (and
`--metric aivat` forces the other way); a forced choice is stated in the printed
block and recorded as `metric_mode` in `summary.json`. The override exists because
AIVAT's benefit is a **property of the run, not of the column**: it is only worth
using when it actually reduces variance, so compare `sd(aivat_value)` against
`sd(hero_chips_delta)` per arm before trusting it — a ratio near 1 means the
estimator is not earning its per-hand cost, and below 1 means it is adding variance.
The runner (§10.1) calls `summarize(dest)` after the
final sync-back against the permanent snapshot, wrapped so a summary failure is
logged rather than failing the already-committed run.

## 9. Implementation Steps

Ordered so each step yields something usable before the next.

1. **Counters (cheap, enables everything).** Add cache hit/miss/size accessors on
   [SolverState](../poker_ai/search/solver_state.py)'s three caches (`legal_at`,
   `leaf_value_cache`, `runout_cache` — none track hits today) and a `node_count` /
   `unique_pubkeys` tally in the walk. Add a `config_fingerprint` hash over
   `SolverConfig` + `LeafConfig` + `table_policy`. Surface the already-computed
   `SearchResult.wall_seconds` / `iterations_run`, and the per-decision `stop_reason`.
   *done*
2. **The SQLite sink.** A small `evaluation/logging` module owning the DB
   connection: `open(path)` (applies the WAL / `synchronous` pragmas and creates
   the schema in §6), `log_game(...)`, `log_seats(...)`, `log_decision(...)`,
   `log_range_quality(...)`, a `snapshot(dest)` wrapping `VACUUM INTO`, and a
   per-game transaction context manager. No search-package dependency beyond the
   result objects. *done*
3. **Evaluation runner (§10.1).** The time-budgeted game loop: blueprint-derived
   opponents (`bp` + bias variants), `table_policy` seat assignment, hero/button
   rotation, deterministic `(run_seed, hand_index)` seeding, the `(run_id,
   hand_index)` resume cursor, and one logging transaction per hand. This is what
   actually produces games; the later steps observe it. *done*
4. **Range-quality hook (§7).** Buffer belief snapshots per seat during a hand; at
   showdown resolve them against revealed holes into `range_quality` rows. The one
   step that reaches inside the play loop (to observe revealed holes) rather than
   pure additive logging. *done*
5. **Cluster I/O.** Point the DB at node-local scratch; add periodic + on-SIGTERM
   `VACUUM INTO` sync-back to the permanent filesystem (§5), plus the SLURM wrapper
   (§10.1). *done*
6. **Summary command (§8).** `evaluation/summarize.py` reading the four tables into
   the headline block and `summary.json`. Read-only; no search-package dependency,
   so it also runs standalone against any past snapshot. Wired to run automatically
   at the end of the run, after the final sync-back. *done*
7. **Smoke test.** A short time-budgeted run against two `table_policy` settings,
   end-to-end: confirm the four tables populate, the join queries work, resume
   continues cleanly after a kill, the sync-back snapshot is readable, and the
   summary prints and writes `summary.json`.
8. **Scale.** First real experiments, reported on raw `hero_chips_delta` (bb/100).
9. **AIVAT (§10.2).** Layer on the variance-reduced estimator: value-function
   accessor, per-hand online correction accumulator, `aivat_value` populated, and
   the **unbiasedness + variance-drop test**. Switches the summary's strength CI onto
   `aivat_value`; unblocks the small-edge comparisons. Cross-run merge tooling
   (`(run_id, game_id)` keying, DuckDB `ATTACH`+`UNION`) follows once there is more
   than one run to compare. *done — action-node scope; see §10.2. Built:*
   [evaluation/aivat.py](../evaluation/aivat.py), wired into the runner behind the
   opt-in `--aivat` flag.
10. **Cross-condition pairing / CRN (§10.1).** The variance lever that makes the
    ~10k-hand-per-arm vanilla/DBR comparison
    ([opponent_modeling.md](opponent_modeling.md) §7) detectable. Three parts:
    (a) a `max_hands` fixed-count run mode (paired mode: disables `time_budget`)
    so arms share the `hand_index` range; (b) a `condition` column + the
    hero-independence invariant (dealing/seat-assignment drawn before any agent
    call), so a shared `run_seed` guarantees equal `deck_seed` per hand across
    arms; (c) summary tooling (§8) that joins arms on `deck_seed`, differences
    `aivat_value` per hand, and bootstraps the CI on the **paired difference**.
    Independent of the modeling package. *done — schema `games.condition` (v3) +
    `idx_games_deck`; `EvalConfig.condition` + total-based paired `max_hands`
    (resume-safe, budget-disabling) + the hero-independence invariant comment in
    `_play_and_log_one`; `summarize._query_paired` + `_bootstrap_ci` in the report
    and human block. Tests: CRN deal-is-hero-independent gate, v2→v3 migration,
    paired-mode total count, `_query_paired` deck-seed join.* (The cumulative-count
    learning-curve binning was dropped with the online learner; the model-quality axis is
    now the injected `SyntheticOpponentModel` error sweep, each level its own `condition`
    paired against the vanilla baseline.)

## 10. Planned Components

The logging backbone (§4–§9) is specified to build now. The two components below —
the **runner** that generates the games and **AIVAT** that makes small-edge
comparisons statistically feasible — are specified here in enough detail to
implement. The **runner is now built** ([evaluation/runner.py](../evaluation/runner.py),
§9 steps 3–6), and **AIVAT is now built** ([evaluation/aivat.py](../evaluation/aivat.py),
§9 step 9) in its action-node scope (see §10.2). §10.3 lists genuinely-future work.

### 10.1 Evaluation Runner

Drives the time-budgeted sequential game loop, calls the §9 logging module per hand,
and runs the §8 summary at the end.

**Opponents (start scope) — blueprint and its bias variants.** The five non-hero
seats are filled by **blueprint-derived bots**: the unaltered blueprint policy, or
one of the fold-/call-/raise-biased variants — the *same* inference-time
reweightings already used for the continuation strategies at MC leaves (subgame doc
§4). The hook already exists: `BlueprintPolicy.strategy(state, bias=…)`
([policy.py](../poker_ai/search/policy.py)) returns the regret-matched σ with the
bias class applied, so a runner samples an opponent action straight from it — no
search, no new artifacts, a cheap per-decision sampler. Two properties the design
leans on:

- **Cheap** — opponents add negligible cost, so the run's compute goes to the hero's
  search.
- **Known policy** — each opponent's action distribution is computable *exactly*
  (blueprint σ + the bias transform). This is precisely what lets AIVAT (§10.2)
  correct opponent actions, not just the hero's.

`game_seats.agent_label` vocabulary for the start scope: `bp`, `bp_fold`, `bp_call`,
`bp_raise` (extensible). The opponent is just "an agent exposing `action_probs(env,
seat)` and `sample(...)`", so heterogeneous/stronger opponents drop in later (§10.3)
without touching the runner or schema.

**Table-composition policy — the per-run knob.** The run config picks how the five
opponent seats are populated; the runner samples an assignment per hand and records
it in `game_seats`:

- `all_blueprint` — all five unaltered `bp`.
- `random` — each seat draws i.i.d. from {`bp`, `bp_fold`, `bp_call`, `bp_raise`}.
- `fixed` — an ordered list of `n_players - 1` opponent identities, one per
  opponent (e.g. `["bp_fold", "bp_call", "bp_raise"]` for a 4-player game). Every
  opponent plays every hand; which physical seat each identity lands in is
  reshuffled per hand (hero-independent, reproducible/CRN-paired the same way as
  `random`), so table position never confounds a given bias.

You choose the policy per experiment run; the schema captures whatever was assigned,
so analysis slices by opponent type regardless of the policy.

**Hero.** The search agent under test occupies one seat, **rotated by hand** (with
button rotation) so all six positions are covered evenly — position is a first-order
strength factor (§6).

**The loop — time-budgeted, like blueprint training.**

```
hand_index = 1 + max(games.hand_index WHERE run_id = ...)   # resume cursor, else 0
while wall_elapsed < time_budget and not SIGTERM:
    deck_seed  = derive(run_seed, hand_index)               # reproducible + resumable
    agent_seed = derive(run_seed, hand_index, "agent")
    seats      = assign(table_policy, hand_index)           # opp variants + hero/button rotation
    play the hand:
        hero decision   -> search agent; log a `decisions` row; buffer belief snapshots
        opp decision    -> blueprint(+bias) sample
        accumulate AIVAT corrections online (§10.2)
    resolve `range_quality` vs revealed holes (§7)
    write games + game_seats + decisions + range_quality  in ONE transaction (§4)
    every N hands / T minutes: VACUUM INTO snapshot -> permanent FS (§5)
    hand_index += 1
final VACUUM INTO; run the §8 summary
```

- **Time budget** mirrors blueprint training's `max_runtime_hours`: simulate until
  the budget ends, stopping only at a **hand boundary** (never mid-hand) so every
  `games` row is complete. No fixed hand count — the run fills the budget.
  **Paired mode (`max_hands` set) overrides this:** the run stops at an exact hand
  count instead of a wall-clock budget, so conditions being compared cover the
  *same* `hand_index` range (§10.1 "Cross-condition pairing"). Time-budget mode
  desyncs conditions — a search agent completes far fewer hands than a
  no-search blueprint_only run in equal wall-clock — which breaks pairing; use `max_hands`
  for any vanilla/DBR comparison.
- **Resume / idempotency** via the `(run_id, hand_index)` cursor (§6): a restarted or
  preempted run reads the max completed `hand_index` and continues from the next.
  Deterministic `derive(run_seed, hand_index)` seeding makes the continuation
  **identical** to an uninterrupted run — no replayed or skipped hands. The per-game
  transaction (§4) guarantees each hand is either fully logged or absent, which is
  what makes the cursor exact.
- **Config persisted** as `config.yaml` next to the snapshot (mirrors the blueprint
  runner); `config_fingerprint = hash(solver + leaf + table_policy)`.

**Cross-condition pairing (common random numbers).** The single highest-leverage
variance lever for comparing approaches (vanilla / DBR of
[opponent_modeling.md](opponent_modeling.md) §7), and it costs **zero extra
compute** — it is the same hands, seeded identically, not more hands. **The
no-exploitation arm — vanilla Pluribus (search with an empty model store, no
opponent model) — is *always* the baseline** (the design doc calls it "B0"; the
no-search `blueprint_only` arm is a pipeline test, not this baseline): every
exploitation number is the paired `Δ = aivat_value(treatment) − aivat_value(vanilla)`
on the shared `deck_seed`, i.e. the *incremental* value of exploiting, never a bare
per-arm EV. The mechanism is already latent in the deterministic seeding; this
locks it:

- **Shared `run_seed` + identical `table_policy` + identical table shape.** Deal
  and seating are pure functions of `(run_seed, hand_index)` — `deck_seed =
  derive(run_seed, hand_index)`, `seats = assign(table_policy, hand_index)`,
  hero/button rotation keyed by `hand_index` — and crucially **none of them depend
  on the hero agent**. So two runs that share `run_seed` and `table_policy` see, at
  each `hand_index`, the *same* cards, the *same* opponent seating, and the hero in
  the *same* seat. Only the hero's own strategy (the treatment) differs.
  *Verified mechanism:* the deal is **cursor-based over a single
  construction-time shuffle** — `Deck.__init__` shuffles once off the global RNG
  (seeded with `deck_seed` immediately before `new_env`), and `deal_private_cards`
  / `deal_community` only advance a cursor, never re-drawing; the hero's search runs
  on deepcopies with its own `default_rng`, so nothing it does can perturb the
  played board (confirmed: interleaving arbitrary global-RNG use between streets
  leaves holes + board bit-identical). *Precondition, because the board's offset in
  the permutation depends on how many holes were dealt first:* the arms must also
  match `n_players` and the deck/stack config (`low/high_card_rank`,
  `starting_stack`) — a different seat count reshuffles which cards land on the
  board. The runner should assert hero-independence (dealing/assignment drawn before
  and separately from any agent call).
- **`deck_seed` is the cross-run join key.** Pair the conditions on `deck_seed`
  (equivalently `hand_index` under a shared `run_seed`) and difference per hand:
  `Δ = aivat_value(DBR) − aivat_value(vanilla)` on the matched deal. The enormous
  shared card-luck cancels in `Δ`, so its CI is far tighter than either arm's —
  and this **stacks multiplicatively with AIVAT** (§10.2), which further corrects
  the residual opponent-action variance after the tree diverges. (`pairing_id`
  remains the *within*-run seat-rotation lever; this is the *across*-run one.)
  The stacking is **conditional on AIVAT's RNG isolation** (§10.2): a value function
  that shares a stream with anything arm-dependent scores an identically-played deal
  differently per arm, which turns AIVAT into a *source* of paired-Δ noise rather
  than a reducer of it. Gated by `test_aivat.py::TestCrnPairing`.
- **Fixed `max_hands`, not a time budget** (see the loop note above), so every
  condition contains the same `hand_index` set to pair against. Each condition is
  its own `run_id`; a `condition` label column on `games` (e.g. `vanilla|DBR|blueprint_only`,
  with the `(p_max, τ)` cell for A) makes the paired join self-describing without a
  run_id→method side table.
- **What cancels vs. what doesn't.** Card luck cancels fully (shared hole cards +
  board). Opponent *action* draws share a seed but desync once the hero diverges
  the tree — that residual is AIVAT's job, not CRN's; sharing the opponent-RNG
  stream still helps on the pre-divergence (early-street) decisions. Report the CI
  on the **paired difference** of `aivat_value` (bootstrapped over hands), never on
  the two arms independently.

**Run config fields:** `run_id`, `run_seed`, `table_policy`, `time_budget`,
`max_hands` (paired mode; mutually exclusive with `time_budget`), `condition`,
`big_blind`, `starting_stack`, `n_players = 6`, `sync_interval`, scratch/permanent
paths.

**Cluster launch script** ([scripts/evaluation.sh](../scripts/evaluation.sh)) — the
SLURM (or equivalent) wrapper around the runner: stages the LUT **and blueprint** to
node-local scratch (both are on the real-time-search hot path, §5), points
`--db-path` at node-local scratch and `--sync-path` at the permanent FS, sets the
sync interval, forwards SIGTERM to the runner (so it writes the final `VACUUM INTO`
before the wall-clock `SIGKILL`), and points analysis at the permanent snapshot. It
requests the ~420 GB local disk (`--tmp`) and RAM (`--mem`) the two staged artifacts
need (§5), runs a preflight free-space check, and — on restart — seeds the
node-local DB from the existing permanent snapshot so the `(run_id, hand_index)`
cursor resumes cleanly. Mirrors [scripts/training.sh](../scripts/training.sh).

### 10.2 AIVAT (variance-reduced strength estimate)

Fills `games.aivat_value` (§6). Per-hand chip variance is huge (~100 bb/100 std), so
raw bb/100 has a wide CI (§11); AIVAT gives the **same mean with far smaller
variance**, which is what makes the small-edge comparisons (approach vs approach, and
the future exploitation solver) detectable.

**Estimator.** For a played hand with hero utility `u(z)`:

```
aivat_value = u(z)  −  Σ correction_terms
```

Each correction term has **zero expectation** by construction, so
`E[aivat_value] = E[u(z)]` for *any* value function — the estimator is **unbiased**;
a better value function only shrinks the variance further, it can never skew the
mean. Corrections are taken at:

- **every action node of a known-policy player** — here *all* of them, since both the
  hero and the blueprint bots have known policies: `term = v(child_sampled) −
  Σ_a π(a)·v(child_a)`, with `π` = the hero's played strategy at hero nodes
  (`SearchResult.policy`, the logged `action_dist`) and the opponent's
  `BlueprintPolicy.strategy(state, bias)` at opponent nodes — both exact, no
  estimation. (Correcting the opponents too — possible
  only because their policies are known — is the "gift" of the start-scope bots and
  gives *full*-AIVAT reduction rather than the hero-only partial case.)
- **every chance node** (hole deal, board cards) — the MIVAT term `v(realized) −
  Σ_c P(c)·v(child_c)`, with `P` range-aware (card removal).

**Value function `v`.** Reuse the search's own value machinery — the solved subgame's
root/leaf values ([leaf.py](../poker_ai/search/leaf.py)) give a public-state value
under the current ranges. Any consistent `v` is unbiased; pick the best cheap one at
build time (candidate: the blueprint's expected value / rollout equity at the public
state given ranges). Better `v` ⇒ more variance reduction, never a correctness risk.

**Computed online, stored as one scalar.** Accumulate corrections during the hand,
while the env, ranges, and values are live, and store only `aivat_value` — no
per-node blob in the DB. The extra cost is evaluating `v` at the sibling actions and
cards *not* taken (make/undo makes those states reachable); it is bounded by
branching × cost(`v`) and lands in the experiment's time budget, not the real-time
search budget.

**Acceptance test = unbiasedness + variance drop.** Over many hands,
`mean(aivat_value)` must equal `mean(hero_chips_delta)` within CI (unbiased), while
`var(aivat_value) ≪ var(hero_chips_delta)`. This single gate catches a sign error or
an information leak in the corrections — the main implementation risk — and confirms
the payoff. Budget the effort in this test, not the arithmetic.

**Sequencing.** The backbone logs raw `hero_chips_delta` first; AIVAT layers on using
data already captured (`action_dist`, values, ranges). Adopting it lets the runner
skip seat rotation (§6). Medium effort, well-bounded.

**As built** ([evaluation/aivat.py](../evaluation/aivat.py)), the scope is
**action-node AIVAT (hero + opponents) plus the terminal all-in runout**; the general
per-street chance (MIVAT) term is **deferred**. The engine has no steppable /
enumerable per-street chance node — hole cards are dealt in `PokerEnv.__init__` and
board cards *inside* `_apply_action_in_place` off a shuffled deck, with no "re-deal a
specific alternative card down the same betting line" primitive — so the per-street
`v(realized) − Σ_c P(c)·v(child_c)` cannot be taken exactly without a new engine
primitive. What *is* taken:

- **Action nodes** — at every hero and opponent decision, `term = v(child_sampled) −
  Σ_a π(a)·v(child_a)`, with the siblings enumerated by the engine's make/undo
  (`step_in_place`/`undo`) on the runner's existing pre-action env copy. `π` is the
  hero's played σ (the exact vector logged as `decisions.action_dist`) or the
  opponent's `BlueprintOpponent.action_probs` — both exact known policies (§10.1).
- **The terminal all-in runout** — the one chance event the engine *can* integrate
  exactly: at a decision-free all-in terminal with **≤2 board cards to come** (a
  flop/turn all-in) the realised single board is replaced by `runout_equity`
  (`term = u(z) − runout_equity[hero]`), collapsing `aivat_value` to
  `runout_equity − Σ action_terms`. A high-variance chance event, removed for free.
  A **pre-flop** all-in (5 cards to come) is **skipped** — its exact runout blows
  past `runout_equity`'s enumeration cap into a thousands-of-boards Monte-Carlo
  sample per hand — so those hands keep only their action-node corrections.

**The value function `v`** reuses the leaf machinery
([leaf.py](../poker_ai/search/leaf.py) `continuation_value`) under a fixed
all-blueprint continuation profile: it materialises a card-disjoint joint hole
assignment sampled from the tracker's belief — the hero seat filled with its *known*
hole — and averages the hero-seat continuation value over a few (`--aivat-hole-samples`,
default 6) such draws. Any consistent `v` is unbiased, so the internal hole sampling
uses cheap sequential-with-removal rather than the exact conditioned joint. **The
information-leak rule** (the main correctness trap): `v` integrates only over the
*observer's* belief and never reads an opponent's concrete hole; `π` at an opponent
node may condition on that opponent's own hole (it is exactly the distribution the
action was sampled from).

**RNG isolation** (see [poker_ai/search/rng.py](../poker_ai/search/rng.py)) — AIVAT
owns its randomness in *both* directions. It runs on its own sub-stream (a 5th
`derive_seeds` child) plus a spawned board-runout stream, and reads the global
`np.random` nowhere; the global stream belongs to the played hand's deal alone.
Outward, that keeps AIVAT **passive**: a hand's raw `hero_chips_delta` is
byte-identical with AIVAT on or off. Inward — the direction that matters for the
headline — it makes `aivat_value` a pure function of `(run_seed, hand_index)` and
the played line, so it **stacks** with the §10.1 CRN pairing instead of eroding it.
Earlier the value function reshuffled the undealt deck off the global stream and the
module merely snapshot-and-restored it: that gave the outward guarantee only, and
because the hero's solver consumes the global stream by an *arm-dependent* amount,
two arms playing an identical hand drew different rollout boards. The resulting
arm-specific noise landed directly in the paired Δ, which is why the first AIVAT run
showed no variance reduction and inflated the DBR headline. Note that a residual,
*legitimate* arm-dependence remains: `v` integrates over `hero.tracker`, whose
boundary update replays under the last search's average policy, and `π` is the
played σ — so arms at different budgets still compute different (still unbiased)
corrections. Making those cancel too needs an arm-independent `v`, a separate
decision. Gated behind the opt-in `EvalConfig.aivat` / `--aivat` flag (extra per-hand
cost, in the experiment budget, off the real-time search hot path). The
**acceptance test** ([test/evaluation/test_aivat.py](../test/evaluation/test_aivat.py))
gates the §10.2 property — paired `mean(aivat) ≈ mean(hero_chips_delta)` within CI
and `var(aivat) < var(hero_chips_delta)` (measured ~2× reduction on the stub) — plus
the deterministic term/finalize algebra and a no-leak belief-sampling check. The
summary's strength CI switches onto `aivat_value` automatically once every game
carries it (§8, no code change).

### 10.3 Future (not planned in detail)

- **Heterogeneous / stronger opponents** — drop in via the agent interface (§10.1);
  no schema change (`game_seats.agent_label` already carries per-seat identity).
- **Opponent-exploitation solver** — a second, exploitative hero agent, compared
  **across runs** by `run_id` / `config_fingerprint` (strength delta vs. the same
  opponent tables), not a per-decision approach within a run. Forward-compatible; no
  columns needed now.
- **Exploitability evaluator** (subgame doc §9) — fills the reserved nullable
  `exploitability` / `game_value` columns in `decisions` once built.

## 11. Risks and Open Questions

| Risk / question | Note |
|---|---|
| **Showdown sampling bias** — range quality is only verifiable at showdown, a non-random subset of hands (hands that fold early are never checked). | Report quality with the resolved-fraction alongside it; treat aggregates as conditional on reaching showdown, not unconditional. Do not silently drop unresolved hands without reporting how many. |
| **Logging perturbs timing measurements.** | Keep the write off the hot path (per-game transaction, `synchronous=NORMAL`, node-local disk); record wall-clock around the solve only, excluding the commit. |
| **Node death between periodic snapshots.** | Bounded by the sync-back interval (§5); tune the interval against game throughput. Accept loss of at most one interval's games. |
| **`config_fingerprint` instability** (dict ordering, float formatting). | Canonicalise before hashing (sorted keys, fixed float repr) so identical configs across runs collide as intended. |
| **Belief-vector buffering memory** for range quality. | Only per-live-seat vectors for the current hand are held; freed at hand end. Bounded by `n_seats × n_combos`. |
| **Schema evolution** as new metrics are added. | SQLite `ALTER TABLE ADD COLUMN` with nullable columns (the exploitability columns are the first planned use); analysis tolerates NULLs. |
| **Under-powered experiment** — per-hand chip variance is huge (~100 bb/100 std), so raw bb/100 CI ≈ `1000/sqrt(N)`: ±28 at 5k hands, ±6 at 100k. Small edges (approach A vs B, exploitation solver) are undetectable at modest N. | AIVAT (§10.2) is the primary mitigation — same mean, far tighter CI, so plan around the AIVAT-adjusted interval. Do the power calc (target edge → required N) *before* the run. Prefer a bootstrap CI over normal-approx (chip outcomes are heavy-tailed). |
| **Cross-run merge corruption** — `game_id` is unique only within a snapshot; unioning files on it alone collides and cross-joins child rows. | Logical key is `(run_id, game_id)` (§6); the merge namespaces/remaps `game_id` on import. DuckDB `ATTACH` + `UNION ALL` is the analysis path; `schema_version` handles column drift. |

---

## Appendix A: Current Logging State

Baseline at the time of writing (see the audit that motivated this document): the
search stack is effectively un-instrumented. The repository has a centralised
`RichHandler` logging config ([poker_ai/__init__.py](../poker_ai/__init__.py),
fixed at `INFO`), but `poker_ai/search/` uses it only for two rare warnings — the
MCCFR joint-sampler fallback ([mccfr.py:152](../poker_ai/search/mccfr.py#L152))
and the range collapse ([ranges.py:245](../poker_ai/search/ranges.py#L245)).
`wall_seconds` and `iterations_run` are computed and returned on `SearchResult`
([solver.py](../poker_ai/search/solver.py)) but never logged; `_decision_log` in
the tracker is buffered but never consumed. There is no structured sink, no
per-run metadata, no cache/tree statistics, and no verbosity or output
configuration. This document defines the sink and the metadata that fills it.
