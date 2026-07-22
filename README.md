# pluribus

A Pluribus-style multiplayer no-limit Texas hold'em poker AI. The codebase is the implementation supporting a master's thesis at KIT.

It started as a fork of [keithlee96/pluribus-poker-AI](https://github.com/keithlee96/pluribus-poker-AI) and has since been substantially rewritten and restructured. The blueprint trainer, the card-information abstraction pipeline, the storage layer, the real-time search, the evaluation harness, the compiled core, the test suite, and the package layout are all new.

## Status

Working today:
- Card-information abstraction build (Monte Carlo and exact methods, chunked + checkpointed for HPC runs)
- Blueprint strategy training via Linear MCCFR with regret pruning, in either single-process or multi-process mode
- LMDB-backed sparse regret/strategy tables that scale beyond RAM
- **Real-time depth-limited subgame search** on top of the offline blueprint — two CFR regimes (range-vs-range vector CFR for heads-up post-flop, traverser-vectorized external-sampling MCCFR for pre-flop / multiway), Bayesian range tracking, a blueprint continuation fleet at the depth limit, and a structural per-subgame iteration budget over independent parallel replicas
- **Evaluation harness** — plays the searching agent against blueprint/opponent tables, logs every decision to SQLite, and reports strength (AIVAT-reduced), range-tracking quality, and search cost
- **Compiled Cython core** — a byte-identical `FastState` betting engine + hot-loop kernels behind one flag per pipeline, ~14× on the training CFR loop
- Terminal client to play hands against a trained blueprint
- Pytest suite covering the environment, abstraction, training, search, and evaluation

Planned next:
- Safe opponent exploitation methods integrated into the real-time search loop

## Repository layout

```
poker_ai/                   Blueprint trainer, tables, real-time search, compiled core, terminal client
├── blueprint/              CFR traversal, training schedule, single- and multi-process runners
├── tables/                 LMDB-backed sparse regret / strategy storage with checkpointing
├── search/                 Real-time depth-limited subgame solver (two CFR regimes, ranges, leaf, budget)
├── _core/                  Compiled Cython core: FastState engine + hot-loop kernels (.pyx source)
└── terminal/               Text-mode client for playing against a trained agent

evaluation/                 Evaluation harness: hand runner, SQLite decision log, AIVAT, range-quality, summarize
cli/                        `poker_ai` Click entry point (composes blueprint + build + terminal)
environment/                Poker game state, action space, hand evaluator + rank table, deck / pot / player
information_abstraction/    Card-info LUT build pipeline (`build/`) and runtime load API
utils/                      Shared atomic-IO primitives (no domain dependencies)
data/                       Generated artifacts (e.g. `data/20cards_exact/card_info_lut.joblib`)
docs/                       Design docs (`subgame_solving.md`, `evaluation.md`, …)
scripts/                    Slurm submission scripts and one-off helpers (LUT rebinding)
test/                       Pytest suite mirroring the package layout
```


## Architecture

A full-deck six-player build blows past what the upstream codebase assumed. The river alone has ~2.8 billion hand/board combinations on a 52-card deck, and a CFR run over six players touches orders of magnitude more information sets than the toy decks the original implementation targeted. Two subsystems were rewritten from scratch to make this tractable on commodity HPC hardware:

### Card-information abstraction pipeline

Lives in [information_abstraction/build/](information_abstraction/build/). The pipeline produces a `{street: {combo: cluster_id}}` lookup table by extracting an expected-hand-strength feature for every combo and then KMeans-clustering those features per street. Two design choices make this workable at full-deck scale:

- **Chunked, resumable processing.** Each street's combos are sliced into fixed-size chunks ([ChunkStore](information_abstraction/build/chunk_store.py)) and dispatched across a process pool. Every completed chunk is written as an atomic `.npy` file with an fsync + round-trip verification step so that a node crash or quota exhaustion on the parallel filesystem cannot leave a silently-truncated chunk behind. A [CheckpointManager](information_abstraction/build/checkpoint.py) tracks per-street completion in a JSON file guarded by an `fcntl` lock, so a job that runs out of wall time simply resubmits itself and resumes with the missing chunks only. After merge, KMeans itself checkpoints inside [Clusterer](information_abstraction/build/clusterer.py) so the clustering stage is resumable too.
- **Memory-efficient river lookup.** A Python dict keyed on ~2.8 billion tuples is not feasible — storing the cluster ids alone as a hash table would require close to a terabyte of RAM and a `joblib.dump` would never complete. Instead the build writes a compact `uint16` memmap (`cluster_ids.dat`) whose row layout is the combinadic index of the (hole, board) combo. Runtime access goes through [MemmapLookup](information_abstraction/lookup.py), which computes the row in O(1) from the cards and reads a single `uint16`. The pickled `MemmapLookup` is a few kilobytes; the OS page cache is shared across all worker processes that open the file.

The streets are processed in order `river → turn → flop` because the turn and flop features are histograms over the downstream street's cluster ids — each upstream `cluster_ids.dat` is lazily memmapped into worker processes and cached at module scope ([ehs.py](information_abstraction/build/ehs.py)).

### CFR tables

Lives in [poker_ai/tables/](poker_ai/tables/). The CFR trainer needs per-infoset integer arrays (regret and average-strategy counts) that are shared across many worker processes, persistent across restarts, and resilient to training runs that allocate far more information sets than fit in RAM. The storage layer is built in three layers plus a checkpointer:

- **[InfosetIndex](poker_ai/tables/index.py) — persistent `string → row` mapping.** An LMDB environment per street maps 16-byte hashes of infoset strings to stable 64-bit row numbers. Row allocation happens inside a single LMDB write transaction, which serialises the `get_or_create` path across processes without any explicit lock. Readers never need a transaction to learn the current row count because it is mirrored in a `multiprocessing.Value` the parent initialises before forking. LMDB reader slots do not survive `fork(2)` — workers reopen their environment handles at startup.
- **[ChunkStore](poker_ai/tables/chunk_store.py) — shared-memory integer arrays.** Rows are grouped into fixed-size chunks, each backed by an mmapped file under `/dev/shm`. Because the files are mapped `MAP_SHARED`, any worker's write is instantly visible to every other worker without IPC. A shared dirty-flag array records which chunks have been modified since the last checkpoint, so only the dirty ones have to be flushed to the (slow) network filesystem.
- **[ChunkedTable](poker_ai/tables/chunked_table.py) — the facade CFR code sees.** Translates `(infoset_string, action_index)` into a shared-memory row and exposes `get_row` / `update_row` / `merge_delta_row`. Concurrent writers to the same chunk are serialised by a bank of POSIX semaphores (`N_STRIPE_LOCKS`) keyed on `chunk_id % N`; different chunks almost never contend.
- **[CFRTables](poker_ai/tables/cfr_tables.py)** bundles the four streets × two tables (regret, strategy) plus the shared indexes, and owns bulk operations that must span every table (LCFR discounting, save/restore).
- **[CheckpointManager](poker_ai/tables/checkpoint.py) — atomic, hardlink-based checkpoints.** Each checkpoint is a self-contained directory. Dirty chunks are written as atomic `.npy` files into a temp directory; clean chunks are **hardlinked** from the previous generation so every checkpoint remains complete without copying unchanged data (same-inode hardlinks are near-free on POSIX). A final `rename` makes the new directory visible atomically, and the previous generation is then deleted — exactly one checkpoint is retained at all times. `SIGTERM` / `SIGINT` are intercepted so SLURM jobs running out of wall time write one final emergency checkpoint before exit. On startup the manager auto-resumes if a valid checkpoint exists and refuses to resume across structural hyperparameter changes that would silently corrupt training.

Together these pieces let a six-player full-deck run train with many workers sharing mutable state at near-zero synchronisation cost, persist only what has changed at each checkpoint, and resume cleanly after a SLURM preemption.

### Real-time search

Lives in [poker_ai/search/](poker_ai/search/) (design: [docs/subgame_solving.md](docs/subgame_solving.md)). At play time the agent re-solves the current [depth-limited subgame](poker_ai/search/context.py) against the blueprint rather than reading a fixed strategy. A [RangeTracker](poker_ai/search/ranges.py) maintains a Bayesian belief over every seat's holdings, updated at each betting-round boundary from the blueprint's own play; the resulting per-seat ranges root the subgame. [`solve`](poker_ai/search/solver.py) then picks one of two CFR regimes:

- **Vector-form CFR** ([vector.py](poker_ai/search/vector.py)) for heads-up flop/turn/river — full-width range-vs-range Linear CFR that plays every hand of both ranges at once, walking to showdown with the future streets keyed by LUT cluster.
- **Traverser-vectorized external-sampling MCCFR** ([mccfr.py](poker_ai/search/mccfr.py)) for the pre-flop root and any multiway subgame — the traverser's whole range is solved in one vectorized pass while opponents and chance are sampled.

Both share the per-public-node regret/strategy matrices in [solver_state.py](poker_ai/search/solver_state.py) and the arithmetic in [vform.py](poker_ai/search/vform.py). At the depth limit, leaves are valued by a **continuation fleet** of biased blueprint strategies ([leaf.py](poker_ai/search/leaf.py)); barely-reached solved rows are shrunk back toward the blueprint. Rather than a machine-specific wall clock, each search runs a **structural iteration budget** ([budget.py](poker_ai/search/budget.py)) derived from the subgame's size, and parallelizes as *W* independent replicas merged once ([parallel.py](poker_ai/search/parallel.py)).

### Evaluation harness

Lives in [evaluation/](evaluation/) (design: [docs/evaluation.md](docs/evaluation.md)). [`run_evaluation`](evaluation/runner.py) seats the searching agent against configurable opponent tables and plays hands under a wall-clock / hand budget, streaming **every decision** to a SQLite database ([sqlite_logging.py](evaluation/sqlite_logging.py)) written to node-local scratch and periodically `VACUUM INTO`-synced to permanent storage. Strength is reported with an [AIVAT](evaluation/aivat.py) control-variate to cut variance; [range_quality.py](evaluation/range_quality.py) scores how well the belief tracker matched the truth, and [summarize.py](evaluation/summarize.py) rolls the log up into a report (strength CIs, range-tracking net info gain, search cost, blueprint-prior health, routing checks).

### Compiled core

Lives in [poker_ai/_core/](poker_ai/_core/). The CFR hot loop and the game engine it walks are compiled to Cython. The centrepiece is a `FastState` betting engine ([_state.pyx](poker_ai/_core/_state.pyx)) that is **byte-identical** to the Python `PokerEnv` — make/undo, legal actions, public keys, and terminal settlement — plus per-street kernels for regret matching, showdown, and per-combo settlement. `.pyx` files are the source of truth (generated `.c`/`.so` are gitignored); every core routine keeps a pure-Python oracle and is covered by byte-parity gates. It is gated by **one operator flag per pipeline** ([flags.py](poker_ai/_core/flags.py)): `PLURIBUS_CFR_CORE=1` lights the compiled walk + every kernel for blueprint training (~14× on the training CFR loop), `PLURIBUS_SEARCH_CORE=1` does the same for real-time search. With neither set (or Cython absent) everything transparently falls back to the Python path.

## Installation

Tested under Python 3.7 in a conda environment named `pluribus`.

```bash
conda create -n pluribus python=3.7
conda activate pluribus
pip install -e .
```

This installs the `poker_ai` console entry point and compiles the Cython core ([poker_ai/_core/](poker_ai/_core/)). After editing a `.pyx`, rebuild in place with `python setup.py build_ext --inplace`. Set `POKER_AI_NO_EXT=1` to skip the compiled build entirely — the code then runs on the pure-Python fallback.

## Workflow

### 1. Build the card-information abstraction

Clusters all post-flop hand strengths into the buckets the trainer treats as equivalent. Required before training. Output goes to `--save_dir`.

```bash
poker_ai build-abstraction \
  --save_dir data/20cards_exact \
  --low_card_rank 10 --high_card_rank 14 \
  --n_river_clusters 200 --n_turn_clusters 200 --n_flop_clusters 200 \
  --method exact
```

For full-deck Monte Carlo runs use `--method monte_carlo`, set `--n_simulations_river`, and submit via `scripts/abstraction_auto_resub.sh` — that Slurm script auto-resumes from the checkpoint when its wall-time expires.

> **Before using a LUT built elsewhere (or after moving its directory), rebind it to your local path:**
>
> ```bash
> python scripts/rebind_lut.py data/20cards_exact
> ```
>
> `card_info_lut.joblib` stores the absolute path to its `cluster_ids.dat` memmap internally. Training and play will fail to load the LUT until those paths match the current location.

### 2. Train the blueprint

```bash
poker_ai train start \
  --multi_process \
  --n_players 6 \
  --lut_path data/20cards_exact \
  --nickname runs/6player_blueprint \
  --strategy_interval 25000 --sync_interval 1000 \
  --discount_interval 5 --discount_duration_iters 250000 \
  --prune_threshold 125000 --c -300000000
```

A run auto-resumes when the resolved save directory already contains a valid checkpoint — re-invoke the same command with the same `--nickname` to continue an interrupted run. The cluster counterpart is `scripts/training.sh`.

### 3. Play against a trained blueprint

```bash
python -m poker_ai.terminal.runner \
  --lut_path ./data/20cards_exact \
  --pickle_dir ./data/20cards_exact \
  --strategy_path ./agent.joblib \
  --agent offline --n_players 3
```

### 4. Evaluate a searching agent

Plays the real-time-search agent against opponent tables and logs every decision to SQLite for analysis.

```bash
PLURIBUS_SEARCH_CORE=1 python -m evaluation.runner run \
  --run-id demo --db-path runs/eval.db \
  --blueprint-path runs/6player_blueprint \
  --lut-path data/20cards_exact \
  --n-players 6 --time-budget-hours 1
```

The cluster counterpart is `scripts/evaluation.sh`. Summarize a finished run with `python -m evaluation.summarize runs/eval.db`.

## Tests

```bash
pytest                            # full suite
pytest -m "not slow"              # skip slow training/abstraction tests
pytest -m "not requires_lut"      # skip tests that need a built LUT on disk
```

LUT-dependent tests resolve the LUT directory from `PLURIBUS_LUT_PATH` and default to `data/20cards_exact/`. They self-skip when the file is absent.

## Helper scripts

- `scripts/training.sh` — Slurm submission wrapper for `poker_ai train start`
- `scripts/evaluation.sh` — Slurm submission wrapper for `python -m evaluation.runner`, with LUT/blueprint staging and a compiled-core liveness preflight
- `scripts/abstraction_auto_resub.sh` — Slurm submission wrapper for `poker_ai build-abstraction` with self-resubmission until all four streets are clustered
- `scripts/rebind_lut.py` — rewrites the absolute paths a `MemmapLookup` stores when its `save_dir` is moved

## Attribution

The original Pluribus algorithm is described in Brown & Sandholm, *Superhuman AI for multiplayer poker* (Science, 2019). This codebase began as a fork of [keithlee96/pluribus-poker-AI](https://github.com/keithlee96/pluribus-poker-AI), which itself derives from earlier work by Leon Fedden and Colin Manko.
