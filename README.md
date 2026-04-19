# pluribus

A Pluribus-style multiplayer no-limit Texas hold'em poker AI. The codebase is the implementation supporting a master's thesis at KIT.

It started as a fork of [keithlee96/pluribus-poker-AI](https://github.com/keithlee96/pluribus-poker-AI) and has since been substantially rewritten and restructured. The blueprint trainer, the card-information abstraction pipeline, the storage layer, the test suite, and the package layout are all new.

## Status

Working today:
- Card-information abstraction build (Monte Carlo and exact methods, chunked + checkpointed for HPC runs)
- Blueprint strategy training via Linear MCCFR with regret pruning, in either single-process or multi-process mode
- LMDB-backed sparse regret/strategy tables that scale beyond RAM
- Terminal client to play hands against a trained blueprint
- Pytest suite covering the environment, abstraction, and training pipeline

Planned next:
- Real-time search on top of the offline blueprint
- Safe opponent exploitation methods integrated into the real-time search loop

## Repository layout

```
poker_ai/                   Blueprint trainer, tables, terminal client
├── blueprint/              CFR traversal, training schedule, single- and multi-process runners
├── tables/                 LMDB-backed sparse regret / strategy storage with checkpointing
└── terminal/               Text-mode client for playing against a trained agent

cli/                        `poker_ai` Click entry point (composes blueprint + build + terminal)
environment/                Poker game state, action space, hand evaluator + rank table, deck / pot / player
information_abstraction/    Card-info LUT build pipeline (`build/`) and runtime load API
utils/                      Shared atomic-IO primitives (no domain dependencies)
data/                       Generated artifacts (e.g. `data/20cards_exact/card_info_lut.joblib`)
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

## Installation

Tested under Python 3.7 in a conda environment named `pluribus`.

```bash
conda create -n pluribus python=3.7
conda activate pluribus
pip install -e .
```

This installs the `poker_ai` console entry point.

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

## Tests

```bash
pytest                            # full suite
pytest -m "not slow"              # skip slow training/abstraction tests
pytest -m "not requires_lut"      # skip tests that need a built LUT on disk
```

LUT-dependent tests resolve the LUT directory from `PLURIBUS_LUT_PATH` and default to `data/20cards_exact/`. They self-skip when the file is absent.

## Helper scripts

- `scripts/training.sh` — Slurm submission wrapper for `poker_ai train start`
- `scripts/abstraction_auto_resub.sh` — Slurm submission wrapper for `poker_ai build-abstraction` with self-resubmission until all four streets are clustered
- `scripts/rebind_lut.py` — rewrites the absolute paths a `MemmapLookup` stores when its `save_dir` is moved

## Attribution

The original Pluribus algorithm is described in Brown & Sandholm, *Superhuman AI for multiplayer poker* (Science, 2019). This codebase began as a fork of [keithlee96/pluribus-poker-AI](https://github.com/keithlee96/pluribus-poker-AI), which itself derives from earlier work by Leon Fedden and Colin Manko.
