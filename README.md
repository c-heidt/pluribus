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
poker_ai/                   Blueprint trainer, CLI, tables, terminal client
├── blueprint/              CFR traversal, training schedule, single- and multi-process runners
├── tables/                 LMDB-backed sparse regret / strategy storage with checkpointing
├── cli/                    `poker_ai` Click entry point
└── terminal/               Text-mode client for playing against a trained agent

environment/                Poker game state, action space, hand evaluator, deck / pot / player
information_abstraction/    Card-info LUT build pipeline (`build/`) and runtime load API
data/                       Generated artifacts (e.g. `data/20cards_exact/card_info_lut.joblib`)
scripts/                    Slurm submission scripts and one-off helpers (LUT rebinding)
test/                       Pytest suite mirroring the package layout
```

`environment/` and `information_abstraction/` are top-level packages, sibling to `poker_ai/`, not nested under it.

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
