#!/bin/bash -l
# Slurm submission script to run training via the package CLI.
# Usage:
#   # Base blueprint (default):
#   sbatch --export=ALL,WORKSPACE=/path/to/ws training.sh
#
#   # Biased blueprint warm-started from a finished base run:
#   sbatch --export=ALL,WORKSPACE=/path/to/ws,BIAS=fold,WARM_START=/path/to/base training.sh
#SBATCH --job-name=pluribus-train
#SBATCH --output=logs/training-%j.out
#SBATCH --error=logs/training-%j_error.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=100000mb
#SBATCH --signal=SIGTERM@300
#SBATCH --mail-type=All


set -euo pipefail

# User-configurable
CONDA_ENV=${CONDA_ENV:-pluribus}
PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ -z "${WORKSPACE:-}" ]; then
  echo "ERROR: WORKSPACE is not set. Export WORKSPACE=/path/to/workspace before submitting (e.g. sbatch --export=ALL,WORKSPACE=...)." >&2
  exit 1
fi

# Training parameters (cycle-based options are counted in sync cycles = N * sync_interval iterations).
#
# The cycle/iteration cadence below is fitted to the cluster's current throughput
# of ~7M traversals-per-player per hour (the ``t`` in the progress logs), which at
# SYNC_INTERVAL=1000 is 7,000 sync cycles/hour.  Re-derive if throughput changes:
#   cycles/hour = 7e6 / SYNC_INTERVAL ;  raw traversals-per-player/hour = 7e6.
N_PLAYERS=${N_PLAYERS:-4}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-96}
SYNC_INTERVAL=${SYNC_INTERVAL:-1000}
# LCFR discount stretched over the first 4h (was ~13 min at this throughput),
# keeping the same 19 discount steps: one every 1400 cycles (12 min), window
# 28000 cycles (4.0h = 20 * DISCOUNT_INTERVAL).
DISCOUNT_INTERVAL=${DISCOUNT_INTERVAL:-1400}
DISCOUNT_DURATION_CYCLES=${DISCOUNT_DURATION_CYCLES:-28000}
# Pre-flop average-strategy warm-up: start accumulating φ after the first 6h
# (42000 cycles = 6.0h at 7,000 cycles/h), matching CHECKPOINT_START_CYCLES so
# the pre-flop average and the post-flop snapshots share one warm-up and both
# skip the near-random early era (past the 28000-cycle LCFR discount window).
UPDATE_THRESHOLD=${UPDATE_THRESHOLD:-42000}
STRATEGY_INTERVAL=${STRATEGY_INTERVAL:-1}
# Checkpoint every 3 hours (21000 cycles = 3.0h at 7,000 cycles/h).  Every
# checkpoint is now RETAINED (previous generations are no longer deleted) and
# doubles as a post-flop average-strategy snapshot for the offline
# `poker_ai train average` tool, so this interval is also the snapshot cadence.
# 3h over the 96h run yields ~30 retained snapshots for the average.
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-21000}
# Start checkpointing/snapshotting after the first 6 hours — the
# average-strategy warm-up.  42000 cycles = 6.0h at 7,000 cycles/h (past the
# 28000-cycle / 4h LCFR discount window), so the retained snapshots exclude the
# near-random early era that pollutes the average.  With the interval above the
# first snapshot lands exactly at 6h (gate is ``sync_step >= start``), then one
# every 3h.  An end-of-run / SIGTERM checkpoint is always written regardless of
# this gate.
CHECKPOINT_START_CYCLES=${CHECKPOINT_START_CYCLES:-42000}
# CFR-P pruning begins at 2.5h.  Raw iterations (traversals-per-player): 2.5h * 7e6/h = 17.5M.
PRUNE_THRESHOLD=${PRUNE_THRESHOLD:-17500000}
C=${C:--3000000}
PICKLE_DIR=${PICKLE_DIR:-false}
N_PROCESSES=${N_PROCESSES:-}
LUT_PATH=${LUT_PATH:-"$WORKSPACE/exact"}
# Bias / warm-start (for biased-blueprint training).  BIAS=none runs the
# standard base blueprint and ignores BIAS_MAGNITUDE / WARM_START.
BIAS=${BIAS:-none}
BIAS_MAGNITUDE=${BIAS_MAGNITUDE:-100}
WARM_START=${WARM_START:-}
# Default save dir varies by bias so concurrent biased runs don't
# collide.  base → ${N}player_52cards; biased → ..._${BIAS}_biased.
if [ "$BIAS" = "none" ]; then
  NICKNAME=${NICKNAME:-"$WORKSPACE/models/${N_PLAYERS}player_52cards"}
else
  NICKNAME=${NICKNAME:-"$WORKSPACE/models/${N_PLAYERS}player_52cards_${BIAS}_biased"}
fi

# Efficiency knobs honoured by the trainer (env-driven so they can be
# overridden per submission without editing code).
export PLURIBUS_CFR_BATCH_SIZE=${PLURIBUS_CFR_BATCH_SIZE:-5}
# Pre-flop UPDATE-STRATEGY playthroughs folded into EACH cfr job (per player,
# after warm-up), interleaved with CFR and flushed with the regret delta at the
# sync barrier.  The average strategy is now pre-flop only with full opponent
# branching, so a single pass already covers the whole pre-flop opponent tree
# per deal — 1 is plenty (the old 30 and its auto-sizing fed the abandoned
# post-flop average).  The post-flop blueprint comes from offline snapshot
# averaging (`poker_ai train average`), not this pass.
export PLURIBUS_STRATEGY_PER_JOB=${PLURIBUS_STRATEGY_PER_JOB:-1}
export PLURIBUS_CHUNK_SIZE=${PLURIBUS_CHUNK_SIZE:-4000000}
# Shared-memory index cache: serves the per-node info-set lookup from shm
# instead of an LMDB read txn (the dominant inner-loop cost).  Capacities are
# per-street SLOT counts (pre_flop,flop,turn,river), each 24 bytes; a street
# holds up to capacity*0.5 infosets before it overflows.  These MUST cover the
# run's saturation — the mmap cannot grow once workers fork, so an undersized
# street fails LOUDLY and early (before real compute is spent), telling you to
# raise the value and restart.  The chosen sizes are persisted in the
# checkpoint so a resume reuses them.  Defaults ≈ 21 GiB of shm; raise --mem
# accordingly (chunk tables need the rest of RAM).
export PLURIBUS_INDEX_CACHE=${PLURIBUS_INDEX_CACHE:-1}
export PLURIBUS_INDEX_CAPACITY=${PLURIBUS_INDEX_CAPACITY:-"67108864,268435456,268435456,268435456"}
# Compiled Cython core for the CFR hot loop.  1 = every worker drives its
# traversals through poker_ai._core (byte-verified, fallback-retaining); set 0
# to force the pure-Python path (e.g. the A/B baseline arm).  REQUIRES the shm
# index cache above (pure-shm reads, no LMDB fallback) — already default-on —
# and the extension to be BUILT on this node (preflight check below).
export PLURIBUS_CFR_CORE=${PLURIBUS_CFR_CORE:-1}
# Deferred-durability allocation.  1 = the shm index cache assigns info-set row
# numbers under its own lightweight lock and LMDB is written in bulk only at
# checkpoints — taking the LMDB single-writer mutex off the allocation hot path
# (the measured 48-worker bottleneck once the compiled core is on).  Byte-
# identical single-process (golden-trace gated), fallback-retaining.  REQUIRES
# the shm index cache above (it IS the allocator); enforced by the guard below.
# Pure-Python — needs no build.  Set 0 to force the legacy per-infoset LMDB
# write-txn allocator (e.g. the A/B baseline arm).
export PLURIBUS_DEFERRED_ALLOC=${PLURIBUS_DEFERRED_ALLOC:-1}

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$(dirname "$NICKNAME")"

# Activate conda 
echo "Activating conda environment: $CONDA_ENV"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

cd "$PROJECT_DIR"

# Preflight: when the compiled core is requested it MUST be built on this node.
# CoreDriver imports the extension in each worker; a missing/stale .so would
# otherwise surface as a worker crash mid-run. Fail here with the build command
# instead. (Build once before submitting; do not build inside the job if
# PROJECT_DIR is shared across concurrent jobs — the .so write would race.)
if [ "$PLURIBUS_CFR_CORE" = "1" ]; then
  if ! python -c "import poker_ai._core._state, poker_ai._core._traverse" 2>/dev/null; then
    echo "ERROR: PLURIBUS_CFR_CORE=1 but the compiled core is not importable." >&2
    echo "       Build it on this node first:  python setup.py build_ext --inplace" >&2
    echo "       (or set PLURIBUS_CFR_CORE=0 to run the pure-Python path)." >&2
    exit 1
  fi
  echo "Compiled CFR core present and importable."
fi

# Deferred allocation IS the shm cache (it allocates rows there), so it is
# meaningless without it.  CFRTables silently disables deferred when the cache
# is off; catch the inconsistent request loudly instead of silently reverting to
# the slow LMDB-writer-mutex allocator.  No build check — deferred is pure Python.
if [ "$PLURIBUS_DEFERRED_ALLOC" = "1" ] && [ "$PLURIBUS_INDEX_CACHE" != "1" ]; then
  echo "ERROR: PLURIBUS_DEFERRED_ALLOC=1 requires PLURIBUS_INDEX_CACHE=1 (the shm" >&2
  echo "       cache is the allocator). Enable the cache or set DEFERRED_ALLOC=0." >&2
  exit 1
fi

# Ensure LUT path exists
if [ ! -d "$LUT_PATH" ]; then
  echo "ERROR: LUT path not found at $LUT_PATH" >&2
  echo "Please run abstraction first using: sbatch scripts/abstraction_auto_resub.sh" >&2
  exit 1
fi

# Per-job private working directory under TMPDIR.  Cluster doc says
# always use TMPDIR; whether TMPDIR is per-job or shared across
# concurrent users on the same node is implementation-defined, so we
# isolate our staged data under a job-specific subdirectory and
# tear the whole thing down on exit.  Everything below (LUT, LMDB,
# any future local artefacts) hangs off WORK_DIR.
WORK_DIR="${TMPDIR:?cluster requires TMPDIR to be set (do not fall back to /tmp)}/pluribus-${SLURM_JOB_ID:-$$}"
mkdir -p "$WORK_DIR"

# Stage the LUT to node-local fast scratch.  Without this, every river
# memmap lookup that misses the page cache becomes a network-FS round
# trip (the LUT typically lives on shared /pfs storage).  Set
# STAGE_LUT_LOCALLY=false to disable when local disk is too small.
STAGE_LUT_LOCALLY=${STAGE_LUT_LOCALLY:-true}
if [ "$STAGE_LUT_LOCALLY" = "true" ]; then
  LOCAL_LUT_PATH="$WORK_DIR/lut"
  echo "Staging LUT from $LUT_PATH to $LOCAL_LUT_PATH ..."
  mkdir -p "$LOCAL_LUT_PATH"
  rsync_start=$(date +%s)
  rsync -a "$LUT_PATH/" "$LOCAL_LUT_PATH/"
  rsync_end=$(date +%s)
  echo "LUT staged in $((rsync_end - rsync_start))s ($(du -sh "$LOCAL_LUT_PATH" | cut -f1))"
  LUT_PATH="$LOCAL_LUT_PATH"
fi

# Stage the LMDB indexes to node-local fast scratch.  Every CFR
# infoset lookup goes through ``InfosetIndex.get`` which opens an
# LMDB read transaction; on network FS the per-txn ``fcntl`` lock
# acquisition under 31 concurrent workers becomes a meaningful
# fraction of wall time.  The python-side ``CheckpointManager``
# mirrors the local LMDB back to ${NICKNAME}/lmdb_index at every
# checkpoint using ``env.copy`` + rsync, so a crash can never lose
# more than the most recent ``CHECKPOINT_INTERVAL`` of index data.
STAGE_LMDB_LOCALLY=${STAGE_LMDB_LOCALLY:-true}
if [ "$STAGE_LMDB_LOCALLY" = "true" ]; then
  LOCAL_LMDB_PATH="$WORK_DIR/lmdb"
  PERSISTENT_LMDB_PATH="$NICKNAME/lmdb_index"
  mkdir -p "$LOCAL_LMDB_PATH"
  if [ -d "$PERSISTENT_LMDB_PATH" ]; then
    echo "Staging LMDB from $PERSISTENT_LMDB_PATH to $LOCAL_LMDB_PATH ..."
    rsync_start=$(date +%s)
    rsync -a "$PERSISTENT_LMDB_PATH/" "$LOCAL_LMDB_PATH/"
    echo "LMDB staged in $(($(date +%s) - rsync_start))s ($(du -sh "$LOCAL_LMDB_PATH" 2>/dev/null | cut -f1))"
  else
    echo "No existing LMDB at $PERSISTENT_LMDB_PATH — starting fresh in $LOCAL_LMDB_PATH"
  fi
  export PLURIBUS_LMDB_LOCAL_DIR="$LOCAL_LMDB_PATH"
fi

# Cleanup: runs on any exit (clean or signalled).  Just removes the
# job-private working directory.  No safety-net rsync to persistent
# storage — the python ``CheckpointManager`` mirrors the LMDB at
# every checkpoint while the run is alive, so the persistent state
# after any crash is exactly the most recent successful checkpoint
# (chunks + LMDB together).  Anything since that checkpoint is at
# most ``CHECKPOINT_INTERVAL`` of training and is preferable to lose
# rather than risk a divergence between persistent LMDB and chunks.
trap 'rm -rf "$WORK_DIR"' EXIT

echo "Starting training with:"
echo "  - Players:                     $N_PLAYERS"
echo "  - Max runtime (hours):         $MAX_RUNTIME_HOURS"
echo "  - Sync interval (iters):       $SYNC_INTERVAL"
echo "  - Discount interval (cycles):  $DISCOUNT_INTERVAL"
echo "  - Discount duration (cycles):  $DISCOUNT_DURATION_CYCLES"
echo "  - Update threshold (cycles):   $UPDATE_THRESHOLD"
echo "  - Strategy interval (cycles):  $STRATEGY_INTERVAL"
echo "  - Checkpoint interval (cycles):$CHECKPOINT_INTERVAL"
echo "  - Checkpoint start (cycles):   $CHECKPOINT_START_CYCLES"
echo "  - Prune threshold (iters):     $PRUNE_THRESHOLD"
echo "  - C (pruning regret):          $C"
echo "  - Pickle dir:                  $PICKLE_DIR"
echo "  - N processes:                 ${N_PROCESSES:-(auto)}"
echo "  - LUT path:                    $LUT_PATH"
echo "  - Nickname:                    $NICKNAME"
echo "  - Bias:                        $BIAS"
if [ "$BIAS" != "none" ]; then
  echo "  - Bias magnitude:              $BIAS_MAGNITUDE"
fi
echo "  - Warm start:                  ${WARM_START:-(none)}"
echo "  - CPUs:                        $SLURM_CPUS_PER_TASK"
echo "  - PLURIBUS_CFR_BATCH_SIZE:     $PLURIBUS_CFR_BATCH_SIZE"
echo "  - PLURIBUS_STRATEGY_PER_JOB:   ${PLURIBUS_STRATEGY_PER_JOB:-1}"
echo "  - PLURIBUS_CFR_CORE:           $PLURIBUS_CFR_CORE"
echo "  - PLURIBUS_DEFERRED_ALLOC:     $PLURIBUS_DEFERRED_ALLOC"
echo "  - PLURIBUS_INDEX_CACHE:        $PLURIBUS_INDEX_CACHE"
echo "  - PLURIBUS_INDEX_CAPACITY:     $PLURIBUS_INDEX_CAPACITY"
echo "  - PLURIBUS_CHUNK_SIZE:         $PLURIBUS_CHUNK_SIZE"
echo "  - STAGE_LUT_LOCALLY:           $STAGE_LUT_LOCALLY"
echo "  - STAGE_LMDB_LOCALLY:          $STAGE_LMDB_LOCALLY"
echo "  - PLURIBUS_LMDB_LOCAL_DIR:     ${PLURIBUS_LMDB_LOCAL_DIR:-(unset)}"

# Build optional flags
EXTRA_ARGS=()
[ -n "$N_PROCESSES" ]      && EXTRA_ARGS+=(--n_processes "$N_PROCESSES")
[ "$PICKLE_DIR" = "true" ] && EXTRA_ARGS+=(--pickle_dir)
if [ "$BIAS" != "none" ]; then
  EXTRA_ARGS+=(--bias "$BIAS" --bias_magnitude "$BIAS_MAGNITUDE")
  if [ -z "$WARM_START" ]; then
    echo "ERROR: BIAS=$BIAS requires WARM_START to point at a finished base blueprint." >&2
    exit 1
  fi
fi
if [ -n "$WARM_START" ]; then
  if [ ! -d "$WARM_START" ]; then
    echo "ERROR: WARM_START path not found: $WARM_START" >&2
    exit 1
  fi
  EXTRA_ARGS+=(--warm_start "$WARM_START")
fi

# Run the trainer in the background so this shell can forward
# slurm's grace-period SIGTERM to the python process.  When SLURM
# signals a batch job, the signal goes to the bash wrapper — not to
# its foreground child.  Without explicit forwarding the trainer
# keeps running as an orphan, never sees SIGTERM, and gets
# SIGKILL'd at the wall-clock limit with no chance to write a final
# checkpoint.
poker_ai train start \
  --multi_process \
  --n_players "$N_PLAYERS" \
  --max_runtime_hours "$MAX_RUNTIME_HOURS" \
  --sync_interval "$SYNC_INTERVAL" \
  --discount_interval "$DISCOUNT_INTERVAL" \
  --discount_duration_cycles "$DISCOUNT_DURATION_CYCLES" \
  --update_threshold "$UPDATE_THRESHOLD" \
  --strategy_interval "$STRATEGY_INTERVAL" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --checkpoint_start_cycles "$CHECKPOINT_START_CYCLES" \
  --prune_threshold "$PRUNE_THRESHOLD" \
  --c "$C" \
  --lut_path "$LUT_PATH" \
  --nickname "$NICKNAME" \
  "${EXTRA_ARGS[@]}" &
TRAINER_PID=$!

_forward_signal() {
  local sig=$1
  echo "[batch] received SIG${sig} — forwarding to trainer (pid=${TRAINER_PID})"
  kill -"${sig}" "${TRAINER_PID}" 2>/dev/null || true
}
trap '_forward_signal TERM' TERM
trap '_forward_signal INT' INT

# `wait` returns when interrupted by a signal (after running the
# trap), even if the child is still running.  Loop until the child
# has actually exited so the EXIT trap (which cleans up the local
# LUT copy) only fires *after* the trainer's final checkpoint
# has finished.
set +e
while true; do
  wait "${TRAINER_PID}"
  EXIT_CODE=$?
  kill -0 "${TRAINER_PID}" 2>/dev/null || break
done
set -e
exit "${EXIT_CODE}"
