#!/bin/bash -l
# Slurm submission script to run training via the package CLI.
# Usage:
#   sbatch training.sh
#SBATCH --job-name=pluribus-train
#SBATCH --output=logs/training-%j.out
#SBATCH --error=logs/training-%j_error.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=50000mb
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

# Training parameters (cycle-based options are counted in sync cycles = N * sync_interval iterations)
N_PLAYERS=${N_PLAYERS:-6}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-71.5}
SYNC_INTERVAL=${SYNC_INTERVAL:-500}
DISCOUNT_INTERVAL=${DISCOUNT_INTERVAL:-50}
DISCOUNT_DURATION_CYCLES=${DISCOUNT_DURATION_CYCLES:-2000}
UPDATE_THRESHOLD=${UPDATE_THRESHOLD:-200}
STRATEGY_INTERVAL=${STRATEGY_INTERVAL:-5}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-150}
PRUNE_THRESHOLD=${PRUNE_THRESHOLD:-500000}
C=${C:--300000000}
PICKLE_DIR=${PICKLE_DIR:-false}
N_PROCESSES=${N_PROCESSES:-}
LUT_PATH=${LUT_PATH:-"$WORKSPACE/exact"}
NICKNAME=${NICKNAME:-"$WORKSPACE/models/${N_PLAYERS}player_52cards"}

# Efficiency knobs honoured by the trainer (env-driven so they can be
# overridden per submission without editing code).
export PLURIBUS_CFR_BATCH_SIZE=${PLURIBUS_CFR_BATCH_SIZE:-5}
export PLURIBUS_CHUNK_SIZE=${PLURIBUS_CHUNK_SIZE:-4000000}

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$(dirname "$NICKNAME")"

# Activate conda 
echo "Activating conda environment: $CONDA_ENV"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

cd "$PROJECT_DIR"

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
echo "  - Prune threshold (iters):     $PRUNE_THRESHOLD"
echo "  - C (pruning regret):          $C"
echo "  - Pickle dir:                  $PICKLE_DIR"
echo "  - N processes:                 ${N_PROCESSES:-(auto)}"
echo "  - LUT path:                    $LUT_PATH"
echo "  - Nickname:                    $NICKNAME"
echo "  - CPUs:                        $SLURM_CPUS_PER_TASK"
echo "  - PLURIBUS_CFR_BATCH_SIZE:     $PLURIBUS_CFR_BATCH_SIZE"
echo "  - PLURIBUS_CHUNK_SIZE:         $PLURIBUS_CHUNK_SIZE"
echo "  - STAGE_LUT_LOCALLY:           $STAGE_LUT_LOCALLY"
echo "  - STAGE_LMDB_LOCALLY:          $STAGE_LMDB_LOCALLY"
echo "  - PLURIBUS_LMDB_LOCAL_DIR:     ${PLURIBUS_LMDB_LOCAL_DIR:-(unset)}"

# Build optional flags
EXTRA_ARGS=()
[ -n "$N_PROCESSES" ]      && EXTRA_ARGS+=(--n_processes "$N_PROCESSES")
[ "$PICKLE_DIR" = "true" ] && EXTRA_ARGS+=(--pickle_dir)

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
