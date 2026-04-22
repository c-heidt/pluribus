#!/bin/bash -l
# Slurm submission script to run training via the package CLI.
# Usage:
#   sbatch training.sh
#SBATCH --job-name=pluribus-train
#SBATCH --output=logs/training-%j.out
#SBATCH --error=logs/training-%j_error.out
#SBATCH --partition=highmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=1000000mb
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
DISCOUNT_INTERVAL=${DISCOUNT_INTERVAL:-5}
DISCOUNT_DURATION_CYCLES=${DISCOUNT_DURATION_CYCLES:-100}
UPDATE_THRESHOLD=${UPDATE_THRESHOLD:-60}
STRATEGY_INTERVAL=${STRATEGY_INTERVAL:-20}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-60}
PRUNE_THRESHOLD=${PRUNE_THRESHOLD:-50000}
C=${C:--300000000}
PICKLE_DIR=${PICKLE_DIR:-false}
N_PROCESSES=${N_PROCESSES:-}
LUT_PATH=${LUT_PATH:-"$WORKSPACE/exact"}
NICKNAME=${NICKNAME:-"$WORKSPACE/models/${N_PLAYERS}player_52cards"}

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

# Build optional flags
EXTRA_ARGS=()
[ -n "$N_PROCESSES" ]      && EXTRA_ARGS+=(--n_processes "$N_PROCESSES")
[ "$PICKLE_DIR" = "true" ] && EXTRA_ARGS+=(--pickle_dir)

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
  "${EXTRA_ARGS[@]}"
