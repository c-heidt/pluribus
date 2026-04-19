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

# Training parameters
N_PLAYERS=${N_PLAYERS:-6}
UPDATE_THRESHOLD=${UPDATE_THRESHOLD:-50000}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-71.5}
DISCOUNT_DURATION_ITERS=${DISCOUNT_DURATION_ITERS:-250000}
STRATEGY_INTERVAL=${STRATEGY_INTERVAL:-25000}
SYNC_INTERVAL=${SYNC_INTERVAL:-1000}
DISCOUNT_INTERVAL=${DISCOUNT_INTERVAL:-5}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-100000}
PRUNE_THRESHOLD=${PRUNE_THRESHOLD:-125000}
C=${C:--300000000}
DUMP_ITERATION=${DUMP_ITERATION:-1000}
PICKLE_DIR=${PICKLE_DIR:-false}
N_PROCESSES=${N_PROCESSES:-}
LUT_PATH=${LUT_PATH:-"$WORKSPACE/exact"}
NICKNAME=${NICKNAME:-"$WORKSPACE/models/6player_52cards"}

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
  echo "Please run clustering first using: sbatch cluster.sh" >&2
  exit 1
fi

echo "Starting training with:"
echo "  - Players:                $N_PLAYERS"
echo "  - Update threshold:       $UPDATE_THRESHOLD"
echo "  - Max runtime (hours):    $MAX_RUNTIME_HOURS"
echo "  - Discount duration iters:$DISCOUNT_DURATION_ITERS"
echo "  - Strategy interval:      $STRATEGY_INTERVAL"
echo "  - Sync interval:          $SYNC_INTERVAL"
echo "  - Discount interval:      $DISCOUNT_INTERVAL"
echo "  - Checkpoint interval:    $CHECKPOINT_INTERVAL"
echo "  - Prune threshold:        $PRUNE_THRESHOLD"
echo "  - C (pruning regret):     $C"
echo "  - Dump iteration:         $DUMP_ITERATION"
echo "  - Pickle dir:             $PICKLE_DIR"
echo "  - N processes:            ${N_PROCESSES:-(auto)}"
echo "  - LUT path:               $LUT_PATH"
echo "  - Nickname:               $NICKNAME"
echo "  - CPUs:                   $SLURM_CPUS_PER_TASK"

# Build optional flags
EXTRA_ARGS=()
if [ -n "$N_PROCESSES" ]; then
  EXTRA_ARGS+=(--n_processes "$N_PROCESSES")
fi
if [ "$PICKLE_DIR" = "true" ]; then
  EXTRA_ARGS+=(--pickle_dir)
fi

# Run training using the installed CLI with multiprocessing
poker_ai train start \
  --multi_process \
  --n_players "$N_PLAYERS" \
  --update_threshold "$UPDATE_THRESHOLD" \
  --max_runtime_hours "$MAX_RUNTIME_HOURS" \
  --discount_duration_iters "$DISCOUNT_DURATION_ITERS" \
  --strategy_interval "$STRATEGY_INTERVAL" \
  --sync_interval "$SYNC_INTERVAL" \
  --discount_interval "$DISCOUNT_INTERVAL" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --prune_threshold "$PRUNE_THRESHOLD" \
  --c "$C" \
  --dump_iteration "$DUMP_ITERATION" \
  --lut_path "$LUT_PATH" \
  --nickname "$NICKNAME" \
  "${EXTRA_ARGS[@]}"
