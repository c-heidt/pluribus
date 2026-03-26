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
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=38G
#SBATCH --signal=SIGTERM@300
#SBATCH --mail-type=All
#SBATCH --mail-user=uvizo@student.kit.edu


set -euo pipefail

# User-configurable
CONDA_ENV=${CONDA_ENV:-pluribus}
PROJECT_DIR=${PROJECT_DIR:-"$HOME/pluribus"}
WORKSPACE=${WORKSPACE:-/pfs/work9/workspace/scratch/ka_gu4593-clustering_52}

# Training parameters (all correspond to poker_ai train start options)
N_PLAYERS=${N_PLAYERS:-6}
UPDATE_THRESHOLD=${UPDATE_THRESHOLD:-1000}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-0.75}
DISCOUNT_DURATION_ITERS=${DISCOUNT_DURATION_ITERS:-10000}
STRATEGY_INTERVAL=${STRATEGY_INTERVAL:-5000}
SYNC_INTERVAL=${SYNC_INTERVAL:-250}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-10000}
PRUNE_THRESHOLD=${PRUNE_THRESHOLD:-5000}
C=${C:--300000000}
DUMP_ITERATION=${DUMP_ITERATION:-500}
PICKLE_DIR=${PICKLE_DIR:-false}
N_PROCESSES=${N_PROCESSES:-}
LUT_PATH=${LUT_PATH:-"$PROJECT_DIR/data/clusterin/20cards_exact"}
NICKNAME=${NICKNAME:-"models/6player_20cards"}

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$(dirname "$NICKNAME")"

# Activate conda (prefer simple `conda activate` since conda is on PATH)
if command -v conda >/dev/null 2>&1; then
  # Source conda base explicitly (avoids "Run 'conda init' before 'conda activate'" in non-interactive shells)
  CONDA_BASE=$(conda info --base 2>/dev/null || true)
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    # Fallback: try shell hook then activate
    source <(conda shell.bash hook)
    conda activate "$CONDA_ENV"
  fi
  echo "Activated conda env: $CONDA_ENV (base: ${CONDA_BASE:-unknown})"
else
  echo "Conda not found on PATH; ensure conda is available." >&2
  exit 1
fi

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
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --prune_threshold "$PRUNE_THRESHOLD" \
  --c "$C" \
  --dump_iteration "$DUMP_ITERATION" \
  --lut_path "$LUT_PATH" \
  --nickname "$NICKNAME" \
  "${EXTRA_ARGS[@]}"
