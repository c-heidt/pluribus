#!/bin/bash -l
# Minimal Slurm submission script to run training via the package CLI.
# Usage:
#   sbatch training.sh
#SBATCH --job-name=pluribus-train
#SBATCH --output=logs/training-%j.out
#SBATCH --error=logs/training-%j_error.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=38G
#SBATCH --mail-type=All
#SBATCH --mail-user=uvizo@student.kit.edu


set -euo pipefail

# User-configurable
CONDA_ENV=${CONDA_ENV:-pluribus}
PROJECT_DIR=${PROJECT_DIR:-"$HOME/pluribus"}
WORKSPACE=${WORKSPACE:-/pfs/work9/workspace/scratch/ka_gu4593-clustering_20}
N_PLAYERS=${N_PLAYERS:-2}
UPDATE_THRESHOLD=${UPDATE_THRESHOLD:-50}
N_ITERATIONS=${N_ITERATIONS:-1000}
DUMP_ITERATION=${DUMP_ITERATION:-10}
LUT_PATH=${LUT_PATH:-"$WORKSPACE/exact"}
NICKNAME=${NICKNAME:-"models/2player_20cards"}

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/models"

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
echo "  - Players: $N_PLAYERS"
echo "  - Update threshold: $UPDATE_THRESHOLD"
echo "  - Iterations: $N_ITERATIONS"
echo "  - Dump iteration: $DUMP_ITERATION"
echo "  - LUT path: $LUT_PATH"
echo "  - Nickname: $NICKNAME"
echo "  - CPUs: $SLURM_CPUS_PER_TASK"

# Run training using the installed CLI with multiprocessing
poker_ai train start \
  --multi_process \
  --n_players "$N_PLAYERS" \
  --update_threshold "$UPDATE_THRESHOLD" \
  --n_iterations "$N_ITERATIONS" \
  --dump_iteration "$DUMP_ITERATION" \
  --lut_path "$LUT_PATH" \
  --nickname "$NICKNAME"
