#!/bin/bash -l
# Minimal Slurm submission script to run clustering via the package CLI.
# Usage:
#   sbatch cluster.sh
#SBATCH --job-name=pluribus-cluster
#SBATCH --output=logs/cluster-%j.out
#SBATCH --error=logs/cluster-%j_error.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=All
#SBATCH --mail-user=uvizo@student.kit.edu


set -euo pipefail

# User-configurable
CONDA_ENV=${CONDA_ENV:-pluribus}
PROJECT_DIR=${PROJECT_DIR:-"$HOME/pluribus"}
# Number of worker processes to use for clustering (defaults to 4)
WORKERS=${WORKERS:-4}

mkdir -p "$PROJECT_DIR/logs"

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

# Run clustering using the installed CLI
SAVE_DIR=${SAVE_DIR:-"$PROJECT_DIR/data/clustering"}
# Ensure save directory exists
mkdir -p "$SAVE_DIR"

echo "Running clustering with $WORKERS worker(s) and $SLURM_CPUS_PER_TASK CPU(s) allocated."
poker_ai cluster --save_dir "$SAVE_DIR" --workers "$WORKERS"

