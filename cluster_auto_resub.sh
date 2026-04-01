#!/bin/bash -l
# Auto-resubmitting Slurm script for long-running clustering jobs.
# The job will automatically resubmit itself every 72 hours until clustering is complete.
# Usage:
#   sbatch cluster_auto_resub.sh
#SBATCH --job-name=pluribus-cluster
#SBATCH --output=logs/cluster-%j.out
#SBATCH --error=logs/cluster-%j_error.out
#SBATCH --partition=highmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000000mb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=uvizo@student.kit.edu


set -euo pipefail

# User-configurable
CONDA_ENV=${CONDA_ENV:-pluribus}
PROJECT_DIR=${PROJECT_DIR:-"$HOME/pluribus"}
WORKERS=${WORKERS:-60}
WORKSPACE=${WORKSPACE:-/pfs/work9/workspace/scratch/ka_gu4593-clustering_52}  # Workspace path!
SAVE_DIR=${SAVE_DIR:-"$WORKSPACE/monte_carlo"}

# Clustering parameters (adjust as needed)
LOW_CARD_RANK=${LOW_CARD_RANK:-2}
HIGH_CARD_RANK=${HIGH_CARD_RANK:-14}
N_RIVER_CLUSTERS=${N_RIVER_CLUSTERS:-200}
N_TURN_CLUSTERS=${N_TURN_CLUSTERS:-200}
N_FLOP_CLUSTERS=${N_FLOP_CLUSTERS:-200}
N_SIMULATIONS_RIVER=${N_SIMULATIONS_RIVER:-200}
CHUNK_SIZE=${CHUNK_SIZE:-25000}
MAX_RESUBMISSIONS=${MAX_RESUBMISSIONS:-2}
# Computation method: monte_carlo or exact 
# If exact, N_SIMULATIONS_* parameters are ignored
METHOD=${METHOD:-monte_carlo}

mkdir -p "$PROJECT_DIR/logs"
[ -d "$SAVE_DIR" ] || mkdir -p "$SAVE_DIR"

# Counter file to track resubmissions
COUNTER_FILE="$SAVE_DIR/.resubmit_counter"

# Handler called by Slurm ~5 min before the wall-time limit via SIGUSR1.
# Resubmits the job so the next run resumes from the checkpoint.
resubmit_on_timeout() {
  echo "⟳ Received pre-timeout signal from Slurm."
  if [ -f "$COUNTER_FILE" ]; then
    RESUBMIT_COUNT=$(cat "$COUNTER_FILE")
  else
    RESUBMIT_COUNT=0
  fi
  if [ "$RESUBMIT_COUNT" -lt "$MAX_RESUBMISSIONS" ]; then
    NEW_COUNT=$((RESUBMIT_COUNT + 1))
    echo "$NEW_COUNT" > "$COUNTER_FILE"
    echo "⟳ Resubmitting job ($NEW_COUNT/$MAX_RESUBMISSIONS)..."
    sbatch "$PROJECT_DIR/cluster_auto_resub.sh"
  else
    echo "⊗ Maximum resubmissions ($MAX_RESUBMISSIONS) reached. Not resubmitting."
  fi
  # Gracefully stop the background clustering process before Slurm kills everything
  [ -n "${CLUSTER_PID:-}" ] && kill "$CLUSTER_PID" 2>/dev/null || true
  exit 0
}
trap 'resubmit_on_timeout' USR1

# Activate conda
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base 2>/dev/null || true)
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    source <(conda shell.bash hook)
    conda activate "$CONDA_ENV"
  fi
  echo "Activated conda env: $CONDA_ENV"
else
  echo "Conda not found on PATH; ensure conda is available." >&2
  exit 1
fi

cd "$PROJECT_DIR"

# Check if clustering is complete
CHECKPOINT_FILE="$SAVE_DIR/checkpoint.json"
CARD_INFO_LUT="$SAVE_DIR/card_info_lut.joblib"

if [ -f "$CARD_INFO_LUT" ]; then
  echo "Checking if clustering is complete..."
  # Check if all streets are done in the card_info_lut
  if python3 -c "
import joblib
import sys
try:
    lut = joblib.load('$CARD_INFO_LUT')
    required_streets = ['pre_flop', 'river', 'turn', 'flop']
    if all(street in lut for street in required_streets):
        print('✓ Clustering is complete!')
        sys.exit(0)
    else:
        missing = [s for s in required_streets if s not in lut]
        print(f'⟳ Clustering incomplete. Missing: {missing}')
        sys.exit(1)
except Exception as e:
    print(f'⟳ Clustering incomplete or error: {e}')
    sys.exit(1)
" 2>/dev/null; then
    echo "All streets completed. No resubmission needed."
    exit 0
  fi
fi

echo "Starting/resuming clustering with $WORKERS workers..."
echo "Configuration:"
echo "  Method: $METHOD"
echo "  Cards: $LOW_CARD_RANK-$HIGH_CARD_RANK"
echo "  Clusters: River=$N_RIVER_CLUSTERS, Turn=$N_TURN_CLUSTERS, Flop=$N_FLOP_CLUSTERS"
if [ "$METHOD" = "monte_carlo" ]; then
  echo "  Simulations: River=$N_SIMULATIONS_RIVER"
fi
echo "  Chunk size: $CHUNK_SIZE"
echo "  Save directory: $SAVE_DIR"

# Run clustering in background so the USR1 trap can fire while it is running.
poker_ai cluster \
  --save_dir "$SAVE_DIR" \
  --workers "$WORKERS" \
  --low_card_rank "$LOW_CARD_RANK" \
  --high_card_rank "$HIGH_CARD_RANK" \
  --n_river_clusters "$N_RIVER_CLUSTERS" \
  --n_turn_clusters "$N_TURN_CLUSTERS" \
  --n_flop_clusters "$N_FLOP_CLUSTERS" \
  --n_simulations_river "$N_SIMULATIONS_RIVER" \
  --chunk_size "$CHUNK_SIZE" \
  --method "$METHOD" &
CLUSTER_PID=$!
wait "$CLUSTER_PID" || true  # 'true' so set -e doesn't fire if we killed it

# Check if clustering completed
if [ -f "$CARD_INFO_LUT" ]; then
  if python3 -c "
import joblib
import sys
try:
    lut = joblib.load('$CARD_INFO_LUT')
    required_streets = ['pre_flop', 'river', 'turn', 'flop']
    if all(street in lut for street in required_streets):
        sys.exit(0)
    else:
        sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null; then
    echo "✓ Clustering completed successfully!"
    echo "Results are in: $SAVE_DIR"
    
    exit 0
  fi
fi

# If we got here, clustering is not complete - check if we should resubmit
# Read or initialize counter
if [ -f "$COUNTER_FILE" ]; then
  RESUBMIT_COUNT=$(cat "$COUNTER_FILE")
else
  RESUBMIT_COUNT=0
fi

if [ "$RESUBMIT_COUNT" -lt "$MAX_RESUBMISSIONS" ]; then
  NEW_COUNT=$((RESUBMIT_COUNT + 1))
  echo "$NEW_COUNT" > "$COUNTER_FILE"
  echo "⟳ Clustering not complete after run. Resubmitting job ($NEW_COUNT/$MAX_RESUBMISSIONS)..."
  sbatch "$PROJECT_DIR/cluster_auto_resub.sh"
else
  echo "⊗ Maximum resubmissions ($MAX_RESUBMISSIONS) reached."
  echo "⊗ Please check intermediate results and manually resubmit if needed."
  echo "⊗ To reset counter: rm $COUNTER_FILE"
  exit 0
fi