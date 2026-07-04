#!/bin/bash -l
# Auto-resubmitting Slurm script for long-running abstraction jobs.
# The job resubmits itself (up to MAX_RESUBMISSIONS times) until abstraction is complete.
# Usage:
#   sbatch abstraction_auto_resub.sh
#SBATCH --job-name=pluribus-abstraction
#SBATCH --output=logs/abstraction-%j.out
#SBATCH --error=logs/abstraction-%j_error.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --cpus-per-task=64
#SBATCH --mem=350000mb
#SBATCH --mail-type=ALL


set -euo pipefail

# ---------- Configuration ----------
CONDA_ENV=${CONDA_ENV:-pluribus}
PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
WORKERS=${WORKERS:-63}  # leave 1 CPU free for the main process and OS
if [ -z "${WORKSPACE:-}" ]; then
  echo "ERROR: WORKSPACE is not set. Export WORKSPACE=/path/to/workspace before submitting (e.g. sbatch --export=ALL,WORKSPACE=...)." >&2
  exit 1
fi
SUBMIT_SCRIPT=${SUBMIT_SCRIPT:-$(readlink -f "${BASH_SOURCE[0]}")}

LOW_CARD_RANK=${LOW_CARD_RANK:-2}
HIGH_CARD_RANK=${HIGH_CARD_RANK:-14}
N_RIVER_CLUSTERS=${N_RIVER_CLUSTERS:-100}
N_TURN_CLUSTERS=${N_TURN_CLUSTERS:-100}
N_FLOP_CLUSTERS=${N_FLOP_CLUSTERS:-200}
N_SIMULATIONS_RIVER=${N_SIMULATIONS_RIVER:-200}
CHUNK_SIZE=${CHUNK_SIZE:-100000}
MAX_RESUBMISSIONS=${MAX_RESUBMISSIONS:-2}
# Computation method: monte_carlo or exact (exact ignores N_SIMULATIONS_*).
METHOD=${METHOD:-exact}

# Always derived so WORKSPACE and METHOD can't drift apart.
SAVE_DIR="$WORKSPACE/$METHOD"
CARD_INFO_LUT="$SAVE_DIR/card_info_lut.joblib"
COUNTER_FILE="$SAVE_DIR/.resubmit_counter"

mkdir -p "$PROJECT_DIR/logs" "$SAVE_DIR"

# ---------- Helpers ----------

# Returns 0 iff the card_info_lut exists and contains all required streets.
is_complete() {
  [ -f "$CARD_INFO_LUT" ] || return 1
  python3 - "$CARD_INFO_LUT" <<'PY' 2>/dev/null
import joblib, sys
try:
    lut = joblib.load(sys.argv[1])
    sys.exit(0 if all(s in lut for s in ('pre_flop', 'river', 'turn', 'flop')) else 1)
except Exception:
    sys.exit(1)
PY
}

read_counter() {
  [ -f "$COUNTER_FILE" ] && cat "$COUNTER_FILE" || echo 0
}

# Resubmit via sbatch if under the cap; otherwise print a notice. Never fails.
try_resubmit() {
  local reason="$1"
  local count
  count=$(read_counter)
  if [ "$count" -lt "$MAX_RESUBMISSIONS" ]; then
    local next=$((count + 1))
    echo "$next" > "$COUNTER_FILE"
    echo "⟳ $reason — resubmitting ($next/$MAX_RESUBMISSIONS)..."
    sbatch "$SUBMIT_SCRIPT"
  else
    echo "⊗ $reason — max resubmissions ($MAX_RESUBMISSIONS) reached. To reset: rm $COUNTER_FILE"
  fi
}

# SIGUSR1 handler: fired ~5min before wall-time. Resubmit, then stop the worker.
on_pre_timeout() {
  echo "⟳ Received pre-timeout signal from Slurm."
  try_resubmit "pre-timeout"
  [ -n "${CLUSTER_PID:-}" ] && kill "$CLUSTER_PID" 2>/dev/null || true
  exit 0
}
trap 'on_pre_timeout' USR1

# ---------- Main ----------

echo "Activating conda environment: $CONDA_ENV"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"

if is_complete; then
  echo "✓ Abstraction already complete at $SAVE_DIR. Nothing to do."
  exit 0
fi

echo "Starting/resuming abstraction with $WORKERS workers..."
echo "Configuration:"
echo "  Method:          $METHOD"
echo "  Cards:           $LOW_CARD_RANK-$HIGH_CARD_RANK"
echo "  Clusters:        River=$N_RIVER_CLUSTERS, Turn=$N_TURN_CLUSTERS, Flop=$N_FLOP_CLUSTERS"
[ "$METHOD" = "monte_carlo" ] && echo "  Simulations:     River=$N_SIMULATIONS_RIVER"
echo "  Chunk size:      $CHUNK_SIZE"
echo "  Save directory:  $SAVE_DIR"

# Run in background so the USR1 trap can fire during execution.
poker_ai build-abstraction \
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
wait "$CLUSTER_PID" || true  # don't trip set -e if the trap killed the child

if is_complete; then
  echo "✓ Abstraction completed successfully! Results in: $SAVE_DIR"
  rm -f "$COUNTER_FILE"
  exit 0
fi

try_resubmit "abstraction not complete after run"
