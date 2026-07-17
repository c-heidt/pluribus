#!/bin/bash -l
# Slurm submission script for the OFFLINE post-flop average-strategy build.
#
# After a training run finishes, its retained ``checkpoint_*`` generations are
# the post-flop strategy snapshots (Pluribus-style).  This job runs
# ``poker_ai train average`` to regret-match each snapshot and average them into
# a single final blueprint whose post-flop strategy tables hold the averaged
# strategy, then leaves a directory the evaluation / search stack loads unchanged.
#
# Usage:
#   sbatch --export=ALL,WORKSPACE=/path/to/ws strategy_average.sh
#   # explicit dirs:
#   sbatch --export=ALL,WORKSPACE=/path/to/ws,TRAIN_DIR=/ws/models/4player_52cards,OUTPUT_DIR=/ws/models/4player_52cards_blueprint strategy_average.sh
#
# MEMORY: the averager is STREAMING — each (street, chunk) task holds one
# snapshot chunk plus one float64 accumulator (~800 MB at the default 4M-row
# PLURIBUS_CHUNK_SIZE) and nothing else, independent of the run's total on-disk
# footprint.  Peak RAM is therefore WORKERS * ~800 MB, so --workers is the memory
# dial: even with >2 TB of retained checkpoints the job fits on one node.
# --mem below covers WORKERS=16 (~13 GB) plus headroom; scale both together.
#
# RUNTIME: the work is embarrassingly parallel per chunk and I/O-bound once the
# pool is wide.  Single-threaded on a 4p/200-100-100 run (~145 post-flop chunks
# per snapshot, ~0.7 s each) it is ~1.5-2.5 h; with WORKERS=16 it drops to the
# filesystem's read speed for the snapshot set.  If a job is cut short, resubmit
# with RESUME=true — completed chunks are skipped.
#SBATCH --job-name=pluribus-average
#SBATCH --output=logs/average-%j.out
#SBATCH --error=logs/average-%j_error.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=24000mb
#SBATCH --signal=SIGTERM@120
#SBATCH --mail-type=All


set -euo pipefail

# User-configurable
CONDA_ENV=${CONDA_ENV:-pluribus}
PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ -z "${WORKSPACE:-}" ]; then
  echo "ERROR: WORKSPACE is not set. Export WORKSPACE=/path/to/workspace before submitting (e.g. sbatch --export=ALL,WORKSPACE=...)." >&2
  exit 1
fi

# Which training run to average and where to write the final blueprint.  The
# defaults mirror training.sh's NICKNAME convention so a base run averages with
# no extra arguments.
N_PLAYERS=${N_PLAYERS:-4}
TRAIN_DIR=${TRAIN_DIR:-"$WORKSPACE/models/${N_PLAYERS}player_52cards"}
OUTPUT_DIR=${OUTPUT_DIR:-"${TRAIN_DIR}_blueprint"}

# Optional averaging knobs (leave unset for the tool defaults):
#   SCALE  — integer scale for the stored post-flop strategy pseudo-counts
#            (default 1,000,000; the readout normalises it away).
#   MIN_T  — exclude snapshots below this iteration t from the average
#            (default = the warm-up recorded in the latest checkpoint; pass 0 to
#            average every retained checkpoint).
SCALE=${SCALE:-}
MIN_T=${MIN_T:-}

# Concurrency / memory dial.  Peak RAM ≈ WORKERS * ~800 MB (one chunk in flight
# per worker), so keep WORKERS <= --cpus-per-task and --mem >= WORKERS*0.8 GB +
# headroom.  Defaults to the allocated CPU count.
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}
# Set RESUME=true to continue an interrupted build in an existing OUTPUT_DIR
# (completed chunks are skipped; every output is written atomically so anything
# present is finished).
RESUME=${RESUME:-false}

mkdir -p "$PROJECT_DIR/logs"

# Activate conda
echo "Activating conda environment: $CONDA_ENV"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

cd "$PROJECT_DIR"

# The averager is pure NumPy (no compiled core, no LUT), so no build/LUT
# preflight is needed — but the numpy element-wise ops it runs are single
# threaded; cap any BLAS pool so it does not oversubscribe the modest core
# allocation.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

# --- Preflight -------------------------------------------------------------
if [ ! -d "$TRAIN_DIR" ]; then
  echo "ERROR: TRAIN_DIR not found: $TRAIN_DIR" >&2
  exit 1
fi
if [ ! -d "$TRAIN_DIR/lmdb_index" ]; then
  echo "ERROR: $TRAIN_DIR has no lmdb_index/ — is it a training run directory?" >&2
  exit 1
fi
# There must be at least one retained checkpoint to average.
if ! compgen -G "$TRAIN_DIR/checkpoint_[0-9]*" > /dev/null; then
  echo "ERROR: no checkpoint_* generations under $TRAIN_DIR." >&2
  echo "       The run must retain checkpoints (CHECKPOINT_START_CYCLES gate in" >&2
  echo "       training.sh); a run that kept only one rolling checkpoint cannot" >&2
  echo "       be averaged into a post-flop blueprint." >&2
  exit 1
fi
N_SNAPSHOTS=$(compgen -G "$TRAIN_DIR/checkpoint_[0-9]*" | wc -l)

# The averager refuses to overwrite an existing blueprint unless resuming.
# Fail early with a clear message instead of deep in Python.
if [ -e "$OUTPUT_DIR/lmdb_index" ] && [ "$RESUME" != "true" ]; then
  echo "ERROR: $OUTPUT_DIR already contains a blueprint (lmdb_index present)." >&2
  echo "       Choose a fresh OUTPUT_DIR, remove it, or resubmit with" >&2
  echo "       RESUME=true to continue an interrupted build." >&2
  exit 1
fi

# NOTE: a partial output is deliberately NOT deleted on failure.  Every artefact
# is written atomically (temp + rename), so whatever is on disk is complete work
# — throwing it away would discard hours of averaging.  Resubmit with
# RESUME=true instead and the finished chunks are skipped.
DONE=0
_keep_partial_notice() {
  if [ "$DONE" -ne 1 ] && [ -d "$OUTPUT_DIR" ]; then
    echo "[exit] build did not complete — partial output KEPT at $OUTPUT_DIR" >&2
    echo "[exit] resubmit with RESUME=true to continue where it stopped:" >&2
    echo "       sbatch --export=ALL,WORKSPACE=$WORKSPACE,TRAIN_DIR=$TRAIN_DIR,OUTPUT_DIR=$OUTPUT_DIR,RESUME=true $0" >&2
  fi
}
trap _keep_partial_notice EXIT

echo "Building averaged blueprint with:"
echo "  - Train dir:        $TRAIN_DIR"
echo "  - Output dir:       $OUTPUT_DIR"
echo "  - Retained snaps:   $N_SNAPSHOTS  ($(du -sh --apparent-size "$TRAIN_DIR" 2>/dev/null | cut -f1) on disk)"
echo "  - Scale:            ${SCALE:-(default 1000000)}"
echo "  - Min t:            ${MIN_T:-(default: latest-checkpoint warm-up)}"
echo "  - CPUs:             ${SLURM_CPUS_PER_TASK:-4}"
echo "  - Workers:          $WORKERS  (peak RAM ≈ $((WORKERS * 800)) MB)"
echo "  - Resume:           $RESUME"

# Build optional flags
EXTRA_ARGS=()
[ -n "$SCALE" ] && EXTRA_ARGS+=(--scale "$SCALE")
[ -n "$MIN_T" ] && EXTRA_ARGS+=(--min_t "$MIN_T")
[ "$RESUME" = "true" ] && EXTRA_ARGS+=(--resume)

# Run in the background so this shell can forward slurm's grace-period SIGTERM
# to the python process (same rationale as training.sh: the signal hits the bash
# wrapper, not its foreground child).  On SIGTERM the averager aborts and the
# EXIT trap removes the partial output so the re-run starts clean.
poker_ai train average \
  --train_dir "$TRAIN_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --workers "$WORKERS" \
  "${EXTRA_ARGS[@]}" &
AVG_PID=$!

_forward_signal() {
  local sig=$1
  echo "[batch] received SIG${sig} — forwarding to averager (pid=${AVG_PID})"
  kill -"${sig}" "${AVG_PID}" 2>/dev/null || true
}
trap '_forward_signal TERM' TERM
trap '_forward_signal INT' INT

set +e
while true; do
  wait "${AVG_PID}"
  EXIT_CODE=$?
  kill -0 "${AVG_PID}" 2>/dev/null || break
done
set -e

if [ "${EXIT_CODE}" -eq 0 ]; then
  DONE=1
  echo "Averaged blueprint written to $OUTPUT_DIR"
  echo "Sanity-check it with:  python -m evaluation.blueprint_metrics --leaf-coverage $OUTPUT_DIR"
fi
exit "${EXIT_CODE}"
