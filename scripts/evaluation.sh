#!/bin/bash -l
# Slurm submission script to run a time-budgeted evaluation via the package CLI
# (docs/evaluation.md §5 cluster I/O, §10.1 runner).
#
# The design mirrors scripts/training.sh: the LUT **and the blueprint** are staged
# to node-local fast scratch, the SQLite sink is written node-local (so per-hand
# commits never touch the network FS), and a permanent-FS snapshot is produced by
# periodic + final ``VACUUM INTO`` sync-back.  SLURM's grace-period SIGTERM is
# forwarded to the python process so the runner stops at a hand boundary and writes
# a final snapshot before the wall-clock SIGKILL.  Analysis reads the permanent
# snapshot, never the live node-local file.
#
# RESOURCE FOOTPRINT (real-time search loads both artifacts on the node):
#   * Node-local disk (--tmp): the LUT (~250 GB) and the blueprint (~150 GB) are
#     both rsync'd to $TMPDIR, so the job needs ~420 GB of local scratch (the two
#     staged copies + the small SQLite db).  --tmp requests it; a preflight check
#     below aborts early if the node cannot hold both.
#   * RAM (--mem): the blueprint's per-street regret/strategy chunks are restored
#     into /dev/shm (tmpfs, RAM-backed) by CFRTables, so ~150 GB of the blueprint
#     is *resident* in addition to the working set of the solver and the LUT page
#     cache.  On most SLURM/cgroup setups /dev/shm usage counts against --mem, so
#     --mem must cover the resident blueprint (≈150 GB) plus headroom.  If the
#     node's /dev/shm is capped below the blueprint size independently of --mem,
#     the run cannot load the blueprint — raise the mount or use a bigger node.
# Both staged copies are read-only for eval (the runner never writes the blueprint
# or LUT back), so disable a stage with STAGE_*_LOCALLY=false to trade hot-path
# speed for local disk if a node is too small to hold both.
#
# Usage:
#   sbatch --export=ALL,WORKSPACE=/path/to/ws,RUN_ID=2026-07-01_6max_mix,\
#     BLUEPRINT_PATH=/path/to/blueprint scripts/evaluation.sh
#
#SBATCH --job-name=pluribus-eval
#SBATCH --output=logs/evaluation-%j.out
#SBATCH --error=logs/evaluation-%j_error.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=200000mb
#SBATCH --tmp=460000
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
if [ -z "${BLUEPRINT_PATH:-}" ]; then
  echo "ERROR: BLUEPRINT_PATH is not set. Export BLUEPRINT_PATH=/path/to/trained/blueprint before submitting." >&2
  exit 1
fi

# Run parameters (docs/evaluation.md §10.1 "Run config fields").
RUN_ID=${RUN_ID:-"eval-${SLURM_JOB_ID:-$$}"}
RUN_SEED=${RUN_SEED:-0}
TABLE_POLICY=${TABLE_POLICY:-all_blueprint}       # all_blueprint | random | fixed
FIXED_SEATS=${FIXED_SEATS:-}                       # JSON map, only for TABLE_POLICY=fixed
TIME_BUDGET_HOURS=${TIME_BUDGET_HOURS:-71.5}
N_PLAYERS=${N_PLAYERS:-6}
BIG_BLIND=${BIG_BLIND:-100}
SMALL_BLIND=${SMALL_BLIND:-50}
STARTING_STACK=${STARTING_STACK:-10000}
MAX_ITERATIONS=${MAX_ITERATIONS:-5000}
MAX_WALL_SECONDS=${MAX_WALL_SECONDS:-10.0}
WORKERS=${WORKERS:-}                               # solver replicas (§6.7); empty → auto
# Sync-back cadence (§5): every SYNC_INTERVAL_HANDS hands and/or SYNC_INTERVAL_MINUTES.
SYNC_INTERVAL_HANDS=${SYNC_INTERVAL_HANDS:-500}
SYNC_INTERVAL_MINUTES=${SYNC_INTERVAL_MINUTES:-15}
LUT_PATH=${LUT_PATH:-"$WORKSPACE/exact"}

# Permanent-FS destination for the snapshot + config.yaml (analysis reads this).
PERM_DIR=${PERM_DIR:-"$WORKSPACE/evaluation/$RUN_ID"}
PERM_SNAPSHOT="$PERM_DIR/experiment.sqlite"
mkdir -p "$PERM_DIR"

mkdir -p "$PROJECT_DIR/logs"

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
if [ ! -d "$BLUEPRINT_PATH" ]; then
  echo "ERROR: BLUEPRINT_PATH not found at $BLUEPRINT_PATH" >&2
  exit 1
fi

# Per-job private working directory under TMPDIR (same discipline as training.sh:
# always use TMPDIR, isolate under a job-specific subdir, tear down on exit).
WORK_DIR="${TMPDIR:?cluster requires TMPDIR to be set (do not fall back to /tmp)}/pluribus-eval-${SLURM_JOB_ID:-$$}"
mkdir -p "$WORK_DIR"

# Tear down the job-private scratch dir on ANY exit — armed here, immediately after
# creation, so it also fires on the preflight disk-space `exit 1` and on any
# `set -e` failure during staging (rsync/du/cp).  Arming it only after staging (as
# training.sh does) would leak a partial 250 GB+ copy on node-local scratch if a
# stage fails or the preflight aborts — exactly the "tmp dir not cleaned up" case.
# `rm -rf` on a not-yet-populated path is a harmless no-op.
trap 'rm -rf "$WORK_DIR"' EXIT

# Node-local SQLite sink.  All per-hand commits land here; the network FS only
# ever sees the periodic/final VACUUM INTO snapshot (§4/§5).
LOCAL_DB_PATH="$WORK_DIR/experiment.sqlite"

# Resume: if a prior run of this RUN_ID left a permanent snapshot, seed the
# node-local db from it so the (run_id, hand_index) cursor continues cleanly
# (§10.1) instead of restarting at hand 0.
if [ -f "$PERM_SNAPSHOT" ]; then
  echo "Resuming from existing snapshot $PERM_SNAPSHOT"
  cp "$PERM_SNAPSHOT" "$LOCAL_DB_PATH"
fi

STAGE_LUT_LOCALLY=${STAGE_LUT_LOCALLY:-true}
STAGE_BLUEPRINT_LOCALLY=${STAGE_BLUEPRINT_LOCALLY:-true}

# Preflight: the LUT (~250 GB) and blueprint (~150 GB) are large, so verify the
# node-local filesystem can hold everything we intend to stage BEFORE rsync starts
# — a half-staged copy that fills the disk mid-run is far worse than failing fast
# here.  Needed = sum of the sources we will stage + a margin for the db/WAL.
need_kb=0
if [ "$STAGE_LUT_LOCALLY" = "true" ]; then
  lut_kb=$(du -sk "$LUT_PATH" | cut -f1)
  need_kb=$((need_kb + lut_kb))
fi
if [ "$STAGE_BLUEPRINT_LOCALLY" = "true" ]; then
  bp_kb=$(du -sk "$BLUEPRINT_PATH" | cut -f1)
  need_kb=$((need_kb + bp_kb))
fi
need_kb=$((need_kb + 10 * 1024 * 1024))   # +10 GB margin (db + WAL + slack)
avail_kb=$(df -Pk "$WORK_DIR" | awk 'NR==2 {print $4}')
echo "Node-local scratch: need ~$((need_kb / 1024 / 1024)) GB, available ~$((avail_kb / 1024 / 1024)) GB at $WORK_DIR"
if [ "$avail_kb" -lt "$need_kb" ]; then
  echo "ERROR: insufficient node-local scratch to stage the LUT + blueprint." >&2
  echo "       Request more with a larger --tmp, use a node with bigger local disk," >&2
  echo "       or disable a stage (STAGE_LUT_LOCALLY=false / STAGE_BLUEPRINT_LOCALLY=false)." >&2
  exit 1
fi

# Stage the LUT to node-local fast scratch (same rationale as training.sh: avoid a
# network-FS round trip on every river memmap lookup that misses the page cache).
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

# Stage the blueprint to node-local fast scratch.  Real-time search hits the
# blueprint on the hot path — the per-street LMDB info-set index is queried on
# every leaf-fleet / opponent / hero-blueprint lookup, and its checkpoint chunks
# are restored into /dev/shm at startup — so a network-FS blueprint is a per-lookup
# round trip (index) plus a slow one-time ~150 GB restore read.  Eval is read-only
# against it (no write-back, unlike the training LMDB), so a plain rsync of the
# whole directory and a repointed --blueprint-path is all that is needed.
if [ "$STAGE_BLUEPRINT_LOCALLY" = "true" ]; then
  LOCAL_BLUEPRINT_PATH="$WORK_DIR/blueprint"
  echo "Staging blueprint from $BLUEPRINT_PATH to $LOCAL_BLUEPRINT_PATH ..."
  mkdir -p "$LOCAL_BLUEPRINT_PATH"
  rsync_start=$(date +%s)
  rsync -a "$BLUEPRINT_PATH/" "$LOCAL_BLUEPRINT_PATH/"
  rsync_end=$(date +%s)
  echo "Blueprint staged in $((rsync_end - rsync_start))s ($(du -sh "$LOCAL_BLUEPRINT_PATH" | cut -f1))"
  BLUEPRINT_PATH="$LOCAL_BLUEPRINT_PATH"
fi

# (The EXIT-trap cleanup of $WORK_DIR was armed right after its creation, above.)
# The `wait` loop at the end delays the actual exit — and thus this cleanup — until
# the python child has exited, so the runner's final VACUUM INTO has already written
# the permanent snapshot before node-local scratch is torn down.

echo "Starting evaluation with:"
echo "  - Run id:                 $RUN_ID"
echo "  - Run seed:               $RUN_SEED"
echo "  - Table policy:           $TABLE_POLICY"
echo "  - Time budget (hours):    $TIME_BUDGET_HOURS"
echo "  - Players:                $N_PLAYERS"
echo "  - Big/small blind:        $BIG_BLIND / $SMALL_BLIND"
echo "  - Starting stack:         $STARTING_STACK"
echo "  - Max iterations:         $MAX_ITERATIONS"
echo "  - Max wall seconds:       $MAX_WALL_SECONDS"
echo "  - Workers:                ${WORKERS:-(auto)}"
echo "  - Sync interval (hands):  $SYNC_INTERVAL_HANDS"
echo "  - Sync interval (mins):   $SYNC_INTERVAL_MINUTES"
echo "  - LUT path:               $LUT_PATH"
echo "  - Blueprint path:         $BLUEPRINT_PATH"
echo "  - Stage LUT locally:      $STAGE_LUT_LOCALLY"
echo "  - Stage blueprint local:  $STAGE_BLUEPRINT_LOCALLY"
echo "  - Node-local db:          $LOCAL_DB_PATH"
echo "  - Permanent snapshot:     $PERM_SNAPSHOT"
echo "  - CPUs:                   ${SLURM_CPUS_PER_TASK:-(unset)}"

# Build optional flags
EXTRA_ARGS=()
[ -n "$WORKERS" ]      && EXTRA_ARGS+=(--workers "$WORKERS")
[ -n "$FIXED_SEATS" ]  && EXTRA_ARGS+=(--fixed-seats "$FIXED_SEATS")

# Run the runner in the background so this shell can forward slurm's grace-period
# SIGTERM to the python process (same pattern as training.sh: a batch job's signal
# goes to the bash wrapper, not its foreground child; without forwarding the runner
# never sees SIGTERM and is SIGKILL'd with no chance to write the final snapshot).
python -m evaluation.runner run \
  --run-id "$RUN_ID" \
  --run-seed "$RUN_SEED" \
  --db-path "$LOCAL_DB_PATH" \
  --sync-path "$PERM_SNAPSHOT" \
  --sync-interval-hands "$SYNC_INTERVAL_HANDS" \
  --sync-interval-minutes "$SYNC_INTERVAL_MINUTES" \
  --blueprint-path "$BLUEPRINT_PATH" \
  --lut-path "$LUT_PATH" \
  --table-policy "$TABLE_POLICY" \
  --time-budget-hours "$TIME_BUDGET_HOURS" \
  --n-players "$N_PLAYERS" \
  --big-blind "$BIG_BLIND" \
  --small-blind "$SMALL_BLIND" \
  --starting-stack "$STARTING_STACK" \
  --max-iterations "$MAX_ITERATIONS" \
  --max-wall-seconds "$MAX_WALL_SECONDS" \
  "${EXTRA_ARGS[@]}" &
RUNNER_PID=$!

_forward_signal() {
  local sig=$1
  echo "[batch] received SIG${sig} — forwarding to runner (pid=${RUNNER_PID})"
  kill -"${sig}" "${RUNNER_PID}" 2>/dev/null || true
}
trap '_forward_signal TERM' TERM
trap '_forward_signal INT' INT

# `wait` returns when interrupted by a signal (after running the trap), even if the
# child is still running.  Loop until the child has actually exited so the EXIT trap
# (which tears down WORK_DIR) only fires after the runner's final VACUUM INTO.
set +e
while true; do
  wait "${RUNNER_PID}"
  EXIT_CODE=$?
  kill -0 "${RUNNER_PID}" 2>/dev/null || break
done
set -e

echo "Evaluation finished (exit ${EXIT_CODE}). Permanent snapshot: $PERM_SNAPSHOT"
exit "${EXIT_CODE}"
