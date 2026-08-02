#!/bin/bash -l
# Slurm submission script for real-time-search BUDGET CALIBRATION on the production
# node (evaluation/calibrate.py).  It plays hands with the trained blueprint, captures
# the exact production subgame roots the hero searches, and re-solves each at a ladder
# of per-replica iteration budgets *at the node's worker count* — emitting a suggested
# ``mccfr_min_per_replica_by_street`` / ``vector_budget_by_street`` block plus a
# per-cell throughput/convergence table (calibration_summary.json + calibration_rows.csv).
#
# It mirrors scripts/evaluation.sh: the LUT (runtime subset) and the blueprint are
# staged to node-local fast scratch, the compiled search core is required-and-verified
# (PLURIBUS_SEARCH_CORE=1), and the worker count is picked up from SLURM_CPUS_PER_TASK
# so the measured throughput/wall matches production.  Output is small (two files),
# written node-local and copied to a permanent dir on exit.
#
# RESOURCE FOOTPRINT is the same as evaluation.sh (real-time search loads both
# artifacts on the node): ~6 GiB LUT runtime subset + ~150 GB blueprint staged to
# $TMPDIR, and the blueprint's chunks resident in /dev/shm (counts against --mem).
#
# Usage:
#   sbatch --export=ALL,WORKSPACE=/path/to/ws,RUN_ID=cal_4p,\
#     BLUEPRINT_PATH=/path/to/blueprint scripts/calibrate_search.sh
#
#SBATCH --job-name=pluribus-calibrate
#SBATCH --output=logs/calibrate-%j.out
#SBATCH --error=logs/calibrate-%j_error.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=100000mb
#SBATCH --signal=SIGTERM@120
#SBATCH --mail-type=All

set -euo pipefail

# ----------------------------------------------------------------------------
# User-configurable
# ----------------------------------------------------------------------------
CONDA_ENV=${CONDA_ENV:-pluribus}
PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ -z "${WORKSPACE:-}" ]; then
  echo "ERROR: WORKSPACE is not set. Export WORKSPACE=/path/to/workspace before submitting." >&2
  exit 1
fi
if [ -z "${BLUEPRINT_PATH:-}" ]; then
  echo "ERROR: BLUEPRINT_PATH is not set. Export BLUEPRINT_PATH=/path/to/trained/blueprint." >&2
  exit 1
fi

RUN_ID=${RUN_ID:-"cal-${SLURM_JOB_ID:-$$}"}
LUT_PATH=${LUT_PATH:-"$WORKSPACE/exact"}

# Calibration parameters (see `python -m evaluation.calibrate run --help`).
N_PLAYERS=${N_PLAYERS:-4}
CONDITIONS=${CONDITIONS:-vanilla,DBR}      # both share the tree → equal-budget check
MODEL_P_MAX=${MODEL_P_MAX:-1.0}            # DBR arm (1.0 = naive best response)
MODEL_ERROR=${MODEL_ERROR:-0.0}
WORKERS=${WORKERS:-}                       # empty → SLURM_CPUS_PER_TASK-1 (production)
COLLECT_HANDS=${COLLECT_HANDS:-400}
PER_CELL_CAP=${PER_CELL_CAP:-3}
REPS=${REPS:-3}                            # ≥3: averaging over reps lowers the MCCFR value-estimate noise floor
LADDER_POINTS=${LADDER_POINTS:-7}          # rungs per cell, clustered around its production budget
LADDER_LO=${LADDER_LO:-0.5}                # ladder min = LO * production budget (feasible, near convergence)
LADDER_HI=${LADDER_HI:-2.0}                # ladder top (value-gap REFERENCE) = HI * production budget (>1, above convergence)
LADDER_MAX=${LADDER_MAX:-40000}            # hard cap on the top rung (bounds the single longest solve's wall)
THRESHOLDS=${THRESHOLDS:-20,10,5}          # mbb value-gap (metric = hero root EV on the table)
COLLECT_ITERS=${COLLECT_ITERS:-64}
TABLE_POLICY=${TABLE_POLICY:-random}       # random → street/live-count coverage
RUN_SEED=${RUN_SEED:-0}
BIG_BLIND=${BIG_BLIND:-100}
SMALL_BLIND=${SMALL_BLIND:-50}
STARTING_STACK=${STARTING_STACK:-10000}
LOW_CARD_RANK=${LOW_CARD_RANK:-2}
HIGH_CARD_RANK=${HIGH_CARD_RANK:-14}
WALL_TARGET=${WALL_TARGET:-30}             # per-decision wall budget (s); drives the A/B head-to-head + flags over-budget cells (set empty to disable)
REGIME_AB_STREETS=${REGIME_AB_STREETS:-turn}  # HU vector-vs-MCCFR A/B streets: turn | flop,turn | none  (flop is SLOW: full-width vector flop ~1.3 it/s)

# Permanent-FS destination for the two output files.
PERM_DIR=${PERM_DIR:-"$WORKSPACE/calibration/$RUN_ID"}
mkdir -p "$PERM_DIR"
mkdir -p "$PROJECT_DIR/logs"

echo "Activating conda environment: $CONDA_ENV"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

# ----------------------------------------------------------------------------
# Compiled search core — required + verified (must not silently fall back).
# ----------------------------------------------------------------------------
export PLURIBUS_SEARCH_CORE=${PLURIBUS_SEARCH_CORE:-1}
export PLURIBUS_CORE_KERNELS=${PLURIBUS_CORE_KERNELS:-}

# Optional: rebuild the compiled core on THIS node (opt-in, REBUILD_EXT=1).  The
# generated .c/.so are gitignored, so a node whose last build predates a newly
# committed .pyx symbol silently keeps the stale kernel and the preflight below
# fails ("kernels requested but NOT live").  A clean rebuild here — on the compute
# node, matching its arch — self-heals it.  Default off preserves the pre-built
# discipline; submit with --export=...,REBUILD_EXT=1 to force a fresh build.
REBUILD_EXT=${REBUILD_EXT:-0}
if [ "$REBUILD_EXT" = "1" ]; then
  echo "REBUILD_EXT=1 → clean rebuild of poker_ai/_core on $(hostname)"
  rm -rf build/
  rm -f poker_ai/_core/*.so poker_ai/_core/*.c   # generated + gitignored
  python setup.py build_ext --inplace
fi

if [ "$PLURIBUS_SEARCH_CORE" = "1" ] || [ -n "$PLURIBUS_CORE_KERNELS" ]; then
  if ! python - <<'PY'
import sys
import evaluation.runner  # real entry module; fixes evaluator import order
from environment.evaluator import default_evaluator
from environment.pot import Pot
import environment.poker_env as pe
import environment.range_showdown as rs
from poker_ai.search import vector as vec
from poker_ai._core.flags import kernel_enabled, search_core_enabled

live = {
    "evaluator": default_evaluator.hand_size_map[7].__module__.startswith("poker_ai._core"),
    "settlement": Pot.compute_utility.__name__ == "_compute_utility_core",
    "showdown": rs.showdown_cfv.__module__.startswith("poker_ai._core"),
    "regret_match_matrix": "poker_ai._core" in getattr(
        getattr(vec, "_regret_match_matrix", None), "__module__", ""),
    "settle_concrete": pe._settle_traverser is not pe._settle_traverser_py,
}
dead = [k for k, ok in live.items() if kernel_enabled(k) and not ok]
if dead:
    sys.exit("kernels requested but NOT live (silent pure-Python fallback): %s"
             % ", ".join(dead))

walk = "off"
if search_core_enabled():
    import poker_ai.search.mccfr as mccfr
    from poker_ai.search.fast_env import (  # noqa: F401
        build_fast_walk_env, build_fast_mccfr_env,
    )
    if mccfr.continuation_value_vector.__module__ != "poker_ai.search.leaf_fast":
        sys.exit("PLURIBUS_SEARCH_CORE=1 but the MCCFR leaf rollout is pure Python")
    if mccfr.build_fast_mccfr_env is not build_fast_mccfr_env:
        sys.exit("PLURIBUS_SEARCH_CORE=1 but the MCCFR walk adapter is not wired")
    walk = "on"

print("Search kernels live: %s  |  walk core: %s"
      % (", ".join(k for k, ok in live.items() if ok) or "(none)", walk))
PY
  then
    echo "ERROR: a requested compiled path silently fell back to pure Python." >&2
    echo "       Rebuild on this node: python setup.py build_ext --inplace" >&2
    echo "       (or set PLURIBUS_SEARCH_CORE=0 for the pure-Python path)." >&2
    exit 1
  fi
fi

# ----------------------------------------------------------------------------
# Path checks
# ----------------------------------------------------------------------------
if [ ! -d "$LUT_PATH" ] || [ ! -f "$LUT_PATH/card_info_lut.joblib" ]; then
  echo "ERROR: LUT not found / incomplete at $LUT_PATH (needs card_info_lut.joblib)." >&2
  exit 1
fi
if [ ! -d "$BLUEPRINT_PATH" ]; then
  echo "ERROR: BLUEPRINT_PATH not found at $BLUEPRINT_PATH" >&2
  exit 1
fi

# ----------------------------------------------------------------------------
# Node-local scratch + staging (same discipline as evaluation.sh)
# ----------------------------------------------------------------------------
WORK_DIR="${TMPDIR:?cluster requires TMPDIR to be set}/pluribus-cal-${SLURM_JOB_ID:-$$}"
LOCAL_OUT="$WORK_DIR/out"
mkdir -p "$LOCAL_OUT"
# Copy whatever results exist back to the permanent dir on ANY exit (partial runs
# still yield the rows/summary written so far), then tear down node-local scratch.
trap 'cp -a "$LOCAL_OUT"/. "$PERM_DIR"/ 2>/dev/null || true; rm -rf "$WORK_DIR"' EXIT

STAGE_LUT_LOCALLY=${STAGE_LUT_LOCALLY:-true}
STAGE_BLUEPRINT_LOCALLY=${STAGE_BLUEPRINT_LOCALLY:-true}

lut_runtime_kb() {
  du -sck "$1"/card_info_lut.joblib "$1"/*/cluster_ids.dat 2>/dev/null \
    | awk '$2 == "total" {print $1}'
}

need_kb=0
if [ "$STAGE_LUT_LOCALLY" = "true" ]; then
  lut_kb=$(lut_runtime_kb "$LUT_PATH")
  [ -n "$lut_kb" ] || { echo "ERROR: could not size the LUT runtime subset." >&2; exit 1; }
  need_kb=$((need_kb + lut_kb))
fi
if [ "$STAGE_BLUEPRINT_LOCALLY" = "true" ]; then
  need_kb=$((need_kb + $(du -sk "$BLUEPRINT_PATH" | cut -f1)))
fi
need_kb=$((need_kb + 5 * 1024 * 1024))   # +5 GB margin
avail_kb=$(df -Pk "$WORK_DIR" | awk 'NR==2 {print $4}')
echo "Node-local scratch: need ~$((need_kb/1024/1024)) GB, available ~$((avail_kb/1024/1024)) GB at $WORK_DIR"
if [ "$avail_kb" -lt "$need_kb" ]; then
  echo "ERROR: insufficient node-local scratch to stage LUT + blueprint." >&2
  echo "       Disable a stage (STAGE_LUT_LOCALLY=false / STAGE_BLUEPRINT_LOCALLY=false)." >&2
  exit 1
fi

LUT_RUNTIME_FILTER=(
  --include='*/'
  --include='card_info_lut.joblib'
  --include='cluster_ids.dat'
  --exclude='*'
)
if [ "$STAGE_LUT_LOCALLY" = "true" ]; then
  LOCAL_LUT_PATH="$WORK_DIR/lut"
  echo "Staging LUT (runtime subset) → $LOCAL_LUT_PATH ..."
  mkdir -p "$LOCAL_LUT_PATH"
  rsync -a "${LUT_RUNTIME_FILTER[@]}" "$LUT_PATH/" "$LOCAL_LUT_PATH/"
  [ -f "$LOCAL_LUT_PATH/card_info_lut.joblib" ] || {
    echo "ERROR: staged LUT is incomplete (missing card_info_lut.joblib)." >&2; exit 1; }
  for src_ids in "$LUT_PATH"/*/cluster_ids.dat; do
    [ -e "$src_ids" ] || continue
    rel="${src_ids#$LUT_PATH/}"
    [ -f "$LOCAL_LUT_PATH/$rel" ] || { echo "ERROR: staged LUT missing $rel." >&2; exit 1; }
  done
  LUT_PATH="$LOCAL_LUT_PATH"
fi
if [ "$STAGE_BLUEPRINT_LOCALLY" = "true" ]; then
  LOCAL_BLUEPRINT_PATH="$WORK_DIR/blueprint"
  echo "Staging blueprint → $LOCAL_BLUEPRINT_PATH ..."
  mkdir -p "$LOCAL_BLUEPRINT_PATH"
  rsync -a "$BLUEPRINT_PATH/" "$LOCAL_BLUEPRINT_PATH/"
  BLUEPRINT_PATH="$LOCAL_BLUEPRINT_PATH"
fi

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
echo "Starting calibration with:"
echo "  - Run id:            $RUN_ID"
echo "  - Players:           $N_PLAYERS"
echo "  - Conditions:        $CONDITIONS"
echo "  - Workers:           ${WORKERS:-(auto = SLURM_CPUS_PER_TASK-1)}"
echo "  - CPUs:              ${SLURM_CPUS_PER_TASK:-(unset)}"
echo "  - Collect hands:     $COLLECT_HANDS  (per-cell cap $PER_CELL_CAP, reps $REPS)"
echo "  - Ladder:            per-cell [${LADDER_LO}..${LADDER_HI}]x production budget, $LADDER_POINTS pts, cap $LADDER_MAX"
echo "  - Thresholds (mbb):  $THRESHOLDS"
echo "  - Wall target (s):   ${WALL_TARGET:-(disabled)}"
echo "  - Regime A/B:        $REGIME_AB_STREETS"
echo "  - Table policy:      $TABLE_POLICY"
echo "  - Search core:       ${PLURIBUS_SEARCH_CORE:-0}"
echo "  - LUT / blueprint:   $LUT_PATH  |  $BLUEPRINT_PATH"
echo "  - Output → perm:     $PERM_DIR"

EXTRA_ARGS=()
[ -n "$WORKERS" ]     && EXTRA_ARGS+=(--workers "$WORKERS")
[ -n "$WALL_TARGET" ] && EXTRA_ARGS+=(--wall-target "$WALL_TARGET")

# Background + forward SLURM's grace-period SIGTERM (same pattern as evaluation.sh),
# so a wall-clock stop still hits the EXIT trap and copies partial output back.
python -m evaluation.calibrate run \
  --blueprint-path "$BLUEPRINT_PATH" \
  --lut-path "$LUT_PATH" \
  --n-players "$N_PLAYERS" \
  --conditions "$CONDITIONS" \
  --model-p-max "$MODEL_P_MAX" \
  --model-error "$MODEL_ERROR" \
  --collect-hands "$COLLECT_HANDS" \
  --per-cell-cap "$PER_CELL_CAP" \
  --reps "$REPS" \
  --ladder-points "$LADDER_POINTS" \
  --ladder-lo "$LADDER_LO" \
  --ladder-hi "$LADDER_HI" \
  --ladder-max "$LADDER_MAX" \
  --thresholds "$THRESHOLDS" \
  --collect-iters "$COLLECT_ITERS" \
  --table-policy "$TABLE_POLICY" \
  --run-seed "$RUN_SEED" \
  --big-blind "$BIG_BLIND" \
  --small-blind "$SMALL_BLIND" \
  --starting-stack "$STARTING_STACK" \
  --low-card-rank "$LOW_CARD_RANK" \
  --high-card-rank "$HIGH_CARD_RANK" \
  --regime-ab-streets "$REGIME_AB_STREETS" \
  --out-dir "$LOCAL_OUT" \
  "${EXTRA_ARGS[@]}" &

PY_PID=$!
term() { echo "Forwarding SIGTERM to calibration (pid $PY_PID)"; kill -TERM "$PY_PID" 2>/dev/null || true; }
trap term TERM
wait "$PY_PID"
echo "Calibration finished; results in $PERM_DIR"
