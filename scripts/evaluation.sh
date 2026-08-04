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
#   * Node-local disk: only the LUT's *runtime subset* (~6 GiB — the cluster-id
#     memmaps; the ~250 GB of hand-strength build artefacts are never staged, see
#     LUT_RUNTIME_FILTER below) and the blueprint (~150 GB) are rsync'd to $TMPDIR,
#     so the job needs ~170 GB of local scratch (the two staged copies + the small
#     SQLite db).  We deliberately do NOT `#SBATCH --tmp=...` for
#     it — like training.sh, staging goes to whatever $TMPDIR the node provides, and
#     the runtime preflight `df` check below aborts early if the node cannot hold
#     both.  (A hard `--tmp` reservation is rejected at submit time — "Temporary disk
#     specification can not be satisfied" — on clusters that don't advertise that
#     much schedulable TmpDisk; the preflight gives the same protection without it.)
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
#   Paired multi-arm sanity test (search vs pure blueprint), one submission:
#   sbatch --export=ALL,WORKSPACE=...,RUN_ID=sanity,BLUEPRINT_PATH=...,\
#     CONDITIONS=vanilla,blueprint_only,MAX_HANDS=3000,N_PLAYERS=2 scripts/evaluation.sh
#   → both arms share one db + run-seed (CRN-paired on deck_seed), each under its own
#     run-id; a paired summary is printed at the end.
#
#SBATCH --job-name=pluribus-eval
#SBATCH --output=logs/evaluation-%j.out
#SBATCH --error=logs/evaluation-%j_error.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=200000mb
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

# Experiment arms (docs/evaluation.md §10.1).  A comma-separated list runs each arm
# in turn into the SAME node-local db, so the summary pairs them on deck_seed (CRN).
# Each arm gets its own per-arm run-id — the resume cursor is keyed per run_id, so a
# shared id would make later arms skip every hand.  A single value = the classic
# one-arm run.  Model-free arms: vanilla | blueprint_only | OX (or OX(beta=X)).  A
# DBR arm (e.g. 'DBR(p_max=0.8)') additionally consumes the MODEL_* knobs below.
#   e.g. CONDITIONS=vanilla,blueprint_only  — the search-vs-pure-blueprint sanity test.
CONDITIONS=${CONDITIONS:-vanilla}
# Paired fixed hand count (§10.1).  When set it is the SOLE stop criterion (every arm
# covers hand_index 0..MAX_HANDS-1 → CRN-paired) and TIME_BUDGET_HOURS is ignored.
# Leave empty for the time-budgeted sweep.  A vanilla/DBR/OX comparison REQUIRES it.
MAX_HANDS=${MAX_HANDS:-}
# DBR model knobs — consumed ONLY by a DBR(...) arm; model-free arms never receive
# them (a DBR arm with MODEL_P_MAX unset aborts rather than silently running vanilla).
MODEL_P_MAX=${MODEL_P_MAX:-}
MODEL_ERROR=${MODEL_ERROR:-0.0}
MODEL_CONFIDENCE=${MODEL_CONFIDENCE:-1.0}
MODEL_SEED=${MODEL_SEED:-0}
# Auto-run the paired CRN summary over the permanent snapshot once all arms finish.
SUMMARIZE=${SUMMARIZE:-true}
N_PLAYERS=${N_PLAYERS:-6}
BIG_BLIND=${BIG_BLIND:-100}
SMALL_BLIND=${SMALL_BLIND:-50}
STARTING_STACK=${STARTING_STACK:-10000}
# The single absolute per-replica iteration CEILING — NOT a throttle: the
# structural budget (poker_ai/search/budget.py) is the primary stop and this only
# clips it in pathology, so it must sit at/above the deepest structural budget any
# subgame requests (multiway flop clamps at 30000).  The old 5000 here silently
# capped every HU-flop / multiway MCCFR search far below convergence.
MAX_ITERATIONS=${MAX_ITERATIONS:-30000}
# Loose per-search wall backstop (seconds).  Sized above the DEEPEST shipped search's
# wall cost so it never clips the iteration budget in normal operation and only
# catches a genuinely stuck subgame.  From the 2026-08 (v4) calibration: the deepest
# is multiway flop MCCFR at the 30000 clamp — 30000 / 38.19 it_s ≈ 786 s under
# 63-worker load — and the calibration's own flop backstop is 993 s; 1000 clears both
# with margin at every N_PLAYERS.  (A per-street tuple is NOT used here: the
# calibration measured only HU vector turn/river, whose tiny river cap 16 s would
# butcher multiway river MCCFR at ~130 s.)  The old 10.0 here cut every search but river.
MAX_WALL_SECONDS=${MAX_WALL_SECONDS:-1000.0}
# AIVAT variance-reduced strength estimate (§10.2).  ON by default — it fills
# games.aivat_value (the summary auto-switches its strength CI onto it) at extra
# per-hand cost that stays in the experiment budget, off the search hot path, and
# never perturbs the played hand.  Set AIVAT=false for the raw-only baseline.
AIVAT=${AIVAT:-true}                                # true | false
AIVAT_HOLE_SAMPLES=${AIVAT_HOLE_SAMPLES:-6}         # belief draws per value eval
# Sync-back cadence (§5): every SYNC_INTERVAL_HANDS hands and/or SYNC_INTERVAL_MINUTES.
SYNC_INTERVAL_HANDS=${SYNC_INTERVAL_HANDS:-500}
SYNC_INTERVAL_MINUTES=${SYNC_INTERVAL_MINUTES:-20}
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

# Compiled real-time SEARCH core — the SINGLE operator switch.  1 = drive the
# whole subgame solve through the compiled core: the walk on the FastState betting
# engine (make/undo collapses out of Python) AND every byte-identical leaf/terminal
# kernel it uses (regret matching, range-showdown CFV, runout, evaluator,
# settlement).  Byte-identical to the pure-Python path (parity-gated by
# test/search/functional/test_search_core_golden.py et al.) and fallback-retaining:
# it reverts to the PokerEnv path when the extension is unavailable or an off-tree
# action was injected.  Set 0 to force the pure-Python baseline arm.
export PLURIBUS_SEARCH_CORE=${PLURIBUS_SEARCH_CORE:-1}

# Developer override — normally UNSET.  A comma-separated kernel allow-list (or
# ``all``) that A/B's individual kernels against their pure-Python oracles in
# isolation; when set it WINS over PLURIBUS_SEARCH_CORE for the per-kernel gates
# (the walk still follows PLURIBUS_SEARCH_CORE).  Leave unset in production — the
# master switch above already lights every kernel.
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

# Preflight: with the search core on, the compiled paths must be ACTUALLY LIVE,
# not merely requested.  Each kernel rebinds itself at import behind
# ``except ImportError: pass``, so a missing/stale .so — or the evaluator kernel's
# import-order trap — leaves the pure-Python reference in place with no error and
# search just runs slower.  Import through ``evaluation.runner`` (the real -m entry
# module, which forces poker_ai first) and assert the rebinds happened for the
# terminal-value layer and the vector/showdown leaf ops, and that the search-walk
# core is enabled + its FastState adapter importable.
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
    # The concrete per-combo settlement rebind keeps the same name, so detect it
    # by identity: the core closure is a *different* object from the _py oracle.
    "settle_concrete": pe._settle_traverser is not pe._settle_traverser_py,
}
dead = [k for k, ok in live.items() if kernel_enabled(k) and not ok]
if dead:
    sys.exit("kernels requested but NOT live (silent pure-Python fallback): %s"
             % ", ".join(dead))

# Search WALK core (gated by the single PLURIBUS_SEARCH_CORE switch): both regimes'
# FastState adapters (vector `build_fast_walk_env` + MCCFR `build_fast_mccfr_env`)
# and the MCCFR depth-limit leaf rollout, which rebinds to the compiled FastState
# rollout at import when the flag is on.  If any did not, the walk/leaf silently ran
# on pure Python.
walk = "off"
if search_core_enabled():
    import poker_ai.search.mccfr as mccfr
    from poker_ai.search.fast_env import (  # noqa: F401
        build_fast_walk_env, build_fast_mccfr_env,
    )
    if mccfr.continuation_value_vector.__module__ != "poker_ai.search.leaf_fast":
        sys.exit("PLURIBUS_SEARCH_CORE=1 but the MCCFR leaf rollout did not bind to "
                 "the compiled core (mccfr.continuation_value_vector is pure Python)")
    if mccfr.build_fast_mccfr_env is not build_fast_mccfr_env:
        sys.exit("PLURIBUS_SEARCH_CORE=1 but the MCCFR walk adapter is not wired "
                 "(mccfr.build_fast_mccfr_env missing — the walk runs on PokerEnv)")
    walk = "on"

print("Search kernels live: %s  |  walk core: %s"
      % (", ".join(k for k, ok in live.items() if ok) or "(none)", walk))
PY
  then
    echo "ERROR: a requested compiled path silently fell back to pure Python" >&2
    echo "       (extension missing/stale, or an import-order regression in" >&2
    echo "       evaluation/runner.py).  Rebuild on this node:" >&2
    echo "       python setup.py build_ext --inplace" >&2
    echo "       (or set PLURIBUS_SEARCH_CORE=0 for the pure-Python path)." >&2
    exit 1
  fi
fi

# Ensure LUT path exists
if [ ! -d "$LUT_PATH" ]; then
  echo "ERROR: LUT path not found at $LUT_PATH" >&2
  echo "Please run abstraction first using: sbatch scripts/abstraction_auto_resub.sh" >&2
  exit 1
fi
if [ ! -f "$LUT_PATH/card_info_lut.joblib" ]; then
  echo "ERROR: $LUT_PATH has no card_info_lut.joblib — the abstraction build did" >&2
  echo "       not finish (or this is the legacy pickle-dir layout, unsupported here)." >&2
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
# training.sh does) would leak a partial 150 GB+ copy on node-local scratch if a
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

# Only the runtime mapping is staged, never the build artefacts — see the
# LUT_RUNTIME_FILTER rationale below.  Sizing must use the same subset the rsync
# will actually copy, or the preflight demands ~250 GB of scratch for a ~6 GiB
# stage and rejects nodes that would have been fine.
lut_runtime_kb() {
  du -sck "$1"/card_info_lut.joblib "$1"/*/cluster_ids.dat 2>/dev/null \
    | awk '$2 == "total" {print $1}'
}

# Preflight: the staged LUT subset (~6 GiB) and blueprint (~150 GB) are large, so
# verify the node-local filesystem can hold everything we intend to stage BEFORE
# rsync starts — a half-staged copy that fills the disk mid-run is far worse than
# failing fast here.  Needed = sum of the sources we will stage + a margin.
need_kb=0
if [ "$STAGE_LUT_LOCALLY" = "true" ]; then
  lut_kb=$(lut_runtime_kb "$LUT_PATH")
  if [ -z "$lut_kb" ]; then
    echo "ERROR: could not size the LUT runtime subset under $LUT_PATH." >&2
    exit 1
  fi
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
#
# Stage the RUNTIME SUBSET ONLY.  The abstraction directory mixes two disjoint
# classes of data, and only one of them is read after the build:
#   * runtime mapping  — card_info_lut.joblib (pre-flop dict + MemmapLookup
#     stubs) and each street's cluster_ids.dat (the uint16 combo→cluster
#     memmap).  These are the ONLY files information_abstraction/lookup.py ever
#     opens.  ~5.9 GiB on a 52-card deck (river 5.2 + turn 0.6 + flop 0.05).
#   * build artefacts — merged_data.dat (the per-combo hand-strength/EHS feature
#     vectors), all_combos.npy, clusters.npy, centroids.*, checkpoint.json.
#     These are the k-means INPUTS, touched only by information_abstraction/
#     build/*.  They dominate the ~250 GB on disk and are dead weight here.
# So the filter below cuts the staged bytes ~40x with no behaviour change.
LUT_RUNTIME_FILTER=(
  --include='*/'
  --include='card_info_lut.joblib'
  --include='cluster_ids.dat'
  --exclude='*'
)
if [ "$STAGE_LUT_LOCALLY" = "true" ]; then
  SRC_LUT_PATH="$LUT_PATH"
  LOCAL_LUT_PATH="$WORK_DIR/lut"
  echo "Staging LUT (runtime subset) from $SRC_LUT_PATH to $LOCAL_LUT_PATH ..."
  mkdir -p "$LOCAL_LUT_PATH"
  rsync_start=$(date +%s)
  rsync -a "${LUT_RUNTIME_FILTER[@]}" "$SRC_LUT_PATH/" "$LOCAL_LUT_PATH/"
  rsync_end=$(date +%s)
  echo "LUT staged in $((rsync_end - rsync_start))s ($(du -sh "$LOCAL_LUT_PATH" | cut -f1) of $(du -sh "$SRC_LUT_PATH" | cut -f1) total)"

  # Verify the subset is COMPLETE.  load_info_set_lut() rebinds each street's
  # MemmapLookup to $LUT_PATH/<street>/cluster_ids.dat only *if that file
  # exists*; otherwise it silently keeps the absolute path baked in at build
  # time and every lookup goes back to the shared FS — precisely the cost this
  # staging exists to avoid, with no error and no log line.  A filter that
  # misses a file must fail loudly here, not degrade silently at runtime.
  lut_missing=()
  [ -f "$LOCAL_LUT_PATH/card_info_lut.joblib" ] || lut_missing+=("card_info_lut.joblib")
  for src_ids in "$SRC_LUT_PATH"/*/cluster_ids.dat; do
    [ -e "$src_ids" ] || continue          # unmatched glob
    rel="${src_ids#$SRC_LUT_PATH/}"
    [ -f "$LOCAL_LUT_PATH/$rel" ] || lut_missing+=("$rel")
  done
  if [ ${#lut_missing[@]} -ne 0 ]; then
    echo "ERROR: staged LUT is incomplete — missing: ${lut_missing[*]}" >&2
    echo "       LUT_RUNTIME_FILTER dropped a file the runtime needs." >&2
    exit 1
  fi
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
echo "  - Run id (base):          $RUN_ID"
echo "  - Conditions:             $CONDITIONS"
echo "  - Max hands (paired):     ${MAX_HANDS:-(time-budgeted)}"
echo "  - Run seed:               $RUN_SEED"
echo "  - Table policy:           $TABLE_POLICY"
echo "  - Time budget (hours):    $TIME_BUDGET_HOURS"
echo "  - Players:                $N_PLAYERS"
echo "  - Big/small blind:        $BIG_BLIND / $SMALL_BLIND"
echo "  - Starting stack:         $STARTING_STACK"
echo "  - Max iterations:         $MAX_ITERATIONS"
echo "  - Max wall seconds:       $MAX_WALL_SECONDS"
echo "  - AIVAT:                  $AIVAT (hole samples: $AIVAT_HOLE_SAMPLES)"
echo "  - Sync interval (hands):  $SYNC_INTERVAL_HANDS"
echo "  - Sync interval (mins):   $SYNC_INTERVAL_MINUTES"
echo "  - Search core:            ${PLURIBUS_SEARCH_CORE:-0}  (1 = walk + all kernels)"
echo "  - Kernel override (dev):  ${PLURIBUS_CORE_KERNELS:-(none — master switch drives all)}"
echo "  - LUT path:               $LUT_PATH"
echo "  - Blueprint path:         $BLUEPRINT_PATH"
echo "  - Stage LUT locally:      $STAGE_LUT_LOCALLY"
echo "  - Stage blueprint local:  $STAGE_BLUEPRINT_LOCALLY"
echo "  - Node-local db:          $LOCAL_DB_PATH"
echo "  - Permanent snapshot:     $PERM_SNAPSHOT"
echo "  - CPUs:                   ${SLURM_CPUS_PER_TASK:-(unset)}"

# Build optional flags shared by every arm.
EXTRA_ARGS=()
[ -n "$FIXED_SEATS" ]  && EXTRA_ARGS+=(--fixed-seats "$FIXED_SEATS")
# AIVAT (§10.2): a boolean --aivat/--no-aivat flag + the belief-sample count.
if [ "$AIVAT" = "true" ]; then
  EXTRA_ARGS+=(--aivat --aivat-hole-samples "$AIVAT_HOLE_SAMPLES")
else
  EXTRA_ARGS+=(--no-aivat)
fi

# Stop criterion, shared by every arm: a fixed paired hand count (MAX_HANDS) makes
# every arm cover hand_index 0..MAX_HANDS-1 (CRN-paired, time budget ignored by the
# runner); otherwise the classic time budget.
STOP_ARGS=()
if [ -n "$MAX_HANDS" ]; then
  STOP_ARGS+=(--max-hands "$MAX_HANDS")
  echo "Paired mode: $MAX_HANDS hands per arm (TIME_BUDGET_HOURS ignored)."
else
  STOP_ARGS+=(--time-budget-hours "$TIME_BUDGET_HOURS")
fi

# Signal forwarding: slurm's grace-period SIGTERM goes to this bash wrapper, not its
# child (same pattern as training.sh).  Forward it to whichever arm is running and
# stop the loop so no further arms start — the running arm writes its final VACUUM
# INTO before the EXIT trap tears down node-local scratch.
CURRENT_PID=""
STOP_REQUESTED=0
_forward_signal() {
  local sig=$1
  STOP_REQUESTED=1
  if [ -n "$CURRENT_PID" ]; then
    echo "[batch] received SIG${sig} — forwarding to runner (pid=${CURRENT_PID})"
    kill -"${sig}" "${CURRENT_PID}" 2>/dev/null || true
  fi
}
trap '_forward_signal TERM' TERM
trap '_forward_signal INT' INT

# Run one arm in the background and block (via a signal-resilient wait loop) until it
# exits.  All arms write the SAME node-local db and sync to the SAME permanent
# snapshot; only --run-id and --condition (and the DBR model knobs) differ.
run_arm() {
  local cond=$1 arm_run_id=$2
  shift 2                                   # remaining args = per-arm model flags
  python -m evaluation.runner run \
    --run-id "$arm_run_id" \
    --condition "$cond" \
    --run-seed "$RUN_SEED" \
    --db-path "$LOCAL_DB_PATH" \
    --sync-path "$PERM_SNAPSHOT" \
    --sync-interval-hands "$SYNC_INTERVAL_HANDS" \
    --sync-interval-minutes "$SYNC_INTERVAL_MINUTES" \
    --blueprint-path "$BLUEPRINT_PATH" \
    --lut-path "$LUT_PATH" \
    --table-policy "$TABLE_POLICY" \
    --n-players "$N_PLAYERS" \
    --big-blind "$BIG_BLIND" \
    --small-blind "$SMALL_BLIND" \
    --starting-stack "$STARTING_STACK" \
    --max-iterations "$MAX_ITERATIONS" \
    --max-wall-seconds "$MAX_WALL_SECONDS" \
    "${STOP_ARGS[@]}" "${EXTRA_ARGS[@]}" "$@" &
  CURRENT_PID=$!
  local code
  set +e
  # `wait` returns when interrupted by a signal (after running the trap), even if the
  # child is still running.  Loop until the child has actually exited.
  while true; do
    wait "${CURRENT_PID}"
    code=$?
    kill -0 "${CURRENT_PID}" 2>/dev/null || break
  done
  set -e
  CURRENT_PID=""
  return "$code"
}

# One arm per condition (comma-separated CONDITIONS), into the shared db.
IFS=',' read -ra COND_ARR <<< "$CONDITIONS"
MULTI=0
[ "${#COND_ARR[@]}" -gt 1 ] && MULTI=1
EXIT_CODE=0
for raw_cond in "${COND_ARR[@]}"; do
  cond="$(echo "$raw_cond" | xargs)"        # trim surrounding whitespace
  [ -z "$cond" ] && continue

  # Per-arm run-id: base id + a filesystem-safe slug of the condition, so arms in one
  # db never share a run_id (which would make later arms resume-skip every hand).  A
  # single-condition run keeps the plain base id (backward compatible).
  arm_run_id="$RUN_ID"
  if [ "$MULTI" -eq 1 ]; then
    arm_slug="$(echo "$cond" | tr -c 'A-Za-z0-9' '_' | sed -E 's/_+/_/g; s/^_//; s/_$//')"
    arm_run_id="${RUN_ID}__${arm_slug}"
  fi

  # A DBR arm consumes the model knobs; model-free arms must NOT receive them.
  ARM_ARGS=()
  case "$(echo "$cond" | tr '[:upper:]' '[:lower:]')" in
    dbr*)
      if [ -z "$MODEL_P_MAX" ]; then
        echo "ERROR: condition '$cond' is a DBR arm but MODEL_P_MAX is unset." >&2
        exit 1
      fi
      ARM_ARGS+=(--model-p-max "$MODEL_P_MAX" --model-error "$MODEL_ERROR" \
                 --model-confidence "$MODEL_CONFIDENCE" --model-seed "$MODEL_SEED")
      ;;
  esac

  echo "=== Arm: condition='$cond'  run-id='$arm_run_id' ==="
  run_arm "$cond" "$arm_run_id" "${ARM_ARGS[@]}"
  code=$?
  [ "$code" -ne 0 ] && EXIT_CODE=$code
  if [ "$STOP_REQUESTED" -eq 1 ]; then
    echo "[batch] stop requested — not starting further arms."
    break
  fi
done

echo "Evaluation finished (exit ${EXIT_CODE}). Permanent snapshot: $PERM_SNAPSHOT"

# Paired CRN summary over the permanent snapshot (search − blueprint advantage, etc.).
# Best-effort: a summary failure never overrides the run's own exit code.
if [ "$SUMMARIZE" = "true" ] && [ -f "$PERM_SNAPSHOT" ]; then
  echo "=== Summary ($PERM_SNAPSHOT) ==="
  python -m evaluation.summarize "$PERM_SNAPSHOT" || \
    echo "[batch] summary step failed (non-fatal)."
fi

exit "${EXIT_CODE}"
