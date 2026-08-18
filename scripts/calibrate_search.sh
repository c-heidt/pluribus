#!/bin/bash -l
# Slurm submission script for real-time-search BUDGET CALIBRATION on the production
# node (evaluation/calibrate.py).  It plays hands with the trained blueprint, captures
# the exact production subgame roots the hero searches, and re-solves each at a ladder
# of per-replica iteration budgets — emitting a suggested
# per-cell ``MCCFR_BUDGET`` / ``VECTOR_BUDGET`` block (one number per
# (approach, street, n_live)) plus a per-cell throughput/convergence table
# (calibration_summary.json + calibration_rows.csv).
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
#SBATCH --time=5:00:00   # sized for N_LIVE=2,3 (~54 core-h → ~1h wall at MAX_CONCURRENT_MULTIWAY=32). The n_live=4 slice is the expensive one (~116 core-h → ~3.5h; incl. the turn-n4 14.5 it/s outlier): when running N_LIVE=4, raise this to 5:00:00. Full grid would be ~4.5h.
#SBATCH --cpus-per-task=64
# --mem: sized for MAX_CONCURRENT_MULTIWAY=32 (the ≤5h target).  ~32 * ~10 GB/multiway-solve
# + lights + blueprint ≈ 340 GB (ESTIMATE — VERIFY real RSS on run 1 and tighten; if it comes
# in well under, lower this for faster scheduling).  ≤5h and low RAM are in tension here: to
# drop --mem you must lower K, which pushes the wall past 5h (K=16 ≈ 200 GB but ~9.5h).
#SBATCH --mem=250000mb
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
CONDITIONS=${CONDITIONS:-vanilla,DBR,OX}   # any of: vanilla | DBR | blueprint_only | OX | OX(beta=X).
                                           # OX arms are auto-split into their OWN calibrate run (gadget = a
                                           # different game tree, β must agree → cannot share a solver_cfg with
                                           # vanilla/DBR) and calibrate VECTOR cells only (OX's MCCFR path == vanilla).
                                           # Bare "OX" uses the code-default β (runner.DEFAULT_OX_BETA).
# DBR model realism (opponent_modeling.md §3/§6.2).  A calibration on a PERFECT model
# (error 0.0, p_max 1.0) is condition B1 — the exact-model, unconstrained "unsafe EV
# ceiling", the sharpest/most-polarised exploitation and NOT what production runs.  We
# calibrate the production DBR (Approach A) instead: an imperfect model at the default cap.
MODEL_ERROR=${MODEL_ERROR:-0.2}           # target ℓ1 error of σ̂ vs the true strategy (0=exact; 0.2≈a 10pp
                                          #   per-infoset action-frequency miss — a good-but-imperfect model;
                                          #   0.3 is the more conservative choice).  Uniform proxy for the
                                          #   street-graded realistic profile (low preflop→high river); the
                                          #   schedule form (schedules.street_error) is not yet on the CLI.
MODEL_P_MAX=${MODEL_P_MAX:-0.8}           # confidence cap (Approach A default).  1.0 = B1 naive-BR ceiling;
                                          #   0.8 blends σ̃=c·σ̂+(1−c)·x, de-polarising the exploitation — the
                                          #   biggest lever on the tail-driven variance we've been chasing.
WORKERS=${WORKERS:-}                       # empty → SLURM_CPUS_PER_TASK-1 (production)
MAX_CONCURRENT_MULTIWAY=${MAX_CONCURRENT_MULTIWAY:-16}  # peak-RAM cap: at most this many multiway (≥3 live) MCCFR solves run at once (biggest tables); cheap solves backfill the rest. Wall is heavy-bound ≈ (multiway core-hours ~152)/K → K=32 gives ~4.7h (≤5h). Uncapped (~52 concurrent) would front-load the RAM and likely exceed the node. Empty → no cap. Peak RAM ≈ K*multiway + (WORKERS-K)*light; size --mem to that. Lower K = less RAM but a longer wall (K=16→~9.5h).
COLLECT_HANDS=${COLLECT_HANDS:-400}
N_LIVE=${N_LIVE:-2,3}                      # live-player counts to calibrate; empty = all. '2,3' EXCLUDES the deep 4-player lines (~2/3 of the cost incl. the turn-n4 outlier) — run them later with N_LIVE=4. Lossless split: roots are seeded per (street, n_live, k). The grid is POST-FLOP only (preflop is played from the blueprint, never searched).
PER_CELL_CAP=${PER_CELL_CAP:-6}            # distinct HANDS (roots) per cell for the sampled cells
PER_CELL_CAP_DETERMINISTIC=${PER_CELL_CAP_DETERMINISTIC:-8}  # distinct HANDS for the DETERMINISTIC cells (vector river): every seed gives a byte-identical solve there, so they run exactly ONE rep and spend the compute on extra hands instead
REPS=${REPS:-4}                            # re-solves per root, SAMPLED cells only (different seed, same hand). Deterministic cells (vector river) always use 1 — extra reps there are byte-identical. Snapshot ladders make one rep cover every rung, so a cell = PER_CELL_CAP hands x REPS searches.
HOT_L1_TOL_MCCFR=${HOT_L1_TOL_MCCFR:-0.20}  # convergence tolerance for the SAMPLED cells. LOOSER than vector on purpose: external-sampling MCCFR leaves Monte-Carlo dispersion that decays like 1/sqrt(T) and never reaches 0, so a vector-tight bar would measure noise, not convergence (and would demand budget bought purely to average out noise the next street's re-solve discards). Each cell also reports its MEASURED cross-seed spread ('xrep' = cross-rep hot_l1 at the top budget: same hand, same iterations, different seed); the run flags any cell whose tol sits at or under it, meaning the budget is being resolved finer than two independent seeds of that cell agree to — loosen this above the reported xrep if so.
HOT_L1_TOL_VECTOR=${HOT_L1_TOL_VECTOR:-0.10}  # convergence tolerance for the FULL-WIDTH cells. Tighter: the vector regime enumerates both ranges, so there is no per-iteration sampling noise to average out (the river cell is exactly deterministic).
# Both: per-cell budget = smallest rung where MEAN hot_l1 ≤ tol. hot_l1 = single-worker strategy self-distance vs its OWN top budget, measured over the hero's WHOLE ROOT STREET (reach-weighted over every hero decision node on that street, since one solve serves the whole street — not the root alone, and not past the street boundary, which is re-solved), averaged over the cell's solves. Cells that never settle are flagged → raise that street LADDER_TOP_SECONDS.
LADDER_POINTS=${LADDER_POINTS:-6}          # rungs per ladder, walking DOWN from the wall-anchored top
LADDER_TOP_SECONDS=${LADDER_TOP_SECONDS:-600,600,60}          # MCCFR per-street (flop,turn,river) wall budget for the ladder TOP rung: top = probed it/s × this. Hard cells (flop/turn: hot_l1 0.23-0.43 at their production budgets) get the full 600s; river converged at ~9000 iters/22s so 60s brackets it without waste.
LADDER_TOP_SECONDS_VECTOR=${LADDER_TOP_SECONDS_VECTOR:-600,600,15}  # same for VECTOR cells — separate because the regimes converge at very different iteration counts (HU river vector settles ~1071 iters vs ~9000 multiway river MCCFR), so one per-street value cannot bracket both.
LADDER_LO=${LADDER_LO:-0.35}               # ladder min as a FRACTION OF THE TOP (not of the production budget) — 0.35 spans the band where hot_l1 crosses 0.1-0.2; lower rungs are known-unconverged and skipped
PROBE_SECONDS=${PROBE_SECONDS:-30}         # box-saturating probe solve per ladder that measures its it/s (sets the top rung). Loaded, so throughput matches the real run instead of an idle-box overestimate.
COLLECT_ITERS=${COLLECT_ITERS:-64}
TABLE_POLICY=${TABLE_POLICY:-fixed}        # 'fixed' → deterministic table from FIXED_SEATS (no random seat-draw noise across cells); 'random' | 'all_blueprint' also allowed
FIXED_SEATS=${FIXED_SEATS:-bp_fold,bp_call,bp_raise}  # ONE opponent per exploitable bias class (fold/call/raise) — exactly N_PLAYERS-1 for 4p; 'none'(bp) is the unbiased baseline, no leak to exploit, so it's excluded. Must have N_PLAYERS-1 entries.
BIAS_MULTIPLIER=${BIAS_MULTIPLIER:-5.0}    # opponent leak strength (biased action prob ×this, renormalized). 1.0=no leak (nothing to exploit); ~20+=near-pure/degenerate (infeasible). 5.0 = established mid-strength (biased action ~2-3× baseline, still mixing).
RUN_SEED=${RUN_SEED:-0}
BIG_BLIND=${BIG_BLIND:-100}
SMALL_BLIND=${SMALL_BLIND:-50}
STARTING_STACK=${STARTING_STACK:-10000}
LOW_CARD_RANK=${LOW_CARD_RANK:-2}
HIGH_CARD_RANK=${HIGH_CARD_RANK:-14}
WALL_TARGET=${WALL_TARGET:-660}            # per-decision wall budget (s) for the REPORT (flags over-budget cells + the A/B comparison point). 660 = the production safety wall (Pluribus 30s*22-core translated to one worker), matching the 2026-08 budgets (binding cell flop-n4 ~657s). Set empty to disable.
REGIME_AB_STREETS=${REGIME_AB_STREETS:-none}  # HU vector-vs-MCCFR A/B streets: none | turn | flop,turn. OFF: HU turn is DECIDED = vector (production routing), so the mccfr arm would only measure the road not taken.
CRN_VALUE=${CRN_VALUE:-false}            # DIAGNOSTIC ONLY (default off): the budget now comes from hot_l1 self-stability, no value estimate needed. true re-enables the CRN value-gap/replica-spread columns (extra compute).
CRN_WORLDS=${CRN_WORLDS:-32}              # fixed card-worlds per CRN value estimate (only used when CRN_VALUE=true)

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
echo "  - Max concurrent mw: ${MAX_CONCURRENT_MULTIWAY:-(uncapped)}  (peak-RAM cap on multiway MCCFR solves)"
echo "  - Live counts:       ${N_LIVE:-(all)}"
echo "  - CPUs:              ${SLURM_CPUS_PER_TASK:-(unset)}"
echo "  - Hands/reps:        $PER_CELL_CAP hands x $REPS reps (sampled cells); $PER_CELL_CAP_DETERMINISTIC hands x 1 rep (deterministic: vector river)"
echo "  - Ladder:            wall-anchored top = probe(${PROBE_SECONDS}s) x mccfr[${LADDER_TOP_SECONDS}]s/vector[${LADDER_TOP_SECONDS_VECTOR}]s (flop,turn,river), down to ${LADDER_LO}x top, $LADDER_POINTS pts"
echo "  - hot_l1 tol:        mccfr $HOT_L1_TOL_MCCFR / vector $HOT_L1_TOL_VECTOR  (budget = smallest rung with mean hot_l1 ≤ tol; hot_l1 spans the hero's whole root street, reach-weighted; MCCFR is looser because sampling leaves dispersion)"
echo "  - Wall target (s):   ${WALL_TARGET:-(disabled)}"
echo "  - Regime A/B:        $REGIME_AB_STREETS"
echo "  - CRN root value:    $CRN_VALUE  (worlds $CRN_WORLDS; false = raw internal MCCFR value)"
echo "  - Table policy:      $TABLE_POLICY  (seats: ${FIXED_SEATS:-n/a}; bias ×$BIAS_MULTIPLIER)"
echo "  - Search core:       ${PLURIBUS_SEARCH_CORE:-0}"
echo "  - LUT / blueprint:   $LUT_PATH  |  $BLUEPRINT_PATH"
echo "  - Output → perm:     $PERM_DIR"

# One calibrate invocation → its own out-dir.  Backgrounded + forwarding SLURM's
# grace-period SIGTERM (same pattern as evaluation.sh) so a wall-clock stop still hits the
# EXIT trap and copies partial output back.  ``CUR_PID`` is the child the trap signals.
CUR_PID=""
term() { [ -n "$CUR_PID" ] && { echo "Forwarding SIGTERM to calibration (pid $CUR_PID)"; kill -TERM "$CUR_PID" 2>/dev/null || true; }; }
trap term TERM

run_calibrate() {  # $1 = conditions, $2 = out-dir
  local conds="$1" outdir="$2"
  mkdir -p "$outdir"
  local extra=()
  [ -n "$WORKERS" ]     && extra+=(--workers "$WORKERS")
  [ -n "$MAX_CONCURRENT_MULTIWAY" ] && extra+=(--max-concurrent-multiway "$MAX_CONCURRENT_MULTIWAY")
  [ -n "$WALL_TARGET" ] && extra+=(--wall-target "$WALL_TARGET")
  if [ "$CRN_VALUE" = "true" ]; then extra+=(--crn-value --crn-worlds "$CRN_WORLDS"); else extra+=(--internal-value); fi
  python -m evaluation.calibrate run \
    --blueprint-path "$BLUEPRINT_PATH" \
    --lut-path "$LUT_PATH" \
    --n-players "$N_PLAYERS" \
    --conditions "$conds" \
    --model-p-max "$MODEL_P_MAX" \
    --model-error "$MODEL_ERROR" \
    --collect-hands "$COLLECT_HANDS" \
    --per-cell-cap "$PER_CELL_CAP" \
    --per-cell-cap-deterministic "$PER_CELL_CAP_DETERMINISTIC" \
    --reps "$REPS" \
    --hot-l1-tol-mccfr "$HOT_L1_TOL_MCCFR" \
    --hot-l1-tol-vector "$HOT_L1_TOL_VECTOR" \
    --ladder-points "$LADDER_POINTS" \
    --ladder-lo "$LADDER_LO" \
    --ladder-top-seconds "$LADDER_TOP_SECONDS" \
    --ladder-top-seconds-vector "$LADDER_TOP_SECONDS_VECTOR" \
    --probe-seconds "$PROBE_SECONDS" \
    --collect-iters "$COLLECT_ITERS" \
    --n-live "$N_LIVE" \
    --table-policy "$TABLE_POLICY" \
    --fixed-seats "$FIXED_SEATS" \
    --bias-multiplier "$BIAS_MULTIPLIER" \
    --run-seed "$RUN_SEED" \
    --big-blind "$BIG_BLIND" \
    --small-blind "$SMALL_BLIND" \
    --starting-stack "$STARTING_STACK" \
    --low-card-rank "$LOW_CARD_RANK" \
    --high-card-rank "$HIGH_CARD_RANK" \
    --regime-ab-streets "$REGIME_AB_STREETS" \
    --out-dir "$outdir" \
    "${extra[@]}" &
  CUR_PID=$!
  wait "$CUR_PID"
  CUR_PID=""
}

# Split CONDITIONS: OX arms calibrate SEPARATELY (gadget = different tree, β must agree, so
# they cannot share a solver_cfg with vanilla/DBR) into $PERM_DIR/ox; the rest run together.
BASE_CONDS=""; OX_CONDS=""
IFS=',' read -ra _CONDS <<< "$CONDITIONS"
for c in "${_CONDS[@]}"; do
  c_trim=$(echo "$c" | xargs); [ -z "$c_trim" ] && continue
  case "$(echo "$c_trim" | tr '[:upper:]' '[:lower:]')" in
    ox|ox\(*) OX_CONDS="${OX_CONDS:+$OX_CONDS,}$c_trim" ;;
    *)        BASE_CONDS="${BASE_CONDS:+$BASE_CONDS,}$c_trim" ;;
  esac
done

if [ -n "$BASE_CONDS" ]; then
  echo "=== Base calibration ($BASE_CONDS) → $PERM_DIR ==="
  run_calibrate "$BASE_CONDS" "$LOCAL_OUT"
fi
if [ -n "$OX_CONDS" ]; then
  echo "=== OX-Search calibration ($OX_CONDS, vector cells only) → $PERM_DIR/ox ==="
  run_calibrate "$OX_CONDS" "$LOCAL_OUT/ox"
fi
echo "Calibration finished; results in $PERM_DIR"
