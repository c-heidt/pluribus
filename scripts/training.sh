#!/bin/bash -l
# Slurm submission script to run training via the package CLI.
# Usage:
#   # Base blueprint (default):
#   sbatch --export=ALL,WORKSPACE=/path/to/ws training.sh
#
#   # Biased blueprint warm-started from a finished base run:
#   sbatch --export=ALL,WORKSPACE=/path/to/ws,BIAS=fold,WARM_START=/path/to/base training.sh
#SBATCH --job-name=pluribus-train
#SBATCH --output=logs/training-%j.out
#SBATCH --error=logs/training-%j_error.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=100000mb
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

# Training parameters (cycle-based options are counted in sync cycles = N * sync_interval iterations)
N_PLAYERS=${N_PLAYERS:-6}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-71.5}
SYNC_INTERVAL=${SYNC_INTERVAL:-750}
DISCOUNT_INTERVAL=${DISCOUNT_INTERVAL:-100}
DISCOUNT_DURATION_CYCLES=${DISCOUNT_DURATION_CYCLES:-2000}
UPDATE_THRESHOLD=${UPDATE_THRESHOLD:-200}
STRATEGY_INTERVAL=${STRATEGY_INTERVAL:-1}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-2000}
PRUNE_THRESHOLD=${PRUNE_THRESHOLD:-500000}
C=${C:--3000000}
PICKLE_DIR=${PICKLE_DIR:-false}
N_PROCESSES=${N_PROCESSES:-}
LUT_PATH=${LUT_PATH:-"$WORKSPACE/exact"}
# Bias / warm-start (for biased-blueprint training).  BIAS=none runs the
# standard base blueprint and ignores BIAS_MAGNITUDE / WARM_START.
BIAS=${BIAS:-none}
BIAS_MAGNITUDE=${BIAS_MAGNITUDE:-100}
WARM_START=${WARM_START:-}
# Default save dir varies by bias so concurrent biased runs don't
# collide.  base → ${N}player_52cards; biased → ..._${BIAS}_biased.
if [ "$BIAS" = "none" ]; then
  NICKNAME=${NICKNAME:-"$WORKSPACE/models/${N_PLAYERS}player_52cards"}
else
  NICKNAME=${NICKNAME:-"$WORKSPACE/models/${N_PLAYERS}player_52cards_${BIAS}_biased"}
fi

# Efficiency knobs honoured by the trainer (env-driven so they can be
# overridden per submission without editing code).
export PLURIBUS_CFR_BATCH_SIZE=${PLURIBUS_CFR_BATCH_SIZE:-5}
# Sampled playthroughs per update_strategy queue item.  Each strategy
# firing runs workers_per_player * this many playthroughs per player;
# playthroughs are single sampled lines, so this is cheap relative to
# the CFR work between firings but is what actually populates the
# average-strategy table.
export PLURIBUS_STRATEGY_BATCH_SIZE=${PLURIBUS_STRATEGY_BATCH_SIZE:-128}
export PLURIBUS_CHUNK_SIZE=${PLURIBUS_CHUNK_SIZE:-4000000}
# Shared-memory index cache: serves the per-node info-set lookup from shm
# instead of an LMDB read txn (the dominant inner-loop cost).  Capacities are
# per-street SLOT counts (pre_flop,flop,turn,river), each 24 bytes; a street
# holds up to capacity*0.5 infosets before it overflows.  These MUST cover the
# run's saturation — the mmap cannot grow once workers fork, so an undersized
# street fails LOUDLY and early (before real compute is spent), telling you to
# raise the value and restart.  The chosen sizes are persisted in the
# checkpoint so a resume reuses them.  Defaults ≈ 21 GiB of shm; raise --mem
# accordingly (chunk tables need the rest of RAM).
export PLURIBUS_INDEX_CACHE=${PLURIBUS_INDEX_CACHE:-1}
export PLURIBUS_INDEX_CAPACITY=${PLURIBUS_INDEX_CAPACITY:-"67108864,268435456,268435456,268435456"}
# Compiled Cython core for the CFR hot loop.  1 = every worker drives its
# traversals through poker_ai._core (byte-verified, fallback-retaining); set 0
# to force the pure-Python path (e.g. the A/B baseline arm).  REQUIRES the shm
# index cache above (pure-shm reads, no LMDB fallback) — already default-on —
# and the extension to be BUILT on this node (preflight check below).
export PLURIBUS_CFR_CORE=${PLURIBUS_CFR_CORE:-1}
# Deferred-durability allocation.  1 = the shm index cache assigns info-set row
# numbers under its own lightweight lock and LMDB is written in bulk only at
# checkpoints — taking the LMDB single-writer mutex off the allocation hot path
# (the measured 48-worker bottleneck once the compiled core is on).  Byte-
# identical single-process (golden-trace gated), fallback-retaining.  REQUIRES
# the shm index cache above (it IS the allocator); enforced by the guard below.
# Pure-Python — needs no build.  Set 0 to force the legacy per-infoset LMDB
# write-txn allocator (e.g. the A/B baseline arm).
export PLURIBUS_DEFERRED_ALLOC=${PLURIBUS_DEFERRED_ALLOC:-1}

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$(dirname "$NICKNAME")"

# Activate conda 
echo "Activating conda environment: $CONDA_ENV"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

cd "$PROJECT_DIR"

# Preflight: when the compiled core is requested it MUST be built on this node.
# CoreDriver imports the extension in each worker; a missing/stale .so would
# otherwise surface as a worker crash mid-run. Fail here with the build command
# instead. (Build once before submitting; do not build inside the job if
# PROJECT_DIR is shared across concurrent jobs — the .so write would race.)
if [ "$PLURIBUS_CFR_CORE" = "1" ]; then
  if ! python -c "import poker_ai._core._state, poker_ai._core._traverse" 2>/dev/null; then
    echo "ERROR: PLURIBUS_CFR_CORE=1 but the compiled core is not importable." >&2
    echo "       Build it on this node first:  python setup.py build_ext --inplace" >&2
    echo "       (or set PLURIBUS_CFR_CORE=0 to run the pure-Python path)." >&2
    exit 1
  fi
  echo "Compiled CFR core present and importable."
fi

# Deferred allocation IS the shm cache (it allocates rows there), so it is
# meaningless without it.  CFRTables silently disables deferred when the cache
# is off; catch the inconsistent request loudly instead of silently reverting to
# the slow LMDB-writer-mutex allocator.  No build check — deferred is pure Python.
if [ "$PLURIBUS_DEFERRED_ALLOC" = "1" ] && [ "$PLURIBUS_INDEX_CACHE" != "1" ]; then
  echo "ERROR: PLURIBUS_DEFERRED_ALLOC=1 requires PLURIBUS_INDEX_CACHE=1 (the shm" >&2
  echo "       cache is the allocator). Enable the cache or set DEFERRED_ALLOC=0." >&2
  exit 1
fi

# Ensure LUT path exists
if [ ! -d "$LUT_PATH" ]; then
  echo "ERROR: LUT path not found at $LUT_PATH" >&2
  echo "Please run abstraction first using: sbatch scripts/abstraction_auto_resub.sh" >&2
  exit 1
fi

# Per-job private working directory under TMPDIR.  Cluster doc says
# always use TMPDIR; whether TMPDIR is per-job or shared across
# concurrent users on the same node is implementation-defined, so we
# isolate our staged data under a job-specific subdirectory and
# tear the whole thing down on exit.  Everything below (LUT, LMDB,
# any future local artefacts) hangs off WORK_DIR.
WORK_DIR="${TMPDIR:?cluster requires TMPDIR to be set (do not fall back to /tmp)}/pluribus-${SLURM_JOB_ID:-$$}"
mkdir -p "$WORK_DIR"

# Stage the LUT to node-local fast scratch.  Without this, every river
# memmap lookup that misses the page cache becomes a network-FS round
# trip (the LUT typically lives on shared /pfs storage).  Set
# STAGE_LUT_LOCALLY=false to disable when local disk is too small.
STAGE_LUT_LOCALLY=${STAGE_LUT_LOCALLY:-true}
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

# Stage the LMDB indexes to node-local fast scratch.  Every CFR
# infoset lookup goes through ``InfosetIndex.get`` which opens an
# LMDB read transaction; on network FS the per-txn ``fcntl`` lock
# acquisition under 31 concurrent workers becomes a meaningful
# fraction of wall time.  The python-side ``CheckpointManager``
# mirrors the local LMDB back to ${NICKNAME}/lmdb_index at every
# checkpoint using ``env.copy`` + rsync, so a crash can never lose
# more than the most recent ``CHECKPOINT_INTERVAL`` of index data.
STAGE_LMDB_LOCALLY=${STAGE_LMDB_LOCALLY:-true}
if [ "$STAGE_LMDB_LOCALLY" = "true" ]; then
  LOCAL_LMDB_PATH="$WORK_DIR/lmdb"
  PERSISTENT_LMDB_PATH="$NICKNAME/lmdb_index"
  mkdir -p "$LOCAL_LMDB_PATH"
  if [ -d "$PERSISTENT_LMDB_PATH" ]; then
    echo "Staging LMDB from $PERSISTENT_LMDB_PATH to $LOCAL_LMDB_PATH ..."
    rsync_start=$(date +%s)
    rsync -a "$PERSISTENT_LMDB_PATH/" "$LOCAL_LMDB_PATH/"
    echo "LMDB staged in $(($(date +%s) - rsync_start))s ($(du -sh "$LOCAL_LMDB_PATH" 2>/dev/null | cut -f1))"
  else
    echo "No existing LMDB at $PERSISTENT_LMDB_PATH — starting fresh in $LOCAL_LMDB_PATH"
  fi
  export PLURIBUS_LMDB_LOCAL_DIR="$LOCAL_LMDB_PATH"
fi

# Cleanup: runs on any exit (clean or signalled).  Just removes the
# job-private working directory.  No safety-net rsync to persistent
# storage — the python ``CheckpointManager`` mirrors the LMDB at
# every checkpoint while the run is alive, so the persistent state
# after any crash is exactly the most recent successful checkpoint
# (chunks + LMDB together).  Anything since that checkpoint is at
# most ``CHECKPOINT_INTERVAL`` of training and is preferable to lose
# rather than risk a divergence between persistent LMDB and chunks.
trap 'rm -rf "$WORK_DIR"' EXIT

echo "Starting training with:"
echo "  - Players:                     $N_PLAYERS"
echo "  - Max runtime (hours):         $MAX_RUNTIME_HOURS"
echo "  - Sync interval (iters):       $SYNC_INTERVAL"
echo "  - Discount interval (cycles):  $DISCOUNT_INTERVAL"
echo "  - Discount duration (cycles):  $DISCOUNT_DURATION_CYCLES"
echo "  - Update threshold (cycles):   $UPDATE_THRESHOLD"
echo "  - Strategy interval (cycles):  $STRATEGY_INTERVAL"
echo "  - Checkpoint interval (cycles):$CHECKPOINT_INTERVAL"
echo "  - Prune threshold (iters):     $PRUNE_THRESHOLD"
echo "  - C (pruning regret):          $C"
echo "  - Pickle dir:                  $PICKLE_DIR"
echo "  - N processes:                 ${N_PROCESSES:-(auto)}"
echo "  - LUT path:                    $LUT_PATH"
echo "  - Nickname:                    $NICKNAME"
echo "  - Bias:                        $BIAS"
if [ "$BIAS" != "none" ]; then
  echo "  - Bias magnitude:              $BIAS_MAGNITUDE"
fi
echo "  - Warm start:                  ${WARM_START:-(none)}"
echo "  - CPUs:                        $SLURM_CPUS_PER_TASK"
echo "  - PLURIBUS_CFR_BATCH_SIZE:     $PLURIBUS_CFR_BATCH_SIZE"
echo "  - PLURIBUS_STRATEGY_BATCH_SIZE:$PLURIBUS_STRATEGY_BATCH_SIZE"
echo "  - PLURIBUS_CFR_CORE:           $PLURIBUS_CFR_CORE"
echo "  - PLURIBUS_DEFERRED_ALLOC:     $PLURIBUS_DEFERRED_ALLOC"
echo "  - PLURIBUS_INDEX_CACHE:        $PLURIBUS_INDEX_CACHE"
echo "  - PLURIBUS_INDEX_CAPACITY:     $PLURIBUS_INDEX_CAPACITY"
echo "  - PLURIBUS_CHUNK_SIZE:         $PLURIBUS_CHUNK_SIZE"
echo "  - STAGE_LUT_LOCALLY:           $STAGE_LUT_LOCALLY"
echo "  - STAGE_LMDB_LOCALLY:          $STAGE_LMDB_LOCALLY"
echo "  - PLURIBUS_LMDB_LOCAL_DIR:     ${PLURIBUS_LMDB_LOCAL_DIR:-(unset)}"

# Build optional flags
EXTRA_ARGS=()
[ -n "$N_PROCESSES" ]      && EXTRA_ARGS+=(--n_processes "$N_PROCESSES")
[ "$PICKLE_DIR" = "true" ] && EXTRA_ARGS+=(--pickle_dir)
if [ "$BIAS" != "none" ]; then
  EXTRA_ARGS+=(--bias "$BIAS" --bias_magnitude "$BIAS_MAGNITUDE")
  if [ -z "$WARM_START" ]; then
    echo "ERROR: BIAS=$BIAS requires WARM_START to point at a finished base blueprint." >&2
    exit 1
  fi
fi
if [ -n "$WARM_START" ]; then
  if [ ! -d "$WARM_START" ]; then
    echo "ERROR: WARM_START path not found: $WARM_START" >&2
    exit 1
  fi
  EXTRA_ARGS+=(--warm_start "$WARM_START")
fi

# When profiling, let the trainer authorize py-spy to attach.  The profiler
# runs in a sibling subshell (below), so under ptrace_scope=1 the trainer must
# opt in via prctl(PR_SET_PTRACER_ANY) — see runner.py _allow_ptrace_if_requested.
# Must be exported BEFORE the trainer launches so it (and its forked workers) see it.
[ -n "${PROFILE:-}" ] && export PLURIBUS_ALLOW_PTRACE=1

# Run the trainer in the background so this shell can forward
# slurm's grace-period SIGTERM to the python process.  When SLURM
# signals a batch job, the signal goes to the bash wrapper — not to
# its foreground child.  Without explicit forwarding the trainer
# keeps running as an orphan, never sees SIGTERM, and gets
# SIGKILL'd at the wall-clock limit with no chance to write a final
# checkpoint.
poker_ai train start \
  --multi_process \
  --n_players "$N_PLAYERS" \
  --max_runtime_hours "$MAX_RUNTIME_HOURS" \
  --sync_interval "$SYNC_INTERVAL" \
  --discount_interval "$DISCOUNT_INTERVAL" \
  --discount_duration_cycles "$DISCOUNT_DURATION_CYCLES" \
  --update_threshold "$UPDATE_THRESHOLD" \
  --strategy_interval "$STRATEGY_INTERVAL" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --prune_threshold "$PRUNE_THRESHOLD" \
  --c "$C" \
  --lut_path "$LUT_PATH" \
  --nickname "$NICKNAME" \
  "${EXTRA_ARGS[@]}" &
TRAINER_PID=$!

_forward_signal() {
  local sig=$1
  echo "[batch] received SIG${sig} — forwarding to trainer (pid=${TRAINER_PID})"
  kill -"${sig}" "${TRAINER_PID}" 2>/dev/null || true
}
trap '_forward_signal TERM' TERM
trap '_forward_signal INT' INT

# ---------------------------------------------------------------------------
# Optional in-run profiler.  Completely inert unless PROFILE is set, so the
# production submission path above is unchanged.  Attaches py-spy to the LIVE
# trainer process tree (server + every worker, via --subprocesses) after a
# warmup, so the samples reflect steady state at real scale rather than the
# cold-start prewarm + initial allocation burst.  --idle is essential: it keeps
# OFF-cpu samples (threads blocked on the alloc lock, IPC _recv, or an LMDB txn)
# so the capture can rank *compute* against *lookup/sync* — the whole point of
# this measurement.  Output is speedscope JSON under $NICKNAME/profiles, viewable
# offline at https://speedscope.app (Left-Heavy + Sandwich views).
#
# Knobs (env, all optional):
#   PROFILE_WARMUP    seconds to wait after launch before the first capture (1200)
#   PROFILE_DURATION  seconds per capture (180)
#   PROFILE_GAP       seconds between captures (600)
#   PROFILE_N_SAMPLES number of captures (3)
#   PROFILE_RATE      samples/sec (default 25; py-spy samples every process
#                     serially per tick, so with ~48 workers it CANNOT sustain a
#                     high rate — it logs "N s behind in sampling" and the
#                     profile skews.  Keep this low for many-worker runs.)
if [ -n "${PROFILE:-}" ]; then
  if ! command -v py-spy >/dev/null 2>&1; then
    echo "[profile] py-spy not found — installing into $CONDA_ENV"
    pip install --quiet py-spy || echo "[profile] py-spy install FAILED; captures will be skipped"
  fi
  PROFILE_DIR="$NICKNAME/profiles"
  PROFILE_WARMUP=${PROFILE_WARMUP:-1200}
  PROFILE_DURATION=${PROFILE_DURATION:-180}
  PROFILE_GAP=${PROFILE_GAP:-600}
  PROFILE_N_SAMPLES=${PROFILE_N_SAMPLES:-3}
  PROFILE_RATE=${PROFILE_RATE:-25}
  if ! mkdir -p "$PROFILE_DIR" 2>/dev/null; then
    echo "[profile] WARNING: cannot create $PROFILE_DIR — captures will have nowhere to go"
  fi
  echo "[profile] output dir: $PROFILE_DIR"
  echo "[profile] ptrace_scope=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo '?') (0/1=attach ok, 2/3=blocked)"
  (
    echo "[profile] warmup ${PROFILE_WARMUP}s before first capture (steady-state)"
    sleep "$PROFILE_WARMUP"
    # Attach self-test BEFORE spending a capture: prove py-spy can ptrace the
    # trainer.  If this fails, every record below fails too — surface why now
    # instead of silently producing no files.
    if py-spy dump --pid "$TRAINER_PID" >/dev/null 2>"$PROFILE_DIR/.spy_selftest.err"; then
      echo "[profile] attach self-test OK (py-spy can read pid $TRAINER_PID)"
    else
      echo "[profile] ATTACH FAILED — py-spy cannot ptrace pid $TRAINER_PID; NO files will be saved."
      echo "[profile]   py-spy: $(tr '\n' ' ' < "$PROFILE_DIR/.spy_selftest.err")"
      echo "[profile]   fix: run on a node with ptrace_scope<=1, or grant the job CAP_SYS_PTRACE."
    fi
    for i in $(seq 1 "$PROFILE_N_SAMPLES"); do
      kill -0 "$TRAINER_PID" 2>/dev/null || { echo "[profile] trainer exited — stopping"; break; }
      ts=$(date +%s)
      base="spy_${ts}_sample${i}.speedscope.json"
      # Capture to node-local scratch first — py-spy writing straight to Lustre
      # is fragile (a client hiccup can lose the file even on exit 0).  Fold
      # py-spy's own stderr (the "Wrote speedscope file to ... Samples: N
      # Errors: M" report) into this log so it is visible inline, not stranded
      # in the separate SLURM _error.out.
      local_out="$WORK_DIR/$base"
      out="$PROFILE_DIR/$base"
      echo "[profile] capture ${i}/${PROFILE_N_SAMPLES} (${PROFILE_DURATION}s @ ${PROFILE_RATE}Hz, subprocesses+idle) -> local $local_out"
      if py-spy record --pid "$TRAINER_PID" --subprocesses --idle \
           --rate "$PROFILE_RATE" --duration "$PROFILE_DURATION" \
           --format speedscope --output "$local_out" 2>&1; then
        if [ -s "$local_out" ]; then
          sz=$(stat -c%s "$local_out" 2>/dev/null || echo 0)
          if cp -f "$local_out" "$out" 2>/dev/null && sync && [ -s "$out" ]; then
            echo "[profile] SAVED $out (${sz}B local, copied to /pfs OK)"
          else
            echo "[profile] captured OK locally (${sz}B) but COPY TO /pfs FAILED: $out — Lustre?  Local copy: $local_out"
          fi
        else
          echo "[profile] py-spy exit 0 but produced NO/empty local file — see its Samples/Errors line above (attach to workers failed?)"
        fi
      else
        rc=$?
        echo "[profile] py-spy capture ${i} FAILED (exit $rc) — no file (see attach self-test above)"
      fi
      sleep "$PROFILE_GAP"
    done
    n=$(ls -1 "$PROFILE_DIR"/spy_*.speedscope.json 2>/dev/null | wc -l)
    echo "[profile] profiler finished — ${n} file(s) in $PROFILE_DIR"
  ) &
  echo "[profile] enabled — warmup=${PROFILE_WARMUP}s, ${PROFILE_N_SAMPLES}×${PROFILE_DURATION}s captures to $PROFILE_DIR"
fi

# `wait` returns when interrupted by a signal (after running the
# trap), even if the child is still running.  Loop until the child
# has actually exited so the EXIT trap (which cleans up the local
# LUT copy) only fires *after* the trainer's final checkpoint
# has finished.
set +e
while true; do
  wait "${TRAINER_PID}"
  EXIT_CODE=$?
  kill -0 "${TRAINER_PID}" 2>/dev/null || break
done
set -e
exit "${EXIT_CODE}"
