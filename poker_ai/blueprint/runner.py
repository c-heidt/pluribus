"""Command-line entry point for CFR training.

Defines the ``poker_ai train`` Click command group and its ``start``
subcommand.  The command parses hyperparameters, writes them to a
``config.yaml`` alongside the save directory, and dispatches to
either the single-process :func:`simple_search` or the multi-process
:class:`Server` depending on the ``--single_process/--multi_process``
flag.

All cycle-based options are counted in sync cycles (= ``sync_interval``
raw iterations): ``strategy_interval``, ``discount_interval``,
``checkpoint_interval``, ``discount_duration_cycles``, and
``update_threshold``.  Only ``prune_threshold`` stays in raw
iterations because pruning is a per-CFR-call decision.

Resume behaviour
----------------
A run auto-resumes whenever the resolved save directory already
contains a valid checkpoint.  The
:class:`~poker_ai.tables.checkpoint.CheckpointManager` compares the saved
structural hyperparameters against the current ones and refuses to
restore on mismatches, so simply re-invoking ``poker_ai train start``
with the same ``--nickname`` continues an interrupted run.  There is
no separate ``resume`` command.
"""
import logging
import os
from pathlib import Path
from typing import Dict

import click
import yaml

from poker_ai.blueprint.multiprocess.server import Server, WorkerError
from poker_ai.blueprint.singleprocess.train import simple_search


log = logging.getLogger("poker_ai.blueprint.runner")

# Default average-strategy sampling knobs for a bare CLI run, chosen so
# ``poker_ai train start`` behaves identically to a cluster submission out of
# the box.  KEEP IN SYNC with the export in scripts/training.sh
# (PLURIBUS_STRATEGY_PER_JOB).  Applied only when neither the --flag nor a
# pre-existing env var is set, so an explicit flag or export still wins (see
# :func:`start`).  The average strategy is now pre-flop only with full opponent
# branching, so one pass per deal already covers the whole pre-flop opponent
# tree — a single playthrough per job is plenty (the old high count and its
# auto-sizing existed to feed the abandoned post-flop average).
_DEFAULT_STRATEGY_PER_JOB = 1


def _safe_search(server: Server):
    """Run :meth:`Server.search` inside a resilient try/except/terminate.

    Wraps the server's main loop with three exit paths:

    - **Clean return.**  Call :meth:`Server.terminate` for an
      orderly worker shutdown.
    - **Worker error.**  :class:`WorkerError` is raised from inside
      the loop when a worker has set the shared error event.  The
      queue may be in an inconsistent state, so call
      :meth:`Server.terminate(safe=False) <Server.terminate>` to
      kill every worker without going through the queue.
    - **User interrupt.**  :class:`KeyboardInterrupt` or
      :class:`SystemExit`.  Fall through to a safe termination so
      the interactive user sees their workers exit cleanly.

    Parameters
    ----------
    server : Server
        Already-constructed training server.
    """
    try:
        server.search()
    except WorkerError as exc:
        log.error(f"Fatal worker error: {exc}")
        server.terminate(safe=False)
    except (KeyboardInterrupt, SystemExit):
        log.info(
            "Early termination of program. Please wait for workers to "
            "terminate."
        )
        server.terminate()
    else:
        server.terminate()
    log.info("All workers terminated. Quitting program - thanks for using me!")


@click.group()
def train():
    """Click group registering the training subcommands."""
    pass


@train.command()
@click.option(
    "--n_players",
    default=2,
    help="The number of players in the game.",
)
@click.option(
    "--max_runtime_hours",
    default=8.0,
    type=float,
    help="Wall-clock budget for this run in hours. Training stops when elapsed time reaches this limit.",
)
@click.option(
    "--sync_interval",
    default=1000,
    help=(
        "How many iterations between worker sync barriers.  This is the base "
        "unit for all cycle-based options.  Higher values keep workers busier "
        "but delay delta merges."
    ),
)
@click.option(
    "--discount_interval",
    default=100,
    help=(
        "Apply LCFR discounting every N sync cycles "
        "(= N * sync_interval iterations)."
    ),
)
@click.option(
    "--discount_duration_cycles",
    default=2000,
    help="Close the LCFR discount window after this many sync cycles (= N * sync_interval iterations).",
)
@click.option(
    "--update_threshold",
    default=3300,
    help=(
        "Start accumulating the pre-flop average strategy after this many sync "
        "cycles (= N * sync_interval iterations) — the average-strategy warm-up "
        "that keeps the near-random early era out of the running mean.  Default "
        "3300 matches --checkpoint_start_cycles (~6.25%% of an 8h/20-card 2p "
        "run), so pre-flop φ and the post-flop snapshots share one warm-up."
    ),
)
@click.option(
    "--strategy_interval",
    default=1,
    help="Update the current strategy every N sync cycles (= N * sync_interval iterations).",
)
@click.option(
    "--checkpoint_interval",
    default=1650,
    help=(
        "Write a training checkpoint every N sync cycles "
        "(= N * sync_interval iterations).  Every checkpoint is RETAINED "
        "(previous generations are no longer deleted) so the offline "
        "average-strategy tool can treat them as post-flop snapshots — this "
        "interval therefore also sets the snapshot cadence.  Defaults are "
        "PROPORTIONAL to the 4p production schedule (scripts/training.sh: 3h "
        "of a 96h run = 3.125%): 1650 = 3.125% of an 8h/20-card 2p run "
        "(~52,500 cycles at the throughput measured from the 8h checkpoint, "
        "t=52.5M, sync_interval=1000), giving ~30 snapshots like production."
    ),
)
@click.option(
    "--checkpoint_start_cycles",
    default=3300,
    help=(
        "Only start writing (and retaining) checkpoints once sync_step reaches "
        "this many cycles; the first checkpoint fires at the first "
        "checkpoint_interval multiple >= this value.  0 checkpoints from the "
        "beginning (no gate).  Default 3300 = 6.25% of an 8h/20-card 2p run "
        "(~52,500 cycles) — PROPORTIONAL to the 4p production warm-up (6h of a "
        "96h run = 6.25%), not a hard hour barrier — so the retained snapshots "
        "skip the near-random early era (and the LCFR discount window) exactly "
        "as production does.  An end-of-run / SIGTERM checkpoint is always "
        "written regardless of this gate so an orderly stop stays resumable."
    ),
)
@click.option(
    "--prune_threshold",
    default=2500000,
    help=(
        "When a uniform random number is less than 95%, and the iteration > "
        "prune_threshold, use CFR with pruning.  Counted in raw iterations "
        "because pruning is a per-CFR-call decision."
    ),
)
@click.option(
    "--c",
    default=-3000000,
    help=(
        "Pruning threshold for regret, which means when we are using CFR with "
        "pruning and have a state with a regret of less than `c`, then we'll "
        "elect to not recusrively visit it and it's child nodes."
    ),
)
@click.option(
    "--pickle_dir",
    default=False,
    help=(
        "Whether or not the lut files are pickle files. This lookup "
        "method is deprecated."
    ),
)
@click.option(
    "--n_processes",
    default=None,
    type=int,
    help=(
        "Number of worker processes to spawn. Defaults to cpu_count-1 "
        "(or SLURM_CPUS_PER_TASK-1 when running under Slurm)."
    ),
)

@click.option(
    "--lut_path",
    default=".",
    help="The path to the files for clustering the infosets.",
)
@click.option("--nickname", default="", help="The nickname of the study.")
@click.option(
    "--n_iterations",
    default=10000,
    hidden=True,
    help="[Single-process only] Number of iterations for the validation baseline.",
)
@click.option(
    "--single_process/--multi_process",
    default=False,
    help="Either use or don't use multiple processes.",
)
@click.option(
    "--cfr_core/--no_cfr_core",
    default=True,
    help=(
        "Drive CFR traversals through the compiled Cython core (~14x faster "
        "hot loop). Sets the PLURIBUS_CFR_CORE environment variable, so an "
        "explicit PLURIBUS_CFR_CORE in the environment (e.g. from a launch "
        "script) is overridden by this flag. The core requires the shm index "
        "cache (PLURIBUS_INDEX_CACHE=1, on by default) and the extension to be "
        "built; pass --no_cfr_core for the pure-Python path (e.g. an A/B "
        "baseline or an unbuilt checkout)."
    ),
)
@click.option(
    "--strategy_per_job",
    type=int,
    default=None,
    help=(
        "Pre-flop UPDATE-STRATEGY playthroughs per player folded into each CFR "
        "job (multi-process) / per strategy firing (single-process), after "
        "warm-up.  Sets the PLURIBUS_STRATEGY_PER_JOB environment variable so "
        "forked/spawned workers inherit it (the knob is env-only otherwise).  "
        f"Precedence: this flag > a pre-existing env var > the default "
        f"({_DEFAULT_STRATEGY_PER_JOB}).  One pass covers the whole pre-flop "
        "opponent tree per deal (full branching), so 1-2 is plenty; more just "
        "samples more deals.  Not a structural checkpoint key, so it may be "
        "changed on resume."
    ),
)
@click.option(
    "--bias",
    type=click.Choice(["none", "fold", "call", "raise"]),
    default="none",
    help=(
        "Bias class for biased-blueprint training.  'none' (default) "
        "is the standard base blueprint."
    ),
)
@click.option(
    "--bias_magnitude",
    type=float,
    default=100,
    help=(
        "Per-occurrence bonus added to terminal payoff for actions in "
        "the biased class.  Ignored when --bias none."
    ),
)
@click.option(
    "--warm_start",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help=(
        "Path to a base-blueprint save directory whose regret + strategy "
        "tables seed this run.  Honoured only on a fresh run; if the "
        "destination already contains a checkpoint the warm-start is "
        "skipped (resume wins)."
    ),
)
def start(
    strategy_interval: int,
    max_runtime_hours: float,
    discount_duration_cycles: int,
    n_iterations: int,
    prune_threshold: int,
    c: int,
    n_players: int,
    update_threshold: int,
    lut_path: str,
    pickle_dir: bool,
    single_process: bool,
    cfr_core: bool,
    sync_interval: int,
    discount_interval: int,
    checkpoint_interval: int,
    checkpoint_start_cycles: int,
    n_processes,
    nickname: str,
    strategy_per_job,
    bias: str,
    bias_magnitude: float,
    warm_start: str,
):
    """Train a CFR agent, auto-resuming if a valid checkpoint exists.

    Resolves the save directory from the ``--nickname`` option,
    persists the full hyperparameter dict to ``config.yaml`` for
    provenance, and hands off to either the single-process or
    multi-process entry point.  When the resolved save directory
    already contains a valid checkpoint the training run continues
    from it; when not, it starts fresh.
    """
    # Select the CFR execution path before any table/worker is constructed:
    # core_runner.core_enabled() and the workers read PLURIBUS_CFR_CORE, and
    # forked/spawned workers inherit this env, so it must be set here in the
    # parent. The flag is authoritative over any pre-existing env value.
    os.environ["PLURIBUS_CFR_CORE"] = "1" if cfr_core else "0"
    log.info(
        "CFR execution path: %s (PLURIBUS_CFR_CORE=%s)",
        "compiled Cython core" if cfr_core else "pure Python",
        os.environ["PLURIBUS_CFR_CORE"],
    )

    # Average-strategy sampling knob. Same rationale as PLURIBUS_CFR_CORE: the
    # server and forked/spawned workers read it from the environment, so the
    # resolved value must be exported here in the parent before any worker is
    # constructed. Precedence: an explicit --flag wins; else a pre-existing env
    # var (e.g. an export from scripts/training.sh) is kept; else the CLI default.
    if strategy_per_job is not None:
        os.environ["PLURIBUS_STRATEGY_PER_JOB"] = str(strategy_per_job)
    elif "PLURIBUS_STRATEGY_PER_JOB" not in os.environ:
        os.environ["PLURIBUS_STRATEGY_PER_JOB"] = str(_DEFAULT_STRATEGY_PER_JOB)
    log.info(
        "Average-strategy knob: PLURIBUS_STRATEGY_PER_JOB=%s",
        os.environ["PLURIBUS_STRATEGY_PER_JOB"],
    )

    config: Dict[str, int] = {**locals()}
    save_path: Path = Path(nickname)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.yaml", "w") as steam:
        yaml.dump(config, steam)
    if single_process:
        log.info(
            "Only one process specified so using poker_ai.blueprint.singleprocess."
            "simple_search for the optimisation."
        )
        simple_search(
            config=config,
            save_path=save_path,
            lut_path=lut_path,
            pickle_dir=pickle_dir,
            strategy_interval=strategy_interval,
            n_iterations=n_iterations,
            discount_duration_cycles=discount_duration_cycles,
            prune_threshold=prune_threshold,
            c=c,
            n_players=n_players,
            update_threshold=update_threshold,
            sync_interval=sync_interval,
            discount_interval=discount_interval,
            bias=bias,  # type: ignore[arg-type]
            bias_magnitude=bias_magnitude,
            warm_start=warm_start,
        )
    else:
        log.info(
            "Mulitple processes specifed so using poker_ai.blueprint.multiprocess."
            "server.Server for the optimisation."
        )
        server = Server(
            strategy_interval=strategy_interval,
            max_runtime_hours=max_runtime_hours,
            discount_duration_cycles=discount_duration_cycles,
            prune_threshold=prune_threshold,
            c=c,
            n_players=n_players,
            update_threshold=update_threshold,
            save_path=save_path,
            lut_path=lut_path,
            pickle_dir=pickle_dir,
            sync_interval=sync_interval,
            discount_interval=discount_interval,
            checkpoint_interval=checkpoint_interval,
            checkpoint_start_cycles=checkpoint_start_cycles,
            n_processes=n_processes,
            bias=bias,  # type: ignore[arg-type]
            bias_magnitude=bias_magnitude,
            warm_start=warm_start,
        )
        _safe_search(server)


@train.command(name="average")
@click.option(
    "--train_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help=(
        "Training directory holding lmdb_index/ and the retained "
        "checkpoint_<t>/ generations to average."
    ),
)
@click.option(
    "--output_dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Destination directory for the final averaged blueprint.",
)
@click.option(
    "--scale",
    type=int,
    default=None,
    help=(
        "Integer scale for the stored post-flop strategy pseudo-counts "
        "(default 1,000,000). Larger just uses more of the int32 range; the "
        "readout normalises it away."
    ),
)
@click.option(
    "--min_t",
    type=int,
    default=None,
    help=(
        "Exclude snapshots below this iteration t from the post-flop average. "
        "Defaults to the warm-up (checkpoint_start_cycles * sync_interval) "
        "recorded in the latest checkpoint. Pass 0 to average every retained "
        "checkpoint."
    ),
)
@click.option(
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help=(
        "Post-flop (street, chunk) tasks to average concurrently.  Each task "
        "streams one chunk at a time, so peak RAM is roughly workers * 800 MB "
        "at the default chunk size — this flag is the memory dial as well as "
        "the speed dial.  The work is embarrassingly parallel (each chunk reads "
        "its own files and writes its own output), so speedup is near-linear "
        "until the filesystem saturates."
    ),
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help=(
        "Continue an interrupted build in an existing --output_dir instead of "
        "refusing it: artefacts already present are skipped.  Every output is "
        "written atomically (temp + rename), so anything on disk is complete "
        "and safe to skip — a killed job never leaves a half-written chunk that "
        "resume would trust."
    ),
)
@click.option(
    "--min_confirming_snapshots",
    type=int,
    default=None,
    help=(
        "Minimum number of independent averaged snapshots that must show "
        "positive regret for a post-flop row before it is published (default "
        "2). This is a FLOOR, not the only requirement — see "
        "--min_confirming_fraction, which usually dominates it on any run "
        "with more than a handful of retained snapshots. Rows short of the "
        "effective requirement are written all-zero and correctly deferred to "
        "the live regret-match fallback at read time, instead of publishing a "
        "falsely-confident average."
    ),
)
@click.option(
    "--min_confirming_fraction",
    type=float,
    default=None,
    help=(
        "Minimum FRACTION of all averaged snapshots that must independently "
        "confirm a row (default 0.5, a majority), IN ADDITION to "
        "--min_confirming_snapshots: the effective requirement is "
        "max(min_confirming_snapshots, ceil(min_confirming_fraction * "
        "n_snapshots_averaged)). The absolute floor alone doesn't scale — on "
        "a run with dozens of retained snapshots, 'confirmed by any 2 of "
        "them' is a very low bar, since a single-touch positive-regret blip "
        "(see the module docstring) only needs to land in 2 out of, say, 81 "
        "snapshots, which is close to certain for anything touched at all "
        "across a long run. Expressing the bar as a fraction of THIS run's "
        "own snapshot count keeps it meaningful regardless of how many "
        "checkpoints were retained."
    ),
)
@click.option(
    "--min_snapshot_regret_magnitude",
    type=int,
    default=None,
    help=(
        "Optional: minimum sum(positive regret) a SINGLE snapshot's row must "
        "show before that snapshot counts toward confirming a row at all (on "
        "top of, not instead of, plain positivity). Unset (default) keeps the "
        "plain 'any positive' per-snapshot test. There is NO built-in default "
        "value for this one, deliberately: what counts as meaningful regret "
        "for a single snapshot depends on this run's chip/payoff scale, so "
        "calibrate it for your specific run rather than trusting a guessed "
        "constant across configurations."
    ),
)
def average_snapshots(
    train_dir, output_dir, scale, min_t, workers, resume,
    min_confirming_snapshots, min_confirming_fraction, min_snapshot_regret_magnitude,
):
    """Build a final blueprint by averaging a run's retained snapshots.

    Reconstructs the post-flop average strategy offline from the retained
    checkpoints (Pluribus-style snapshot averaging) and writes a blueprint
    directory loadable by the evaluation / search stack unchanged.
    """
    from poker_ai.blueprint.offline_average import (
        MIN_CONFIRMING_FRACTION_DEFAULT,
        MIN_CONFIRMING_SNAPSHOTS_DEFAULT,
        SIGMA_SCALE_DEFAULT,
        build_final_blueprint,
    )

    build_final_blueprint(
        Path(train_dir),
        Path(output_dir),
        scale=SIGMA_SCALE_DEFAULT if scale is None else scale,
        min_t=min_t,
        workers=workers,
        resume=resume,
        min_confirming_snapshots=(
            MIN_CONFIRMING_SNAPSHOTS_DEFAULT if min_confirming_snapshots is None
            else min_confirming_snapshots
        ),
        min_confirming_fraction=(
            MIN_CONFIRMING_FRACTION_DEFAULT if min_confirming_fraction is None
            else min_confirming_fraction
        ),
        min_snapshot_regret_magnitude=min_snapshot_regret_magnitude,
    )


if __name__ == "__main__":
    train()
