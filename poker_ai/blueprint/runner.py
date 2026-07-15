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
# the box.  KEEP IN SYNC with the exports in scripts/training.sh
# (PLURIBUS_STRATEGY_PER_JOB / PLURIBUS_STRATEGY_BATCH_SIZE).  These are applied
# only when neither the corresponding --flag nor a pre-existing env var is set,
# so an explicit flag or export still wins (see :func:`start`).
_DEFAULT_STRATEGY_PER_JOB = 30
_DEFAULT_STRATEGY_BATCH_SIZE = 128


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
    default=70,
    help=(
        "Start updating the strategy after this many sync cycles "
        "(= N * sync_interval iterations)."
    ),
)
@click.option(
    "--strategy_interval",
    default=1,
    help="Update the current strategy every N sync cycles (= N * sync_interval iterations).",
)
@click.option(
    "--checkpoint_interval",
    default=2500,
    help=(
        "Write a training checkpoint every N sync cycles "
        "(= N * sync_interval iterations)."
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
    default=-300000000,
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
        "Average-strategy sample playthroughs folded into each CFR job, per "
        "player, after warm-up.  Sets the PLURIBUS_STRATEGY_PER_JOB environment "
        "variable so forked/spawned workers inherit it (the knob is env-only "
        f"otherwise).  Precedence: this flag > a pre-existing env var > the "
        f"cluster-matching default ({_DEFAULT_STRATEGY_PER_JOB}, in sync with "
        "scripts/training.sh).  Raise it to convert more search-leaf lookups from "
        "the last-iterate regret fallback to the converged average strategy "
        "(measure with `python -m evaluation.blueprint_metrics --leaf-coverage`)."
        "  Not a structural checkpoint key, so it may be changed on resume."
    ),
)
@click.option(
    "--strategy_batch_size",
    type=int,
    default=None,
    help=(
        "Average-strategy sample mass per player per sync cycle (single-process: "
        "playthroughs per strategy firing).  Sets the PLURIBUS_STRATEGY_BATCH_SIZE "
        "environment variable.  On the multi-process path it is inert while "
        "--strategy_per_job (or its env var) is set, since that pins the per-job "
        "count directly; otherwise it sizes the auto-default.  Precedence: this "
        f"flag > a pre-existing env var > the cluster-matching default "
        f"({_DEFAULT_STRATEGY_BATCH_SIZE}, in sync with scripts/training.sh)."
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
    n_processes,
    nickname: str,
    strategy_per_job,
    strategy_batch_size,
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

    # Average-strategy sampling knobs. Same rationale as PLURIBUS_CFR_CORE: the
    # server and forked/spawned workers read these from the environment, so the
    # resolved value must be exported here in the parent before any worker is
    # constructed. Precedence per knob: an explicit --flag wins; else a
    # pre-existing env var (e.g. an export from scripts/training.sh) is kept;
    # else the cluster-matching CLI default is applied so a bare `train start`
    # behaves like a cluster submission rather than falling to the low
    # sync-coupled auto-size.
    if strategy_per_job is not None:
        os.environ["PLURIBUS_STRATEGY_PER_JOB"] = str(strategy_per_job)
    elif "PLURIBUS_STRATEGY_PER_JOB" not in os.environ:
        os.environ["PLURIBUS_STRATEGY_PER_JOB"] = str(_DEFAULT_STRATEGY_PER_JOB)
    if strategy_batch_size is not None:
        os.environ["PLURIBUS_STRATEGY_BATCH_SIZE"] = str(strategy_batch_size)
    elif "PLURIBUS_STRATEGY_BATCH_SIZE" not in os.environ:
        os.environ["PLURIBUS_STRATEGY_BATCH_SIZE"] = str(_DEFAULT_STRATEGY_BATCH_SIZE)
    log.info(
        "Average-strategy knobs: PLURIBUS_STRATEGY_PER_JOB=%s "
        "PLURIBUS_STRATEGY_BATCH_SIZE=%s",
        os.environ["PLURIBUS_STRATEGY_PER_JOB"],
        os.environ["PLURIBUS_STRATEGY_BATCH_SIZE"],
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
            n_processes=n_processes,
            bias=bias,  # type: ignore[arg-type]
            bias_magnitude=bias_magnitude,
            warm_start=warm_start,
        )
        _safe_search(server)


if __name__ == "__main__":
    train()
