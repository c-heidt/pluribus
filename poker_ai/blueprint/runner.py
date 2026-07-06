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
from pathlib import Path
from typing import Dict

import click
import yaml

from poker_ai.blueprint.multiprocess.server import Server, WorkerError
from poker_ai.blueprint.singleprocess.train import simple_search


log = logging.getLogger("poker_ai.blueprint.runner")


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
    default=6,
    help="The number of players in the game.",
)
@click.option(
    "--max_runtime_hours",
    default=71.5,
    type=float,
    help="Wall-clock budget for this run in hours. Training stops when elapsed time reaches this limit.",
)
@click.option(
    "--sync_interval",
    default=500,
    help=(
        "How many iterations between worker sync barriers.  This is the base "
        "unit for all cycle-based options.  Higher values keep workers busier "
        "but delay delta merges."
    ),
)
@click.option(
    "--discount_interval",
    default=5,
    help=(
        "Apply LCFR discounting every N sync cycles "
        "(= N * sync_interval iterations)."
    ),
)
@click.option(
    "--discount_duration_cycles",
    default=100,
    help="Close the LCFR discount window after this many sync cycles (= N * sync_interval iterations).",
)
@click.option(
    "--update_threshold",
    default=60,
    help=(
        "Start updating the strategy after this many sync cycles "
        "(= N * sync_interval iterations)."
    ),
)
@click.option(
    "--strategy_interval",
    default=20,
    help="Update the current strategy every N sync cycles (= N * sync_interval iterations).",
)
@click.option(
    "--checkpoint_interval",
    default=60,
    help=(
        "Write a training checkpoint every N sync cycles "
        "(= N * sync_interval iterations)."
    ),
)
@click.option(
    "--prune_threshold",
    default=50000,
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
    sync_interval: int,
    discount_interval: int,
    checkpoint_interval: int,
    n_processes,
    nickname: str,
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
