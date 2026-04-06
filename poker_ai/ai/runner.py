"""Script for using multiprocessing to train agent.

All cycle-based options are counted in sync cycles (= ``sync_interval``
raw iterations): ``strategy_interval``, ``discount_interval``,
``checkpoint_interval``, ``discount_duration_cycles``, and
``update_threshold``.  Only ``prune_threshold`` stays in raw
iterations, because pruning is a per-CFR-call decision.

Resume: a run auto-resumes if ``save_path`` already contains a valid
checkpoint.  The ``CheckpointManager`` validates that the saved config
matches the current config and refuses to restore on structural
mismatches.  There is no separate ``resume`` command.
"""
import logging
from pathlib import Path
from typing import Dict

import click
import yaml

from poker_ai import utils
from poker_ai.ai.multiprocess.server import Server, WorkerError
from poker_ai.ai.singleprocess.train import simple_search


log = logging.getLogger("poker_ai.ai.runner")


def _safe_search(server: Server):
    """Safely run the server, and allow user to control c."""
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
    """Train a poker AI."""
    pass


@train.command()
@click.option(
    "--strategy_interval",
    default=20,
    help="Update the current strategy every N sync cycles (= N * sync_interval iterations).",
)
@click.option(
    "--max_runtime_hours",
    default=1.0,
    type=float,
    help="Wall-clock budget for this run in hours. Training stops when elapsed time reaches this limit.",
)
@click.option(
    "--discount_duration_cycles",
    default=40,
    help="Close the LCFR discount window after this many sync cycles (= N * sync_interval iterations).",
)
@click.option(
    "--n_iterations",
    default=10000,
    hidden=True,
    help="[Single-process only] Number of iterations for the validation baseline.",
)
@click.option(
    "--prune_threshold",
    default=5000,
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
    "--n_players",
    default=3,
    help="The number of players in the game."
)
@click.option(
    "--update_threshold",
    default=4,
    help=(
        "Start updating the strategy after this many sync cycles "
        "(= N * sync_interval iterations)."
    ),
)
@click.option(
    "--lut_path",
    default=".",
    help=(
        "The path to the files for clustering the infosets."
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
    "--single_process/--multi_process",
    default=False,
    help="Either use or don't use multiple processes.",
)
@click.option(
    "--sync_interval",
    default=250,
    help=(
        "How many iterations between worker sync barriers.  This is the base "
        "unit for all cycle-based options.  Higher values keep workers busier "
        "but delay delta merges."
    ),
)
@click.option(
    "--discount_interval",
    default=10,
    help=(
        "Apply LCFR discounting every N sync cycles "
        "(= N * sync_interval iterations).  Default 1 discounts every sync cycle."
    ),
)
@click.option(
    "--checkpoint_interval",
    default=40,
    help=(
        "Write a training checkpoint every N sync cycles "
        "(= N * sync_interval iterations)."
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
@click.option("--nickname", default="", help="The nickname of the study.")
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
):
    """Train agent from scratch — auto-resumes if a valid checkpoint exists."""
    config: Dict[str, int] = {**locals()}
    save_path: Path = utils.io.create_dir(nickname)
    with open(save_path / "config.yaml", "w") as steam:
        yaml.dump(config, steam)
    if single_process:
        log.info(
            "Only one process specified so using poker_ai.ai.singleprocess."
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
        )
    else:
        log.info(
            "Mulitple processes specifed so using poker_ai.ai.multiprocess."
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
        )
        _safe_search(server)


if __name__ == "__main__":
    train()
