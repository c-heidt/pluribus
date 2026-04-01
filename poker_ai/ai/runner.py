"""Script for using multiprocessing to train agent.

CLI Use
-------

Below you can run `poker_ai train start --help` to see available options::

    Usage: poker_ai train start [OPTIONS]

      Train agent from scratch.

    Options:
      --strategy_interval INTEGER     Update the current strategy whenever the
                                      iteration % strategy_interval == 0.
      --max_runtime_hours FLOAT       Wall-clock budget for this run in hours.
                                      Training stops when elapsed time reaches
                                      this limit.
      --discount_duration_iters INTEGER
                                      Total number of iterations for which the
                                      LCFR discount window is active.
      --prune_threshold INTEGER       When a uniform random number is less than
                                      95%, and the iteration > prune_threshold,
                                      use CFR with pruning.
      --c INTEGER                     Pruning threshold for regret below which
                                      subtrees are not recursively visited
                                      during CFR with pruning.
      --n_players INTEGER             The number of players in the game.
      --dump_iteration INTEGER        Compute and accumulate a new strategy
                                      snapshot every dump_iteration iterations.
      --update_threshold INTEGER      Start updating the strategy after this
                                      many iterations.
      --lut_path TEXT                 Path to the clustering infoset files.
      --pickle_dir                    Whether the LUT files are pickle files
                                      (deprecated).
      --single_process / --multi_process
                                      Either use or don't use multiple
                                      processes.
      --sync_interval INTEGER         Iterations between worker sync barriers.
                                      Higher values keep workers busier but
                                      delay delta merges.
      --discount_interval INTEGER     Apply LCFR discounting every N sync
                                      barriers (i.e. every sync_interval * N
                                      iterations). Default 1 discounts at
                                      every sync.
      --checkpoint_interval INTEGER   Write a training checkpoint every N
                                      iterations.
      --n_processes INTEGER           Number of worker processes to spawn.
                                      Defaults to cpu_count-1 (or
                                      SLURM_CPUS_PER_TASK-1 under Slurm).
      --nickname TEXT                 The nickname of the study.
      --help                          Show this message and exit.
"""
import logging
from pathlib import Path
from typing import Dict

import click
import joblib
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
    "--server_config_path",
    default="./server.gz",
    help="The path to the previous server.gz file from a previous study.",
)
def resume(server_config_path: str):
    """
    Continue training agent from config loaded from file.

    ...

    Parameters
    ----------
    server_config_path : str
        Path to server configurations.
    """
    try:
        config = joblib.load(server_config_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Server config file not found at the path: {server_config_path}\n "
            f"Please set the path to a valid file dumped by a previous session."
        )
    server = Server.from_dict(config)
    _safe_search(server)


@train.command()
@click.option(
    "--strategy_interval",
    default=5000,
    help="Update the current strategy whenever the iteration % strategy_interval == 0.",
)
@click.option(
    "--max_runtime_hours",
    default=1.0,
    type=float,
    help="Wall-clock budget for this run in hours. Training stops when elapsed time reaches this limit.",
)
@click.option(
    "--discount_duration_iters",
    default=10000,
    help="Total number of iterations for which the discount window is active.",
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
        "prune_threshold, use CFR with pruning."
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
    "--dump_iteration",
    default=1000,
    help=(
        "When the iteration % dump_iteration == 0, we will compute a new strategy "
        "and write that to the accumlated strategy, which gets normalised at a "
        "later time."
    ),
)
@click.option(
    "--update_threshold",
    default=1000,
    help=(
        "When the iteration is greater than update_threshold we can start "
        "updating the strategy."
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
        "How many iterations between worker sync barriers. Higher values keep "
        "workers busier but delay delta merges."
    ),
)
@click.option(
    "--discount_interval",
    default=10,
    help=(
        "Apply LCFR discounting every N sync barriers (i.e. every "
        "sync_interval * discount_interval iterations). Default 1 discounts "
        "at every sync."
    ),
)
@click.option(
    "--checkpoint_interval",
    default=10000,
    help=(
        "Write a training checkpoint every N iterations. "
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
    discount_duration_iters: int,
    n_iterations: int,
    prune_threshold: int,
    c: int,
    n_players: int,
    dump_iteration: int,
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
    """Train agent from scratch."""
    # Write config to file, and create directory to save results in.
    config: Dict[str, int] = {**locals()}
    save_path: Path = utils.io.create_dir(nickname)
    with open(save_path / "config.yaml", "w") as steam:
        yaml.dump(config, steam)
    if single_process:
        log.info(
            "Only one process specified so using poker_ai.ai.singleprocess."
            "simple_search for the optimisation."
        )
        # simple_search is the validation baseline — it keeps its own
        # iteration-based parameters and is not affected by Phase 7.
        simple_search(
            config=config,
            save_path=save_path,
            lut_path=lut_path,
            pickle_dir=pickle_dir,
            strategy_interval=strategy_interval,
            n_iterations=n_iterations,
            lcfr_threshold=discount_duration_iters,
            c=c,
            n_players=n_players,
            dump_iteration=dump_iteration,
            update_threshold=update_threshold,
        )
    else:
        log.info(
            "Mulitple processes specifed so using poker_ai.ai.multiprocess."
            "server.Server for the optimisation."
        )
        # Create the server that controls/coordinates the workers.
        server = Server(
            strategy_interval=strategy_interval,
            max_runtime_hours=max_runtime_hours,
            discount_duration_iters=discount_duration_iters,
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
