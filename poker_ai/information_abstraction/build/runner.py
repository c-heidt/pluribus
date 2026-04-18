"""CLI entry point ``poker_ai build-abstraction``.

Builds the card-information abstraction with chunked processing and
checkpointing.  Flags match the pre-refactor ``poker_ai cluster`` command
one-for-one so existing scripts only need the command-name swap.
"""
from typing import Optional

import click

from poker_ai.information_abstraction.build.builder import AbstractionBuilder


@click.command(name="build-abstraction")
@click.option(
    "--low_card_rank",
    default=10,
    help=(
        "The starting hand rank from 2 through 14 for the deck we want to "
        "cluster. We recommend starting small."
    ),
)
@click.option(
    "--high_card_rank",
    default=14,
    help=(
        "The ending hand rank from 2 through 14 for the deck we want to "
        "cluster. We recommend starting small."
    ),
)
@click.option(
    "--n_river_clusters",
    default=50,
    help=(
        "The number of card information buckets we would like to create for "
        "the river. We recommend to start small."
    ),
)
@click.option(
    "--n_turn_clusters",
    default=50,
    help=(
        "The number of card information buckets we would like to create for "
        "the turn. We recommend to start small."
    ),
)
@click.option(
    "--n_flop_clusters",
    default=50,
    help=(
        "The number of card information buckets we would like to create for "
        "the flop. We recommend to start small."
    ),
)
@click.option(
    "--n_simulations_river",
    default=6,
    help=(
        "The number of opponent hand simulations we would like to run on the "
        "river. We recommend to start small. (Ignored if --method exact)"
    ),
)
@click.option(
    "--save_dir",
    default="",
    help=(
        "Path to directory to save card info lookup table and betting stage "
        "centroids."
    ),
)
@click.option(
    "--workers",
    default=None,
    type=int,
    help=(
        "Number of worker processes to use for clustering. "
        "Defaults to number of CPUs detected by the system."
    ),
)
@click.option(
    "--chunk_size",
    default=10000,
    type=int,
    help=(
        "Number of card combinations to process per chunk. Larger chunks use "
        "more memory but may be faster. Default 10000."
    ),
)
@click.option(
    "--use_mini_batch/--no_mini_batch",
    default=True,
    help=(
        "Use MiniBatchKMeans for large datasets (>50000 samples). "
        "More memory efficient but slightly less accurate. Default True."
    ),
)
@click.option(
    "--method",
    default="monte_carlo",
    type=click.Choice(["monte_carlo", "exact"], case_sensitive=False),
    help=(
        "Computation method for hand strength. 'monte_carlo' uses sampling "
        "(faster, configurable via n_simulations_*). 'exact' uses exhaustive "
        "enumeration (slower but precise, ignores n_simulations_* options). "
        "Default 'monte_carlo'."
    ),
)
def build_abstraction(
    low_card_rank: int,
    high_card_rank: int,
    n_river_clusters: int,
    n_turn_clusters: int,
    n_flop_clusters: int,
    n_simulations_river: int,
    save_dir: str,
    workers: Optional[int],
    chunk_size: int,
    use_mini_batch: bool,
    method: str,
):
    """Build the card-information abstraction with checkpointing."""
    builder = AbstractionBuilder(
        method=method,
        n_simulations_river=n_simulations_river,
        low_card_rank=low_card_rank,
        high_card_rank=high_card_rank,
        save_dir=save_dir,
        workers=workers,
        chunk_size=chunk_size,
        use_mini_batch=use_mini_batch,
    )
    builder.compute(
        n_river_clusters,
        n_turn_clusters,
        n_flop_clusters,
    )


if __name__ == "__main__":
    build_abstraction()
