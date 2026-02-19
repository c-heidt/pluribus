"""
Usage: poker_ai cluster [OPTIONS]

  Run clustering with memory-efficient chunked processing and checkpointing.

Options:
  --low_card_rank INTEGER        The starting hand rank from 2 through 14 for
                                 the deck we want to cluster. We recommend
                                 starting small.
  --high_card_rank INTEGER       The ending hand rank from 2 through 14 for
                                 the deck we want to cluster. We recommend
                                 starting small.
  --n_river_clusters INTEGER     The number of card information buckets we
                                 would like to create for the river. We
                                 recommend to start small.
  --n_turn_clusters INTEGER      The number of card information buckets we
                                 would like to create for the turn. We
                                 recommend to start small.
  --n_flop_clusters INTEGER      The number of card information buckets we
                                 would like to create for the flop. We
                                 recommend to start small.
  --n_simulations_river INTEGER  The number of opponent hand simulations we
                                 would like to run on the river. We recommend
                                 to start small. (Ignored if --method exact)
  --n_simulations_turn INTEGER   The number of river card hand simulations we
                                 would like to run on the turn. We recommend
                                 to start small. (Ignored if --method exact)
  --n_simulations_flop INTEGER   The number of turn card hand simulations we
                                 would like to run on the flop. We recommend
                                 to start small. (Ignored if --method exact)
  --save_dir TEXT                Path to directory to save card info lookup
                                 table, betting stage centroids, and checkpoints.
  --workers INTEGER              Number of worker processes to use for clustering.
                                 Defaults to number of CPUs detected by the system.
  --chunk_size INTEGER           Number of card combinations to process per chunk.
                                 Larger chunks use more memory but may be faster.
                                 Default 10000.
  --use_mini_batch/--no_mini_batch  Use MiniBatchKMeans for large datasets
                                 (>50000 samples). More memory efficient but
                                 slightly less accurate. Default True.
  --method [monte_carlo|exact]   Computation method for hand strength. 'monte_carlo'
                                 uses sampling (faster). 'exact' uses exhaustive
                                 enumeration (slower but precise). Default 'monte_carlo'.
  --help                         Show this message and exit.
"""
import click
from typing import Optional

from poker_ai.clustering.card_info_lut_builder import CardInfoLutBuilder
from poker_ai.clustering.exact_lut_builder import ExactHandStrengthBuilder


@click.command()
@click.option(
    "--low_card_rank",
    default=10,
    help=(
        "The starting hand rank from 2 through 14 for the deck we want to "
        "cluster. We recommend starting small."
    )
)
@click.option(
    "--high_card_rank",
    default=14,
    help=(
        "The starting hand rank from 2 through 14 for the deck we want to "
        "cluster. We recommend starting small."
    )
)
@click.option(
    "--n_river_clusters",
    default=50,
    help=(
        "The number of card information buckets we would like to create for "
        "the river. We recommend to start small."
    )
)
@click.option(
    "--n_turn_clusters",
    default=50,
    help=(
        "The number of card information buckets we would like to create for "
        "the turn. We recommend to start small."
    )
)
@click.option(
    "--n_flop_clusters",
    default=50,
    help=(
        "The number of card information buckets we would like to create for "
        "the flop. We recommend to start small."
    )
)
@click.option(
    "--n_simulations_river",
    default=6,
    help=(
        "The number of opponent hand simulations we would like to run on the "
        "river. We recommend to start small."
    )
)
@click.option(
    "--n_simulations_turn",
    default=6,
    help=(
        "The number of river card hand simulations we would like to run on the "
        "turn. We recommend to start small."
    )
)
@click.option(
    "--n_simulations_flop",
    default=6,
    help=(
        "The number of turn card hand simulations we would like to run on the "
        "flop. We recommend to start small."
    )
)
@click.option(
    "--save_dir",
    default="",
    help=(
        "Path to directory to save card info lookup table and betting stage "
        "centroids."
    )
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
def cluster(
    low_card_rank: int,
    high_card_rank: int,
    n_river_clusters: int,
    n_turn_clusters: int,
    n_flop_clusters: int,
    n_simulations_river: int,
    n_simulations_turn: int,
    n_simulations_flop: int,
    save_dir: str,
    workers: Optional[int],
    chunk_size: int,
    use_mini_batch: bool,
    method: str,
):
    """Run clustering with memory-efficient chunked processing and checkpointing."""
    
    # Choose builder based on method
    if method.lower() == "exact":
        # Exact computation doesn't use simulation parameters
        builder = ExactHandStrengthBuilder(
            low_card_rank=low_card_rank,
            high_card_rank=high_card_rank,
            save_dir=save_dir,
            workers=workers,
            chunk_size=chunk_size,
            use_mini_batch=use_mini_batch,
        )
    else:
        # Monte Carlo computation (default)
        builder = CardInfoLutBuilder(
            n_simulations_river,
            n_simulations_turn,
            n_simulations_flop,
            low_card_rank,
            high_card_rank,
            save_dir,
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
    cluster()
