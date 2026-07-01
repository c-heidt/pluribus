import click

from poker_ai.blueprint.runner import train
from information_abstraction.build.runner import build_abstraction
from poker_ai.terminal.runner import run_terminal_app
from evaluation.runner import evaluate


@click.group()
def cli():
    """The CLI for the poker_ai package that groups the various scripts.

    The root command will allow you to do the following. The "train" option
    builds a model and manages the search for the offline strategy. The "play"
    option allows you to play against the strategy you have trained. The
    "build-abstraction" option runs the card-information abstraction build
    required as a pre-requisite for training.
    """
    pass


cli.add_command(train, name="train")
cli.add_command(build_abstraction, name="build-abstraction")
cli.add_command(run_terminal_app, name="play")
cli.add_command(evaluate, name="evaluate")
