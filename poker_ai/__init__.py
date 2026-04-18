from __future__ import annotations

import logging

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    format=FORMAT, datefmt="[%X] ", handlers=[RichHandler()], level=logging.INFO,
)

from . import blueprint
from . import cli
from . import environment
from . import information_abstraction
from . import tables
from . import terminal

__version__ = "1.0.0rc3"
