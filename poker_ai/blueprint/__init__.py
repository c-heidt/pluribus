"""Blueprint computation: CFR traversal, training schedule, CLI.

The `tables` sibling package owns the sparse LMDB-backed regret/strategy
storage consumed by this package.  The environment owns the canonical action
list exposed via :mod:`poker_ai.environment.action_space`.
"""
from . import cfr
from . import multiprocess
from . import runner
from . import singleprocess
from . import strategy
from . import training
from . import tree_utils
