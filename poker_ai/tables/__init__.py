"""Sparse LMDB-backed regret/strategy table stack.

The `blueprint` sibling package drives CFR traversals against this storage.
"""
from . import checkpoint
from . import chunk_store
from . import chunked_table
from . import cfr_tables
from . import index
