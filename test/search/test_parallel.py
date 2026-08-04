"""Fork-safety + worker-count helpers used by the per-hand pool (§6.7).

Within-search parallelism (W replicas of a single search, merged once) was retired —
production runs one hand per core with a serial search — so the replica-merge / planner /
``solve(workers>1)`` machinery is gone.  What remains and is exercised here:

- :func:`_reopen_leaf_fleet_lmdb` — reopens each unique LMDB-backed leaf/model policy
  once after a ``fork`` (MDB_BAD_RSLOT repair), skips policies without the hook;
- :func:`resolve_workers` — resolves a ``None`` worker count to a cpu-based default.
"""

import numpy as np
import pytest

from poker_ai.search.mccfr import _BIAS_CLASSES
from poker_ai.search.policy import Policy
from poker_ai.search.parallel import _reopen_leaf_fleet_lmdb, resolve_workers


class UniformPolicy(Policy):
    def strategy(self, state, bias="none"):
        n = len(state.legal_actions)
        return np.full(n, 1.0 / n, dtype=np.float32) if n else np.array([], np.float32)


def _policies():
    return {c: UniformPolicy() for c in _BIAS_CLASSES}


# --------------------------------------------------------------------------- #
# Fork-safety: reopen LMDB-backed leaf policies in each worker (MDB_BAD_RSLOT)
# --------------------------------------------------------------------------- #

class _RecordingPolicy(Policy):
    """A leaf policy exposing ``reopen_after_fork`` (stands in for a blueprint)."""

    def __init__(self):
        self.reopens = 0

    def strategy(self, state, bias="none"):
        n = len(state.legal_actions)
        return np.full(n, 1.0 / n, dtype=np.float32) if n else np.array([], np.float32)

    def reopen_after_fork(self):
        self.reopens += 1


class TestReopenForkedLmdb:
    """`_reopen_leaf_fleet_lmdb` reopens each unique LMDB policy once, skips the rest."""

    def _ctx(self, policies):
        from types import SimpleNamespace
        return SimpleNamespace(leaf=SimpleNamespace(policies=policies))

    def test_shared_blueprint_reopened_once(self):
        # The four §4 bias variants share ONE blueprint object → reopen exactly once.
        bp = _RecordingPolicy()
        _reopen_leaf_fleet_lmdb(self._ctx({c: bp for c in _BIAS_CLASSES}))
        assert bp.reopens == 1

    def test_distinct_policies_each_reopened(self):
        a, b = _RecordingPolicy(), _RecordingPolicy()
        _reopen_leaf_fleet_lmdb(self._ctx({"none": a, "fold": b}))
        assert a.reopens == 1 and b.reopens == 1

    def test_policies_without_reopen_are_skipped(self):
        # A mix: the in-memory UniformPolicy has no reopen hook → skipped, no error.
        bp = _RecordingPolicy()
        _reopen_leaf_fleet_lmdb(self._ctx({"none": UniformPolicy(), "fold": bp}))
        assert bp.reopens == 1

    def test_no_reopenable_policies_is_a_noop(self):
        _reopen_leaf_fleet_lmdb(self._ctx(_policies()))   # all UniformPolicy → no crash


# --------------------------------------------------------------------------- #
# Worker-count resolution (used by the per-hand pool + calibration sweep)
# --------------------------------------------------------------------------- #

class TestResolveWorkers:

    @pytest.mark.parametrize("val,expected", [(1, 1), (4, 4), (0, 1), (-2, 1)])
    def test_resolve_explicit(self, val, expected):
        assert resolve_workers(val) == expected

    def test_resolve_none_is_cpu_based(self):
        assert resolve_workers(None) >= 1
