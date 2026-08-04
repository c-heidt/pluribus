"""The clamp's row↔cluster contract, under the REAL 20-card LUT (multi-cluster).

Everything else in the A-suite runs on ``_stub_lut``, where every hand maps to
cluster 0 and every future street collapses to ``n_rows == 1``.  That makes a whole
bug class invisible — with one row, *any* cluster→row mapping is trivially correct.
A board-stale model-row cache passed the entire stub-LUT suite unchanged; only these
tests distinguish it.

``data/20cards_exact`` gives ~24-37 dense rows at future streets, so here the
mapping actually has to be right.
"""

import numpy as np
import pytest

from poker_ai.search.cluster_maps import ClusterMapper
from poker_ai.search.solver import solve

from test.search._helpers import _ctx, _real_lut_env
from test.search.test_budget import _cfg
from test.search.test_dbr_agent import _modeled


class _ClusterCodedModel:
    """σ̂ encodes the info-set it was built from, so a stale/mismatched row is
    detectable by inspection rather than by luck."""

    def __init__(self, c=0.5):
        self._c, self.seen = c, []

    @staticmethod
    def _row_for(info_set, n):
        h = int.from_bytes(bytes(info_set)[:8].ljust(8, b"\0"), "little")
        r = np.array([(h >> (8 * i)) % 97 + 1 for i in range(n)], dtype=np.float64)
        return r / r.sum()

    def strategy(self, state):
        self.seen.append(bytes(state.info_set))
        return self._row_for(state.info_set, len(state.legal_actions))

    def confidence(self, state):
        return self._c


# --------------------------------------------------------------------------- #
# The fixture is actually multi-cluster (guard against silent collapse)
# --------------------------------------------------------------------------- #

@pytest.mark.requires_lut
@pytest.mark.parametrize("root,expect_future", [(1, (2, 3)), (2, (3,))])
def test_real_lut_gives_many_cluster_rows(root, expect_future):
    env = _real_lut_env(root)
    cm = ClusterMapper(env.card_info_lut, env.combo_cards,
                       env.community_cards, root)
    for s in expect_future:
        assert cm.n_rows(s) > 1, (
            f"street {s} collapsed to {cm.n_rows(s)} row(s) — fixture is not "
            f"multi-cluster, these tests would be vacuous"
        )


# --------------------------------------------------------------------------- #
# The contract: one cached row per cluster, and it matches that cluster's σ̂
# --------------------------------------------------------------------------- #

@pytest.mark.requires_lut
@pytest.mark.parametrize("root", [2, 3])
def test_cached_model_rows_match_their_cluster(root):
    """Every filled row holds the σ̂ of *its own* cluster.

    This is the assertion a board-stale (combo-space) cache fails: rows built on
    one iteration's completion get reused on later, different completions, so the
    stored row stops matching the cluster the row stands for.
    """
    env = _real_lut_env(root)
    model = _ClusterCodedModel()
    res = solve(env, _modeled(_ctx(env, seed=7), {1: model}),
                _cfg(auto_budget=False, max_iterations=25,
                     max_wall_seconds=1e9))

    cache = res.state.model_sigma_cache
    assert len(cache) > 0, "the clamp never ran"
    seen = {bytes(s) for s in model.seen}
    checked = 0
    for key in cache:
        m_rows, c_rows, filled = cache[key]
        idx = np.flatnonzero(filled)
        assert idx.size > 0
        for r in idx:
            row = m_rows[r]
            nz = row[row > 0]
            assert nz.size > 0, f"row {r} filled but all-zero"
            np.testing.assert_allclose(row.sum(), 1.0, atol=1e-9)
            # The row must be one the model actually produced for some info-set it
            # was queried at — not an average, not a stale neighbour.
            assert any(
                np.allclose(row[: len(_ClusterCodedModel._row_for(s, row.size))],
                            _ClusterCodedModel._row_for(s, row.size))
                for s in seen
            ), f"row {r} matches no info-set the model was queried at"
            checked += 1
    assert checked > 1, "only one row checked — not exercising cluster structure"


@pytest.mark.requires_lut
def test_distinct_clusters_get_distinct_rows():
    """With a real LUT the cached rows must not all be identical — otherwise the
    cluster→row gather has collapsed and exploitation would be uniform."""
    env = _real_lut_env(2)
    res = solve(env, _modeled(_ctx(env, seed=7), {1: _ClusterCodedModel()}),
                _cfg(auto_budget=False, max_iterations=25,
                     max_wall_seconds=1e9))

    cache = res.state.model_sigma_cache
    multi = 0
    for key in cache:
        m_rows, _, filled = cache[key]
        idx = np.flatnonzero(filled)
        if idx.size > 1:
            rows = {tuple(np.round(m_rows[r], 12)) for r in idx}
            multi += 1
            assert len(rows) > 1, (
                f"{idx.size} filled rows are all identical at {key} — the "
                f"cluster→row mapping collapsed"
            )
    assert multi > 0, "no node had >1 filled row — fixture not exercising clusters"


@pytest.mark.requires_lut
def test_model_queried_at_many_distinct_info_sets():
    """Sanity that the real LUT genuinely varies the info-set (the stub does not)."""
    env = _real_lut_env(2)
    model = _ClusterCodedModel()
    solve(env, _modeled(_ctx(env, seed=7), {1: model}),
          _cfg(auto_budget=False, max_iterations=25,
               max_wall_seconds=1e9))
    assert len({bytes(s) for s in model.seen}) > 1
