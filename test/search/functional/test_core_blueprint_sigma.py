"""Phase-4c gate: the in-core ``CoreTables.blueprint_sigma`` vs the Python
``BlueprintPolicy.strategy``.

4c moves the per-decision blueprint policy read into the compiled core (the ~16%
Python callback of a realistic-blueprint MCCFR iteration).  The rollout is already
*equilibrium-gated* (the board draw differs from Python) and the resulting sigma only
feeds ``sample_index``, so the acceptance bar is **tolerance-equivalence** (<1e-6),
not byte-identity: the regret-match fallback reuses the proven byte-identical
``_sigma``; only the average-strategy normalise / bias reweight / canonical->legal
remap are float64-accumulate -> float32.

This gate is deliberately **LUT-free** — it writes synthetic strategy + regret rows at
arbitrary info-set byte keys via ``ChunkedTable.merge_delta_row`` (no card-info LUT, no
pre-train), so it runs anywhere the core is built.  It sweeps the full behaviour matrix
that ``BlueprintPolicy.strategy`` distinguishes:

* average-strategy HIT (row mass over the legal cols >= ``min_strategy_mass``),
* average-strategy UNDER-MASS -> regret-match fallback,
* index MISS -> uniform over legal,
* all four §4 bias classes (none/fold/call/raise) x multiplier 1.0 and 5.0,
* ``min_strategy_mass`` boundary (just-below / just-above),
* ``legal`` a strict subset of the canonical set WITH a bias (the case most likely to
  expose the bias-renormalise-over-full-canonical ordering).

Skips when the compiled core is not built.
"""

import numpy as np
import pytest

from poker_ai import _core

pytestmark = pytest.mark.skipif(
    not _core.CORE_AVAILABLE, reason="compiled core extension not built"
)

if _core.CORE_AVAILABLE:
    from environment.action_space import (
        ACTION_TO_IDX,
        CANONICAL_ACTIONS,
        MAX_ACTIONS_PER_STREET,
    )
    from environment.poker_env import PolicyState
    from poker_ai._core import _traverse as cyt
    from poker_ai.search.policy import BlueprintPolicy
    from poker_ai.tables.cfr_tables import CFRTables
    from poker_ai.tables.index import lmdb_map_size_for_players

    _BIAS_CODE = {"none": 0, "fold": 1, "call": 2, "raise": 3}
    _CAPS = {r: 1 << 14 for r in range(4)}


def _tables(tmp_path):
    shm = tmp_path / "shm"
    shm.mkdir(parents=True, exist_ok=True)
    return CFRTables(
        index_path=tmp_path / "lmdb_index",
        shm_dir=str(shm),
        actions_per_street=MAX_ACTIONS_PER_STREET,
        lmdb_map_size=lmdb_map_size_for_players(2),
        enable_index_cache=True,
        index_capacities=_CAPS,
    )


def _core_tables(tables):
    return cyt.CoreTables(
        tables, CANONICAL_ACTIONS, ACTION_TO_IDX, MAX_ACTIONS_PER_STREET
    )


def _policy_state(r, key, legal):
    legal_set = set(legal)
    valid_mask = np.array(
        [a in legal_set for a in CANONICAL_ACTIONS[r]], dtype=bool
    )
    return PolicyState(
        player_i=0, betting_round=r, info_set=key,
        valid_mask=valid_mask, legal_actions=tuple(legal),
    )


def _core_sigma(ct, r, key, legal, bias, mult, min_mass):
    legal_cols = np.array([ACTION_TO_IDX[r][a] for a in legal], dtype=np.int64)
    return np.asarray(
        ct.blueprint_sigma(r, key, legal_cols, _BIAS_CODE[bias], float(mult), int(min_mass))
    )


def _assert_match(ct, tables, r, key, legal, bias, mult, min_mass, label):
    py = BlueprintPolicy(
        tables, bias_multiplier=mult, min_strategy_mass=min_mass
    ).strategy(_policy_state(r, key, legal), bias=bias)
    core = _core_sigma(ct, r, key, legal, bias, mult, min_mass)
    assert core.shape == py.shape, f"{label}: shape {core.shape} != {py.shape}"
    np.testing.assert_allclose(
        core, py, atol=1e-6,
        err_msg=f"{label}: core={np.round(core,5)} py={np.round(py,5)} "
                f"(r={r} legal={legal} bias={bias} mult={mult} min={min_mass})",
    )
    # A well-formed non-empty node is a distribution.
    assert abs(float(core.sum()) - 1.0) < 1e-5


@pytest.mark.parametrize("r", [0, 1, 2, 3])
def test_blueprint_sigma_matches_python_randomized(tmp_path, r):
    """Full randomized sweep on street ``r``: 3 row regimes x random legal subsets
    x 4 biases x {mult 1, 5}, all within 1e-6 of ``BlueprintPolicy.strategy``."""
    w = MAX_ACTIONS_PER_STREET[r]
    canonical = list(CANONICAL_ACTIONS[r])
    rng = np.random.RandomState(1000 + r)
    tables = _tables(tmp_path)
    try:
        # Three key regimes.  Writing a strategy row allocates the shared index
        # entry, so both stores resolve a flat for it (regret defaults to the
        # zero mmap row unless also written).
        keys = {}
        # (a) average-strategy HIT: big visit counts.
        for i in range(6):
            k = b"avg-%d" % i
            tables.strategy[r].merge_delta_row(
                k, rng.randint(1, 400, size=w).astype(np.int32))
            tables.regret[r].merge_delta_row(
                k, rng.randint(-300, 300, size=w).astype(np.int32))
            keys[k] = "avg"
        # (b) under-mass strategy (tiny counts) -> regret-match fallback.
        for i in range(6):
            k = b"reg-%d" % i
            tables.strategy[r].merge_delta_row(
                k, rng.randint(0, 2, size=w).astype(np.int32))
            tables.regret[r].merge_delta_row(
                k, rng.randint(-500, 800, size=w).astype(np.int32))
            keys[k] = "reg"
        # (c) MISS: never written -> uniform over legal.
        for i in range(4):
            keys[b"miss-%d" % i] = "miss"

        tables.prewarm_caches()
        ct = _core_tables(tables)

        for k, regime in keys.items():
            for _ in range(6):
                m = rng.randint(1, w + 1)
                legal = list(rng.choice(canonical, size=m, replace=False))
                for bias in ("none", "fold", "call", "raise"):
                    for mult in (1.0, 5.0):
                        _assert_match(
                            ct, tables, r, k, legal, bias, mult, 10,
                            f"{regime}/{k!r}")
    finally:
        tables.close()


def test_min_strategy_mass_boundary(tmp_path):
    """The mass gate (avg-strategy vs regret fallback) flips exactly as Python's
    ``total < min_strategy_mass`` does, on both sides of the threshold."""
    r = 1
    w = MAX_ACTIONS_PER_STREET[r]
    canonical = list(CANONICAL_ACTIONS[r])
    tables = _tables(tmp_path)
    try:
        # Row whose sum over ALL actions is 100; a legal subset selects part of it.
        row = np.zeros(w, dtype=np.int32)
        row[0] = 40
        row[1] = 60
        k = b"boundary"
        tables.strategy[r].merge_delta_row(k, row)
        tables.regret[r].merge_delta_row(
            k, np.array([7] * w, dtype=np.int32))  # positive -> non-uniform fallback
        tables.prewarm_caches()
        ct = _core_tables(tables)

        legal = [canonical[0], canonical[1]]        # legal-col mass = 100
        # min just below and just above the legal-col mass -> avg vs fallback.
        for min_mass in (99, 100, 101, 150):
            for bias in ("none", "raise"):
                _assert_match(ct, tables, r, k, legal, bias, 5.0, min_mass,
                              f"boundary(min={min_mass})")
        # A legal subset that excludes the mass (only zero-count actions) -> the
        # masked sum is 0 < any positive min -> regret-match fallback.
        if w > 2:
            legal2 = [canonical[2]] + ([canonical[3]] if w > 3 else [])
            for min_mass in (1, 10):
                _assert_match(ct, tables, r, k, legal2, "call", 5.0, min_mass,
                              "boundary/zero-mass-subset")
    finally:
        tables.close()


def test_strategy_probe_row_matches_get_row_if_exists(tmp_path):
    """Row-reader isolation (analog of ``probe_row`` for regret): the in-core
    strategy read equals ``ChunkedTable.get_row_if_exists`` byte-for-byte on hits,
    and returns ``None`` on a genuine index miss."""
    r = 2
    w = MAX_ACTIONS_PER_STREET[r]
    tables = _tables(tmp_path)
    try:
        written = {}
        for i in range(8):
            k = b"srow-%d" % i
            row = np.random.RandomState(i).randint(0, 5000, size=w).astype(np.int32)
            tables.strategy[r].merge_delta_row(k, row)
            written[k] = row
        tables.prewarm_caches()
        ct = _core_tables(tables)
        for k, row in written.items():
            core = ct.strategy_probe_row(r, k)
            ref = tables.strategy[r].get_row_if_exists(k)
            assert core is not None and ref is not None
            assert np.array_equal(np.asarray(core), np.asarray(ref)), k
        for k in (b"never-1", b"never-2"):
            assert ct.strategy_probe_row(r, k) is None
            assert tables.strategy[r].get_row_if_exists(k) is None
    finally:
        tables.close()


def test_single_legal_action_is_certain(tmp_path):
    """A lone legal action gets probability 1 regardless of row/bias — matches
    Python and guards the final gather renormalise."""
    r = 0
    w = MAX_ACTIONS_PER_STREET[r]
    tables = _tables(tmp_path)
    try:
        k = b"solo"
        tables.strategy[r].merge_delta_row(
            k, np.arange(1, w + 1, dtype=np.int32) * 10)
        tables.prewarm_caches()
        ct = _core_tables(tables)
        for a in CANONICAL_ACTIONS[r]:
            for bias in ("none", "fold", "raise"):
                core = _core_sigma(ct, r, k, [a], bias, 5.0, 10)
                assert core.shape == (1,)
                np.testing.assert_allclose(core, [1.0], atol=1e-6)
    finally:
        tables.close()
