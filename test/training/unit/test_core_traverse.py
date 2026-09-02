"""Phase-3 in-core CFR traversal — byte-exact against the pure-Python reference.

Three gates, in ascending scope:

* **3a — row reader.**  ``CoreTables.probe_row`` (pure-shm probe + chunk read)
  must equal ``ChunkedTable.get_row_if_exists`` for every info set, hits *and*
  misses.  This is the highest silent-divergence risk (the inverted high/low
  hash word, the ``divmod(flat, CHUNK_SIZE)`` chunk indexing) — a wrong port
  reads "uniform everywhere" with no crash, so it is gated on its own.

* **3b — traverse (the magnitude win).**  Record an opponent sequence with the
  Python ``_traverse`` (:mod:`test.training.core_diff`), replay it into the
  compiled core, and assert the two ``local_delta`` dicts are **byte-identical**
  (``assert_local_delta_equal``).  This is RNG-free and end-to-end exact — the
  strongest correctness gate.  A truncation test proves it is non-vacuous.

* **rng sampler — distributional.**  The production opponent sampler is not
  certified by RNG byte-parity with numpy (deliberately), so it is checked
  distributionally: many draws match ``sigma`` restricted to the legal actions.

Uses cache-enabled tables (``requires_lut``): the in-core read path is pure-shm,
so a shm index cache is mandatory.  Capacities are pinned small so the test's
``/dev/shm`` footprint stays tiny.
"""

import random
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

import poker_ai.blueprint.cfr as cfr_mod
from environment.action_space import (
    ACTION_TO_IDX,
    CANONICAL_ACTIONS,
    MAX_ACTIONS_PER_STREET,
)
from environment.poker_env import (
    MAX_RAISES_PER_ROUND,
    RAISE_SIZES_BY_STAGE,
    _ACTION_BYTE,
    _STAGE_ID,
    new_game,
)
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.tables.index import lmdb_map_size_for_players
from poker_ai._core import _state as cy
from poker_ai._core import _traverse as cyt
from test.training.core_diff import (
    RecordingSampler,
    ReplaySampler,
    assert_local_delta_equal,
    run_recording,
    run_replay,
)

# Dump the encoding alphabet + raise grid into the compiled state engine once.
cy.configure(
    _STAGE_ID, _ACTION_BYTE, RAISE_SIZES_BY_STAGE, MAX_RAISES_PER_ROUND
)

N_PLAYERS = 2
_T = 100  # arbitrary iteration index (unweighted increment; unused by the core)
# Power-of-two headroom well above the few-thousand rows a 40-iter pre-train
# allocates, but tiny in /dev/shm (64k slots * 24 B ~ 1.5 MiB per street).
_CAPS = {r: 1 << 16 for r in range(4)}


@pytest.fixture
def cached_tables(tmp_path, lut):
    """Cache-enabled CFR tables, lightly pre-trained for non-uniform regrets.

    The seeded pre-train both populates realistic regrets and (with the cache
    enabled) mirrors every allocated row into the shm index cache, so the core's
    pure-shm reads resolve exactly what ``get_row_if_exists`` resolves.
    """
    shm = tmp_path / "shm"
    shm.mkdir(parents=True, exist_ok=True)
    tables = CFRTables(
        index_path=tmp_path / "lmdb_index",
        shm_dir=str(shm),
        actions_per_street=MAX_ACTIONS_PER_STREET,
        lmdb_map_size=lmdb_map_size_for_players(N_PLAYERS),
        enable_index_cache=True,
        index_capacities=_CAPS,
    )
    np.random.seed(123)
    for t in range(1, 40):
        for i in range(N_PLAYERS):
            cfr_mod.cfr(tables, new_game(N_PLAYERS, lut), i, t)
    tables.prewarm_caches()
    yield tables
    tables.close()


def _core_tables(tables):
    return cyt.CoreTables(
        tables, CANONICAL_ACTIONS, ACTION_TO_IDX, MAX_ACTIONS_PER_STREET
    )


def _build_cached_tables(tmp_path, lut, n_players, n_iters=30, seed=123):
    """Cache-enabled, lightly pre-trained tables for ``n_players`` players."""
    shm = tmp_path / "shm"
    shm.mkdir(parents=True, exist_ok=True)
    tables = CFRTables(
        index_path=tmp_path / "lmdb_index",
        shm_dir=str(shm),
        actions_per_street=MAX_ACTIONS_PER_STREET,
        lmdb_map_size=lmdb_map_size_for_players(n_players),
        enable_index_cache=True,
        index_capacities=_CAPS,
    )
    np.random.seed(seed)
    for t in range(1, n_iters + 1):
        for i in range(n_players):
            cfr_mod.cfr(tables, new_game(n_players, lut), i, t)
    tables.prewarm_caches()
    return tables


def _record_cfrp(tables, state, i, t, c, seed):
    """Record a CFR-P (pruned) traversal's opponent choices + local_delta."""
    sampler = RecordingSampler(seed=seed)
    ld = {}
    orig = cfr_mod.sample_action
    cfr_mod.sample_action = sampler
    try:
        cfr_mod.cfrp(tables, deepcopy(state), i, t, c, local_delta=ld)
    finally:
        cfr_mod.sample_action = orig
    return ld, sampler.choices


def _replay_cfrp(tables, state, i, t, c, choices):
    """Replay a recorded sequence into the Python CFR-P traversal."""
    sampler = ReplaySampler(choices)
    ld = {}
    orig = cfr_mod.sample_action
    cfr_mod.sample_action = sampler
    try:
        cfr_mod.cfrp(tables, deepcopy(state), i, t, c, local_delta=ld)
    finally:
        cfr_mod.sample_action = orig
    assert sampler.exhausted()
    return ld


def _collect_infosets(lut, n_hands=40, seed=0):
    """Random-walk several hands, collecting ``(betting_round, info_set)`` at
    each decision node — a corpus mixing allocated (hit) and unseen (miss) keys.
    """
    rng = random.Random(seed)
    out = []
    for h in range(n_hands):
        np.random.seed(seed * 1000 + h)
        env = new_game(N_PLAYERS, lut)
        steps = 0
        while not env.is_terminal and steps < 200:
            out.append((env.betting_round, env.info_set))
            legal = [a for a in env.legal_actions if a is not None]
            if not legal:
                break
            env.step_in_place(legal[rng.randrange(len(legal))])
            steps += 1
    return out


@pytest.mark.requires_lut
class TestRowReader:
    def test_probe_row_matches_get_row_if_exists(self, cached_tables, lut):
        """Core row read == ChunkedTable.get_row_if_exists on hits and misses."""
        ct = _core_tables(cached_tables)
        hits = misses = 0
        for r, iset in _collect_infosets(lut):
            ref = cached_tables.regret[r].get_row_if_exists(iset)
            got = ct.probe_row(r, iset)
            if ref is None:
                assert got is None
                misses += 1
            else:
                assert got is not None
                assert np.array_equal(np.asarray(ref), np.asarray(got)), (
                    f"row mismatch on street {r} for {iset!r}"
                )
                hits += 1
        assert hits > 0, "corpus exercised no allocated rows"

    def test_synthetic_keys_miss(self, cached_tables, lut):
        """Never-allocated keys resolve to None (unseen → uniform), no crash."""
        ct = _core_tables(cached_tables)
        for k in range(64):
            fake = b"\xfe_core_phase3_miss_" + bytes([k])
            for r in range(4):
                assert ct.probe_row(r, fake) is None


@pytest.mark.requires_lut
class TestCoreTraverseByteIdentical:
    def test_local_delta_byte_identical(self, cached_tables, lut):
        """Core replay reproduces the Python traversal's local_delta exactly."""
        ct = _core_tables(cached_tables)
        total_opponent_choices = 0
        compared = 0
        for deal_seed in range(12):
            np.random.seed(3000 + deal_seed)
            state = new_game(N_PLAYERS, lut)
            for i in range(N_PLAYERS):
                delta_py, choices = run_recording(
                    cached_tables, state, i, _T, seed=deal_seed
                )
                # Cross-check the Python replay reproduces the recording too
                # (guards the harness itself), then gate the core against it.
                delta_py_replay = run_replay(cached_tables, state, i, _T, choices)
                assert_local_delta_equal(delta_py, delta_py_replay)

                fast_state = cy.FastState.from_poker_env(state)
                delta_core = cyt.traverse_replay(ct, fast_state, i, _T, choices)
                assert_local_delta_equal(delta_py, delta_core)

                total_opponent_choices += len(choices)
                compared += 1
        assert compared > 0
        # The gate must actually exercise opponent (external-sampling) nodes.
        assert total_opponent_choices > 0

    def test_local_delta_byte_identical_cfrp_prune(self, cached_tables, lut):
        """CFR-P: core traversal with a prune threshold matches Python cfrp.

        Uses ``c = 0`` so on the lightly-trained tables many non-positive-regret
        off-river actions are actually pruned (river is always explored in full)
        — otherwise the prune branch is never taken and the gate is vacuous.
        Record and replay both use the same predicate so the pruned tree, its
        opponent-node count, and its local_delta all line up.
        """
        ct = _core_tables(cached_tables)
        c = 0
        compared = 0
        for deal_seed in range(12):
            np.random.seed(8000 + deal_seed)
            state = new_game(N_PLAYERS, lut)
            for i in range(N_PLAYERS):
                delta_py, choices = _record_cfrp(
                    cached_tables, state, i, _T, c, seed=deal_seed
                )
                delta_py_replay = _replay_cfrp(
                    cached_tables, state, i, _T, c, choices
                )
                assert_local_delta_equal(delta_py, delta_py_replay)

                fast_state = cy.FastState.from_poker_env(state)
                delta_core = cyt.traverse_replay(
                    ct, fast_state, i, _T, choices, prune=c
                )
                assert_local_delta_equal(delta_py, delta_core)
                compared += 1
        assert compared > 0

    def test_prune_actually_prunes(self, cached_tables, lut):
        """Guard non-vacuity: with c=0 the prune skip branch is really taken.

        Record a CFR-P (c=0) sequence, then replay the *same* choices into the
        core with **no** prune.  The unpruned walk explores every my-node action
        (a superset of the pruned walk), so it reaches strictly more opponent
        nodes and out-runs the recorded sequence → ``replay exhausted``.  That
        raise can only occur if c=0 removed at least one opponent-bearing subtree
        — i.e. the prune branch fired.  (A pruned action that opened no opponent
        node leaves the counts equal and no raise; we scan seeds until one fires.)
        """
        ct = _core_tables(cached_tables)
        saw_pruning = False
        for deal_seed in range(24):
            np.random.seed(8500 + deal_seed)
            state = new_game(N_PLAYERS, lut)
            for i in range(N_PLAYERS):
                _, choices = _record_cfrp(
                    cached_tables, state, i, _T, 0, seed=deal_seed
                )
                # Pruned replay consumes exactly the recorded choices (sanity).
                cyt.traverse_replay(
                    ct, cy.FastState.from_poker_env(state), i, _T,
                    choices, prune=0,
                )
                # Same choices, no prune → unpruned superset out-runs them.
                try:
                    cyt.traverse_replay(
                        ct, cy.FastState.from_poker_env(state), i, _T,
                        choices, prune=None,
                    )
                except AssertionError:
                    saw_pruning = True
                    break
            if saw_pruning:
                break
        assert saw_pruning, "c=0 pruned nothing detectable — prune gate vacuous"

    def test_truncated_replay_raises(self, cached_tables, lut):
        """A short replay into the core raises (non-vacuity), like the harness."""
        ct = _core_tables(cached_tables)
        found = False
        for deal_seed in range(20):
            np.random.seed(6000 + deal_seed)
            state = new_game(N_PLAYERS, lut)
            for i in range(N_PLAYERS):
                _, choices = run_recording(
                    cached_tables, state, i, _T, seed=deal_seed
                )
                if len(choices) >= 1:
                    found = True
                    fast_state = cy.FastState.from_poker_env(state)
                    with pytest.raises(AssertionError):
                        cyt.traverse_replay(
                            ct, fast_state, i, _T, choices[:-1]
                        )
                    break
            if found:
                break
        assert found, "no hand with an opponent node found to test truncation"

    def test_rng_path_runs_and_shapes(self, cached_tables, lut):
        """The production rng path returns a well-formed local_delta.

        Byte-exactness of the rng path is not claimed (RNG parity is not
        pursued); this only guards structure — keys are ``(round, bytes)`` and
        rows are int64 of the street's width.
        """
        ct = _core_tables(cached_tables)
        rng = np.random.RandomState(7)
        np.random.seed(4242)
        produced = 0
        for _ in range(20):
            state = new_game(N_PLAYERS, lut)
            for i in range(N_PLAYERS):
                fast_state = cy.FastState.from_poker_env(state)
                delta = cyt.traverse_rng(ct, fast_state, i, _T, rng)
                for (r, iset), row in delta.items():
                    assert isinstance(r, int) and isinstance(iset, bytes)
                    assert row.dtype == np.int64
                    assert row.shape == (MAX_ACTIONS_PER_STREET[r],)
                    produced += 1
        assert produced > 0


@pytest.mark.requires_lut
class TestCoreTraverseMultiway:
    """3-player gate: directly exercises the ``not is_seat_active(i)`` early
    return at a *non-terminal* node — a folded traversing player whose payout is
    already locked in while other seats keep acting.  Heads-up never reaches this
    branch distinctly (a fold there also ends the hand), so multiway is the only
    coverage of the post-fold descent and its replay-consumption count.
    """

    def test_local_delta_byte_identical_3player(self, tmp_path, lut):
        try:
            new_game(3, lut)
        except Exception as exc:  # pragma: no cover - LUT may be HU-only
            pytest.skip(f"3-player new_game unsupported on this LUT: {exc}")
        tables = _build_cached_tables(tmp_path, lut, n_players=3)
        try:
            ct = _core_tables(tables)
            compared = 0
            for deal_seed in range(16):
                np.random.seed(7000 + deal_seed)
                state = new_game(3, lut)
                for i in range(3):
                    delta_py, choices = run_recording(
                        tables, state, i, _T, seed=deal_seed
                    )
                    fast_state = cy.FastState.from_poker_env(state)
                    delta_core = cyt.traverse_replay(
                        ct, fast_state, i, _T, choices
                    )
                    assert_local_delta_equal(delta_py, delta_core)
                    compared += 1
            assert compared > 0
        finally:
            tables.close()


class TestRngSamplerDistribution:
    def test_frequencies_match_sigma(self):
        """The C external-sampling draw is proportional to sigma over legals."""
        legal = ["fold", "call", "raise:1.0"]
        a2i = {"fold": 0, "call": 1, "all_in": 2, "raise:1.0": 3}
        # sigma indexed by column; all_in (col 2) is illegal → mass on legals
        # already sums to 1 (0.2 + 0.5 + 0.3).
        sigma = [0.2, 0.5, 0.0, 0.3]
        rng = np.random.RandomState(0)
        n = 40000
        counts = Counter(
            cyt.rng_sample_test(rng, legal, sigma, a2i) for _ in range(n)
        )
        assert abs(counts["fold"] / n - 0.2) < 0.01
        assert abs(counts["call"] / n - 0.5) < 0.01
        assert abs(counts["raise:1.0"] / n - 0.3) < 0.01

    def test_degenerate_sigma_uniform_fallback(self):
        """All-zero sigma over legals → uniform draw (no degenerate vector)."""
        legal = ["fold", "call"]
        a2i = {"fold": 0, "call": 1, "all_in": 2}
        sigma = [0.0, 0.0, 0.0]
        rng = np.random.RandomState(1)
        n = 20000
        counts = Counter(
            cyt.rng_sample_test(rng, legal, sigma, a2i) for _ in range(n)
        )
        assert abs(counts["fold"] / n - 0.5) < 0.02
        assert abs(counts["call"] / n - 0.5) < 0.02
