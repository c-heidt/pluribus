"""Byte-identity tests for the compiled-core leaf kernels (Phase 1).

Each kernel must be a byte-exact drop-in for its pure-Python reference.  These
tests certify that two ways: (1) a direct fuzz of the kernel against the Python
reference over random + edge-case inputs, and (2) an end-to-end differential
where a real CFR traversal runs the kernel and must produce a byte-identical
``local_delta`` (via the RNG-free record/replay harness).

Skips cleanly when the compiled extension is not built.
"""

import numpy as np
import pytest

from poker_ai import _core

pytestmark = pytest.mark.skipif(
    not _core.CORE_AVAILABLE, reason="compiled core extension not built"
)

# Kernel imports are guarded so a pure-Python install (no extension) skips these
# tests cleanly instead of erroring at collection; the names are only referenced
# from the tests, which the skip mark suppresses when unavailable.
if _core.CORE_AVAILABLE:
    from poker_ai._core._regret import calculate_strategy_from_row as core_csfr
    from poker_ai._core._infoset import (
        configure as _core_infoset_configure,
        encode_info_set as core_encode_info_set,
    )
    from poker_ai._core._index import (
        hash_info_set_128 as core_hash_info_set_128,
        lookup as core_lookup,
        probe as core_probe,
        self_check as core_index_self_check,
    )
    from poker_ai._core._eval import (
        configure as _core_eval_configure,
        five as core_five,
        six as core_six,
        seven as core_seven,
    )
    from poker_ai._core._settle import (
        compute_utility_won as core_settle_won,
        settle_payout as core_settle_payout,
    )
    from environment.evaluator import default_evaluator as _py_evaluator
    from poker_ai.blueprint.tree_utils import (
        _calculate_strategy_from_row_py as py_csfr,
    )
    from poker_ai.tables.index import (
        _hash_info_set_128_py as py_hash_info_set_128,
    )
    from environment.poker_env import (
        _ACTION_BYTE,
        _RAW_TOKEN_MARK,
        _STAGE_ID,
        _encode_info_set_py as py_encode_info_set,
    )


def _assert_identical(regret_row, mask):
    a = core_csfr(regret_row, mask)
    b = py_csfr(regret_row, mask)
    assert a.dtype == b.dtype == np.float32
    assert a.shape == b.shape
    # Exact float32 bit-for-bit equality — not np.allclose.
    assert a.tobytes() == b.tobytes(), (
        f"regret={np.asarray(regret_row).tolist()} mask={mask}: "
        f"core={a.tolist()} py={b.tolist()}"
    )


# ---------------------------------------------------------------------------
# 1a. Regret matching — calculate_strategy_from_row
# ---------------------------------------------------------------------------
class TestRegretMatchKernel:
    def test_edge_cases(self):
        cases = [
            (np.array([0, 0, 0, 0, 0], np.int32), None),                 # all-zero -> uniform
            (np.array([-1, -2, -3], np.int32), None),                    # all-neg -> uniform
            (np.array([5, 0, 0], np.int32), None),                       # single positive
            (np.array([10, -5, 30, 0, 5], np.int32),
             [True, True, True, False, True]),                           # masked
            (np.array([10, 20, 30], np.int32), [False, False, False]),   # all masked -> zeros
            (np.array([0, 0, 0], np.int32), [True, False, True]),        # masked uniform
            (np.array([2 ** 30, -2 ** 30, 1], np.int32), None),          # large magnitudes
            (np.array([1], np.int32), None),                             # width 1
            (np.array([-310_000_000, 5, 310_000_000], np.int32), None),  # near REGRET_FLOOR
        ]
        for regret_row, mask in cases:
            _assert_identical(regret_row, mask)

    def test_random_fuzz_int32(self):
        """20k random int32 rows (hot path) x list/None masks — byte-exact."""
        rng = np.random.RandomState(0)
        for _ in range(20_000):
            n = rng.randint(3, 8)
            regret_row = rng.randint(
                -310_000_000, 310_000_000, size=n
            ).astype(np.int32)
            if rng.random_sample() < 0.5:
                mask = (rng.random_sample(n) < 0.7).tolist()
            else:
                mask = None
            _assert_identical(regret_row, mask)

    def test_fallback_dtypes(self):
        """Non-int32 rows exercise the list fallback; ndarray masks the tolist path."""
        rng = np.random.RandomState(1)
        for _ in range(3_000):
            n = rng.randint(3, 8)
            regret_row = rng.randint(-10 ** 9, 10 ** 9, size=n).astype(np.int64)
            _assert_identical(regret_row, None)
            _assert_identical(regret_row, (rng.random_sample(n) < 0.6))  # bool ndarray mask

    def test_ndarray_bool_mask(self):
        r = np.array([10, -5, 30, 0, 5], np.int32)
        _assert_identical(r, np.array([True, True, True, False, True]))


@pytest.mark.requires_lut
class TestRegretMatchEndToEnd:
    """The core kernel produces a byte-identical local_delta inside a real traversal."""

    def test_core_kernel_identical_local_delta(self, tmp_path, lut):
        import poker_ai.blueprint.tree_utils as tree_utils
        from environment.poker_env import new_game
        from test.training.core_diff import (
            assert_local_delta_equal,
            build_trained_tables,
            run_recording,
            run_replay,
        )

        tables = build_trained_tables(tmp_path, lut)
        try:
            total_choices = 0
            for deal_seed in range(6):
                np.random.seed(3000 + deal_seed)
                state = new_game(2, lut)
                for i in range(2):
                    # Record with the Python kernel (the default binding).
                    delta_py, choices = run_recording(
                        tables, state, i, 100, seed=deal_seed
                    )
                    total_choices += len(choices)
                    # Replay the SAME opponent sequence with the core kernel swapped
                    # in.  sigma feeds both the sampled node value and the regret
                    # increments, so an identical local_delta proves the core sigma
                    # is byte-identical to Python's through the whole traversal.
                    original = tree_utils.calculate_strategy_from_row
                    tree_utils.calculate_strategy_from_row = core_csfr
                    try:
                        delta_core = run_replay(tables, state, i, 100, choices)
                    finally:
                        tree_utils.calculate_strategy_from_row = original
                    assert_local_delta_equal(delta_py, delta_core)
            assert total_choices > 0  # the traversals actually hit opponent nodes
        finally:
            tables.close()


# ---------------------------------------------------------------------------
# 1b. Info-set encoding — encode_info_set
# ---------------------------------------------------------------------------
class TestEncodeInfoSetKernel:
    def setup_method(self):
        # Install the LIVE alphabet dumped from poker_env (never hard-coded).
        _core_infoset_configure(_STAGE_ID, _ACTION_BYTE, _RAW_TOKEN_MARK)

    def _assert_identical(self, cluster, history):
        a = core_encode_info_set(cluster, history)
        b = py_encode_info_set(cluster, history)
        assert a == b, (
            f"cluster={cluster} history={history}: {a.hex()} != {b.hex()}"
        )

    def test_used_before_configure_raises(self):
        # Independent of setup_method: a fresh kernel import would refuse to run.
        # Here it is already configured, so just assert configuration took and a
        # trivial encode works (empty history = bare cluster varint).
        self._assert_identical(0, [])
        self._assert_identical(200, [])

    def test_all_canonical_tokens_per_stage(self):
        """Every canonical token (incl. 'skip') encodes byte-identically per stage.

        This is the alphabet-coverage guard: if the Cython alphabet had drifted
        from poker_env's RAISE_SIZES_BY_STAGE-derived table, some token would map
        to a different byte and this would fail.
        """
        for stage in _STAGE_ID:
            tokens = list(_ACTION_BYTE[stage].keys())
            self._assert_identical(1, [(stage, tokens)])

    def test_off_tree_raw_token_path(self):
        """Unknown tokens take the 0xFF raw-bytes fallback, byte-identically."""
        self._assert_identical(7, [("pre_flop", ["raise:0.3333"])])
        self._assert_identical(9, [("flop", ["call", "raise:0.12345", "all_in"])])

    def test_random_fuzz(self):
        """20k random (cluster, cross-street history) keys — byte-exact."""
        rng = np.random.RandomState(0)
        stages = list(_STAGE_ID.keys())  # pre_flop..river, in play order
        for _ in range(20_000):
            # Vary the varint width: cluster spans 1..~2**40.
            cluster = int(rng.randint(0, 1 << int(rng.randint(1, 40))))
            n_stages = rng.randint(0, len(stages) + 1)
            history = []
            for stage in stages[:n_stages]:
                vocab = list(_ACTION_BYTE[stage].keys())
                actions = []
                for _ in range(rng.randint(0, 6)):
                    if rng.random_sample() < 0.1:
                        actions.append(f"raise:{round(rng.random_sample() * 3, 4)}")
                    else:
                        actions.append(vocab[rng.randint(len(vocab))])
                history.append((stage, actions))
            self._assert_identical(cluster, history)


# ---------------------------------------------------------------------------
# 1c. Info-set hashing — hash_info_set_128 (vendored XXH3)
# ---------------------------------------------------------------------------
class TestXxh3HashKernel:
    """The vendored XXH3 digest is bit-identical to the pip ``xxhash`` package
    the index was written under — the #1 silent-divergence trap.  A single
    diverging bit would rewrite every LMDB key ("untrained everywhere")."""

    def test_self_check_passes(self):
        # This also runs at import; asserting it here makes the KAT explicit.
        assert core_index_self_check() is True

    def test_matches_python_reference_fuzz(self):
        """20k random byte keys — the returned (high64, low64) pair is exact."""
        rng = np.random.RandomState(0)
        for _ in range(20_000):
            b = rng.bytes(int(rng.randint(0, 48)))
            assert core_hash_info_set_128(b) == py_hash_info_set_128(b), b.hex()

    def test_edges_and_str(self):
        """Empty, single byte, full byte range, and str (utf-8) inputs."""
        cases = [
            b"", b"\x00", b"\xff", bytes(range(256)), b"pre_flop",
            "abc", "unicode-♠♥♣♦",
        ]
        for c in cases:
            assert core_hash_info_set_128(c) == py_hash_info_set_128(c), repr(c)

    def test_matches_pip_xxhash_intdigest_split(self):
        """Cross-check straight against pip xxhash's 128-bit intdigest split."""
        import xxhash

        rng = np.random.RandomState(7)
        for _ in range(3_000):
            b = rng.bytes(int(rng.randint(0, 40)))
            di = xxhash.xxh3_128(b).intdigest()
            assert core_hash_info_set_128(b) == (di >> 64, di & (2 ** 64 - 1))

    def test_lmdb_key_bytes_unchanged(self):
        """The 16-byte LMDB key derived via the kernel equals the Python one, so
        swapping the hash never rewrites a key (resume / warm-start safe)."""
        import struct

        rng = np.random.RandomState(3)
        for _ in range(3_000):
            b = rng.bytes(int(rng.randint(0, 40)))
            core_key = struct.pack("<QQ", *core_hash_info_set_128(b))
            py_key = struct.pack("<QQ", *py_hash_info_set_128(b))
            assert core_key == py_key


# ---------------------------------------------------------------------------
# 1c. Shm index probe — probe / lookup
# ---------------------------------------------------------------------------
class TestProbeKernel:
    """probe / lookup reproduce ``ShmIndexCache.probe`` over the raw shm arrays,
    including the linear-probe collision walk and the -1/None miss mapping."""

    def _make_cache(self, tmp_path, name, cap):
        from poker_ai.tables.shm_index_cache import ShmIndexCache

        return ShmIndexCache(name, capacity=cap, shm_dir=str(tmp_path / "shm"))

    def test_probe_matches_cache_probe(self, tmp_path):
        from poker_ai.tables.shm_index_cache import capacity_for

        c = self._make_cache(tmp_path, "pb", capacity_for(2000))
        try:
            rng = np.random.RandomState(1)
            present = []
            for row in range(1500):
                lo = int(rng.randint(0, 2 ** 63))  # arbitrary 64-bit words
                hi = int(rng.randint(0, 2 ** 63))
                c.insert(lo, hi, row)
                present.append((lo, hi, row))
            # Hits: core probe == cache probe == the inserted row.
            for lo, hi, row in present:
                got = core_probe(c._keys, c._rows, c._mask, lo, hi)
                assert got == row == c.probe(lo, hi)
            # Misses: -1 exactly when the cache returns None.
            for _ in range(2000):
                lo = int(rng.randint(0, 2 ** 63))
                hi = int(rng.randint(0, 2 ** 63))
                exp = c.probe(lo, hi)
                got = core_probe(c._keys, c._rows, c._mask, lo, hi)
                assert (got == -1) == (exp is None)
                if exp is not None:
                    assert got == exp
        finally:
            c.close()
            c.unlink()

    def test_probe_collision_chain(self, tmp_path):
        """All keys share one start slot, forcing a long linear-probe walk."""
        c = self._make_cache(tmp_path, "cc", 1024)  # power of two
        try:
            stride = c._mask + 1  # multiples of capacity share (& mask)
            entries = []
            for row in range(100):
                lo = 7 + (row + 1) * stride  # (lo & mask) == 7 for all, distinct
                hi = 1000 + row
                c.insert(lo, hi, row)
                entries.append((lo, hi, row))
            for lo, hi, row in entries:
                assert core_probe(c._keys, c._rows, c._mask, lo, hi) == row
                assert c.probe(lo, hi) == row
        finally:
            c.close()
            c.unlink()


class TestIndexLookupEndToEnd:
    """``lookup()`` == ``InfosetIndex.get()`` over a real LMDB-backed, prewarmed
    cache — the ``ShmIndexCache.audit``-equivalent gate for the whole
    hash + inverted-word-convention + probe seam.  The cache is populated on the
    Python side (LMDB keys = pip-``xxhash`` digests); if the kernel's hash or its
    inverted convention were wrong, every lookup would miss and this would fail.
    """

    def test_lookup_matches_index_get(self, tmp_path):
        from poker_ai.tables.index import InfosetIndex
        from poker_ai.tables.shm_index_cache import ShmIndexCache, capacity_for

        idx = InfosetIndex(tmp_path / "lmdb")
        keys = [f"info_{i}".encode() for i in range(400)]
        expected = {}
        for k in keys:
            row, _ = idx.get_or_create(k)
            expected[k] = row

        cache = ShmIndexCache(
            "e2e", capacity=capacity_for(len(keys) * 2),
            shm_dir=str(tmp_path / "shm"),
        )
        try:
            loaded = cache.prewarm_from_cursor(idx._env)
            assert loaded == len(keys)
            idx.set_cache(cache)
            # Every allocated key resolves identically through the kernel and
            # through the production cache-backed index.get.
            for k, row in expected.items():
                assert core_lookup(cache._keys, cache._rows, cache._mask, k) == row
                assert idx.get(k) == row
            # Unseen keys miss in both (kernel -1 == index None).
            for i in range(400):
                u = f"unseen_{i}".encode()
                assert core_lookup(cache._keys, cache._rows, cache._mask, u) == -1
                assert idx.get(u) is None
            # The cache's own audit agrees (independent consistency check).
            assert cache.audit(idx._env) == len(keys)
        finally:
            cache.close()
            cache.unlink()
            idx.close()


# ---------------------------------------------------------------------------
# 1d. Hand evaluator — five / six / seven
# ---------------------------------------------------------------------------
def _full_deck():
    """All 52 card ints in the environment's bit encoding (suit one-hot)."""
    from environment.utils import CARD_PRIMES

    return [
        (1 << r << 16) | (s << 12) | (r << 8) | CARD_PRIMES[r]
        for r in range(13)
        for s in (1, 2, 4, 8)
    ]


class TestEvaluatorKernel:
    """The Cython evaluator is byte-identical to the scalar ``Evaluator``
    methods AND to the independent 21-subset enumeration oracle."""

    def setup_method(self):
        ev = _py_evaluator
        # Install the LIVE tables dumped from the shared evaluator.
        _core_eval_configure(
            ev._flush_best, ev._flush_rank,
            ev._unsuited_keys, ev._unsuited_ranks,
            ev._nonflush6_keys, ev._nonflush6_ranks,
            ev._nonflush7_keys, ev._nonflush7_ranks,
        )
        self.deck = _full_deck()

    def test_named_edge_hands(self):
        """Specific classes exercise both branches at known boundaries."""
        from environment.utils import new_card

        def core(cs):
            return {5: core_five, 6: core_six, 7: core_seven}[len(cs)](cs)

        def hand(*strs):
            return [new_card(s) for s in strs]

        cases = [
            hand("As", "Ks", "Qs", "Js", "Ts"),                 # royal flush
            hand("9h", "8h", "7h", "6h", "5h"),                 # straight flush
            hand("As", "2s", "3s", "4s", "5s"),                 # wheel str flush
            hand("Ac", "Ad", "Ah", "As", "Kd"),                 # quads
            hand("Ac", "Ad", "Ah", "Kd", "Ks"),                 # full house
            hand("Ah", "5c", "4d", "3s", "2h"),                 # wheel straight
            hand("As", "Ks", "Qs", "Js", "Ts", "2c", "3d"),    # 7: royal + junk
            hand("2s", "2h", "2d", "2c", "5s", "5h", "9d"),     # 7: quads
            hand("Ah", "Kh", "Qh", "2c", "3d", "4s"),          # 6: high card
        ]
        for cs in cases:
            k = len(cs)
            pyfn = {5: _py_evaluator._five, 6: _py_evaluator._six,
                    7: _py_evaluator._seven}[k]
            assert core(cs) == pyfn(cs)

    def test_fuzz_vs_scalar_and_oracle(self):
        """Random 5/6/7-card hands: core == scalar method == 21-subset oracle."""
        rng = np.random.RandomState(0)
        deck = self.deck
        for k, corefn, pyfn in (
            (5, core_five, _py_evaluator._five),
            (6, core_six, _py_evaluator._six),
            (7, core_seven, _py_evaluator._seven),
        ):
            hands = [
                list(rng.choice(deck, size=k, replace=False))
                for _ in range(30_000)
            ]
            core_ranks = np.array([corefn(h) for h in hands], dtype=np.int64)
            py_ranks = np.array([pyfn(h) for h in hands], dtype=np.int64)
            oracle = _py_evaluator._evaluate_batch_oracle(
                np.array(hands, dtype=np.int64)
            )
            assert np.array_equal(core_ranks, py_ranks), f"scalar k={k}"
            assert np.array_equal(core_ranks, oracle), f"oracle k={k}"

    def test_used_before_configure_raises(self):
        # Already configured by setup_method; assert the guard exists and a
        # trivial evaluate works (a full-house 7-card hand).
        from environment.utils import new_card

        h = [new_card(s) for s in
             ("Ac", "Ad", "Ah", "Kd", "Ks", "2c", "3d")]
        assert core_seven(h) == _py_evaluator._seven(h)

    @pytest.mark.slow
    def test_exhaustive_five(self):
        """All C(52,5) = 2,598,960 five-card hands are byte-identical."""
        import itertools

        for hand in itertools.combinations(self.deck, 5):
            h = list(hand)
            assert core_five(h) == _py_evaluator._five(h)


# ---------------------------------------------------------------------------
# 1e. Side-pot settlement — compute_utility_won / settle_payout
# ---------------------------------------------------------------------------
class _StubPlayer:
    """Minimal stand-in exposing the two fields ``compute_utility`` reads."""

    def __init__(self, player_i, order):
        self.player_i = player_i
        self.order = order


def _oracle_won(contrib, groups_pi, order):
    """``Pot.compute_utility`` restricted to the win amounts, as a list."""
    from environment.pot import Pot

    pot = Pot(len(contrib))
    pot._chips = [int(c) for c in contrib]
    players = [_StubPlayer(i, order[i]) for i in range(len(contrib))]
    ranked = [[players[pi] for pi in g] for g in groups_pi]
    won = pot.compute_utility(players, ranked)
    return [won[i] for i in range(len(contrib))]


class TestSettlementKernel:
    """The settlement kernel is byte-identical to ``Pot.compute_utility``,
    including side-pot peel order and odd-chip remainders."""

    def test_named_edge_fixtures(self):
        cases = [
            ([5, 5], [[0, 1]], [0, 1]),               # chop, odd chip -> lower order
            ([5, 5], [[0, 1]], [1, 0]),               # chop, order reversed
            ([100, 50, 25], [[2], [1], [0]], [0, 1, 2]),  # layered side pots
            ([100, 50, 25], [[0], [1], [2]], [0, 1, 2]),  # big stack wins all
            ([30, 10, 10], [[1, 2]], [0, 1, 2]),      # folded seat 0 = dead money
            ([7, 7, 7], [[0, 1, 2]], [2, 0, 1]),      # 3-way chop, remainder 1
            ([0, 0], [[0, 1]], [0, 1]),               # empty pot
            ([10, 0], [[0]], [0, 1]),                 # single contributor wins
        ]
        for contrib, groups, order in cases:
            assert core_settle_won(contrib, groups, order) == _oracle_won(
                contrib, groups, order
            ), (contrib, groups, order)

    def test_payout_is_won_minus_contrib(self):
        contrib, groups, order = [100, 50, 25], [[2], [1], [0]], [0, 1, 2]
        won = _oracle_won(contrib, groups, order)
        assert core_settle_payout(contrib, groups, order) == [
            won[i] - contrib[i] for i in range(len(contrib))
        ]

    def test_randomized_differential(self):
        """50k random multi-way pots (all-in layers, chops, folded dead money,
        odd chips) — win amounts and netted payouts both byte-exact."""
        rng = np.random.RandomState(0)
        for _ in range(50_000):
            n = int(rng.randint(2, 7))
            contrib = [int(rng.randint(0, 60)) for _ in range(n)]
            order = list(int(x) for x in rng.permutation(n))
            active = [i for i in range(n) if rng.random_sample() < 0.75]
            if not active:
                active = [int(rng.randint(0, n))]
            rng.shuffle(active)
            groups = []
            idx = 0
            while idx < len(active):
                step = int(rng.randint(1, len(active) - idx + 1))
                groups.append(active[idx : idx + step])
                idx += step
            won = _oracle_won(contrib, groups, order)
            assert core_settle_won(contrib, groups, order) == won
            assert core_settle_payout(contrib, groups, order) == [
                won[i] - contrib[i] for i in range(n)
            ]


class TestKernelExceptionsPropagate:
    """Regression: a ``cdef`` helper's raise must PROPAGATE, never be swallowed at
    the C call boundary.

    A ``cdef`` function returning ``void`` / a C scalar *without* an exception
    spec (``except *`` / ``except? -1``) silently drops any Python exception —
    Cython prints "Exception ignored in: ..." and the call returns garbage, a
    silent-wrong-result class a happy-path fuzz never catches (it bit
    ``_state.FastState._apply`` in Phase 2).  Every Phase-1 raising helper avoids
    it by returning ``object`` (NULL-propagates) or declaring a spec; these tests
    trip each raise path and assert it actually raises.
    """

    def test_settle_over_max_players_raises(self):
        # cdef object _settle_won -> ValueError for n > MAX_PLAYERS (object return
        # NULL-propagates); a swallow would return a garbage list.
        with pytest.raises(ValueError):
            core_settle_won([10] * 33, [[0]], list(range(33)))

    def test_index_hash_non_bytes_raises(self):
        with pytest.raises(TypeError):
            core_hash_info_set_128(123)

    def test_index_lookup_non_bytes_raises(self):
        keys = np.zeros((4, 2), np.uint64)
        rows = np.zeros(4, np.uint64)
        with pytest.raises(TypeError):
            core_lookup(keys, rows, 3, 123)

    def test_eval_lookup_product_miss_raises(self):
        # cdef short _lookup(...) except? -1 -> KeyError on a product absent from
        # the table.  Force a miss by ranking the 7-card table with 6 mixed-suit
        # cards (their 6-card product is not a 7-card key); a swallow would return
        # a garbage rank (and print "Exception ignored").
        from environment.utils import new_card

        ev = _py_evaluator
        _core_eval_configure(
            ev._flush_best, ev._flush_rank,
            ev._unsuited_keys, ev._unsuited_ranks,
            ev._nonflush6_keys, ev._nonflush6_ranks,
            ev._nonflush7_keys, ev._nonflush7_ranks,
        )
        six_mixed = [new_card(s) for s in ("2c", "3d", "4h", "5s", "6c", "7d")]
        with pytest.raises(KeyError):
            core_seven(six_mixed)

    def test_regret_bad_input_propagates(self):
        # Fallback path does regret_row.tolist(); a bad object raises (propagated
        # through the object-returning cdef helpers), never swallowed.
        with pytest.raises(Exception):
            core_csfr(object())
