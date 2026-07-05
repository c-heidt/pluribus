# cython: language_level=3
"""In-core external-sampling CFR traversal (Phase 3).

Ports ``poker_ai.blueprint.cfr._traverse`` (the MCCFR hot loop) into the compiled
core.  A traversal walks a :class:`poker_ai._core._state.FastState` (the Phase-2
betting engine) and produces a ``local_delta`` dict — ``(betting_round,
info_set_bytes) -> int64 ndarray`` — **byte-identical** to what the pure-Python
``_traverse`` produces for the same dealt hand, the same read-only regret
snapshot, and the same opponent-action choices.  Python's existing
``merge_local_delta`` consumes that dict verbatim (Phase 4 wiring).

What moves into C here (the per-node Python costs the profile flagged):

* **Row resolution is a pure-shm probe.**  :class:`CoreTables` holds each
  street's :class:`~poker_ai.tables.shm_index_cache.ShmIndexCache` arrays and its
  regret chunk mmaps; a lookup is ``_index.lookup`` (the Phase-1c hash + inverted
  high/low-word open-addressing probe) then ``divmod(flat_row, CHUNK_SIZE)`` into
  a held ``int32`` memoryview — **no LMDB txn, no ``get_row_if_exists`` Python
  chain**.  A miss returns ``None`` (unseen → uniform), exactly as
  ``InfosetIndex.get`` does with a cache attached (``index.py`` has *no* LMDB
  fallback on a cache miss), which is *required* for equivalence, not a shortcut.
* **Regret matching writes a C ``float`` buffer**, reproducing
  ``_regret._from_int32`` bit-for-bit (float64 positive-regret total in ascending
  index order, ``1.0/total``, float64→float32 round-to-nearest-even last) — no
  per-node numpy ``sigma`` allocation.
* **``vo`` accumulation and the regret increment are C.**  ``vo += (double)
  sigma[col] * voa`` matches numpy's ``float32 * pyfloat -> float64`` promotion;
  the increment is ``llrint(voa - vo)`` — round-half-to-even, identical to
  Python 3 ``int(round(cfv - vo))``.

The betting transitions (``legal_actions`` / ``step_in_place`` / ``undo`` /
``info_set`` / ``payout``) stay in ``FastState`` — already compiled and proven
byte-identical to ``PokerEnv`` by the Phase-2 differential — and are called as
its public methods; this module never re-implements them.

Opponent (external-sampling) nodes get their action from a pluggable sampler:
:class:`_Replay` feeds a recorded Python sequence (the Phase-0 mock-sampler gate —
byte-exact and RNG-free) while an injected ``numpy.random.RandomState`` drives the
production path (RNG byte-parity with numpy is deliberately *not* pursued; the
production path is certified distributionally + at equilibrium, per the plan).

Read semantics require the shm index cache (``enable_index_cache=True`` +
``prewarm_caches``); :class:`CoreTables` raises if it is absent.
"""

import numpy as np
from libc.math cimport llrint
from libc.stdint cimport uint64_t

from poker_ai._core._index import lookup as _index_lookup

# Widest canonical action row across streets is 6 (pre-flop); 16 is generous
# headroom.  Enforced against the dumped widths in CoreTables.__init__ so a
# future abstraction that exceeds it fails loudly, never silently truncates.
DEF MAX_ACTIONS = 16

# River betting round — the CFR-P prune predicate always explores it in full
# (``_prune``: ``state._betting_stage == "river"``).
DEF ST_RIVER = 3


cdef class _Replay:
    """Opponent sampler that replays a recorded action sequence in DFS order.

    Mirrors :class:`test.training.core_diff.ReplaySampler`: pop the next recorded
    action per opponent node, assert it is legal at the node and that the
    sequence is neither over- nor under-consumed — a divergence between the
    recording walk and the core walk shows up here as a loud assertion, which is
    the whole point of the byte-exact gate.
    """

    cdef list choices
    cdef Py_ssize_t pos

    def __init__(self, choices):
        self.choices = list(choices)
        self.pos = 0

    cdef str take(self, list legal):
        if self.pos >= len(self.choices):
            raise AssertionError(
                "replay exhausted: the driven core traversal visited more "
                "opponent nodes than were recorded — the two walks diverged"
            )
        action = self.choices[self.pos]
        self.pos += 1
        if action not in legal:
            raise AssertionError(
                "replayed action %r illegal at node with legal actions %r — "
                "the two walks diverged" % (action, legal)
            )
        return action

    def exhausted(self):
        return self.pos == len(self.choices)


cdef class CoreTables:
    """In-core read view over a :class:`~poker_ai.tables.cfr_tables.CFRTables`.

    Holds, per street: the shm index-cache arrays (for the lock-free
    ``info_set -> flat_row`` probe) and the regret :class:`ChunkStore` (for the
    ``flat_row -> int32 row`` read), plus the dumped canonical action list /
    action→column map / row width.  Reads only — the traversal accumulates into a
    caller-owned ``local_delta`` and never writes the shared tables.

    Requires the shm index cache: the in-core read path is pure-shm with no LMDB
    fallback (matching ``InfosetIndex.get`` when a cache is attached), so a cache
    is mandatory, not optional.
    """

    cdef list _keys            # per-street: cache keys ndarray (cap, 2) uint64
    cdef list _rows            # per-street: cache rows ndarray (cap,) uint64
    cdef uint64_t _mask[4]
    cdef list _regret_store    # per-street: regret ChunkStore
    cdef list _chunk_arr       # per-street: list of int32 (CHUNK_SIZE, nact) views
    cdef long _chunk_size
    cdef list _canonical       # per-street: list[str] canonical actions
    cdef list _a2i             # per-street: dict action -> column index
    cdef int _nact[4]
    cdef list _caches          # keep ShmIndexCache refs alive (mmaps)

    def __init__(self, tables, canonical_actions, action_to_idx, max_actions):
        from poker_ai.tables.chunk_store import CHUNK_SIZE
        cdef int r, w
        caches = getattr(tables, "_index_caches", None)
        if caches is None:
            raise RuntimeError(
                "CoreTables requires an attached shm index cache "
                "(CFRTables(enable_index_cache=True) + prewarm_caches()); the "
                "in-core read path is pure-shm with no LMDB fallback."
            )
        self._chunk_size = <long>CHUNK_SIZE
        self._keys = [None] * 4
        self._rows = [None] * 4
        self._caches = [None] * 4
        self._regret_store = [None] * 4
        self._chunk_arr = [None] * 4
        self._canonical = [None] * 4
        self._a2i = [None] * 4
        for r in range(4):
            w = int(max_actions[r])
            if w > MAX_ACTIONS:
                raise RuntimeError(
                    "street %d action width %d exceeds MAX_ACTIONS=%d — raise "
                    "the compile-time bound in _traverse.pyx" % (r, w, MAX_ACTIONS)
                )
            self._nact[r] = w
            cache = caches[r]
            self._caches[r] = cache
            self._keys[r] = cache._keys
            self._rows[r] = cache._rows
            self._mask[r] = <uint64_t>cache._mask
            self._regret_store[r] = tables.regret[r].store
            self._chunk_arr[r] = []
            self._canonical[r] = list(canonical_actions[r])
            self._a2i[r] = dict(action_to_idx[r])

    def probe_row(self, int r, bytes iset):
        """Test accessor: the ``int32`` shm regret row for ``iset``, or ``None``.

        Byte-identical to ``ChunkedTable.get_row_if_exists`` — the isolated
        Phase-3a row-reader differential gate.  Not on the hot path (the
        traversal uses the ``cdef`` :meth:`_regret_row` directly).
        """
        return self._regret_row(r, iset)

    cdef long _probe(self, int r, bytes iset) except *:
        """Flat row for ``iset`` on street ``r``, or ``-1`` on a cache miss.

        ``except *`` is REQUIRED: a bare ``cdef long`` returning a C scalar would
        swallow any raise from the Python ``_index_lookup`` call (leaving the
        exception set-but-unchecked while the caller proceeds on a garbage row) —
        the Phase-2 ``_apply`` lesson.  ``except *`` (not ``except -1``) because
        ``-1`` is a legitimate miss sentinel, not an error flag.
        """
        cdef uint64_t[:, ::1] keys = self._keys[r]
        cdef uint64_t[::1] rows = self._rows[r]
        return _index_lookup(keys, rows, self._mask[r], iset)

    cdef object _regret_row(self, int r, bytes iset):
        """Return the ``int32`` shm regret row for ``iset``, or ``None`` on miss.

        Byte-identical to ``ChunkedTable.get_row_if_exists``: probe → flat row →
        ``divmod(flat, CHUNK_SIZE)`` → the row slice of the held chunk view.  New
        chunks (rare; append-only, never remapped) are opened through the store
        on demand.
        """
        cdef long flat = self._probe(r, iset)
        if flat < 0:
            return None
        cdef long cid = flat // self._chunk_size
        cdef long loc = flat % self._chunk_size
        cdef list arrs = self._chunk_arr[r]
        store = self._regret_store[r]
        while len(arrs) <= cid:
            arrs.append(store.view(len(arrs)))
        cdef int[:, ::1] view = arrs[cid]
        return view[loc]

    cdef void _sigma(self, object row_mv, bint has_row, bint* mask, int n,
                     float* out) except *:
        """Regret matching into ``out`` — bit-exact to ``_regret._from_int32``.

        ``row_mv`` is the ``int32`` shm row (``has_row``) or ``None`` (miss → the
        all-zero row → uniform over the legal subset).  ``mask`` marks legal
        columns (always supplied, mirroring ``get_node_strategy``'s
        ``valid_mask``).  ``except *`` REQUIRED (``cdef void`` with a raising
        memoryview coercion — the Phase-2 rule).
        """
        cdef int[::1] row
        cdef Py_ssize_t k
        cdef double total = 0.0, inv, p
        cdef int nvalid = 0
        cdef int rr
        if has_row:
            row = row_mv
        for k in range(n):
            out[k] = 0.0
        for k in range(n):
            if not mask[k]:
                continue
            nvalid += 1
            if has_row:
                rr = row[k]
                if rr > 0:
                    total += rr
        if total > 0.0:
            inv = 1.0 / total
            for k in range(n):
                if not mask[k]:
                    continue
                if has_row:
                    rr = row[k]
                    if rr > 0:
                        out[k] = <float>((<double>rr) * inv)
            return
        if nvalid > 0:
            p = 1.0 / nvalid
            for k in range(n):
                if mask[k]:
                    out[k] = <float>p


cdef str _rng_sample(rng, list legal, float* sigma, dict a2i):
    """External-sampling draw over ``legal`` from ``sigma`` via ``rng``.

    Reproduces ``tree_utils.sample_action``'s inverse-CDF walk (build the legal
    sub-distribution in float64, normalise, single ``rng`` draw) but sourced from
    the injected ``RandomState`` — RNG byte-parity with global numpy is not
    required (the production path is certified distributionally + at
    equilibrium), only that the draw is proportional to ``sigma``.
    """
    cdef Py_ssize_t m = len(legal), k
    cdef double total = 0.0, threshold, cumulative
    cdef list probs = [0.0] * m
    cdef double v
    for k in range(m):
        v = <double>sigma[<int>a2i[legal[k]]]
        probs[k] = v
        total += v
    if total > 0.0:
        for k in range(m):
            probs[k] = (<double>probs[k]) / total
    else:
        for k in range(m):
            probs[k] = 1.0 / m
    threshold = rng.random_sample()
    cumulative = 0.0
    for k in range(m):
        cumulative += probs[k]
        if threshold < cumulative:
            return legal[k]
    return legal[m - 1]


def rng_sample_test(rng, list legal, sigma, dict a2i):
    """Test hook: exercise the C external-sampling draw in isolation.

    ``sigma`` is indexed by canonical column (like the per-node ``sigma``
    buffer).  Used by the distributional gate — draw many times and check the
    empirical frequencies match ``sigma`` restricted to ``legal`` — since the
    production sampler is not certified by RNG byte-parity.
    """
    cdef float buf[MAX_ACTIONS]
    cdef Py_ssize_t k, m = len(sigma)
    if m > MAX_ACTIONS:
        raise ValueError("sigma wider than MAX_ACTIONS")
    for k in range(m):
        buf[k] = <float>sigma[k]
    return _rng_sample(rng, legal, buf, a2i)


cdef double _traverse(CoreTables ct, s, int i, dict local_delta,
                      _Replay replay, rng, prune) except *:
    """Recursive external-sampling CFR traversal; returns player ``i``'s value.

    ``s`` is a :class:`FastState` (mutated in place via ``step_in_place`` /
    ``undo``).  ``replay`` xor ``rng`` selects opponent actions.  ``prune`` is
    ``None`` (explore every action — standard CFR) or an ``int`` regret threshold
    (CFR-P: explore an action iff on the river or its cumulative regret exceeds
    the threshold).  ``except *`` propagates any raise (a bare ``cdef double``
    would silently swallow it — the Phase-2 lesson).
    """
    # Terminal for i: hand over, or i has folded (no future decision can change
    # i's locked-in payout).  Mirrors tree_utils.is_terminal.
    if s.is_terminal or not s.is_seat_active(i):
        return <double>(s.payout()[i])

    cdef list legal = [a for a in s.legal_actions() if a is not None]
    if len(legal) == 0:
        return <double>(s.payout()[i])

    cdef int r = s.betting_round
    cdef int n = ct._nact[r]
    cdef bint mine = (s.player_i == i)

    cdef Py_ssize_t nleg = len(legal), li
    cdef int k, col, tok
    cdef bint has_row
    cdef bint mask[MAX_ACTIONS]
    cdef float sigma[MAX_ACTIONS]
    cdef double voa[MAX_ACTIONS]
    cdef bint explored[MAX_ACTIONS]
    cdef double vo = 0.0, cval
    cdef long pc = 0
    cdef bint has_prune = prune is not None
    cdef int rv
    cdef int[::1] rrow

    if not mine:
        # Opponent node: external sampling — descend exactly one branch.
        if replay is not None:
            action = replay.take(legal)
        else:
            # RNG path needs sigma over the legal actions at this node.
            iset = s.info_set()
            row_mv = ct._regret_row(r, iset)
            has_row = row_mv is not None
            canonical = ct._canonical[r]
            a2i = ct._a2i[r]
            legal_set = set(legal)
            for k in range(n):
                mask[k] = 1 if (canonical[k] in legal_set) else 0
            ct._sigma(row_mv, has_row, mask, n, sigma)
            action = _rng_sample(rng, legal, sigma, a2i)
        tok = s.step_in_place(action)
        cval = _traverse(ct, s, i, local_delta, replay, rng, prune)
        s.undo(tok)
        return cval

    # Traversing player's node.  info_set is resolved once and reused as both the
    # row-lookup key and the local_delta key (PokerEnv.info_set re-encodes on
    # every read — one of the hottest per-node costs the core removes).
    iset = s.info_set()
    row_mv = ct._regret_row(r, iset)
    has_row = row_mv is not None
    if has_row:
        rrow = row_mv
    canonical = ct._canonical[r]
    a2i = ct._a2i[r]
    legal_set = set(legal)
    for k in range(n):
        mask[k] = 1 if (canonical[k] in legal_set) else 0
    ct._sigma(row_mv, has_row, mask, n, sigma)

    for k in range(n):
        explored[k] = 0
    if has_prune:
        pc = <long>prune

    for li in range(nleg):
        action = legal[li]
        col = <int>a2i[action]
        if has_prune and r != ST_RIVER:
            rv = rrow[col] if has_row else 0
            if not (rv > pc):
                continue
        tok = s.step_in_place(action)
        cval = _traverse(ct, s, i, local_delta, replay, rng, prune)
        s.undo(tok)
        voa[col] = cval
        explored[col] = 1
        vo += (<double>sigma[col]) * cval

    # accumulate_regrets: the (r, info_set) key is created at EVERY traversing
    # node even when no action was explored (matching the Python reference's
    # lazy-zero allocation), so the two local_delta key sets stay identical.
    key = (r, iset)
    arr = local_delta.get(key)
    if arr is None:
        arr = np.zeros(n, dtype=np.int64)
        local_delta[key] = arr
    cdef long[::1] av = arr
    for k in range(n):
        if explored[k]:
            av[k] += <long>llrint(voa[k] - vo)
    return vo


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def traverse_replay(CoreTables ct, fast_state, int i, int t, choices,
                    prune=None):
    """Run one in-core traversal driven by a recorded opponent sequence.

    Returns the ``local_delta`` dict (``(round, info_set_bytes) -> int64
    ndarray``).  Asserts the replay is fully consumed — the byte-exact Phase-3
    gate: record a sequence with the Python traversal, replay it into both the
    Python and the core traversal, and assert the two ``local_delta`` dicts are
    identical.  ``t`` is accepted for signature parity with ``cfr`` (the
    increment is unweighted; discounting is applied elsewhere) and is unused.
    """
    cdef _Replay replay = _Replay(choices)
    cdef dict local_delta = {}
    _traverse(ct, fast_state, i, local_delta, replay, None, prune)
    if not replay.exhausted():
        raise AssertionError(
            "replay under-consumed: the driven core traversal visited fewer "
            "opponent nodes than were recorded — the two walks diverged"
        )
    return local_delta


def traverse_rng(CoreTables ct, fast_state, int i, int t, rng, prune=None):
    """Run one in-core traversal sampling opponents from ``rng`` (production path).

    ``rng`` is a ``numpy.random.RandomState``.  Returns the ``local_delta`` dict
    for ``merge_local_delta``.  RNG byte-parity with the Python path is not
    pursued; correctness is certified RNG-free (``traverse_replay``) and
    distributionally.
    """
    cdef dict local_delta = {}
    _traverse(ct, fast_state, i, local_delta, None, rng, prune)
    return local_delta
