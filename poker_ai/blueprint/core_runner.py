"""Compiled-core driver for the CFR hot loop (Phase 4 wiring).

Bridges the pure-Python training schedule
(:func:`poker_ai.blueprint.training.cfr_step`) to the Cython core
(:mod:`poker_ai._core._traverse`).  The schedule, the deal, the shared
tables, ``merge_local_delta``, discounting, and checkpointing all stay in
Python and are untouched; the only thing that moves is the per-traversal
recursion, which the core owns.

Opt-in, per process
--------------------
The core is selected by the ``PLURIBUS_CFR_CORE`` environment variable and is
strictly a **fallback-retaining** switch: when it is unset (or the run is
biased — the biased traversal is not ported yet), the Python path runs
unchanged.  :func:`core_enabled` is the single decision point; both the
single-process loop and every worker consult it and, when it returns ``True``,
build one :class:`CoreDriver` for the lifetime of the process.

Why a driver object (not a free function)
-----------------------------------------
The in-core read path is pure-shm: :class:`~poker_ai._core._traverse.CoreTables`
holds each street's shm index-cache arrays and regret chunk mmaps.  Those must
be the *live* process's own mappings, so the driver is constructed **after** the
fork (in the worker's ``run``) / after ``prewarm_caches`` (single-process), and
then reused for every traversal — building it once amortises the
:func:`~poker_ai._core._state.configure` dump and the ``CoreTables`` wiring over
the whole run.

Importing this module never requires the compiled extension — the Cython
imports happen inside :meth:`CoreDriver.__init__`, so a build-less checkout can
still import :mod:`poker_ai.blueprint.training` and run the Python path.
"""

import logging
import os

log = logging.getLogger("poker_ai.blueprint.core_runner")

CORE_FLAG_ENV = "PLURIBUS_CFR_CORE"
"""Environment variable selecting the compiled core (``"1"`` enables it)."""


def core_enabled(bias: str = "none") -> bool:
    """Return ``True`` iff the compiled core should drive CFR this process.

    Two gates, both of which must pass:

    * ``PLURIBUS_CFR_CORE=1`` — the operator opted in.
    * ``bias == "none"`` — the biased traversal (``_traverse_biased``) is not
      ported into the core yet (deferred), so any biased run silently and
      correctly stays on the Python path rather than dropping the bias.

    Everything else about the run (schedule, tables, merge, discount,
    checkpoint) is identical between the two paths, so this is a pure
    dispatch switch.
    """
    if os.environ.get(CORE_FLAG_ENV, "0") != "1":
        return False
    if bias != "none":
        log.info(
            "PLURIBUS_CFR_CORE is set but bias=%r — the biased traversal is not "
            "ported into the compiled core yet; using the Python path.", bias
        )
        return False
    return True


class CoreDriver:
    """Per-process handle that runs one CFR traversal through the Cython core.

    Construct once, after the shm index caches are prewarmed (single-process)
    or after the fork + LMDB reopen (worker).  Holds the configured
    :class:`~poker_ai._core._traverse.CoreTables` read view and the RNG the
    external-sampling opponent draws come from.
    """

    def __init__(self, tables, n_players: int):
        """Wire the core against ``tables`` and verify the pure-shm invariant.

        Parameters
        ----------
        tables : poker_ai.tables.cfr_tables.CFRTables
            The shared tables this process reads.  Must have the shm index
            cache attached and prewarmed — the in-core read path has no LMDB
            fallback, so a missing/stale cache would read "uniform everywhere"
            with no crash.  :meth:`_verify_caches` turns that silent failure
            into a loud one at startup.
        n_players : int
            This run's player count — resolves :func:`max_raises_per_round`
            before it is dumped into the (process-global) state engine.
        """
        # Deferred so an unbuilt checkout can still import the training module.
        from environment.action_space import (
            ACTION_TO_IDX,
            CANONICAL_ACTIONS,
            MAX_ACTIONS_PER_STREET,
        )
        from environment.poker_env import (
            RAISE_SIZES_BY_STAGE,
            _ACTION_BYTE,
            _STAGE_ID,
            max_raises_per_round,
        )
        from poker_ai._core import _state as _cy_state
        from poker_ai._core import _traverse as _cy_traverse
        import numpy as np

        if getattr(tables, "_index_caches", None) is None:
            raise RuntimeError(
                "PLURIBUS_CFR_CORE requires the shm index cache "
                "(PLURIBUS_INDEX_CACHE=1 + CFRTables(enable_index_cache=True)); "
                "the in-core read path is pure-shm with no LMDB fallback."
            )
        self._verify_caches(tables)

        # Dump the encoding alphabet + raise grid into the state engine once
        # per process.  Never a hard-coded copy — these derive from the raise
        # grid and would silently drift.  Idempotent across processes.
        if not _cy_state.is_configured():
            _cy_state.configure(
                _STAGE_ID, _ACTION_BYTE, RAISE_SIZES_BY_STAGE,
                max_raises_per_round(n_players),
            )

        self._FastState = _cy_state.FastState
        self._traverse_rng = _cy_traverse.traverse_rng
        self._strategy_rng = _cy_traverse.strategy_rng
        self._core_tables = _cy_traverse.CoreTables(
            tables, CANONICAL_ACTIONS, ACTION_TO_IDX, MAX_ACTIONS_PER_STREET
        )
        # The module object exposes ``random_sample`` and shares the global
        # numpy stream, so the worker's per-process os.urandom seed (and the
        # pruning coin drawn in cfr_step) drive the opponent sampling too —
        # independent streams across workers, one stream per process.
        self._rng = np.random

    @staticmethod
    def _verify_caches(tables) -> None:
        """Hard-assert each street's cache is a complete mirror of its index.

        The core read path trusts the shm cache to hold *every* allocated row
        (a miss is treated as an unseen infoset → uniform).  That holds only if
        the cache was prewarmed and never overflowed.  Compare, per street, the
        O(1) cache occupancy counter against the index's allocated-row
        watermark; a mismatch means a botched prewarm/resume and would corrupt
        training quietly, so fail loud at startup instead.
        """
        caches = tables._index_caches
        indexes = tables._indexes
        for r in range(4):
            occ = caches[r].occupancy()
            alloc = indexes[r].n_allocated_rows
            if occ != alloc:
                raise RuntimeError(
                    "shm index cache for street %d is not a complete mirror of "
                    "the index: cache occupancy=%d but allocated rows=%d — "
                    "prewarm_caches() must run (and never overflow) before the "
                    "core reads. The pure-shm path has no LMDB fallback, so a "
                    "short cache would silently train on uniform strategies."
                    % (r, occ, alloc)
                )

    def run_cfr(self, state, i, t, c, prune_on, local_delta) -> None:
        """Run one in-core CFR traversal, accumulating into ``local_delta``.

        Byte-for-byte replacement for the ``cfr`` / ``cfrp`` call inside
        :func:`poker_ai.blueprint.training.cfr_step`: same regret increments,
        same ``(round, info_set)`` keys, accumulated **in place** into the
        caller-owned ``local_delta`` (so a worker's persistent batch buffer
        keeps working exactly as it does on the Python path).

        Parameters
        ----------
        state : environment.poker_env.PokerEnv
            The freshly dealt root state for this traversal.
        i : int
            Traversing player.
        t : int
            Iteration counter (unweighted increment; forwarded for parity).
        c : int
            CFR-P regret threshold.  Used only when ``prune_on``.
        prune_on : bool
            Whether this traversal prunes — the pruning coin is drawn once, in
            ``cfr_step``, so the core and Python paths make the identical
            cfr/cfrp choice for a given RNG state.
        local_delta : dict[tuple[int, bytes], numpy.ndarray]
            Caller-owned regret accumulator, mutated in place.
        """
        fast_state = self._FastState.from_poker_env(state)
        prune = c if prune_on else None
        self._traverse_rng(
            self._core_tables, fast_state, i, t, self._rng,
            prune=prune, local_delta=local_delta,
        )

    def run_strategy(self, state, i, local_strategy_delta) -> None:
        """Run one in-core strategy-sampling playthrough for player ``i``.

        Byte-for-byte replacement for the ``update_strategy`` call inside
        :func:`poker_ai.blueprint.training.strategy_step`: a single sampled line
        that accumulates player ``i``'s average-strategy visit counts **in place**
        into the caller-owned ``local_strategy_delta`` (mirror of the regret
        ``local_delta``), so the worker's persistent strategy buffer keeps working
        exactly as the regret buffer does across a batch.  Opponent sampling is
        drawn from the same per-process ``numpy.random`` stream as
        :meth:`run_cfr`.

        Parameters
        ----------
        state : environment.poker_env.PokerEnv
            The freshly dealt root state for this playthrough.
        i : int
            Player whose average strategy is being updated.
        local_strategy_delta : dict[tuple[int, bytes], numpy.ndarray]
            Caller-owned visit-count accumulator, mutated in place.  Flushed by
            :func:`poker_ai.blueprint.cfr.merge_local_strategy_delta`.
        """
        fast_state = self._FastState.from_poker_env(state)
        self._strategy_rng(
            self._core_tables, fast_state, i, self._rng,
            local_strategy_delta=local_strategy_delta,
        )
