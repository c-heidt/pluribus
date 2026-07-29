"""Golden-trace regression for the single-process CFR trainer (Phase 0 safety net).

``simple_search(seed=42)`` is fully deterministic given a fixed LUT (it seeds both
``numpy`` and ``random`` at entry), so a short run on the committed 20-card LUT
produces byte-identical regret + strategy tables every time.  This module freezes
a digest of those tables as the **forward regression oracle** for the Tier-1
compiled-core rewrite: every phase that touches the traversal must reproduce this
exact digest through the pure-Python path, and Phase 3 will additionally assert
that the compiled core reproduces it too (``digest(core) == digest(python)``).

The digest is a pure function of the *trained values*, independent of chunk size
or the shm index cache, so it is a faithful fingerprint of the algorithm's output.

Regeneration
------------
The frozen digest is tied to the game abstraction (``RAISE_SIZES_BY_STAGE`` /
``CANONICAL_ACTIONS``), the 20-card LUT, and the CFR maths.  A *deliberate* change
to any of those changes the digest; that is the point of a regression anchor.
When such a change is intended, regenerate the literal with::

    python -m pytest test/training/functional/test_core_golden_trace.py \
        -k regenerate -s -q

which prints the current digest to copy into :data:`GOLDEN_DIGEST`.  An
*unexpected* mismatch means the traversal / regret maths drifted — investigate,
do not blindly re-freeze.
"""

import hashlib
from pathlib import Path

import pytest

from environment.action_space import MAX_ACTIONS_PER_STREET
from poker_ai.tables.cfr_tables import CFRTables
from poker_ai.tables.chunk_store import CHUNK_SIZE
from poker_ai.tables.index import lmdb_map_size_for_players
from poker_ai.blueprint.singleprocess.train import simple_search

# Fixed run configuration — small enough to be a fast functional test (~3 s),
# large enough to allocate rows on every street (good coverage of the trace).
N_ITERATIONS = 60
N_PLAYERS = 2
_LUT_PATH = "data/20cards_exact"

# Frozen fingerprint of the merged regret + strategy tables after the run above.
# See the module docstring for when and how to regenerate this.
# Re-baselined 2026-07-17 after the Pluribus-style strategy switch (commit
# 598e567): ``update_strategy`` now returns at the end of the pre-flop round and
# branches over ALL opponent actions instead of sampling one.  That moves the
# digest on BOTH table kinds, as expected:
#   * strategy — post-flop phi is never written online any more (streets 1-3 are
#     now all-zero here; the post-flop blueprint comes from ``poker_ai train
#     average`` over retained checkpoints).  Only street 0 carries phi mass.
#   * regret   — NOT a regret-maths change.  ``strategy_step`` and ``cfr_step``
#     share the global numpy RNG stream (singleprocess/train.py), so changing how
#     many draws the strategy walk consumes shifts the stream and every later
#     traversal takes a different (equally valid) trajectory.
# Verified at re-baseline time: digest is identical with PLURIBUS_CORE_KERNELS
# unset vs "all", and reproducible across runs.
# Re-baselined 2026-07-29 after widening RAISE_SIZES_BY_STAGE for preflop/flop
# and making MAX_RAISES_PER_ROUND player-count-dependent (was flat 3) — both
# change the action abstraction the digest is explicitly tied to (see module
# docstring). Full test/environment/ suite green (3655 passed) before this
# re-baseline; regenerated via train_and_digest(), not via `-k regenerate -s`
# (that marker is @pytest.mark.skip unconditionally, so `-k` alone doesn't
# run it — called the helper directly instead).
# Previous digests:
#   00cb45e7dc0c67a2a2a7beb14068c3061a8f26be952ecf1c546be42ff147aed2  (pre raise-grid widening)
#   7fef05fc8c7c421e8c667bd049530aeb3121c16aeb41ecbeb7f33bdd83243022  (pre preflop-only phi)
#   4ca50490d570bd4a8318b03112dd40f0466796d89c4fd8ee5a9ddb350b4f99c9  (after _hand_over all-in fix)
#   e23d93b6b08842048d37b58ee0c166e3c9260fd803712ead082d8efe5f221251  (pre all-in fixes)
GOLDEN_DIGEST = "e3395c03920c6bfcbe134e183077b3140a5b7cd3bae6b105b5f2b58133a901c0"


def train_and_digest(save_path: Path, *, n_iterations: int = N_ITERATIONS):
    """Run ``simple_search(seed=42)`` into *save_path* and digest the tables.

    Returns ``(digest, counts)`` where ``digest`` is a SHA-256 over every
    allocated regret then strategy row (per street, in flat-row / allocation
    order) and ``counts`` maps ``(kind, street) -> n_allocated``.  Reading only
    allocated rows makes the digest independent of ``CHUNK_SIZE`` and of the
    unallocated (zero) tail of each chunk.

    Reusable across phases: pass the same ``n_iterations`` and compare the
    digest produced by the compiled core against the pure-Python one.
    """
    simple_search(
        config={},
        save_path=save_path,
        lut_path=_LUT_PATH,
        pickle_dir=False,
        strategy_interval=1,
        n_iterations=n_iterations,
        discount_duration_cycles=100,
        prune_threshold=9_999_999,  # keep CFR-P out of this short baseline
        c=-300_000_000,
        n_players=N_PLAYERS,
        update_threshold=0,
        sync_interval=5,
        discount_interval=1,
    )
    tables = CFRTables(
        index_path=save_path / "lmdb_index",
        shm_dir=str(save_path / "shm"),
        actions_per_street=MAX_ACTIONS_PER_STREET,
        lmdb_map_size=lmdb_map_size_for_players(N_PLAYERS),
    )
    try:
        digest = hashlib.sha256()
        counts = {}
        for kind, table_set in (("regret", tables.regret), ("strategy", tables.strategy)):
            for r in range(4):
                # Use the per-street InfosetIndex row count that regret[r] and
                # strategy[r] SHARE — not either table's own ``n_allocated``.
                # ``_locate_row`` bumps a table's counter only for infosets that
                # table allocated *first* (``is_new`` on the shared index), so on
                # live tables regret and strategy each under-count and disagree;
                # only the shared index has the true dense row range ``[0, n)``.
                # On a freshly reopened CFRTables both per-table counters already
                # equal this (the digest is therefore unchanged), but reading the
                # index keeps this helper correct if a future phase digests live
                # in-process tables instead of reopening.
                n = tables._indexes[r].n_allocated_rows
                counts[(kind, r)] = n
                for flat_row in range(n):
                    chunk_id, local_row = divmod(flat_row, CHUNK_SIZE)
                    row = table_set[r].get_row_by_location(chunk_id, local_row)
                    digest.update(row.tobytes())
        return digest.hexdigest(), counts
    finally:
        tables.close()


@pytest.mark.requires_lut
class TestGoldenTrace:
    def test_matches_frozen_digest(self, tmp_path):
        """The trained tables reproduce the frozen golden digest byte-for-byte."""
        digest, _ = train_and_digest(tmp_path)
        assert digest == GOLDEN_DIGEST, (
            "single-process trainer output drifted from the golden trace. If this "
            "was an intentional abstraction/algorithm change, regenerate "
            "GOLDEN_DIGEST (see module docstring); otherwise investigate a "
            "regression in the traversal or regret maths."
        )

    def test_matches_frozen_digest_deferred_alloc(self, tmp_path, monkeypatch):
        """Deferred-durability allocation is byte-identical to the LMDB path.

        With ``PLURIBUS_DEFERRED_ALLOC=1`` the shm cache assigns row numbers and
        LMDB is bulk-flushed at run-end instead of one txn per new info set.
        Single-process allocation is sequential, so the info-set→row mapping (and
        therefore every chunk row the digest reads) is unchanged — the golden
        digest must still hold.  This is the byte-exact gate for the allocator
        rework.
        """
        monkeypatch.setenv("PLURIBUS_DEFERRED_ALLOC", "1")
        monkeypatch.setenv("PLURIBUS_INDEX_CACHE", "1")
        digest, _ = train_and_digest(tmp_path)
        assert digest == GOLDEN_DIGEST, (
            "deferred-allocation trainer output drifted from the golden trace — "
            "the shm-cache allocator is not byte-identical to the LMDB allocator."
        )

    def test_deterministic_across_runs(self, tmp_path):
        """Two independent runs produce the identical digest (nondeterminism guard).

        Abstraction-independent: this must hold regardless of the frozen literal,
        and is the property the whole make/undo + seeded-RNG design relies on.

        Scope: both runs share this interpreter's ``PYTHONHASHSEED``, so this
        guards run-to-run nondeterminism but not hash-seed portability of the
        checked-in ``GOLDEN_DIGEST``.  Portability holds by construction — row
        allocation order is hash-independent (canonical-ordered legal actions,
        ``sorted`` overlay, insertion-ordered history, ``sorted`` cards,
        DFS-ordered deltas) — so no ``PYTHONHASHSEED``-varying run is needed.
        """
        d1, c1 = train_and_digest(tmp_path / "run1")
        d2, c2 = train_and_digest(tmp_path / "run2")
        assert d1 == d2
        assert c1 == c2

    def test_all_streets_covered(self, tmp_path):
        """Every street allocates rows, so the trace exercises the full tree."""
        _, counts = train_and_digest(tmp_path)
        for r in range(4):
            assert counts[("regret", r)] > 0, f"no regret rows on street {r}"
            assert counts[("strategy", r)] > 0, f"no strategy rows on street {r}"


@pytest.mark.requires_lut
@pytest.mark.skip(reason="regeneration helper — run explicitly with -k regenerate -s")
def test_regenerate_golden_digest(tmp_path):
    """Print the current digest for pasting into ``GOLDEN_DIGEST``.

    Skipped by default; run with ``-k regenerate -s`` after a deliberate
    abstraction / algorithm change.
    """
    digest, counts = train_and_digest(tmp_path)
    print(f"\nGOLDEN_DIGEST = \"{digest}\"")
    print(f"counts = {counts}")
