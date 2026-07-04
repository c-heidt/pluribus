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
GOLDEN_DIGEST = "e23d93b6b08842048d37b58ee0c166e3c9260fd803712ead082d8efe5f221251"


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
