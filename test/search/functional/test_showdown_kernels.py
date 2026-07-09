"""Byte-identity gate for the Phase-1 vector kernels (``_showdown`` / ``_regret``).

Each compiled kernel must be **bit-for-bit** identical to its numpy reference so
the frozen vector golden digest survives turning it on (float64 — a last-ULP
drift would break the anchor).  Two levels:

- direct differential **fuzz** of each kernel vs its ``_*_py`` oracle over random
  inputs (realistic evaluator ranks *and* adversarial arbitrary-rank / pairwise-
  width cases), and
- an **end-to-end** check that a full vector solve with the kernels swapped in
  reproduces the exact frozen ``GOLDEN_DIGEST_VECTOR`` (and that the MCCFR digest
  is untouched — the kernels are vector-only, a blast-radius guard).
"""

import numpy as np
import pytest

from environment import range_showdown as rs
from environment.utils import enumerate_combos
from poker_ai._core._regret import calculate_strategy_matrix as _core_rmm
from poker_ai._core._showdown import (
    reach_after_removal as _core_reach,
    showdown_cfv as _core_showdown,
)
from poker_ai.search import vector as vec_mod

from test.search.functional import test_search_core_golden as golden


# --------------------------------------------------------------------------- #
# showdown_cfv / reach_after_removal fuzz
# --------------------------------------------------------------------------- #

_LOW, _HIGH = 11, 14  # 16-card deck, 120 combos


def _deck_combos():
    cards, _ = enumerate_combos(_LOW, _HIGH)
    return cards, rs.removal_index(cards)


def test_showdown_cfv_bit_identical_realistic():
    """Kernel == numpy over realistic (evaluator-ranked) boards, all combos live."""
    cards, removal = _deck_combos()
    deck = np.unique(cards)
    rng = np.random.RandomState(0)
    for _ in range(200):
        board = list(rng.choice(deck, size=5, replace=False))
        ranks, valid = rs.rank_combos_on_board(cards, [int(c) for c in board])
        reach = rng.random_sample(cards.shape[0]) * rng.choice([1.0, 10.0, 0.001])
        stake = float(rng.randint(1, 5000))
        dead = float(rng.choice([0, rng.randint(0, 3000)]))
        a = rs.showdown_cfv_py(ranks, valid, cards, reach, stake, dead=dead, removal=removal)
        b = _core_showdown(ranks, valid, cards, reach, stake, dead=dead, removal=removal)
        assert np.array_equal(a, b), np.abs(a - b).max()


def test_showdown_cfv_bit_identical_arbitrary_ranks():
    """Kernel == numpy for arbitrary int64 ranks (exercises the unique fallback + ties)."""
    cards, removal = _deck_combos()
    n = cards.shape[0]
    rng = np.random.RandomState(1)
    for _ in range(200):
        # Heavy ties + out-of-domain values (negative, > 7462) → numpy fallback path.
        ranks = rng.randint(-5, 40, size=n).astype(np.int64)
        valid = rng.random_sample(n) > 0.3
        reach = rng.random_sample(n)
        stake = float(rng.randint(1, 1000))
        dead = float(rng.randint(0, 1000))
        a = rs.showdown_cfv_py(ranks, valid, cards, reach, stake, dead=dead, removal=removal)
        b = _core_showdown(ranks, valid, cards, reach, stake, dead=dead, removal=removal)
        assert np.array_equal(a, b)


def test_showdown_cfv_edge_cases():
    cards, removal = _deck_combos()
    n = cards.shape[0]
    # All board-incompatible (every combo the sentinel → one group, all zeroed).
    ranks = np.full(n, 1 << 30, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    reach = np.ones(n)
    a = rs.showdown_cfv_py(ranks, valid, cards, reach, 100.0, dead=50.0, removal=removal)
    b = _core_showdown(ranks, valid, cards, reach, 100.0, dead=50.0, removal=removal)
    assert np.array_equal(a, b)
    # Single combo.
    one = cards[:1]
    rem1 = rs.removal_index(one)
    a = rs.showdown_cfv_py(np.array([5], np.int64), np.array([True]), one,
                           np.array([2.0]), 10.0, dead=1.0, removal=rem1)
    b = _core_showdown(np.array([5], np.int64), np.array([True]), one,
                       np.array([2.0]), 10.0, dead=1.0, removal=rem1)
    assert np.array_equal(a, b)


def test_reach_after_removal_bit_identical():
    cards, removal = _deck_combos()
    n = cards.shape[0]
    rng = np.random.RandomState(2)
    for _ in range(200):
        valid = rng.random_sample(n) > 0.25
        reach = np.where(valid, rng.random_sample(n) * rng.choice([1.0, 100.0]), 0.0)
        a = rs.reach_after_removal_py(cards, reach, removal)
        b = _core_reach(cards, reach, removal)
        assert np.array_equal(a, b)


def test_reach_after_removal_removal_none_path():
    """The standalone ``removal=None`` branch computes densification internally."""
    cards, _ = _deck_combos()
    reach = np.random.RandomState(3).random_sample(cards.shape[0])
    assert np.array_equal(
        rs.reach_after_removal_py(cards, reach), _core_reach(cards, reach)
    )


# --------------------------------------------------------------------------- #
# Production-scale (full 52-card deck, 1326 combos) bit-identity.
#
# The golden digest and the fuzz above use the 16-card / 120-combo deck; but 120
# < 128, so ``reach_after_removal``'s ``pairwise_sum`` never leaves its base case
# there, and ``showdown_cfv`` is only exercised on a small tree.  Production runs
# at 1326 combos where the pairwise summation *recurses* and the counting sort
# spans thousands of real ranks — this is the scale the whole "core == numpy
# bit-for-bit" bet actually rides on, so assert it directly (the blueprint
# lesson: a green small-deck differential proves only happy-path parity).
# --------------------------------------------------------------------------- #

def _full_deck():
    cards, _ = enumerate_combos(2, 14)  # 52-card deck → 1326 combos
    return cards, rs.removal_index(cards)


def test_showdown_cfv_full_deck_bit_identical():
    cards, removal = _full_deck()
    assert cards.shape[0] == 1326
    deck = np.unique(cards)
    rng = np.random.RandomState(0)
    for _ in range(40):
        board = [int(c) for c in rng.choice(deck, size=5, replace=False)]
        ranks, valid = rs.rank_combos_on_board(cards, board)
        reach = rng.random_sample(1326) * rng.choice([1.0, 100.0])
        stake = float(rng.randint(1, 5000))
        dead = float(rng.choice([0, rng.randint(0, 3000)]))
        a = rs.showdown_cfv_py(ranks, valid, cards, reach, stake, dead=dead, removal=removal)
        b = _core_showdown(ranks, valid, cards, reach, stake, dead=dead, removal=removal)
        assert np.array_equal(a, b), np.abs(a - b).max()


def test_reach_after_removal_full_deck_bit_identical():
    """1326 > 128, so ``total`` goes through the pairwise-sum *recursion* here."""
    cards, removal = _full_deck()
    rng = np.random.RandomState(1)
    for _ in range(40):
        valid = rng.random_sample(1326) > 0.1
        reach = np.where(valid, rng.random_sample(1326) * rng.choice([1.0, 1000.0]), 0.0)
        a = rs.reach_after_removal_py(cards, reach, removal)
        b = _core_reach(cards, reach, removal)
        assert np.array_equal(a, b), np.abs(a - b).max()


def test_showdown_cfv_all_valid_no_sentinel():
    """Fast counting-sort path with every combo live (no sentinel group)."""
    cards, removal = _full_deck()
    n = cards.shape[0]
    rng = np.random.RandomState(2)
    ranks = rng.randint(1, 7463, size=n).astype(np.int64)  # all real, no sentinel
    valid = np.ones(n, dtype=bool)
    reach = rng.random_sample(n)
    a = rs.showdown_cfv_py(ranks, valid, cards, reach, 200.0, dead=40.0, removal=removal)
    b = _core_showdown(ranks, valid, cards, reach, 200.0, dead=40.0, removal=removal)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# calculate_strategy_matrix fuzz (pairwise-sum + division bit-identity)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("width", [1, 2, 5, 7, 8, 9, 16, 130, 257])
def test_regret_matrix_bit_identical(width):
    """Kernel == numpy across widths that span the pairwise-sum base cases + recursion."""
    rng = np.random.RandomState(width)
    for _ in range(30):
        n = int(rng.randint(1, 400))
        reg = (rng.standard_normal((n, width)) * rng.choice([1.0, 1e6, 1e-6]))
        # Force some rows all-negative (uniform 1/width fallback) and some zero.
        if n > 3:
            reg[0] = -np.abs(reg[0])
            reg[1] = 0.0
        a = vec_mod._regret_match_matrix_py(reg)
        b = _core_rmm(reg)
        assert np.array_equal(a, b), np.abs(a - b).max()


def test_regret_matrix_3d_and_strided():
    """The river-conditioned ``(n, n_rivers, width)`` tensor + a strided 2-D slice."""
    rng = np.random.RandomState(99)
    reg3 = rng.standard_normal((40, 6, 5)) * 1e5
    assert np.array_equal(vec_mod._regret_match_matrix_py(reg3), _core_rmm(reg3))
    # A non-contiguous slice (what ``_walk`` passes below the river chance node).
    sl = reg3[:, 3, :]
    assert not sl.flags.c_contiguous
    assert np.array_equal(vec_mod._regret_match_matrix_py(sl), _core_rmm(sl))


# --------------------------------------------------------------------------- #
# End-to-end golden digest with the kernels swapped in
# --------------------------------------------------------------------------- #

def test_vector_digest_unchanged_with_kernels(monkeypatch):
    """A full vector solve with all three kernels swapped in matches the frozen digest."""
    monkeypatch.setattr(rs, "showdown_cfv", _core_showdown)
    monkeypatch.setattr(rs, "reach_after_removal", _core_reach)
    monkeypatch.setattr(vec_mod, "_regret_match_matrix", _core_rmm)
    assert golden._vector_digest() == golden.GOLDEN_DIGEST_VECTOR


def test_mccfr_digest_unchanged_with_kernels(monkeypatch):
    """The MCCFR digest is untouched — the Phase-1 kernels are vector-only."""
    monkeypatch.setattr(rs, "showdown_cfv", _core_showdown)
    monkeypatch.setattr(rs, "reach_after_removal", _core_reach)
    monkeypatch.setattr(vec_mod, "_regret_match_matrix", _core_rmm)
    assert golden._mccfr_digest() == golden.GOLDEN_DIGEST_MCCFR
