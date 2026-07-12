"""Phase-3e gate: the fused ``fold_cfv`` core kernel vs the numpy oracle.

The vector regime's fold-terminal settlement (``vector_payout``'s fold branch)
is fused into one core sweep: ``signed_gain * where(valid, reach_after_removal(
where(valid, opp_reach, 0)), 0)``.  This gate proves the Cython kernel is
**byte-identical** to the pure-numpy ``range_showdown.fold_cfv_py`` (which is in
turn byte-identical to the inline expression it replaced) over a broad fuzz,
including the ``-0.0`` edge for a negative ``signed_gain`` on board-incompatible
combos.  Skips cleanly when the core is not built.
"""

import numpy as np
import pytest

from poker_ai import _core

pytestmark = pytest.mark.skipif(
    not _core.CORE_AVAILABLE, reason="compiled core extension not built"
)

if _core.CORE_AVAILABLE:
    from environment import range_showdown as rs
    from environment.utils import enumerate_combos
    from poker_ai._core._showdown import fold_cfv as core_fold_cfv


def _deck_removal(low, high):
    combo_cards, _ = enumerate_combos(low, high)
    return combo_cards, rs.removal_for(low, high)


@pytest.mark.parametrize("low,high", [(11, 14), (9, 14), (2, 14)])
def test_fold_cfv_matches_numpy_oracle(low, high):
    """Core ``fold_cfv`` == numpy ``fold_cfv_py`` bit-for-bit over random reach,
    random board-compat masks, and both signs of the gain."""
    combo_cards, removal = _deck_removal(low, high)
    n = combo_cards.shape[0]
    rng = np.random.RandomState(low * 100 + high)
    for trial in range(200):
        reach = rng.random_sample(n)
        if trial % 3 == 0:
            reach = reach * (rng.random_sample(n) > 0.4)  # sparse
        # Random board-compatibility mask (some combos invalid → exercise removal
        # + the where masking + the -0.0 path).
        valid = (rng.random_sample(n) > 0.15)
        if trial % 5 == 0:
            valid[:] = True                      # all-valid fast path
        signed_gain = rng.uniform(-1500, 1500)
        if trial % 7 == 0:
            signed_gain = -signed_gain           # force negatives (−0.0 edge)
        ref = rs.fold_cfv_py(valid, combo_cards, reach, signed_gain, removal)
        got = core_fold_cfv(valid, combo_cards, reach, signed_gain, removal)
        assert got.dtype == ref.dtype == np.float64
        # Byte-identical: exact equality AND identical sign bits on zeros.
        assert np.array_equal(got, ref), (
            f"value mismatch low={low} high={high} trial={trial}"
        )
        assert np.array_equal(
            np.signbit(got), np.signbit(ref)
        ), f"sign-bit (−0.0) mismatch low={low} high={high} trial={trial}"


def test_fold_cfv_removal_none_path():
    """The standalone ``removal=None`` densify path also matches the oracle."""
    combo_cards, _ = _deck_removal(11, 14)
    n = combo_cards.shape[0]
    rng = np.random.RandomState(1)
    reach = rng.random_sample(n)
    valid = rng.random_sample(n) > 0.2
    for sg in (600.0, -600.0, 0.0):
        ref = rs.fold_cfv_py(valid, combo_cards, reach, sg, None)
        got = core_fold_cfv(valid, combo_cards, reach, sg, None)
        assert np.array_equal(got, ref)
        assert np.array_equal(np.signbit(got), np.signbit(ref))


def test_wiring_rebinds_under_flag():
    """When ``PLURIBUS_CORE_KERNELS`` includes ``showdown``, ``range_showdown.fold_cfv``
    is the core kernel; otherwise the numpy reference."""
    import os
    from poker_ai._core.flags import kernel_enabled

    if kernel_enabled("showdown"):
        assert rs.fold_cfv.__module__ == "poker_ai._core._showdown"
    else:
        assert rs.fold_cfv is rs.fold_cfv_py
