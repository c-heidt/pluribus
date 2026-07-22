"""Phase-0 build skeleton: the compiled core builds and its memoryview seam works.

The compiled extension (``poker_ai._core``) is optional — a pure-Python install
with no C toolchain is valid — so these tests **skip** when the extension is not
built rather than fail.  Where it *is* built (the development / cluster
toolchain), they assert the one thing the whole Tier-1 architecture bets on:
that a Cython typed memoryview binds **zero-copy** to the exact numpy array
layouts the real core reads on its hot path — a C-contiguous ``int32`` matrix
(the shm regret/strategy chunk tables) and a ``uint64`` vector (the index
cache) — and agrees with numpy byte-for-byte.
"""

import os

import numpy as np
import pytest

from poker_ai import _core

# Opt-in hard-failure guard for CI / cluster builds that must ship the compiled
# core.  A missing *or broken* extension both surface as ``CORE_AVAILABLE=False``
# (a half-linked / ABI-skewed .so also raises ImportError), which would otherwise
# pass green-by-skip below.  When ``POKER_AI_REQUIRE_EXT`` is set we fail loudly
# at collection time instead — this runs even when the extension is absent,
# unlike the tests below which the module-level skip mark suppresses.
if os.environ.get("POKER_AI_REQUIRE_EXT") and not _core.CORE_AVAILABLE:
    raise RuntimeError(
        "POKER_AI_REQUIRE_EXT is set but poker_ai._core did not import "
        "(compiled extension missing or broken). Rebuild with "
        "`pip install -e .` or `python setup.py build_ext --inplace`, "
        "or unset POKER_AI_REQUIRE_EXT to allow the pure-Python fallback."
    )

pytestmark = pytest.mark.skipif(
    not _core.CORE_AVAILABLE,
    reason="compiled core extension not built (pure-Python install)",
)


def test_extension_imported():
    """The compiled probe module loaded."""
    assert _core.core_build_ok() is True


def test_self_check_passes():
    """The built-in round-trip of both memoryview bindings succeeds."""
    assert _core.self_check() is True


def test_int32_matrix_seam_matches_numpy():
    """Zero-copy int32-matrix sum agrees with numpy over random C-contiguous data."""
    from poker_ai._core._probe import sum_int32_matrix

    rng = np.random.RandomState(0)
    for shape in [(1, 1), (3, 4), (128, 7), (4096, 6)]:
        mat = rng.randint(-1000, 1000, size=shape).astype(np.int32)
        assert mat.flags["C_CONTIGUOUS"]
        assert sum_int32_matrix(mat) == int(mat.sum())


def test_uint64_slot_seam_matches_numpy():
    """Zero-copy uint64-slot read agrees with numpy, including near the sentinel."""
    from poker_ai._core._probe import read_uint64_slot

    rows = np.array([0, 1, 2 ** 63, 2 ** 64 - 1], dtype=np.uint64)
    for slot in range(len(rows)):
        assert read_uint64_slot(rows, slot) == int(rows[slot])
