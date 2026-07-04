"""Compiled Cython core for the MCCFR blueprint hot loop.

The compiled extension is **optional**: every code path it accelerates has a
pure-Python reference implementation that stays the source of truth, so a build
without a C toolchain (or with ``POKER_AI_NO_EXT=1``) still yields a working
install.  :data:`CORE_AVAILABLE` reports whether the extension compiled and
imported; callers gate on it and fall back to Python when it is ``False``.

Nothing here is imported on a training hot path yet — Phase 3 wires the core in
behind the ``PLURIBUS_CFR_CORE`` flag.  Until then this package only carries the
Phase-0 build probe (:mod:`poker_ai._core._probe`).
"""

# Re-exported (when built) so callers use ``poker_ai._core.core_build_ok()``
# without reaching into the private ``_probe`` extension module.  ``__all__``
# tracks availability so ``from poker_ai._core import *`` never names a symbol
# that a pure-Python (extension-less) install does not define.
try:
    from poker_ai._core._probe import (  # noqa: F401  (re-exported public API)
        core_build_ok,
        self_check,
    )

    CORE_AVAILABLE = True
    __all__ = ["CORE_AVAILABLE", "core_build_ok", "self_check"]
except ImportError:
    CORE_AVAILABLE = False
    __all__ = ["CORE_AVAILABLE"]
