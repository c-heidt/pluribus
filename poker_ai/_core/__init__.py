"""Compiled Cython core for the MCCFR blueprint hot loop.

This is a single, deliberately cross-cutting acceleration layer: its kernels
mirror functions from several *different* pure-Python packages (``environment``
and ``poker_ai``), and they are kept flat in one package — rather than split to
follow the Python tree — so that Phase 3's in-core ``_traverse`` can ``cimport``
them at the C level and call them with zero Python overhead between nodes (the
same rationale behind, e.g., ``pandas/_libs``).  Each kernel names the module it
ports in its own docstring; the map is collected here for reference:

===========================  ==================================================
Kernel                       Pure-Python oracle (source of truth)
===========================  ==================================================
``_regret.pyx``              ``poker_ai.blueprint.tree_utils`` regret matching
``_infoset.pyx``             ``environment.poker_env.encode_info_set``
``_index.pyx``               ``poker_ai.tables.index`` hash + ``ShmIndexCache``
``_eval.pyx``                ``environment.evaluator`` (_five/_six/_seven)
``_settle.pyx``              ``environment.pot.Pot.compute_utility``
``_showdown.pyx``            ``environment.range_showdown`` CFV kernels
``_runout.pyx``              ``environment.poker_env`` ``_settle_runout`` (all-in
                             board-average) + ``_settle_traverser`` (per-combo)
``_probe.pyx``               memoryview-seam build probe (Phase 0)
===========================  ==================================================

The compiled extension is **optional**: every code path it accelerates keeps its
pure-Python reference as the source of truth, so a build without a C toolchain
(or with ``POKER_AI_NO_EXT=1``) still yields a working install.
:data:`CORE_AVAILABLE` reports whether the extension compiled and imported;
callers gate on it and fall back to Python when it is ``False``.

The compiled core is driven by **one master switch per pipeline** —
``PLURIBUS_SEARCH_CORE`` (real-time search) and ``PLURIBUS_CFR_CORE`` (blueprint
training) — each of which lights the compiled walk *and* every byte-identical
kernel that pipeline uses (see :mod:`poker_ai._core.flags`).
``PLURIBUS_CORE_KERNELS`` is a developer-only per-kernel override for parity A/B.
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
