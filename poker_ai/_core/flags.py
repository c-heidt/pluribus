"""Runtime enable flags for the compiled-core kernels (pure Python, always importable).

There is **one master switch per pipeline**, and that is all an operator sets:

    PLURIBUS_SEARCH_CORE=1     # real-time search: the compiled walk + every search kernel
    PLURIBUS_CFR_CORE=1        # blueprint training: the compiled walk + every training kernel

Either master flag, when on, turns on the compiled *walk* for its pipeline **and**
every byte-identical leaf/terminal kernel that pipeline uses (settlement, showdown,
regret matching, evaluator, …).  A kernel is used only when it is also *available*
(the extension built — ``poker_ai._core.CORE_AVAILABLE``); otherwise the pure-Python
reference runs, so an extension-less install is always functional (each call site
owns that ``and``).

``PLURIBUS_CORE_KERNELS`` is a **developer-only override** for differential / parity
A/B — a comma-separated kernel allow-list, or ``all``.  It is normally unset; when
set it *wins* over the master flags (so a single kernel can be exercised in isolation
against its pure-Python oracle without turning on a whole pipeline):

    PLURIBUS_CORE_KERNELS=showdown         # just that kernel, master flags ignored
    PLURIBUS_CORE_KERNELS=regret_match,evaluator
    PLURIBUS_CORE_KERNELS=all              # every built kernel
"""

import os


def search_core_enabled() -> bool:
    """Return ``True`` iff the compiled real-time *search* core is enabled.

    The single operator switch (``PLURIBUS_SEARCH_CORE=1``) for the subgame solver:
    it gates the recursive *walk* ports (the vector ``_walk`` and the MCCFR walk /
    leaf rollout on the ``FastState`` engine) **and**, via :func:`kernel_enabled`,
    every search leaf/terminal kernel.  The walk additionally requires the extension
    to be built (``poker_ai._core.CORE_AVAILABLE``); the callers own that ``and`` so
    an extension-less install always falls back to Python.

    Read fresh from the environment on every call so tests can toggle it with
    ``monkeypatch.setenv`` without re-importing.
    """
    return os.environ.get("PLURIBUS_SEARCH_CORE", "") == "1"


def train_core_enabled() -> bool:
    """Return ``True`` iff the compiled blueprint *training* core is enabled.

    The single operator switch (``PLURIBUS_CFR_CORE=1``) for the training pipeline —
    the training-side analogue of :func:`search_core_enabled`.  It gates the compiled
    CFR walk (:mod:`poker_ai.blueprint.core_runner`, which layers its own bias/index
    preconditions on top) **and**, via :func:`kernel_enabled`, every training kernel.

    Read fresh each call (same rationale as above).
    """
    return os.environ.get("PLURIBUS_CFR_CORE", "") == "1"


def kernel_enabled(name: str) -> bool:
    """Return ``True`` iff compiled kernel ``name`` should be used.

    Normal operation: a kernel is on whenever *either* pipeline master flag is on
    (one flag lights up the whole pipeline, walk + kernels).  The developer override
    ``PLURIBUS_CORE_KERNELS``, when set, takes precedence and is the explicit
    allow-list (``all`` or a comma-separated set) — so a single kernel can be A/B'd
    against its pure-Python oracle in isolation, without turning on any walk.

    Read fresh from the environment on every call so tests can toggle kernels with
    ``monkeypatch.setenv`` without re-importing; the cost is a dict lookup plus a
    small split, negligible next to a traversal.
    """
    override = os.environ.get("PLURIBUS_CORE_KERNELS", "")
    if override:
        if override == "all":
            return True
        return name in {token.strip() for token in override.split(",")}
    return search_core_enabled() or train_core_enabled()
