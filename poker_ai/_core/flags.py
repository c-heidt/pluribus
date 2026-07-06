"""Runtime enable flags for the compiled-core kernels (pure Python, always importable).

Each compiled kernel ships behind an opt-in flag so it can be A/B'd against its
pure-Python reference independently and rolled out one at a time.  Selection is a
single env var, ``PLURIBUS_CORE_KERNELS`` — a comma-separated list of kernel
names, or ``all``:

    PLURIBUS_CORE_KERNELS=regret_match         # just one kernel
    PLURIBUS_CORE_KERNELS=regret_match,evaluator
    PLURIBUS_CORE_KERNELS=all                  # every built kernel

A kernel is used only when it is both *enabled* here and *available* (the
extension built — ``poker_ai._core.CORE_AVAILABLE``); otherwise the pure-Python
reference runs, so an extension-less install is always functional.
"""

import os


def kernel_enabled(name: str) -> bool:
    """Return ``True`` iff compiled kernel ``name`` is enabled via the env var.

    Read fresh from the environment on every call so tests can toggle kernels
    with ``monkeypatch.setenv`` without re-importing; the cost is a dict lookup
    plus a small split, negligible next to a traversal.
    """
    value = os.environ.get("PLURIBUS_CORE_KERNELS", "")
    if not value:
        return False
    if value == "all":
        return True
    return name in {token.strip() for token in value.split(",")}


def search_core_enabled() -> bool:
    """Return ``True`` iff the compiled real-time *search* walk is enabled.

    Top-level dispatch flag ``PLURIBUS_SEARCH_CORE`` for the subgame solver's
    compiled core — the search-side analogue of ``PLURIBUS_CFR_CORE`` (blueprint
    training).  It gates the recursive *walk* ports (the vector ``_walk`` and the
    MCCFR ``_traverse``/leaf rollout, plan Phases 3-4), which additionally require
    the extension to be built (``poker_ai._core.CORE_AVAILABLE``); the callers own
    that ``and`` so an extension-less install always falls back to Python.

    The per-node *leaf kernels* (``showdown`` / ``regret_match_matrix`` / ``runout``,
    Phases 1-2) gate independently through :func:`kernel_enabled` — they are pure
    byte-identical swaps and can be A/B'd one at a time without turning on the walk.

    Read fresh from the environment on every call so tests can toggle it with
    ``monkeypatch.setenv`` without re-importing.
    """
    return os.environ.get("PLURIBUS_SEARCH_CORE", "") == "1"
