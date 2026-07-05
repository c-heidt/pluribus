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
