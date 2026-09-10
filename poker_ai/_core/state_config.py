"""One entry point for configuring the compiled ``FastState`` engine.

``_state.configure()`` installs the action alphabet and raise grid **process-wide** and
is once-only (guarded by ``is_configured()``), while the raise grid can now legitimately
differ between callers: real-time search may narrow it (see
``environment.poker_env.SEARCH_RAISE_SIZES_BY_STAGE``) while training must always use the
canonical one.  Left unchecked that is a silent-corruption path in both directions — a
search-configured trim would leak into a training core that guards on ``is_configured()``,
and a training-configured full grid would make the compiled walk enumerate raise sizes the
Python walk no longer offers.

So every caller goes through :func:`ensure_state_configured`, which remembers the grid it
installed and raises :class:`CoreGridMismatch` if a later caller wants a different one.
Failing loudly is the whole point: the divergence it prevents is invisible at runtime and
would show up only as a subtly wrong strategy.
"""

from __future__ import annotations

_CONFIGURED_KEY = None


class CoreGridMismatch(RuntimeError):
    """A second, different raise grid was requested for an already-configured core."""


def _key(raise_sizes_by_stage) -> tuple:
    return tuple(sorted(
        (str(stage), tuple(tuple(float(f) for f in cell) for cell in levels))
        for stage, levels in raise_sizes_by_stage.items()
    ))


def configured_key():
    """The grid this process installed, or ``None`` if the core is unconfigured here."""
    return _CONFIGURED_KEY


def ensure_state_configured(raise_sizes_by_stage) -> None:
    """Configure the compiled state engine with ``raise_sizes_by_stage``, once.

    Raises :class:`CoreGridMismatch` when the core is already configured with a different
    grid — never silently reuses it, because the caller would then walk a different game
    than the ``PokerEnv`` it built its state from.
    """
    global _CONFIGURED_KEY
    from poker_ai._core import _state as _cystate
    from environment.poker_env import (
        _ACTION_BYTE,
        _STAGE_ID,
        MAX_RAISES_PER_ROUND,
        ALL_IN_ALLOWED_BY_STAGE,
        CALL_ALLOWED_BY_STAGE,
        RAISE_SIZES_BY_STAGE,
    )

    want = _key(raise_sizes_by_stage)
    canonical = _key(RAISE_SIZES_BY_STAGE)
    if _cystate.is_configured():
        if _CONFIGURED_KEY is None:
            # Configured by a path that bypassed this helper (a test fixture calling
            # ``_state.configure`` directly, say).  The core exposes no getter, so its grid
            # cannot be read back — but refusing outright would break every legitimate
            # direct caller while protecting nothing.  Refuse only in the direction that
            # can actually corrupt: asking for a NARROWED grid against a core whose grid is
            # unverifiable.  A canonical request is what every direct caller in this repo
            # uses, so it is allowed through and recorded.
            if want != canonical:
                raise CoreGridMismatch(
                    "The compiled state core was already configured by a path that bypassed "
                    "ensure_state_configured(), so its raise grid cannot be verified — and "
                    "this caller wants a NARROWED grid. Route that configure() through "
                    "poker_ai._core.state_config, or run without the search-grid trim."
                )
            _CONFIGURED_KEY = want
            return
        if _CONFIGURED_KEY != want:
            raise CoreGridMismatch(
                "The compiled state core is already configured with a DIFFERENT raise "
                "grid in this process. It is process-wide and once-only, so search (which "
                "may narrow the grid) and training (which must not) cannot share a "
                "process. Run them separately, or pass --no-trim-search-grid so search "
                "uses the canonical grid."
            )
        return

    _cystate.configure(
        _STAGE_ID, _ACTION_BYTE, raise_sizes_by_stage, MAX_RAISES_PER_ROUND,
        CALL_ALLOWED_BY_STAGE, ALL_IN_ALLOWED_BY_STAGE,
    )
    _CONFIGURED_KEY = want


def grid_for_env(env):
    """The raise grid ``env`` actually plays: its search trim, else the canonical grid."""
    from environment.poker_env import RAISE_SIZES_BY_STAGE

    return getattr(env, "_search_raise_grid", None) or RAISE_SIZES_BY_STAGE
