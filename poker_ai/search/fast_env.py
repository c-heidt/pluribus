"""Compiled-engine adapter for the vector-regime walk (search Cython core, §6.5).

Phase 3 drives the vector CFR walk on the compiled :class:`poker_ai._core._state.FastState`
betting engine (Phase-2-proven byte-identical to :class:`~environment.poker_env.PokerEnv`)
so make/undo collapses out of Python.  :class:`FastEnvAdapter` presents the *exact*
read-only + step/undo surface :meth:`poker_ai.search.vector._VectorSolver._walk` /
``_child`` consume, delegating each call to the ``FastState`` — so the **unchanged**
Python walk produces byte-identical ``vregret`` / ``vstrat`` tables whether it walks a
``PokerEnv`` or this adapter (proven by the record/replay differential + the golden digest).

**Scope — canonical histories only.**  ``FastState``'s history is a compact byte-code
stream with no code for off-tree *injected* raise strings, so :attr:`FastState.public_key`
is exact only for on-tree histories.  The solver therefore uses this adapter **only when
the search has no off-tree injections** (``root_env._extra_legal_actions`` empty); a
re-search that injected an off-tree size falls back to the Python ``PokerEnv`` walk.  This
keeps the (dominant) fresh-search path in-core without an overlay merge, and never risks
byte-identity on the rare injected re-search.  See :func:`build_fast_walk_env`.
"""

from __future__ import annotations

from typing import List, Optional


class _SeatView:
    """Minimal ``env.players[s]`` stand-in exposing the live ``is_active`` flag.

    ``_walk`` / ``_terminal_needs_river`` read ``env.players[s].is_active`` at
    terminals; the flag mutates with make/undo, so this reads it straight off the
    ``FastState`` each access (never caches a stale value).
    """

    __slots__ = ("_fast", "_seat")

    def __init__(self, fast, seat: int) -> None:
        self._fast = fast
        self._seat = seat

    @property
    def is_active(self) -> bool:
        return bool(self._fast.is_seat_active(self._seat))


class FastEnvAdapter:
    """``PokerEnv``-shaped facade over a :class:`FastState` for the vector walk.

    Exposes only the members ``_VectorSolver._walk`` / ``_child`` /
    ``_terminal_needs_river`` touch — every one a thin delegate to the compiled
    engine — so the Python walk runs unchanged.  ``combo_cards`` is threaded to
    :meth:`FastState.vector_payout` (the engine owns everything else).
    """

    __slots__ = ("_fast", "_combo_cards", "_players")

    def __init__(self, fast, combo_cards) -> None:
        self._fast = fast
        self._combo_cards = combo_cards
        self._players: List[_SeatView] = [
            _SeatView(fast, s) for s in range(fast.n_players)
        ]

    # --- read-only public-state surface (byte-identical to PokerEnv, Phase 3a) ---
    @property
    def player_i(self) -> int:
        return self._fast.player_i

    @property
    def legal_actions(self):
        # Canonical set only (no overlay — this path runs only when the overlay is
        # empty; see build_fast_walk_env).  The walk filters ``None`` itself.
        return self._fast.legal_actions()

    @property
    def public_key(self):
        return self._fast.public_key

    @property
    def betting_round(self) -> int:
        return self._fast.betting_round

    @property
    def is_terminal(self) -> bool:
        return self._fast.is_terminal

    @property
    def n_raises_this_round(self) -> int:
        return self._fast.n_raises_this_round

    @property
    def n_players_started_round(self) -> int:
        return self._fast.n_players_started_round

    @property
    def terminal_board_len(self) -> Optional[int]:
        return self._fast.terminal_board_len

    @property
    def players(self) -> List[_SeatView]:
        return self._players

    # --- make/undo + terminal settlement (Phase 2 engine + Phase 3b payout) ---
    def step_in_place(self, action: str, settle_winners: bool = True):
        # ``settle_winners`` is a PokerEnv concept (concrete winner settlement);
        # the vector regime always passes ``False`` and the compiled engine never
        # settles concretely — so the kwarg is accepted and ignored.
        return self._fast.step_in_place(action)

    def undo(self, token) -> None:
        self._fast.undo(token)

    def vector_payout(self, seat: int, opp_seat: int, opp_reach, runout=None):
        return self._fast.vector_payout(
            seat, opp_seat, opp_reach, runout, self._combo_cards
        )


class _McSeatView:
    """``env.players[s]`` stand-in for the MCCFR walk: live ``is_active`` + the
    seat's concrete ``cards`` (both read straight off the FastState each access, so
    make/undo and per-iteration reseat are always reflected)."""

    __slots__ = ("_fast", "_seat")

    def __init__(self, fast, seat: int) -> None:
        self._fast = fast
        self._seat = seat

    @property
    def is_active(self) -> bool:
        return bool(self._fast.is_seat_active(self._seat))

    @property
    def cards(self):
        return self._fast.hole_cards(self._seat)


class FastMCCFRAdapter:
    """``PokerEnv``-shaped facade over a :class:`FastState` for the MCCFR walk.

    Richer than :class:`FastEnvAdapter` (the vector regime is leaf-free and settles
    range-vs-range): the traverser-vectorized MCCFR walk also reads per-seat holes,
    the board *prefix* (``community_cards``), the shared ``combo_cards`` / LUT, and
    settles concretely (:meth:`FastState.vector_payout_concrete`) — and its
    depth-limit leaf needs a frontier it can clone and draw boards on.  Every member
    is a thin delegate to the compiled engine (or a threaded solver constant), so the
    unchanged Python ``_vwalk`` / ``_vchild`` / ``_vmeta_game`` run on it; the leaf
    (:func:`poker_ai.search.leaf_fast.continuation_value_vector_fast`) detects the
    exposed ``_fast`` and clones it rather than rebuilding from a ``PokerEnv``.
    """

    __slots__ = ("_fast", "combo_cards", "n_combos", "card_info_lut", "_players")

    def __init__(self, fast, combo_cards, card_info_lut) -> None:
        self._fast = fast
        self.combo_cards = combo_cards
        self.n_combos = int(combo_cards.shape[0])
        self.card_info_lut = card_info_lut
        self._players = [_McSeatView(fast, s) for s in range(fast.n_players)]

    # --- read-only public-state surface (byte-identical to PokerEnv) ---
    @property
    def player_i(self) -> int:
        return self._fast.player_i

    @property
    def legal_actions(self):
        return self._fast.legal_actions()

    @property
    def public_key(self):
        return self._fast.public_key

    @property
    def betting_round(self) -> int:
        return self._fast.betting_round

    @property
    def is_terminal(self) -> bool:
        return self._fast.is_terminal

    @property
    def n_raises_this_round(self) -> int:
        return self._fast.n_raises_this_round

    @property
    def n_players(self) -> int:
        return self._fast.n_players

    @property
    def n_players_started_round(self) -> int:
        return self._fast.n_players_started_round

    @property
    def terminal_board_len(self) -> Optional[int]:
        return self._fast.terminal_board_len

    @property
    def community_cards(self):
        return self._fast.community_cards()

    @property
    def players(self) -> List[_McSeatView]:
        return self._players

    # --- make/undo + concrete terminal settlement ---
    def step_in_place(self, action: str, settle_winners: bool = True):
        return self._fast.step_in_place(action)

    def undo(self, token) -> None:
        self._fast.undo(token)

    def vector_payout_concrete(self, seat: int):
        return self._fast.vector_payout_concrete(seat, self.combo_cards)


def build_fast_walk_env(root_env):
    """Return a :class:`FastEnvAdapter` over ``root_env``, or ``None`` to fall back.

    Fall back (return ``None``, → the caller walks the ``PokerEnv``) when: the
    compiled core is unavailable; the search injected an off-tree action (overlay
    non-empty → histories not representable in ``FastState``'s byte-codes); or the
    ``FastState`` cannot be constructed (e.g. missing LUT).  Never raises — the
    opt-in only ever *accelerates* the fresh-search path, never changes results.
    """
    overlay = getattr(root_env, "_extra_legal_actions", None)
    if overlay:  # any off-tree injection anywhere in the tree
        return None
    try:
        from poker_ai._core import CORE_AVAILABLE
        if not CORE_AVAILABLE:
            return None
        from poker_ai._core import _state as _cystate
        if not _cystate.is_configured():
            # Dump the live alphabet + raise grid once per process (idempotent) —
            # never a hard-coded copy (anti-drift, same as the blueprint runner).
            from environment.poker_env import (
                _ACTION_BYTE,
                _STAGE_ID,
                RAISE_SIZES_BY_STAGE,
                MAX_RAISES_PER_ROUND,
            )
            _cystate.configure(
                _STAGE_ID, _ACTION_BYTE, RAISE_SIZES_BY_STAGE, MAX_RAISES_PER_ROUND
            )
        fast = _cystate.FastState.from_poker_env(root_env)
        return FastEnvAdapter(fast, root_env.combo_cards)
    except Exception:
        return None


def build_fast_mccfr_env(root_env):
    """Return a :class:`FastMCCFRAdapter` over ``root_env``, or ``None`` to fall back.

    Called **per MCCFR iteration**: the traverser-vectorized walk re-holes the root
    each traversal (in-place ``reseat_private_cards`` on the ``PokerEnv``), so the
    ``FastState`` is rebuilt from the freshly-reseated ``root_env`` (a cheap O(n)
    copy) rather than mutated.  Falls back (→ the ``PokerEnv`` walk) on the same
    conditions as :func:`build_fast_walk_env` — core unavailable, off-tree overlay,
    or construction failure; never raises.
    """
    overlay = getattr(root_env, "_extra_legal_actions", None)
    if overlay:
        return None
    try:
        from poker_ai._core import CORE_AVAILABLE
        if not CORE_AVAILABLE:
            return None
        from poker_ai._core import _state as _cystate
        if not _cystate.is_configured():
            from environment.poker_env import (
                _ACTION_BYTE,
                _STAGE_ID,
                RAISE_SIZES_BY_STAGE,
                MAX_RAISES_PER_ROUND,
            )
            _cystate.configure(
                _STAGE_ID, _ACTION_BYTE, RAISE_SIZES_BY_STAGE, MAX_RAISES_PER_ROUND
            )
        fast = _cystate.FastState.from_poker_env(root_env)
        return FastMCCFRAdapter(fast, root_env.combo_cards, root_env.card_info_lut)
    except Exception:
        return None
