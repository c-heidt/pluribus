"""Action translation for runtime -> abstraction mapping.

The bot's search tree and policies are defined over a discrete set of
abstract actions per street (``fold``, ``call``, ``raise:<f>``,
``all_in``); the runtime opponent acts in chips.  This module is the
sole place where chip amounts and fraction strings convert into one
another.  It also decides when an observed raise is far enough from
the abstraction grid to warrant injecting a new ``raise:<f_obs>``
action into the env's overlay (§6.1).

Pure functions: nothing here mutates the env, the tracker, or the
policy.  Off-tree injection is performed by the *caller* via
``env.inject_action(action_str)``; this module returns only strings.

See §6.3 of ``docs/subgame_solving.md`` for the design rationale.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import List, Tuple

from environment.poker_env import MAX_RAISES_PER_ROUND, PokerEnv


_OFF_TREE_DECIMALS = 4


class Classification(Enum):
    """Result of mapping an observed chip raise onto the abstraction."""

    ON_TREE = auto()
    NEAR_TREE = auto()
    OFF_TREE = auto()


def canonical_raise_fractions(env: PokerEnv) -> List[float]:
    """Currently-playable raise fractions for ``env``'s actor.

    Mirrors the gating that ``env.legal_actions`` applies before it
    calls ``_get_available_raise_sizes``: an inactive player, a call
    amount that meets or exceeds the stack, or having already hit
    ``MAX_RAISES_PER_ROUND`` all yield an empty list.
    Stack-clamping and min-raise enforcement come from
    ``_get_available_raise_sizes`` itself.
    """
    if not env.current_player.is_active:
        return []
    biggest_bet = max(p.n_bet_chips for p in env.players)
    n_chips_to_call = biggest_bet - env.current_player.n_bet_chips
    if n_chips_to_call >= env.current_player.n_chips:
        return []
    if env._n_raises >= MAX_RAISES_PER_ROUND:
        return []
    raise_strs = env._get_available_raise_sizes()
    return [
        float(s.split(":", 1)[1]) for s in raise_strs if s.startswith("raise:")
    ]


def classify_observed(
    env_before: PokerEnv,
    chip_amount: int,
    tol: float = 0.15,
) -> Tuple[Classification, str]:
    """Classify an observed raise of ``chip_amount`` chips.

    ``chip_amount`` is the total chips the actor added to the pot at
    this decision, matching
    :meth:`environment.poker_env.PokerEnv._compute_raise_chip_amount`'s
    ``n_chips_to_add`` convention.  Fold / call / check routing is the
    caller's job; this function handles raises (including ``all_in``).

    Returns
    -------
    (Classification, str)
        - ``ON_TREE``: ``chip_amount`` equals ``all_in`` (full
          remaining stack) or matches some canonical
          ``raise:<f>`` after the env's min-raise clamping.
        - ``NEAR_TREE``: relative pot-fraction distance to the nearest
          canonical fraction is ``<= tol`` — the snapped
          ``raise:<f>`` string is returned.
        - ``OFF_TREE``: distance exceeds ``tol``, or no canonical
          fractions are playable; returned string is
          ``raise:<f_obs>`` with ``f_obs = chip_amount / pot_size``
          rounded to a stable precision.

    All-in is *always* ON_TREE per §6.3.
    """
    stack = env_before.current_player.n_chips
    if chip_amount == stack:
        return Classification.ON_TREE, "all_in"

    canonical = canonical_raise_fractions(env_before)
    for f in canonical:
        clamped = env_before._compute_raise_chip_amount(f, enforce_minimum=True)
        if clamped == chip_amount:
            return Classification.ON_TREE, f"raise:{f}"

    pot_size = env_before.pot_size
    f_obs = chip_amount / pot_size

    if canonical:
        f_near = min(canonical, key=lambda f: abs(f - f_obs))
        dist = abs(f_obs - f_near) / f_near
        if dist <= tol:
            return Classification.NEAR_TREE, f"raise:{f_near}"

    return Classification.OFF_TREE, f"raise:{round(f_obs, _OFF_TREE_DECIMALS)}"


def abstract_to_chips(env: PokerEnv, action: str) -> int:
    """Inverse mapping from an abstract action string to chips-to-add.

    - ``"fold"``    -> ``0``
    - ``"call"``    -> ``biggest_bet - actor.n_bet_chips``
    - ``"all_in"``  -> ``actor.n_chips`` (full remaining stack)
    - ``"raise:<f>"`` -> ``env._compute_raise_chip_amount(f, True)``

    Raises ``ValueError`` on unknown action strings.
    """
    if action == "fold":
        return 0
    if action == "call":
        biggest_bet = max(p.n_bet_chips for p in env.players)
        return biggest_bet - env.current_player.n_bet_chips
    if action == "all_in":
        return env.current_player.n_chips
    if action.startswith("raise:"):
        try:
            fraction = float(action.split(":", 1)[1])
        except (ValueError, IndexError):
            raise ValueError(
                f"abstract_to_chips: unparseable raise fraction in {action!r}"
            )
        return env._compute_raise_chip_amount(fraction, enforce_minimum=True)
    raise ValueError(f"abstract_to_chips: unknown action {action!r}")
