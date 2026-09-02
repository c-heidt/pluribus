"""Grid-independent helpers for driving a hand under the action abstraction.

``environment.poker_env``'s abstraction is a per-stage table
(:data:`~environment.poker_env.RAISE_SIZES_BY_STAGE`), cut into a
``"first_raise"`` cell and a ``"subsequent_raise"`` cell.  Which sizes exist is
a *tuning* decision that changes; tests must not encode it.

Everything here derives what it needs from the live tables or from
``env.legal_actions``, so re-cutting the grid never edits a test:

* :func:`passive_action` / :func:`advance_to_round` — walk a hand forward.
* :func:`smallest_raise` / :func:`largest_raise` — a raise without naming one.
* :func:`shove_prelude` / :func:`raise_until_shove_legal` — reach a node where
  a voluntary all-in is legal (deep stacks make it a raise, not a call).
* :func:`bracketing_sample` / :func:`level_exclusive_fraction` — inputs for the
  pseudo-harmonic translation tests, read off the live grid.
"""

import copy
from typing import List, Optional, Tuple

from environment.poker_env import RAISE_SIZES_BY_STAGE, PokerEnv

_STAGE_FOR_ROUND = {0: "pre_flop", 1: "flop", 2: "turn", 3: "river"}


def _legal(env) -> List[str]:
    return [a for a in env.legal_actions if a is not None]


def _raises(env) -> List[str]:
    return sorted(
        (a for a in _legal(env) if a.startswith("raise:")),
        key=lambda a: float(a.split(":", 1)[1]),
    )


# ---------------------------------------------------------------------------
# Driving a hand
# ---------------------------------------------------------------------------


def passive_action(env) -> str:
    """The cheapest way to keep the hand alive without folding.

    ``call`` (a check when nothing is owed) when the abstraction offers it,
    otherwise the smallest legal raise, otherwise a shove.  Never ``fold``
    unless nothing else is legal — a plain ``"call"`` at a level that has no
    call would be remapped to a fold and silently end the hand.
    """
    legal = _legal(env)
    if "call" in legal:
        return "call"
    raises = _raises(env)
    if raises:
        return raises[0]
    if "all_in" in legal:
        return "all_in"
    return legal[0]


def smallest_raise(env) -> Optional[str]:
    """Smallest legal ``raise:<f>`` at the current node, or ``None``."""
    raises = _raises(env)
    return raises[0] if raises else None


def largest_raise(env) -> Optional[str]:
    """Largest legal ``raise:<f>`` at the current node, or ``None``."""
    raises = _raises(env)
    return raises[-1] if raises else None


def advance_to_round(env, betting_round: int, max_steps: int = 60):
    """Play passive actions until ``env.betting_round == betting_round``.

    Returns ``env`` (mutated in place).  Raises ``AssertionError`` if the hand
    ends or stalls first, so a test never silently asserts on the wrong street.
    """
    steps = 0
    while not env.is_terminal and env.betting_round < betting_round:
        env.step_in_place(passive_action(env))
        steps += 1
        assert steps < max_steps, "hand did not reach the requested round"
    assert not env.is_terminal, "hand ended before the requested round"
    assert env.betting_round == betting_round
    return env


def shove_prelude(env, max_steps: int = 6) -> List[str]:
    """Actions that walk ``env`` to a node where ``"all_in"`` is legal.

    Computed on a copy — ``env`` is not stepped — so a caller driving two
    engines in lockstep (the ``FastState`` differentials) can replay the same
    list into both.  Empty wherever the shove is already offered.
    """
    probe = copy.deepcopy(env)
    out: List[str] = []
    while "all_in" not in probe.legal_actions and not probe.is_terminal:
        action = smallest_raise(probe) or passive_action(probe)
        probe.step_in_place(action)
        out.append(action)
        assert len(out) < max_steps, "no shove became legal"
    return out


def raise_until_shove_legal(env, max_steps: int = 6):
    """Step ``env`` through :func:`shove_prelude`; returns ``env``."""
    for action in shove_prelude(env, max_steps):
        env.step_in_place(action)
    return env


# ---------------------------------------------------------------------------
# Reading the grid (for the action-translation tests)
# ---------------------------------------------------------------------------


def grid(stage_or_round, level: int = 0) -> List[float]:
    """Sorted abstraction fractions for a ``(stage, raise level)`` cell."""
    stage = _STAGE_FOR_ROUND.get(stage_or_round, stage_or_round)
    return PokerEnv._abstraction_fractions(stage, level)


def bracketing_sample(
    stage_or_round, level: int = 0, side: str = "mid"
) -> Tuple[float, float, float, float]:
    """An off-tree fraction inside the first bracket of a live grid cell.

    Returns ``(a, x, b, expected)`` where ``a < x < b`` are grid neighbours and
    ``expected`` is the deterministic pseudo-harmonic image of ``x`` (``a`` iff
    ``P_A >= 0.5``).  ``side`` picks where in the bracket ``x`` sits relative to
    the ``P_A = 0.5`` crossing: ``"mid"`` (halfway), ``"below"`` (just under the
    crossing, so ``expected is a``) or ``"above"`` (just over, ``expected is b``).
    """
    cell = grid(stage_or_round, level)
    assert len(cell) >= 2, f"cell {stage_or_round}/{level} has no bracket"
    a, b = cell[0], cell[1]
    # P_A(x) = 0.5  =>  x* = (2b(1+a) - (b-a)) / ((b-a) + 2(1+a))
    crossing = (2 * b * (1 + a) - (b - a)) / ((b - a) + 2 * (1 + a))
    if side == "below":
        x = a + (crossing - a) * 0.5
    elif side == "above":
        x = b - (b - crossing) * 0.5
    else:
        x = (a + b) / 2
    expected = a if PokerEnv._pseudo_harmonic_prob(a, x, b) >= 0.5 else b
    return a, x, b, expected


def level_exclusive_fraction(stage_or_round, level_a: int = 0, level_b: int = 1):
    """A fraction on-tree at ``level_a`` but off-tree at ``level_b``.

    Returns ``(fraction, image_at_level_b)`` — the value that proves a raise
    index advanced, and what it canonicalises to on the other level.  ``None``
    when the two cells share every fraction (nothing to distinguish them by).
    """
    stage = _STAGE_FOR_ROUND.get(stage_or_round, stage_or_round)
    cell_a, cell_b = grid(stage, level_a), grid(stage, level_b)
    for f in cell_a:
        if f not in cell_b:
            # ``_translate_fraction`` never touches ``self`` — calling it
            # unbound keeps this helper free of a throwaway env instance.
            image = PokerEnv._translate_fraction(
                PokerEnv, f, stage, level_b, randomized=False
            )
            return f, image
    return None


def off_tree_fractions(stage_or_round, level: int = 0, count: int = 4) -> List[float]:
    """``count`` distinct positive fractions guaranteed *off* the grid cell.

    For overlay / injection tests that need sizes the abstraction does not
    contain, whatever the grid currently is.  Values stay in a range a normal
    stack can actually raise to, so ``inject_action`` accepts them.
    """
    cell = set(grid(stage_or_round, level))
    out: List[float] = []
    f = 1.0
    while len(out) < count:
        f = round(f + 0.13, 4)
        if f not in cell:
            out.append(f)
    return out
