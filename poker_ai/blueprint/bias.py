"""Action-class bias hooks for biased MCCFR training.

A biased run modifies the terminal payoff with a bonus proportional
to the number of decisions on the trajectory whose action belongs to
the configured class (fold / call / raise).  Regret updates therefore
see a payoff that already encodes the trajectory's bias contribution,
which shifts the equilibrium toward the biased class and causes the
rest of the tree to adapt to that tendency.

This module owns:

- The :data:`BiasClass` literal and the action-string classification
  used at every recursive step when bias is enabled.
- The constant terminal-bonus formula
  ``bonus = bias_magnitude * count_biased_actions_on_trajectory``.

Both helpers are deliberately allocation-free; they sit on the hot
path of every CFR traversal in a biased run.
"""

from typing_extensions import Literal

BiasClass = Literal["none", "fold", "call", "raise"]
"""Bias mode for a CFR run.

``"none"`` is the standard base-blueprint training path with no
modified utility; the other three values bias the corresponding
action class.
"""


def is_biased(action: str, bias: BiasClass) -> bool:
    """Return ``True`` iff *action* belongs to the *bias* action class.

    Action-string format follows
    :func:`environment.poker_env.PokerEnv.get_canonical_actions`:
    ``"fold"``, ``"call"`` / ``"check"``, anything starting with
    ``"raise"`` (e.g. ``"raise:0.5"``, ``"raise:1.0"``) and
    ``"all_in"``.

    Parameters
    ----------
    action : str
        Action string emitted by the poker environment.
    bias : BiasClass
        The configured bias class.  ``"none"`` always returns
        ``False``; biased runs short-circuit this function only after
        checking ``bias != "none"`` at the call site.

    Returns
    -------
    bool
        Whether *action* counts toward the trajectory bias.
    """
    if bias == "fold":
        return action == "fold"
    if bias == "call":
        return action == "call" or action == "check"
    if bias == "raise":
        return action.startswith("raise") or action == "all_in"
    return False
