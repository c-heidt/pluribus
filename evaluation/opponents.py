"""Opponent agents and table-composition assignment for the runner (doc §10.1).

The five non-hero seats are **blueprint-derived bots**: the unaltered blueprint
policy or one of its fold-/call-/raise-biased variants — the same inference-time
reweightings used at the MC leaves (subgame doc §4), applied via
:meth:`BlueprintPolicy.strategy(state, bias=…)`.  An opponent therefore needs no
search and no new artifact: it samples one action straight from the (biased)
blueprint σ at the current node.

Two seams the rest of the design leans on:

- The opponent is just "an agent exposing :meth:`action_probs` and :meth:`sample`",
  so a heterogeneous / stronger opponent (§10.3) drops in without touching the
  runner or the schema.
- Each opponent's action distribution is computable **exactly** (blueprint σ + the
  bias transform), which is what will later let AIVAT (§10.2) correct opponent
  actions too — hence :meth:`action_probs` is exposed, not just :meth:`sample`.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from poker_ai.search.policy import BiasClass, Policy

# `game_seats.agent_label` vocabulary for the start scope, mapped to the bias
# class each label applies to the shared blueprint (doc §10.1).  Extensible:
# a new label is one entry here (+ its policy) — the runner and schema are agnostic.
LABEL_TO_BIAS: Mapping[str, BiasClass] = {
    "bp": "none",
    "bp_fold": "fold",
    "bp_call": "call",
    "bp_raise": "raise",
}
OPPONENT_LABELS: Tuple[str, ...] = tuple(LABEL_TO_BIAS)

# Reserved label for the seat the search agent under test occupies.
HERO_LABEL = "hero"


class BlueprintOpponent:
    """A non-searching opponent: samples the (biased) blueprint σ at the node.

    ``label`` is the ``game_seats.agent_label`` value (``bp`` / ``bp_fold`` / …);
    it selects the bias class applied to the shared ``policy``.  The bot reads the
    **current actor's** state — so :meth:`action_probs` / :meth:`sample` are only
    valid when ``env.player_i == seat`` (the runner guarantees this).
    """

    def __init__(self, label: str, policy: Policy) -> None:
        if label not in LABEL_TO_BIAS:
            raise ValueError(
                f"unknown opponent label {label!r}; expected one of {OPPONENT_LABELS}"
            )
        self.label = label
        self._policy = policy
        self._bias: BiasClass = LABEL_TO_BIAS[label]

    def action_probs(self, env, seat: int) -> Tuple[List[str], np.ndarray]:
        """``(legal_actions, probs)`` for ``seat`` at the current node.

        Queries the blueprint under the seat's *actual* hole with
        ``for_blueprint=True`` (canonicalising any off-tree history, exactly as the
        leaf evaluator and the hero agent query it).  ``probs`` is a float64 vector
        aligned to ``legal_actions`` and summing to 1.
        """
        hole = tuple(int(c) for c in env.players[seat].cards)
        state = env.policy_state_for(hole, for_blueprint=True)
        legal = list(state.legal_actions)
        probs = np.asarray(self._policy.strategy(state, bias=self._bias), dtype=np.float64)
        total = probs.sum()
        if total > 0.0:
            probs = probs / total
        else:
            probs = np.full(len(legal), 1.0 / len(legal), dtype=np.float64)
        return legal, probs

    def sample(self, env, seat: int, rng: np.random.Generator) -> str:
        """Sample one legal action for ``seat`` from :meth:`action_probs`."""
        legal, probs = self.action_probs(env, seat)
        if not legal:
            raise ValueError(
                f"BlueprintOpponent.sample: no legal actions for seat {seat}"
            )
        idx = int(rng.choice(len(legal), p=probs))
        return legal[idx]


def assign_seats(
    table_policy: str,
    hero_seat: int,
    n_players: int,
    rng: np.random.Generator,
    fixed_seats: Optional[Mapping[int, str]] = None,
) -> Dict[int, str]:
    """Map every seat to an agent label for one hand (doc §10.1).

    The hero occupies ``hero_seat`` (label :data:`HERO_LABEL`); the other seats are
    filled per ``table_policy``:

    - ``all_blueprint`` — every opponent is the unaltered ``bp``.
    - ``random`` — each opponent seat draws i.i.d. from the four variants (via
      ``rng`` for per-hand reproducibility).
    - ``fixed`` — an explicit seat→label map (``fixed_seats``); the hero's own seat
      entry, if present, is ignored since the hero occupies it this hand.

    The schema records whatever is assigned, so analysis slices by opponent type
    regardless of the policy.
    """
    labels: Dict[int, str] = {hero_seat: HERO_LABEL}
    opp_seats = [s for s in range(n_players) if s != hero_seat]
    if table_policy == "all_blueprint":
        for s in opp_seats:
            labels[s] = "bp"
    elif table_policy == "random":
        for s in opp_seats:
            labels[s] = OPPONENT_LABELS[int(rng.integers(len(OPPONENT_LABELS)))]
    elif table_policy == "fixed":
        if fixed_seats is None:
            raise ValueError("table_policy='fixed' requires fixed_seats")
        for s in opp_seats:
            if s not in fixed_seats:
                raise ValueError(
                    f"table_policy='fixed' missing a label for seat {s} "
                    f"(hero at {hero_seat})"
                )
            label = fixed_seats[s]
            if label not in LABEL_TO_BIAS:
                raise ValueError(
                    f"fixed_seats[{s}]={label!r} is not a known opponent label"
                )
            labels[s] = label
    else:
        raise ValueError(
            f"unknown table_policy {table_policy!r}; expected "
            "'all_blueprint' | 'random' | 'fixed'"
        )
    return labels
