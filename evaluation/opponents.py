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

import json
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from poker_ai.modeling import schedules
from poker_ai.modeling.model import OpponentModel, SyntheticOpponentModel
from poker_ai.modeling.schedules import Spec
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


# --------------------------------------------------------------------------- #
# Opponent-model specification (DBR model construction)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelSpec:
    """How a **DBR** arm builds the hero's model of each opponent.

    The runner's opponents are :class:`BlueprintOpponent` bias variants, whose true
    strategy is *exactly* ``blueprint.strategy(state, bias)``.  So a synthetic model
    of a seat is exact by wrapping the same blueprint under the same bias
    (:class:`~poker_ai.modeling.model.SyntheticOpponentModel`), then optionally
    injecting a target ℓ1 error and capping confidence — the design-doc §6.2 sweep
    axes.  This spec captures the two knobs plus the perturbation seed:

    - ``p_max`` — confidence cap.  ``1.0`` with ``error = 0`` is **naive best
      response** (the exact-model, unconstrained EV ceiling — the "unsafe" envelope);
      a lower cap is the safe DBR mixture.
    - ``error`` — target ℓ1 perturbation of ``σ̂`` (0.0 = exact).  A single float here
      is a constant target across info-sets; a **schedule-shaped** error (street-graded
      or per-infoset-noisy) is supplied via ``error_schedule`` below.
    - ``confidence`` — constant ``c`` before the ``p_max`` clamp (default 1.0, so
      ``p_max`` alone sets the cap); a **schedule** (calibrated / anti-calibrated /
      flat) is supplied via ``confidence_schedule`` below.
    - ``error_schedule`` / ``confidence_schedule`` — optional JSON-string descriptors
      (mirroring the runner's ``--fixed-seats`` JSON) resolved by
      :mod:`poker_ai.modeling.schedules` (:func:`~poker_ai.modeling.schedules.error_from_spec`
      / :func:`~poker_ai.modeling.schedules.confidence_from_spec`).  When present they
      **override** the corresponding scalar, letting the *pure-vs-noisy* sweep axis
      (design §6.2) be launched from the CLI/config rather than only in code.  Kept as
      strings so the frozen spec stays hashable and round-trips through ``config.yaml``.
    - ``seed`` — perturbation seed; per-seat offset applied in
      :func:`synthetic_models_for` so distinct seats perturb independently.

    ``None`` model spec on the config ⇒ **no models** (vanilla Pluribus).
    """

    p_max: float = 1.0
    error: float = 0.0
    confidence: float = 1.0
    seed: int = 0
    error_schedule: Optional[str] = None
    confidence_schedule: Optional[str] = None

    def resolve(self) -> Tuple[Spec, Spec]:
        """Resolve ``(error, confidence)`` into the scalar-or-callable forms the model uses.

        Falls back to the constant ``error`` / ``confidence`` scalars when no schedule
        JSON is set.  The confidence schedule may reference the error schedule
        (calibrated / anti-calibrated), so error is resolved first and passed in.  A
        malformed descriptor raises here, so callers can validate a spec up front.
        """
        err_desc = json.loads(self.error_schedule) if self.error_schedule else self.error
        error_spec = schedules.error_from_spec(err_desc)
        conf_desc = (
            json.loads(self.confidence_schedule)
            if self.confidence_schedule
            else self.confidence
        )
        confidence_spec = schedules.confidence_from_spec(conf_desc, error_spec=error_spec)
        return error_spec, confidence_spec

    def as_json(self) -> Dict[str, object]:
        """Compact record for the ``games.opponent_models`` provenance column."""
        out: Dict[str, object] = {
            "p_max": float(self.p_max),
            "error": float(self.error),
            "confidence": float(self.confidence),
            "seed": int(self.seed),
        }
        if self.error_schedule:
            out["error_schedule"] = json.loads(self.error_schedule)
        if self.confidence_schedule:
            out["confidence_schedule"] = json.loads(self.confidence_schedule)
        return out


def synthetic_models_for(
    seat_labels: Mapping[int, str],
    blueprint_policy: Policy,
    spec: ModelSpec,
) -> Dict[int, OpponentModel]:
    """Build the hero's ``seat → OpponentModel`` map for one hand.

    One :class:`~poker_ai.modeling.model.SyntheticOpponentModel` per non-hero seat,
    wrapping ``blueprint_policy`` under that seat's actual bias — so with
    ``error = 0`` the model reproduces the opponent's play exactly.  The hero's own
    seat (label :data:`HERO_LABEL`) is skipped (and the agent drops it again
    defensively).  The perturbation seed is offset by seat so two seats holding the
    same bias still perturb independently.

    The (possibly scheduled) error/confidence are resolved **once** here — they are
    seat-independent (the descriptor is shared); only the per-infoset perturbation
    *direction* varies by seat, via the seat-offset model seed.
    """
    error_spec, confidence_spec = spec.resolve()
    models: Dict[int, OpponentModel] = {}
    for seat, label in seat_labels.items():
        if label == HERO_LABEL:
            continue
        if label not in LABEL_TO_BIAS:
            raise ValueError(
                f"synthetic_models_for: unknown opponent label {label!r} at seat {seat}"
            )
        models[int(seat)] = SyntheticOpponentModel(
            blueprint_policy,
            confidence=confidence_spec,
            p_max=spec.p_max,
            error=error_spec,
            seed=spec.seed + int(seat),
            bias=LABEL_TO_BIAS[label],
        )
    return models


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
    fixed_seats: Optional[Sequence[str]] = None,
) -> Dict[int, str]:
    """Map every seat to an agent label for one hand (doc §10.1).

    The hero occupies ``hero_seat`` (label :data:`HERO_LABEL`); the other seats are
    filled per ``table_policy``:

    - ``all_blueprint`` — every opponent is the unaltered ``bp``.
    - ``random`` — each opponent seat draws i.i.d. from the four variants (via
      ``rng`` for per-hand reproducibility).
    - ``fixed`` — an ordered list of exactly ``n_players - 1`` opponent identities
      (``fixed_seats``), one per opponent — like real poker, where the players keep
      their identity and only the hero's table position rotates hand to hand. Every
      opponent is present (and each keeps its own bias) on every hand; WHICH physical
      seat each identity lands in is reshuffled per hand (via ``rng``, hero-independent
      and reproducible/CRN-paired same as ``random``), so an identity's table position
      — and any interaction between position and bias — doesn't confound the sample.

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
        if len(fixed_seats) != len(opp_seats):
            raise ValueError(
                f"fixed_seats must list exactly {len(opp_seats)} opponent "
                f"identities (one per non-hero seat), got {len(fixed_seats)}"
            )
        bad = [l for l in fixed_seats if l not in LABEL_TO_BIAS]
        if bad:
            raise ValueError(
                f"fixed_seats contains unknown labels {bad}; expected one of "
                f"{OPPONENT_LABELS}"
            )
        shuffled = [fixed_seats[i] for i in rng.permutation(len(fixed_seats))]
        for s, label in zip(opp_seats, shuffled):
            labels[s] = label
    else:
        raise ValueError(
            f"unknown table_policy {table_policy!r}; expected "
            "'all_blueprint' | 'random' | 'fixed'"
        )
    return labels
