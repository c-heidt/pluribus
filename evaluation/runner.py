"""Time-budgeted evaluation runner — the game producer (doc §9.3, §10.1).

This is the component that *plays the games*: a sequential, single-node loop that
seats the search agent under test against blueprint-derived opponents, plays a full
hand with the engine, and writes one logging transaction per hand (§4).  Everything
downstream — the summary (§8), AIVAT (§10.2) — observes what this produces.

Structure (a testable core + a thin CLI):

- :class:`EvalConfig` — the per-run knobs (``run_id``, seeds, table policy, budget,
  table size / blinds / stacks).
- :class:`EvalSession` — the run's artifacts (solver config, blueprint policy, card
  LUT) and the per-hand factories (fresh env / hero / opponent).  Injected, so the
  core runs against stubs (``UniformPolicy`` + a small deck) with no blueprint on
  disk, and against a real blueprint via :func:`build_blueprint_session`.
- :func:`play_hand` — drives one hand through the :class:`SearchAgent` lifecycle
  (subgame doc §6.6) and the opponents, capturing a ``decisions`` row per hero
  decision and the hand outcome.
- :func:`run_evaluation` — the loop: deterministic ``(run_seed, hand_index)``
  seeding, the ``(run_id, hand_index)`` resume cursor, hero/position rotation, and
  the time / SIGTERM budget (stops only at a hand boundary).

Range-tracking quality (§7, step 4) is captured here too: :func:`play_hand` buffers
each live opponent's belief at every round boundary via a
:class:`~evaluation.range_quality.RangeQualityRecorder` and resolves them against the
holes revealed at showdown into ``range_quality`` rows, logged in the same per-hand
transaction.

Cluster I/O (§5, step 5) is wired here: the db is written to node-local scratch and
:func:`run_evaluation` calls an injected ``sync_fn`` (a ``VACUUM INTO`` to the
permanent FS, :meth:`~evaluation.sqlite_logging.ExperimentLog.sync_to`) on a
periodic cadence and once more at the end / on SIGTERM.  The SLURM wrapper that
stages scratch and forwards SIGTERM is ``scripts/evaluation.sh``.

The end-of-run summary (§8, step 6) runs automatically at the end of the CLI
``run`` command, after the final sync-back, against the permanent snapshot
(:func:`evaluation.summarize.summarize`).

AIVAT (§10.2, step 9) is wired here too, behind the opt-in ``EvalConfig.aivat``
flag: :func:`play_hand` feeds an :class:`~evaluation.aivat.AivatAccumulator` the
known-policy action nodes (hero + opponents) and the terminal all-in runout, and
the resulting ``aivat_value`` scalar is logged on the ``games`` row.  A dedicated
RNG sub-stream keeps it from perturbing the played hand.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import datetime
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Import order matters for the ``evaluator`` compiled kernel.  Its bind lives in
# ``environment.evaluator`` behind ``except ImportError: pass``; if that module is
# imported before ``poker_ai``, pulling in ``poker_ai._core`` re-enters the
# ``environment`` package mid-import and the swallowed circular ImportError leaves
# ``default_evaluator`` on pure Python — silently, with PLURIBUS_CORE_KERNELS set.
# The ``poker_ai`` console-script entry imports the package first so it is unaffected,
# but ``python -m evaluation.runner`` reaches ``environment`` first.  Import poker_ai
# up front so the kernel binds regardless of entry point.  (Verified: without this,
# default_evaluator.evaluate stays Python under -m even with the flag on.)
import poker_ai  # noqa: F401  (ordering side-effect, not a name use)

from environment.player import Player
from environment.poker_env import PokerEnv, raise_level
from environment.utils import card_str
from evaluation.aivat import AivatAccumulator, LeafValue
from evaluation.opponents import (
    HERO_LABEL,
    LABEL_TO_BIAS,
    OPPONENT_LABELS,
    BlueprintOpponent,
    ModelSpec,
    assign_seats,
    synthetic_models_for,
)
from evaluation.range_quality import RangeQualityRecorder
from poker_ai.modeling.model import OpponentModel
from evaluation.sqlite_logging import (
    DecisionRow,
    ExperimentLog,
    GameRow,
    HandFailureRow,
    RangeQualityRow,
    SeatRow,
)
from poker_ai.search.agent import SearchAgent
from poker_ai.search.policy import Policy
from poker_ai.search.solver import SolverConfig, config_fingerprint

logger = logging.getLogger(__name__)

# betting_stage → the schema's stage vocabulary (drops the underscore).
_STAGE: Mapping[str, str] = {
    "pre_flop": "preflop",
    "flop": "flop",
    "turn": "turn",
    "river": "river",
}
# terminal_board_len → the street the hand ended on (§6 games.terminal_street).
_BOARD_LEN_TO_STREET: Mapping[int, str] = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


# ---------------------------------------------------------------------------
# Config + session
# ---------------------------------------------------------------------------

import re as _re

# ---------------------------------------------------------------------------
# Experiment-arm labels (docs/evaluation.md §10.1)
# ---------------------------------------------------------------------------
# An arm is written ``NAME`` or ``NAME(key=value, ...)``.  Every knob an arm needs
# rides in its own label, so the sweep this eval exists to run — *model quality at a
# fixed approach* — is a plain list of conditions:
#
#   vanilla ; DBR(confidence=0.8,error=0.2) ; DBR(confidence=0.8,error=0.3)
#           ; OX(k_beta=50,error=0.2) ; OX(k_beta=50,error=0.3)
#
# What a label omits falls back to the run-wide default (the ``--model-*`` /
# ``--ox-k-beta`` options), which is why kβ and ``p_max`` need not be repeated on every
# arm: they are held FIXED across the sweep while ``error`` is the axis that varies.
_CONDITION_RE = _re.compile(
    r"^\s*(?P<kind>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\((?P<params>[^()]*)\))?\s*$"
)

# Per-kind label vocabulary.  An unknown key is a typo, not a silent no-op:
# ``OX(p_max=0.8)`` would otherwise read as a configured safety cap that OX never
# consumes (it is reach-only — see :meth:`EvalConfig.for_condition`), and the arm would
# run at the wrong setting without a word.
_ARM_PARAMS: Mapping[str, frozenset] = {
    "vanilla": frozenset(),
    "blueprint_only": frozenset(),
    "ox": frozenset({"k_beta", "error", "seed"}),
    "dbr": frozenset({"p_max", "error", "confidence", "seed"}),
}

# ``_ARM_PARAMS`` is also the arm-NAME vocabulary: a name outside it is a typo and
# raises, rather than falling back to a default approach.  That closes the same hole the
# per-kind parameter check closes, one level up — an eval whose whole purpose is
# comparing approaches cannot afford ``OXX(error=0.2)`` to run silently as a DBR arm and
# land in the results as though the intended approach had been measured.


# Code default **kβ** for a bare 'OX' condition.  kβ — not β — is the knob, because β's
# meaning moves with ``k`` and would silently mean different things on different decks:
# ``vector._ox_setup`` derives ``β = kβ / k`` once k is known, where k = the
# board-compatible opponent root combos (LOSSLESS root), i.e. ``C(deck − board, 2)``.
# The hero's own hole is NOT removed by that mask.  So the SAME β=0.05 is kβ≈56 on the
# 52-card deck (turn k=C(48,2)=1128) but kβ=6 on the 20-card test deck (k=C(16,2)=120)
# — an 8× different regime.  Fixing kβ instead makes an arm label portable.
#
# 50 reproduces the paper's OX-Search gadget mix: their FHP setting (Appendix B) fixes
# the root entry weight 1/(kβ+1) = 1/51, i.e. kβ = 50.  It is a STARTING point, not a
# derived optimum: Thm 4.6's bound (exp(σ') − exp(σ) ≤ Δ/β) is vacuous at any usable
# setting here — with Δ ≈ 2 × stack = 200 bb, kβ=50 gives Δ/β ≈ 4500 bb, and making the
# bound worth 0.1 bb would need kβ ≈ 2.3e6, at which the exploitation branch weight
# 1/(kβ+1) ≈ 4e-7 and OX degenerates into pure safe resolving.  The operative constraint
# is Thm 4.5 instead: raise kβ only while ``decisions.ox_enter_prob`` saturates at ≈ 1
# (λ* clipped at the β boundary ⇒ safety may be violated).  Going higher than that costs
# exploitation for nothing.
DEFAULT_OX_KBETA = 50.0

# Code default ``p_max`` for a DBR arm that names none (neither in its label nor via
# ``--model-p-max``).  1.0 leaves the scale INERT so ``confidence`` alone sets the DBR
# mixture — the predictable reading of ``DBR(confidence=0.8, error=0.2)`` — and keeps
# ``p_max`` as the ceiling that matters when confidence is a *schedule*.  (Naive best
# response needs p_max=1 AND confidence=1 AND error=0, i.e. an unclamped exact model;
# p_max=1 on its own is not it.)
DEFAULT_DBR_P_MAX = 1.0


def _parse_condition(condition: str) -> Tuple[str, Dict[str, float]]:
    """Split an arm label into ``(kind, params)``: ``'DBR(error=0.2)'`` → ``('dbr', {...})``.

    ``kind`` is the lower-cased name, which must be one of :data:`_ARM_PARAMS`; ``params``
    holds the label's ``key=value`` floats, empty for a bare name.  Both an unknown NAME
    (``'OXX'``, ``'oxsearch'``) and an unknown KNOB (``'OX(p_max=1)'``) raise, as does a
    label that *looks* like a spec but does not parse (``'OX(3.0)'``, ``'DBR(error=)'``).
    Nothing silently defaults: a mistyped knob would otherwise run a whole arm at the
    wrong model quality, and a mistyped name would run a whole arm as the *wrong
    approach* — both surface only as a puzzling curve, and the second is worse because
    the results table still says the approach you asked for.
    """
    m = _CONDITION_RE.match(condition or "")
    if m is None:
        raise ValueError(
            f"malformed condition {condition!r} — expected 'NAME' or 'NAME(key=value,...)', "
            "e.g. 'vanilla', 'OX(k_beta=50,error=0.2)', 'DBR(confidence=0.8,error=0.2)'."
        )
    kind = m.group("kind").lower()
    if kind not in _ARM_PARAMS:
        raise ValueError(
            f"unknown arm {m.group('kind')!r} in condition {condition!r} — expected one "
            f"of {sorted(_ARM_PARAMS)}. (It is NOT treated as a DBR arm: a mistyped "
            f"approach name would otherwise run and be reported as the approach you "
            f"asked for.)"
        )
    params: Dict[str, float] = {}
    for item in (m.group("params") or "").split(","):
        if not item.strip():
            continue
        key, sep, value = item.partition("=")
        key = key.strip().lower()
        if not sep or not key or not value.strip():
            raise ValueError(
                f"malformed parameter {item.strip()!r} in condition {condition!r} — "
                "expected 'key=value' (e.g. 'error=0.2')."
            )
        if key in params:
            raise ValueError(f"duplicate parameter {key!r} in condition {condition!r}.")
        try:
            params[key] = float(value)
        except ValueError:
            raise ValueError(
                f"parameter {key}={value.strip()!r} in condition {condition!r} is not a number."
            ) from None
    unknown = sorted(set(params) - _ARM_PARAMS[kind])
    if unknown:
        raise ValueError(
            f"condition {condition!r} sets {unknown}, which is not a parameter of a "
            f"'{kind}' arm; accepted here: {sorted(_ARM_PARAMS[kind]) or '(none)'}."
        )
    return kind, params


def split_conditions(raw: str) -> List[str]:
    """Split a multi-arm string into labels, on ``;`` or on a **top-level** ``,``.

    A condition's own parameter list contains commas
    (``'DBR(confidence=0.8,error=0.2)'``), so a naive ``split(',')`` would tear every
    parameterised arm in half.  Splitting only at paren depth 0 keeps the historical
    comma-separated form working for bare labels while making the parameterised form
    expressible; ``;`` is accepted as the unambiguous separator.
    """
    out: List[str] = []
    cur: List[str] = []
    depth = 0
    for ch in raw or "":
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch in ",;" and depth <= 0:
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    out.append("".join(cur))
    return [c.strip() for c in out if c.strip()]


def _parse_ox_kbeta(condition: str, *, default: Optional[float] = None) -> float:
    """Extract ``kβ`` from an OX-Search condition label like ``'OX(k_beta=50)'``.

    OX-Search requires a finite non-negative kβ (the safety parameter; the solver
    derives ``β = kβ / k`` once k is known).  A label with no ``k_beta=`` takes
    ``default`` — the run-wide ``--ox-k-beta`` — which itself falls back to
    :data:`DEFAULT_OX_KBETA`: kβ is held fixed while ``error`` sweeps, so it belongs on
    the run rather than on every arm.  A label that *looks* like a (botched) spec still
    raises, via :func:`_parse_condition`.
    """
    _, params = _parse_condition(condition)
    kbeta = float(params.get("k_beta", DEFAULT_OX_KBETA if default is None else default))
    if kbeta < 0.0:
        raise ValueError(f"OX-Search k_beta must be >= 0, got {kbeta} in {condition!r}.")
    return kbeta


def _arm_model_spec(
    kind: str,
    condition: str,
    params: Mapping[str, float],
    defaults: Optional["ModelSpec"],
) -> "ModelSpec":
    """The opponent model an arm runs with: run-wide ``defaults`` overridden by its label.

    ``defaults`` is the run's shared ``--model-*`` bundle; the label's own parameters win
    over it, which is exactly what makes an error sweep a list of conditions.  Called only
    for the two EXPLOITING kinds, both of which always get a spec — the model-free arms
    (``vanilla``, ``blueprint_only``) are settled in :meth:`EvalConfig.for_condition`
    without reaching here.

    **DBR** takes the full spec — ``p_max`` scale, ``confidence`` shape, ``error`` — the
    model driving both the belief likelihood and the solver clamp.

    **OX-Search is reach-only** (plan decision 6), which is a statement about *where* the
    model is consumed, not whether one exists: OX needs a model exactly as DBR does, and
    it enters SOLELY through the tracked beliefs.  So only ``error``/``seed`` — the two
    knobs that shape ``σ̂`` itself — are honoured, and ``p_max``/``confidence`` are pinned
    to their inert 1.0 (they are read through ``OpponentModel.confidence``, which nothing
    on the OX path calls).

    An OX arm therefore ALWAYS carries a spec, ``error = 0`` included — there it is the
    *exact* model of the seat, and the e=0 ceiling of the OX curve means the same thing
    as DBR's.  Returning ``None`` there instead would silently drop the arm onto the
    agent's unmodeled likelihood, which is a different distribution, not a perfect one:
    it reads the blueprint at bias ``"none"`` (never the seat's actual ``bp_fold`` /
    ``bp_call`` / ``bp_raise`` bias) and prefers the last search's average policy over the
    opponent's strategy whenever a search ran that round.
    """
    base = defaults if defaults is not None else ModelSpec(p_max=DEFAULT_DBR_P_MAX)
    # A scalar in the label + a schedule on the run is ambiguous in the direction that
    # loses data: ``ModelSpec.resolve`` lets the schedule win, so the per-arm value the
    # sweep is built around would be silently discarded.  Refuse instead.
    if "error" in params and base.error_schedule:
        raise ValueError(
            f"condition {condition!r} sets error={params['error']} but the run carries "
            "--model-error-schedule, which overrides it; drop one of the two."
        )
    error = float(params.get("error", base.error))
    seed = int(params.get("seed", base.seed))
    if kind == "ox":
        return ModelSpec(
            p_max=1.0, confidence=1.0, error=error, seed=seed,
            error_schedule=base.error_schedule,
        )
    if "confidence" in params and base.confidence_schedule:
        raise ValueError(
            f"condition {condition!r} sets confidence={params['confidence']} but the run "
            "carries --model-confidence-schedule, which overrides it; drop one of the two."
        )
    return ModelSpec(
        p_max=float(params.get("p_max", base.p_max)),
        error=error,
        confidence=float(params.get("confidence", base.confidence)),
        seed=seed,
        error_schedule=base.error_schedule,
        confidence_schedule=base.confidence_schedule,
    )


@dataclass
class EvalConfig:
    """Per-run knobs (doc §10.1 "Run config fields")."""

    run_id: str
    run_seed: int = 0
    # Experiment arm for cross-condition CRN pairing (§10.1): 'vanilla' |
    # 'DBR(p_max=...)' | 'blueprint_only' etc.  Logged verbatim onto every ``games``
    # row so the summary can group/pair arms; ``None`` for single-arm runs.  Does NOT
    # affect play — arms are paired by sharing ``run_seed``/``table_policy``/table
    # shape, which makes ``deck_seed`` match per hand (verified hero-independent, §10.1).
    condition: Optional[str] = None
    # --- experiment arm behaviour ---------------------------------------------
    # ``condition`` above is only the *label*; these two fields are what actually
    # make an arm behave.  The arms:
    #   vanilla       : search_enabled=True,  model_spec=None   (vanilla Pluribus —
    #                   real-time search, NO opponent model; THE baseline)
    #   DBR           : search_enabled=True,  model_spec=<spec> (Data-Biased Response;
    #                   naive best response is the unclamped exact model — p_max=1 AND
    #                   confidence=1 AND error=0)
    #   blueprint_only: search_enabled=False, model_spec=None   (NOT an approach —
    #                   a no-search pipeline / blueprint-quality test only)
    #   OX(k_beta=X)  : search_enabled=True,  model_spec=<spec>, k_beta=X,
    #                   model_scope='belief_only' (OX-Search / Approach B —
    #                   adaptation-safe exploitation; the gadget is HU turn/river only,
    #                   inactive elsewhere).  REACH-ONLY names WHERE the model is
    #                   consumed, not whether there is one: OX needs a model as DBR
    #                   does, it just never reaches the solve — it shapes the tracked
    #                   ranges alone.  Multiplayer OX cancelled.
    # Vanilla Pluribus *searches*; only the blueprint-only test skips search.  Kept
    # independent of ``condition`` so a caller can label freely; the CLI and
    # :meth:`for_condition` set both together so they never disagree.  Defaults
    # (search on, no models) are exactly vanilla Pluribus — an existing run is
    # unchanged.
    search_enabled: bool = True
    model_spec: Optional[ModelSpec] = None
    # How that ``model_spec`` is CONSUMED.  ``'full'`` (DBR) hands the models to both the
    # belief likelihood and the solver clamp.  ``'belief_only'`` (OX-Search) hands them to
    # the belief likelihood ALONE and leaves ``ctx.models`` empty, so the gadget solve
    # carries no DBR clamp, no σ̂ leaf rollouts and no VR baseline — the reach-only
    # exploitation of the OX plan (decision 6), and what ``vector._ox_setup`` asserts.
    # Inert when ``model_spec is None``.
    model_scope: str = "full"
    # OX-Search safety parameter **kβ** (Approach B); ``None`` for every non-OX arm, so
    # the gadget is OFF and the solve is byte-identical vanilla/DBR.  Threaded into the
    # ``SolverConfig`` built by :func:`build_blueprint_session` as ``ox_kbeta``, which is
    # where ``β = kβ / k`` is derived — see :data:`DEFAULT_OX_KBETA` for why the knob is
    # kβ rather than β.
    k_beta: Optional[float] = None
    table_policy: str = "all_blueprint"       # all_blueprint | random | fixed
    # required for table_policy='fixed': an ordered list of exactly n_players - 1
    # opponent identities (e.g. ["bp_fold", "bp_call", "bp_raise"]), one per
    # opponent — like real poker, opponents keep their identity and only the
    # hero's seat rotates (see evaluation.opponents.assign_seats).
    fixed_seats: Optional[Sequence[str]] = None
    time_budget_hours: float = 1.0            # wall-clock budget; 0 → unbounded
    # Paired mode (§10.1): a fixed hand count for this run.  When set it is the
    # sole stop criterion — the wall-clock ``time_budget_hours`` is ignored — so
    # every arm of a comparison covers the *same* ``hand_index`` range and pairs
    # cleanly on ``deck_seed``.  Time-budget mode desyncs arms (a search agent
    # completes far fewer hands than a blueprint-only one at equal wall-clock), so
    # any vanilla/DBR comparison must use ``max_hands``.  ``None`` → budget-bound.
    max_hands: Optional[int] = None
    n_players: int = 6
    big_blind: int = 100
    small_blind: int = 50
    starting_stack: int = 10_000
    # Deck bounds — full deck (2..14) for the real game, a small deck for tests.
    low_card_rank: int = 2
    high_card_rank: int = 14
    # Cluster I/O sync-back cadence (§5).  The run writes the db to node-local
    # scratch and VACUUM-INTOs a permanent-FS snapshot every ``sync_interval_hands``
    # hands (0 → only the final sync) and/or every ``sync_interval_minutes`` minutes
    # (0 → disabled).  Both are inert unless the runner is given a ``sync_fn``.
    sync_interval_hands: int = 500
    sync_interval_minutes: float = 0.0
    # AIVAT variance-reduced strength estimate (§10.2, step 9).  Off by default —
    # it adds per-hand cost (in the experiment budget, off the search hot path) and
    # is driven by a dedicated RNG sub-stream, so a hand's raw ``hero_chips_delta``
    # is identical whether AIVAT is on or off.  ``aivat_rollouts`` is the number of
    # baseline playouts averaged per value-function evaluation, and is the estimator's
    # dominant variance knob (measured var_x 1.28 at 6 vs 1.82 at 48).  ``aivat_chance``
    # additionally corrects the per-street (turn/river) chance nodes by exact
    # enumeration over the undealt deck — the only family of term that removes
    # *board* variance; ``aivat_chance_rollouts`` is its per-alternative playout count.
    # See :mod:`evaluation.aivat` for both measurements.
    aivat: bool = False
    aivat_rollouts: int = 48
    aivat_chance: bool = False
    aivat_chance_rollouts: int = 48

    def fingerprint_table_policy(self) -> Dict[str, object]:
        """The table-composition + arm identity folded into ``config_fingerprint`` (§6).

        Includes the experiment-arm behaviour (``search_enabled`` + the model spec)
        so vanilla / A fingerprint as distinct configs — the summary groups raw
        records by ``config_fingerprint`` and pairs arms by ``condition``, and a
        modeled arm is genuinely a different config, not a tweak.
        """
        return {
            "policy": self.table_policy,
            "fixed_seats": self.fixed_seats,
            "search_enabled": self.search_enabled,
            "model_spec": (self.model_spec.as_json()
                           if self.model_spec is not None else None),
            # Same spec, different consumer: a belief-only (OX) model and a DBR clamp
            # model are genuinely different configs, so they must not share a hash.
            "model_scope": (self.model_scope if self.model_spec is not None else None),
        }

    @classmethod
    def for_condition(
        cls,
        condition: str,
        *,
        model_spec: Optional[ModelSpec] = None,
        ox_k_beta: Optional[float] = None,
        **kwargs,
    ) -> "EvalConfig":
        """Build a config for an experiment arm, keeping label and behaviour in sync.

        The label is the whole arm spec — ``NAME`` or ``NAME(key=value, ...)``, parsed by
        :func:`_parse_condition`:

        - ``'vanilla'`` ⇒ vanilla Pluribus: **search, no opponent model** (THE baseline).
        - ``'blueprint_only'`` ⇒ no search at all — a pipeline / blueprint-quality test,
          not an approach.
        - ``'OX'`` / ``'OX(k_beta=X, error=e)'`` ⇒ OX-Search (Approach B): search with the
          gadget root at safety parameter β, exploiting **reach-only** — the (optionally
          error-injected) model shapes the tracked beliefs and nothing else.
        - anything else ⇒ a DBR arm, e.g. ``'DBR(confidence=0.8, error=0.2)'``: search
          **with** a model driving both the beliefs and the solver clamp.

        ``model_spec`` and ``ox_k_beta`` are the run-wide **defaults**; each label overrides
        what it names (see :func:`_arm_model_spec`).  That is what lets one submission
        sweep model quality — ``vanilla, DBR(error=0.2), DBR(error=0.3), OX(error=0.2),
        OX(error=0.3)`` — with β and ``p_max`` set once for the whole run.  A DBR arm no
        longer *requires* an explicit ``p_max``: it falls back to
        :data:`DEFAULT_DBR_P_MAX`.
        """
        kind, params = _parse_condition(condition)
        k_beta = None
        scope = "full"
        if kind == "vanilla":
            search_enabled, spec = True, None      # real Pluribus: search, no model
        elif kind == "blueprint_only":
            search_enabled, spec = False, None     # pipeline test only (no search)
        elif kind == "ox":
            k_beta = _parse_ox_kbeta(condition, default=ox_k_beta)
            search_enabled = True
            spec = _arm_model_spec(kind, condition, params, model_spec)
            # Reach-only: whatever model this arm has informs the BELIEF, never the
            # solve, so ``ctx.models`` stays empty — the invariant ``vector._ox_setup``
            # asserts, and the reason OX consumes none of DBR's clamp machinery.
            scope = "belief_only"
        else:
            search_enabled = True
            spec = _arm_model_spec(kind, condition, params, model_spec)
        return cls(
            condition=condition,
            search_enabled=search_enabled,
            model_spec=spec,
            model_scope=scope,
            k_beta=k_beta,
            **kwargs,
        )


@dataclass
class EvalSession:
    """A run's shared artifacts + per-hand factories (injected into the core).

    ``blueprint_policy`` backs both the opponents (via the bias transform) and the
    hero's round-1 / boundary blueprint queries; ``solver_cfg`` carries the leaf
    fleet the hero searches with.  ``card_info_lut`` is assigned onto each fresh env.
    """

    config: EvalConfig
    solver_cfg: SolverConfig
    blueprint_policy: Policy
    card_info_lut: object

    def new_env(self) -> PokerEnv:
        """A freshly-dealt env for one hand (the caller seeds ``np.random`` first).

        The engine deals off the **global** ``np.random`` (subgame doc / engine
        convention), so :func:`run_evaluation` calls ``np.random.seed(deck_seed)``
        immediately before this, making the deal reproducible from ``deck_seed``.
        """
        cfg = self.config
        players = [Player(i, cfg.starting_stack) for i in range(cfg.n_players)]
        env = PokerEnv(
            players=players,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
            low_card_rank=cfg.low_card_rank,
            high_card_rank=cfg.high_card_rank,
        )
        env.card_info_lut = self.card_info_lut
        return env

    def build_models(
        self, seat_labels: Mapping[int, str]
    ) -> Dict[int, "OpponentModel"]:
        """The hero's ``seat → OpponentModel`` map for this hand (DBR).

        Empty for vanilla (``model_spec is None``) — so the returned hero is
        the untouched baseline.  For a modeled arm, one exact-or-noisy synthetic
        model per opponent seat, built from that seat's actual bias label
        (:func:`synthetic_models_for`) so it models the very bot sitting there.

        Built **per hand** because ``seat_labels`` rotate with ``hero_seat``.  The
        model wraps the shared blueprint, adds no per-hand RNG of its own (the
        perturbation is seeded by ``spec.seed`` + seat), and so does not perturb the
        CRN streams — deck/seat are fixed before this is called (§10.1).
        """
        if self.config.model_spec is None:
            return {}
        return synthetic_models_for(
            seat_labels, self.blueprint_policy, self.config.model_spec
        )

    def new_hero(
        self,
        rng: np.random.Generator,
        models: Optional[Mapping[int, "OpponentModel"]] = None,
    ) -> SearchAgent:
        """A hero :class:`SearchAgent` seeded by ``rng`` (fresh per hand).

        ``models`` (from :meth:`build_models`) attaches the opponent models for a
        modeled hand; ``None``/empty is vanilla Pluribus (search, no model).
        ``search_enabled`` comes from the config: it is ``True`` for both vanilla and
        DBR (both search), and only the ``blueprint_only`` pipeline test turns search
        off.  ``model_scope`` decides how far those models reach — everywhere for DBR,
        the belief likelihood only for OX-Search (reach-only).
        """
        return SearchAgent(
            leaf_policies=self.solver_cfg.leaf.policies,
            blueprint_policy=self.blueprint_policy,
            solver_cfg=self.solver_cfg,
            rng=rng,
            models=models,
            model_scope=self.config.model_scope,
            search_enabled=self.config.search_enabled,
        )

    def new_opponent(self, label: str) -> BlueprintOpponent:
        """A blueprint(+bias) opponent for ``label`` (``bp`` / ``bp_fold`` / …)."""
        return BlueprintOpponent(label, self.blueprint_policy)


# ---------------------------------------------------------------------------
# Deterministic per-hand seeding (doc §10.1)
# ---------------------------------------------------------------------------


def derive_seeds(
    run_seed: int, hand_index: int
) -> Tuple[
    int,
    int,
    np.random.SeedSequence,
    np.random.SeedSequence,
    np.random.SeedSequence,
    np.random.SeedSequence,
]:
    """``(deck_seed, agent_seed, hero_seq, opp_seq, table_seq, aivat_seq)`` per hand.

    Everything is a pure function of ``(run_seed, hand_index)`` via one
    :class:`numpy.random.SeedSequence`, so a resumed run reproduces each hand
    bit-for-bit (§10.1).  ``deck_seed`` seeds the global RNG for the deal;
    ``agent_seed`` (logged) seeds the hero's search sampling; the returned
    sub-sequences seed the hero, opponent-sampling, seat-assignment, and AIVAT
    RNGs.  The AIVAT sub-sequence is spawned as a **5th** child (index 4): because
    ``SeedSequence.spawn`` children are determined by their spawn index, the first
    four are byte-identical to the previous four-child layout — turning AIVAT on
    does not shift the deck / hero / opponent / table streams.
    """
    deck_ss, hero_ss, opp_ss, table_ss, aivat_ss = np.random.SeedSequence(
        [int(run_seed), int(hand_index)]
    ).spawn(5)
    deck_seed = int(deck_ss.generate_state(1, dtype=np.uint32)[0])
    agent_seed = int(hero_ss.generate_state(1, dtype=np.uint32)[0])
    return deck_seed, agent_seed, hero_ss, opp_ss, table_ss, aivat_ss


# ---------------------------------------------------------------------------
# One hand
# ---------------------------------------------------------------------------


@dataclass
class HandOutcome:
    """Result of one played hand — the ``games`` outcome fields + its child rows."""

    hero_chips_delta: float
    went_to_showdown: int
    terminal_street: Optional[str]
    final_pot: Optional[float]
    hero_hole: str
    final_board: str
    decisions: List[DecisionRow]
    range_quality: List[RangeQualityRow]
    aivat_value: Optional[float] = None       # §10.2 estimate; None unless AIVAT on
    hu_from_street: Optional[int] = None      # HU coverage (opp-modeling doc §11.4)


def _to_call(env: PokerEnv) -> int:
    """Chips the current actor must add to call (0 when it can check)."""
    biggest = max(p.n_bet_chips for p in env.players)
    return int(biggest - env.current_player.n_bet_chips)


def _dist_json(actions: List[str], probs) -> str:
    """JSON map ``action → probability`` for ``decisions.action_dist`` (§6)."""
    return json.dumps({a: float(p) for a, p in zip(actions, probs)})


def _opponent_models_json(
    cfg: EvalConfig,
    models: Mapping[int, "OpponentModel"],
    seat_labels: Mapping[int, str],
) -> Optional[str]:
    """Provenance for ``games.opponent_models`` — which seats were modeled + how.

    ``None`` when the hand has no models (vanilla), so the column reads NULL
    for every baseline row and a non-NULL row is exactly a modeled hand.
    Otherwise a compact JSON object: the shared model spec, how far it reached
    (``scope``: DBR's full clamp vs OX's belief-only), plus the per-seat bias the model
    was built under, keyed by seat.  This is what lets the summary confirm *which*
    opponents the exploitation slice covers, and at what model quality.
    """
    if not models:
        return None
    return json.dumps(
        {
            "spec": cfg.model_spec.as_json() if cfg.model_spec is not None else None,
            "scope": cfg.model_scope,
            "seats": {str(s): seat_labels.get(s) for s in sorted(models)},
        },
        sort_keys=True,
    )


def _capture_hero_decision(
    hero: SearchAgent, env: PokerEnv, action: str
) -> DecisionRow:
    """Build the ``decisions`` row for the hero's just-chosen (un-stepped) action.

    Called with ``env`` still at the decision node (``act`` does not step), so the
    decision context (pot / to-call / stack / live count) and the played action
    distribution are read directly.  ``searched`` distinguishes a real solve
    (rounds 2–4, or a triggered round-1 search) from a round-1 blueprint play; the
    solver-run columns come from ``hero.last_search`` (populated in step 1).
    """
    stage = _STAGE.get(env.betting_stage, env.betting_stage)
    # The action grid's second axis (env ``RAISE_SIZES_BY_STAGE[stage][level]``):
    # raises already in this round, clamped the way the env clamps it (last level
    # repeats).  Logged so the summary can report the played action mix at exactly
    # the granularity the abstraction is cut at, not pooled over a street.
    level = raise_level(env.betting_stage, env.n_raises_this_round)
    num_live = sum(1 for p in env.players if p.is_active)
    pot_before = float(env.pot_size)
    to_call = float(_to_call(env))
    hero_stack = float(env.players[hero.my_seat].n_chips)

    # The EXACT σ the bot played (agent.play_distribution — the single source of
    # truth ``act`` also uses).  ``searched`` is False not only for a round-1
    # blueprint play but also when the search exists yet does not cover this node (a
    # decision past a depth-limit leaf, or an off-tree line): the bot then played the
    # blueprint fallback, so the decision is logged as a blueprint play — never the
    # search's regime over a uniform guess.
    played_legal, played_probs, searched = hero.play_distribution(env)
    if searched:
        res = hero.last_search
        wall = float(res.wall_seconds)
        stats = res.stats
        return DecisionRow(
            betting_stage=stage,
            regime=res.regime,
            searched=1,
            is_research=0,          # blueprint opponents never play off-tree (§10.1)
            num_live=num_live,
            n_live=int(res.n_live),  # live ranges the solver sized its budget on (calibration axis)
            pot_before=pot_before,
            to_call=to_call,
            hero_stack=hero_stack,
            iterations=int(res.iterations_run),
            wall_seconds=wall,
            iters_per_sec=(res.iterations_run / wall) if wall > 0 else None,
            stop_reason=res.stop_reason,
            node_count=int(stats.node_count),
            unique_pubkeys=int(stats.unique_pubkeys),
            cache_hits=int(stats.cache_hits),
            cache_misses=int(stats.cache_misses),
            action_played=action,
            action_dist=_dist_json(played_legal, played_probs),
            raise_level=level,
            # 1 iff a modeled solve produced this play (DBR) — the
            # coverage flag the summary restricts the exploitation slice to (§9 A7).
            modeled_decision=1 if hero.has_models else 0,
            # OX-Search (Approach B) opt-out saturation for this solve — NULL for
            # vanilla/DBR and for any non-vector subgame (the gadget was inactive).
            # ≈1 ⇒ β too small (Thm 4.5 guard).  Also marks a genuinely OX decision.
            ox_enter_prob=res.ox_enter_prob,
        )

    # Blueprint play — round 1 (no search), or the search-miss / failed-solve hard
    # fallback (the bot played the blueprint at this node).  ``searched=0`` IS the
    # fallback signal.  ``modeled_decision`` is 0: a blueprint play consulted no
    # model, even in a DBR hand.
    return DecisionRow(
        betting_stage=stage,
        regime="blueprint",
        searched=0,
        num_live=num_live,
        pot_before=pot_before,
        to_call=to_call,
        hero_stack=hero_stack,
        action_played=action,
        action_dist=_dist_json(played_legal, played_probs),
        raise_level=level,
        modeled_decision=0,
    )


def _hero_played_dist(
    hero: SearchAgent, env: PokerEnv
) -> Tuple[List[str], np.ndarray]:
    """The hero's just-played action distribution ``(legal, probs)`` at ``env``.

    Delegates to :meth:`SearchAgent.play_distribution` — the exact σ the hero
    sampled from (the searched final-iteration mix, read post-``act`` so a pinned
    played row returns unchanged; or the blueprint fallback when the search does not
    cover this node).  This is the ``π`` AIVAT corrects with (§10.2) and it matches
    the logged ``decisions.action_dist`` because both come from the same method.
    """
    legal, probs, _ = hero.play_distribution(env)
    return legal, probs


def play_hand(
    env: PokerEnv,
    hero_seat: int,
    hero: SearchAgent,
    opponents: Mapping[int, BlueprintOpponent],
    opp_rng: np.random.Generator,
    blueprint_policy: Policy,
    *,
    aivat: Optional[AivatAccumulator] = None,
) -> HandOutcome:
    """Play one full hand; return its outcome + per-hero-decision rows.

    Drives the agent lifecycle (subgame doc §6.6): ``on_hand_start`` at the deal,
    ``on_board_update`` at each round boundary (belief update + the new round's
    solve), ``on_observed_action`` for **every** action (deep-copying the pre-action
    env, as the agent's contract requires), and ``act`` at the hero's decisions.
    Opponents sample the (biased) blueprint at their nodes.

    At each round boundary — after the agent's belief update, before the hero acts —
    every live opponent's belief is buffered for the range-quality resolution at
    showdown (§7, step 4), but only while the hero is still live: once it folds the
    tracker stops updating, so its belief would be stale.

    When an ``aivat`` accumulator is passed (step 9, §10.2) every known-policy action
    node — the hero's plays and the opponents' — contributes a control-variate
    correction, and the terminal all-in runout a chance correction; the resulting
    ``aivat_value`` is returned on the :class:`HandOutcome`.  ``None`` → AIVAT off
    (the default), and the loop is byte-for-byte the prior behaviour.
    """
    hero.on_hand_start(env, hero_seat)
    decisions: List[DecisionRow] = []
    recorder = RangeQualityRecorder(hero_seat, opponents.keys())
    seen_board = {int(c) for c in env.community_cards}
    prev_round = env.betting_round
    # HU coverage (opp-modeling doc §11.4): earliest street at which a betting
    # round *began* with exactly two active seats, the hero among them — the
    # earliest point a OX-Search-HU (design doc §4.2b) solve could activate.  Tracked
    # for every condition: it is a property of the play trajectory, not of the
    # method under test.
    hu_from_street: Optional[int] = _hu_street(env, hero_seat, prev_round)

    while not env.is_terminal:
        r = env.betting_round
        if r != prev_round:
            if hu_from_street is None:
                hu_from_street = _hu_street(env, hero_seat, r)
            # Round boundary: the engine has dealt the new street's cards and it is
            # pre-action on the new round.  Announce the belief update + solve
            # before the hero acts (a no-op once the hero has folded).
            new_cards = [c for c in env.community_cards if int(c) not in seen_board]
            hero.on_board_update(env, new_cards)
            seen_board = {int(c) for c in env.community_cards}
            prev_round = r
            # Buffer the just-updated opponent beliefs for showdown resolution — but
            # only while the hero drives the tracker (folded ⇒ dormant ⇒ stale).
            if env.players[hero_seat].is_active and hero.tracker is not None:
                recorder.capture(
                    hero.tracker, _STAGE.get(env.betting_stage, env.betting_stage)
                )

        seat = env.player_i
        env_before = copy.deepcopy(env)          # act/sample don't step, so this is
        if seat == hero_seat:                    # still the pre-action state
            action = hero.act(env)
            decisions.append(_capture_hero_decision(hero, env, action))
            # AIVAT action-node correction at the hero's known-policy node (§10.2):
            # the played σ is the exact control-variate weighting.
            if aivat is not None:
                legal, probs = _hero_played_dist(hero, env)
                aivat.correct_action(env_before, seat, action, legal, probs)
        elif aivat is not None:
            # Sample the opponent from its exact known policy AND correct that node.
            # Replicating sample() (one opp_rng draw over the same probs) keeps the
            # played action bit-identical to the AIVAT-off path.  Corrections stop
            # once the hero has folded — its outcome is then fixed, so the terms are
            # ~0 and the (dormant) tracker belief would be stale.
            legal, probs = opponents[seat].action_probs(env, seat)
            action = legal[int(opp_rng.choice(len(legal), p=probs))]
            if env.players[hero_seat].is_active:
                aivat.correct_action(env_before, seat, action, legal, probs)
        else:
            action = opponents[seat].sample(env, seat, opp_rng)
        hero.on_observed_action(env_before, seat, action)
        env.step_in_place(action)

    return _finish_hand(
        env, hero_seat, decisions, recorder, aivat, hu_from_street=hu_from_street
    )


def _hu_street(env: PokerEnv, hero_seat: int, street: int) -> Optional[int]:
    """``street`` if the (pre-action) round is heads-up with the hero, else None."""
    active = [i for i, p in enumerate(env.players) if p.is_active]
    if len(active) == 2 and hero_seat in active:
        return int(street)
    return None


def _finish_hand(
    env: PokerEnv,
    hero_seat: int,
    decisions: List[DecisionRow],
    recorder: RangeQualityRecorder,
    aivat: Optional[AivatAccumulator] = None,
    *,
    hu_from_street: Optional[int] = None,
) -> HandOutcome:
    """Read the terminal env into a :class:`HandOutcome` (§6 games outcome cols)."""
    hero_delta = float(env.payout[hero_seat])
    # A hand went to showdown iff ≥2 seats are still live (un-folded) at terminal —
    # they showed down (incl. all-in runouts).  The engine reports ``betting_stage``
    # as ``'terminal'`` even for showdowns (``'show_down'`` is only transient), so
    # the active-seat count, not the stage string, is the reliable signal.
    n_active = sum(1 for p in env.players if p.is_active)
    went_to_showdown = 1 if n_active >= 2 else 0
    terminal_street = _BOARD_LEN_TO_STREET.get(env.terminal_board_len)
    contrib = env.terminal_contributions
    final_pot = float(sum(contrib)) if contrib is not None else None
    hero_hole = " ".join(card_str(int(c)) for c in env.players[hero_seat].cards)
    final_board = " ".join(card_str(int(c)) for c in env.community_cards)
    return HandOutcome(
        hero_chips_delta=hero_delta,
        went_to_showdown=went_to_showdown,
        terminal_street=terminal_street,
        final_pot=final_pot,
        hero_hole=hero_hole,
        final_board=final_board,
        decisions=decisions,
        # Resolve buffered opponent beliefs vs. the holes revealed at showdown (§7).
        range_quality=recorder.resolve(env, went_to_showdown),
        # AIVAT scalar for the hand (u(z) − Σ corrections), incl. the all-in runout
        # chance correction taken at the terminal (§10.2); None when AIVAT is off.
        aivat_value=aivat.finalize(env) if aivat is not None else None,
        hu_from_street=hu_from_street,
    )


def _dealer_seat(env: PokerEnv) -> int:
    """The button seat (stable across a hand); logged for position analysis (§6)."""
    for i, p in enumerate(env.players):
        if p.is_dealer:
            return i
    return env.n_players - 1  # defensive: engine always marks a dealer


# ---------------------------------------------------------------------------
# The run loop
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _validate_config(cfg: EvalConfig) -> None:
    """Fail fast on a misconfigured run (before the loop, not 20 hands in).

    Catches the errors that would otherwise make **every** hand raise in seat
    assignment — a systematic error worth surfacing up front rather than via the
    circuit breaker after a batch of identical ``hand_failures`` rows.
    """
    if cfg.n_players < 2:
        raise ValueError(f"n_players must be >= 2, got {cfg.n_players}")
    if cfg.max_hands is not None and cfg.max_hands <= 0:
        raise ValueError(f"max_hands must be > 0 when set, got {cfg.max_hands}")
    if cfg.table_policy not in ("all_blueprint", "random", "fixed"):
        raise ValueError(
            f"unknown table_policy {cfg.table_policy!r}; expected "
            "'all_blueprint' | 'random' | 'fixed'"
        )
    if cfg.table_policy == "fixed":
        if not cfg.fixed_seats:
            raise ValueError("table_policy='fixed' requires fixed_seats")
        # One identity per opponent, like real poker: the hero's seat rotates, the
        # opponents keep their identity (evaluation.opponents.assign_seats).
        n_opponents = cfg.n_players - 1
        if len(cfg.fixed_seats) != n_opponents:
            raise ValueError(
                f"fixed_seats must list exactly {n_opponents} opponent identities "
                f"(n_players - 1), got {len(cfg.fixed_seats)}"
            )
        bad = [l for l in cfg.fixed_seats if l not in OPPONENT_LABELS]
        if bad:
            raise ValueError(
                f"fixed_seats has unknown labels {bad}; expected {OPPONENT_LABELS}"
            )


def _sync_due(cfg: EvalConfig, hands_since_sync: int, seconds_since_sync: float) -> bool:
    """Whether a periodic sync-back is due under the §5 cadence (hands or minutes)."""
    if cfg.sync_interval_hands > 0 and hands_since_sync >= cfg.sync_interval_hands:
        return True
    if (
        cfg.sync_interval_minutes > 0
        and seconds_since_sync >= cfg.sync_interval_minutes * 60.0
    ):
        return True
    return False


def _best_effort_sync(sync_fn: Callable[[], None], run_id: str) -> None:
    """Run a *periodic* sync-back, swallowing failures (§5 bounded-loss).

    A transient permanent-FS hiccup mid-run must not kill a long evaluation: the
    periodic snapshot is a bounded-loss checkpoint, and the next interval (or the
    strict final sync) retries.  The failure is logged loudly, never silent.
    """
    try:
        sync_fn()
    except Exception:
        logger.exception(
            "periodic sync-back failed for run %s (retrying next interval)", run_id
        )


PROGRESS_INTERVAL_HANDS = 100
"""Default cadence of the per-arm progress line (0 disables it)."""


def _fmt_hms(seconds: float) -> str:
    """``H:MM:SS``, the one format that stays readable from seconds to hours."""
    return str(datetime.timedelta(seconds=int(max(0.0, seconds))))


class _ProgressReporter:
    """Periodic "how far along is this arm" log line — one stream per method.

    Every experiment arm (``vanilla``, ``DBR(...)``, ``OX(...)``, ``blueprint_only``)
    is its own ``evaluate run`` invocation — scripts/evaluation.sh runs them in turn
    into one shared db — so a line tagged with the condition IS a line per method.
    A log line every ``interval`` hands rather than a live bar: these runs are watched
    through a Slurm job log far more often than a terminal, and a redrawing bar turns
    into thousands of log lines there.

    ``unit`` names what is being counted, so the calibration sweep
    (:mod:`evaluation.calibrate`, which counts *solves*) can share this rather than
    grow a near-copy.

    Elapsed and remaining are measured over the hands played **this call**, so a
    resumed run never extrapolates from the pace of the attempt that was interrupted;
    the resume offset still counts toward the displayed total.  Remaining comes from
    the paired-mode hand target when there is one and from the wall budget otherwise
    (the two stop criteria are mutually exclusive, §10.1) — and is only ever an
    extrapolation of the average hand cost so far, which is lumpy by nature: most
    hands fold pre-flop, a few are minutes-long turn solves.
    """

    def __init__(
        self,
        label: Optional[str],
        total: Optional[int],
        *,
        done: int = 0,
        interval: int = PROGRESS_INTERVAL_HANDS,
        budget_s: float = 0.0,
        unit: str = "hand",
    ) -> None:
        self._label = label or "eval"
        self._unit = str(unit)
        self._total = total if total and total > 0 else None
        self._budget_s = float(budget_s)
        self._interval = int(interval)
        self._start_done = int(done)
        self._done = int(done)
        self._t0 = time.monotonic()
        self._next = self._threshold_after(self._done)
        self._last_emit: Optional[int] = None

    @property
    def enabled(self) -> bool:
        return self._interval > 0

    def _threshold_after(self, n: int) -> int:
        """Next multiple of the interval strictly above ``n`` (absolute, not relative).

        Anchoring on absolute hand counts keeps a resumed run's lines on the same
        round numbers as the run it continues.
        """
        if self._interval <= 0:
            return 0
        return ((n // self._interval) + 1) * self._interval

    def update_to(self, n_done: int) -> None:
        """Report cumulative progress (absolute count, monotonic)."""
        if not self.enabled or n_done <= self._done:
            return
        self._done = int(n_done)
        if self._done >= self._next:
            self._emit()
            self._next = self._threshold_after(self._done)

    def bump(self, n: int = 1) -> None:
        self.update_to(self._done + n)

    def finish(self) -> None:
        """Final line, unless the last periodic one already reported this same count."""
        if self.enabled and self._done > self._start_done and self._done != self._last_emit:
            self._emit(final=True)

    def _emit(self, final: bool = False) -> None:
        self._last_emit = self._done
        elapsed = time.monotonic() - self._t0
        played = self._done - self._start_done
        rate = played / elapsed if elapsed > 0 and played > 0 else 0.0
        if self._total is not None:
            pct = 100.0 * self._done / self._total
            head = f"{self._done}/{self._total} {self._unit}s ({pct:.1f}%)"
            remaining = (self._total - self._done) / rate if rate > 0 else None
        else:
            head = f"{self._done} {self._unit}s"
            remaining = (self._budget_s - elapsed) if self._budget_s > 0 else None
        logger.info(
            "[%s] %s | elapsed %s | remaining %s | %.2f " + self._unit + "/s%s",
            self._label,
            head,
            _fmt_hms(elapsed),
            ("~" + _fmt_hms(remaining)) if remaining is not None else "unknown",
            rate,
            " (done)" if final else "",
        )


def run_evaluation(
    session: EvalSession,
    log: ExperimentLog,
    *,
    now_fn: Callable[[], str] = _now_iso,
    git_sha: Optional[str] = None,
    hostname: Optional[str] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    sync_fn: Optional[Callable[[], None]] = None,
    max_hands: Optional[int] = None,
    max_consecutive_failures: Optional[int] = 20,
    progress_interval: int = PROGRESS_INTERVAL_HANDS,
) -> int:
    """Play hands until the budget ends (or ``should_stop``); return hands attempted.

    Stops **only at a hand boundary** — never mid-hand — so every ``games`` row is
    complete (§10.1).  Resumes from the ``(run_id, hand_index)`` cursor (§6): the
    max attempted ``hand_index`` for this ``run_id`` is read up front and the loop
    continues from the next, and the deterministic seeding makes the continuation
    identical to an uninterrupted run.

    **Per-hand fault tolerance (§9.3).**  A hand that raises does not abort the run:
    it is rolled back (§4), recorded in ``hand_failures`` with its seeds + traceback,
    and the loop moves on.  ``max_consecutive_failures`` is a circuit breaker — that
    many failures **in a row** re-raises (a systematic error, e.g. a bad config, is
    not bad luck and should stop loudly rather than spin to the time budget); pass
    ``None`` to disable it.  The return value counts hands *attempted* (successes +
    failures) this call.

    Parameters
    ----------
    now_fn, git_sha, hostname
        Provenance passed onto each ``games`` / ``hand_failures`` row (timestamps
        are *passed in*, §6).
    should_stop
        Polled at each hand boundary — the SIGTERM hook wires it (§5), so a
        preempted run stops with a complete final hand and then syncs back.
    sync_fn
        The permanent-FS sync-back (``VACUUM INTO`` via
        :meth:`ExperimentLog.sync_to`), called on the §5 cadence
        (``EvalConfig.sync_interval_hands`` / ``_minutes``) and once more at the
        end — including after a ``should_stop`` (SIGTERM) break, so nothing since
        the last periodic snapshot is lost.  Periodic calls are best-effort
        (bounded loss); the final call is strict (it *is* the run's product, so a
        failure there propagates).  ``None`` → no sync-back (the node-local db is
        the only artifact — a local run with no separate permanent path).
    max_hands
        Optional **per-call** hard cap on hands this invocation (tests/back-compat).
        Takes precedence over ``cfg.max_hands``.  For the paired-mode fixed count
        (§10.1) prefer ``EvalConfig.max_hands``, which is *total*-based (resume-safe)
        and disables the wall-clock budget; ``None`` here + ``cfg.max_hands=None`` →
        budget-bound only.
    progress_interval
        Log a progress line (arm label, hands done, elapsed, remaining) every this
        many hands; ``0`` silences it.  See :class:`_ProgressReporter`.
    """
    cfg = session.config
    _validate_config(cfg)
    fingerprint = config_fingerprint(session.solver_cfg, cfg.fingerprint_table_policy())
    hand_index = log.next_hand_index(cfg.run_id)
    # Paired mode (§10.1): a fixed ``cfg.max_hands`` is the sole stop criterion —
    # the wall-clock budget is disabled so every arm covers the same hand_index
    # range and pairs on ``deck_seed``.  It is *total*-based (the resume cursor
    # already counts logged hands), so a preempted-then-resumed arm still stops at
    # exactly ``max_hands`` total.  The explicit ``max_hands`` param stays a
    # per-call cap (tests/back-compat) and takes precedence when passed.
    budget_s = 0.0 if cfg.max_hands is not None else cfg.time_budget_hours * 3600.0
    call_cap = max_hands
    if call_cap is None and cfg.max_hands is not None:
        call_cap = max(0, cfg.max_hands - hand_index)
    start = time.monotonic()
    n_attempted = 0
    n_failed = 0
    consecutive_failures = 0
    hands_since_sync = 0
    last_sync = time.monotonic()
    # ``hand_index`` is the resume cursor, i.e. hands already logged for this run_id —
    # so it is both the starting count and, with ``call_cap``, the run's total.
    progress = _ProgressReporter(
        cfg.condition,
        (hand_index + call_cap) if call_cap is not None else None,
        done=hand_index,
        interval=progress_interval,
        budget_s=budget_s,
    )

    while True:
        if should_stop is not None and should_stop():
            break
        if budget_s > 0 and (time.monotonic() - start) >= budget_s:
            break
        if call_cap is not None and n_attempted >= call_cap:
            break
        ok = _play_and_log_one(
            session, log, cfg, fingerprint, hand_index, now_fn, git_sha, hostname
        )
        hand_index += 1
        n_attempted += 1
        hands_since_sync += 1
        progress.update_to(hand_index)
        if ok:
            consecutive_failures = 0
        else:
            n_failed += 1
            consecutive_failures += 1
            if (
                max_consecutive_failures is not None
                and consecutive_failures >= max_consecutive_failures
            ):
                # Preserve everything logged so far before aborting (best-effort;
                # the run is dying anyway, so a sync failure here must not mask the
                # circuit-breaker error the operator needs to see).
                if sync_fn is not None:
                    _best_effort_sync(sync_fn, cfg.run_id)
                raise RuntimeError(
                    f"aborting run {cfg.run_id!r}: {consecutive_failures} consecutive "
                    f"hand failures (through hand_index {hand_index - 1}) — a "
                    "systematic error, not bad luck; see the hand_failures table."
                )
        # Periodic VACUUM INTO sync-back to the permanent FS (§5) — bounds how much
        # a node failure between snapshots can cost; the final sync below is strict.
        if sync_fn is not None and _sync_due(
            cfg, hands_since_sync, time.monotonic() - last_sync
        ):
            _best_effort_sync(sync_fn, cfg.run_id)
            hands_since_sync = 0
            last_sync = time.monotonic()

    progress.finish()
    if n_failed:
        logger.warning(
            "run %s: %d of %d hands failed this session (see hand_failures)",
            cfg.run_id, n_failed, n_attempted,
        )
    # Final sync-back — on normal budget end and on a SIGTERM (should_stop) break
    # alike (§5).  Strict: the permanent snapshot is the run's product, so a
    # failure here propagates rather than silently leaving stale permanent data.
    if sync_fn is not None:
        sync_fn()
    return n_attempted


def _reopen_session_after_fork(session: EvalSession) -> None:
    """Reopen the session's fork-inherited LMDB handles in a worker (``MDB_BAD_RSLOT``).

    Mirrors :func:`poker_ai.search.parallel._reopen_leaf_fleet_lmdb` but over the
    :class:`EvalSession`: the blueprint policy and the leaf-fleet variants (which share
    one ``CFRTables``/LMDB index) each get a fresh reader slot after the fork.  Per-hand
    opponent models are built from this same reopened blueprint, so they need no
    separate reopen.  Deduped by object id (the four §4 leaf variants share one object).
    """
    seen: set = set()

    def _reopen(obj) -> None:
        if obj is None or id(obj) in seen:
            return
        fn = getattr(obj, "reopen_after_fork", None)
        if fn is not None:
            seen.add(id(obj))
            fn()

    _reopen(session.blueprint_policy)
    for pol in session.solver_cfg.leaf.policies.values():
        _reopen(pol)


@contextlib.contextmanager
def _parent_lmdb_closed(session: EvalSession):
    """Close the parent's blueprint LMDB envs for the duration of a fork.

    **The other half of the fork protocol**, and the half
    :func:`_reopen_session_after_fork` cannot supply.  Reopening in the child is
    not enough on its own: python-lmdb holds per-environment transaction state
    that *survives* ``Environment.close()`` in a forked child, so a child that
    inherits an **open** env trips ``mdb_txn_renew: MDB_BAD_RSLOT`` on its first
    read transaction even after reopening — see
    :meth:`~poker_ai.tables.index.InfosetIndex.close_env`, which documents exactly
    this, and :meth:`~poker_ai.tables.cfr_tables.CFRTables.close_envs`.  Forking
    while the parent's envs are closed guarantees the child inherits nothing; each
    side then opens its own.  The training server does the same dance around its
    worker start (``blueprint/multiprocess/server.py``).

    Measured: without this, a 6-worker parallel eval lost hands to ``BadRslotError``
    in the first dozen indices — the pool logs the traceback and moves on, so the
    hands are simply *absent* from the run rather than reported as failures.

    Deduped on the ``CFRTables`` itself, not on the policy: the four §4 bias
    variants and the blueprint may be distinct objects over one table set, and
    ``open_envs`` is **not** idempotent (a second call leaks the first handle),
    even though ``close_envs`` is.  Yields immediately and reopens in a ``finally``,
    so an exception in the pool still restores the parent.

    Safe when no fork happens (``n_workers == 1`` runs the pool inline): the
    worker's own :func:`_reopen_session_after_fork` opens the envs it needs before
    the first hand either way.
    """
    tables = {}
    for pol in (session.blueprint_policy, *session.solver_cfg.leaf.policies.values()):
        t = getattr(pol, "_tables", None)
        if t is not None and hasattr(t, "close_envs"):
            tables[id(t)] = t
    for t in tables.values():
        t.close_envs()
    try:
        yield
    finally:
        for t in tables.values():
            t.open_envs()


# --- Parallel eval: pool hooks (module-level so they are fork-inherited cleanly) ---
def _eval_worker_setup(worker_id: int, shared: dict):
    """Per-worker init after fork: reopen LMDB, open this worker's node-local DB."""
    _reopen_session_after_fork(shared["session"])
    wdir = shared["worker_dir"]
    os.makedirs(wdir, exist_ok=True)
    wpath = os.path.join(wdir, f"w{worker_id:03d}.sqlite")
    if os.path.exists(wpath):
        os.remove(wpath)   # a stale DB from a crashed prior attempt (never merged)
    return {"log": ExperimentLog.open(wpath), "path": wpath, "n_ok": 0, "n_fail": 0}


def _eval_worker_process(hand_index: int, state: dict, shared: dict) -> bool:
    """Play + log one hand.  Returns success, which is what arms the pool's breaker."""
    ok = _play_and_log_one(
        shared["session"], state["log"], shared["cfg"], shared["fingerprint"],
        hand_index, shared["now_fn"], shared["git_sha"], shared["hostname"],
    )
    state["n_ok" if ok else "n_fail"] += 1
    return ok


def _eval_worker_teardown(state: dict) -> dict:
    state["log"].close()
    return {"path": state["path"], "n_ok": state["n_ok"], "n_fail": state["n_fail"]}


def _worker_db_paths(worker_dir: str) -> List[str]:
    """Every per-worker DB on disk, whether or not its worker lived to hand one back.

    The pool's teardown payloads die with a worker, but the paths are deterministic
    (``_eval_worker_setup``), so the hands a killed run already committed stay
    recoverable from the directory alone.
    """
    import glob

    return sorted(glob.glob(os.path.join(worker_dir, "w*.sqlite")))


def _checkpoint_parallel(target_db_path, worker_dir: str, sync_path) -> None:
    """Snapshot the run-so-far to the permanent FS **without disturbing the workers**.

    The parallel runner merges only at the very end, so until then the target DB holds
    nothing this attempt produced and the ordinary sync-back would copy an empty file.
    A hard kill — the cgroup OOM killer, a node failure, a wall-clock overrun past the
    grace period — would therefore lose every hand played.  So the parent, which is
    idle while the pool works, builds the snapshot itself on the §5 cadence:

    1. ``VACUUM INTO`` each live worker DB through a **read-only** connection.  That is
       safe to run against a worker that is actively writing: it takes only a read
       transaction, so the copy is a consistent point-in-time cut, and the per-hand
       transaction (§4) means the cut never lands inside a hand.  Verified against a
       concurrent writer, not assumed.
    2. Merge those copies into a fresh copy of the target.  Rebuilt from scratch every
       time, so a checkpoint is a pure function of what exists now and repeating one
       cannot double-count rows the way merging into a live target would.
    3. ``os.replace`` over the permanent path — the same atomic swap
       :meth:`ExperimentLog.sync_to` uses, so a reader never sees a half-written file
       and the previous good checkpoint survives until this one is complete.

    The result is a COMPLETE, readable experiment DB, not a partial artifact needing
    reassembly: if the run dies, the checkpoint is the run minus at most one cadence.
    """
    import shutil
    import sqlite3
    import tempfile

    from evaluation.sqlite_logging import merge_logs

    workers = _worker_db_paths(worker_dir)
    # Stage NODE-LOCAL, beside the target: the copies and the merge are scratch, and
    # doing them on the permanent (network) FS would make a checkpoint N+1 network
    # writes instead of the one that actually has to go there.
    staging = tempfile.mkdtemp(
        prefix=".ckpt.", dir=os.path.dirname(os.fspath(target_db_path)) or "."
    )
    try:
        # (1) consistent copies of the live worker DBs
        copies = []
        for i, wpath in enumerate(workers):
            dst = os.path.join(staging, f"w{i:03d}.sqlite")
            con = sqlite3.connect(f"file:{wpath}?mode=ro", uri=True)
            try:
                con.execute("VACUUM INTO ?", (dst,))
            finally:
                con.close()
            copies.append(dst)
        # (2) a fresh target copy (carrying any prior attempt's merged rows), + merge
        base = os.path.join(staging, "base.sqlite")
        tlog = ExperimentLog.open(os.fspath(target_db_path))
        try:
            tlog.snapshot(base)
        finally:
            tlog.close()
        merge_logs(base, copies)
        # (3) one network write, atomically swapped — the same path sync_to takes
        blog = ExperimentLog.open(base)
        try:
            blog.sync_to(sync_path)
        finally:
            blog.close()
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _rescue_worker_logs(target_db_path, worker_dir: str, sync_fn, run_id: str) -> None:
    """Merge + sync whatever the workers finished, after the pool died mid-run.

    A worker killed outright (OOM) makes :func:`~evaluation.hand_pool.run_index_pool`
    raise, and that raise lands BEFORE the end-of-run merge — so without this the hands
    the *surviving* workers had already committed are complete rows in their own
    node-local DBs that nothing ever reads, and the job script's ``EXIT`` trap then
    deletes the scratch directory holding them.  Measured: one worker killed out of
    three cost 399 finished jobs.

    Every step is best-effort and logged: the run is already failing, and a rescue that
    raises would replace the operator's real error with its own.
    """
    from evaluation.sqlite_logging import merge_logs

    paths = _worker_db_paths(worker_dir)
    if not paths:
        return
    try:
        merge_logs(os.fspath(target_db_path), paths)
        logger.warning("run %s: pool aborted — merged %d worker DBs so the completed "
                       "hands survive", run_id, len(paths))
    except Exception:
        logger.exception("run %s: could not merge worker DBs after the pool aborted; "
                         "they are still on disk under %s", run_id, worker_dir)
        return
    if sync_fn is not None:
        _best_effort_sync(sync_fn, run_id)


def run_evaluation_parallel(
    session: EvalSession,
    target_db_path,
    *,
    n_workers: int,
    now_fn: Callable[[], str] = _now_iso,
    git_sha: Optional[str] = None,
    hostname: Optional[str] = None,
    sync_fn: Optional[Callable[[], None]] = None,
    sync_path=None,
    max_hands: Optional[int] = None,
    stop_event=None,
    progress_interval: int = PROGRESS_INTERVAL_HANDS,
    max_consecutive_failures: Optional[int] = 20,
) -> int:
    """Play hands **in parallel** — one hand per core, search ``workers=1`` (no nested pool).

    Each of ``n_workers`` forked workers pulls the next ``hand_index`` from a shared
    dynamic counter (skipping the completed-set for resume), plays + logs it to its OWN
    node-local DB; the parent then merges those into ``target_db_path`` with disjoint id
    ranges (:func:`~evaluation.sqlite_logging.merge_logs`) and syncs.  Dynamic pull
    matters: most hands fold pre-flop (no search) while a few are minutes-long turn
    solves, so static sharding would idle cores.

    **Reproducibility — what is and is not guaranteed.**  Every hand's *inputs* are a
    pure function of ``hand_index`` — deal, seating, and all five seed sub-streams
    (:func:`derive_seeds`) — so nothing depends on which worker picked the hand up or
    on what ran before it.  Whether the *outputs* then reproduce depends on the
    solver's stop condition:

    - **iteration-bound solves reproduce bit-for-bit**, for any ``n_workers`` and
      against a sequential run.  This is the guarantee the pool is built on and the
      one :mod:`test.evaluation.test_parallel_reproducibility` gates.
    - **wall-capped solves do NOT** (``SolverConfig.max_wall_seconds``, logged as
      ``decisions.stop_reason == 'wall_cap'``).  How many iterations fit in the cap
      depends on machine load, so the same hand can be played differently on two
      identical invocations — measured at 32/40 identical hands on a wall-bound
      config, and the *whole batch's* wall-cap share swinging 16→42 of 55 searches
      between two runs of one script.  CRN pairing across conditions degrades the
      same way, since the arms then diverge for a reason unrelated to the method.

    The wall cap is a backstop for a genuinely stuck solve, so hitting it *often*
    means the iteration budget is mis-sized for the hardware, not that the run is
    fine.  ``summarize`` flags it (``budget_bound``) once a street exceeds 40%
    wall-cap stops; treat that flag as "these results are not reproducible", not
    merely "searches were slow".

    Stop: ``max_hands`` (or ``cfg.max_hands``) as a **total** target (resume-safe), else
    the wall budget ``cfg.time_budget_hours``; ``stop_event`` (an ``mp.Event``) is polled
    at each hand boundary for SIGTERM.  Returns hands attempted this call.

    **Losing as little as possible when this dies.**  Three things, because the
    default execution mode used to keep everything it produced in per-worker DBs that
    were read exactly once, at the end:

    - the parent checkpoints to ``sync_path`` on the ``EvalConfig.sync_interval_*``
      cadence (:func:`_checkpoint_parallel`) — before this, those knobs were live in
      the sequential loop only and silently inert here, which is the mode the cluster
      script actually runs;
    - a pool abort (a worker killed by the OOM killer, or the breaker below) still
      merges + syncs what the other workers finished (:func:`_rescue_worker_logs`)
      before re-raising;
    - ``max_consecutive_failures`` in a row **within one worker** aborts the pool, the
      same circuit breaker :func:`run_evaluation` has.  A systematic error otherwise
      spends the entire wall budget writing ``hand_failures`` rows and exits 0.

    Progress (arm label, hands done, elapsed, remaining) is logged every
    ``progress_interval`` hands — ``0`` silences it.  The count is the pool's shared
    *finished* counter read on the parent's worker-health poll, so it is a real
    completion count and not the (up to ``n_workers`` ahead) hand-out cursor; it can
    therefore land a few hands past a round number before it is read.
    """
    from evaluation.hand_pool import run_index_pool
    from evaluation.sqlite_logging import merge_logs

    cfg = session.config
    _validate_config(cfg)
    fingerprint = config_fingerprint(session.solver_cfg, cfg.fingerprint_table_policy())
    # Completed-set resume cursor (dynamic out-of-order completion ⇒ max+1 is unsafe).
    tlog = ExperimentLog.open(os.fspath(target_db_path))
    try:
        skip = tlog.completed_hand_indices(cfg.run_id)
    finally:
        tlog.close()
    target = max_hands if max_hands is not None else cfg.max_hands
    budget_s = 0.0 if target is not None else cfg.time_budget_hours * 3600.0
    # Resume: ``skip`` is what a prior attempt already logged, so it is the starting
    # count, and the pool's counter only ever counts hands played THIS call.
    progress = _ProgressReporter(
        cfg.condition, target, done=len(skip),
        interval=progress_interval, budget_s=budget_s,
    )
    n_skipped = len(skip)
    # The pool reports jobs finished THIS call; both the progress line and the
    # checkpoint cadence read it, so keep the latest value where each can see it.
    _seen = {"n": 0}

    def _on_progress(n: int) -> None:
        _seen["n"] = n_skipped + n
        progress.update_to(_seen["n"])

    def progress_done() -> int:
        return _seen["n"]

    shared = {
        "session": session, "cfg": cfg, "fingerprint": fingerprint,
        "now_fn": now_fn, "git_sha": git_sha, "hostname": hostname,
        "worker_dir": f"{os.fspath(target_db_path)}.workers",
    }
    # Periodic checkpoint (§5): the parent is idle while the pool works, and it is the
    # only process that can see every worker at once, so it is what snapshots the run.
    # ``None`` sync_path ⇒ the node-local db is the only artifact, nothing to checkpoint.
    ckpt_state = {"hands": 0, "t": time.monotonic(), "cost": 0.0}

    def _heartbeat() -> None:
        n_done = progress_done() - n_skipped
        idle = time.monotonic() - ckpt_state["t"]
        if not _sync_due(cfg, n_done - ckpt_state["hands"], idle):
            return
        # Self-throttle: a checkpoint rewrites the whole DB to the permanent (network)
        # FS, and its cost grows with the run.  The hand-count cadence is sized for the
        # sequential loop's cheap per-hand sync, so on a fast-folding arm it can come
        # due far quicker than a checkpoint takes — hold off until at least twice the
        # last one's duration has passed, capping the write amplification instead of
        # letting checkpoints chase each other.
        if idle < 2.0 * ckpt_state["cost"]:
            return
        t0 = time.monotonic()
        _checkpoint_parallel(target_db_path, shared["worker_dir"], sync_path)
        ckpt_state["cost"] = time.monotonic() - t0
        ckpt_state["hands"] = n_done
        ckpt_state["t"] = time.monotonic()
        logger.info("checkpointed run %s (%d hands, %.1fs) -> %s",
                    cfg.run_id, n_done, ckpt_state["cost"], sync_path)

    # Both halves of the fork protocol: the parent's envs are closed across the fork
    # (here) and each worker opens its own in ``_eval_worker_setup``.  Neither half
    # is sufficient alone — see :func:`_parent_lmdb_closed`.
    try:
        with _parent_lmdb_closed(session):
            payloads = run_index_pool(
                n_workers=n_workers, setup=_eval_worker_setup,
                process=_eval_worker_process, teardown=_eval_worker_teardown,
                shared=shared, target=target, skip=skip,
                wall_budget_s=budget_s, stop_event=stop_event,
                progress=_on_progress,
                heartbeat=(_heartbeat if sync_path is not None else None),
                max_consecutive_failures=max_consecutive_failures,
            )
    except BaseException:
        # The pool died (a worker was killed, or the breaker fired) and the teardown
        # payloads went with it — but the finished hands are committed rows in the
        # worker DBs on disk.  Merge + sync them before the error propagates and the
        # caller's cleanup removes the scratch directory.
        progress.finish()
        _rescue_worker_logs(target_db_path, shared["worker_dir"], sync_fn, cfg.run_id)
        raise
    progress.finish()
    payloads = [p for p in payloads if p is not None]
    merge_logs(os.fspath(target_db_path), [p["path"] for p in payloads])
    n_ok = sum(p["n_ok"] for p in payloads)
    n_fail = sum(p["n_fail"] for p in payloads)
    if n_fail:
        logger.warning("parallel eval run %s: %d of %d hands failed",
                       cfg.run_id, n_fail, n_ok + n_fail)
    if sync_fn is not None:
        sync_fn()
    logger.info("parallel eval run %s: %d hands (%d workers)",
                cfg.run_id, n_ok + n_fail, n_workers)
    return n_ok + n_fail


def _play_and_log_one(
    session: EvalSession,
    log: ExperimentLog,
    cfg: EvalConfig,
    fingerprint: str,
    hand_index: int,
    now_fn: Callable[[], str],
    git_sha: Optional[str],
    hostname: Optional[str],
) -> bool:
    """Seed, seat, play, and log ONE hand in a single transaction (§4).

    Returns ``True`` on success.  On any exception the hand is rolled back (the
    ``with log.game()`` boundary) and recorded in ``hand_failures`` with its seeds
    and traceback, and ``False`` is returned so the run continues (§9.3).  The seeds
    / ``hero_seat`` are derived up front (deterministic, cannot fail) so they are
    available for the failure row even if play raises immediately.
    """
    deck_seed, agent_seed, hero_ss, opp_ss, table_ss, aivat_ss = derive_seeds(
        cfg.run_seed, hand_index
    )
    hero_seat = hand_index % cfg.n_players           # position rotation (§10.1)
    try:
        # CRN invariant (§10.1): the deal and seat assignment are drawn here from
        # ``(deck_seed, table_ss)`` — both pure functions of ``(run_seed,
        # hand_index)`` — and completed BEFORE the hero agent is constructed, so
        # they are hero-independent.  Arms sharing run_seed/table_policy/table shape
        # therefore see identical cards + seating per hand (paired on ``deck_seed``).
        # Do not move any agent construction above this block.
        np.random.seed(deck_seed)
        env = session.new_env()
        seat_labels = assign_seats(
            cfg.table_policy,
            hero_seat,
            cfg.n_players,
            np.random.default_rng(table_ss),
            cfg.fixed_seats,
        )
        # Opponent models for this hand (DBR) — keyed by the just-drawn
        # seat labels, so each seat is modeled under its own bias.  Empty for
        # vanilla.  Built AFTER seat assignment but adds no RNG draw, so the CRN
        # invariant above still holds (deck + seating are already fixed).
        models = session.build_models(seat_labels)
        hero = session.new_hero(np.random.default_rng(hero_ss), models=models)
        opponents = {
            s: session.new_opponent(lbl)
            for s, lbl in seat_labels.items()
            if lbl != HERO_LABEL
        }

        # AIVAT accumulator (§10.2) — a dedicated RNG sub-stream so the value
        # function's belief sampling / rollouts never perturb the played hand.
        aivat = None
        if cfg.aivat:
            aivat_rng = np.random.default_rng(aivat_ss)
            # Baseline continuation profile: the table's real composition, so the
            # rollout continues the way these opponents actually play.  Derived
            # from ``seat_labels``, which was drawn before the hero existed, so it
            # is identical across arms and keeps ``v`` arm-independent.
            value_fn = LeafValue(
                hero, session.solver_cfg.leaf, aivat_rng,
                n_rollouts=cfg.aivat_rollouts,
                seat_bias={
                    s: LABEL_TO_BIAS[lbl]
                    for s, lbl in seat_labels.items()
                    if lbl in LABEL_TO_BIAS
                },
            )
            aivat = AivatAccumulator(
                hero_seat, value_fn, aivat_rng,
                chance=cfg.aivat_chance,
                chance_rollouts=cfg.aivat_chance_rollouts,
            )

        started_at = now_fn()
        outcome = play_hand(
            env, hero_seat, hero, opponents, np.random.default_rng(opp_ss),
            session.blueprint_policy, aivat=aivat,
        )
        button_seat = _dealer_seat(env)

        with log.game():
            game_id = log.log_game(
                GameRow(
                    run_id=cfg.run_id,
                    condition=cfg.condition,
                    opponent_models=_opponent_models_json(cfg, models, seat_labels),
                    hand_index=hand_index,
                    config_fingerprint=fingerprint,
                    table_label=cfg.table_policy,
                    table_config=json.dumps(
                        {str(s): lbl for s, lbl in seat_labels.items()},
                        sort_keys=True,
                    ),
                    hero_seat=hero_seat,
                    button_seat=button_seat,
                    n_players=cfg.n_players,
                    big_blind=float(cfg.big_blind),
                    starting_stack=float(cfg.starting_stack),
                    deck_seed=deck_seed,
                    agent_seed=agent_seed,
                    aivat_value=outcome.aivat_value,
                    hero_chips_delta=outcome.hero_chips_delta,
                    went_to_showdown=outcome.went_to_showdown,
                    terminal_street=outcome.terminal_street,
                    hu_from_street=outcome.hu_from_street,
                    final_pot=outcome.final_pot,
                    hero_hole=outcome.hero_hole,
                    final_board=outcome.final_board,
                    git_sha=git_sha,
                    hostname=hostname,
                    started_at=started_at,
                )
            )
            log.log_seats(
                game_id,
                [
                    SeatRow(
                        seat=s,
                        is_hero=1 if lbl == HERO_LABEL else 0,
                        agent_label=lbl,
                    )
                    for s, lbl in sorted(seat_labels.items())
                ],
            )
            for decision in outcome.decisions:
                log.log_decision(game_id, decision)
            for rq in outcome.range_quality:
                log.log_range_quality(game_id, rq)
    except Exception as exc:  # isolated bad hand → log + skip, don't kill the run
        logger.warning(
            "hand_index %d (run %s) failed: %s: %s",
            hand_index, cfg.run_id, type(exc).__name__, exc,
        )
        log.log_failure(
            HandFailureRow(
                run_id=cfg.run_id,
                hand_index=hand_index,
                deck_seed=deck_seed,
                agent_seed=agent_seed,
                hero_seat=hero_seat,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
                git_sha=git_sha,
                hostname=hostname,
                failed_at=now_fn(),
            )
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Real-artifact session builder + CLI
#
# The core above is artifact-agnostic (tests inject a stub session).  This seam
# loads a **trained** blueprint + card LUT off disk for a real run; it is exercised
# only against real artifacts (no in-repo unit test — there is no trained blueprint
# in the tree), so it is kept small and delegates to the existing loaders.
# ---------------------------------------------------------------------------

_BIAS_CLASSES = ("none", "fold", "call", "raise")


def _assert_index_caches_complete(tables) -> None:
    """Hard-assert each street's shm index cache mirrors the whole index (Phase 4c).

    The in-core policy read path (``CoreTables``) trusts the shm cache to hold every
    allocated row — a miss is read as an unseen infoset → uniform.  That holds only
    if ``prewarm_caches()`` ran and never overflowed.  Mirrors the training core's
    ``CoreDriver._verify_caches``; fail loud here rather than silently serve uniform.
    """
    caches = getattr(tables, "_index_caches", None)
    if caches is None:
        return
    for r in range(4):
        occ = caches[r].occupancy()
        alloc = tables._indexes[r].n_allocated_rows
        if occ != alloc:
            raise RuntimeError(
                "Phase 4c: shm index cache for street %d is not a complete mirror "
                "of the index (occupancy=%d, allocated=%d) — prewarm_caches() must "
                "run and never overflow before the in-core policy reads." % (r, occ, alloc)
            )


def build_blueprint_session(
    cfg: EvalConfig,
    *,
    blueprint_path: str,
    lut_path: str,
    max_iterations: int = 60_000,   # the single absolute per-replica ITERATION ceiling
    # Absolute per-search WALL cap, one-worker translation of Pluribus's real-time budget:
    # its 30 s TOP on a 22-core shared-table 6-player MCCFR search = 30*22 = 660 core-seconds
    # of compute (its ~20 s average = 440).  One worker does that same work in 660 s of wall,
    # so 660 s is our absolute per-search cap.  4-player is SMALLER than 6-player, so a 4p
    # search needs no more than the 6p top ⇒ this is a safe ceiling most searches finish well
    # under (like Pluribus's 20 s avg).  The structural iteration budget stays the PRIMARY
    # stop; this only backstops a genuinely stuck solve.  The per-regime/per-street TIME
    # profile is encoded in the ITERATION budgets themselves (below), sized so every cell
    # finishes inside this cap at measured throughput — vector needs no special cap because
    # its small structural budgets already stop it at ~6 s (river) / ~150 s (turn).
    max_wall_seconds: float = 660.0,
    bias_multiplier: float = 5.0,
    pickle_dir: bool = False,
) -> EvalSession:
    """Build an :class:`EvalSession` from a trained blueprint + LUT on disk.

    ``blueprint_path`` is a trained-blueprint directory (LMDB index + a checkpoint);
    the LUT is loaded with the repo's :func:`load_info_set_lut`.  A single
    :class:`BlueprintPolicy` over the restored tables backs both the opponents (via
    the bias transform) and the hero's leaf fleet / blueprint queries.
    """
    from pathlib import Path

    from environment.action_space import MAX_ACTIONS_PER_STREET
    from information_abstraction.lookup import load_info_set_lut
    from poker_ai._core.flags import search_core_enabled
    from poker_ai.search.leaf import LeafConfig
    from poker_ai.search.policy import BlueprintPolicy
    from poker_ai.tables.cfr_tables import CFRTables
    from poker_ai.tables.warm_start import apply_warm_start_to_tables

    lut = load_info_set_lut(lut_path, pickle_dir=pickle_dir)
    # Phase 4c: when the compiled search core is enabled, attach the shm index
    # cache so the leaf rollout can read the blueprint policy in-core (the ~16%
    # Python callback).  Gated on the flag so a non-core run pays no shm cost.
    use_cache = search_core_enabled()
    # The trained index lives under ``<blueprint>/lmdb_index/street_{r}`` (the
    # server constructs CFRTables with ``save_path / "lmdb_index"``).  Passing
    # the blueprint root here would silently CREATE empty ``street_*`` indexes
    # next to it — every blueprint lookup would miss and the whole eval would
    # run on a uniform blueprint (chunks restore fine; only the key->row index
    # would be empty).  Accept either the blueprint root or the lmdb dir itself.
    bp_root = Path(blueprint_path)
    index_root = bp_root / "lmdb_index" if (bp_root / "lmdb_index").exists() else bp_root
    tables = CFRTables(
        index_path=index_root,
        actions_per_street=MAX_ACTIONS_PER_STREET,
        enable_index_cache=use_cache,
    )
    apply_warm_start_to_tables(tables, blueprint_path, cfg.n_players)
    n_indexed = sum(tables._indexes[r].n_allocated_rows for r in range(4))
    if n_indexed == 0:
        raise RuntimeError(
            f"Blueprint index at {index_root} is empty (0 rows across all "
            f"streets) — the restored chunks are unreachable and every policy "
            f"query would silently return uniform. Check --blueprint-path."
        )
    if use_cache:
        # Mirror the whole index into the shm cache ONCE in the parent, after the
        # warm-start restore and before the per-hand pool forks — children inherit
        # the fork-shared mmaps and CoreTables reads pure-shm (no LMDB on the hot
        # path).  Fail loud if the mirror is incomplete: the in-core reader has no
        # LMDB fallback, so a short cache would silently serve uniform strategies.
        tables.prewarm_caches()
        _assert_index_caches_complete(tables)
    blueprint = BlueprintPolicy(tables, bias_multiplier=bias_multiplier)

    leaf = LeafConfig(policies={c: blueprint for c in _BIAS_CLASSES})
    # ``discount_interval`` and ``auto_budget`` are NOT set here — they propagate
    # from ``SolverConfig``'s own defaults (poker_ai/search/solver_state.py), the
    # single source of truth.  Only the leaf and the genuine run-knobs
    # (``max_iterations`` / ``max_wall_seconds``) are supplied.
    solver_cfg = SolverConfig(
        leaf=leaf,
        max_iterations=max_iterations,
        max_wall_seconds=max_wall_seconds,
        # OX-Search gadget (Approach B); None ⇒ off (vanilla/DBR).  ``ox_kbeta`` is the
        # deck-agnostic form: the solver derives β = kβ / k once k is known.
        ox_kbeta=cfg.k_beta,
        # VR-MCCFR control-variate baseline (opponent_modeling §5.5).  Requested
        # unconditionally here, same as evaluation/calibrate.py: the flag is
        # self-gating on ``ctx.models`` in mccfr.py (``self._vr = variance_reduction
        # and bool(ctx.models)``), so it stays a true no-op for vanilla/OX/
        # blueprint_only (byte-identical) and only activates for DBR arms, which is
        # exactly where it earns its keep (flop MCCFR is DBR's noisiest regime).
        variance_reduction=True,
    )
    return EvalSession(
        config=cfg,
        solver_cfg=solver_cfg,
        blueprint_policy=blueprint,
        card_info_lut=lut,
    )


def _git_sha() -> Optional[str]:
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _install_sigterm_stop():
    """Trap SIGTERM/SIGINT into an event; return its ``is_set`` as ``should_stop``.

    The loop polls this at each hand boundary and stops cleanly with a complete
    final hand (§10.1).  The on-stop ``VACUUM INTO`` sync-back is step 5.
    """
    import signal
    import threading

    stop = threading.Event()

    def _handle(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    return stop.is_set


def _install_sigterm_stop_event():
    """SIGTERM/SIGINT → a **multiprocessing** Event, visible to forked pool workers.

    The threading Event of :func:`_install_sigterm_stop` lives only in the parent and
    is invisible across a fork, so the parallel runner needs an ``mp.Event`` (created
    from the same ``fork`` context the pool uses) — set it in the parent's signal
    handler, inherited by the workers, polled at each hand boundary.
    """
    import multiprocessing as mp
    import signal

    try:
        ev = mp.get_context("fork").Event()
    except ValueError:  # platform without fork
        ev = mp.Event()

    def _handle(signum, frame) -> None:
        ev.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    return ev


def _cli():
    import socket
    from pathlib import Path

    import click
    import yaml

    @click.group()
    def evaluate():
        """Evaluation harness for the real-time-search agent (docs/evaluation.md)."""

    @evaluate.command(name="run")
    @click.option("--run-id", required=True, help="Groups games of one experiment batch.")
    @click.option("--db-path", required=True, help="SQLite sink path (node-local, §5).")
    @click.option(
        "--sync-path",
        default=None,
        help="Permanent-FS snapshot path (VACUUM INTO sync-back, §5). Omit for a "
        "local run where --db-path is the only artifact.",
    )
    @click.option(
        "--sync-interval-hands",
        default=500,
        type=int,
        show_default=True,
        help="Sync-back cadence in hands (0 → only the final sync).",
    )
    @click.option(
        "--sync-interval-minutes",
        default=0.0,
        type=float,
        show_default=True,
        help="Sync-back cadence in minutes (0 → disabled).",
    )
    @click.option("--blueprint-path", required=True, help="Trained blueprint directory.")
    @click.option("--lut-path", required=True, help="Card-info LUT directory.")
    @click.option("--run-seed", default=0, type=int, show_default=True)
    @click.option(
        "--table-policy",
        default="all_blueprint",
        type=click.Choice(["all_blueprint", "random", "fixed"]),
        show_default=True,
    )
    @click.option(
        "--fixed-seats",
        default=None,
        help="For --table-policy=fixed: JSON array of exactly n_players - 1 "
        'opponent identities, one per opponent, e.g. \'["bp_fold","bp_call",'
        '"bp_raise"]\' for a 4-player game. Every opponent plays every hand; the '
        "hero's seat rotates and each hand's opponent-to-seat placement is "
        "reshuffled, so table position never confounds a given bias.",
    )
    @click.option("--time-budget-hours", default=1.0, type=float, show_default=True)
    @click.option(
        "--max-hands",
        default=None,
        type=int,
        help="Paired mode (§10.1): fixed TOTAL hand count; the sole stop criterion "
        "(ignores --time-budget-hours). REQUIRED for a vanilla/DBR comparison so "
        "every arm covers the same hand_index range and pairs on deck_seed.",
    )
    @click.option("--n-players", default=6, type=int, show_default=True)
    @click.option("--big-blind", default=100, type=int, show_default=True)
    @click.option("--small-blind", default=50, type=int, show_default=True)
    @click.option("--starting-stack", default=10_000, type=int, show_default=True)
    @click.option(
        "--low-card-rank", default=2, type=int, show_default=True,
        help="Deck low rank. Use 10 for the 20-card LUT (ranks 10-14).",
    )
    @click.option(
        "--high-card-rank", default=14, type=int, show_default=True,
        help="Deck high rank (inclusive).",
    )
    @click.option("--max-iterations", default=30_000, type=int, show_default=True,
                  help="The single absolute per-replica ceiling; the structural budget "
                       "is the primary stop and this only clips it in pathology.")
    @click.option("--max-wall-seconds", default=1000.0, type=float, show_default=True,
                  help="Loose per-search wall backstop (s). Sized above the deepest "
                       "shipped search (multiway flop MCCFR at the 30000 clamp ≈ 786 s "
                       "under load; v4 calibration flop backstop 993 s) so it never "
                       "clips the structural budget in normal operation.")
    @click.option(
        "--aivat/--no-aivat",
        default=False,
        show_default=True,
        help="Compute the AIVAT variance-reduced strength estimate (§10.2). Adds "
        "per-hand cost (experiment budget); the strength summary auto-switches to it.",
    )
    @click.option(
        "--aivat-rollouts",
        default=48,
        type=int,
        show_default=True,
        help="Baseline rollouts averaged per AIVAT value-function evaluation. The "
        "estimator's dominant variance knob; below ~6 it reduces no variance at all.",
    )
    @click.option(
        "--aivat-chance/--no-aivat-chance",
        default=False,
        show_default=True,
        help="Also take AIVAT chance corrections at the per-street (turn/river) "
        "deals, by exact enumeration over the undealt deck. Requires --aivat.",
    )
    @click.option(
        "--aivat-chance-rollouts",
        default=48,
        type=int,
        show_default=True,
        help="Baseline rollouts per alternative card in a chance correction.",
    )
    @click.option(
        "--condition",
        default="vanilla",
        show_default=True,
        help="Experiment arm, as 'NAME' or 'NAME(key=value,...)': 'vanilla' (vanilla "
        "Pluribus — search, no opponent model; the baseline), 'blueprint_only' (no "
        "search — pipeline / blueprint-quality test), 'OX(k_beta=50,error=0.2)' "
        "(OX-Search, reach-only), or a DBR arm 'DBR(confidence=0.8,error=0.2)'. Label "
        "parameters override the run-wide --model-* / --ox-k-beta defaults, so a sweep "
        "varies error per arm and sets p_max/k_beta once. Arms sharing "
        "--run-seed/--table-policy pair on deck_seed in the summary.",
    )
    @click.option(
        "--ox-k-beta",
        default=None,
        type=float,
        help="Default OX-Search safety parameter kβ for an OX arm whose label does not "
        f"give one (code default {DEFAULT_OX_KBETA}). The solver derives β = kβ / k, so "
        "this is deck-agnostic where a raw β is not. Inert for every non-OX arm.",
    )
    @click.option(
        "--model-p-max",
        default=DEFAULT_DBR_P_MAX,
        type=float,
        show_default=True,
        help="DBR's exploitation/exploitability dial for an arm (overridable per arm as "
        "'DBR(p_max=...)'). It SCALES the confidence (c = p_max * --model-confidence, the "
        "paper's mixture), so it is inert at 1.0, leaving --model-confidence as the "
        "DBR mixture knob; naive best response is p_max=1 AND confidence=1 AND error=0. "
        "Ignored by vanilla / blueprint_only / OX (reach-only) arms.",
    )
    @click.option(
        "--model-error",
        default=0.0,
        type=float,
        show_default=True,
        help="Default constant target ℓ1 perturbation of each opponent model (0 = "
        "exact) — the sweep's independent variable, overridable per arm as "
        "'DBR(error=...)' / 'OX(error=...)'. Overridden by --model-error-schedule when "
        "given (in which case a per-arm 'error=' is rejected rather than silently "
        "dropped).",
    )
    @click.option(
        "--model-confidence",
        default=1.0,
        type=float,
        show_default=True,
        help="Default constant confidence c before the --model-p-max clamp, "
        "overridable per arm as 'DBR(confidence=...)'. Overridden by "
        "--model-confidence-schedule when given. Inert for an OX arm: OX-Search is "
        "reach-only and never reads a model's confidence.",
    )
    @click.option(
        "--model-error-schedule",
        default=None,
        type=str,
        help="JSON error schedule (design §6.2 sweep axis), overriding --model-error. "
        'E.g. \'{"kind":"street","by_round":{"0":0.05,"3":0.4}}\' or a "noise" sub-map '
        "for per-infoset jitter. See poker_ai.modeling.schedules.error_from_spec.",
    )
    @click.option(
        "--model-confidence-schedule",
        default=None,
        type=str,
        help="JSON confidence schedule, overriding --model-confidence. E.g. "
        '\'{"kind":"calibrated","gain":1.0}\' (confident where accurate) or '
        '"anti_calibrated"/"flat". See schedules.confidence_from_spec.',
    )
    @click.option(
        "--model-seed",
        default=0,
        type=int,
        show_default=True,
        help="Default seed for the per-info-set model perturbation (offset per seat); "
        "overridable per arm as 'DBR(seed=...)' / 'OX(seed=...)'.",
    )
    @click.option(
        "--parallel-workers",
        default=None,
        type=int,
        help="Number of hands to play concurrently, one per core (each hand runs one "
        "serial search).  Default (unset) = auto (cpu-1).  This is the DEFAULT execution "
        "mode: hands are i.i.d. + seeded by index, so it packs the box (most hands fold "
        "pre-flop) and reproduces the sequential run bit-for-bit -- provided solves stop "
        "on the ITERATION cap.  Wall-capped solves are load-dependent and reproduce "
        "neither across worker counts nor across runs.  Ignored under --sequential.",
    )
    @click.option(
        "--progress-interval",
        default=PROGRESS_INTERVAL_HANDS,
        type=int,
        show_default=True,
        help="Log a progress line for this arm every N hands (label, hands done, "
        "elapsed, estimated remaining); 0 silences it. Each arm is its own run, so "
        "this is one progress stream per method under test.",
    )
    @click.option(
        "--sequential",
        is_flag=True,
        default=False,
        help="Force the one-hand-at-a-time loop instead of the default per-hand parallel "
        "runner (a slower reference/debug path; the search is serial either way).",
    )
    def run(**opts):
        """Play a time-budgeted evaluation run, logging one transaction per hand."""
        fixed_seats = None
        if opts["fixed_seats"]:
            fixed_seats = json.loads(opts["fixed_seats"])
        # Experiment arm.  The --model-* / --ox-k-beta options are the run-wide DEFAULTS;
        # the condition's own label overrides whatever it names, and ``for_condition``
        # decides which of them the arm actually consumes (vanilla and blueprint_only
        # consume none; OX takes error/seed only — it is reach-only).  So a sweep is a
        # list of labels sharing one set of defaults, rather than one flag set per arm.
        model_defaults = ModelSpec(
            p_max=float(opts["model_p_max"]),
            error=float(opts["model_error"]),
            confidence=float(opts["model_confidence"]),
            seed=int(opts["model_seed"]),
            error_schedule=opts["model_error_schedule"] or None,
            confidence_schedule=opts["model_confidence_schedule"] or None,
        )
        # Fail fast on a malformed schedule descriptor — before setup / any hands.
        try:
            model_defaults.resolve()
        except Exception as exc:
            raise click.UsageError(f"invalid model schedule: {exc}") from exc
        try:
            arm = EvalConfig.for_condition(
                opts["condition"],
                model_spec=model_defaults,
                ox_k_beta=opts["ox_k_beta"],
                run_id=opts["run_id"],
            )
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        cfg = EvalConfig(
            run_id=opts["run_id"],
            condition=arm.condition,
            search_enabled=arm.search_enabled,
            model_spec=arm.model_spec,
            model_scope=arm.model_scope,
            k_beta=arm.k_beta,
            run_seed=opts["run_seed"],
            table_policy=opts["table_policy"],
            fixed_seats=fixed_seats,
            time_budget_hours=opts["time_budget_hours"],
            max_hands=opts["max_hands"],
            n_players=opts["n_players"],
            big_blind=opts["big_blind"],
            small_blind=opts["small_blind"],
            starting_stack=opts["starting_stack"],
            low_card_rank=opts["low_card_rank"],
            high_card_rank=opts["high_card_rank"],
            sync_interval_hands=opts["sync_interval_hands"],
            sync_interval_minutes=opts["sync_interval_minutes"],
            aivat=opts["aivat"],
            aivat_rollouts=opts["aivat_rollouts"],
            aivat_chance=opts["aivat_chance"],
            aivat_chance_rollouts=opts["aivat_chance_rollouts"],
        )
        db_path = Path(opts["db_path"])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        sync_path = Path(opts["sync_path"]) if opts["sync_path"] else None
        if sync_path is not None:
            sync_path.parent.mkdir(parents=True, exist_ok=True)

        # config.yaml lives next to the permanent snapshot when syncing back (that
        # dir is what analysis reads, §5/§8) — the node-local dir is ephemeral;
        # otherwise next to the db.  Mirrors the blueprint runner.
        config_dir = sync_path.parent if sync_path is not None else db_path.parent
        with open(config_dir / "config.yaml", "w") as fh:
            yaml.dump(opts, fh)

        session = build_blueprint_session(
            cfg,
            blueprint_path=opts["blueprint_path"],
            lut_path=opts["lut_path"],
            max_iterations=opts["max_iterations"],
            max_wall_seconds=opts["max_wall_seconds"],
        )
        if not opts["sequential"]:
            # DEFAULT: per-hand parallel — one hand per core, one serial search each.
            # Workers write their own node-local DBs, merged into ``db_path`` with
            # disjoint id ranges, then synced.  Bit-reproducible vs sequential (hands
            # seeded by index).
            from poker_ai.search.parallel import resolve_workers
            n_workers = resolve_workers(opts["parallel_workers"])
            stop_event = _install_sigterm_stop_event()

            def _sync_par() -> None:  # fresh handle: the parent holds no open log here
                _l = ExperimentLog.open(db_path)
                try:
                    _l.sync_to(sync_path)
                finally:
                    _l.close()

            n = run_evaluation_parallel(
                session,
                db_path,
                n_workers=n_workers,
                git_sha=_git_sha(),
                hostname=socket.gethostname(),
                sync_fn=(_sync_par if sync_path is not None else None),
                sync_path=sync_path,
                stop_event=stop_event,
                progress_interval=opts["progress_interval"],
            )
            dest = sync_path if sync_path is not None else db_path
            click.echo(f"played {n} hands for run_id={cfg.run_id} "
                       f"({n_workers} parallel workers) → {dest}")
            from evaluation.summarize import summarize
            try:
                summarize(dest)
            except Exception:
                logger.exception("end-of-run summary failed for run %s", cfg.run_id)
            return

        should_stop = _install_sigterm_stop()
        log = ExperimentLog.open(db_path)
        # Periodic + on-SIGTERM + final VACUUM INTO sync-back to the permanent FS
        # (§5); None when no --sync-path (the node-local db is the only artifact).
        sync_fn = (lambda: log.sync_to(sync_path)) if sync_path is not None else None
        try:
            n = run_evaluation(
                session,
                log,
                git_sha=_git_sha(),
                hostname=socket.gethostname(),
                should_stop=should_stop,
                sync_fn=sync_fn,
                progress_interval=opts["progress_interval"],
            )
        finally:
            log.close()
        dest = sync_path if sync_path is not None else db_path
        click.echo(f"played {n} hands for run_id={cfg.run_id} → {dest}")

        # End-of-run summary (§8, step 6): runs after the final sync-back, against
        # the permanent-FS snapshot (never the live node-local file).  A summary
        # failure must not fail the run — the data is already safely committed and
        # synced — so it is logged, not raised.
        from evaluation.summarize import summarize

        try:
            summarize(dest)
        except Exception:
            logger.exception("end-of-run summary failed for run %s", cfg.run_id)

    return evaluate


# The click group, built lazily so importing the runner core costs no CLI deps.
evaluate = _cli()


if __name__ == "__main__":
    evaluate()
