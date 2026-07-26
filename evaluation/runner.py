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

import copy
import datetime
import json
import logging
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Tuple

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
from environment.poker_env import PokerEnv
from environment.utils import card_str
from evaluation.aivat import AivatAccumulator, LeafValue
from evaluation.opponents import (
    HERO_LABEL,
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
    #                   naive best response is DBR at p_max=1)
    #   blueprint_only: search_enabled=False, model_spec=None   (NOT an approach —
    #                   a no-search pipeline / blueprint-quality test only)
    # (OX-Search — the heads-up gadget — is a future approach, not yet wired; the
    # multiplayer OX-Search variant is cancelled.)
    # Vanilla Pluribus *searches*; only the blueprint-only test skips search.  Kept
    # independent of ``condition`` so a caller can label freely; the CLI and
    # :meth:`for_condition` set both together so they never disagree.  Defaults
    # (search on, no models) are exactly vanilla Pluribus — an existing run is
    # unchanged.
    search_enabled: bool = True
    model_spec: Optional[ModelSpec] = None
    table_policy: str = "all_blueprint"       # all_blueprint | random | fixed
    fixed_seats: Optional[Dict[int, str]] = None   # required for table_policy='fixed'
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
    # is identical whether AIVAT is on or off.  ``aivat_hole_samples`` is the number
    # of belief hole-draws averaged per value-function evaluation.
    aivat: bool = False
    aivat_hole_samples: int = 6

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
        }

    @classmethod
    def for_condition(
        cls,
        condition: str,
        *,
        model_spec: Optional[ModelSpec] = None,
        **kwargs,
    ) -> "EvalConfig":
        """Build a config for an experiment arm, keeping label and behaviour in sync.

        ``condition='vanilla'`` ⇒ vanilla Pluribus: **search, no opponent model** (THE
        baseline).  ``'blueprint_only'`` ⇒ no search at all — a pipeline / blueprint-
        quality test, not an approach.  Anything else (a ``DBR`` arm, e.g.
        ``'DBR(p_max=0.8)'`` — naive best response is DBR at ``p_max=1``) ⇒ search
        **with** the given ``model_spec`` (required — a modeled arm with no spec is a
        mistake, not vanilla Pluribus).
        """
        low = condition.strip().lower()
        if low == "vanilla":
            search_enabled, spec = True, None      # real Pluribus: search, no model
        elif low == "blueprint_only":
            search_enabled, spec = False, None     # pipeline test only (no search)
        else:
            if model_spec is None:
                raise ValueError(
                    f"condition {condition!r} is a modeled (DBR) arm but no "
                    "model_spec was given ('vanilla' is the model-free search "
                    "baseline; 'blueprint_only' is the no-search pipeline test)"
                )
            search_enabled, spec = True, model_spec
        return cls(
            condition=condition,
            search_enabled=search_enabled,
            model_spec=spec,
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
        DBR hand; ``None``/empty is vanilla Pluribus (search, no model).
        ``search_enabled`` comes from the config: it is ``True`` for both vanilla and
        DBR (both search), and only the ``blueprint_only`` pipeline test turns search
        off.
        """
        return SearchAgent(
            leaf_policies=self.solver_cfg.leaf.policies,
            blueprint_policy=self.blueprint_policy,
            solver_cfg=self.solver_cfg,
            rng=rng,
            models=models,
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
    for every baseline row and a non-NULL row is exactly a DBR hand.
    Otherwise a compact JSON object: the shared model spec plus the per-seat bias
    the model was built under, keyed by seat.  This is what lets the summary confirm
    *which* opponents the exploitation slice covers.
    """
    if not models:
        return None
    return json.dumps(
        {
            "spec": cfg.model_spec.as_json() if cfg.model_spec is not None else None,
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
    played_legal, played_probs, searched, blueprint_weight = hero.play_distribution(env)
    if searched:
        res = hero.last_search
        wall = float(res.wall_seconds)
        stats = res.stats
        return DecisionRow(
            betting_stage=stage,
            regime=res.regime,
            searched=1,
            leaf_mode=res.leaf_mode,
            is_research=0,          # blueprint opponents never play off-tree (§10.1)
            num_live=num_live,
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
            # How much blueprint prior the search read was shrunk toward at this
            # (covered) node — 0.0 when well-trained, →1 when starved (§8).
            blueprint_weight=float(blueprint_weight),
            # 1 iff a modeled solve produced this play (DBR) — the
            # coverage flag the summary restricts the exploitation slice to (§9 A7).
            modeled_decision=1 if hero.has_models else 0,
        )

    # Blueprint play — round 1 (no search), or the search-miss / failed-solve
    # fallback (the bot played the blueprint at this node).  ``blueprint_weight``
    # is 1.0 here (a pure-blueprint play), so the column reads uniformly across
    # both the full fallback and the shrinkage.  ``modeled_decision`` is 0: a
    # blueprint play consulted no model, even in a DBR hand.
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
        blueprint_weight=float(blueprint_weight),
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
    legal, probs, _, _ = hero.play_distribution(env)
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
        # The hero rotates through every seat, so a fixed map must label all of
        # them (the hero's own seat entry is ignored on the hands it occupies it).
        missing = [s for s in range(cfg.n_players) if s not in cfg.fixed_seats]
        if missing:
            raise ValueError(f"fixed_seats is missing labels for seats {missing}")
        bad = {s: l for s, l in cfg.fixed_seats.items() if l not in OPPONENT_LABELS}
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
            value_fn = LeafValue(
                hero, session.solver_cfg.leaf, aivat_rng,
                n_hole_samples=cfg.aivat_hole_samples,
            )
            aivat = AivatAccumulator(hero_seat, value_fn, aivat_rng)

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
    use_decision_free_equity: bool = True,
    max_iterations: int = 5_000,
    max_wall_seconds: float = 60.0,
    workers: Optional[int] = None,
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
        # warm-start restore and before ``run_parallel`` forks — children inherit
        # the fork-shared mmaps and CoreTables reads pure-shm (no LMDB on the hot
        # path).  Fail loud if the mirror is incomplete: the in-core reader has no
        # LMDB fallback, so a short cache would silently serve uniform strategies.
        tables.prewarm_caches()
        _assert_index_caches_complete(tables)
    blueprint = BlueprintPolicy(tables, bias_multiplier=bias_multiplier)

    # ``n_rollouts`` is NOT set here — it propagates from ``LeafConfig``'s own
    # default (poker_ai/search/leaf.py), the single source of truth for the leaf
    # config.  Overriding it here would silently mask that config.
    leaf = LeafConfig(
        policies={c: blueprint for c in _BIAS_CLASSES},
        use_decision_free_equity=use_decision_free_equity,
    )
    # ``discount_interval`` and ``auto_budget`` are NOT set here — they propagate
    # from ``SolverConfig``'s own defaults (poker_ai/search/solver_state.py), the
    # single source of truth.  Only the leaf and the genuine run-knobs
    # (``max_iterations`` / ``max_wall_seconds`` / ``workers``) are supplied.
    solver_cfg = SolverConfig(
        leaf=leaf,
        max_iterations=max_iterations,
        max_wall_seconds=max_wall_seconds,
        workers=workers,
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
        help='For --table-policy=fixed: JSON seat→label map covering every seat, '
        'e.g. \'{"0":"bp","1":"bp_raise","2":"bp_call","3":"bp","4":"bp_fold",'
        '"5":"bp"}\'.',
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
    @click.option("--max-iterations", default=5_000, type=int, show_default=True)
    @click.option("--max-wall-seconds", default=10.0, type=float, show_default=True)
    @click.option("--workers", default=None, type=int, help="Solver replicas (§6.7, "
                  "≤6 on this host).")
    @click.option(
        "--aivat/--no-aivat",
        default=False,
        show_default=True,
        help="Compute the AIVAT variance-reduced strength estimate (§10.2). Adds "
        "per-hand cost (experiment budget); the strength summary auto-switches to it.",
    )
    @click.option(
        "--aivat-hole-samples",
        default=6,
        type=int,
        show_default=True,
        help="Belief hole-draws averaged per AIVAT value-function evaluation.",
    )
    @click.option(
        "--condition",
        default="vanilla",
        show_default=True,
        help="Experiment arm: 'vanilla' (vanilla Pluribus — search, no opponent "
        "model; the baseline), 'blueprint_only' (no search — pipeline / blueprint-"
        "quality test), or a DBR arm (e.g. 'DBR(p_max=0.8)'; naive best response is "
        "DBR at p_max=1) which then REQUIRES --model-p-max. Arms sharing "
        "--run-seed/--table-policy pair on deck_seed in the summary.",
    )
    @click.option(
        "--model-p-max",
        default=None,
        type=float,
        help="Confidence cap for a DBR arm. Presence selects a modeled arm; omit "
        "for vanilla / blueprint_only. '1.0' with --model-error=0 is naive best "
        "response (the unsafe EV ceiling).",
    )
    @click.option(
        "--model-error",
        default=0.0,
        type=float,
        show_default=True,
        help="Target ℓ1 perturbation of each opponent model (0 = exact).",
    )
    @click.option(
        "--model-seed",
        default=0,
        type=int,
        show_default=True,
        help="Seed for the per-info-set model perturbation (offset per seat).",
    )
    def run(**opts):
        """Play a time-budgeted evaluation run, logging one transaction per hand."""
        fixed_seats = None
        if opts["fixed_seats"]:
            fixed_seats = {int(k): v for k, v in json.loads(opts["fixed_seats"]).items()}
        # Experiment arm: a DBR (modeled) arm is signalled by --model-p-max.  Guard
        # the easy mistake — a modeled-looking --condition with no p_max would
        # silently run as vanilla Pluribus (search, no model), i.e. a mislabeled
        # baseline.
        cond = opts["condition"]
        model_spec = None
        if opts["model_p_max"] is not None:
            model_spec = ModelSpec(
                p_max=float(opts["model_p_max"]),
                error=float(opts["model_error"]),
                seed=int(opts["model_seed"]),
            )
        elif cond.strip().lower() not in ("vanilla", "blueprint_only"):
            raise click.UsageError(
                f"--condition={cond!r} is a DBR arm but --model-p-max was not "
                "given ('vanilla' and 'blueprint_only' are the only model-free arms)."
            )
        arm = EvalConfig.for_condition(cond, model_spec=model_spec, run_id=opts["run_id"])
        cfg = EvalConfig(
            run_id=opts["run_id"],
            condition=arm.condition,
            search_enabled=arm.search_enabled,
            model_spec=arm.model_spec,
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
            aivat_hole_samples=opts["aivat_hole_samples"],
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
            workers=opts["workers"],
        )
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
