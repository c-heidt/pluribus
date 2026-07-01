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

Out of scope here (later steps): the periodic / on-SIGTERM ``VACUUM INTO`` sync-back
and SLURM wrapper (§5, step 5), the end-of-run summary (§8, step 6), and
``aivat_value`` (§10.2, step 9).
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

from environment.player import Player
from environment.poker_env import PokerEnv
from environment.utils import card_str
from evaluation.opponents import (
    HERO_LABEL,
    OPPONENT_LABELS,
    BlueprintOpponent,
    assign_seats,
)
from evaluation.range_quality import RangeQualityRecorder
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
    table_policy: str = "all_blueprint"       # all_blueprint | random | fixed
    fixed_seats: Optional[Dict[int, str]] = None   # required for table_policy='fixed'
    time_budget_hours: float = 1.0            # wall-clock budget; 0 → unbounded
    n_players: int = 6
    big_blind: int = 100
    small_blind: int = 50
    starting_stack: int = 10_000
    # Deck bounds — full deck (2..14) for the real game, a small deck for tests.
    low_card_rank: int = 2
    high_card_rank: int = 14

    def fingerprint_table_policy(self) -> Dict[str, object]:
        """The table-composition identity folded into ``config_fingerprint`` (§6)."""
        return {"policy": self.table_policy, "fixed_seats": self.fixed_seats}


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

    def new_hero(self, rng: np.random.Generator) -> SearchAgent:
        """A hero :class:`SearchAgent` seeded by ``rng`` (fresh per hand)."""
        return SearchAgent(
            leaf_policies=self.solver_cfg.leaf.policies,
            blueprint_policy=self.blueprint_policy,
            solver_cfg=self.solver_cfg,
            rng=rng,
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
    int, int, np.random.SeedSequence, np.random.SeedSequence, np.random.SeedSequence
]:
    """``(deck_seed, agent_seed, hero_seq, opp_seq, table_seq)`` for one hand.

    Everything is a pure function of ``(run_seed, hand_index)`` via one
    :class:`numpy.random.SeedSequence`, so a resumed run reproduces each hand
    bit-for-bit (§10.1).  ``deck_seed`` seeds the global RNG for the deal;
    ``agent_seed`` (logged) seeds the hero's search sampling; the returned
    sub-sequences seed the hero, opponent-sampling, and seat-assignment RNGs.
    """
    deck_ss, hero_ss, opp_ss, table_ss = np.random.SeedSequence(
        [int(run_seed), int(hand_index)]
    ).spawn(4)
    deck_seed = int(deck_ss.generate_state(1, dtype=np.uint32)[0])
    agent_seed = int(hero_ss.generate_state(1, dtype=np.uint32)[0])
    return deck_seed, agent_seed, hero_ss, opp_ss, table_ss


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


def _to_call(env: PokerEnv) -> int:
    """Chips the current actor must add to call (0 when it can check)."""
    biggest = max(p.n_bet_chips for p in env.players)
    return int(biggest - env.current_player.n_bet_chips)


def _dist_json(actions: List[str], probs) -> str:
    """JSON map ``action → probability`` for ``decisions.action_dist`` (§6)."""
    return json.dumps({a: float(p) for a, p in zip(actions, probs)})


def _capture_hero_decision(
    hero: SearchAgent, env: PokerEnv, action: str, blueprint_policy: Policy
) -> DecisionRow:
    """Build the ``decisions`` row for the hero's just-chosen (un-stepped) action.

    Called with ``env`` still at the decision node (``act`` does not step), so the
    decision context (pot / to-call / stack / live count) and the played action
    distribution are read directly.  ``searched`` distinguishes a real solve
    (rounds 2–4, or a triggered round-1 search) from a round-1 blueprint play; the
    solver-run columns come from ``hero.last_search`` (populated in step 1).
    """
    stage = _STAGE.get(env.betting_stage, env.betting_stage)
    legal = [a for a in env.legal_actions if a is not None]
    num_live = sum(1 for p in env.players if p.is_active)
    pot_before = float(env.pot_size)
    to_call = float(_to_call(env))
    hero_stack = float(env.players[hero.my_seat].n_chips)

    if hero.last_search is not None:
        res = hero.last_search
        # Read the played distribution under the SAME key the agent sampled from in
        # ``act`` (agent._solved_public_key): a translated near-canonical off-tree
        # node resolves to the canonical branch, so the raw ``env.public_key`` would
        # miss the solved tree and ``strategy_for`` would log a uniform fallback.
        # (Currently on-tree for blueprint opponents, but robust for off-tree ones.)
        pk = hero._solved_public_key(env)
        hr = hero._hand_row(env)
        dist = np.asarray(res.policy.strategy_for(pk, hr, legal), dtype=np.float64)
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
            action_dist=_dist_json(legal, dist),
        )

    # Round-1 blueprint play (no search fired).
    state = env.policy_state_for(hero.my_hole, for_blueprint=True)
    probs = np.asarray(blueprint_policy.strategy(state, "none"), dtype=np.float64)
    return DecisionRow(
        betting_stage=stage,
        regime="blueprint",
        searched=0,
        num_live=num_live,
        pot_before=pot_before,
        to_call=to_call,
        hero_stack=hero_stack,
        action_played=action,
        action_dist=_dist_json(list(state.legal_actions), probs),
    )


def play_hand(
    env: PokerEnv,
    hero_seat: int,
    hero: SearchAgent,
    opponents: Mapping[int, BlueprintOpponent],
    opp_rng: np.random.Generator,
    blueprint_policy: Policy,
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
    """
    hero.on_hand_start(env, hero_seat)
    decisions: List[DecisionRow] = []
    recorder = RangeQualityRecorder(hero_seat, opponents.keys())
    seen_board = {int(c) for c in env.community_cards}
    prev_round = env.betting_round

    while not env.is_terminal:
        r = env.betting_round
        if r != prev_round:
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
            decisions.append(
                _capture_hero_decision(hero, env, action, blueprint_policy)
            )
        else:
            action = opponents[seat].sample(env, seat, opp_rng)
        hero.on_observed_action(env_before, seat, action)
        env.step_in_place(action)

    return _finish_hand(env, hero_seat, decisions, recorder)


def _finish_hand(
    env: PokerEnv,
    hero_seat: int,
    decisions: List[DecisionRow],
    recorder: RangeQualityRecorder,
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


def run_evaluation(
    session: EvalSession,
    log: ExperimentLog,
    *,
    now_fn: Callable[[], str] = _now_iso,
    git_sha: Optional[str] = None,
    hostname: Optional[str] = None,
    should_stop: Optional[Callable[[], bool]] = None,
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
        Polled at each hand boundary — the SIGTERM hook wires it (§5); the
        ``VACUUM INTO`` sync-back on stop is a later step.
    max_hands
        Optional hard cap on hands this call (tests; ``None`` → budget-bound only).
    """
    cfg = session.config
    _validate_config(cfg)
    fingerprint = config_fingerprint(session.solver_cfg, cfg.fingerprint_table_policy())
    hand_index = log.next_hand_index(cfg.run_id)
    budget_s = cfg.time_budget_hours * 3600.0
    start = time.monotonic()
    n_attempted = 0
    n_failed = 0
    consecutive_failures = 0

    while True:
        if should_stop is not None and should_stop():
            break
        if budget_s > 0 and (time.monotonic() - start) >= budget_s:
            break
        if max_hands is not None and n_attempted >= max_hands:
            break
        ok = _play_and_log_one(
            session, log, cfg, fingerprint, hand_index, now_fn, git_sha, hostname
        )
        hand_index += 1
        n_attempted += 1
        if ok:
            consecutive_failures = 0
        else:
            n_failed += 1
            consecutive_failures += 1
            if (
                max_consecutive_failures is not None
                and consecutive_failures >= max_consecutive_failures
            ):
                raise RuntimeError(
                    f"aborting run {cfg.run_id!r}: {consecutive_failures} consecutive "
                    f"hand failures (through hand_index {hand_index - 1}) — a "
                    "systematic error, not bad luck; see the hand_failures table."
                )

    if n_failed:
        logger.warning(
            "run %s: %d of %d hands failed this session (see hand_failures)",
            cfg.run_id, n_failed, n_attempted,
        )
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
    deck_seed, agent_seed, hero_ss, opp_ss, table_ss = derive_seeds(
        cfg.run_seed, hand_index
    )
    hero_seat = hand_index % cfg.n_players           # position rotation (§10.1)
    try:
        np.random.seed(deck_seed)
        env = session.new_env()
        seat_labels = assign_seats(
            cfg.table_policy,
            hero_seat,
            cfg.n_players,
            np.random.default_rng(table_ss),
            cfg.fixed_seats,
        )
        hero = session.new_hero(np.random.default_rng(hero_ss))
        opponents = {
            s: session.new_opponent(lbl)
            for s, lbl in seat_labels.items()
            if lbl != HERO_LABEL
        }

        started_at = now_fn()
        outcome = play_hand(
            env, hero_seat, hero, opponents, np.random.default_rng(opp_ss),
            session.blueprint_policy,
        )
        button_seat = _dealer_seat(env)

        with log.game():
            game_id = log.log_game(
                GameRow(
                    run_id=cfg.run_id,
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
                    hero_chips_delta=outcome.hero_chips_delta,
                    went_to_showdown=outcome.went_to_showdown,
                    terminal_street=outcome.terminal_street,
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


def build_blueprint_session(
    cfg: EvalConfig,
    *,
    blueprint_path: str,
    lut_path: str,
    n_rollouts: int = 20,
    use_decision_free_equity: bool = True,
    max_iterations: int = 5_000,
    max_wall_seconds: float = 10.0,
    discount_interval: int = 100,
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
    from environment.action_space import MAX_ACTIONS_PER_STREET
    from information_abstraction.lookup import load_info_set_lut
    from poker_ai.search.leaf import LeafConfig
    from poker_ai.search.policy import BlueprintPolicy
    from poker_ai.tables.cfr_tables import CFRTables
    from poker_ai.tables.warm_start import apply_warm_start_to_tables

    lut = load_info_set_lut(lut_path, pickle_dir=pickle_dir)
    tables = CFRTables(
        index_path=blueprint_path, actions_per_street=MAX_ACTIONS_PER_STREET
    )
    apply_warm_start_to_tables(tables, blueprint_path, cfg.n_players)
    blueprint = BlueprintPolicy(tables, bias_multiplier=bias_multiplier)

    leaf = LeafConfig(
        policies={c: blueprint for c in _BIAS_CLASSES},
        n_rollouts=n_rollouts,
        use_decision_free_equity=use_decision_free_equity,
    )
    solver_cfg = SolverConfig(
        leaf=leaf,
        max_iterations=max_iterations,
        max_wall_seconds=max_wall_seconds,
        discount_interval=discount_interval,
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
    @click.option("--n-players", default=6, type=int, show_default=True)
    @click.option("--big-blind", default=100, type=int, show_default=True)
    @click.option("--small-blind", default=50, type=int, show_default=True)
    @click.option("--starting-stack", default=10_000, type=int, show_default=True)
    @click.option("--max-iterations", default=5_000, type=int, show_default=True)
    @click.option("--max-wall-seconds", default=10.0, type=float, show_default=True)
    @click.option("--workers", default=None, type=int, help="Solver replicas (§6.7).")
    def run(**opts):
        """Play a time-budgeted evaluation run, logging one transaction per hand."""
        fixed_seats = None
        if opts["fixed_seats"]:
            fixed_seats = {int(k): v for k, v in json.loads(opts["fixed_seats"]).items()}
        cfg = EvalConfig(
            run_id=opts["run_id"],
            run_seed=opts["run_seed"],
            table_policy=opts["table_policy"],
            fixed_seats=fixed_seats,
            time_budget_hours=opts["time_budget_hours"],
            n_players=opts["n_players"],
            big_blind=opts["big_blind"],
            small_blind=opts["small_blind"],
            starting_stack=opts["starting_stack"],
        )
        db_path = Path(opts["db_path"])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(db_path.parent / "config.yaml", "w") as fh:
            yaml.dump(opts, fh)  # provenance, mirrors the blueprint runner

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
        try:
            n = run_evaluation(
                session,
                log,
                git_sha=_git_sha(),
                hostname=socket.gethostname(),
                should_stop=should_stop,
            )
        finally:
            log.close()
        click.echo(f"played {n} hands for run_id={cfg.run_id} → {db_path}")

    return evaluate


# The click group, built lazily so importing the runner core costs no CLI deps.
evaluate = _cli()


if __name__ == "__main__":
    evaluate()
