"""Search iteration/wall-budget calibration for the real-time solver (cluster).

The per-subgame iteration budget (:mod:`poker_ai.search.budget`) and the wall
backstop (``SolverConfig.max_wall_seconds``) are *uncalibrated* on real 52-card
multiway hardware — the defaults were sized on the 20-card proxy.  Now that a real
multiway blueprint exists, this tool measures, **per solver path and betting round**,
the two quantities that make a budget feasible:

- **Throughput** (it/s at the production worker count) — machine-dependent; sets
  ``wall = per_replica_iters / it_s`` (+ the fork/merge fixed cost of a parallel
  solve).
- **Convergence** — machine-independent; how much **value is still on the table** at
  budget ``t`` vs a long reference budget: ``|v_t − v_ref|`` of the hero's root
  counterfactual EV for its actual hand, in mbb.  A *value* gap (not a strategy-
  distribution L1) because value is invariant across equivalent equilibria — the
  full-policy L1 never vanishes at an indifferent infoset (its mixture is free) and
  over-weights cold, never-played tail mass, which also makes it unfair to the
  noisier MCCFR average.  Hot-action L1 and argmax-stability on the played decision
  are kept as secondary diagnostics.  On 52-card there is no equilibrium oracle, so
  this self-referential value gap is the honest, oracle-free budget signal.

It reuses the evaluation stack end-to-end: it plays real hands with the trained
blueprint + bias opponents (:mod:`evaluation.runner` / :mod:`evaluation.opponents`),
and a :class:`CalibrationAgent` — a thin :class:`~poker_ai.search.agent.SearchAgent`
subclass — intercepts each searched decision to capture the *exact* production
:class:`~poker_ai.search.context.SubgameContext` (ranges, models, regime routing,
live-player count).  Those captured roots are then re-solved at a geometric ladder
of per-replica budgets; the resulting per-cell curves yield a suggested
``mccfr_per_player_by_street`` / ``vector_budget_by_street`` block.

A **cell** is ``(condition, regime, street, n_live)`` — i.e. exactly the axes the
budget keys on (``budget._is_vector`` + ``budget.iteration_budget``'s per-street /
per-live-player scaling).  ``condition`` is ``vanilla`` or a ``DBR`` arm; vanilla and
DBR share the regime, engine, and game tree, so their budgets are expected to match —
running both here *verifies* that rather than assuming it.

Run it on the production node via ``scripts/calibrate_search.sh`` (stages the LUT +
blueprint to node-local scratch, sets ``PLURIBUS_SEARCH_CORE=1``, and picks the
worker count up from ``SLURM_CPUS_PER_TASK``).  Standalone::

    python -m evaluation.calibrate run \
        --blueprint-path <bp> --lut-path <lut> --n-players 4 \
        --conditions vanilla,DBR --model-p-max 1.0 \
        --out-dir calibration_out
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from evaluation.aivat import AivatAccumulator, LeafValue, _preserve_global_random
from evaluation.opponents import HERO_LABEL, ModelSpec
from evaluation.runner import (
    EvalConfig,
    EvalSession,
    assign_seats,
    build_blueprint_session,
    derive_seeds,
    play_hand,
)
from poker_ai.search.agent import SearchAgent
from poker_ai.search.budget import iteration_budget
from poker_ai.search.context import SubgameContext
from poker_ai.search.parallel import resolve_workers
from poker_ai.search.solver import _select_regime, solve
from poker_ai.search.solver_state import SolverConfig

logger = logging.getLogger(__name__)

# A cell — the axes the budget keys on (mirrors budget.iteration_budget).
Cell = Tuple[str, str, int, int]  # (condition, regime, street, n_live)

_STREET_NAME = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}


# --------------------------------------------------------------------------- #
# Captured root
# --------------------------------------------------------------------------- #
@dataclass
class RootSample:
    """One captured production subgame root, re-solvable at any budget.

    ``env`` is a private deepcopy (a solve may walk it in place but restores it; we
    still deepcopy per re-solve defensively).  ``ctx`` is the frozen
    :class:`SubgameContext` the agent built — reused verbatim, only its ``rng`` is
    swapped per replicate.  ``(pk, hr, legal)`` locate the hero's root decision row
    so the swept solve's average strategy can be read back for the L1 metric.
    """

    condition: str
    regime: str
    street: int
    n_live: int
    env: object
    ctx: SubgameContext
    pk: object
    hr: int
    legal: List[str]

    @property
    def cell(self) -> Cell:
        return (self.condition, self.regime, int(self.street), int(self.n_live))


class _Collector:
    """Accumulates captured roots, capped ``per_cell_cap`` per cell."""

    def __init__(self, condition: str, per_cell_cap: int) -> None:
        self.condition = condition
        self.per_cell_cap = int(per_cell_cap)
        self.samples: Dict[Cell, List[RootSample]] = defaultdict(list)

    def capture(self, agent: "CalibrationAgent", root_env) -> None:
        ctx = agent._ctx
        if ctx is None:
            return
        regime = _select_regime(ctx)
        street = int(ctx.street_at_root)
        n_live = len(ctx.ranges)
        cell = (self.condition, regime, street, n_live)
        if len(self.samples[cell]) >= self.per_cell_cap:
            return
        my_hole = tuple(int(c) for c in agent.my_hole)
        try:
            pk = root_env.public_key
            hr = int(root_env.combo_index[my_hole])
            legal = [a for a in root_env.legal_actions if a is not None]
        except Exception:  # pragma: no cover - defensive
            logger.exception("root capture failed to read the decision row")
            return
        if not legal:
            return
        self.samples[cell].append(
            RootSample(self.condition, regime, street, n_live,
                       copy.deepcopy(root_env), ctx, pk, hr, list(legal))
        )

    def total(self) -> int:
        return sum(len(v) for v in self.samples.values())

    def all_capped(self) -> bool:
        return bool(self.samples) and all(
            len(v) >= self.per_cell_cap for v in self.samples.values()
        )


class CalibrationAgent(SearchAgent):
    """A :class:`SearchAgent` that records every searched root it builds.

    It runs the normal (cheap-budget) solve so the hand advances realistically, then
    hands the just-built context to the collector.  The cheap budget only advances
    play — the captured ``ctx`` is budget-independent (the budget lives on the
    ``SolverConfig``, not the context), so the calibration sweep re-solves it at the
    real production budgets afterward.
    """

    def __init__(self, *args, collector: _Collector, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._collector = collector

    def _solve_and_store(self, root_env) -> None:  # type: ignore[override]
        super()._solve_and_store(root_env)
        try:
            self._collector.capture(self, root_env)
        except Exception:  # pragma: no cover - capture must never break a hand
            logger.exception("calibration root capture raised")


# --------------------------------------------------------------------------- #
# Phase 1 — collect realistic roots by playing hands
# --------------------------------------------------------------------------- #
def collect_roots(
    session: EvalSession,
    cfg: EvalConfig,
    collect_cfg: SolverConfig,
    condition: str,
    *,
    n_hands: int,
    per_cell_cap: int,
    run_seed: int,
) -> Dict[Cell, List[RootSample]]:
    """Play up to ``n_hands`` hands, capturing ≤ ``per_cell_cap`` roots per cell.

    Mirrors :func:`evaluation.runner.run_evaluation`'s per-hand assembly (same deal /
    seat / model / opponent construction, CRN-seeded), swapping in a
    :class:`CalibrationAgent` hero on the cheap ``collect_cfg`` budget.  Stops early
    once every discovered cell is capped.
    """
    collector = _Collector(condition, per_cell_cap)
    for hand_index in range(n_hands):
        deck_seed, _agent_seed, hero_ss, opp_ss, table_ss, _aivat_ss = derive_seeds(
            run_seed, hand_index
        )
        hero_seat = hand_index % cfg.n_players
        try:
            np.random.seed(deck_seed)
            env = session.new_env()
            seat_labels = assign_seats(
                cfg.table_policy, hero_seat, cfg.n_players,
                np.random.default_rng(table_ss), cfg.fixed_seats,
            )
            models = session.build_models(seat_labels)
            hero = CalibrationAgent(
                leaf_policies=collect_cfg.leaf.policies,
                blueprint_policy=session.blueprint_policy,
                solver_cfg=collect_cfg,
                rng=np.random.default_rng(hero_ss),
                models=models,
                search_enabled=True,
                collector=collector,
            )
            opponents = {
                s: session.new_opponent(lbl)
                for s, lbl in seat_labels.items()
                if lbl != HERO_LABEL
            }
            play_hand(
                env, hero_seat, hero, opponents,
                np.random.default_rng(opp_ss), session.blueprint_policy, aivat=None,
            )
        except Exception:  # a single bad hand must not abort collection
            logger.exception("calibration hand %d failed during collection", hand_index)
        if collector.all_capped():
            break
    logger.info(
        "collected %d roots across %d cells (condition=%s)",
        collector.total(), len(collector.samples), condition,
    )
    return dict(collector.samples)


# --------------------------------------------------------------------------- #
# Phase 1' — CONSTRUCT roots directly (deterministic, full cell coverage)
# --------------------------------------------------------------------------- #
def _target_cells(n_players: int) -> List[Tuple[int, int]]:
    """The ``(street, n_live)`` grid the sweep should cover for an ``n_players`` game.

    Pre-flop starts with everyone in (one cell at ``n_live == n_players``); each
    post-flop street can be reached with any live count from heads-up up to the full
    table.  The regime (vector vs mccfr) is *derived* per constructed root, not chosen
    here, so ``(turn, 2)`` becomes a vector cell and ``(flop, 3)`` an mccfr cell.
    """
    cells = [(0, n_players)]
    for street in (1, 2, 3):                       # flop, turn, river
        for n_live in range(2, n_players + 1):
            cells.append((street, n_live))
    return cells


def _uniform_range(env) -> np.ndarray:
    """No-information belief: uniform over board-compatible combos, normalised.

    The production belief (``RangeTracker.snapshot``) is a board-masked, sum-1 weight
    vector over ``combo_cards``; this is its maximum-entropy stand-in.  Approximate by
    design (§ construction) — the iteration budget depends on the subgame's structural
    size, not the exact opponent range."""
    combo_cards = env.combo_cards
    n = combo_cards.shape[0]
    community = [int(c) for c in env.community_cards]
    if community:
        board = np.asarray(community, dtype=combo_cards.dtype)
        bc = ~(np.isin(combo_cards[:, 0], board) | np.isin(combo_cards[:, 1], board))
    else:
        bc = np.ones(n, dtype=bool)
    w = bc.astype(np.float64)
    s = w.sum()
    return (w / s) if s > 0 else np.full(n, 1.0 / n, dtype=np.float64)


def _drive_to(env, hero_seat: int, target_street: int, n_live: int,
              max_steps: int = 400) -> bool:
    """Drive ``env`` down a passive scripted line to ``hero``'s decision on
    ``target_street`` with exactly ``n_live`` seats still in.

    Pre-flop, fold ``n_players - n_live`` non-hero seats (never the hero); everyone
    else plays the cheapest stay-in action (check, else call).  No raises are ever
    made, so each round is a single pass and the hero always reaches its turn.
    Returns ``True`` iff the env is now non-terminal with the hero to act on
    ``target_street`` (the constructed root); ``False`` on any degenerate line."""
    to_fold = env.n_players - n_live
    steps = 0
    while not env.is_terminal and steps < max_steps:
        steps += 1
        if env.betting_round == target_street and env.player_i == hero_seat:
            return True
        seat = env.player_i
        legal = [a for a in env.legal_actions if a is not None]
        if not legal:
            return False
        if (env.betting_round == 0 and to_fold > 0 and seat != hero_seat
                and "fold" in legal):
            env.step_in_place("fold")
            to_fold -= 1
            continue
        act = next((a for a in ("check", "call") if a in legal), None)
        if act is None:                            # facing a bet with no passive reply
            if seat == hero_seat:                  # the hero must never fold itself out
                act = next((a for a in ("call", "all_in", "check") if a in legal),
                           legal[0])
            else:
                act = "fold" if "fold" in legal else legal[0]
        env.step_in_place(act)
    return (not env.is_terminal and env.betting_round == target_street
            and env.player_i == hero_seat)


def _construct_one(session: EvalSession, cfg: EvalConfig, collect_cfg: SolverConfig,
                   condition: str, street: int, n_live: int, run_seed: int,
                   idx: int) -> Optional[RootSample]:
    """Build ONE deterministic root for cell ``(street, n_live)`` (seed = ``idx``)."""
    deck_seed, _a, hero_ss, _o, table_ss, _v = derive_seeds(run_seed, idx)
    hero_seat = idx % cfg.n_players
    np.random.seed(deck_seed)                      # engine deals off global np.random
    env = session.new_env()
    seat_labels = assign_seats(
        cfg.table_policy, hero_seat, cfg.n_players,
        np.random.default_rng(table_ss), cfg.fixed_seats,
    )
    # Hero excludes its own seat from the model map, exactly as ``on_hand_start`` does.
    models = {int(s): m for s, m in session.build_models(seat_labels).items()
              if int(s) != hero_seat}
    if not _drive_to(env, hero_seat, street, n_live):
        return None
    live = [s for s in range(cfg.n_players) if env.players[s].is_active]
    if len(live) != n_live or hero_seat not in live:
        return None
    uni = _uniform_range(env)
    ranges = {s: uni.copy() for s in live}
    folded = {s: uni.copy() for s in range(cfg.n_players)
              if not env.players[s].is_active}
    my_hole = tuple(int(c) for c in env.players[hero_seat].cards)
    ctx = SubgameContext.from_runtime(
        env, hero_seat, my_hole, ranges, folded, collect_cfg.leaf,
        np.random.default_rng(hero_ss), models=models,
    )
    legal = [a for a in env.legal_actions if a is not None]
    if not legal:
        return None
    try:
        hr = int(env.combo_index[tuple(sorted(my_hole))])
    except KeyError:                               # pragma: no cover - defensive
        return None
    regime = _select_regime(ctx)
    return RootSample(condition, regime, int(street), int(len(ranges)),
                      copy.deepcopy(env), ctx, env.public_key, hr, list(legal))


def construct_roots(
    session: EvalSession,
    cfg: EvalConfig,
    collect_cfg: SolverConfig,
    condition: str,
    *,
    per_cell: int,
    run_seed: int,
) -> Dict[Cell, List[RootSample]]:
    """CONSTRUCT ``per_cell`` roots for EVERY ``(street, n_live)`` cell, deterministically.

    Playing hands under-samples rare production spots — multiway and heads-up
    turn/river are a tiny fraction of hands — so a sampled sweep starves those cells
    (and the vector-vs-mccfr A/B never fires without HU turn/river roots).  Instead we
    build each situation directly: a fresh seeded deal driven down a passive line
    (:func:`_drive_to`) to the target street and live count, with maximum-entropy
    (uniform, board-masked) beliefs.  Deterministic per ``(run_seed, street, n_live,
    k)``; the roots are approximate (not the production belief distribution) but carry
    the right ``(street, n_live, regime)`` structure, which is what the iteration
    budget is calibrated on.  Same cell/regime routing (:func:`_select_regime`) and
    :class:`RootSample` shape as :func:`collect_roots`, so the sweep is unchanged."""
    n_players = int(cfg.n_players)
    out: Dict[Cell, List[RootSample]] = defaultdict(list)
    for street, n_live in _target_cells(n_players):
        for k in range(per_cell):
            idx = (street * (n_players + 1) + n_live) * per_cell + k
            try:
                sample = _construct_one(session, cfg, collect_cfg, condition,
                                        street, n_live, run_seed, idx)
            except Exception:  # a single bad construction must not abort the grid
                logger.exception("root construction failed (street=%s n_live=%s k=%s)",
                                 street, n_live, k)
                sample = None
            if sample is not None:
                out[sample.cell].append(sample)
    logger.info("constructed %d roots across %d cells (condition=%s)",
                sum(len(v) for v in out.values()), len(out), condition)
    return dict(out)


# --------------------------------------------------------------------------- #
# Phase 1'' — low-variance root value via AIVAT (§ calibration variance)
# --------------------------------------------------------------------------- #
class _BeliefTracker:
    """Minimal ``RangeTracker`` stand-in exposing the ctx belief to AIVAT's sampler.

    :meth:`LeafValue._sample_joint` reads ``hero.tracker.snapshot()`` /
    ``folded_snapshot()`` for the per-seat opponent ranges; the calibration has no
    live agent, so we hand it the ctx's frozen ``ranges`` / ``folded_ranges``."""

    def __init__(self, ranges, folded) -> None:
        self._live = {int(s): np.asarray(w, dtype=np.float64) for s, w in ranges.items()}
        self._folded = {int(s): np.asarray(w, dtype=np.float64)
                        for s, w in (folded or {}).items()}

    def snapshot(self):
        return self._live

    def folded_snapshot(self):
        return self._folded


class _CalibHero:
    """The ``hero`` interface AIVAT's :class:`LeafValue` needs (seat, hole, belief)."""

    def __init__(self, ctx) -> None:
        self.my_seat = int(ctx.my_seat)
        self.my_hole = tuple(int(c) for c in ctx.my_hole)
        self.tracker = _BeliefTracker(ctx.ranges, getattr(ctx, "folded_ranges", {}))


def _blueprint_none(ctx):
    """The unbiased (``"none"``) leaf policy — the blueprint used off the solved tree."""
    pols = ctx.leaf.policies
    return pols.get("none") or next(iter(pols.values()))


def _aligned_probs(policy, env, hole, legal: List[str]) -> np.ndarray:
    """Blueprint action distribution for ``hole`` at ``env``, aligned to ``legal``."""
    state = env.policy_state_for(tuple(int(c) for c in hole), for_blueprint=True)
    p = np.asarray(policy.strategy(state, "none"), dtype=np.float64)
    idx = {a: j for j, a in enumerate(state.legal_actions)}
    out = np.zeros(len(legal), dtype=np.float64)
    for j, a in enumerate(legal):
        if a in idx:
            out[j] = p[idx[a]]
    s = out.sum()
    return out / s if s > 0 else np.full(len(legal), 1.0 / len(legal))


def aivat_root_value(res, root_env, ctx, *, rollouts: int, hole_samples: int,
                     rng: np.random.Generator) -> Optional[float]:
    """AIVAT estimate of the hero's root EV under the SOLVED strategy ``res``.

    Replaces the solver's raw internal accumulator (``res.root_value``, a high-variance
    single-combo MC estimate) with a control-variate estimate of the SAME quantity — the
    hero's chip EV playing the solved σ from the root against its belief-sampled
    opponents.  Each of ``rollouts`` playouts samples every seat's hole (hero fixed,
    others from the ctx belief), plays the subgame to a terminal with every actor on the
    solved policy (blueprint off the solved tree), folds an AIVAT control-variate term at
    each action node (:meth:`AivatAccumulator.correct_action`) and the exact runout
    chance-correction at all-in terminals (``max_runout_cards=5`` also covers a pre-flop
    all-in, ``runout_cap`` bounding its Monte-Carlo board sample).  The average is far
    lower variance than the internal counter, so the calibration's value gap carries the
    convergence signal instead of the noise floor.  ``None`` if nothing evaluated."""
    pol = (res.average_policy if getattr(res, "ox_enter_prob", None) is not None
           else res.policy)
    legal_at = res.state.legal_at
    combo_index = root_env.combo_index
    hero_seat = int(ctx.my_seat)
    root_street = int(ctx.street_at_root)
    lv = LeafValue(_CalibHero(ctx), ctx.leaf, rng, n_hole_samples=int(hole_samples))
    bp = _blueprint_none(ctx)

    def _one_rollout() -> float:
        holes = lv._sample_joint(root_env)          # hero fixed, others from belief
        env = root_env.with_hole_cards(holes)
        acc = AivatAccumulator(hero_seat, lv, rng, max_runout_cards=5, runout_cap=256)
        guard = 0
        while not env.is_terminal and guard < 400:
            guard += 1
            legal = [a for a in env.legal_actions if a is not None]
            if not legal:
                break
            actor = int(env.player_i)
            pk = env.public_key
            # The solved σ is externally readable ONLY at the root street (combo-keyed);
            # future-street nodes are cluster-keyed internals (the bot re-solves each
            # street — here the continuation is the blueprint, matching the depth-limit
            # leaf).  So consult the search only on the root street + a covered node.
            if int(env.betting_round) == root_street and pk in legal_at:
                hr = combo_index[tuple(sorted(int(c) for c in holes[actor]))]
                probs = np.asarray(pol.strategy_for(pk, hr, legal), dtype=np.float64)
                s = probs.sum()
                probs = probs / s if s > 0 else np.full(len(legal), 1.0 / len(legal))
            else:
                probs = _aligned_probs(bp, env, holes[actor], legal)
            action = legal[int(rng.choice(len(legal), p=probs))]
            # child_values reads env (via an internal with_hole_cards copy) without
            # mutating it, so the pre-action env IS a valid env_before — no deep copy.
            acc.correct_action(env, actor, action, legal, probs)
            env.step_in_place(action)
        return float(acc.finalize(env))

    # The env deals boards off the GLOBAL numpy RNG (with_hole_cards shuffle +
    # step_in_place runouts), so pin it from ``rng`` and restore on exit (CRN-safe):
    # same ``rng`` ⇒ identical playouts ⇒ deterministic value.
    with _preserve_global_random():
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        vals = [_one_rollout() for _ in range(int(rollouts))]
    return float(np.mean(vals)) if vals else None


# --------------------------------------------------------------------------- #
# Phase 2 — sweep each root over an iteration ladder at the production W
# --------------------------------------------------------------------------- #
def _ladder_around(center: int, points: int, *, lo: float, hi: float,
                   ladder_max: int) -> List[int]:
    """Geometric ladder CLUSTERED around a convergence estimate ``center``.

    Spans ``[lo*center, hi*center]``.  ``hi > 1`` puts the top rung — the value-gap
    REFERENCE — safely ABOVE the estimate, so the gap is measured against a (near-)
    converged point rather than the estimate itself (a self-defeating reference); the
    sweep can exceed the production ceiling to reach it.  ``lo < 1`` keeps the min
    feasible (near the estimate, not a token 250 that nothing converges at).  The top is
    capped at ``ladder_max`` to bound the single longest solve's wall (a 3 h SLURM box).
    ``center`` is the cell's own production budget (``iteration_budget`` at W=1) — where
    we believe it converges — so the ladder auto-adapts per street / n_live / regime.
    """
    center = max(1, int(center))
    top = min(int(ladder_max), int(round(hi * center)))
    bot = min(max(1, int(round(lo * center))), top)
    if points < 2 or top <= bot:
        return [top]
    raw = np.geomspace(bot, top, int(points))
    return sorted({int(round(x)) for x in raw})


def _root_sigma(res, s: RootSample) -> Optional[np.ndarray]:
    """The hero's root **average** strategy from a solve, or ``None`` on a miss.

    ``strategy_for`` silently returns uniform for an unseen node, which would look
    (falsely) converged — so we require the root key to be present in the solved
    tree, and treat its absence as a miss (recorded NaN), not a distribution.
    """
    if s.pk not in res.state.legal_at:
        return None
    try:
        sig = res.average_policy.strategy_for(s.pk, s.hr, s.legal)
    except Exception:  # pragma: no cover - defensive
        return None
    return np.asarray(sig, dtype=np.float64)


def _hot_l1(sig: Optional[np.ndarray], ref: Optional[np.ndarray],
            floor: float = 0.05) -> float:
    """L1 restricted to the *hot* actions (max prob ≥ ``floor`` in ``sig`` or ``ref``).

    The full-distribution L1 never vanishes at an indifferent infoset — the mixture
    is free there, so it drifts among equal-value distributions — and it charges
    convergence for cold, never-played tail mass.  Restricting to the actions that
    carry real probability measures the *played* decision instead.
    """
    if sig is None or ref is None:
        return float("nan")
    hot = np.maximum(sig, ref) >= floor
    if not hot.any():
        return 0.0
    return float(np.abs(sig[hot] - ref[hot]).sum())


def _argmax_match(sig: Optional[np.ndarray], ref: Optional[np.ndarray]) -> float:
    """1.0 if the top (played) action matches the reference, else 0.0 (nan on miss).

    Since the bot plays the final iterate's top action, "the played action stops
    flipping" is the operational convergence bar the value gap does not show directly.
    """
    if sig is None or ref is None:
        return float("nan")
    return 1.0 if int(np.argmax(sig)) == int(np.argmax(ref)) else 0.0


def _value_gap_mbb(val: Optional[float], ref_val: Optional[float],
                   big_blind: int) -> float:
    """``|val − ref|`` in milli-big-blinds; nan if either root value is missing.

    ``val`` is the hero's root counterfactual EV for its actual hand (chips) — an
    equilibrium-invariant, hot-path-weighted signal (cold actions contribute ~0 to a
    value), so "value still on the table per budget" is the honest convergence measure.
    """
    if val is None or ref_val is None:
        return float("nan")
    return abs(float(val) - float(ref_val)) / float(big_blind) * 1000.0


@dataclass
class SweepRow:
    condition: str
    regime: str
    street: int
    n_live: int
    per_replica: int
    workers: int
    pooled_iters: int
    wall_seconds: float
    stop_reason: str
    sample: int
    rep: int
    # Primary convergence signal: value still on the table vs the top-budget
    # reference, in milli-big-blinds (equilibrium-invariant, hot-path-weighted).
    value_gap_mbb: float
    root_value: float           # hero root EV for the played hand (chips), or nan
    # Secondary diagnostics on the played decision (the value gap does not show these).
    hot_l1: float               # L1 over hot actions only (≥ 5% mass in t or ref)
    argmax_match: float         # 1.0 if the top action matches the reference, else 0.0
    # Legacy full-policy L1 — kept as a column for continuity, no longer the driver.
    l1_to_ref: float

    @property
    def cell(self) -> Cell:
        return (self.condition, self.regime, int(self.street), int(self.n_live))


# --- Sweep pool hooks (module-level ⇒ fork-inherited cleanly by the hand pool) ------
# Rough per-(regime, street) throughput (it/s) — HU anchors from the calibration, only
# used to ORDER the pool longest-first (never for correctness).  street: flop=1,turn=2,
# river=3 (vector); mccfr also preflop=0.
_LPT_THROUGHPUT = {
    ("vector", 1): 1.3, ("vector", 2): 15.0, ("vector", 3): 186.0,
    ("mccfr", 0): 150.0, ("mccfr", 1): 84.0, ("mccfr", 2): 158.0, ("mccfr", 3): 150.0,
}


def _sweep_setup(worker_id: int, shared: dict):
    """After fork: reopen the leaf-fleet LMDB (MDB_BAD_RSLOT), start a result list."""
    try:
        from poker_ai.search.parallel import _reopen_leaf_fleet_lmdb
        for samples, _fr in shared["jobs"]:
            if samples:
                _reopen_leaf_fleet_lmdb(samples[0].ctx)
                break
    except Exception:  # pragma: no cover - reopen is best-effort
        pass
    return {"results": []}


def _sweep_process(spec_idx: int, state: dict, shared: dict) -> None:
    """Solve one ``(job, sample, rep, budget)`` at search ``workers=1`` (deployment model).

    The seed is ``base + 1000*si + rep`` — job-independent by design, so the A/B pair
    (vector + mccfr jobs over the *same* roots) draws the same seeds it would in a
    sequential per-cell sweep → byte-identical results to the un-pooled version.
    """
    j, si, rep, t = shared["all_specs"][spec_idx]
    samples, force_regime = shared["jobs"][j]
    s = samples[si]
    seed = shared["base_seed"] + 1000 * si + rep
    env_t = copy.deepcopy(s.env)
    ctx_t = dataclasses.replace(s.ctx, rng=np.random.default_rng(seed))
    cfg_t = dataclasses.replace(
        shared["prod_cfg"], auto_budget=False, max_iterations=int(t),
        max_wall_seconds=1e9,
    )
    res = solve(env_t, ctx_t, cfg_t, regime_override=force_regime)
    # Root value for the convergence metric.  The vector regime's internal value is
    # enumeration-EXACT (near-zero variance), so keep it; the MCCFR regime's is a noisy
    # single-combo MC estimate, so replace it with the low-variance AIVAT estimate of the
    # SAME quantity (opt-in ``aivat_value``; the blueprint-continuation proxy's bias
    # cancels in the value gap, and the reduced estimator noise collapses replica_spread).
    row_regime = force_regime if force_regime is not None else s.regime
    root_value = res.root_value
    if shared.get("aivat_value") and str(row_regime) == "mccfr":
        av = aivat_root_value(
            res, env_t, ctx_t,
            rollouts=int(shared["aivat_rollouts"]),
            hole_samples=int(shared["aivat_hole_samples"]),
            rng=np.random.default_rng(seed * 7 + 1),
        )
        if av is not None:
            root_value = av
    state["results"].append((
        j, si, rep, int(t), _root_sigma(res, s), root_value,
        int(res.iterations_run), float(res.wall_seconds), str(res.stop_reason),
    ))


def _sweep_teardown(state: dict):
    return state["results"]


def sweep_jobs(
    jobs: Sequence[dict],
    prod_cfg: SolverConfig,
    *,
    pool_workers: int,
    reps: int,
    base_seed: int,
    big_blind: int,
    aivat_value: bool = False,
    aivat_rollouts: int = 12,
    aivat_hole_samples: int = 8,
) -> List[SweepRow]:
    """Solve EVERY ``(job, sample, rep, budget)`` in ONE core-parallel pool.

    Pooling across cells — instead of one pool per cell — is what keeps all cores busy:
    a cell's cheap low-budget solves backfill the cores freed while another cell's 30k
    top rung is still running (per-cell pools left ~85% of the box idle during each
    cell's deepest rung).  Each solve runs the search at ``workers=1`` (the deployment
    model, one hand per core, no nested pool), forcing ``auto_budget=False`` +
    ``max_iterations = t`` + a huge wall cap so the wall each reports is the honest
    per-search cost and the budget always completes.

    A **job** = ``{cell, samples, force_regime, ladder, ref_from}``.  ``force_regime``
    overrides regime routing (the A/B solves the same roots under both).  ``ref_from``
    (a job index, or ``None``) sets the value-gap reference: the A/B mccfr job points at
    its paired *vector* job so its gap is measured against the vector-exact value; every
    other job self-references its own top budget.  The reference is resolved in
    post-processing (solves don't depend on it), which is what lets all jobs share one
    pool.  The **primary** convergence signal is the hero root-value gap (mbb, hot-path);
    hot-L1 + argmax-stability are diagnostics; full-policy L1 is a column only.
    """
    from evaluation.hand_pool import run_index_pool

    jobs = list(jobs)
    all_specs = [(j, si, rep, int(t))
                 for j, job in enumerate(jobs)
                 for si in range(len(job["samples"]))
                 for rep in range(reps) for t in job["ladder"]]
    if not all_specs:
        return []
    # Longest-processing-time-first: hand out the most expensive solves FIRST so the deep
    # 30k+ rungs overlap the bulk instead of trailing it on a few cores (the pool serves
    # specs in list order).  Cost ~ budget / rough throughput; the throughput guess only
    # affects ORDERING (utilisation), never correctness.
    def _spec_cost(spec):
        j, _si, _rep, t = spec
        cell = jobs[j]["cell"]
        regime = jobs[j].get("force_regime") or cell[1]
        thr = _LPT_THROUGHPUT.get((regime, int(cell[2])), 80.0)
        if regime == "mccfr" and int(cell[3]) > 2:
            thr *= 2.0 / int(cell[3])  # multiway is slower per iteration
        return t / max(1e-6, thr)
    all_specs.sort(key=_spec_cost, reverse=True)
    shared = {
        "jobs": [(job["samples"], job.get("force_regime")) for job in jobs],
        "prod_cfg": prod_cfg, "base_seed": int(base_seed), "all_specs": all_specs,
        "aivat_value": bool(aivat_value), "aivat_rollouts": int(aivat_rollouts),
        "aivat_hole_samples": int(aivat_hole_samples),
    }
    payloads = run_index_pool(
        n_workers=min(int(pool_workers), len(all_specs)) or 1,
        setup=_sweep_setup, process=_sweep_process, teardown=_sweep_teardown,
        shared=shared, target=len(all_specs),
    )
    # Parent-side LMDB reopen after the pool fork (MDB_BAD_RSLOT).
    try:
        from poker_ai.search.parallel import _reopen_leaf_fleet_lmdb
        for job in jobs:
            if job["samples"]:
                _reopen_leaf_fleet_lmdb(job["samples"][0].ctx)
                break
    except Exception:  # pragma: no cover
        pass

    # Demux results per job, keyed (sample, rep, budget).
    by_job: Dict[int, Dict[Tuple[int, int, int], tuple]] = defaultdict(dict)
    for pl in payloads:
        for (j, si, rep, t, sig, val, iters, wall, stop) in (pl or []):
            by_job[j][(si, rep, t)] = (sig, val, iters, wall, stop)

    # First pass: each job's own top-budget value per (sample, rep) — the self-reference
    # and the A/B mccfr job's cross-reference source.
    job_refs: Dict[int, Dict[Tuple[int, int], Optional[float]]] = {}
    for j, job in enumerate(jobs):
        t_ref = list(job["ladder"])[-1]
        by = by_job.get(j, {})
        job_refs[j] = {
            (si, rep): (by.get((si, rep, t_ref)) or (None,) * 2)[1]
            for si in range(len(job["samples"])) for rep in range(reps)
        }

    # Second pass: build rows (now that every job's reference value is known).
    rows: List[SweepRow] = []
    for j, job in enumerate(jobs):
        ladder = list(job["ladder"])
        t_ref = ladder[-1]
        by = by_job.get(j, {})
        samples = job["samples"]
        force_regime = job.get("force_regime")
        ref_from = job.get("ref_from")
        ref_values = job_refs[ref_from] if ref_from is not None else None
        for si, s in enumerate(samples):
            for rep in range(reps):
                top = by.get((si, rep, t_ref))
                ref_sig = top[0] if top is not None else None
                ref_val = (ref_values.get((si, rep)) if ref_values is not None
                           else job_refs[j][(si, rep)])
                row_regime = force_regime if force_regime is not None else s.regime
                for t in ladder:
                    g = by.get((si, rep, t))
                    if g is None:
                        continue
                    sig, val, iters, wall, stop = g
                    l1 = (float(np.abs(sig - ref_sig).sum())
                          if sig is not None and ref_sig is not None else float("nan"))
                    rows.append(SweepRow(
                        condition=s.condition, regime=row_regime, street=s.street,
                        n_live=s.n_live, per_replica=int(t), workers=1,
                        pooled_iters=int(iters), wall_seconds=float(wall), stop_reason=stop,
                        sample=si, rep=rep,
                        value_gap_mbb=_value_gap_mbb(val, ref_val, big_blind),
                        root_value=(float(val) if val is not None else float("nan")),
                        hot_l1=_hot_l1(sig, ref_sig), argmax_match=_argmax_match(sig, ref_sig),
                        l1_to_ref=l1,
                    ))
    return rows


# --------------------------------------------------------------------------- #
# Phase 3 — aggregate → suggested budgets + config block
# --------------------------------------------------------------------------- #
@dataclass
class CellSummary:
    cell: Cell
    ladder: List[int]
    mean_value_gap_mbb: Dict[int, float]      # PRIMARY: value on the table (mbb) per budget
    mean_hot_l1: Dict[int, float]             # hot-action L1 per budget (diagnostic)
    argmax_stability: Dict[int, float]        # P(top action == reference) per budget
    mean_l1: Dict[int, float]                 # legacy full-policy L1 per budget
    mean_wall: Dict[int, float]               # seconds at `workers`
    throughput_it_s: float                    # per-replica it/s at the top budget
    pooled_it_s: float                        # pooled it/s at the top budget
    # Top-budget cross-replica root-value std (mbb): within-sample std across reps,
    # averaged over samples.  A LOW value gap is only trustworthy convergence if THIS is
    # also small — for a no-reference cell (HU flop / multiway, self-referenced gap) it is
    # the sole check that the flattened value is genuine, not a Monte-Carlo noise floor.
    replica_spread_mbb: float
    # Effective convergence bar (mbb) = max(threshold, spread_k * replica_spread): the gap
    # a cell must fall under to count as converged, raised to the noise floor so an
    # unreachable sub-noise absolute threshold does not read as "never converged".
    noise_floor_mbb: float
    suggested: Dict[str, Optional[int]]       # mbb threshold -> smallest converged budget
    n_samples: int
    workers: int


def _first_below(ladder: Sequence[int], curve: Mapping[int, float], thr: float) -> Optional[int]:
    """Smallest ladder budget whose mean value gap is below ``thr`` (mbb).

    ``ladder`` here EXCLUDES the reference (top) budget: the gap at the reference is
    trivially 0 (it is compared against itself), so a cell that only "converges"
    there has not actually converged — it returns ``None`` (unresolved), signalling
    that ``--max-iters`` should be raised.
    """
    for t in ladder:
        v = curve.get(t, float("nan"))
        if not np.isnan(v) and v < thr:
            return int(t)
    return None


def _mean_ignoring_nan(vals: Sequence[float]) -> float:
    clean = [v for v in vals if not np.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


def summarize_cell(cell: Cell, rows: Sequence[SweepRow],
                   thresholds: Sequence[float], big_blind: int = 100,
                   spread_k: float = 1.5) -> CellSummary:
    ladder = sorted({r.per_replica for r in rows})
    gap_t = defaultdict(list)
    hot_t = defaultdict(list)
    amatch_t = defaultdict(list)
    l1_t = defaultdict(list)
    wall_t = defaultdict(list)
    pooled_t = defaultdict(list)
    for r in rows:
        gap_t[r.per_replica].append(r.value_gap_mbb)
        hot_t[r.per_replica].append(r.hot_l1)
        amatch_t[r.per_replica].append(r.argmax_match)
        l1_t[r.per_replica].append(r.l1_to_ref)
        wall_t[r.per_replica].append(r.wall_seconds)
        pooled_t[r.per_replica].append(r.pooled_iters)
    mean_gap = {t: _mean_ignoring_nan(gap_t[t]) for t in ladder}
    mean_hot = {t: _mean_ignoring_nan(hot_t[t]) for t in ladder}
    amatch = {t: _mean_ignoring_nan(amatch_t[t]) for t in ladder}
    mean_l1 = {t: _mean_ignoring_nan(l1_t[t]) for t in ladder}
    mean_wall = {t: float(np.mean(wall_t[t])) for t in ladder}
    t_top = ladder[-1]
    w = rows[0].workers
    top_wall = mean_wall[t_top]
    per_rep_it_s = (t_top / top_wall) if top_wall > 0 else float("nan")
    pooled_it_s = (float(np.mean(pooled_t[t_top])) / top_wall) if top_wall > 0 else float("nan")
    n_samples = len({r.sample for r in rows})
    # Cross-replica disagreement at the top budget: within-sample std of the hero root
    # value across reps, averaged over samples (isolates MCCFR replica noise from the real
    # between-root spread).  Needs ≥2 reps to be defined.
    by_sample_val: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        if r.per_replica == t_top and not np.isnan(r.root_value):
            by_sample_val[r.sample].append(r.root_value)
    within = [float(np.std(v, ddof=1)) for v in by_sample_val.values() if len(v) >= 2]
    replica_spread_mbb = (float(np.mean(within)) / big_blind * 1000.0
                          if within else float("nan"))
    # Convergence bar is SPREAD-RELATIVE: a value gap below an absolute mbb threshold is
    # unreachable when the Monte-Carlo noise floor exceeds it (an MCCFR cell with a 96-mbb
    # replica spread can never dip under 10 mbb), so the effective bar is
    # ``max(threshold, spread_k * replica_spread)`` — "gap has fallen into the noise".  For
    # a deterministic cell (river vector, spread≈0) it collapses back to the absolute
    # threshold.  The suggestion search excludes the reference (top) budget (gap 0 there by
    # construction, so "converged only at the reference" == did not converge).
    below_ref = ladder[:-1]
    noise_floor = (spread_k * replica_spread_mbb
                   if not np.isnan(replica_spread_mbb) else 0.0)
    suggested = {f"{thr:g}": _first_below(below_ref, mean_gap, max(float(thr), noise_floor))
                 for thr in thresholds}
    return CellSummary(
        cell=cell, ladder=ladder, mean_value_gap_mbb=mean_gap, mean_hot_l1=mean_hot,
        argmax_stability=amatch, mean_l1=mean_l1, mean_wall=mean_wall,
        throughput_it_s=per_rep_it_s, pooled_it_s=pooled_it_s,
        replica_spread_mbb=replica_spread_mbb, noise_floor_mbb=noise_floor,
        suggested=suggested, n_samples=n_samples, workers=w,
    )


def _round_up(x: int, step: int = 50) -> int:
    return int(np.ceil(x / step) * step)


def _production_regime(street: int, n_live: int) -> str:
    """The regime production actually routes ``(street, n_live)`` to.

    Mirrors :func:`poker_ai.search.solver._select_regime` on the two axes a cell keys
    on (that function is the source of truth): HU turn/river → vector, else MCCFR.
    Used to drop the turn-A/B *forced-alternate* cells from the production
    ``suggest_config`` (they are measurements of the road-not-taken, not a production
    budget) while keeping them for the A/B report.
    """
    return "vector" if (int(n_live) == 2 and int(street) in (2, 3)) else "mccfr"


def suggest_config(summaries: Sequence[CellSummary], *, threshold: float,
                   default_mccfr_base: Tuple[int, int, int, int] = (3000, 6000, 4000, 3000),
                   default_vector: Tuple[int, int, int] = (1500, 1000, 500),
                   ) -> Dict[str, object]:
    """Fold per-cell suggested budgets into a ``SolverConfig`` budget block.

    Vector per street comes from the heads-up vector cells.  MCCFR ``base[street]`` is
    the per-live-player budget: the suggested per-replica budget divided by the cell's
    live-player count (production budget is ``base * n_live``), taken as the **max** over
    live-player counts so the slowest-converging live-count is covered, rounded up.
    Cells with no converged suggestion at ``threshold`` (mbb value gap) keep the current
    default and are flagged.

    No wall cap is emitted: production uses a single flat ``max_wall_seconds`` backstop
    (the per-street tuple was retired — the calibration only measured HU vector
    turn/river, whose tiny river cap would clip multiway river MCCFR).
    """
    key = f"{threshold:g}"
    mccfr_by_street: Dict[int, int] = {}
    vector_by_street: Dict[int, int] = {}
    unresolved: List[str] = []
    for s in summaries:
        _cond, regime, street, n_live = s.cell
        # Skip turn-A/B forced-alternate cells (a regime production never routes this
        # (street, n_live) to) — they belong in the A/B report, not the prod budget.
        if regime != _production_regime(street, n_live):
            continue
        val = s.suggested.get(key)
        if val is None:
            unresolved.append(
                f"{_STREET_NAME.get(street, street)}/{regime}/n_live={n_live}"
            )
            continue
        if regime == "mccfr":
            # Production budget is base * n_live, so back out the per-player base.
            base = int(np.ceil(int(val) / max(2, int(n_live))))
            mccfr_by_street[street] = max(mccfr_by_street.get(street, 0), base)
        elif regime == "vector":
            vector_by_street[street] = max(vector_by_street.get(street, 0), int(val))

    mccfr_base = [
        _round_up(mccfr_by_street[st]) if st in mccfr_by_street else default_mccfr_base[st]
        for st in (0, 1, 2, 3)
    ]
    # vector_budget_by_street is (flop, turn, river) = streets 1,2,3.
    vector = [
        _round_up(vector_by_street[st]) if st in vector_by_street else default_vector[i]
        for i, st in enumerate((1, 2, 3))
    ]
    return {
        "threshold_mbb": threshold,
        "mccfr_per_player_by_street": tuple(mccfr_base),
        "vector_budget_by_street": tuple(vector),
        "unresolved_cells": unresolved,
    }


# --------------------------------------------------------------------------- #
# Emit
# --------------------------------------------------------------------------- #
def _write_rows_csv(rows: Sequence[SweepRow], path: Path) -> None:
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "regime", "street", "n_live", "per_replica",
                    "workers", "pooled_iters", "wall_seconds", "stop_reason",
                    "sample", "rep", "value_gap_mbb", "root_value", "hot_l1",
                    "argmax_match", "l1_to_ref"])
        for r in rows:
            w.writerow([r.condition, r.regime, r.street, r.n_live, r.per_replica,
                        r.workers, r.pooled_iters, f"{r.wall_seconds:.6f}",
                        r.stop_reason, r.sample, r.rep, f"{r.value_gap_mbb:.6f}",
                        f"{r.root_value:.6f}", f"{r.hot_l1:.6f}",
                        f"{r.argmax_match:.6f}", f"{r.l1_to_ref:.6f}"])


def _gap_at_wall(summary: CellSummary, target_wall: float) -> Tuple[int, float, float]:
    """``(budget, wall, mean value gap mbb)`` at the largest ladder rung within budget.

    "How converged is this regime inside the wall budget."  Picks the deepest budget
    whose mean wall is ≤ ``target_wall`` (most convergence you can afford); if every
    rung already exceeds the target, falls back to the cheapest rung.
    """
    below = [t for t in summary.ladder if summary.mean_wall[t] <= target_wall]
    t = (max(below, key=lambda x: summary.mean_wall[x]) if below
         else min(summary.ladder, key=lambda x: summary.mean_wall[x]))
    return t, summary.mean_wall[t], summary.mean_value_gap_mbb.get(t, float("nan"))


def _regime_ab_pairs(
    summaries: Sequence[CellSummary],
) -> Dict[Tuple[str, int], Dict[str, CellSummary]]:
    """Per ``(condition, street)`` ``{regime: summary}`` for HU cells with BOTH regimes.

    Keyed by ``(condition, street)`` so the vector-vs-MCCFR comparison covers every HU
    postflop street the A/B ran on (flop and/or turn), not just the turn.
    """
    pairs: Dict[Tuple[str, int], Dict[str, CellSummary]] = {}
    for s in summaries:
        cond, regime, street, n_live = s.cell
        if int(n_live) == 2:
            pairs.setdefault((cond, int(street)), {})[regime] = s
    return {k: d for k, d in pairs.items() if "vector" in d and "mccfr" in d}


def _print_regime_ab(summaries: Sequence[CellSummary],
                     wall_target: Optional[float]) -> None:
    """Head-to-head: at a fixed wall, which regime is closer to the exact answer.

    Both regimes' gaps are measured against the SAME vector-exact reference (set up in
    the sweep — vector is exact-per-iteration), so this is a genuine 'distance to truth
    at equal wall' comparison per HU postflop street (flop / turn), not a per-regime
    self-convergence readout.
    """
    pairs = _regime_ab_pairs(summaries)
    if not pairs:
        return
    target = wall_target if wall_target is not None else 20.0
    print("\n" + "=" * 92)
    print(f"REGIME A/B (HU) — value still on the table at ~{target:.0f}s wall "
          "(both vs the vector-exact ref; lower = closer to truth)")
    print("=" * 92)
    hdr = (f"{'condition':<10} {'street':<7} {'regime':<7} {'budget':>7} "
           f"{'wall':>8} {'gap(mbb)':>9}")
    print(hdr)
    print("-" * len(hdr))
    for (cond, street), d in sorted(pairs.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        sname = _STREET_NAME.get(street, street)
        gaps = {}
        for regime in ("vector", "mccfr"):
            t, wall, gap = _gap_at_wall(d[regime], target)
            gaps[regime] = gap
            print(f"{cond:<10} {sname:<7} {regime:<7} {t:>7} {wall:>8.2f} {gap:>9.1f}")
        v, m = gaps["vector"], gaps["mccfr"]
        if not (np.isnan(v) or np.isnan(m)):
            winner = "vector" if v <= m else "mccfr"
            print(f"  -> {sname} at ~{target:.0f}s wall, {winner.upper()} is closer to "
                  f"the exact answer (vector {v:.1f} vs mccfr {m:.1f} mbb)")
    print("=" * 92)


def _print_report(summaries: Sequence[CellSummary], config: Dict[str, object],
                  wall_target: Optional[float], *, core_on: bool = True) -> None:
    print("\n" + "=" * 92)
    print("SEARCH BUDGET CALIBRATION — per-cell throughput & convergence")
    print("=" * 92)
    if not core_on:
        print("!! compiled search core is OFF — throughput/wall are PURE-PYTHON, "
              "NOT production. Re-run with PLURIBUS_SEARCH_CORE=1.")
    hdr = (f"{'condition':<10} {'regime':<7} {'street':<8} {'n_live':>6} "
           f"{'W':>4} {'top_it':>7} {'it/s':>8} {'wall@top':>9} {'gap@top':>8} "
           f"{'sugg':>7} {'wall@sugg':>10}")
    print(hdr)
    print("-" * len(hdr))
    key = f"{config['threshold_mbb']:g}"
    for s in sorted(summaries, key=lambda x: (x.cell[0], x.cell[1], x.cell[2], x.cell[3])):
        cond, regime, street, n_live = s.cell
        t_top = s.ladder[-1]
        sugg = s.suggested.get(key)
        wall_sugg = s.mean_wall.get(sugg) if sugg is not None else None
        # Gap at the *penultimate* budget: gap@top is 0 by construction (self-compare),
        # so the last-below-reference rung is what shows the residual value on the table.
        t_pen = s.ladder[-2] if len(s.ladder) > 1 else t_top
        gap_pen = s.mean_value_gap_mbb.get(t_pen, float("nan"))
        print(f"{cond:<10} {regime:<7} {_STREET_NAME.get(street, street):<8} "
              f"{n_live:>6} {s.workers:>4} {t_top:>7} {s.throughput_it_s:>8.1f} "
              f"{s.mean_wall[t_top]:>9.2f} {gap_pen:>8.1f} "
              f"{(str(sugg) if sugg is not None else '>max'):>7} "
              f"{(f'{wall_sugg:.2f}' if wall_sugg is not None else '-'):>10}")
    if wall_target is not None:
        print("-" * len(hdr))
        print(f"Wall target per decision: {wall_target:.2f}s — cells whose wall@sugg "
              f"exceeds it need a lower budget or heavier blueprint fallback.")
        for s in summaries:
            sugg = s.suggested.get(key)
            wall_sugg = s.mean_wall.get(sugg) if sugg is not None else None
            if wall_sugg is not None and wall_sugg > wall_target:
                cond, regime, street, n_live = s.cell
                print(f"  ! {cond}/{regime}/{_STREET_NAME.get(street, street)}/"
                      f"n_live={n_live}: wall@sugg={wall_sugg:.2f}s > {wall_target:.2f}s")
    print("\nSuggested SolverConfig block (value-gap threshold "
          f"{config['threshold_mbb']:g} mbb):")
    print(f"    mccfr_per_player_by_street = {config['mccfr_per_player_by_street']}"
          "   # (preflop, flop, turn, river); budget = base * n_live")
    print(f"    vector_budget_by_street    = {config['vector_budget_by_street']}"
          "   # (flop, turn, river)")
    if config["unresolved_cells"]:
        print("  NOTE: did not converge below threshold within the ladder (kept default): "
              + ", ".join(config["unresolved_cells"]))
    print("=" * 92 + "\n")
    _print_regime_ab(summaries, wall_target)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_calibration(
    *,
    blueprint_path: str,
    lut_path: str,
    conditions: Sequence[str],
    model_spec: Optional[ModelSpec],
    n_players: int,
    workers: Optional[int],
    variance_reduction: bool = True,
    construct_roots_mode: bool = True,
    aivat_value: bool = True,
    aivat_rollouts: int = 12,
    aivat_hole_samples: int = 8,
    collect_hands: int,
    per_cell_cap: int,
    reps: int,
    ladder_points: int,
    ladder_lo: float,
    ladder_hi: float,
    ladder_max: int,
    thresholds: Sequence[float],
    spread_k: float = 1.5,
    collect_iters: int,
    table_policy: str,
    run_seed: int,
    big_blind: int,
    small_blind: int,
    starting_stack: int,
    low_card_rank: int,
    high_card_rank: int,
    use_decision_free_equity: bool,
    out_dir: Path,
    wall_target: Optional[float],
    regime_ab_streets: Sequence[int] = (2,),
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Each cell's ladder is CLUSTERED around its own convergence estimate — the cell's
    # production budget (``iteration_budget`` at W=1) — with the reference rung above it
    # (``ladder_hi>1``) and a feasible min (``ladder_lo<1``); see ``_ladder_around``.  This
    # auto-adapts per street / n_live / regime (so vector and MCCFR, and each n_live, cluster
    # around their OWN budget) — no hand-set per-street tops.  MCCFR only converges the HOT
    # PATH (cold infosets fall back to blueprint), so its metric is hero-root value STABILITY
    # (self-referenced) NOT exploitability; the reference above the estimate is what makes
    # that gap meaningful rather than self-defeating.

    # The compiled search core is an env var read at IMPORT time (the kernels rebind
    # behind ``PLURIBUS_SEARCH_CORE`` when the search modules are imported), so it must
    # be exported BEFORE this process starts — ``scripts/calibrate_search.sh`` does
    # that (default 1) and verifies liveness.  A bare ``python -m evaluation.calibrate``
    # with the flag unset runs pure-Python, and the measured throughput is then NOT
    # production-representative.  Surface it loudly and record it so the two cannot be
    # confused.
    from poker_ai._core.flags import search_core_enabled
    core_on = bool(search_core_enabled())
    if core_on:
        logger.info("compiled search core: ON (PLURIBUS_SEARCH_CORE) — production throughput")
    else:
        logger.warning(
            "compiled search core is OFF — search runs PURE PYTHON and the measured "
            "throughput/wall are NOT production-representative. Export "
            "PLURIBUS_SEARCH_CORE=1 (and rebuild the extension) before calibrating, "
            "or submit via scripts/calibrate_search.sh."
        )

    # One session per process; SHARE it across conditions so the blueprint LMDB is
    # opened exactly once (opening the same blueprint twice in a process corrupts it).
    # ``build_models`` reads ``session.config.model_spec``, so we replace the session's
    # config per condition (cheap dataclass swap) rather than rebuilding the session.
    # Build each condition's config through ``for_condition`` so its guards fire PER
    # ARM — OX-Search REJECTS a model_spec, DBR REQUIRES one, and only OX sets ``beta``.
    # The old ``dataclasses.replace(base_cfg, condition=..., model_spec=...)`` bypassed
    # those guards: an ``OX(beta=X)`` arm silently kept ``beta`` from ``conditions[0]``
    # (usually None) *and* picked up the DBR ``model_spec``, i.e. it ran as DBR
    # mislabelled "OX" — a silent DBR/OX conflation.
    def _cfg_for(condition: str) -> EvalConfig:
        low = condition.strip().lower()
        # vanilla / blueprint_only / OX take NO opponent model; DBR arms require one.
        spec = (None if (low in ("vanilla", "blueprint_only") or low.startswith("ox"))
                else model_spec)
        return EvalConfig.for_condition(
            condition, model_spec=spec,
            run_id="calibrate", run_seed=run_seed, table_policy=table_policy,
            n_players=n_players, big_blind=big_blind, small_blind=small_blind,
            starting_stack=starting_stack, low_card_rank=low_card_rank,
            high_card_rank=high_card_rank,
        )

    cond_cfgs = {c: _cfg_for(c) for c in conditions}
    # Calibrate opens the blueprint LMDB once and SHARES a single ``solver_cfg`` (hence
    # one ``beta``) across every condition's sweep, so it cannot mix an OX arm (``beta``
    # set) with vanilla/DBR (``beta`` None) in one run.  Assert agreement so that is a
    # LOUD error rather than a silent inflation of one arm into the other's regime.
    _betas = {c: cfg.beta for c, cfg in cond_cfgs.items()}
    if len(set(_betas.values())) > 1:
        raise ValueError(
            "calibrate shares one solver_cfg across conditions, so all conditions must "
            f"share beta; got mixed {_betas}. Run OX-Search conditions in a separate "
            "calibrate invocation from vanilla/DBR."
        )
    base_cfg = cond_cfgs[conditions[0]]
    # Keep the PRODUCTION ``max_iterations`` (build_blueprint_session's default == the
    # config ceiling): the per-cell ladder centre is that production budget, so this must
    # NOT be widened to the ladder max (the sweep exceeds it per-solve via an explicit t).
    session = build_blueprint_session(
        base_cfg, blueprint_path=blueprint_path, lut_path=lut_path,
        use_decision_free_equity=use_decision_free_equity,
        max_wall_seconds=1e9,
    )
    prod_cfg = session.solver_cfg
    # VR-MCCFR (opponent_modeling §5.5): set the flag on the shared prod_cfg so every
    # derived sweep cfg inherits it; it is gated on ``ctx.models`` inside the solver, so
    # it only affects the DBR condition (vanilla/OX carry no models → byte-identical).
    prod_cfg = dataclasses.replace(prod_cfg, variance_reduction=bool(variance_reduction))
    # ``workers`` here sizes the SWEEP's job pool (one core per concurrent solve), not
    # search replicas — every search runs serially (the production deployment model).
    resolved_workers = resolve_workers(workers)
    collect_cfg = dataclasses.replace(
        prod_cfg, auto_budget=False, max_iterations=int(collect_iters),
        max_wall_seconds=1e9,
    )

    # Collect roots for EVERY condition first, building the full job list, then sweep all
    # jobs in ONE cross-cell pool (below) so the whole box stays busy.  A job is a
    # (cell, samples, force_regime, ladder, ref_from) unit; the A/B adds a second
    # (mccfr) job on the same roots whose gap references the paired vector job.
    def _ladder_for(ctx, regime_override) -> List[int]:
        # Centre = the cell's production budget for THIS regime (regime_override forces it,
        # so the A/B mccfr arm on a HU turn reads the mccfr budget, not the vector one).
        center = iteration_budget(ctx, prod_cfg, regime_override=regime_override)
        return _ladder_around(int(center), ladder_points, lo=ladder_lo, hi=ladder_hi,
                              ladder_max=ladder_max)

    jobs: List[dict] = []
    for condition in conditions:
        # Per-arm config from ``for_condition`` (guards enforced): OX ⇒ beta set,
        # no model; DBR ⇒ model, no beta; vanilla ⇒ neither.
        cond_cfg = cond_cfgs[condition]
        session = dataclasses.replace(session, config=cond_cfg)
        if construct_roots_mode:
            # CONSTRUCT roots (deterministic, full cell coverage) — the default, so
            # rare multiway / HU-turn/river cells are never starved by sampling.
            logger.info("constructing roots for condition=%s", condition)
            samples = construct_roots(
                session, cond_cfg, collect_cfg, condition,
                per_cell=per_cell_cap, run_seed=run_seed,
            )
        else:
            logger.info("collecting roots (sampled play) for condition=%s", condition)
            samples = collect_roots(
                session, cond_cfg, collect_cfg, condition,
                n_hands=collect_hands, per_cell_cap=per_cell_cap, run_seed=run_seed,
            )
        # Regime A/B: solve HU roots on the ``regime_ab_streets`` under BOTH regimes.
        # Per condition (the cell key carries the condition), so vanilla AND each DBR arm
        # get their own comparison — the decision can legitimately flip for DBR, because
        # the model clamp concentrates the opponent's effective range, changing both the
        # full-range settlement cost (vector) and the sampling variance (MCCFR).  Vector
        # is exact-per-iteration on every HU postflop street, so vector@top is the shared
        # exact reference both regimes' gaps are measured against.  Disabled under
        # OX-Search (``beta`` set): the gadget root exists solely in the vector regime.
        ab_streets = set(int(s) for s in regime_ab_streets)
        ab_on = bool(ab_streets) and getattr(prod_cfg, "beta", None) is None
        # OX-Search (``beta`` set) only changes the VECTOR regime (the gadget root lives in
        # HU turn/river vector); its MCCFR path is byte-identical to vanilla, so re-sweeping
        # MCCFR cells here would just re-measure the vanilla budget.  So under OX we skip
        # non-vector cells entirely and calibrate the gadget game's vector budgets only.
        ox_on = getattr(prod_cfg, "beta", None) is not None
        for cell, sample_list in samples.items():
            _cond, _regime, street, n_live = cell
            if ox_on and str(_regime) != "vector":
                continue
            ctx0 = sample_list[0].ctx  # all roots in a cell share street / n_live
            is_ab = (int(n_live) == 2 and int(street) in ab_streets)
            if ab_on and is_ab:
                vidx = len(jobs)  # the paired vector job the mccfr job references
                jobs.append(dict(cell=cell, samples=sample_list, force_regime="vector",
                                 ladder=_ladder_for(ctx0, "vector"), ref_from=None))
                jobs.append(dict(cell=cell, samples=sample_list, force_regime="mccfr",
                                 ladder=_ladder_for(ctx0, "mccfr"), ref_from=vidx))
            else:
                jobs.append(dict(cell=cell, samples=sample_list, force_regime=None,
                                 ladder=_ladder_for(ctx0, None), ref_from=None))

    n_solves = sum(len(j["samples"]) * reps * len(list(j["ladder"])) for j in jobs)
    logger.info("sweeping %d jobs (%d cells, %d solves) across %d cores in ONE pool",
                len(jobs), len({j["cell"] for j in jobs}), n_solves, resolved_workers)
    all_rows = sweep_jobs(jobs, prod_cfg, pool_workers=resolved_workers, reps=reps,
                          base_seed=run_seed, big_blind=big_blind,
                          aivat_value=aivat_value, aivat_rollouts=aivat_rollouts,
                          aivat_hole_samples=aivat_hole_samples)

    # Aggregate.
    by_cell: Dict[Cell, List[SweepRow]] = defaultdict(list)
    for r in all_rows:
        by_cell[r.cell].append(r)
    summaries = [summarize_cell(c, rs, thresholds, big_blind=big_blind, spread_k=spread_k)
                 for c, rs in by_cell.items()]

    # Emit — default suggestion uses the middle threshold.
    thr_default = sorted(thresholds)[len(thresholds) // 2]
    config = suggest_config(summaries, threshold=thr_default)

    _write_rows_csv(all_rows, out_dir / "calibration_rows.csv")
    summary_json = {
        "search_core": "on" if core_on else "off (PURE PYTHON — not production)",
        "workers": resolved_workers,
        # Whether VR-MCCFR (opponent_modeling §5.5) was active this run — DBR-only, so
        # it affects only modeled MCCFR cells; recorded so a run is self-documenting
        # (comparing a VR-on vs VR-off summary is meaningless without knowing which).
        "variance_reduction": bool(variance_reduction),
        # Low-variance root value via AIVAT for the MCCFR cells (vector stays exact).
        "aivat_value": {"enabled": bool(aivat_value), "rollouts": int(aivat_rollouts),
                        "hole_samples": int(aivat_hole_samples)} if aivat_value else False,
        "ladder_design": {
            "centre": "per-cell production budget (iteration_budget at W=1)",
            "span": [ladder_lo, ladder_hi], "points": ladder_points, "max": ladder_max,
            "note": "each cell's rungs cluster in [lo*centre, hi*centre]; the top rung "
                    "(hi>1) is the value-gap reference, above the convergence estimate",
        },
        "metric": "value_gap_mbb",  # hero root EV (mbb): hot-path VALUE stability, not
                                    # exploitability (MCCFR converges only the hot path)
        "thresholds_mbb": list(thresholds),
        "suggested_config": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in config.items()
        },
        "cells": [
            {
                "condition": s.cell[0], "regime": s.cell[1],
                "street": _STREET_NAME.get(s.cell[2], s.cell[2]), "n_live": s.cell[3],
                "n_samples": s.n_samples, "throughput_it_s": s.throughput_it_s,
                "pooled_it_s": s.pooled_it_s,
                "replica_spread_mbb": s.replica_spread_mbb,  # noise-floor check on the gap
                "noise_floor_mbb": s.noise_floor_mbb,        # effective convergence bar (mbb)
                "mean_value_gap_mbb": {str(t): s.mean_value_gap_mbb[t] for t in s.ladder},
                "argmax_stability": {str(t): s.argmax_stability[t] for t in s.ladder},
                "mean_hot_l1": {str(t): s.mean_hot_l1[t] for t in s.ladder},
                "mean_l1": {str(t): s.mean_l1[t] for t in s.ladder},
                "mean_wall_seconds": {str(t): s.mean_wall[t] for t in s.ladder},
                "suggested": s.suggested,
            }
            for s in summaries
        ],
    }
    # Regime A/B head-to-head (per condition × HU street): both regimes' value gap at a
    # fixed wall vs the SHARED vector-exact reference — "which regime is closer to truth
    # at equal wall".  Empty unless the A/B ran (vanilla/DBR, ``beta`` off, HU flop/turn).
    ab_target = wall_target if wall_target is not None else 20.0
    ab_pairs = _regime_ab_pairs(summaries)
    if ab_pairs:
        summary_json["regime_ab"] = {
            "wall_target_s": ab_target,
            "note": ("both gaps vs the vector-exact reference; lower = closer to the "
                     "true value; per (condition, street) — DBR can differ from vanilla, "
                     "and the flop can differ from the turn"),
            "cells": [
                {
                    "condition": cond,
                    "street": _STREET_NAME.get(street, street),
                    **{
                        regime: {
                            "budget": _gap_at_wall(d[regime], ab_target)[0],
                            "wall_s": _gap_at_wall(d[regime], ab_target)[1],
                            "value_gap_mbb": _gap_at_wall(d[regime], ab_target)[2],
                        }
                        for regime in ("vector", "mccfr")
                    },
                }
                for (cond, street), d in sorted(ab_pairs.items())
            ],
        }
    (out_dir / "calibration_summary.json").write_text(json.dumps(summary_json, indent=2))
    _print_report(summaries, config, wall_target, core_on=core_on)
    logger.info("wrote %s and %s", out_dir / "calibration_rows.csv",
                out_dir / "calibration_summary.json")
    return summary_json


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli():
    import click

    @click.group()
    def calibrate() -> None:
        """Real-time-search budget calibration."""

    @calibrate.command()
    @click.option("--blueprint-path", required=True, type=str)
    @click.option("--lut-path", required=True, type=str)
    @click.option("--n-players", default=4, type=int, show_default=True)
    @click.option("--conditions", default="vanilla", show_default=True,
                  help="Comma list: 'vanilla', 'DBR', 'blueprint_only'. Both vanilla "
                  "and DBR share the tree, so running both verifies equal budgets.")
    @click.option("--model-p-max", default=1.0, type=float, show_default=True,
                  help="DBR confidence cap (used only for a DBR condition; 1.0 = "
                  "naive best response).")
    @click.option("--model-error", default=0.0, type=float, show_default=True)
    @click.option("--model-seed", default=0, type=int, show_default=True)
    @click.option("--workers", default=None, type=int,
                  help="Sweep pool size — concurrent solves, one core each; each search "
                       "runs serially (default: SLURM_CPUS_PER_TASK-1).")
    @click.option("--variance-reduction/--no-variance-reduction", default=True,
                  show_default=True,
                  help="VR-MCCFR baseline on the DBR MCCFR path (opponent_modeling §5.5). "
                       "Gated on opponent models, so it only affects the DBR condition "
                       "(vanilla/OX are byte-identical either way).  --no-variance-reduction "
                       "reproduces the pre-VR behaviour for an A/B against an earlier run.")
    @click.option("--construct-roots/--sample-roots", default=True, show_default=True,
                  help="Root source. --construct-roots (default) BUILDS one root per "
                       "(street, n_live) cell directly — deterministic, full coverage, "
                       "so rare multiway / HU-turn/river cells are never starved (uniform "
                       "board-masked beliefs; approximate but structurally exact). "
                       "--sample-roots harvests roots by playing --collect-hands hands.")
    @click.option("--aivat-value/--internal-value", default=True, show_default=True,
                  help="Root-value estimator for the convergence metric.  --aivat-value "
                       "(default) replaces the MCCFR cells' noisy internal accumulator with "
                       "a low-variance AIVAT estimate (control variates + exact runout) of "
                       "the same root EV, so replica_spread reflects convergence not MC "
                       "noise; the enumeration-exact VECTOR cells keep their internal value. "
                       "--internal-value reproduces the pre-AIVAT (raw) metric.")
    @click.option("--aivat-rollouts", default=12, type=int, show_default=True,
                  help="Playouts per solve for the AIVAT root-value estimate.")
    @click.option("--aivat-hole-samples", default=8, type=int, show_default=True,
                  help="Opponent-hole belief draws per AIVAT baseline node (higher = "
                       "tighter control variate, more compute).")
    @click.option("--collect-hands", default=400, type=int, show_default=True,
                  help="Hands played to harvest roots when --sample-roots is set.")
    @click.option("--per-cell-cap", default=4, type=int, show_default=True,
                  help="Roots kept per (condition, regime, street, n_live) cell.")
    @click.option("--reps", default=4, type=int, show_default=True,
                  help="Independent re-solve replicates per root; averaging over reps is "
                       "what lowers the MCCFR value-estimate noise floor, so keep ≥4.")
    @click.option("--spread-k", default=1.5, type=float, show_default=True,
                  help="Spread-relative convergence: a cell counts as converged when its "
                       "value gap falls below max(threshold, spread-k * replica_spread) — "
                       "i.e. into its Monte-Carlo noise floor, since a sub-noise absolute "
                       "mbb threshold is unreachable. 0 restores pure absolute thresholds.")
    @click.option("--ladder-points", default=7, type=int, show_default=True,
                  help="Rungs per cell, clustered around its production budget.")
    @click.option("--ladder-lo", default=0.5, type=float, show_default=True,
                  help="Ladder min = lo * (cell production budget) — feasible, near the "
                       "convergence estimate (not a token 250 nothing converges at).")
    @click.option("--ladder-hi", default=2.0, type=float, show_default=True,
                  help="Ladder top (the value-gap REFERENCE) = hi * (production budget); "
                       ">1 so the reference sits ABOVE the convergence estimate. Higher = "
                       "more trustworthy reference but a slower deepest solve.")
    @click.option("--ladder-max", default=50_000, type=int, show_default=True,
                  help="Hard cap on the top rung (bounds the single longest solve's wall on "
                       "a time-limited box); the reference is min(ladder-max, hi*budget).")
    @click.option("--thresholds", default="20,10,5", show_default=True,
                  help="Value-gap convergence thresholds in mbb (hero root EV still on "
                  "the table); the middle one drives the suggested config.")
    @click.option("--collect-iters", default=64, type=int, show_default=True,
                  help="Cheap per-solve budget used only to advance collection hands.")
    @click.option("--table-policy", default="random", show_default=True,
                  help="'random' mixes fold/call/raise bias for street/live-count "
                  "coverage; 'all_blueprint' | 'fixed' also allowed.")
    @click.option("--run-seed", default=0, type=int, show_default=True)
    @click.option("--big-blind", default=100, type=int, show_default=True)
    @click.option("--small-blind", default=50, type=int, show_default=True)
    @click.option("--starting-stack", default=10_000, type=int, show_default=True)
    @click.option("--low-card-rank", default=2, type=int, show_default=True)
    @click.option("--high-card-rank", default=14, type=int, show_default=True)
    @click.option("--no-decision-free-equity", is_flag=True, default=False,
                  help="Disable the exact decision-free leaf equity (debug).")
    @click.option("--wall-target", default=None, type=float,
                  help="Per-decision wall budget (s); flags cells whose suggested "
                  "budget would exceed it, and sets the wall for the turn regime A/B.")
    @click.option("--regime-ab-streets", default="turn", show_default=True,
                  help="Comma list of HU postflop streets (flop,turn,river) to solve "
                  "under BOTH vector and MCCFR vs a shared vector-exact reference — "
                  "compares which regime is closer to truth at equal wall (per condition). "
                  "'none' disables. Auto-off under OX-Search (gadget is vector-only). "
                  "WARNING: 'flop' is SLOW (full-width vector flop ~1.3 it/s) — use a "
                  "small --max-iters/--collect-hands when including it.")
    @click.option("--out-dir", default="calibration_out", type=str, show_default=True)
    def run(**o):
        """Collect roots, sweep budgets, and emit a suggested SolverConfig block."""
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        conditions = [c.strip() for c in o["conditions"].split(",") if c.strip()]
        thresholds = [float(x) for x in o["thresholds"].split(",") if x.strip()]
        if not (0 < o["ladder_lo"] < 1 < o["ladder_hi"]):
            raise click.BadParameter("need 0 < --ladder-lo < 1 < --ladder-hi "
                                     "(min below, reference above, the convergence estimate)")
        need_model = any(c.strip().lower() not in ("vanilla", "blueprint_only")
                         for c in conditions)
        model_spec = (ModelSpec(p_max=float(o["model_p_max"]),
                                error=float(o["model_error"]),
                                seed=int(o["model_seed"]))
                      if need_model else None)
        _street_id = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
        ab_raw = o["regime_ab_streets"].strip().lower()
        regime_ab_streets = () if ab_raw in ("none", "") else tuple(
            _street_id[s.strip()] for s in ab_raw.split(",")
            if s.strip() and s.strip() in _street_id
        )
        run_calibration(
            blueprint_path=o["blueprint_path"], lut_path=o["lut_path"],
            conditions=conditions, model_spec=model_spec,
            n_players=o["n_players"], workers=o["workers"],
            variance_reduction=o["variance_reduction"],
            construct_roots_mode=o["construct_roots"],
            aivat_value=o["aivat_value"], aivat_rollouts=o["aivat_rollouts"],
            aivat_hole_samples=o["aivat_hole_samples"],
            collect_hands=o["collect_hands"], per_cell_cap=o["per_cell_cap"],
            reps=o["reps"], ladder_points=o["ladder_points"], ladder_lo=o["ladder_lo"],
            ladder_hi=o["ladder_hi"], ladder_max=o["ladder_max"], thresholds=thresholds,
            spread_k=o["spread_k"],
            collect_iters=o["collect_iters"], table_policy=o["table_policy"],
            run_seed=o["run_seed"], big_blind=o["big_blind"],
            small_blind=o["small_blind"], starting_stack=o["starting_stack"],
            low_card_rank=o["low_card_rank"], high_card_rank=o["high_card_rank"],
            use_decision_free_equity=not o["no_decision_free_equity"],
            out_dir=Path(o["out_dir"]), wall_target=o["wall_target"],
            regime_ab_streets=regime_ab_streets,
        )

    return calibrate


calibrate = _cli()

if __name__ == "__main__":
    calibrate()
