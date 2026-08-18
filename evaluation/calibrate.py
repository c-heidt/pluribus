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
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from evaluation.aivat import LeafValue, _preserve_global_random
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
    swapped per replicate.  ``(pk, hr, legal)`` locate the hero's root decision row;
    ``pk`` additionally anchors the *street* (``pk[0]``) and identifies the hero seat
    (``actor_at[pk]``) that :func:`_street_sigma` sweeps for the L1 metric.
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
def _target_cells(n_players: int,
                  n_live_filter: Optional[Sequence[int]] = None) -> List[Tuple[int, int]]:
    """The ``(street, n_live)`` grid the sweep should cover for an ``n_players`` game.

    **POST-FLOP ONLY.**  Pre-flop is not calibrated: the bot plays it from the blueprint
    (Pluribus does the same — search starts on the flop), so a pre-flop search budget
    would never be used.  ``mccfr_per_player_by_street[0]`` therefore keeps its default
    and is simply never measured.

    Each post-flop street can be reached with any live count from heads-up up to the full
    table.  The regime (vector vs mccfr) is *derived* per constructed root, not chosen
    here, so ``(turn, 2)`` becomes a vector cell and ``(flop, 3)`` an mccfr cell.

    ``n_live_filter`` (``None`` ⇒ all) restricts the grid to those live counts, so an
    expensive slice can be split into its own run — e.g. ``[2, 3]`` now and ``[4]``
    later.  Splitting is LOSSLESS: each root's seed is keyed on ``(street, n_live, k)``
    (see :func:`construct_roots`), not on its position in this list, so a cell yields
    byte-identical roots whether or not the other cells ran alongside it.
    """
    keep = None if n_live_filter is None else {int(x) for x in n_live_filter}
    cells = []
    for street in (1, 2, 3):                       # flop, turn, river (NO pre-flop)
        for n_live in range(2, n_players + 1):
            cells.append((street, n_live))
    return [c for c in cells if keep is None or c[1] in keep]


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
    per_cell_deterministic: Optional[int] = None,
    n_live_filter: Optional[Sequence[int]] = None,
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
    per_cell_deterministic = (per_cell if per_cell_deterministic is None
                              else int(per_cell_deterministic))
    out: Dict[Cell, List[RootSample]] = defaultdict(list)
    for street, n_live in _target_cells(n_players, n_live_filter):
        # DETERMINISTIC cells (vector river) get MORE hands instead of reps: their reps are
        # byte-identical, so root variety is the only thing extra compute can buy there.
        n_roots = (per_cell_deterministic
                   if _is_deterministic(_production_regime(street, n_live), street)
                   else per_cell)
        for k in range(int(n_roots)):
            # Root index uses a FIXED stride, not ``n_roots`` — so root k of a cell is the
            # same root regardless of how many roots that cell asks for (changing a count
            # never reshuffles the others, and the n_live split stays lossless).
            idx = (street * (n_players + 1) + n_live) * _ROOT_STRIDE + k
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
# Phase 1'' — low-variance root value via COMMON RANDOM NUMBERS (§ calibration variance)
# --------------------------------------------------------------------------- #
class _BeliefTracker:
    """Minimal ``RangeTracker`` stand-in exposing the ctx belief to the hole sampler.

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
    """The ``hero`` interface :class:`LeafValue`'s sampler needs (seat, hole, belief)."""

    def __init__(self, ctx) -> None:
        self.my_seat = int(ctx.my_seat)
        self.my_hole = tuple(int(c) for c in ctx.my_hole)
        self.tracker = _BeliefTracker(ctx.ranges, getattr(ctx, "folded_ranges", {}))


_ROOT_STRIDE = 64
"""Per-cell root-index stride in :func:`construct_roots` — a constant, so a cell's root
``k`` is seeded the same however many roots that cell requests."""


def _is_deterministic(regime: str, street: int) -> bool:
    """Does this cell's solve depend on the RNG at all?

    The **vector river** is fully deterministic: the vector regime enumerates both ranges
    exactly, and the river has no future chance node left to sample — so every seed yields
    a byte-identical solve (the 2026-08 vanilla calibration measured its cross-rep spread
    at exactly 0.0, against 62-75 mbb for the MCCFR cells).  Extra *reps* there are pure
    waste; extra *hands* still buy root variety.  Vector flop/turn still sample their
    remaining board cards, so they are NOT deterministic (turn's spread was 15.8).
    """
    return str(regime) == "vector" and int(street) == 3


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


def crn_root_value(res, root_env, ctx, *, worlds: int,
                   seed: int) -> Optional[float]:
    """Common-random-numbers estimate of the hero's root EV under the SOLVED σ ``res``.

    Replaces the MCCFR solver's raw internal accumulator (``res.root_value``, a
    single-combo value against belief-*sampled* opponents) for the convergence metric.
    The estimate averages the hero-seat chip result of playing the solved σ from the
    root — blueprint continuation off the solved tree, matching the depth-limit leaf —
    over ``worlds`` fixed *worlds*, each an opponent-hole draw plus a board runout drawn
    from ``seed``.

    The point is ``seed``: it is keyed to the ROOT only (the caller shares it across
    replicas AND budgets), so every replica evaluates the SAME cards.  The card-outcome
    variance that dominates a ~100 bb subgame becomes common-mode and cancels in
    ``replica_spread`` and the value gap, leaving only the genuine strategy differences
    the calibration is trying to measure — no control variate, no rollout averaging to
    beat down (the internal accumulator's opponent sampling was the noise; CRN removes
    it at the source).  ``None`` if nothing evaluated.

    CRN robustness under a diverging σ: the worlds' opponent holes and per-world board
    seeds are drawn UP FRONT from ``seed`` (before any trajectory), the betting actions
    are sampled from a SEPARATE per-world stream, and the board (dealt off the GLOBAL
    numpy RNG — ``with_hole_cards`` shuffle + ``step_in_place`` runout) is reseeded to
    that fixed per-world seed.  So a replica whose actions diverge can never desync the
    card stream: the two replicas still meet the identical holes and runout.  The whole
    thing is wrapped to restore the caller's global stream (CRN-safe, deterministic)."""
    pol = (res.average_policy if getattr(res, "ox_enter_prob", None) is not None
           else res.policy)
    legal_at = res.state.legal_at
    combo_index = root_env.combo_index
    hero_seat = int(ctx.my_seat)
    root_street = int(ctx.street_at_root)
    eval_rng = np.random.default_rng(int(seed))
    lv = LeafValue(_CalibHero(ctx), ctx.leaf, eval_rng, n_hole_samples=1)
    bp = _blueprint_none(ctx)
    n = max(1, int(worlds))
    # Draw every world's opponent holes + board seed FIRST, off the shared eval stream,
    # so nothing a divergent trajectory does downstream can perturb the cards.
    joints = [lv._sample_joint(root_env) for _ in range(n)]         # hero fixed, others belief
    board_seeds = [int(x) for x in eval_rng.integers(0, 2 ** 31 - 1, size=n)]

    def _play(holes, board_seed: int) -> Optional[float]:
        act_rng = np.random.default_rng(int(board_seed) ^ 0x9E3779B9)   # own stream (CRN)
        env = root_env.with_hole_cards(holes)
        guard = 0
        while not env.is_terminal and guard < 400:
            guard += 1
            legal = [a for a in env.legal_actions if a is not None]
            if not legal:
                break
            actor = int(env.player_i)
            pk = env.public_key
            # Solved σ is externally readable ONLY at the root street (combo-keyed);
            # future-street nodes are cluster-keyed internals (the bot re-solves each
            # street — here the continuation is the blueprint, as in the depth-limit leaf).
            if int(env.betting_round) == root_street and pk in legal_at:
                hr = combo_index[tuple(sorted(int(c) for c in holes[actor]))]
                probs = np.asarray(pol.strategy_for(pk, hr, legal), dtype=np.float64)
                s = probs.sum()
                probs = probs / s if s > 0 else np.full(len(legal), 1.0 / len(legal))
            else:
                probs = _aligned_probs(bp, env, holes[actor], legal)
            env.step_in_place(legal[int(act_rng.choice(len(legal), p=probs))])
        return float(env.payout[hero_seat]) if env.is_terminal else None

    with _preserve_global_random():
        vals = []
        for holes, bseed in zip(joints, board_seeds):
            np.random.seed(bseed)          # fixed (CRN) hole shuffle + board runout for THIS world
            v = _play(holes, bseed)
            if v is not None:
                vals.append(v)
    return float(np.mean(vals)) if vals else None


# --------------------------------------------------------------------------- #
# Phase 2 — sweep each root over an iteration ladder at the production W
# --------------------------------------------------------------------------- #
def _ladder_wall_anchored(throughput_it_s: float, top_seconds: float, points: int,
                          *, lo_frac: float) -> List[int]:
    """Geometric ladder anchored at **the most iterations that fit in ``top_seconds``**.

    The old ladder was anchored on the production budget (``[0.5C, 2C]``), which wastes
    most of its rungs: the 2026-08 vanilla calibration showed the low rungs are hopelessly
    unconverged (hot_l1 0.5-0.8) and the answer always sits in the top half.  So instead:

    - **top rung** ``T = top_seconds * throughput`` — the deepest solve that fits the wall
      budget for this cell.  This is the reference the hot_l1 self-distance is measured
      against, so it is as converged as the box can afford;
    - **rungs walk DOWN from T** to ``lo_frac * T`` (default 0.35), i.e. the band where
      hot_l1 actually crosses the 0.1-0.2 decision region — no compute spent on budgets
      that are certainly unconverged.

    ``top_seconds`` is per-cell: hard cells (flop, turn) get the full wall, easy ones
    (river converges in seconds) get much less, so no compute is burned proving what is
    already converged.
    """
    top = max(1, int(round(float(throughput_it_s) * float(top_seconds))))
    bot = min(max(1, int(round(lo_frac * top))), top)
    if points < 2 or top <= bot:
        return [top]
    raw = np.geomspace(bot, top, int(points))
    return sorted({int(round(x)) for x in raw})


# --- Throughput probe (sets each cell's wall-anchored ladder top) ------------------
def _probe_setup(worker_id: int, shared: dict):
    try:
        from poker_ai.search.parallel import _reopen_leaf_fleet_lmdb
        for s in shared["probe_samples"]:
            _reopen_leaf_fleet_lmdb(s.ctx)
            break
    except Exception:  # pragma: no cover
        pass
    return {"out": []}


def _probe_process(idx: int, state: dict, shared: dict) -> None:
    """Time ONE wall-bounded solve; record the cell's achieved iterations/second."""
    s = shared["probe_samples"][idx % len(shared["probe_samples"])]
    cfg = dataclasses.replace(
        shared["prod_cfg"], auto_budget=False, max_iterations=10 ** 9,
        max_wall_seconds=float(shared["probe_seconds"]),
    )
    env = copy.deepcopy(s.env)
    ctx = dataclasses.replace(s.ctx, rng=np.random.default_rng(shared["base_seed"] + idx))
    res = solve(env, ctx, cfg, regime_override=shared["probe_regimes"][
        idx % len(shared["probe_samples"])])
    if res.wall_seconds > 0:
        state["out"].append((idx % len(shared["probe_samples"]),
                             float(res.iterations_run) / float(res.wall_seconds)))


def _probe_teardown(state: dict):
    return state["out"]


def probe_throughput(cell_samples: Sequence[tuple], prod_cfg: SolverConfig, *,
                     seconds: float, workers: int, base_seed: int) -> Dict[int, float]:
    """Measure it/s per cell by running short solves that SATURATE the box.

    Throughput under a full box is materially lower than on an idle one (memory
    bandwidth), and the ladder top is a *wall* target — so the probe must run loaded or
    every top rung overshoots its budget.  Each of ``workers`` slots runs one
    ``seconds``-bounded solve, round-robin over the cells, so total probe wall is ~one
    ``seconds`` regardless of cell count; the per-cell **median** is returned.

    ``cell_samples`` is ``[(sample, force_regime), ...]``, one entry per ladder to build.
    """
    from evaluation.hand_pool import run_index_pool

    if not cell_samples:
        return {}
    shared = {
        "probe_samples": [s for s, _r in cell_samples],
        "probe_regimes": [r for _s, r in cell_samples],
        "prod_cfg": prod_cfg, "probe_seconds": float(seconds),
        "base_seed": int(base_seed),
    }
    n = max(len(cell_samples), int(workers))
    payloads = run_index_pool(
        n_workers=min(int(workers), n) or 1, setup=_probe_setup,
        process=_probe_process, teardown=_probe_teardown, shared=shared, target=n,
    )
    by_cell: Dict[int, List[float]] = defaultdict(list)
    for pl in payloads:
        for (i, it_s) in (pl or []):
            by_cell[i].append(it_s)
    return {i: float(np.median(v)) for i, v in by_cell.items() if v}


def _top_seconds_for(regime: str, street: int, mccfr_secs: Sequence[float],
                     vector_secs: Sequence[float]) -> float:
    """Wall budget for a ladder's TOP rung, per regime × street (flop, turn, river).

    Data-driven from the 2026-08 vanilla calibration.  Two things drive it:

    - **hard vs easy** — flop/turn are still moving at their production budgets (hot_l1
      0.23-0.43) so they get the full wall; the river converges in seconds (mccfr n3:
      hot_l1 0.06 at 9000 iters / 22 s), and giving it 600 s would only burn compute.
    - **the ladder must BRACKET the convergence point** — otherwise a cell that settles
      below the lowest rung reports that rung as its budget, a large over-estimate.  The
      river cells settle at wildly different iteration counts by regime (vector 1071 vs
      mccfr 9000), so per-street alone cannot bracket both: vector and MCCFR carry their
      own tops.
    """
    secs = vector_secs if str(regime) == "vector" else mccfr_secs
    flop, turn, river = secs
    return float({1: flop, 2: turn, 3: river}.get(int(street), flop))


# A snapshot of the hero's strategy over the ROOT STREET: ``{public_key: (sigma, reach)}``.
StreetSigma = Dict[Any, Tuple[np.ndarray, float]]


def _street_sigma(policy, s: RootSample) -> Optional[StreetSigma]:
    """The hero's **average** strategy at every decision it can face on the root street.

    Scope — why the street and not just the root, nor the whole depth-limited tree:
    one solve serves the hero for the *whole street*.  The agent re-searches only when
    an off-tree raise is injected (:mod:`poker_ai.search.agent`), so after the root
    decision every further hero node on this street is read out of **this same tree** at
    a deeper public key; only the street boundary triggers a fresh solve.  So the played
    set is exactly ``{hero decision nodes with pk[0] == root stage}``.  Measuring the root
    alone under-measures (it ignores the later, rarer, slowest-converging nodes the bot
    will still play from this solve); measuring to the depth limit over-measures (those
    nodes exist only to give the street's nodes correct leaf values and are thrown away
    and re-solved, so charging convergence for them buys a wastefully large budget).

    ``public_key`` is ``(betting_stage, history)``, so ``pk[0]`` is the street — the
    filter is exact, not a heuristic.  ``actor_at`` picks out the hero's own nodes; the
    row index ``s.hr`` stays valid across the whole street because the board (and hence
    ``combo_index``) does not change within one.

    Each node carries its **reach mass** ``vstrat[pk][hr].sum()`` — the cumulative
    reach-weighted strategy mass this row accumulated, i.e. how often this decision
    actually comes up for the hero's hand.  That weights the aggregate toward the
    decisions the hero really faces instead of letting an all-but-unreachable node
    dominate a flat average.

    Returns ``None`` on a miss (root key absent from the solved tree).  ``strategy_for``
    silently returns uniform for an unseen node, which would read as falsely converged,
    so absence must never be mistaken for a distribution.
    """
    st = policy._state
    if s.pk not in st.legal_at:
        return None
    hero = st.actor_at.get(s.pk)
    if hero is None:
        return None
    stage = s.pk[0]
    out: StreetSigma = {}
    for pk, legal in st.legal_at.items():
        # DBR's opponent meta-game registers ``(pk_base, "META", seat)`` rows in this same
        # table (:mod:`poker_ai.search.mccfr`).  Those are bias-class draws, not the hero's
        # betting decisions, and their action set is ``_BIAS_CLASSES`` — scoring them would
        # corrupt the DBR arm's metric only, i.e. exactly the comparison the calibration
        # exists to make.  A real public key is ``(stage, history)``; require that shape
        # explicitly rather than relying on the meta key's ``[0]`` merely not matching.
        if len(pk) != 2 or pk[0] != stage or st.actor_at.get(pk) != hero:
            continue
        try:
            sig = np.asarray(policy.strategy_for(pk, s.hr, list(legal)),
                             dtype=np.float64)
        except Exception:  # pragma: no cover - defensive
            continue
        mat = st.vstrat.get(pk)
        reach = (float(mat[s.hr].sum())
                 if mat is not None and 0 <= s.hr < mat.shape[0] else 0.0)
        out[pk] = (sig, reach)
    return out or None


def _weighted_node_mean(snap: Optional[StreetSigma], ref: Optional[StreetSigma],
                        per_node) -> float:
    """Reach-weighted mean of ``per_node(sig, ref_sig)`` over the reference's nodes.

    The **reference** (top-budget) snapshot defines both the node set and the weights,
    so every rung of a ladder is scored on the same decisions with the same weights and
    the rungs are directly comparable.  A node the reference reached but the snapshot
    has not yet is scored against **uniform** — not skipped — because uniform is
    literally what ``strategy_for`` returns there, hence what the bot would play at that
    budget.  Falls back to an unweighted mean if every reach is zero (no iterations yet).
    """
    if snap is None or ref is None:
        return float("nan")
    num = den = 0.0
    flat_num = flat_den = 0.0
    for pk, (rsig, w) in ref.items():
        got = snap.get(pk)
        sig = got[0] if got is not None else np.full(rsig.shape, 1.0 / len(rsig))
        if sig.shape != rsig.shape:      # node widened between rungs — skip, not compare
            continue
        d = float(per_node(sig, rsig))
        num += w * d
        den += w
        flat_num += d
        flat_den += 1.0
    if den > 0:
        return num / den
    return flat_num / flat_den if flat_den > 0 else float("nan")


def _hot_l1_node(sig: np.ndarray, ref: np.ndarray, floor: float = 0.05) -> float:
    """L1 restricted to the *hot* actions (max prob ≥ ``floor`` in ``sig`` or ``ref``).

    The full-distribution L1 never vanishes at an indifferent infoset — the mixture
    is free there, so it drifts among equal-value distributions — and it charges
    convergence for cold, never-played tail mass.  Restricting to the actions that
    carry real probability measures the *played* decision instead.
    """
    hot = np.maximum(sig, ref) >= floor
    if not hot.any():
        return 0.0
    return float(np.abs(sig[hot] - ref[hot]).sum())


def _hot_l1(snap: Optional[StreetSigma], ref: Optional[StreetSigma]) -> float:
    """Reach-weighted mean hot-action L1 over the hero's root-street decisions."""
    return _weighted_node_mean(snap, ref, _hot_l1_node)


def _full_l1(snap: Optional[StreetSigma], ref: Optional[StreetSigma]) -> float:
    """Reach-weighted mean *full-distribution* L1 — diagnostic column only."""
    return _weighted_node_mean(
        snap, ref, lambda a, b: float(np.abs(a - b).sum()))


def _argmax_match(snap: Optional[StreetSigma], ref: Optional[StreetSigma]) -> float:
    """Reach-weighted fraction of root-street decisions whose top action matches.

    Since the bot plays the final iterate's top action, "the played action stops
    flipping" is the operational convergence bar the value gap does not show directly.
    Diagnostic: it ignores the mixture, so it flickers at an indifference point and
    cannot drive the budget on its own.
    """
    return _weighted_node_mean(
        snap, ref,
        lambda a, b: 1.0 if int(np.argmax(a)) == int(np.argmax(b)) else 0.0)


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
    # Reach-weighted mean over the hero's ROOT-STREET decisions (see _street_sigma).
    hot_l1: float               # per node: L1 over hot actions only (≥ 5% mass in t or ref)
    argmax_match: float         # per node: 1.0 if the top action matches the reference
    # Full-distribution L1 (cold tail mass included) — diagnostic column, not the driver.
    l1_to_ref: float
    # SAMPLING SCALE (top rung only, sampled cells only; nan elsewhere): mean hot_l1 of
    # this rep's top-budget strategy against the SAME HAND's other reps at the SAME budget.
    # Same hand, same iteration count, different seed ⇒ pure Monte-Carlo dispersion: how
    # far apart two equally-valid solves of this cell land.  NOT a hard floor on ``hot_l1``
    # — that is a NESTED comparison (rung t vs the same solve's top shares iterations
    # 1..t), so its noise partially cancels and it can legitimately sit below this.  It is
    # the scale check: a tolerance far under it certifies a precision the solver does not
    # reproducibly have.
    hot_l1_cross_rep: float = float("nan")

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
    """Run ONE search per ``(job, sample, rep)`` and snapshot EVERY ladder rung from it.

    A solve's trajectory depends only on ``(env, seeded ctx.rng, cfg)`` — never on
    ``max_iterations`` — so the strategy after ``t`` iterations of the top-rung search is
    byte-identical to a separate solve capped at ``t``.  Re-solving per rung therefore
    re-walked the same trajectory ~3.8x over; instead we solve once to the top and read the
    average strategy at each rung (``solve(snapshot_at=...)``).  The saved compute buys
    more reps, which is what actually tightens the metric.

    The seed is ``base + 1000*si + rep`` — job-independent by design, so paired jobs over
    the *same* roots draw the same seeds a sequential per-cell sweep would.
    """
    j, si, rep = shared["all_specs"][spec_idx]
    samples, force_regime = shared["jobs"][j]
    ladder = shared["ladders"][j]
    s = samples[si]
    seed = shared["base_seed"] + 1000 * si + rep
    # Peak-RAM cap: multiway MCCFR solves hold the largest vregret/vstrat tables, so a
    # semaphore bounds how many run at once (the cheap solves keep the other cores busy).
    # ``None`` ⇒ no cap (every worker free-runs, the old behaviour).
    sem = shared.get("mw_sem")
    heavy = sem is not None and spec_idx in shared.get("heavy_idx", frozenset())
    if heavy:
        sem.acquire()
    try:
        env_t = copy.deepcopy(s.env)
        ctx_t = dataclasses.replace(s.ctx, rng=np.random.default_rng(seed))
        top = int(ladder[-1])
        cfg_t = dataclasses.replace(
            shared["prod_cfg"], auto_budget=False, max_iterations=top,
            max_wall_seconds=1e9,
        )
        # One row per rung, captured mid-search.  ``sig`` must be COPIED: the policy reads
        # the live state, which keeps evolving after the snapshot returns.
        snaps: List[tuple] = []

        def _grab(t: int, avg, elapsed: float) -> None:
            # The hero's whole ROOT-STREET strategy (root + every deeper same-street node
            # it will still play out of this one solve), with per-node reach weights.
            try:
                sig = _street_sigma(avg, s)
            except Exception:  # pragma: no cover - defensive
                sig = None
            snaps.append((int(t), sig, float(elapsed)))

        res = solve(env_t, ctx_t, cfg_t, regime_override=force_regime,
                    snapshot_at=ladder, on_snapshot=_grab)
        # Root value — DIAGNOSTIC only (the budget comes from hot_l1 self-stability), so it
        # is read once from the FINAL state and attached to the top rung's row.
        row_regime = force_regime if force_regime is not None else s.regime
        root_value = res.root_value
        if shared.get("crn_value") and str(row_regime) == "mccfr":
            cv = crn_root_value(
                res, env_t, ctx_t,
                worlds=int(shared["crn_worlds"]),
                seed=(int(shared["base_seed"]) + 1) * 100003 + si,
            )
            if cv is not None:
                root_value = cv
        for (t, sig, elapsed) in snaps:
            # A rung below the top completed its full count by construction; only the top
            # can carry the solve's real stop reason.
            stop = str(res.stop_reason) if t >= top else "iteration_cap"
            state["results"].append((
                j, si, rep, int(t), sig,
                (root_value if t >= top else None),
                int(t), float(elapsed), stop,
            ))
    finally:
        if heavy:
            sem.release()


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
    crn_value: bool = False,
    crn_worlds: int = 32,
    max_concurrent_multiway: Optional[int] = None,
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
    # ONE spec per (job, sample, rep): each runs a single search to its ladder TOP and
    # snapshots every rung on the way (see ``_sweep_process``), so the ladder costs one
    # search instead of one per rung.
    # Per-JOB rep count: a deterministic cell (vector river) needs exactly one — every seed
    # gives a byte-identical solve there — while the sampled cells need ``reps``.
    job_reps = [1 if _is_deterministic(job.get("force_regime") or job["cell"][1],
                                       job["cell"][2]) else int(reps)
                for job in jobs]
    all_specs = [(j, si, rep)
                 for j, job in enumerate(jobs)
                 for si in range(len(job["samples"]))
                 for rep in range(job_reps[j])]
    if not all_specs:
        return []
    # Longest-processing-time-first: hand out the most expensive searches FIRST so the deep
    # ones overlap the bulk instead of trailing it on a few cores (the pool serves specs in
    # list order).  Cost ~ top budget / rough throughput; the throughput guess only affects
    # ORDERING (utilisation), never correctness.
    def _spec_cost(spec):
        j, _si, _rep = spec
        t = int(jobs[j]["ladder"][-1])
        cell = jobs[j]["cell"]
        regime = jobs[j].get("force_regime") or cell[1]
        thr = _LPT_THROUGHPUT.get((regime, int(cell[2])), 80.0)
        if regime == "mccfr" and int(cell[3]) > 2:
            thr *= 2.0 / int(cell[3])  # multiway is slower per iteration
        return t / max(1e-6, thr)
    all_specs.sort(key=_spec_cost, reverse=True)

    def _is_multiway(spec):  # the RAM-heavy solves: multiway (>=3 live) MCCFR
        j = spec[0]
        cell = jobs[j]["cell"]
        regime = jobs[j].get("force_regime") or cell[1]
        return regime == "mccfr" and int(cell[3]) >= 3

    # Peak-RAM cap.  Multiway MCCFR holds the biggest tables; front-loading (LPT) makes ALL
    # workers grab them at once — the RAM spike.  With a cap we (1) spread the heavy specs
    # evenly through the schedule so cheap solves stay available to backfill idle cores, and
    # (2) gate the heavy solves through a semaphore so at most K run at once.  Peak RAM then
    # ≈ K * (multiway solve) + (W-K) * (light solve), decoupled from the worker count — you
    # keep throughput but request far less --mem.  ``None`` ⇒ no cap (unchanged behaviour).
    mw_sem = None
    heavy_idx: frozenset = frozenset()
    if max_concurrent_multiway is not None:
        heavy = [s for s in all_specs if _is_multiway(s)]
        light = [s for s in all_specs if not _is_multiway(s)]
        if heavy and light:
            stride = max(1, len(light) // len(heavy))
            ordered, li = [], 0
            for hs in heavy:
                ordered.extend(light[li:li + stride]); li += stride
                ordered.append(hs)
            ordered.extend(light[li:])
            all_specs = ordered
        heavy_idx = frozenset(i for i, s in enumerate(all_specs) if _is_multiway(s))
        k = max(1, int(max_concurrent_multiway))
        if heavy_idx:
            import multiprocessing as _mp
            try:
                mw_sem = _mp.get_context("fork").Semaphore(k)
            except ValueError:  # no fork (non-POSIX) — cap is a no-op there
                mw_sem = None
    shared = {
        "jobs": [(job["samples"], job.get("force_regime")) for job in jobs],
        "ladders": [[int(t) for t in job["ladder"]] for job in jobs],
        "prod_cfg": prod_cfg, "base_seed": int(base_seed), "all_specs": all_specs,
        "crn_value": bool(crn_value), "crn_worlds": int(crn_worlds),
        "mw_sem": mw_sem, "heavy_idx": heavy_idx,
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
            for si in range(len(job["samples"])) for rep in range(job_reps[j])
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
            # Sampling noise floor of the metric itself: this hand's top-budget strategy
            # across reps.  Identical hand and identical iteration count, only the seed
            # differs, so whatever hot_l1 remains here is irreducible MCCFR dispersion —
            # the floor no budget can beat.  Needs >= 2 reps (deterministic cells run 1,
            # and have no sampling noise to measure anyway).
            tops = {r: by[(si, r, t_ref)][0] for r in range(job_reps[j])
                    if by.get((si, r, t_ref)) is not None}
            cross: Dict[int, float] = {}
            if len(tops) >= 2:
                for rep, mine in tops.items():
                    cross[rep] = _mean_ignoring_nan(
                        [_hot_l1(mine, o) for r, o in tops.items() if r != rep])
            for rep in range(job_reps[j]):
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
                    rows.append(SweepRow(
                        condition=s.condition, regime=row_regime, street=s.street,
                        n_live=s.n_live, per_replica=int(t), workers=1,
                        pooled_iters=int(iters), wall_seconds=float(wall), stop_reason=stop,
                        sample=si, rep=rep,
                        value_gap_mbb=_value_gap_mbb(val, ref_val, big_blind),
                        root_value=(float(val) if val is not None else float("nan")),
                        hot_l1=_hot_l1(sig, ref_sig), argmax_match=_argmax_match(sig, ref_sig),
                        l1_to_ref=_full_l1(sig, ref_sig),
                        hot_l1_cross_rep=(cross.get(rep, float("nan"))
                                          if t == t_ref else float("nan")),
                    ))
    return rows


# --------------------------------------------------------------------------- #
# Phase 3 — aggregate → suggested budgets + config block
# --------------------------------------------------------------------------- #
@dataclass
class CellSummary:
    cell: Cell
    ladder: List[int]
    mean_value_gap_mbb: Dict[int, float]      # DIAGNOSTIC: value on the table (mbb) per budget
    mean_hot_l1: Dict[int, float]             # PRIMARY: mean hot-action L1 vs own top budget per budget
    argmax_stability: Dict[int, float]        # P(top action == reference) per budget (diagnostic)
    mean_l1: Dict[int, float]                 # legacy full-policy L1 per budget
    mean_wall: Dict[int, float]               # seconds at `workers`
    throughput_it_s: float                    # per-replica it/s at the top budget
    pooled_it_s: float                        # pooled it/s at the top budget
    # Top-budget cross-replica root-value std (mbb): within-sample std across reps,
    # averaged over samples.  A LOW value gap is only trustworthy convergence if THIS is
    # also small — for a no-reference cell (HU flop / multiway, self-referenced gap) it is
    # the sole check that the flattened value is genuine, not a Monte-Carlo noise floor.
    replica_spread_mbb: float                 # DIAGNOSTIC ONLY (no longer drives the budget)
    # PRIMARY OUTPUT: smallest single-worker budget at which the strategy has stopped moving
    # (worst-case hot_l1 <= tol), EXCLUDING the top reference rung.  ``None`` ⇒ not converged
    # within the ladder ⇒ raise LADDER_HI/MAX.  See :func:`summarize_cell`.
    suggested_budget: Optional[int]
    converged: bool                           # False ⇒ ladder too short (see suggested_budget)
    # True ⇒ already converged at the LOWEST rung, so the real budget is below the ladder
    # and ``suggested_budget`` is an OVER-estimate: lower this regime/street's
    # ``ladder_top_seconds`` to bracket the crossing.  Not used for the emitted config.
    below_ladder: bool
    hot_l1_tol: float                         # the tolerance THIS cell was judged against
    # MEASURED sampling scale: mean cross-rep hot_l1 at the top budget (same hand, same
    # iteration count, different seed).  nan for the deterministic cells and any cell run
    # with a single rep.  See ``SweepRow.hot_l1_cross_rep`` — a scale, not a hard floor.
    cross_rep_hot_l1: float
    # True ⇒ ``hot_l1_tol`` sits at or under that scale: the budget is being picked at a
    # precision finer than two independent seeds of this cell agree to, so it is largely
    # an artefact of which seeds ran.  Loosen the tolerance (or accept a noisy budget).
    tol_below_cross_rep: bool
    n_samples: int
    workers: int


def _first_at_or_below(
    ladder: Sequence[int], curve: Mapping[int, float], tol: float
) -> Optional[int]:
    """Smallest ladder budget whose ``curve`` value is <= ``tol``.

    ``ladder`` here EXCLUDES the reference (top) budget: the self-distance at the reference
    is trivially 0 (compared against itself), so a cell that only "settles" there has not
    actually converged — returns ``None`` (unresolved), signalling that the ladder top
    must be raised.
    """
    for t in ladder:
        v = curve.get(t, float("nan"))
        if not np.isnan(v) and v <= tol:
            return t
    return None


def _first_below(ladder: Sequence[int], curve: Mapping[int, float], thr: float) -> Optional[int]:
    """Smallest ladder budget whose value gap is strictly below ``thr`` (mbb) — DIAGNOSTIC
    only now; the production budget comes from :func:`_first_at_or_below` on ``mean_hot_l1``.
    """
    for t in ladder:
        v = curve.get(t, float("nan"))
        if not np.isnan(v) and v < thr:
            return int(t)
    return None


def _mean_ignoring_nan(vals: Sequence[float]) -> float:
    clean = [v for v in vals if not np.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


# Convergence tolerance is PER REGIME, because the two regimes have different floors.
#
# ``vector`` enumerates both ranges full-width: given the subgame it samples nothing on
# the river and only the river card on the turn, so its average strategy has no
# per-iteration sampling noise to shake off and a tight bar is meaningful.
#
# ``mccfr`` is EXTERNAL SAMPLING.  Its average strategy carries Monte-Carlo dispersion
# that decays like 1/sqrt(T) and never reaches zero, and the street-scoped metric now
# includes the rarely-visited later nodes where the per-node visit count — hence the
# noise — is worst.  Holding it to the vector bar would not measure convergence, it
# would measure sampling noise, and the ladder would report "unresolved" or demand a
# budget bought entirely to average out noise the re-solve at the next street discards.
DEFAULT_HOT_L1_TOL_MCCFR = 0.20
DEFAULT_HOT_L1_TOL_VECTOR = 0.10


def _tol_for(cell: Cell, tol_mccfr: float, tol_vector: float) -> float:
    """Convergence tolerance for ``cell`` — MCCFR is sampled, vector is not (see above)."""
    return float(tol_mccfr if str(cell[1]) == "mccfr" else tol_vector)


def summarize_cell(cell: Cell, rows: Sequence[SweepRow],
                   hot_l1_tol_mccfr: float = DEFAULT_HOT_L1_TOL_MCCFR,
                   hot_l1_tol_vector: float = DEFAULT_HOT_L1_TOL_VECTOR,
                   big_blind: int = 100) -> CellSummary:
    """Per-cell single-worker budget = smallest rung at which the strategy has stopped
    moving.  The signal is MEAN ``hot_l1``: each solve's hot-action strategy over its
    whole ROOT STREET (reach-weighted across the hero's decision nodes, :func:`_street_sigma`)
    against its OWN top-budget strategy, averaged over the cell's solves.  Value gap /
    replica spread / argmax are kept as diagnostics only; they no longer pick the budget
    (value is payoff-leverage-noisy on deep cells, and replica spread is reproducibility,
    not convergence).

    The tolerance is chosen PER REGIME (:func:`_tol_for`) — sampled MCCFR carries
    Monte-Carlo dispersion that full-width vector does not.  ``cross_rep_hot_l1`` measures
    that dispersion directly from the cell's own reps, so the chosen tolerance can be
    checked against the cell's real precision instead of trusted."""
    hot_l1_tol = _tol_for(cell, hot_l1_tol_mccfr, hot_l1_tol_vector)
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
    # DIAGNOSTIC: top-budget cross-rep root-value std (no longer used to pick the budget).
    by_sample_val: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        if r.per_replica == t_top and not np.isnan(r.root_value):
            by_sample_val[r.sample].append(r.root_value)
    within = [float(np.std(v, ddof=1)) for v in by_sample_val.values() if len(v) >= 2]
    replica_spread_mbb = (float(np.mean(within)) / big_blind * 1000.0
                          if within else float("nan"))
    # BUDGET: smallest rung (excluding the top reference) where the MEAN single-worker
    # strategy has settled to within ``hot_l1_tol`` of its own top budget.  None ⇒ the
    # strategy is still moving at the last real rung ⇒ ladder too short (raise LADDER_HI/MAX).
    below_ref = ladder[:-1]
    suggested_budget = _first_at_or_below(below_ref, mean_hot, float(hot_l1_tol))
    # Already converged at the LOWEST rung ⇒ the real budget is somewhere below the ladder,
    # so the lowest rung is an over-estimate, not the answer.  Flag it (``below_ladder``)
    # rather than emitting a budget the data does not support; the fix is a smaller
    # ``ladder_top_seconds`` for that regime/street so the ladder brackets the crossing.
    below_ladder = bool(below_ref) and suggested_budget == below_ref[0]
    # Measured sampling scale: cross-rep hot_l1 at the TOP budget (same hand, same
    # iterations, different seed).  Judging convergence to a tolerance under it means
    # resolving a difference finer than two independent seeds of this cell agree to.
    cross_rep = _mean_ignoring_nan([r.hot_l1_cross_rep for r in rows])
    tol_below_cross = bool(not np.isnan(cross_rep) and hot_l1_tol <= cross_rep)
    return CellSummary(
        cell=cell, ladder=ladder, mean_value_gap_mbb=mean_gap, mean_hot_l1=mean_hot,
        argmax_stability=amatch, mean_l1=mean_l1, mean_wall=mean_wall,
        throughput_it_s=per_rep_it_s, pooled_it_s=pooled_it_s,
        replica_spread_mbb=replica_spread_mbb,
        suggested_budget=suggested_budget, converged=suggested_budget is not None,
        below_ladder=below_ladder,
        hot_l1_tol=float(hot_l1_tol), cross_rep_hot_l1=cross_rep,
        tol_below_cross_rep=tol_below_cross,
        n_samples=n_samples, workers=w,
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


def suggest_config(summaries: Sequence[CellSummary],
                   default_mccfr_base: Tuple[int, int, int, int] = (3000, 5000, 4000, 3000),
                   # MUST track SolverConfig.vector_budget_by_street — this is what an
                   # UNRESOLVED cell falls back to, so a stale value here silently emits a
                   # DOWNGRADE (it read (1500,1000,500) while production ran (1500,1350,850)).
                   default_vector: Tuple[int, int, int] = (1500, 1350, 850),
                   ) -> Dict[str, object]:
    """Fold per-cell suggested budgets into a ``SolverConfig`` budget block.

    Each cell's ``suggested_budget`` is the smallest single-worker budget at which the
    strategy has stopped moving (mean ``hot_l1 <= tol``, the tolerance chosen per regime
    — see :func:`_tol_for`).  Vector per street comes
    from the heads-up vector cells.  MCCFR ``base[street]`` is the per-live-player budget:
    the suggested per-replica budget divided by the cell's live-player count (production
    budget is ``base * n_live``), taken as the **max** over live-player counts so the
    slowest-converging live-count is covered, rounded up.  Cells that never settled within
    the ladder keep the current default and are flagged (raise LADDER_HI/MAX for those).

    No wall cap is emitted: production uses a single flat ``max_wall_seconds`` backstop.
    """
    mccfr_by_street: Dict[int, int] = {}
    vector_by_street: Dict[int, int] = {}
    unresolved: List[str] = []
    for s in summaries:
        _cond, regime, street, n_live = s.cell
        # Skip turn-A/B forced-alternate cells (a regime production never routes this
        # (street, n_live) to) — they belong in the A/B report, not the prod budget.
        if regime != _production_regime(street, n_live):
            continue
        # ``below_ladder`` ⇒ converged at the lowest rung, so this is an over-estimate, not
        # a measurement: treat it as unresolved (keep the default) and flag it.
        val = None if s.below_ladder else s.suggested_budget
        if val is None:
            why = ("converges BELOW the ladder — lower this street's ladder_top_seconds"
                   if s.below_ladder else
                   "still moving at the top rung — raise this street's ladder_top_seconds")
            unresolved.append(
                f"{_STREET_NAME.get(street, street)}/{regime}/n_live={n_live} ({why})"
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
        "hot_l1_tol": {
            r: next((s.hot_l1_tol for s in summaries if s.cell[1] == r), None)
            for r in ("mccfr", "vector")
        },
        "mccfr_per_player_by_street": tuple(mccfr_base),
        "vector_budget_by_street": tuple(vector),
        "unresolved_cells": unresolved,  # never settled within the ladder → raise LADDER_HI/MAX
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
                    "argmax_match", "l1_to_ref", "hot_l1_cross_rep"])
        for r in rows:
            w.writerow([r.condition, r.regime, r.street, r.n_live, r.per_replica,
                        r.workers, r.pooled_iters, f"{r.wall_seconds:.6f}",
                        r.stop_reason, r.sample, r.rep, f"{r.value_gap_mbb:.6f}",
                        f"{r.root_value:.6f}", f"{r.hot_l1:.6f}",
                        f"{r.argmax_match:.6f}", f"{r.l1_to_ref:.6f}",
                        f"{r.hot_l1_cross_rep:.6f}"])


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
           f"{'W':>4} {'top_it':>7} {'it/s':>8} {'wall@top':>9} {'hotL1@pen':>10} "
           f"{'xrep':>7} {'tol':>5} {'sugg':>7} {'wall@sugg':>10}")
    print(hdr)
    print("-" * len(hdr))
    for s in sorted(summaries, key=lambda x: (x.cell[0], x.cell[1], x.cell[2], x.cell[3])):
        cond, regime, street, n_live = s.cell
        t_top = s.ladder[-1]
        sugg = s.suggested_budget
        wall_sugg = s.mean_wall.get(sugg) if sugg is not None else None
        # mean hot_l1 at the *penultimate* rung (the last real rung; @top is 0 by
        # self-compare) — the residual "still moving" the criterion reads.
        t_pen = s.ladder[-2] if len(s.ladder) > 1 else t_top
        hl1_pen = s.mean_hot_l1.get(t_pen, float("nan"))
        print(f"{cond:<10} {regime:<7} {_STREET_NAME.get(street, street):<8} "
              f"{n_live:>6} {s.workers:>4} {t_top:>7} {s.throughput_it_s:>8.1f} "
              f"{s.mean_wall[t_top]:>9.2f} {hl1_pen:>10.3f} "
              f"{(f'{s.cross_rep_hot_l1:.3f}' if not np.isnan(s.cross_rep_hot_l1) else '-'):>7} "
              f"{s.hot_l1_tol:>5.2f} "
              f"{(str(sugg) if sugg is not None else '>max'):>7} "
              f"{(f'{wall_sugg:.2f}' if wall_sugg is not None else '-'):>10}")
    if wall_target is not None:
        print("-" * len(hdr))
        print(f"Wall target per decision: {wall_target:.2f}s — cells whose wall@sugg "
              f"exceeds it need a lower budget or heavier blueprint fallback.")
        for s in summaries:
            sugg = s.suggested_budget
            wall_sugg = s.mean_wall.get(sugg) if sugg is not None else None
            if wall_sugg is not None and wall_sugg > wall_target:
                cond, regime, street, n_live = s.cell
                print(f"  ! {cond}/{regime}/{_STREET_NAME.get(street, street)}/"
                      f"n_live={n_live}: wall@sugg={wall_sugg:.2f}s > {wall_target:.2f}s")
    floored = [s for s in summaries if s.tol_below_cross_rep]
    if floored:
        print("-" * len(hdr))
        print("  ! tolerance is AT OR BELOW the measured cross-seed spread ('xrep') — the "
              "budget for these cells is resolved finer than independent seeds agree to, "
              "so it is largely an artefact of which seeds ran. Loosen their tol:")
        for s in floored:
            cond, regime, street, n_live = s.cell
            print(f"    {cond}/{regime}/{_STREET_NAME.get(street, street)}/n_live="
                  f"{n_live}: tol={s.hot_l1_tol:.2f} <= xrep="
                  f"{s.cross_rep_hot_l1:.3f}")
    print("\nSuggested SolverConfig block (hot_l1 tol mccfr="
          f"{config['hot_l1_tol']['mccfr']} vector={config['hot_l1_tol']['vector']}):")
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
    n_live_filter: Optional[Sequence[int]] = None,
    max_concurrent_multiway: Optional[int] = None,
    crn_value: bool = False,
    crn_worlds: int = 32,
    collect_hands: int,
    per_cell_cap: int,
    per_cell_cap_deterministic: Optional[int] = None,
    reps: int,
    ladder_points: int,
    ladder_lo: float,
    ladder_top_seconds: Sequence[float] = (600.0, 600.0, 60.0),
    ladder_top_seconds_vector: Sequence[float] = (600.0, 600.0, 15.0),
    probe_seconds: float = 30.0,
    hot_l1_tol_mccfr: float = DEFAULT_HOT_L1_TOL_MCCFR,
    hot_l1_tol_vector: float = DEFAULT_HOT_L1_TOL_VECTOR,
    collect_iters: int,
    table_policy: str,
    fixed_seats: Optional[Sequence[str]] = None,
    bias_multiplier: float = 5.0,
    run_seed: int,
    big_blind: int,
    small_blind: int,
    starting_stack: int,
    low_card_rank: int,
    high_card_rank: int,
    out_dir: Path,
    wall_target: Optional[float],
    regime_ab_streets: Sequence[int] = (2,),
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Each cell's ladder is WALL-ANCHORED: its top rung is the deepest solve that fits the
    # cell's wall budget (``ladder_top_seconds``, measured by a box-saturating probe), and
    # the rungs walk DOWN to ``ladder_lo * top`` — the band where hot_l1 crosses the 0.1-0.2
    # decision region.  No compute is spent on the low budgets the 2026-08 vanilla run
    # showed are certainly unconverged, and the reference rung is as converged as the box
    # can afford.  Ladders are keyed (regime, street, n_live) and SHARED across conditions,
    # so vanilla/DBR are compared on identical rungs.  MCCFR only converges the HOT
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
            fixed_seats=fixed_seats,
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
        max_wall_seconds=1e9, bias_multiplier=bias_multiplier,
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
                per_cell_deterministic=per_cell_cap_deterministic,
                n_live_filter=n_live_filter,
            )
        else:
            logger.info("collecting roots (sampled play) for condition=%s", condition)
            samples = collect_roots(
                session, cond_cfg, collect_cfg, condition,
                n_hands=collect_hands, per_cell_cap=per_cell_cap, run_seed=run_seed,
            )
        if n_live_filter is not None:
            # Enforce the live-count slice in BOTH root modes (construct_roots already
            # skips the work; sampled play harvests whatever it hits).
            keep = {int(x) for x in n_live_filter}
            samples = {c: v for c, v in samples.items() if int(c[3]) in keep}
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
                                 ref_from=None))
                jobs.append(dict(cell=cell, samples=sample_list, force_regime="mccfr",
                                 ref_from=vidx))
            else:
                jobs.append(dict(cell=cell, samples=sample_list, force_regime=None,
                                 ref_from=None))

    # --- Wall-anchored ladders (one probe pass, shared by every condition) -----------
    # The ladder is keyed on (regime, street, n_live) ONLY — conditions share it, so a
    # vanilla/DBR comparison is made on identical rungs (and the probe runs once).
    def _lkey(job) -> Tuple[str, int, int]:
        cond, regime, street, n_live = job["cell"]
        return (str(job.get("force_regime") or regime), int(street), int(n_live))

    keys = sorted({_lkey(j) for j in jobs})
    rep_for = {}
    for j in jobs:
        rep_for.setdefault(_lkey(j), (j["samples"][0], j.get("force_regime")))
    logger.info("probing throughput for %d ladders (%.0fs, box-saturating)",
                len(keys), probe_seconds)
    thr_by_i = probe_throughput([rep_for[k] for k in keys], prod_cfg,
                                seconds=probe_seconds, workers=resolved_workers,
                                base_seed=run_seed)
    ladders: Dict[Tuple[str, int, int], List[int]] = {}
    for i, k in enumerate(keys):
        regime, street, _n_live = k
        thr = thr_by_i.get(i)
        secs = _top_seconds_for(regime, street, ladder_top_seconds,
                                ladder_top_seconds_vector)
        if thr is None or thr <= 0:        # probe failed — fall back to the prod budget
            top_iters = max(1, int(iteration_budget(rep_for[k][0].ctx, prod_cfg,
                                                    regime_override=rep_for[k][1])))
            ladders[k] = _ladder_wall_anchored(top_iters / max(1e-9, secs), secs,
                                               ladder_points, lo_frac=ladder_lo)
            logger.warning("ladder %s: probe failed, falling back to the production budget", k)
        else:
            ladders[k] = _ladder_wall_anchored(thr, secs, ladder_points, lo_frac=ladder_lo)
        logger.info("ladder %-22s %6.1f it/s x %4.0fs -> top %d  rungs %s",
                    str(k), thr or float("nan"), secs, ladders[k][-1], ladders[k])
    for j in jobs:
        j["ladder"] = ladders[_lkey(j)]

    # One SEARCH per (sample, rep) — the ladder is snapshotted from it, so rungs no longer
    # multiply the solve count.  Deterministic cells run a single rep.
    n_solves = sum(len(j["samples"])
                   * (1 if _is_deterministic(j.get("force_regime") or j["cell"][1],
                                             j["cell"][2]) else reps)
                   for j in jobs)
    logger.info("sweeping %d jobs (%d cells, %d solves) across %d cores in ONE pool",
                len(jobs), len({j["cell"] for j in jobs}), n_solves, resolved_workers)
    all_rows = sweep_jobs(jobs, prod_cfg, pool_workers=resolved_workers, reps=reps,
                          base_seed=run_seed, big_blind=big_blind,
                          crn_value=crn_value, crn_worlds=crn_worlds,
                          max_concurrent_multiway=max_concurrent_multiway)

    # Aggregate.
    by_cell: Dict[Cell, List[SweepRow]] = defaultdict(list)
    for r in all_rows:
        by_cell[r.cell].append(r)
    summaries = [summarize_cell(c, rs, hot_l1_tol_mccfr=hot_l1_tol_mccfr,
                                hot_l1_tol_vector=hot_l1_tol_vector,
                                big_blind=big_blind)
                 for c, rs in by_cell.items()]

    config = suggest_config(summaries)

    _write_rows_csv(all_rows, out_dir / "calibration_rows.csv")
    summary_json = {
        "search_core": "on" if core_on else "off (PURE PYTHON — not production)",
        "workers": resolved_workers,
        "max_concurrent_multiway": max_concurrent_multiway,  # peak-RAM cap (None ⇒ uncapped)
        # Live-count slice this run covered (None ⇒ the full grid).  A partial run's
        # suggested_config keeps the DEFAULT budget for every street it did not measure.
        "n_live_filter": list(n_live_filter) if n_live_filter else None,
        # Whether VR-MCCFR (opponent_modeling §5.5) was active this run — DBR-only, so
        # it affects only modeled MCCFR cells; recorded so a run is self-documenting.
        "variance_reduction": bool(variance_reduction),
        # Opponent table this calibration exploited — recorded so a run is self-documenting
        # (the DBR budget depends on WHO the hero models and how hard they leak).
        "opponents": {
            "table_policy": table_policy,
            "fixed_seats": list(fixed_seats) if fixed_seats else None,
            "bias_multiplier": float(bias_multiplier),
        },
        "ladder_design": {
            "anchor": "wall-anchored: top rung = probed it/s * ladder_top_seconds",
            "top_seconds_mccfr_flop_turn_river": list(ladder_top_seconds),
            "top_seconds_vector_flop_turn_river": list(ladder_top_seconds_vector),
            "probe_seconds": probe_seconds,
            "lo_frac": ladder_lo, "points": ladder_points,
            "ladders": {f"{r}/{s}/n{n}": v for (r, s, n), v in sorted(ladders.items())},
            "note": "rungs walk down from the deepest solve that fits the cell's wall "
                    "budget to lo_frac*top; the top rung is the self-reference the hot_l1 "
                    "convergence is measured against.  Shared across conditions.",
        },
        # CONVERGENCE = single-worker strategy self-stability: smallest budget where the
        # MEAN hot_l1 (each solve's hot-action strategy vs its own top budget) <= tol.
        # Value gap / replica spread are DIAGNOSTIC columns only (value is payoff-leverage-
        # noisy on deep cells; replica spread is reproducibility, not convergence).
        "metric": "mean_hot_l1",
        # PER REGIME: sampled MCCFR has a Monte-Carlo floor that full-width vector lacks.
        "hot_l1_tol": {"mccfr": float(hot_l1_tol_mccfr),
                       "vector": float(hot_l1_tol_vector)},
        "suggested_config": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in config.items()
        },
        "cells": [
            {
                "condition": s.cell[0], "regime": s.cell[1],
                "street": _STREET_NAME.get(s.cell[2], s.cell[2]), "n_live": s.cell[3],
                "n_samples": s.n_samples, "throughput_it_s": s.throughput_it_s,
                "pooled_it_s": s.pooled_it_s,
                "suggested_budget": s.suggested_budget,
                "converged": s.converged, "below_ladder": s.below_ladder,
                "hot_l1_tol": s.hot_l1_tol,
                # Measured Monte-Carlo spread of the metric (cross-rep at the top
                # budget); null for the deterministic / single-rep cells.  A tol at or
                # under it means the budget is resolved finer than seeds agree to.
                "cross_rep_hot_l1": (None if np.isnan(s.cross_rep_hot_l1)
                                     else s.cross_rep_hot_l1),
                "tol_below_cross_rep": s.tol_below_cross_rep,
                "mean_hot_l1": {str(t): s.mean_hot_l1[t] for t in s.ladder},
                "argmax_stability": {str(t): s.argmax_stability[t] for t in s.ladder},
                "mean_value_gap_mbb": {str(t): s.mean_value_gap_mbb[t] for t in s.ladder},
                "replica_spread_mbb": s.replica_spread_mbb,  # diagnostic
                "mean_l1": {str(t): s.mean_l1[t] for t in s.ladder},
                "mean_wall_seconds": {str(t): s.mean_wall[t] for t in s.ladder},
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
    @click.option("--n-live", default="", show_default=True,
                  help="Comma list of live-player counts to calibrate (e.g. '2,3'); empty "
                       "= all.  Splits an expensive grid into separate runs — the deep "
                       "n_live=4 cells are ~2/3 of a 4p run's cost, so '2,3' now and '4' "
                       "later is much cheaper to schedule.  LOSSLESS: each root's seed is "
                       "keyed on (street, n_live, k), so a cell yields byte-identical roots "
                       "either way.  (The grid is post-flop only — pre-flop is played from "
                       "the blueprint, never searched.)")
    @click.option("--max-concurrent-multiway", default=None, type=int,
                  help="Peak-RAM cap: at most this many multiway (>=3 live) MCCFR solves — "
                       "the biggest tables — run at once; cheap solves backfill the other "
                       "cores.  Lets you keep --workers high but request far less --mem "
                       "(peak ≈ K*multiway + (W-K)*light).  Default: no cap.")
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
    @click.option("--crn-value/--internal-value", default=False, show_default=True,
                  help="DIAGNOSTIC ONLY (default OFF).  The convergence budget now comes from "
                       "strategy self-stability (worst-case hot_l1), which needs no value "
                       "estimate — so the CRN value-estimator is off by default and no compute "
                       "is spent on it.  --crn-value re-enables it purely for the diagnostic "
                       "value-gap/replica-spread columns.")
    @click.option("--crn-worlds", default=32, type=int, show_default=True,
                  help="Fixed card-worlds per CRN value estimate (only used with --crn-value).")
    @click.option("--collect-hands", default=400, type=int, show_default=True,
                  help="Hands played to harvest roots when --sample-roots is set.")
    @click.option("--per-cell-cap", default=6, type=int, show_default=True,
                  help="Distinct HANDS (roots) per cell for the sampled cells.")
    @click.option("--per-cell-cap-deterministic", default=8, type=int, show_default=True,
                  help="Distinct HANDS for the DETERMINISTIC cells (vector river): every "
                       "seed there gives a byte-identical solve, so they run exactly ONE "
                       "rep and spend their compute on extra hands instead.")
    @click.option("--reps", default=12, type=int, show_default=True,
                  help="Independent re-solves per root.  ONE search per rep now covers the "
                       "whole ladder (mid-search snapshots), so reps — not rungs — is where "
                       "extra compute buys a tighter mean hot_l1.")
    @click.option("--ladder-points", default=6, type=int, show_default=True,
                  help="Rungs per ladder, walking DOWN from the wall-anchored top.")
    @click.option("--ladder-top-seconds", default="600,600,60", show_default=True,
                  help="MCCFR per-street wall budget (flop,turn,river) for the ladder's TOP "
                       "rung: top = probed it/s * this, i.e. the deepest solve that fits.  "
                       "Hard cells (flop/turn — still moving at their production budgets) get "
                       "the full wall; the river converges in seconds, so a big budget there "
                       "only burns compute AND pushes the whole ladder above the crossing.")
    @click.option("--ladder-top-seconds-vector", default="600,600,15", show_default=True,
                  help="Same, for the VECTOR cells.  Separate because the regimes converge at "
                       "very different iteration counts (2026-08: HU river vector settles at "
                       "~1071 iters vs ~9000 for multiway river MCCFR), so one per-street "
                       "value cannot bracket both.")
    @click.option("--ladder-lo", default=0.35, type=float, show_default=True,
                  help="Ladder min as a FRACTION OF THE TOP rung (not of the production "
                       "budget).  0.35 spans the band where hot_l1 crosses 0.1-0.2; lower "
                       "just re-measures budgets already known to be unconverged.")
    @click.option("--probe-seconds", default=30.0, type=float, show_default=True,
                  help="Wall-bounded probe solve per ladder that measures its it/s (sets the "
                       "top rung).  Run box-saturating, so the throughput matches the loaded "
                       "run rather than an idle-box overestimate.")
    @click.option("--hot-l1-tol-mccfr", default=DEFAULT_HOT_L1_TOL_MCCFR, type=float,
                  show_default=True,
                  help="Convergence tolerance for the SAMPLED (MCCFR) cells. The per-cell "
                       "budget is the smallest rung where the MEAN hot_l1 (each single-worker "
                       "solve's hot-action strategy over its whole root street vs its own top "
                       "budget, averaged over the cell's solves) <= this. Looser than the "
                       "vector bar on purpose: external sampling leaves Monte-Carlo dispersion "
                       "that decays like 1/sqrt(T) and never reaches zero, so a tight bar would "
                       "measure noise, not convergence. The run reports each cell's MEASURED "
                       "cross-seed spread (cross-rep hot_l1 at the top budget: same hand, same "
                       "iterations, different seed) and flags any cell whose tol sits at or "
                       "below it — there the emitted budget is resolved finer than independent "
                       "seeds agree to, so loosen this above the reported spread.")
    @click.option("--hot-l1-tol-vector", default=DEFAULT_HOT_L1_TOL_VECTOR, type=float,
                  show_default=True,
                  help="Convergence tolerance for the FULL-WIDTH (vector) cells. Tighter than "
                       "the MCCFR bar: the vector regime enumerates both ranges, so it has no "
                       "per-iteration sampling noise to average out (the river cell is exactly "
                       "deterministic) and a tight bar is meaningful there.")
    @click.option("--collect-iters", default=64, type=int, show_default=True,
                  help="Cheap per-solve budget used only to advance collection hands.")
    @click.option("--table-policy", default="random", show_default=True,
                  help="'random' mixes fold/call/raise bias for street/live-count "
                  "coverage; 'all_blueprint' | 'fixed' also allowed.")
    @click.option("--fixed-seats", default="", show_default=True,
                  help="Comma-separated opponent labels (bp/bp_fold/bp_call/bp_raise), "
                       "exactly n_players-1, used only when --table-policy=fixed.  "
                       "'bp_fold,bp_call,bp_raise' seats ONE opponent per exploitable "
                       "bias class (4-player) — deterministic, full leak coverage, no "
                       "random seat-draw noise across cells.")
    @click.option("--bias-multiplier", default=5.0, type=float, show_default=True,
                  help="Opponent leak strength: the biased action class's probability is "
                       "scaled by this then renormalized.  1.0 = unbiased (nothing to "
                       "exploit); very large ⇒ near-pure/degenerate opponent.  5.0 is the "
                       "established mid-strength leak (biased action ~2-3x its baseline "
                       "frequency, still mixing).")
    @click.option("--run-seed", default=0, type=int, show_default=True)
    @click.option("--big-blind", default=100, type=int, show_default=True)
    @click.option("--small-blind", default=50, type=int, show_default=True)
    @click.option("--starting-stack", default=10_000, type=int, show_default=True)
    @click.option("--low-card-rank", default=2, type=int, show_default=True)
    @click.option("--high-card-rank", default=14, type=int, show_default=True)
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
        if not (0 < o["ladder_lo"] < 1):
            raise click.BadParameter("need 0 < --ladder-lo < 1 (a fraction of the top rung)")
        def _secs(raw, flag):
            v = tuple(float(x) for x in raw.split(",") if x.strip())
            if len(v) != 3 or any(s <= 0 for s in v):
                raise click.BadParameter(f"{flag} needs 3 positive values (flop,turn,river)")
            return v
        top_secs = _secs(o["ladder_top_seconds"], "--ladder-top-seconds")
        top_secs_vec = _secs(o["ladder_top_seconds_vector"], "--ladder-top-seconds-vector")
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
        fixed_seats = tuple(
            s.strip() for s in o["fixed_seats"].split(",") if s.strip()
        ) or None
        n_live_filter = tuple(
            int(x) for x in o["n_live"].split(",") if x.strip()
        ) or None
        if n_live_filter and any(v < 2 or v > o["n_players"] for v in n_live_filter):
            raise click.BadParameter(
                f"--n-live values must be in [2, n_players={o['n_players']}], "
                f"got {list(n_live_filter)}")
        run_calibration(
            blueprint_path=o["blueprint_path"], lut_path=o["lut_path"],
            conditions=conditions, model_spec=model_spec,
            n_players=o["n_players"], workers=o["workers"],
            max_concurrent_multiway=o["max_concurrent_multiway"],
            variance_reduction=o["variance_reduction"],
            construct_roots_mode=o["construct_roots"], n_live_filter=n_live_filter,
            crn_value=o["crn_value"], crn_worlds=o["crn_worlds"],
            collect_hands=o["collect_hands"], per_cell_cap=o["per_cell_cap"],
            per_cell_cap_deterministic=o["per_cell_cap_deterministic"],
            reps=o["reps"], ladder_points=o["ladder_points"], ladder_lo=o["ladder_lo"],
            ladder_top_seconds=top_secs, ladder_top_seconds_vector=top_secs_vec,
            probe_seconds=o["probe_seconds"],
            hot_l1_tol_mccfr=o["hot_l1_tol_mccfr"],
            hot_l1_tol_vector=o["hot_l1_tol_vector"],
            collect_iters=o["collect_iters"], table_policy=o["table_policy"],
            fixed_seats=fixed_seats, bias_multiplier=o["bias_multiplier"],
            run_seed=o["run_seed"], big_blind=o["big_blind"],
            small_blind=o["small_blind"], starting_stack=o["starting_stack"],
            low_card_rank=o["low_card_rank"], high_card_rank=o["high_card_rank"],
            out_dir=Path(o["out_dir"]), wall_target=o["wall_target"],
            regime_ab_streets=regime_ab_streets,
        )

    return calibrate


calibrate = _cli()

if __name__ == "__main__":
    calibrate()
