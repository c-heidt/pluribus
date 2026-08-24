"""AIVAT — variance-reduced strength estimate (doc §10.2, §9 step 9).

Per-hand chip variance is huge (~100 bb/100 std, §11), so the raw ``hero_chips_delta``
bb/100 CI is wide.  AIVAT replaces it with an estimator that has the **same mean**
(unbiased) and **far smaller variance**, which is what makes the small-edge
comparisons detectable.  The estimator is a control variate:

    aivat_value = u(z)  −  Σ correction_terms

Each correction term has **zero expectation** by construction, so
``E[aivat_value] == E[u(z)]`` for *any* value function ``v`` — a better ``v`` only
shrinks variance, it can never skew the mean.  Two families of term are taken:

- **Action nodes of a known-policy player** (here BOTH the hero and the
  blueprint-derived opponents have known, exactly-computable policies — the "gift"
  of the start-scope bots, §10.1/§10.2, giving *full*-AIVAT over the players'
  actions rather than the hero-only partial case)::

      term = v(child_sampled) − Σ_a π(a)·v(child_a)

  with ``π`` the hero's played strategy at hero nodes (the same vector logged as
  ``decisions.action_dist``) and the opponent's ``BlueprintPolicy.strategy(state,
  bias)`` at opponent nodes.

- **The terminal all-in runout** (a chance node the engine *can* integrate
  exactly).  When the hand ends as a decision-free all-in with **≤2 board cards to
  come** (a flop/turn all-in), the single dealt board is one draw of a chance event
  whose exact mean the engine already computes via :meth:`PokerEnv.runout_equity`::

      term = u(z) − runout_equity[hero]

  Subtracting it replaces the realised single-board payout with the exact
  board-average — a high-variance chance event, removed for free.  A **pre-flop**
  all-in (5 cards to come) is **skipped**: its exact runout exceeds
  ``runout_equity``'s enumeration cap and would Monte-Carlo sample thousands of
  boards per hand, so those hands keep only their action-node corrections.  (General
  per-street MIVAT chance corrections are **not** taken either: this engine has no
  steppable/enumerable per-street chance node — board cards are dealt inside
  ``_apply_action_in_place`` off a shuffled deck, with no "re-deal a specific
  alternative card down the same betting line" primitive — so only the terminal
  all-in runout, which ``runout_equity`` already integrates, is corrected.  This is
  documented in evaluation.md §10.2 as a bounded scope.)

**The value function ``v``.**  Reuses the search's own leaf machinery
(:func:`poker_ai.search.leaf.continuation_value`) under a fixed all-blueprint
continuation profile: it materialises a card-disjoint joint hole assignment sampled
from the tracker's belief (the hero's own seat filled with its *known* hole), then
takes the mean hero-seat continuation value over a handful of such samples.  Any
consistent ``v`` is unbiased, so the internal hole sampling need not be the exact
conditioned joint — sequential-with-removal is used for robustness.

**Information-leak rule (the #1 correctness trap).**  ``v`` integrates over the
*observer's* belief and must never read an opponent's actual concrete hole; the
belief-sampled holes come from ``tracker.snapshot()`` / ``folded_snapshot()``,
which already exclude the board and the hero's hole (card removal).  The hero seat
is filled with its own hole (known information, not a leak).  ``π`` at an opponent
node *may* condition on that opponent's own hole — that is exactly the distribution
the action was sampled from, and AIVAT corrects the action *given the policy*.

**Cost.**  All of this lands in the *experiment* time budget, never the real-time
search hot path (§10.2): each ``v`` call is a few belief-sampled rollouts, and it is
evaluated at the (2–4) legal siblings of each decision.  It is gated behind an
opt-in flag (``EvalConfig.aivat``).

**RNG isolation — two directions, both required.**  AIVAT owns dedicated streams
(``rng`` for beliefs + rollout actions, a spawned child for board runouts) and reads
the global ``np.random`` nowhere.  That buys:

- *outward* — turning AIVAT on does not perturb the deck / hero / opponent sampling,
  so a hand's raw ``hero_chips_delta`` is identical with AIVAT on or off;
- *inward* — AIVAT's own draws do not depend on how much randomness anything else
  consumed.  This is the direction that matters for the headline: the evaluation
  compares CRN-paired arms, and while the hero's solver burns an arm-dependent
  amount of RNG, a value function sharing that stream would score an identically
  played hand differently in each arm.  ``aivat_value`` is instead a pure function
  of ``(run_seed, hand_index)`` and the played line, so it cancels exactly in the
  paired Δ and AIVAT *stacks* with CRN (evaluation.md §10.1) instead of eroding it.

Passivity used to be enforced by snapshot-and-restore around the global stream
(:func:`_preserve_global_random`, kept as a backstop); that delivered the outward
direction only, and the missing inward one is why the first AIVAT run showed no
variance reduction and inflated the DBR headline (evaluation.md §10.2).

A residual, *legitimate* arm-dependence remains and is not an RNG matter: ``v``
integrates over ``hero.tracker`` (updated under the last search's average policy)
and ``π`` is the played σ, so arms at different budgets still compute different —
still unbiased — corrections.  Making those cancel too needs an arm-independent
``v``, which is a separate design decision.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np

from environment.poker_env import PokerEnv
from poker_ai.search.leaf import LeafConfig, continuation_value
from poker_ai.search.rng import spawn_one

logger = logging.getLogger(__name__)

# Max board cards still to come for the terminal all-in chance correction to run.
# ≤2 (a flop/turn all-in) is always under ``runout_equity``'s exact-enumeration cap,
# so the correction is exact and cheap.  A pre-flop all-in (5 to come) would exceed
# the cap and fall back to Monte-Carlo *sampling thousands of boards* per hand — an
# expensive, log-noisy operation — so it is skipped (the hand keeps its action-node
# corrections), mirroring the solver's own ``street_at_root != 0`` equity guard.
_MAX_RUNOUT_CARDS = 2


@contextlib.contextmanager
def _preserve_global_random():
    """Run a block, then restore the **global** ``np.random`` state it consumed.

    A **defensive backstop, not the isolation mechanism.**  AIVAT draws every
    random number it needs from its own streams (``self._rng`` for beliefs and
    rollout actions, ``self._board_rng`` for board runouts), so in the normal case
    this manager sees an unchanged state and restores a no-op.  It stays because
    passivity — the played hand's ``hero_chips_delta`` being bit-identical with
    AIVAT on or off — is a hard guarantee that should not depend on every engine
    path *below* the value function continuing to honour its ``rng`` argument.

    Restoring the global state was previously how AIVAT stayed passive, but it
    never made AIVAT's own draws *independent* of the global stream: the value
    function still consumed whatever state the hero's solver had left behind, so
    two arms playing an identical hand drew different boards and the evaluation's
    CRN pairing was destroyed.  Owning the streams outright is what fixes that;
    see :mod:`poker_ai.search.rng`.
    """
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


class _LeafCtx:
    """Minimal ``SubgameContext`` stand-in for :func:`continuation_value`.

    ``continuation_value`` reads only ``ctx.leaf`` (the fleet + rollout knobs),
    ``ctx.rng`` (the rollout action-sampling source) and ``ctx.board_rng`` (the
    rollout board runout) — never the ranges / board mask — so the value function
    passes this lightweight duck-typed carrier instead of building a full
    :class:`SubgameContext` (which would need ranges we integrate over ourselves).
    """

    __slots__ = ("leaf", "rng", "board_rng")

    def __init__(
        self,
        leaf: LeafConfig,
        rng: np.random.Generator,
        board_rng: np.random.Generator,
    ) -> None:
        self.leaf = leaf
        self.rng = rng
        self.board_rng = board_rng


class LeafValue:
    """Public-state value function ``v(state | belief) → hero-seat chip delta``.

    Averages :func:`continuation_value`'s hero-seat entry over
    ``n_hole_samples`` card-disjoint joint hole draws from the current belief,
    under a fixed all-blueprint continuation profile.  It reads the belief from the
    live ``hero.tracker`` at call time (the tracker is stable within a betting round
    — updated only at round boundaries — so a correction taken mid-round sees the
    belief as of that round's start, a valid observer information set).  A hero with
    ``tracker is None`` (a blueprint-only / non-search agent) is supported: opponent
    holes are then drawn belief-free (uniform over available combos), still unbiased.

    Parameters
    ----------
    hero
        Any agent exposing ``my_seat`` / ``my_hole`` (the known hero hole) and a
        ``tracker`` — a :class:`~poker_ai.search.ranges.RangeTracker` supplying the
        opponent belief, or ``None`` for a blueprint-only agent (belief-free draws).
    leaf_cfg
        The continuation leaf config — reuse the session's ``solver_cfg.leaf`` for
        its **fleet** (``policies``); passed straight through (``continuation_value``
        always takes exactly one rollout, so there is nothing left to override).
    rng
        Dedicated AIVAT RNG (a distinct seed sub-stream), used for the belief hole
        sampling and the rollout action draws.  A second stream for the rollout
        **board** runouts is spawned from it — split for the same reason the solver
        splits its own (a board draw interleaved into the sampling stream desyncs
        the sampling trajectory), and spawned rather than drawn from so ``rng``'s
        byte-stream is unchanged by the split.

        Both streams are AIVAT's alone.  Nothing here reads the global
        ``np.random``, which belongs to the played hand's deal: that independence
        is what makes ``aivat_value`` a pure function of ``(run_seed, hand_index)``
        and the played line, so two CRN-paired arms that play a hand identically
        produce an identical correction and it cancels exactly in the paired Δ.
    n_hole_samples
        Number of joint hole draws averaged per ``v`` evaluation.
    """

    def __init__(
        self,
        hero,
        leaf_cfg: LeafConfig,
        rng: np.random.Generator,
        *,
        n_hole_samples: int = 6,
    ) -> None:
        self._hero = hero
        self._rng = rng
        self._board_rng = spawn_one(rng)
        self._m = int(n_hole_samples)
        self._leaf = leaf_cfg

    def child_values(
        self, env_before: PokerEnv, legal: Sequence[str]
    ) -> Dict[str, float]:
        """Mean hero-seat value of every ``legal`` action's child at ``env_before``.

        For each of ``n_hole_samples`` belief draws, materialises the sampled holes
        at ``env_before`` (board = the current street, so no board conflict), then
        steps each legal action via make/undo and scores the resulting child with
        :func:`continuation_value`.  ``env_before`` is never mutated (the sampled
        env is a fresh ``with_hole_cards`` copy).
        """
        legal = list(legal)
        sums: Dict[str, float] = {a: 0.0 for a in legal}
        hero_seat = int(self._hero.my_seat)
        n_players = env_before.n_players
        profile = {s: "none" for s in range(n_players)}
        m = max(self._m, 1)
        ctx = _LeafCtx(self._leaf, self._rng, self._board_rng)
        with _preserve_global_random():
            for _ in range(m):
                holes = self._sample_joint(env_before)
                base = env_before.with_hole_cards(holes, rng=self._board_rng)
                for a in legal:
                    tok = base.step_in_place(a)
                    v = continuation_value(base, profile, ctx)
                    sums[a] += float(v[hero_seat])
                    base.undo(tok)
        return {a: sums[a] / m for a in legal}

    # ------------------------------------------------------------------ #
    # Belief joint hole sampling (card-disjoint, board-masked)
    # ------------------------------------------------------------------ #
    def _sample_joint(self, env: PokerEnv) -> List[Tuple[int, int]]:
        """One card-disjoint hole assignment for every seat (hero = its own hole).

        The hero seat is fixed to ``hero.my_hole`` (known); every other seat is
        drawn from its tracked belief (live or fold-time), masked to the current
        board so the draw never conflicts with the community.  Sequential-with-
        removal keeps the draw card-disjoint — a mild approximation to the exact
        conditioned joint that is harmless here (``v`` may be *any* consistent
        function; unbiasedness does not depend on the sampling being exact).

        A hero **without** a belief tracker (``hero.tracker is None`` — e.g. a
        blueprint-only / non-search agent under test) has no per-seat range, so the
        non-hero seats fall back to belief-free sampling: uniform over the
        board-compatible, still-available combos (``_masked_weights(None, ...)``).
        Still unbiased — ``v`` need only be consistent — it just integrates over a
        wider (uninformative) opponent range.
        """
        tracker = self._hero.tracker
        hero_seat = int(self._hero.my_seat)
        my_hole = tuple(int(c) for c in self._hero.my_hole)
        cc = env.combo_cards
        board_ok = _board_compatible(env)

        live = tracker.snapshot() if tracker is not None else {}   # {seat: range}
        folded = tracker.folded_snapshot() if tracker is not None else {}

        holes: List[Tuple[int, int]] = [(-1, -1)] * env.n_players
        holes[hero_seat] = my_hole
        used = {my_hole[0], my_hole[1]}

        other = [s for s in range(env.n_players) if s != hero_seat]
        # Draw the more-concentrated (folded) beliefs first so removal bites least.
        for s in other:
            src = live.get(s)
            if src is None:
                src = folded.get(s)
            w = _masked_weights(src, board_ok, cc, used)
            ci = int(self._rng.choice(len(w), p=w))
            c0, c1 = int(cc[ci, 0]), int(cc[ci, 1])
            holes[s] = (c0, c1)
            used.add(c0)
            used.add(c1)
        return holes


def _board_compatible(env: PokerEnv) -> np.ndarray:
    """Boolean mask over ``env.combo_cards``: True iff the combo shares no card
    with the current community (every combo pre-flop)."""
    if not env.community_cards:
        return np.ones(env.n_combos, dtype=bool)
    board = np.asarray(env.community_cards, dtype=np.int32)
    cc = env.combo_cards
    return ~(np.isin(cc[:, 0], board) | np.isin(cc[:, 1], board))


def _masked_weights(
    src, board_ok: np.ndarray, cc: np.ndarray, used: set
) -> np.ndarray:
    """Normalised sampling weights for one seat: its belief, zeroed on the board
    and on already-drawn cards; falls back to uniform-over-available if the belief
    has no board-compatible mass left (a defensive, ``v``-only approximation)."""
    if src is None:
        w = board_ok.astype(np.float64)
    else:
        w = np.asarray(src, dtype=np.float64) * board_ok
    if used:
        forbidden = np.fromiter(used, dtype=np.int64)
        conflict = np.isin(cc[:, 0], forbidden) | np.isin(cc[:, 1], forbidden)
        w = w.copy()
        w[conflict] = 0.0
    total = w.sum()
    if total <= 0.0:
        # Belief collapsed against the board / removal — fall back to a uniform
        # draw over the still-available combos so v stays defined.
        w = board_ok.astype(np.float64)
        if used:
            w = w.copy()
            w[conflict] = 0.0
        total = w.sum()
        if total <= 0.0:
            raise ValueError("AIVAT joint sampler: card exhaustion for a seat.")
    return w / total


class AivatAccumulator:
    """Accumulates AIVAT correction terms across one hand → the ``aivat_value`` scalar.

    Fed one ``(env_before, seat, action, legal, probs)`` event per known-policy
    action node during play; :meth:`finalize` reads the terminal payout, applies the
    all-in runout chance correction, and returns
    ``u(z) − Σ action_terms − chance_term``.
    """

    def __init__(self, hero_seat: int, value_fn: LeafValue, rng: np.random.Generator,
                 *, max_runout_cards: int = _MAX_RUNOUT_CARDS,
                 runout_cap: int = 5000) -> None:
        self._hero_seat = int(hero_seat)
        self._v = value_fn
        self._rng = rng
        self._sum_terms = 0.0
        # Terminal all-in runout chance-correction reach.  Default 2 (flop/turn all-ins,
        # ``C(deck, <=2)`` under the exact cap) — the played-game default, unchanged.  The
        # calibration raises it to 5 so a PRE-FLOP all-in is also Rao-Blackwellised, with
        # ``runout_cap`` bounding the Monte-Carlo board sample so the cost stays modest.
        self._max_runout_cards = int(max_runout_cards)
        self._runout_cap = int(runout_cap)

    def correct_action(
        self,
        env_before: PokerEnv,
        seat: int,
        action: str,
        legal: Sequence[str],
        probs: Sequence[float],
    ) -> None:
        """Fold in one action-node term ``v(child_sampled) − Σ_a π(a)·v(child_a)``.

        ``legal`` / ``probs`` must be aligned (``probs`` sums to 1) and ``action``
        must be in ``legal`` (the sampled action); a mismatch skips the term with a
        warning rather than corrupting the estimate.
        """
        legal = list(legal)
        if action not in legal:
            logger.warning(
                "AIVAT: sampled action %r not in legal %s at seat %d — skipping term",
                action, legal, seat,
            )
            return
        vals = self._v.child_values(env_before, legal)
        baseline = sum(float(p) * vals[a] for p, a in zip(probs, legal))
        self._sum_terms += vals[action] - baseline

    def finalize(self, terminal_env: PokerEnv) -> float:
        """Return ``aivat_value`` for the finished hand.

        ``u(z)`` is the hero's realised chip delta (identical to
        ``HandOutcome.hero_chips_delta``).  At a decision-free all-in terminal with a
        **cheap** runout (≤ ``_MAX_RUNOUT_CARDS`` board cards to come — a flop/turn
        all-in) the realised single board is replaced by the exact runout average (a
        zero-mean chance correction) — the holes are revealed at this showdown, so
        reading them for the integration is not a leak.  A **pre-flop** all-in (5
        cards to come) is skipped: its exact runout exceeds ``runout_equity``'s cap
        and would Monte-Carlo sample thousands of boards per hand, so the hand keeps
        only its action-node corrections.  Unbiased either way.
        """
        hero_delta = float(terminal_env.payout[self._hero_seat])
        chance_term = 0.0
        if terminal_env.is_decision_free and self._cheap_runout(terminal_env):
            with _preserve_global_random():
                eq = terminal_env.runout_equity(rng=self._rng, cap=self._runout_cap)
            chance_term = hero_delta - float(eq[self._hero_seat])
        return hero_delta - self._sum_terms - chance_term

    def _cheap_runout(self, terminal_env: PokerEnv) -> bool:
        """Whether the terminal all-in's runout is worth the chance correction.

        True iff at most ``self._max_runout_cards`` board cards remain.  At the default
        (2) only flop/turn all-ins qualify (``C(deck, ≤2)`` under the exact enumeration
        cap); raising it to 5 also admits a pre-flop all-in, whose 5-card runout
        ``runout_equity`` Monte-Carlo samples (bounded by ``runout_cap``).  An unknown
        board length is treated as expensive (skip) — the safe default.
        """
        board_len = terminal_env.terminal_board_len
        if board_len is None:
            return False
        return (5 - int(board_len)) <= self._max_runout_cards
