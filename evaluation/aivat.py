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
opt-in flag (``EvalConfig.aivat``) and driven by a dedicated RNG sub-stream so
turning it on does not perturb the deck / hero / opponent sampling — the raw
``hero_chips_delta`` of a hand is identical with AIVAT on or off.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np

from environment.poker_env import PokerEnv
from poker_ai.search.leaf import LeafConfig, continuation_value

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

    The leaf value machinery reshuffles the undealt deck via
    :meth:`Deck.shuffle_undealt`, which draws from the *global* ``np.random``
    (chance.py) — the same stream the played hand deals its board from.  AIVAT is a
    passive post-hoc estimator: it must not move that stream, or turning it on would
    change the real hand's later board cards.  Snapshotting and restoring the global
    state around each value evaluation makes AIVAT's deck reshuffling a no-op on the
    played hand (its own ``aivat_rng`` :class:`~numpy.random.Generator` is a separate
    object, unaffected by ``set_state``, so AIVAT's own sampling still advances
    normally and reproducibly).
    """
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


class _LeafCtx:
    """Minimal ``SubgameContext`` stand-in for :func:`continuation_value`.

    ``continuation_value`` reads only ``ctx.leaf`` (the fleet + rollout knobs) and
    ``ctx.rng`` (the rollout action-sampling source) — never the ranges / board mask
    — so the value function passes this lightweight duck-typed carrier instead of
    building a full :class:`SubgameContext` (which would need ranges we integrate
    over ourselves).
    """

    __slots__ = ("leaf", "rng")

    def __init__(self, leaf: LeafConfig, rng: np.random.Generator) -> None:
        self.leaf = leaf
        self.rng = rng


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
        its **fleet** (``policies``).  Its ``n_rollouts`` is *overridden* to
        ``value_rollouts`` (default 1): the estimator is unbiased at any rollout
        count and the ``n_hole_samples`` averaging already tames ``v``'s noise, so
        paying the session's search ``n_rollouts`` (e.g. 20) per value evaluation
        would be ~20× cost for negligible gain.
    rng
        Dedicated AIVAT RNG (a distinct seed sub-stream), used for both the belief
        hole sampling and the rollout runouts — kept separate from the hand's
        deck / hero / opponent RNGs so AIVAT never perturbs the played hand.
    n_hole_samples
        Number of joint hole draws averaged per ``v`` evaluation.
    value_rollouts
        Rollouts per :func:`continuation_value` call (default 1); see ``leaf_cfg``.
    """

    def __init__(
        self,
        hero,
        leaf_cfg: LeafConfig,
        rng: np.random.Generator,
        *,
        n_hole_samples: int = 6,
        value_rollouts: int = 1,
    ) -> None:
        self._hero = hero
        self._rng = rng
        self._m = int(n_hole_samples)
        self._rollouts = int(value_rollouts)
        # Reuse the session fleet but pin a cheap rollout count.  Whether to take the
        # exact decision-free runout is decided per call from the node's street (see
        # child_values), so it is not baked in here.
        self._base_use_equity = bool(leaf_cfg.use_decision_free_equity)
        self._leaf_exact = dataclasses.replace(
            leaf_cfg, n_rollouts=self._rollouts, use_decision_free_equity=True
        )
        self._leaf_sampled = dataclasses.replace(
            leaf_cfg, n_rollouts=self._rollouts, use_decision_free_equity=False
        )
        # Exact decision-free runout equities are memoised across all v calls of a
        # hand (keyed on (holes, runout snapshot) inside continuation_value), so a
        # repeated all-in integration runs once — the same shared-cache trick the
        # solver uses.
        self._runout_cache: dict = {}

    def child_values(
        self, env_before: PokerEnv, legal: Sequence[str]
    ) -> Dict[str, float]:
        """Mean hero-seat value of every ``legal`` action's child at ``env_before``.

        For each of ``n_hole_samples`` belief draws, materialises the sampled holes
        at ``env_before`` (board = the current street, so no board conflict), then
        steps each legal action via make/undo and scores the resulting child with
        :func:`continuation_value`.  ``env_before`` is never mutated (the sampled
        env is a fresh ``with_hole_cards`` copy).

        The exact decision-free runout is used only when the node is **post-flop**:
        a pre-flop-rooted rollout can reach a pre-flop all-in whose exact 5-card
        runout blows past ``runout_equity``'s enumeration cap (thousands of sampled
        boards per terminal).  So pre-flop nodes fall back to the sampled single
        board — mirroring the solver's own ``_use_equity = flag AND street_at_root
        != 0`` guard.  ``v`` is unbiased either way.
        """
        legal = list(legal)
        sums: Dict[str, float] = {a: 0.0 for a in legal}
        hero_seat = int(self._hero.my_seat)
        n_players = env_before.n_players
        profile = {s: "none" for s in range(n_players)}
        m = max(self._m, 1)
        use_exact = self._base_use_equity and env_before.betting_round != 0
        leaf = self._leaf_exact if use_exact else self._leaf_sampled
        ctx = _LeafCtx(leaf, self._rng)
        cache = self._runout_cache if use_exact else None
        with _preserve_global_random():
            for _ in range(m):
                holes = self._sample_joint(env_before)
                base = env_before.with_hole_cards(holes)
                for a in legal:
                    tok = base.step_in_place(a)
                    v = continuation_value(base, profile, ctx, cache)
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

    def __init__(self, hero_seat: int, value_fn: LeafValue, rng: np.random.Generator) -> None:
        self._hero_seat = int(hero_seat)
        self._v = value_fn
        self._rng = rng
        self._sum_terms = 0.0

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
                eq = terminal_env.runout_equity(rng=self._rng)
            chance_term = hero_delta - float(eq[self._hero_seat])
        return hero_delta - self._sum_terms - chance_term

    @staticmethod
    def _cheap_runout(terminal_env: PokerEnv) -> bool:
        """Whether the terminal all-in's runout is on ``runout_equity``'s exact path.

        True iff at most ``_MAX_RUNOUT_CARDS`` board cards remain (a flop/turn all-in
        — ``C(deck, ≤2)`` is always under the enumeration cap).  A pre-flop all-in (0
        board cards ⇒ 5 to come) returns False, and an unknown board length is
        treated as expensive (skip) — the safe default.
        """
        board_len = terminal_env.terminal_board_len
        if board_len is None:
            return False
        return (5 - int(board_len)) <= _MAX_RUNOUT_CARDS
