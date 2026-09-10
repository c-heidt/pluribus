"""AIVAT — variance-reduced strength estimate (doc §10.2, §9 step 9).

Per-hand chip variance is huge, so a raw ``hero_chips_delta`` bb/100 CI is wide.
AIVAT replaces it with a control variate of the **same mean** and smaller variance::

    aivat_value = u(z)  −  Σ correction_terms

Every term has **zero expectation** by construction, so ``E[aivat_value] == E[u(z)]``
for *any* value function ``v`` — a better ``v`` only shrinks variance, it can never
skew the mean.  Three families are taken:

- **Action nodes of a known-policy player** — here both the hero and the
  blueprint-derived opponents qualify, giving *full*-AIVAT rather than the hero-only
  partial case::

      term = v(child_sampled) − Σ_a π(a)·v(child_a)

  ``π`` is the hero's played σ at hero nodes (the vector logged as
  ``decisions.action_dist``) and ``BlueprintPolicy.strategy(state, bias)`` at
  opponent nodes.  Both exact, no estimation.

- **The terminal all-in runout**, when the hand ends decision-free with **≤2 board
  cards to come** (a flop/turn all-in) so ``PokerEnv.runout_equity`` can enumerate it
  exactly::

      term = u(z) − runout_equity[hero]

  A pre-flop all-in (5 to come) exceeds that cap and is skipped.

- **Per-street chance nodes at the turn and river** (opt-in, ``EvalConfig
  .aivat_chance``)::

      term = v(h·c_dealt) − Σ_c P(c)·v(h·c)

  ``P`` is **exactly** uniform over ``deck.remaining``: conditioned on the full
  history (holes included, as ``v`` already is), that is the true conditional, so the
  baseline is an enumeration rather than a sample.  The **flop** deals three cards at
  once — an unordered triple, ``C(48,3)`` of them — so it is not enumerable and is
  left out.  Note the action-node terms remove **no** board variance: each rollout
  reshuffles the undealt deck, so ``v(child_a)`` is already a board-average on both
  sides of the difference while ``u(z)`` carries the realised board in full.

**The value function ``v``.**  The paper's ``u^σ``: the expected value of history
``h`` under a fixed baseline profile σ, evaluated at the hand's **actual** holes and
Monte-Carlo'd by ``n_rollouts`` playouts through
:func:`poker_ai.search.leaf.continuation_value` under an all-blueprint continuation
fleet.  Reading the true holes is not an information leak — the estimator is offline
and never chooses an action, and unbiasedness holds regardless: conditioned on the
full history the sampled action came from ``π`` and ``v`` is a fixed function of the
resulting state, so ``E[v(child_A)] = Σ_a π(a)·v(child_a)`` whatever ``v`` reads.

**RNG isolation.**  AIVAT owns three stream pairs — action-node sampling/board,
chance sampling/board — and reads the global ``np.random`` nowhere; that stream
belongs to the played hand's deal.  Outward this keeps AIVAT **passive** (a hand's
raw ``hero_chips_delta`` is identical with it on or off).  Inward it makes
``aivat_value`` a pure function of ``(run_seed, hand_index)`` and the played line, so
it cancels in the CRN-paired Δ instead of injecting arm-specific noise.  See
:mod:`poker_ai.search.rng`.

**Cost** lands in the experiment budget, never the real-time search path.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from environment.poker_env import PokerEnv
from poker_ai.search.leaf import LeafConfig, continuation_value
from poker_ai.search.rng import spawn, spawn_one

logger = logging.getLogger(__name__)

# Max board cards still to come for the terminal all-in chance correction to run.
# ≤2 (flop/turn all-in) is under ``runout_equity``'s exact-enumeration cap; a pre-flop
# all-in (5 to come) would Monte-Carlo thousands of boards per hand, so it is skipped.
_MAX_RUNOUT_CARDS = 2

# Baseline playouts averaged per action-node ``v`` evaluation.
# MEASURED (1555 real-blueprint hands, every m scored on the same played hands):
#     m=3  var_x 1.114   m=6  1.278   m=16  1.559   m=48  1.817   m=144  1.755
# Raising this from the old default of 6 to 48 is the single largest variance-reduction
# win available to the estimator: gain 1.422, 95% CI [1.21, 1.68].  A ``var = a + b/m``
# fit puts the m->inf ceiling at 1.78-1.84, so 48 is essentially at it.  Note m=3 gives
# var_x ~1.0 — no reduction at all — so the working range is narrower than it looks.
# Cost is ~5% of hand wall-clock at a reduced search budget, less at production budgets.
_ACTION_ROLLOUTS = 48

# Baseline playouts per alternative card in a per-street chance correction.
# MEASURED, not reasoned (3105 real-blueprint hands): m=2 makes the correction
# *add* variance (gain 0.885, CI [0.81, 0.96]).  The tempting argument — "the baseline
# enumerates 12–45 cards, so √|remaining| already kills the noise" — covers only the
# BASELINE half; ``v̂(c_dealt)`` stands alone with weight ``(1 − 1/N)`` and needs
# rollouts of its own.  48 sits at the ceiling (higher buys nothing) and costs <1% of
# hand wall-clock.
_CHANCE_ROLLOUTS = 48


@contextlib.contextmanager
def _preserve_global_random():
    """Run a block, then restore the **global** ``np.random`` state it consumed.

    A defensive backstop, not the isolation mechanism: AIVAT draws from its own
    streams, so this normally restores a no-op.  It stays so that passivity does not
    depend on every engine path below ``v`` honouring its ``rng`` argument.
    """
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


class _LeafCtx:
    """Minimal ``SubgameContext`` stand-in for :func:`continuation_value`.

    That function reads only ``ctx.leaf``, ``ctx.rng`` and ``ctx.board_rng`` — never
    the ranges / board mask — so this duck-typed carrier is enough.
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
    """Baseline value function ``v(h) → hero-seat chip delta`` (the paper's ``u^σ``).

    Averages :func:`continuation_value`'s hero-seat entry over ``n_rollouts`` playouts
    under a fixed continuation profile σ, at the hand's **actual** holes.

    Parameters
    ----------
    hero
        Any agent exposing ``my_seat`` — the seat whose chip delta ``v`` returns.
        ``my_hole`` / ``tracker`` are read only by the legacy :meth:`_sample_joint`.
    leaf_cfg
        Continuation leaf config; reuse the session's ``solver_cfg.leaf`` for its
        policy fleet.
    rng
        Dedicated AIVAT stream.  Three further streams are **spawned** from it (so its
        own byte-stream is untouched): a board-runout partner, and a separate
        sampling/board pair for chance corrections.  Splitting board draws out of the
        sampling stream keeps a board draw from desyncing the sampling trajectory;
        giving chance its own pair means the action-node terms are bit-identical
        whether chance corrections are on or off.
    n_rollouts
        Baseline playouts averaged per ``v`` evaluation.  See ``_ACTION_ROLLOUTS``:
        this is the estimator's dominant variance knob, and small values are
        measurably useless (m=3 reduces no variance at all).
    seat_bias
        ``seat → BiasClass`` for σ.  Supply the table's actual composition so σ
        *reproduces* the opponents rather than resembling them — their true policy is
        exactly ``blueprint.strategy(state, bias)``, and the runner hands the same
        ``BlueprintPolicy`` object to the leaf fleet and the opponents.  Missing seats
        fall back to ``"none"``.  The **hero's** seat deliberately stays ``"none"``:
        its real continuation is a search result with no policy form at arbitrary
        future nodes, and substituting one would make ``v`` arm-dependent.  ``v`` may
        be wrong about the hero; it only has to be consistent.
    """

    def __init__(
        self,
        hero,
        leaf_cfg: LeafConfig,
        rng: np.random.Generator,
        *,
        n_rollouts: int = _ACTION_ROLLOUTS,
        seat_bias: Optional[Mapping[int, str]] = None,
    ) -> None:
        self._hero = hero
        self._rng = rng
        self._board_rng = spawn_one(rng)
        # Spawned AFTER ``_board_rng`` so enabling chance corrections leaves every
        # existing stream — and every number it produces — untouched.
        self._chance_rng, self._chance_board_rng = spawn(rng, 2)
        self._m = int(n_rollouts)
        self._leaf = leaf_cfg
        self._seat_bias = dict(seat_bias or {})

    def _rollout_setup(self, env_before: PokerEnv, *, chance: bool = False):
        """``(profile, holes, ctx)`` shared by every ``v`` evaluation at this node.

        ``holes`` covers every seat, folded ones included, since ``v(h)`` is the
        baseline value of the true history.  ``ctx`` carries the action-node stream
        pair, or (``chance=True``) the separate chance pair.
        """
        n_players = env_before.n_players
        profile = {s: self._seat_bias.get(s, "none") for s in range(n_players)}
        holes = [
            tuple(int(c) for c in env_before.players[s].cards)
            for s in range(n_players)
        ]
        streams = (
            (self._chance_rng, self._chance_board_rng) if chance
            else (self._rng, self._board_rng)
        )
        return profile, holes, _LeafCtx(self._leaf, *streams)

    def child_values(
        self, env_before: PokerEnv, legal: Sequence[str]
    ) -> Dict[str, float]:
        """Mean hero-seat value of every ``legal`` action's child at ``env_before``.

        Steps each legal action via make/undo and scores the child with
        :func:`continuation_value`.  ``env_before`` is never mutated — each rollout
        works on a fresh ``with_hole_cards`` copy, whose reshuffle of the undealt deck
        is what makes rollouts independent draws.

        **Siblings are scored under common random numbers**: both streams are rewound
        before every legal action, so within a rollout the children differ *only* by
        the action taken.  This matters because :func:`continuation_value` is a
        one-rollout estimate carrying large MC noise, and the correction uses ``v̂``
        only inside ``v̂(child_taken) − Σ_a π(a)·v̂(child_a)`` — under CRN the shared
        noise cancels there instead of accumulating.  Scoring siblings independently
        was measured to make MC noise 69% of the correction's variance.

        Unbiasedness is untouched: ``E[v̂(child_a)] = v(child_a)`` holds per action,
        and the control-variate identity needs nothing about how the noise correlates.
        """
        legal = list(legal)
        sums: Dict[str, float] = {a: 0.0 for a in legal}
        hero_seat = int(self._hero.my_seat)
        profile, holes, ctx = self._rollout_setup(env_before)
        m = max(self._m, 1)
        env_legal = set(env_before.legal_actions)
        to_inject = [a for a in legal if a not in env_legal]
        with _preserve_global_random():
            for _ in range(m):
                base = env_before.with_hole_cards(holes, rng=self._board_rng)
                for a in to_inject:
                    base.inject_action(a)
                sample_rng = self._rng.bit_generator.state
                sample_board = self._board_rng.bit_generator.state
                for a in legal:
                    self._rng.bit_generator.state = sample_rng
                    self._board_rng.bit_generator.state = sample_board
                    tok = base.step_in_place(a)
                    v = continuation_value(base, profile, ctx)
                    sums[a] += float(v[hero_seat])
                    base.undo(tok)
                # Streams are left wherever the last sibling ended — a deterministic
                # function of the situation, so the next node stays reproducible.
        return {a: sums[a] / m for a in legal}

    def chance_values(
        self, env_before: PokerEnv, action: str, *, n_rollouts: int = _CHANCE_ROLLOUTS
    ) -> Optional[Tuple[float, float]]:
        """``(v(dealt card), Σ_c P(c)·v(c))`` for the street deal fused into ``action``.

        The engine has no standalone chance node — a street is dealt *inside*
        ``_apply_action_in_place`` when an action closes the betting round — so the
        event is discovered by taking the step and seeing whether the board grew.

        Returns ``None`` unless the step deals **exactly one** card and leaves a
        non-terminal state, which restricts this to the turn and river:

        - 0 cards → the action did not close the round; no chance node.
        - 3 cards → the flop; not enumerable (see the module docstring).
        - terminal → the board is force-dealt by ``_settle_terminal`` and that runout
          is already integrated by :meth:`AivatAccumulator.finalize`, so correcting
          here too would double-count.

        ``remaining[0]`` is the card the played hand actually receives, so the realised
        outcome needs no lookahead and is scored as one more alternative.  Alternatives
        are scored under common random numbers, as in :meth:`child_values`.

        Unbiased for any ``v``: conditioned on ``h·action`` the dealt card ``C`` comes
        from ``P``, so ``E[v(C)] = Σ_c P(c)·v(c)``.  The caller's term is
        ``v_dealt − baseline``.
        """
        hero_seat = int(self._hero.my_seat)
        profile, holes, ctx = self._rollout_setup(env_before, chance=True)
        rng, board_rng = ctx.rng, ctx.board_rng
        m = max(int(n_rollouts), 1)
        n_before = len(env_before.community_cards)
        # Support comes off the LIVE deck: a copy's reshuffle changes the undealt
        # *order* but not the undealt *set*, which is all the event depends on.
        alternatives = [int(c) for c in env_before.deck.remaining]
        if not alternatives:
            return None

        needs_inject = action not in set(env_before.legal_actions)
        with _preserve_global_random():
            base = env_before.with_hole_cards(holes, rng=board_rng)
            if needs_inject:
                base.inject_action(action)
            # Probe the step for its chance event.  The betting engine is
            # board-independent, so the card count and terminality do not depend on
            # the copy's reshuffled deck order.
            tok = base.step_in_place(action)
            n_dealt = len(base.community_cards) - n_before
            terminal = base.is_terminal
            base.undo(tok)
            if terminal or n_dealt != 1:
                return None

            sums = [0.0] * len(alternatives)
            for r in range(m):
                if r:
                    base = env_before.with_hole_cards(holes, rng=board_rng)
                    if needs_inject:
                        base.inject_action(action)
                round_rng = rng.bit_generator.state
                round_board = board_rng.bit_generator.state
                for i, card in enumerate(alternatives):
                    rng.bit_generator.state = round_rng
                    board_rng.bit_generator.state = round_board
                    forced = base.deck.force_next((card,))
                    tok = base.step_in_place(action)
                    v = continuation_value(base, profile, ctx)
                    sums[i] += float(v[hero_seat])
                    base.undo(tok)
                    # ``undo`` restores the deal cursor but NOT ``_cards`` — every
                    # force must be reversed by hand (see ``Deck.force_next``).
                    base.deck.unforce(forced)

        v_dealt = sums[0] / m                       # alternatives[0] == the real card
        baseline = sum(sums) / (m * len(sums))      # uniform P(c) over ``remaining``
        return v_dealt, baseline

    # ------------------------------------------------------------------ #
    # Belief joint hole sampling (card-disjoint, board-masked)
    # ------------------------------------------------------------------ #
    def _sample_joint(self, env: PokerEnv) -> List[Tuple[int, int]]:
        """One card-disjoint hole assignment for every seat (hero = its own hole).

        .. note::
           **Not used by the AIVAT estimator** — ``child_values`` evaluates ``v`` at
           the actual holes, as the paper does.  Retained for
           :mod:`evaluation.calibrate`, which wants belief-drawn opponent worlds.

        Non-hero seats are drawn from their tracked belief (live or fold-time), masked
        to the board; sequential-with-removal keeps the draw disjoint.  A hero without
        a tracker falls back to uniform over board-compatible combos.  Both are mild
        approximations, harmless because ``v`` need only be consistent.
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
    has no board-compatible mass left."""
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
        # Belief collapsed against the board / removal — uniform over what is left.
        w = board_ok.astype(np.float64)
        if used:
            w = w.copy()
            w[conflict] = 0.0
        total = w.sum()
        if total <= 0.0:
            raise ValueError("AIVAT joint sampler: card exhaustion for a seat.")
    return w / total


class AivatAccumulator:
    """Accumulates one hand's correction terms → the ``aivat_value`` scalar.

    Fed one ``(env_before, seat, action, legal, probs)`` event per known-policy action
    node during play; :meth:`finalize` reads the terminal payout, applies the all-in
    runout correction, and returns
    ``u(z) − Σ action_terms − Σ chance_terms − runout_term``.
    """

    def __init__(self, hero_seat: int, value_fn: LeafValue, rng: np.random.Generator,
                 *, max_runout_cards: int = _MAX_RUNOUT_CARDS,
                 runout_cap: int = 5000,
                 chance: bool = False,
                 chance_rollouts: int = _CHANCE_ROLLOUTS) -> None:
        self._hero_seat = int(hero_seat)
        self._v = value_fn
        self._rng = rng
        # Kept apart so a caller can score BOTH variants from one played hand:
        # ``aivat_without_chance == finalize(...) + sum_chance_terms``, exactly.
        self._sum_terms = 0.0
        self._sum_chance = 0.0
        self._chance = bool(chance)
        self._chance_rollouts = int(chance_rollouts)
        self.n_action_terms = 0
        self.n_chance_terms = 0
        # Terminal all-in runout reach.  2 = flop/turn all-ins (exact enumeration);
        # the calibration raises it to 5 to also cover a pre-flop all-in, with
        # ``runout_cap`` bounding the Monte-Carlo board sample.
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

        ``legal`` / ``probs`` must be aligned (``probs`` sums to 1) and ``action`` must
        be in ``legal``; a mismatch skips the term with a warning rather than
        corrupting the estimate.

        With chance corrections enabled this **also** takes the term for the street
        ``action`` deals, if it deals one — the engine fuses the deal into the closing
        action.  The two families are independent; a ``legal`` mismatch skips only the
        action one.  A round closed after the hero folds takes no chance term, which
        costs nothing: the hero's payout is then settled, so ``v`` is constant in the
        board and the term is exactly zero.
        """
        legal = list(legal)
        if self._chance:
            chance = self._v.chance_values(
                env_before, action, n_rollouts=self._chance_rollouts
            )
            if chance is not None:
                v_dealt, baseline = chance
                self._sum_chance += v_dealt - baseline
                self.n_chance_terms += 1
        if action not in legal:
            logger.warning(
                "AIVAT: sampled action %r not in legal %s at seat %d — skipping term",
                action, legal, seat,
            )
            return
        vals = self._v.child_values(env_before, legal)
        baseline = sum(float(p) * vals[a] for p, a in zip(probs, legal))
        self._sum_terms += vals[action] - baseline
        self.n_action_terms += 1

    def finalize(self, terminal_env: PokerEnv) -> float:
        """Return ``aivat_value`` for the finished hand.

        ``u(z)`` is the hero's realised chip delta.  At a decision-free all-in terminal
        with ≤ ``_MAX_RUNOUT_CARDS`` board cards to come, the realised single board is
        replaced by the exact runout average — the holes are revealed at that showdown,
        so integrating over them is not a leak.  Unbiased either way.
        """
        hero_delta = float(terminal_env.payout[self._hero_seat])
        chance_term = 0.0
        if terminal_env.is_decision_free and self._cheap_runout(terminal_env):
            with _preserve_global_random():
                eq = terminal_env.runout_equity(rng=self._rng, cap=self._runout_cap)
            chance_term = hero_delta - float(eq[self._hero_seat])
        return hero_delta - self._sum_terms - self._sum_chance - chance_term

    @property
    def sum_chance_terms(self) -> float:
        """Σ of this hand's per-street chance terms (0.0 when they are off).

        ``finalize(...) + sum_chance_terms`` is exactly the ``aivat_value`` this hand
        would have produced with ``chance=False`` — how the two variants are compared
        on identical played hands.
        """
        return self._sum_chance

    @property
    def sum_action_terms(self) -> float:
        """Σ of this hand's action-node terms."""
        return self._sum_terms

    def _cheap_runout(self, terminal_env: PokerEnv) -> bool:
        """True iff at most ``self._max_runout_cards`` board cards remain.

        An unknown board length is treated as expensive (skip) — the safe default.
        """
        board_len = terminal_env.terminal_board_len
        if board_len is None:
            return False
        return (5 - int(board_len)) <= self._max_runout_cards
