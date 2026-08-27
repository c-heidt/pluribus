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
  boards per hand, so those hands keep only their action-node corrections.

- **Per-street chance nodes — the turn and the river** (opt-in, ``EvalConfig
  .aivat_chance``).  When an action closes a betting round the engine deals the next
  street inside ``_apply_action_in_place``; ``Deck.force_next`` re-deals that street
  as each alternative undealt card down the same betting line, giving::

      term = v(h·c_dealt) − Σ_c P(c)·v(h·c)

  with ``P`` **exactly** uniform over ``deck.remaining`` (conditioned on the full
  history, holes included, that is the true conditional).  A single card means the
  baseline is an exact enumeration, not a sample.  The **flop** deals three cards at
  once — an unordered triple, ``C(48,3)`` of them — so it is not enumerable and is
  left out pending a sampled variant.

  This is the term that can still move the number.  The action-node corrections
  above remove **no** board variance: each rollout re-copies through
  ``with_hole_cards``, which reshuffles the undealt deck, so ``v(child_a)`` is
  already a board-average on both sides of the difference, while ``u(z)`` carries
  the realised board in full.

**The value function ``v``.**  As in the AIVAT paper: ``v(h)`` is the expected value
of history ``h`` under a fixed baseline strategy profile, evaluated at the hand's
**actual** holes.  Implemented by reusing the search's own leaf machinery
(:func:`poker_ai.search.leaf.continuation_value`) under an all-blueprint continuation
profile — that profile is the baseline σ — and averaging a handful of rollouts,
because the exact expectation is far too large to enumerate here.

**Why the actual holes, and why that is not an information leak.**  The estimator is
**offline**: it never chooses an action, so nothing it reads can make the bot cheat
or inflate the strength number.  The rule that private information must not be read
constrains the *agent*, not the evaluator, and conflating the two costs real
accuracy.  Unbiasedness does not depend on it either — condition on the full history
(every player's holes included); the sampled action ``A`` is drawn from ``π`` and
``v`` is a fixed function of the resulting state, so
``E[v(child_A)] = Σ_a π(a)·v(child_a)`` and the term is zero-mean *whatever* ``v``
reads.  That argument covers opponent nodes too, where ``π`` conditions on that
opponent's own hole — exactly the distribution the action was sampled from.

Reading the true holes is also what the rest of this module already does: the
terminal runout correction integrates ``runout_equity`` over them.  An earlier
version instead integrated ``v`` over the tracker's *belief*, which was a deviation
from the paper that bought nothing (the unbiasedness argument above never needed it)
and cost twice: it added belief-sampling noise on top of the rollout noise, and it
made ``aivat_value`` depend on the arm under test — two methods hold different
posteriors after their searches diverge, so the correction stopped cancelling in the
CRN-paired Δ that the evaluation actually reports.

**Cost.**  All of this lands in the *experiment* time budget, never the real-time
search hot path (§10.2): each ``v`` call is a few baseline rollouts, and it is
evaluated at the (2–4) legal siblings of each decision.  It is gated behind an
opt-in flag (``EvalConfig.aivat``).

**RNG isolation — two directions, both required.**  AIVAT owns dedicated streams
(``rng`` for rollout actions, a spawned child for board runouts) and reads
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

``v`` is now arm-independent: it reads the actual holes and a fixed baseline
profile, so two methods sharing a deal compute the *same* ``v`` at the same node.
The only remaining arm-dependence is ``π`` itself — the played σ, which genuinely
differs between methods — so the correction cancels in the paired Δ wherever the
two arms' play coincides.
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
# ≤2 (a flop/turn all-in) is always under ``runout_equity``'s exact-enumeration cap,
# so the correction is exact and cheap.  A pre-flop all-in (5 to come) would exceed
# the cap and fall back to Monte-Carlo *sampling thousands of boards* per hand — an
# expensive, log-noisy operation — so it is skipped (the hand keeps its action-node
# corrections), mirroring the solver's own ``street_at_root != 0`` equity guard.
_MAX_RUNOUT_CARDS = 2

# Baseline playouts per alternative card in a per-street chance correction.
#
# ⚠️MEASURED, not reasoned.  The obvious argument — "the baseline enumerates 12-45
# cards, so ``√|remaining|`` already suppresses the noise and this can be tiny" — is
# WRONG, and a default of 2 derived that way was measured to make the correction
# ADD variance (gain 0.885, 95% CI [0.81, 0.96] over 3105 real-blueprint hands).
# The flaw: only the BASELINE half of ``v̂(c_dealt) − Σ_c P(c)·v̂(c)`` gets that
# averaging.  ``v̂(c_dealt)`` stands alone with weight ``(1 − 1/N)`` and its own
# Monte-Carlo noise is untouched by the enumeration, so it needs real rollouts of
# its own — the same trap that made the action-node terms harmful before sibling CRN.
#
# Measured gain vs chance-off, same played hands (3105 hands, 2p 20-card blueprint):
#     m=2  0.885 [0.81, 0.96]   m=6  1.007   m=16  1.011   m=48  1.035 [0.99, 1.10]
# A ``var = a + b/m`` fit puts the m→∞ ceiling at 1.031–1.046 depending on which
# points it is fitted on — i.e. 48 is already AT the ceiling and raising it further
# buys nothing.  Cost is not the constraint either: the whole four-value sweep was
# 2.4% of hand wall-clock, so a single m=48 is well under 1%.
_CHANCE_ROLLOUTS = 48


@contextlib.contextmanager
def _preserve_global_random():
    """Run a block, then restore the **global** ``np.random`` state it consumed.

    A **defensive backstop, not the isolation mechanism.**  AIVAT draws every
    random number it needs from its own streams (``self._rng`` for rollout actions,
    ``self._board_rng`` for board runouts), so in the normal case
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
    """Baseline value function ``v(h) → hero-seat chip delta`` (AIVAT paper's ``u^σ``).

    Averages :func:`continuation_value`'s hero-seat entry over ``n_rollouts``
    playouts under a fixed all-blueprint continuation profile (the baseline σ),
    evaluated at the hand's **actual** holes.  The average is a Monte-Carlo stand-in
    for the exact expectation, which is too large to enumerate; more rollouts ⇒ a
    lower-noise ``v``, never a different mean.

    Parameters
    ----------
    hero
        Any agent exposing ``my_seat`` — the seat whose chip delta ``v`` returns.
        (``my_hole`` / ``tracker`` are no longer read by the estimator; the belief
        sampler :meth:`_sample_joint` still uses them and is still consumed by
        :mod:`evaluation.calibrate`.)
    leaf_cfg
        The continuation leaf config — reuse the session's ``solver_cfg.leaf`` for
        its **fleet** (``policies``); passed straight through (``continuation_value``
        always takes exactly one rollout, so there is nothing left to override).
    rng
        Dedicated AIVAT RNG (a distinct seed sub-stream), used for the rollout
        action draws.  A second stream for the rollout
        **board** runouts is spawned from it — split for the same reason the solver
        splits its own (a board draw interleaved into the sampling stream desyncs
        the sampling trajectory), and spawned rather than drawn from so ``rng``'s
        byte-stream is unchanged by the split.

        Both streams are AIVAT's alone.  Nothing here reads the global
        ``np.random``, which belongs to the played hand's deal: that independence
        is what makes ``aivat_value`` a pure function of ``(run_seed, hand_index)``
        and the played line, so two CRN-paired arms that play a hand identically
        produce an identical correction and it cancels exactly in the paired Δ.
    n_rollouts
        Number of baseline playouts averaged per ``v`` evaluation.
    seat_bias
        ``seat → BiasClass`` for the baseline continuation profile σ.  Supply the
        table's **actual** composition (``bp_fold`` → ``"fold"`` and so on) so the
        rollout continues the way the real opponents do; the runner derives it from
        the same ``seat_labels`` that built the opponents.  Missing seats — and the
        hero's own — fall back to ``"none"``.

        The opponents' true policy is *exactly* ``blueprint.strategy(state, bias)``,
        so getting the bias right makes σ match them rather than merely resemble
        them — **at any bias magnitude**, not just the default.  The magnitude lives
        in ``BlueprintPolicy.bias_multiplier``, and
        :func:`evaluation.runner.build_blueprint_session` hands the *same* policy
        object to the leaf fleet and to the opponents, so the two cannot drift apart
        whatever it is set to.  That sharing is what makes the match exact rather
        than approximate; it is pinned by
        ``test_runner.py::test_leaf_fleet_shares_the_opponents_policy_object``.  A
        session that built the fleet and the opponents from *separate* policies
        (the stub does) still benefits from the right bias class, but σ would then
        only approximate the opponents rather than reproduce them.  The **hero's** seat deliberately stays ``"none"``: its real
        continuation is a search result that does not exist as a policy at arbitrary
        future nodes, and substituting one would make ``v`` arm-dependent again —
        the thing the actual-holes change just removed.  ``v`` is allowed to be
        wrong about the hero; it only has to be *consistent*, and a better σ for the
        seats it can model is a strictly better control variate.
    """

    def __init__(
        self,
        hero,
        leaf_cfg: LeafConfig,
        rng: np.random.Generator,
        *,
        n_rollouts: int = 6,
        seat_bias: Optional[Mapping[int, str]] = None,
    ) -> None:
        self._hero = hero
        self._rng = rng
        self._board_rng = spawn_one(rng)
        # Chance corrections draw from their OWN pair of streams, not the two
        # above.  Same "one stream per consumer" rule the search follows
        # (:mod:`poker_ai.search.rng`), and here it buys something concrete: the
        # action-node terms come out bit-identical whether chance corrections are
        # on or off, so the two variants can be scored on the same hand and
        # differenced exactly.  Spawned *after* ``_board_rng`` so enabling this
        # leaves the existing streams — and every number they produce — untouched.
        self._chance_rng, self._chance_board_rng = spawn(rng, 2)
        self._m = int(n_rollouts)
        self._leaf = leaf_cfg
        self._seat_bias = dict(seat_bias or {})

    def _rollout_setup(self, env_before: PokerEnv, *, chance: bool = False):
        """``(profile, holes, ctx)`` shared by every ``v`` evaluation at this node.

        - ``profile`` — the baseline σ: each seat continues under its ACTUAL bias
          class where known (see ``seat_bias``), else the unbiased blueprint.
        - ``holes`` — the hand's real holes, every seat and folded ones included,
          since ``v(h)`` is the baseline value of the true history ``h``.
        - ``ctx`` — the duck-typed leaf carrier over the caller's stream pair:
          the action-node streams, or (``chance=True``) the separate chance pair.
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

        Runs ``n_rollouts`` baseline playouts at the hand's **actual** holes (the
        paper's ``u^σ``), stepping each legal action via make/undo and scoring the
        resulting child with :func:`continuation_value`.  ``env_before`` is never
        mutated — each rollout works on a fresh ``with_hole_cards`` copy, which also
        reshuffles the undealt deck so successive rollouts see different runouts.

        **Siblings are scored under common random numbers.**  Both streams are
        rewound to the same state before every legal action, so within one rollout
        the children differ *only* by the action taken — same board runout, same
        rollout action line wherever the line is shared.

        This matters because :func:`continuation_value` is a **one-rollout** estimate
        (one sampled action line on one sampled board — the solver's design, not an
        accident here), so an individual ``v̂`` carries enormous Monte-Carlo noise.
        The correction only ever uses ``v̂`` inside the difference
        ``v̂(child_taken) − Σ_a π(a)·v̂(child_a)``, and under CRN the noise the
        siblings share cancels in that difference instead of accumulating into it.
        Scoring them independently was measured to make MC noise **69% of the
        correction's variance**, which is what left AIVAT *adding* variance rather
        than removing it.

        Unbiasedness is untouched: ``E[v̂(child_a)] = v(child_a)`` still holds for
        each ``a`` separately, and the control-variate identity needs nothing about
        how the siblings' noise is correlated — only about each one's mean.
        """
        legal = list(legal)
        sums: Dict[str, float] = {a: 0.0 for a in legal}
        hero_seat = int(self._hero.my_seat)
        profile, holes, ctx = self._rollout_setup(env_before)
        m = max(self._m, 1)
        with _preserve_global_random():
            for _ in range(m):
                # Re-copy per rollout: ``with_hole_cards`` reshuffles the undealt
                # deck off ``board_rng``, which is what makes each rollout an
                # independent draw of the continuation.
                base = env_before.with_hole_cards(holes, rng=self._board_rng)
                sample_rng = self._rng.bit_generator.state
                sample_board = self._board_rng.bit_generator.state
                for a in legal:
                    self._rng.bit_generator.state = sample_rng
                    self._board_rng.bit_generator.state = sample_board
                    tok = base.step_in_place(a)
                    v = continuation_value(base, profile, ctx)
                    sums[a] += float(v[hero_seat])
                    base.undo(tok)
                # Leaves both streams wherever the last sibling ended — a
                # deterministic function of the situation, so the next rollout and
                # the next node stay reproducible.
        return {a: sums[a] / m for a in legal}

    def chance_values(
        self, env_before: PokerEnv, action: str, *, n_rollouts: int = 2
    ) -> Optional[Tuple[float, float]]:
        """``(v(dealt card), Σ_c P(c)·v(c))`` for the street deal fused into ``action``.

        The engine has no standalone chance node: a street is dealt *inside*
        ``_apply_action_in_place`` when an action closes the betting round
        (:meth:`PokerEnv._increment_stage`).  So the chance event that follows
        ``h·action`` is discovered by taking the step and seeing whether the board
        grew — no duplication of "does this action close the round?" logic.

        Returns ``None`` — no correction available — unless the step deals
        **exactly one** card and leaves a non-terminal state.  That restricts this
        to the **turn and river**:

        - 0 cards → the action did not close the round; there is no chance node.
        - 3 cards → the flop.  Its event is an unordered triple, ``C(48,3) = 17296``
          of them on a full deck, so it cannot be enumerated and needs a *sampled*
          baseline with its own knob; deliberately deferred.
        - a terminal step → the board is force-dealt to five by
          :meth:`PokerEnv._settle_terminal`, and that runout is already
          Rao-Blackwellised exactly by :meth:`AivatAccumulator.finalize`'s
          ``runout_equity`` term.  Correcting it here as well would double-count.

        **The event's distribution is exactly uniform over** ``deck.remaining``, and
        that is not an approximation: conditioning on the full history ``h`` — every
        seat's holes included, as ``v`` already does — the next card is uniform over
        the cards that are neither dealt to a player nor on the board, which is
        precisely what ``remaining`` holds.  ``remaining[0]`` is the card the played
        hand actually receives (``deal_community`` slices from the cursor), so the
        realised outcome needs no lookahead and is scored as just one more
        alternative, under the same code path as the rest.

        **Why this is the term that can still pay.**  The action-node correction in
        :meth:`child_values` removes **no** board variance at all: every rollout
        there re-copies through ``with_hole_cards``, which reshuffles the undealt
        deck, so ``v(child_a)`` is already a board-*average* on both sides of
        ``v(child_taken) − Σ_a π(a)·v(child_a)``.  Meanwhile ``u(z)`` carries the
        full realised board.  Integrating the street deal is what puts the two on
        the same footing.

        **Cost, and why ``n_rollouts`` is small.**  One card means the baseline is an
        exact enumeration over ``|remaining|`` alternatives (12–13 heads-up on the
        20-card LUT, 44–45 on a full deck), and averaging over that many draws
        already suppresses rollout noise by ``√|remaining|`` — so this needs far
        fewer rollouts *per alternative* than the 2–4-way action nodes do.  Siblings
        are scored under common random numbers exactly as in :meth:`child_values`:
        both streams are rewound before each alternative, so within a round the
        boards differ only by the forced card.

        Unbiasedness is the standard control-variate argument and needs nothing from
        ``v`` beyond consistency: conditioned on ``h·action``, the dealt card ``C``
        is drawn from ``P``, so ``E[v(C)] = Σ_c P(c)·v(c)`` and the difference the
        caller forms is zero-mean whatever ``v`` reads or how noisy it is.

        Parameters
        ----------
        env_before
            Pre-action state at the node (never mutated — every rollout works on a
            ``with_hole_cards`` copy, which is also the only env the deck force is
            ever applied to).
        action
            The action about to be played, whose step may deal the street.
        n_rollouts
            Baseline playouts per alternative card.

        Returns
        -------
        tuple[float, float] or None
            ``(v_dealt, baseline)``; the caller's term is ``v_dealt − baseline``.
        """
        hero_seat = int(self._hero.my_seat)
        profile, holes, ctx = self._rollout_setup(env_before, chance=True)
        rng, board_rng = ctx.rng, ctx.board_rng
        m = max(int(n_rollouts), 1)
        n_before = len(env_before.community_cards)
        # Read the support off the LIVE deck: ``with_hole_cards`` reshuffles the
        # undealt region, so a copy's order differs — but the undealt *set*, which
        # is all that defines the event, does not.
        alternatives = [int(c) for c in env_before.deck.remaining]
        if not alternatives:
            return None

        with _preserve_global_random():
            base = env_before.with_hole_cards(holes, rng=board_rng)
            # Probe the step for its chance event.  The betting engine is
            # board-independent, so how many cards this deals and whether the state
            # is terminal do not depend on the copy's reshuffled deck order.
            tok = base.step_in_place(action)
            n_dealt = len(base.community_cards) - n_before
            terminal = base.is_terminal
            base.undo(tok)
            if terminal or n_dealt != 1:
                return None

            sums = [0.0] * len(alternatives)
            for r in range(m):
                if r:
                    # Fresh copy per round, as in ``child_values``: the reshuffle is
                    # what makes each round an independent draw of the continuation
                    # *beyond* the forced card.
                    base = env_before.with_hole_cards(holes, rng=board_rng)
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
                    # ``undo`` restores the deal cursor but NOT ``_cards`` — the
                    # force has to be reversed by hand (see ``Deck.force_next``).
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
           **Not used by the AIVAT estimator any more** — ``child_values`` evaluates
           ``v`` at the hand's actual holes, as the paper does.  This belief sampler
           is retained for :mod:`evaluation.calibrate`, which genuinely wants
           belief-drawn opponent worlds for its CRN root-value estimate.


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
                 runout_cap: int = 5000,
                 chance: bool = False,
                 chance_rollouts: int = _CHANCE_ROLLOUTS) -> None:
        self._hero_seat = int(hero_seat)
        self._v = value_fn
        self._rng = rng
        # Kept apart so a caller can score BOTH variants from one played hand:
        # ``aivat_without_chance == finalize(...) + sum_chance_terms``.  The two
        # families draw from separate streams, so that identity is exact, not an
        # approximation (see :class:`LeafValue`).
        self._sum_terms = 0.0
        self._sum_chance = 0.0
        # Per-street (turn/river) chance corrections — opt-in, see
        # :meth:`LeafValue.chance_values`.  Off by default so the estimator's
        # scope stays the measured one until the A/B says otherwise.
        self._chance = bool(chance)
        self._chance_rollouts = int(chance_rollouts)
        # Diagnostics: how many terms of each family this hand actually took.
        self.n_action_terms = 0
        self.n_chance_terms = 0
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

        When per-street chance corrections are enabled, this **also** takes the
        chance term for the street ``action`` deals, if it deals one — the engine
        fuses the deal into the closing action, so ``h·action`` is where that event
        lives (:meth:`LeafValue.chance_values`).  The two families are independent
        terms; a ``legal`` mismatch skips only the action one.

        Coverage note: the play loop calls this at *known-policy* nodes while the
        hero is still active, so a round closed after the hero folds takes no chance
        term.  That costs nothing — with the hero folded its payout is already
        settled, so ``v`` is constant in the board and the term is exactly zero.
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
        return hero_delta - self._sum_terms - self._sum_chance - chance_term

    @property
    def sum_chance_terms(self) -> float:
        """Σ of this hand's per-street chance terms (0.0 when they are off).

        ``finalize(...) + sum_chance_terms`` is exactly the ``aivat_value`` the same
        hand would have produced with ``chance=False``, which is what makes the
        two variants comparable on identical played hands.
        """
        return self._sum_chance

    @property
    def sum_action_terms(self) -> float:
        """Σ of this hand's action-node terms."""
        return self._sum_terms

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
