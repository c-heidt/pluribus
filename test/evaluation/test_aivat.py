"""Tests for AIVAT — variance-reduced strength estimate (evaluation/aivat.py, §10.2).

Three levels, mirroring the doc's "budget the effort in the acceptance test":

- **Deterministic arithmetic** — the :class:`AivatAccumulator` term / finalize sign
  against a fake value function, so the control-variate algebra
  (``u(z) − Σ(v(sampled) − Σπ·v(a))`` and the all-in runout term) is pinned exactly.
- **Belief / no-leak** — :meth:`LeafValue._sample_joint` draws opponent holes only
  from the tracked belief, never the opponent's revealed cards (the #1 correctness
  trap).
- **Statistical acceptance** — a full stub run with AIVAT on gates the two
  properties §10.2 specifies: ``mean(aivat) ≈ mean(hero_chips_delta)`` (unbiased,
  paired CI) and ``var(aivat) ≪ var(hero_chips_delta)`` (the payoff, and the
  sign-error trap).  Plus the plumbing (column populated iff on) and the passive
  guarantee (AIVAT's RNG isolation leaves the played hand's chip delta unchanged).

All fast: small deck, heads-up, tiny solver budget — no ``slow`` / ``requires_lut``.
"""

import contextlib
import math
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.aivat import AivatAccumulator, LeafValue, _board_compatible
from evaluation.runner import run_evaluation
from evaluation.sqlite_logging import ExperimentLog
from poker_ai.search.agent import SearchAgent
from poker_ai.search.leaf import LeafConfig
from poker_ai.search.solver import SolverConfig
from test.evaluation.test_runner import _stub_session
from test.search._helpers import UniformPolicy, _flop_env, _policies


class _NonUniformPolicy(UniformPolicy):
    """A fixed non-uniform strategy: favours call/check, seldom folds/shoves.

    AIVAT is a control-variate method — it only *reduces* variance when the
    value function actually predicts play.  That requires a realistic
    (non-uniform) policy: under a uniform random policy the corrections are
    uncorrelated with outcomes, so AIVAT's variance reduction is unreliable
    (and the correct full-stack all-in dynamics tip borderline cases either
    way).  Used by the variance-reduction acceptance test.
    """

    def strategy(self, state, bias="none"):
        la = state.legal_actions
        n = len(la)
        if not n:
            return np.array([], np.float32)
        w = np.empty(n, dtype=np.float64)
        for i, a in enumerate(la):
            w[i] = (
                1.0 if a is None
                else 8.0 if a in ("call", "check")
                else 1.0 if a == "fold"
                else 0.5 if a == "all_in"
                else 2.0  # raise:<f>
            )
        w /= w.sum()
        return w.astype(np.float32)


# --------------------------------------------------------------------------- #
# Deterministic arithmetic (fake value function)
# --------------------------------------------------------------------------- #

class _FakeValue:
    """Returns a fixed hero-seat value per action, ignoring the env."""

    def __init__(self, table):
        self._table = table

    def child_values(self, env_before, legal):
        return {a: self._table[a] for a in legal}


class _FakeTerminal:
    # board_len defaults to 3 (a flop all-in → cheap runout, chance correction on).
    def __init__(self, payout, *, decision_free=False, runout=None, board_len=3):
        self._payout = payout
        self._df = decision_free
        self._runout = runout
        self._board_len = board_len

    @property
    def payout(self):
        return self._payout

    @property
    def is_decision_free(self):
        return self._df

    @property
    def terminal_board_len(self):
        return self._board_len

    def runout_equity(self, *, rng=None, cap=5000):
        return self._runout


class TestAccumulatorArithmetic:

    def test_action_term_and_finalize_sign(self):
        v = _FakeValue({"call": 10.0, "fold": -2.0})
        acc = AivatAccumulator(0, v, np.random.default_rng(0))
        # π = (0.25 call, 0.75 fold), sampled "call".
        acc.correct_action(None, 0, "call", ["call", "fold"], [0.25, 0.75])
        baseline = 0.25 * 10.0 + 0.75 * (-2.0)          # = 1.0
        term = 10.0 - baseline                           # = 9.0
        # No showdown → finalize is just u(z) − Σ terms.
        val = acc.finalize(_FakeTerminal({0: 30.0}))
        assert math.isclose(val, 30.0 - term)

    def test_multiple_terms_accumulate(self):
        v = _FakeValue({"a": 4.0, "b": 0.0})
        acc = AivatAccumulator(0, v, np.random.default_rng(0))
        acc.correct_action(None, 0, "a", ["a", "b"], [0.5, 0.5])   # term = 4 - 2 = 2
        acc.correct_action(None, 1, "b", ["a", "b"], [0.5, 0.5])   # term = 0 - 2 = -2
        val = acc.finalize(_FakeTerminal({0: 7.0}))
        assert math.isclose(val, 7.0 - (2.0 + (-2.0)))            # = 7.0

    def test_action_not_in_legal_is_skipped(self):
        v = _FakeValue({"a": 4.0, "b": 0.0})
        acc = AivatAccumulator(0, v, np.random.default_rng(0))
        acc.correct_action(None, 0, "raise:1.0", ["a", "b"], [0.5, 0.5])  # bogus action
        val = acc.finalize(_FakeTerminal({0: 5.0}))
        assert math.isclose(val, 5.0)                             # no term applied

    def test_allin_runout_chance_correction(self):
        # Cheap (flop, board_len=3 → 2 cards to come) decision-free terminal: aivat
        # collapses to runout_equity − Σ action terms.
        acc = AivatAccumulator(0, _FakeValue({}), np.random.default_rng(0))
        val = acc.finalize(
            _FakeTerminal({0: 100.0}, decision_free=True, runout={0: 60.0}, board_len=3)
        )
        # chance_term = 100 − 60 = 40 → aivat = 100 − 0 − 40 = 60 (the exact average).
        assert math.isclose(val, 60.0)

    def test_preflop_allin_skips_expensive_runout(self):
        # A pre-flop all-in (board_len=0 → 5 cards to come) must NOT call the
        # (5000-board) runout_equity; it keeps only the action-node corrections.
        class _Boom(_FakeTerminal):
            def runout_equity(self, *, rng=None, cap=5000):
                raise AssertionError("runout_equity must not run for a preflop all-in")

        acc = AivatAccumulator(0, _FakeValue({}), np.random.default_rng(0))
        val = acc.finalize(_Boom({0: 100.0}, decision_free=True, board_len=0))
        assert math.isclose(val, 100.0)         # no chance term applied


# --------------------------------------------------------------------------- #
# Belief / no-leak (LeafValue._sample_joint)
# --------------------------------------------------------------------------- #

def _hero_on_flop(seed=0, low=11, high=14, stacks=(300, 300)):
    """A heads-up flop env + a hero SearchAgent whose tracker is initialised."""
    env = _flop_env(low=low, high=high, stacks=stacks, seed=seed)
    leaf = LeafConfig(policies=_policies())
    cfg = SolverConfig(leaf=leaf, max_iterations=4, max_wall_seconds=30.0,
                       discount_interval=20)
    hero_seat = env.player_i
    hero = SearchAgent(_policies(), UniformPolicy(), cfg, np.random.default_rng(seed))
    hero.on_hand_start(env, hero_seat)
    return env, hero


class TestBeliefSampling:

    def test_sampler_draws_only_from_belief_support(self):
        # Force the opponent's belief onto a single board-compatible combo; every
        # sampled opponent hole must be that combo — proving the sampler reads the
        # belief, never the opponent's actual (revealed) cards (no information leak).
        env, hero = _hero_on_flop(seed=3)
        opp = next(s for s in hero.tracker.snapshot() if s != hero.my_seat)
        w = hero.tracker._ranges[opp]
        board_ok = _board_compatible(env)
        my = set(hero.my_hole)
        cc = env.combo_cards
        # A nonzero, board-compatible combo disjoint from the hero's known hole.
        idx = next(
            i for i in np.nonzero(w)[0]
            if board_ok[i] and not (my & {int(cc[i, 0]), int(cc[i, 1])})
        )
        w[:] = 0.0
        w[idx] = 1.0
        target = (int(cc[idx, 0]), int(cc[idx, 1]))

        vf = LeafValue(hero, hero._cfg.leaf, np.random.default_rng(0), n_hole_samples=1)
        for _ in range(40):
            holes = vf._sample_joint(env)
            assert holes[hero.my_seat] == tuple(sorted(hero.my_hole)) or \
                set(holes[hero.my_seat]) == set(hero.my_hole)   # hero = own hole
            assert holes[opp] == target                          # belief respected

    def test_child_values_are_finite_per_action(self):
        env, hero = _hero_on_flop(seed=5)
        legal = [a for a in env.legal_actions if a is not None]
        vf = LeafValue(hero, hero._cfg.leaf, np.random.default_rng(1), n_hole_samples=3)
        vals = vf.child_values(env, legal)
        assert set(vals) == set(legal)
        assert all(np.isfinite(v) for v in vals.values())

    def test_tracker_less_hero_samples_belief_free(self):
        # A blueprint-only / non-search hero has ``tracker is None``.  AIVAT must
        # still draw card-disjoint holes (hero = own hole, opponents uniform over
        # the available combos) and evaluate v end-to-end — the "make AIVAT ready
        # for a blueprint hero" contract.
        env = _flop_env(low=11, high=14, stacks=(300, 300), seed=7)
        hero_seat = env.player_i
        my_hole = tuple(sorted(int(c) for c in env.players[hero_seat].cards))
        hero = SimpleNamespace(my_seat=hero_seat, my_hole=my_hole, tracker=None)
        board = {int(c) for c in env.community_cards}
        leaf = LeafConfig(policies=_policies())

        vf = LeafValue(hero, leaf, np.random.default_rng(0), n_hole_samples=1)
        for _ in range(30):
            holes = vf._sample_joint(env)
            flat = [c for pair in holes for c in pair]
            assert len(set(flat)) == len(flat)               # card-disjoint
            assert not (set(flat) & board)                    # none on the board
            assert set(holes[hero_seat]) == set(my_hole)      # hero = own hole

        legal = [a for a in env.legal_actions if a is not None]
        vals = LeafValue(hero, leaf, np.random.default_rng(1),
                         n_hole_samples=2).child_values(env, legal)
        assert set(vals) == set(legal)
        assert all(np.isfinite(v) for v in vals.values())


# --------------------------------------------------------------------------- #
# Statistical acceptance + plumbing (full stub run)
# --------------------------------------------------------------------------- #

def _run(tmp_path, *, aivat, run_id="A", n=120, seed=13, hole_samples=4,
         starting_stack=400, blueprint=None):
    session = _stub_session(run_id=run_id, run_seed=seed, n_players=2,
                            starting_stack=starting_stack, blueprint=blueprint)
    session.config.aivat = aivat
    session.config.aivat_hole_samples = hole_samples
    log = ExperimentLog.open(tmp_path / f"{run_id}.sqlite")
    try:
        run_evaluation(log=log, session=session, max_hands=n)
        rows = log._con.execute(
            "SELECT hand_index, aivat_value, hero_chips_delta FROM games "
            "ORDER BY hand_index"
        ).fetchall()
    finally:
        log.close()
    return rows


class TestAcceptance:

    def test_unbiased(self, tmp_path):
        # Unbiasedness is the robust, regime-independent gate.  Use a short-stack
        # (all-in-heavy) run so the kept flop/turn runout corrections dominate — the
        # mean must still match the raw delta.
        rows = _run(tmp_path, aivat=True, n=160)
        aiv = np.array([r[1] for r in rows], dtype=float)
        dl = np.array([r[2] for r in rows], dtype=float)
        assert not np.isnan(aiv).any()                    # every hand populated
        # aivat = delta − Σterms, so d = aivat − delta has zero expectation.
        # |mean(d)| must sit inside a 4σ band (non-flaky; a mean-shifting bug — a
        # dropped/double-counted/sign-flipped term — breaks it).
        d = aiv - dl
        se = d.std(ddof=1) / math.sqrt(len(d))
        assert abs(d.mean()) <= 4.0 * se + 1e-9

    def test_reduces_variance(self, tmp_path):
        # The payoff.  The scoped AIVAT (action-node + flop/turn all-in corrections,
        # no per-street chance MIVAT) reduces variance once the value function is
        # low-noise enough to be a good control variate — so use deeper stacks (fewer
        # uncorrected pre-flop shoves) and more hole samples (a smoother v).  A
        # sign-flipped correction would *inflate* variance and fail this.
        # AIVAT reduces variance only against a realistic (non-uniform) policy —
        # a uniform policy leaves the corrections uncorrelated with outcomes.
        rows = _run(tmp_path, aivat=True, n=140, seed=13,
                    starting_stack=6000, hole_samples=12,
                    blueprint=_NonUniformPolicy())
        aiv = np.array([r[1] for r in rows], dtype=float)
        dl = np.array([r[2] for r in rows], dtype=float)
        assert aiv.var() < dl.var()

    def test_column_populated_iff_enabled(self, tmp_path):
        off = _run(tmp_path, aivat=False, run_id="OFF", n=6)
        on = _run(tmp_path, aivat=True, run_id="ON", n=6)
        assert all(r[1] is None for r in off)            # NULL when off
        assert all(r[1] is not None for r in on)         # populated when on

    def test_aivat_is_passive_on_the_played_hand(self, tmp_path):
        # RNG isolation, OUTWARD direction: turning AIVAT on must not perturb the
        # deal / hero / opponent sampling, so the raw hero_chips_delta is identical
        # with AIVAT on or off.
        off = _run(tmp_path, aivat=False, run_id="P0", n=20, seed=21)
        on = _run(tmp_path, aivat=True, run_id="P1", n=20, seed=21)
        assert [r[2] for r in off] == [r[2] for r in on]

    def test_reproducible(self, tmp_path):
        a = _run(tmp_path, aivat=True, run_id="R1", n=20, seed=8)
        b = _run(tmp_path, aivat=True, run_id="R2", n=20, seed=8)
        # Same seed → identical aivat_value and delta (deterministic AIVAT sampling).
        assert [(r[1], r[2]) for r in a] == [(r[1], r[2]) for r in b]


# --------------------------------------------------------------------------- #
# AIVAT runout-coverage knob: max_runout_cards
# --------------------------------------------------------------------------- #

def test_aivat_max_runout_cards_default_unchanged():
    from evaluation.aivat import _MAX_RUNOUT_CARDS
    acc = AivatAccumulator(0, object(), np.random.default_rng(0))
    assert acc._max_runout_cards == _MAX_RUNOUT_CARDS == 2   # played-game default

    class _T:
        terminal_board_len = 0                              # pre-flop all-in (5 to come)
    assert acc._cheap_runout(_T()) is False                 # skipped at default

    acc5 = AivatAccumulator(0, object(), np.random.default_rng(0), max_runout_cards=5)
    assert acc5._cheap_runout(_T()) is True                 # covered when raised


# --------------------------------------------------------------------------- #
# RNG isolation, INWARD direction — AIVAT must not inherit anyone else's stream
# --------------------------------------------------------------------------- #

def _global_state():
    """Hashable snapshot of the global MT19937 state (key *and* position)."""
    st = np.random.get_state()
    return (st[0], st[1].tobytes(), st[2], st[3], st[4])


@contextlib.contextmanager
def _corrected_nodes():
    """Record, per hand, the AIVAT-visible inputs at every corrected node.

    Each entry is ``(seat, action, legal, probs, live_belief, folded_belief)`` —
    **every** input the correction is a function of.  Equal chip deltas is not a
    substitute for this key (two different lines can net the same chips, so it
    dilutes the sample with coincidences), and neither is equal *play*: two arms
    can sample the same action from different strategies.

    All four non-action components are genuinely arm-dependent, and legitimately so:

    - ``probs`` is the played σ, so a different iteration budget gives a different
      control-variate baseline ``Σ π(a)·v(a)`` at the very same node;
    - the beliefs come from ``hero.tracker``, whose boundary update replays buffered
      actions under *the last search's* average policy
      (``SearchAgent._apply_boundary_belief_update``), so different budgets hold
      different posteriors.

    None of that is an RNG leak — it is the control variate honestly differing
    between arms.  What the RNG fix guarantees, and what these tests gate, is that
    once every one of these inputs agrees, ``aivat_value`` agrees *exactly*.

    A self-restoring context manager rather than a ``monkeypatch`` helper: two arms
    are recorded in one test, and a second ``monkeypatch.setattr`` would wrap the
    first arm's patch instead of the original — both recorders would then fire and
    the two arms' sequences would cross-contaminate.
    """
    from evaluation.aivat import AivatAccumulator
    hands, cur = [], []
    orig_correct = AivatAccumulator.correct_action
    orig_finalize = AivatAccumulator.finalize

    def _digest(snapshot):
        return tuple(sorted(
            (int(seat), np.asarray(w, dtype=np.float64).tobytes())
            for seat, w in snapshot.items()
        ))

    def correct(self, env_before, seat, action, legal, probs):
        tracker = getattr(self._v._hero, "tracker", None)
        cur.append((
            int(seat), action, tuple(legal),
            tuple(round(float(p), 12) for p in probs),
            _digest(tracker.snapshot()) if tracker is not None else None,
            _digest(tracker.folded_snapshot()) if tracker is not None else None,
        ))
        return orig_correct(self, env_before, seat, action, legal, probs)

    def finalize(self, terminal_env):
        hands.append(tuple(cur))
        cur.clear()
        return orig_finalize(self, terminal_env)

    AivatAccumulator.correct_action = correct
    AivatAccumulator.finalize = finalize
    try:
        yield hands
    finally:
        AivatAccumulator.correct_action = orig_correct
        AivatAccumulator.finalize = orig_finalize


class TestRngIsolation:

    def test_consumes_no_global_randomness(self, tmp_path):
        """AIVAT owns its streams outright — it must not read the global one at all.

        Stronger than passivity: snapshot-and-restore would also leave the global
        state equal, but would still have made AIVAT's own draws depend on whatever
        the hero's solver left behind.  This asserts the value function never
        touches the stream in the first place.
        """
        env, hero = _hero_on_flop(seed=5)
        legal = [a for a in env.legal_actions if a is not None]
        vf = LeafValue(hero, hero._cfg.leaf, np.random.default_rng(1),
                       n_hole_samples=3)
        before = _global_state()
        vf.child_values(env, legal)
        assert _global_state() == before

    def test_value_is_independent_of_global_stream_position(self):
        """An unrelated consumer of the global stream must not move v."""
        env, hero = _hero_on_flop(seed=5)
        legal = [a for a in env.legal_actions if a is not None]

        def once():
            return LeafValue(hero, hero._cfg.leaf, np.random.default_rng(1),
                             n_hole_samples=3).child_values(env, legal)

        np.random.seed(1)
        a = once()
        np.random.seed(2)
        np.random.random(53)                       # burn an arbitrary amount
        b = once()
        assert a == b

    def test_value_is_independent_of_solver_consumption(self):
        """**The regression test for the pairing defect.**

        The hero's solver burns an arm-dependent amount of randomness — a DBR solve,
        an OX solve and a vanilla solve differ, and a skipped solve burns none.  When
        the value function shared that stream, two arms scored an identically-played
        hand differently, so ``aivat_value`` could not cancel in the CRN-paired Δ and
        the estimator injected arm-specific noise into the headline.
        """
        env, hero = _hero_on_flop(seed=5)
        legal = [a for a in env.legal_actions if a is not None]

        def once():
            return LeafValue(hero, hero._cfg.leaf, np.random.default_rng(1),
                             n_hole_samples=3).child_values(env, legal)

        a = once()
        hero._solve_rng.random(4096)               # a much heavier solve
        np.random.random(77)                       # and the traffic it used to cause
        b = once()
        assert a == b

    def test_board_stream_is_separate_from_sampling_stream(self):
        _env, hero = _hero_on_flop(seed=5)
        vf = LeafValue(hero, hero._cfg.leaf, np.random.default_rng(1))
        assert vf._board_rng is not vf._rng
        assert [float(vf._rng.random()) for _ in range(8)] != \
               [float(vf._board_rng.random()) for _ in range(8)]

    def test_deriving_board_stream_leaves_sampling_stream_untouched(self):
        """Spawned, not drawn: adding the board split must not change the belief
        sampler's draws."""
        _env, hero = _hero_on_flop(seed=5)
        vf = LeafValue(hero, hero._cfg.leaf, np.random.default_rng(1))
        baseline = np.random.default_rng(1)
        assert [float(vf._rng.random()) for _ in range(8)] == \
               [float(baseline.random()) for _ in range(8)]


class TestCrnPairing:
    """AIVAT must STACK with CRN, not erode it (evaluation.md §10.1).

    The evaluation's headline is a paired Δ over arms sharing a deck seed.  Where
    two arms see an identical deal, the raw metric contributes exactly zero to that
    difference; ``aivat_value`` has to do the same, or the estimator injects
    arm-specific noise straight into the statistic it is meant to sharpen.

    "Identical" here means identical *AIVAT inputs*: the corrected-node sequence,
    the played σ at each node, and the belief ``v`` integrates over (see
    :func:`_corrected_nodes`).  Equal play alone is not enough — two arms can sample
    the same action from different strategies, and their trackers can hold different
    posteriors, both of which legitimately change the control variate without any
    RNG being involved.  Making the correction cancel on *those* deals too would
    need an arm-INDEPENDENT value function, which is a separate design decision and
    is what still caps the pairing benefit; it is not an RNG fix.
    """

    def _arm(self, tmp_path, run_id, iters, n=60, seed=31):
        import dataclasses
        session = _stub_session(run_id=run_id, run_seed=seed, n_players=2,
                                starting_stack=400,
                                blueprint=_NonUniformPolicy())
        session.solver_cfg = dataclasses.replace(session.solver_cfg,
                                                 max_iterations=iters)
        session.config.aivat = True
        session.config.aivat_hole_samples = 3
        log = ExperimentLog.open(tmp_path / f"{run_id}.sqlite")
        try:
            with _corrected_nodes() as nodes:
                run_evaluation(log=log, session=session, max_hands=n)
            rows = log._con.execute(
                "SELECT hand_index, aivat_value, hero_chips_delta FROM games "
                "ORDER BY hand_index").fetchall()
        finally:
            log.close()
        assert len(nodes) == len(rows), "recorder/rows misaligned"
        return {r[0]: (r[1], r[2], nodes[i]) for i, r in enumerate(rows)}

    def test_identical_inputs_give_identical_aivat_value(self, tmp_path):
        a = self._arm(tmp_path, "CRN_A", iters=4)
        b = self._arm(tmp_path, "CRN_B", iters=40)
        paired = [(a[k], b[k]) for k in sorted(set(a) & set(b))]
        same = [(x, y) for x, y in paired if x[2] == y[2]]
        assert len(same) >= 5, (
            f"only {len(same)} deals with identical AIVAT inputs — too few to "
            "gate on; raise n or lower the arms' divergence"
        )
        for x, y in same:
            assert x[1] == y[1]                    # raw delta agrees (sanity)
            assert x[0] == pytest.approx(y[0], abs=1e-9), (
                "aivat_value differs between arms despite identical inputs — the "
                "value function is inheriting arm-dependent randomness, which "
                "breaks the CRN pairing the paired Δ relies on"
            )

    def test_cancels_exactly_in_the_paired_difference(self, tmp_path):
        """The consequence: those deals contribute exactly 0 to the paired Δ,
        matching what the raw metric contributes."""
        a = self._arm(tmp_path, "CRN_C", iters=4)
        b = self._arm(tmp_path, "CRN_D", iters=40)
        deltas = [b[k][0] - a[k][0]
                  for k in sorted(set(a) & set(b)) if a[k][2] == b[k][2]]
        assert deltas
        assert max(abs(d) for d in deltas) < 1e-9
