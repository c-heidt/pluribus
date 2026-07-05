"""Tests for the depth-limited subgame solver — MCCFR path (§6.5, row 6.2).

The solver is exercised at three levels:

- **Exact units** — ``SolverState`` table ops (regret matching, widening, discount,
  averaging), the regime selector, the joint root sampler, ``SearchPolicy`` reads,
  and the freezing gate.  These are deterministic and assert exact behaviour.
- **Integration** — ``solve()`` on a heads-up **flop** subgame (the MCCFR regime's
  terminal-only setting: no depth-limit leaf, exact small-deck runouts) runs,
  produces valid strategies, is deterministic, and reuses warm-start state.
- **Convergence (fast)** — the average strategy stabilises over iterations.

The heads-up flop subgame (`street_at_root == 1`, two live seats) is chosen
deliberately: ``_select_regime`` routes it to MCCFR, and ``DepthLimit.classify``
makes it terminal-only, so the traversal walks flop→turn→river to real showdowns
without the continuation meta-game — isolating the core CFR loop.
"""

import collections
import copy
from typing import Dict

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv
from poker_ai.search.context import SubgameContext
from poker_ai.search.leaf import LeafConfig
from poker_ai.search.mccfr import _MCCFRSolver, _BIAS_CLASSES
from poker_ai.search.policy import Policy, SearchPolicy
from environment.range_showdown import reach_after_removal
from poker_ai.search.solver import solve, SolverConfig, SolverState, _select_regime
from poker_ai.search.solver_state import _hand_row
from poker_ai.search.vector import _VectorSolver, _regret_match_matrix


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class UniformPolicy(Policy):
    """Uniform over the legal actions (a self-contained leaf fleet stand-in)."""

    def strategy(self, state, bias="none"):
        n = len(state.legal_actions)
        return np.full(n, 1.0 / n, dtype=np.float32) if n else np.array([], np.float32)


def _policies():
    return {c: UniformPolicy() for c in _BIAS_CLASSES}


def _stub_lut(env: PokerEnv) -> None:
    env.card_info_lut = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0)
    )


def _flop_env(low=11, high=14, stacks=(200, 200), seed=0) -> PokerEnv:
    """Heads-up env advanced to the flop over a small deck (exact runouts)."""
    np.random.seed(seed)
    env = PokerEnv(
        players=[Player(i, s) for i, s in enumerate(stacks)],
        low_card_rank=low,
        high_card_rank=high,
    )
    _stub_lut(env)
    guard = 0
    while env.betting_round < 1 and not env.is_terminal and guard < 20:
        env.step_in_place("call" if "call" in env.legal_actions else "check")
        guard += 1
    return env


def _advance_to(env: PokerEnv, target_round: int) -> PokerEnv:
    """Walk a heads-up env (calls/checks only) to ``target_round``."""
    guard = 0
    while env.betting_round < target_round and not env.is_terminal and guard < 60:
        env.step_in_place("call" if "call" in env.legal_actions else "check")
        guard += 1
    return env


def _late_env(target_round, low=11, high=14, stacks=(200, 200), seed=0) -> PokerEnv:
    """Heads-up small-deck env advanced to ``target_round`` (2=turn, 3=river)."""
    np.random.seed(seed)
    env = PokerEnv(
        players=[Player(i, s) for i, s in enumerate(stacks)],
        low_card_rank=low,
        high_card_rank=high,
    )
    _stub_lut(env)
    return _advance_to(env, target_round)


def _ctx(env, *, n_rollouts=2, seed=0, ranges=None, folded=None) -> SubgameContext:
    if ranges is None:
        ranges = {s: np.ones(env.n_combos, np.float32) / env.n_combos for s in range(2)}
    leaf = LeafConfig(policies=_policies(), n_rollouts=n_rollouts)
    return SubgameContext.from_runtime(
        env=env,
        my_seat=0,
        my_hole=tuple(int(c) for c in env.players[0].cards),
        ranges=ranges,
        folded_ranges=folded or {},
        leaf=leaf,
        rng=np.random.default_rng(seed),
    )


def _cfg(env_ctx, *, iters=60, discount=20) -> SolverConfig:
    return SolverConfig(
        leaf=env_ctx.leaf, max_iterations=iters, max_wall_seconds=30.0,
        discount_interval=discount, workers=1,
    )


# --------------------------------------------------------------------------- #
# Regime selection
# --------------------------------------------------------------------------- #

class TestRegimeSelect:

    @pytest.mark.parametrize("street,n,expected", [
        (0, 2, "mccfr"),   # preflop HU
        (1, 2, "mccfr"),   # flop HU (round-2)
        (2, 2, "vector"),  # turn HU
        (3, 2, "vector"),  # river HU
        (2, 3, "mccfr"),   # turn 3-way (multiway → MCCFR)
        (1, 3, "mccfr"),   # flop 3-way
    ])
    def test_regime_rule(self, street, n, expected):
        ranges = {s: np.ones(4, np.float32) for s in range(n)}
        ctx = SubgameContext(
            my_seat=0, my_hole=(0, 1), ranges=ranges, folded_ranges={},
            board_compatible=np.ones(4, bool), street_at_root=street,
            depth_limit=None, leaf=None, rng=np.random.default_rng(0),
        )
        assert _select_regime(ctx) == expected

    def test_vector_regime_dispatches_and_runs(self):
        # Drive a HU env to the turn; a search must dispatch to the vector regime
        # and run (no NotImplementedError), populating the per-node matrices.
        env = _late_env(2, seed=1)
        assert env.betting_round == 2
        ctx = _ctx(env)
        assert _select_regime(ctx) == "vector"
        res = solve(env, ctx, _cfg(ctx, iters=5))
        assert res.iterations_run == 5
        assert len(res.state.vregret) > 0


# --------------------------------------------------------------------------- #
# SolverState — exact table ops
# --------------------------------------------------------------------------- #

class TestSolverState:

    def _node(self, legal=("fold", "call", "raise")):
        st = SolverState.empty()
        pk = ("flop", ())
        st.ensure_node(pk, legal, actor=0)
        return st, pk

    def test_unseen_sigma_is_uniform(self):
        st, pk = self._node()
        key = (pk, 7)
        np.testing.assert_allclose(st.sigma(key), [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_regret_matching(self):
        st, pk = self._node()
        key = (pk, 7)
        st.add_regret(key, np.array([0.0, 3.0, 1.0]))
        np.testing.assert_allclose(st.sigma(key), [0.0, 0.75, 0.25], atol=1e-6)

    def test_average_sigma_normalises(self):
        st, pk = self._node()
        key = (pk, 7)
        assert st.average_sigma(key) is None
        st.add_strat(key, np.array([1.0, 0.0, 0.0]))
        st.add_strat(key, np.array([0.0, 1.0, 1.0]))
        np.testing.assert_allclose(st.average_sigma(key), [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_discount_scales_both_tables(self):
        st, pk = self._node()
        key = (pk, 7)
        st.add_regret(key, np.array([2.0, 4.0, 6.0]))
        st.add_strat(key, np.array([1.0, 2.0, 3.0]))
        st.discount(0.5)
        np.testing.assert_allclose(st.regret[key], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(st.strat_sum[key], [0.5, 1.0, 1.5])

    def test_widening_preserves_columns(self):
        # A re-search injects a 4th action; old regrets keep their place, the new
        # column is zero-initialised.
        st, pk = self._node(("fold", "call", "raise"))
        key = (pk, 7)
        st.add_regret(key, np.array([1.0, 2.0, 3.0]))
        st.add_strat(key, np.array([4.0, 5.0, 6.0]))
        st.ensure_node(pk, ("fold", "call", "raise", "all_in"), actor=0)
        np.testing.assert_allclose(st.regret[key], [1.0, 2.0, 3.0, 0.0])
        np.testing.assert_allclose(st.strat_sum[key], [4.0, 5.0, 6.0, 0.0])
        assert st.width(pk) == 4

    def test_widening_grows_frozen_rows(self):
        st, pk = self._node(("fold", "call"))
        key = (pk, 7)
        st.frozen[key] = np.array([0.3, 0.7])
        st.ensure_node(pk, ("fold", "call", "all_in"), actor=0)
        np.testing.assert_allclose(st.frozen[key], [0.3, 0.7, 0.0])  # still sums to 1


# --------------------------------------------------------------------------- #
# Joint root sampler
# --------------------------------------------------------------------------- #

class TestJointSampler:

    def _solver(self, **ctx_kw):
        env = _flop_env(seed=3)
        ctx = _ctx(env, **ctx_kw)
        return _MCCFRSolver(env, SolverState.empty(), ctx, _cfg(ctx), ctx.rng), env

    def test_draws_are_card_disjoint_and_board_compatible(self):
        solver, env = self._solver()
        board = set(int(c) for c in env.community_cards)
        for _ in range(300):
            holes = solver._sample_root_holes()
            cards = [c for h in holes.values() for c in h]
            assert len(cards) == len(set(cards))          # pairwise disjoint
            assert not (set(cards) & board)                # board-compatible
            assert set(holes) == {0, 1}

    def test_marginals_track_the_range(self):
        # Seat 1 gets a peaked range over two combos; empirical draws should match.
        env = _flop_env(seed=4)
        w = np.zeros(env.n_combos, np.float32)
        # pick two board-compatible combos
        board = set(int(c) for c in env.community_cards)
        compat = [j for j in range(env.n_combos)
                  if not ({int(env.combo_cards[j, 0]), int(env.combo_cards[j, 1])} & board)]
        a, b = compat[0], compat[1]
        w[a], w[b] = 0.75, 0.25
        ranges = {0: np.ones(env.n_combos, np.float32) / env.n_combos, 1: w}
        ctx = _ctx(env, ranges=ranges)
        solver = _MCCFRSolver(env, SolverState.empty(), ctx, _cfg(ctx), ctx.rng)
        counts = collections.Counter()
        N = 4000
        for _ in range(N):
            counts[env.combo_index[tuple(sorted(solver._sample_root_holes()[1]))]] += 1
        # only the two non-zero combos appear; ratio ≈ 3:1
        assert set(counts) <= {a, b}
        assert abs(counts[a] / N - 0.75) < 0.05

    def test_zero_mass_combo_never_drawn(self):
        solver, env = self._solver()
        # seat 0's range is uniform; confirm a board-conflicting combo is never seen
        board_card = int(env.community_cards[0])
        for _ in range(300):
            for h in solver._sample_root_holes().values():
                assert board_card not in h

    def test_folded_seat_is_covered(self):
        # Three-handed: seat 2 folded pre-root; it must still be drawn (card removal).
        np.random.seed(5)
        env = PokerEnv(players=[Player(i, 200) for i in range(3)],
                       low_card_rank=11, high_card_rank=14)
        _stub_lut(env)
        ranges = {s: np.ones(env.n_combos, np.float32) / env.n_combos for s in (0, 1)}
        folded = {2: np.ones(env.n_combos, np.float32) / env.n_combos}
        leaf = LeafConfig(policies=_policies(), n_rollouts=1)
        ctx = SubgameContext.from_runtime(
            env=env, my_seat=0, my_hole=tuple(int(c) for c in env.players[0].cards),
            ranges=ranges, folded_ranges=folded, leaf=leaf, rng=np.random.default_rng(0),
        )
        solver = _MCCFRSolver(env, SolverState.empty(), ctx, _cfg(ctx), ctx.rng)
        holes = solver._sample_root_holes()
        assert set(holes) == {0, 1, 2}
        cards = [c for h in holes.values() for c in h]
        assert len(cards) == len(set(cards))


# --------------------------------------------------------------------------- #
# Freezing gate
# --------------------------------------------------------------------------- #

class TestFreezing:

    def _solver(self):
        env = _flop_env(seed=6)
        ctx = _ctx(env)
        return _MCCFRSolver(env, SolverState.empty(), ctx, _cfg(ctx), ctx.rng), env, ctx

    def test_frozen_only_when_sampled_hand_is_actual(self):
        solver, env, ctx = self._solver()
        my = ctx.my_hole
        other = next(
            tuple(env.combo_cards[j]) for j in range(env.n_combos)
            if tuple(sorted(int(c) for c in env.combo_cards[j])) != tuple(sorted(my))
        )
        key = (("flop", ()), 0)
        solver.state.frozen[key] = np.array([1.0, 0.0])
        # bot seat, actual hand → frozen
        assert solver._is_frozen(key, ctx.my_seat, {ctx.my_seat: my})
        # bot seat, different hand in (possibly) same cluster → NOT frozen
        assert not solver._is_frozen(key, ctx.my_seat, {ctx.my_seat: tuple(int(c) for c in other)})
        # non-bot seat → never frozen
        assert not solver._is_frozen(key, 1, {1: my})

    def test_frozen_sigma_is_returned_verbatim(self):
        solver, env, ctx = self._solver()
        key = (("flop", ()), 0)
        solver.state.legal_at[("flop", ())] = ("fold", "call")
        pinned = np.array([0.9, 0.1])
        solver.state.frozen[key] = pinned
        out = solver._frozen_or(np.array([0.5, 0.5]), key, ctx.my_seat, {ctx.my_seat: ctx.my_hole})
        np.testing.assert_array_equal(out, pinned)


# --------------------------------------------------------------------------- #
# Regret / strategy pass separation (the blueprint's two-pass structure)
# --------------------------------------------------------------------------- #

class TestPassSeparation:
    """The average strategy must be accumulated ONLY by the separate sampled
    strategy pass (`_update_strategy`), where the traverser's own actions are
    *sampled* (π_i-reach weighted) — never inside the regret pass, where the
    traverser *explores* all actions (which would mis-weight by opponent reach).
    Mirrors blueprint `cfr.py` (regret only) vs `strategy.py` (`update_strategy`)."""

    def _solver(self, seed=5):
        env = _flop_env(seed=seed)
        ctx = _ctx(env, seed=1)
        solver = _MCCFRSolver(env, SolverState.empty(), ctx, _cfg(ctx), ctx.rng)
        holes = solver._sample_root_holes()
        return solver, env, holes

    def test_regret_pass_does_not_touch_strategy(self):
        solver, env, holes = self._solver()
        hl = [holes[s] for s in range(2)]
        # Traverse as the seat that actually acts at the flop root — heads-up
        # that is the big blind (seat 1), who leads post-flop — so the traverser
        # is guaranteed a decision node in the sampled line.
        root_actor = env.player_i
        solver._traverse(env.with_hole_cards(hl), root_actor, holes)
        assert solver.state.regret, "regret pass should populate regret"
        assert not solver.state.strat_sum, "regret pass must NOT populate strat_sum"

    def test_strategy_pass_accumulates_strategy(self):
        solver, env, holes = self._solver()
        hl = [holes[s] for s in range(2)]
        root_actor = env.player_i
        solver._update_strategy(env.with_hole_cards(hl), root_actor, holes)
        assert solver.state.strat_sum, "strategy pass should populate strat_sum"
        # only the traverser's own rows are accumulated in its own pass
        for (pk, _hr) in solver.state.strat_sum:
            assert solver.state.actor_at[pk] == root_actor


# --------------------------------------------------------------------------- #
# solve() integration
# --------------------------------------------------------------------------- #

class TestSolveIntegration:

    def test_runs_and_builds_valid_strategies(self):
        env = _flop_env(seed=7)
        ctx = _ctx(env)
        res = solve(env, ctx, _cfg(ctx, iters=40))
        assert res.iterations_run == 40
        assert len(res.state.legal_at) > 0
        # every play / average distribution at a registered node is valid
        for pk, legal in res.state.legal_at.items():
            for use_avg in (False, True):
                p = (res.average_policy if use_avg else res.policy).strategy_for(pk, 0, legal)
                assert p.shape == (len(legal),)
                assert abs(p.sum() - 1.0) < 1e-5
                assert (p >= -1e-9).all()

    def test_deterministic_under_fixed_seed(self):
        # Determinism needs BOTH seeds pinned: action sampling uses ctx.rng, but
        # the board runout is dealt with the engine's global np.random (§6.4.1,
        # RNG unification deferred).  Same ctx seed + same global seed → identical
        # tables.
        def run():
            env = _flop_env(seed=8)
            ctx = _ctx(env, seed=11)
            np.random.seed(99)
            return solve(env, ctx, _cfg(ctx, iters=30))

        r1, r2 = run(), run()
        assert set(r1.state.regret) == set(r2.state.regret)
        for k in r1.state.regret:
            np.testing.assert_allclose(r1.state.regret[k], r2.state.regret[k])

    def test_warm_start_reuses_state(self):
        env = _flop_env(seed=9)
        ctx = _ctx(env, seed=2)
        first = solve(env, ctx, _cfg(ctx, iters=20))
        keys_before = set(first.state.regret)
        # re-search the same root with the carried state
        env2 = _flop_env(seed=9)
        ctx2 = _ctx(env2, seed=3)
        second = solve(env2, ctx2, _cfg(ctx2, iters=20), warm_start=first.state)
        assert second.state is first.state                 # reused in place
        assert keys_before <= set(second.state.regret)     # rows preserved/extended

    def test_warm_start_resets_per_search_counters(self):
        # Regression: the walk/cache tallies live on the reused SolverState, so a
        # warm re-search must report only ITS work — like iterations_run, which is
        # counted fresh per solve().  Without the reset every re-searched `decisions`
        # row would inflate node_count/cache stats cumulatively over the hand.
        env = _flop_env(seed=9)
        ctx = _ctx(env, seed=2)
        first = solve(env, ctx, _cfg(ctx, iters=20))
        n1 = first.stats.node_count
        assert n1 > 0 and first.stats.legal_at_misses > 0
        env2 = _flop_env(seed=9)
        ctx2 = _ctx(env2, seed=3)
        second = solve(env2, ctx2, _cfg(ctx2, iters=20), warm_start=first.state)
        # Same tree + same iters → per-search node count, NOT ~2x (cumulative).
        assert second.stats.node_count < 1.5 * n1
        # Most nodes already registered by the first solve → far fewer "new key"
        # misses this re-search than the fresh solve had (cumulative would be ≥).
        assert second.stats.legal_at_misses < first.stats.legal_at_misses

    def test_runout_terminal_flag_changes_values(self):
        # Flag on (exact equity) vs off (sampled payout) should generally differ on
        # a subgame that reaches all-in runouts; both must produce valid strategies.
        env = _flop_env(seed=10)
        ctx_on = _ctx(env, seed=1)
        res = solve(env, ctx_on, _cfg(ctx_on, iters=20))
        assert res.iterations_run == 20


# --------------------------------------------------------------------------- #
# SearchPolicy — exact reads
# --------------------------------------------------------------------------- #

class TestSearchPolicy:

    def _state(self, legal=("fold", "call", "raise")):
        st = SolverState.empty()
        pk = ("flop", ())
        st.ensure_node(pk, legal, actor=0)
        return st, pk

    def test_unseen_node_is_uniform(self):
        st = SolverState.empty()
        sp = SearchPolicy(st, use_average=False)
        out = sp.strategy_for(("nope", ()), 0, ("fold", "call"))
        np.testing.assert_allclose(out, [0.5, 0.5])

    def test_play_reads_regret_matched(self):
        st, pk = self._state()
        st.add_regret((pk, 5), np.array([0.0, 3.0, 1.0]))
        sp = SearchPolicy(st, use_average=False)
        np.testing.assert_allclose(
            sp.strategy_for(pk, 5, ("fold", "call", "raise")), [0.0, 0.75, 0.25], atol=1e-6
        )

    def test_average_reads_strat_sum(self):
        st, pk = self._state()
        st.add_strat((pk, 5), np.array([2.0, 1.0, 1.0]))
        sp = SearchPolicy(st, use_average=True)
        np.testing.assert_allclose(
            sp.strategy_for(pk, 5, ("fold", "call", "raise")), [0.5, 0.25, 0.25], atol=1e-6
        )

    def test_alignment_remaps_to_caller_order(self):
        st, pk = self._state(("fold", "call", "raise"))
        st.add_regret((pk, 5), np.array([0.0, 3.0, 1.0]))
        sp = SearchPolicy(st, use_average=False)
        # caller lists actions in a different order
        out = sp.strategy_for(pk, 5, ("raise", "fold", "call"))
        np.testing.assert_allclose(out, [0.25, 0.0, 0.75], atol=1e-6)

    def test_overlay_action_absent_from_row_gets_zero_mass(self):
        st, pk = self._state(("fold", "call", "raise"))
        st.add_regret((pk, 5), np.array([0.0, 3.0, 1.0]))
        sp = SearchPolicy(st, use_average=False)
        # an injected off-tree action the row has no column for → 0, renormalised
        out = sp.strategy_for(pk, 5, ("fold", "call", "raise", "raise:0.5"))
        np.testing.assert_allclose(out, [0.0, 0.75, 0.25, 0.0], atol=1e-6)
        assert abs(out.sum() - 1.0) < 1e-6

    def test_play_uses_frozen_row(self):
        st, pk = self._state(("fold", "call", "raise"))
        st.frozen[(pk, 5)] = np.array([0.2, 0.5, 0.3])
        play = SearchPolicy(st, use_average=False).strategy_for(pk, 5, ("fold", "call", "raise"))
        np.testing.assert_allclose(play, [0.2, 0.5, 0.3], atol=1e-6)
        # the average policy ignores frozen (belief uses accumulated strategy)
        avg = SearchPolicy(st, use_average=True).strategy_for(pk, 5, ("fold", "call", "raise"))
        np.testing.assert_allclose(avg, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)  # no strat_sum → uniform


# --------------------------------------------------------------------------- #
# Convergence
# --------------------------------------------------------------------------- #

class TestConvergence:

    @staticmethod
    def _root_range_average(state, root_pk: tuple, width: int) -> np.ndarray:
        """Range-aggregated average strategy at the root node.

        Sums ``strat_sum`` over every hand row at the root public node and
        normalises.  Unlike a single hand row — which is only updated on the rare
        traversals that sample exactly that hole — this aggregate accumulates on
        **every** traversal where the root actor traverses, so it is the
        low-variance quantity that actually converges under external sampling.
        """
        agg = np.zeros(width, dtype=np.float64)
        for (pk, _hr), row in state.strat_sum.items():
            if pk == root_pk:
                agg += row
        total = agg.sum()
        return agg / total if total > 0 else np.full(width, 1.0 / width)

    def test_average_strategy_stabilises(self, _seeded):
        # A converging solver's *average* strategy settles: the root range-average
        # should drift only a little as iterations grow.  We solve the same
        # heads-up flop subgame twice from identical seeds — N and 2N iterations —
        # and bound the change.  This is a genuine convergence property; it does
        # not pin the exact equilibrium (a brute-force tabular-CFR cross-check
        # would — a candidate strengthening).
        #
        # All seeds are pinned per trial (the board runout is dealt with the global
        # np.random, §6.4.1), so the 2N run replays the N run's trajectory and then
        # extends it — the difference is purely the averaging tail, and the result
        # is deterministic.  The trial seed varies the scenario across the 5 trials.
        trial = _seeded

        def root_range_average(iters):
            env = _flop_env(seed=12 + trial)
            ctx = _ctx(env, seed=4 + trial)
            pk = env.public_key
            width = len([a for a in env.legal_actions if a is not None])
            np.random.seed(123 + trial)  # pin the board-deal RNG too
            res = solve(env, ctx, _cfg(ctx, iters=iters))
            return self._root_range_average(res.state, pk, width)

        a_n = root_range_average(200)
        a_2n = root_range_average(400)
        # deterministic; observed worst-case drift across the 5 trials is ~0.16.
        assert np.abs(a_n - a_2n).max() < 0.25


# --------------------------------------------------------------------------- #
# Vector regime (heads-up turn/river)
# --------------------------------------------------------------------------- #

class TestVectorRegime:
    """The vector-form Linear CFR path (§6.5): per-combo matrices, chance-sampled
    river, showdown/fold terminals, all read back through the shared state."""

    # -- units -------------------------------------------------------------- #

    def test_regret_match_matrix(self):
        regret = np.array([[0.0, 3.0, 1.0], [-1.0, -2.0, -3.0], [0.0, 0.0, 0.0]])
        sigma = _regret_match_matrix(regret)
        np.testing.assert_allclose(sigma[0], [0.0, 0.75, 0.25], atol=1e-6)
        np.testing.assert_allclose(sigma[1], [1 / 3, 1 / 3, 1 / 3], atol=1e-6)  # no +regret
        np.testing.assert_allclose(sigma[2], [1 / 3, 1 / 3, 1 / 3], atol=1e-6)  # fresh

    def test_reach_after_removal_matches_bruteforce(self):
        env = _late_env(2, seed=1)
        cc = env.combo_cards
        opp = np.random.default_rng(0).random(cc.shape[0])
        got = reach_after_removal(cc, opp)
        sets = [set(row) for row in cc.tolist()]
        exp = np.array([
            sum(opp[j] for j in range(len(sets)) if not (sets[i] & sets[j]))
            for i in range(len(sets))
        ])
        # exercises the inclusion-exclusion add-back (j==i and one-card overlaps drop out)
        np.testing.assert_allclose(got, exp, rtol=1e-9, atol=1e-9)

    # -- SolverState matrix storage ---------------------------------------- #

    def test_ensure_vnode_allocates_and_discount_scales(self):
        st = SolverState.empty()
        pk = ("turn", ())
        st.ensure_vnode(pk, ("fold", "call"), actor=0, n_combos=6)
        assert st.vregret[pk].shape == (6, 2) and st.vstrat[pk].shape == (6, 2)
        st.vregret[pk][:] = 2.0
        st.vstrat[pk][:] = 4.0
        st.discount(0.5)
        assert np.allclose(st.vregret[pk], 1.0) and np.allclose(st.vstrat[pk], 2.0)

    def test_vnode_widening_grows_columns_preserving_values(self):
        st = SolverState.empty()
        pk = ("turn", ())
        st.ensure_vnode(pk, ("fold", "call"), actor=0, n_combos=3)
        st.vregret[pk][:] = [[1.0, 2.0]]
        st.ensure_vnode(pk, ("fold", "call", "raise:1.0"), actor=0, n_combos=3)
        assert st.vregret[pk].shape == (3, 3)
        np.testing.assert_allclose(st.vregret[pk][:, :2], [[1.0, 2.0]] * 3)
        np.testing.assert_allclose(st.vregret[pk][:, 2], [0.0, 0.0, 0.0])

    def test_state_reads_matrix_rows(self):
        st = SolverState.empty()
        pk = ("turn", ())
        st.ensure_vnode(pk, ("fold", "call", "raise"), actor=0, n_combos=4)
        st.vregret[pk][2] = [0.0, 3.0, 1.0]
        np.testing.assert_allclose(st.sigma((pk, 2)), [0.0, 0.75, 0.25], atol=1e-6)
        st.vstrat[pk][2] = [2.0, 1.0, 1.0]
        np.testing.assert_allclose(st.average_sigma((pk, 2)), [0.5, 0.25, 0.25], atol=1e-6)
        assert st.average_sigma((pk, 0)) is None  # unaccumulated row

    # -- terminal / value correctness -------------------------------------- #

    def test_root_value_is_zero_sum(self):
        # Under uniform play (fresh state) the reach-weighted root value to each
        # seat sums to zero — a property that holds only if the showdown/fold
        # stake is the matched contribution and card removal is consistent.
        env = _late_env(3, seed=5)  # river: deterministic, no chance node
        ctx = _ctx(env)
        cfg = _cfg(ctx, iters=1)
        s0, s1 = sorted(ctx.ranges)
        bc = np.asarray(ctx.board_compatible, dtype=np.float64)
        r0 = np.asarray(ctx.ranges[s0], np.float64) * bc
        r1 = np.asarray(ctx.ranges[s1], np.float64) * bc
        v0 = _VectorSolver(env, SolverState.empty(), ctx, cfg, ctx.rng)._walk(env, s0, r0, r1, None)
        v1 = _VectorSolver(env, SolverState.empty(), ctx, cfg, ctx.rng)._walk(env, s1, r1, r0, None)
        big = abs(float(r0 @ v0)) + abs(float(r1 @ v1)) + 1.0
        assert abs(float(r0 @ v0) + float(r1 @ v1)) < 1e-6 * big

    # -- regime behaviour --------------------------------------------------- #

    def test_runs_on_turn_and_river_with_valid_strategies(self):
        for target in (2, 3):
            env = _late_env(target, seed=6)
            ctx = _ctx(env)
            assert _select_regime(ctx) == "vector"
            res = solve(env, ctx, _cfg(ctx, iters=25))
            assert len(res.state.vregret) > 0
            for pk, legal in res.state.legal_at.items():
                mat = res.state.vregret.get(pk)
                if mat is not None and mat.ndim == 3:
                    # River-conditioned turn-subgame node (§6.5): a per-river
                    # betting strategy, internal to the solve (not externally
                    # read — the river is played from a fresh river subgame).
                    # Validate the per-(combo, river) regret-matched rows directly.
                    sig = _regret_match_matrix(mat)
                    assert sig.shape == (mat.shape[0], mat.shape[1], len(legal))
                    np.testing.assert_allclose(sig.sum(axis=-1), 1.0, atol=1e-5)
                    continue
                for pol in (res.policy, res.average_policy):
                    d = pol.strategy_for(pk, 0, legal)
                    assert d.shape == (len(legal),)
                    assert abs(d.sum() - 1.0) < 1e-5 and (d >= -1e-9).all()

    def test_determinism_independent_of_global_rng(self):
        # The river is sampled from ctx.rng, so the result must not depend on the
        # engine's global np.random state (unlike the MCCFR path).
        def run(global_seed):
            env = _late_env(2, seed=2)
            ctx = _ctx(env, seed=7)
            np.random.seed(global_seed)  # perturb global RNG after the subgame is fixed
            return solve(env, ctx, _cfg(ctx, iters=30))

        a, b = run(111), run(999)
        assert set(a.state.vregret) == set(b.state.vregret)
        for k in a.state.vregret:
            np.testing.assert_allclose(a.state.vregret[k], b.state.vregret[k])

    def test_no_reranking_in_iteration_loop(self, monkeypatch):
        # §6.4.2 pt 3: ranking is reach-independent, so the env memoises each board
        # and never re-ranks it across iterations.  Over many iterations the
        # underlying rank_combos_on_board is called at most once per distinct
        # candidate river — not once per node per iteration.
        import environment.range_showdown as rs
        rs._ranked_cached.cache_clear()
        env = _late_env(2, seed=3)
        ctx = _ctx(env)
        calls = {"n": 0}
        real = rs.rank_combos_on_board

        def spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(rs, "rank_combos_on_board", spy)
        solver = _VectorSolver(env, SolverState.empty(), ctx, _cfg(ctx, iters=1), ctx.rng)
        n_candidate_rivers = len(solver._rivers)
        for _ in range(40):
            solver.iterate()
        # bounded by distinct boards, independent of the 40 iterations / node count
        assert 0 < calls["n"] <= n_candidate_rivers

    def test_search_policy_reads_vector_result_and_frozen(self):
        env = _late_env(2, seed=4)
        ctx = _ctx(env)
        res = solve(env, ctx, _cfg(ctx, iters=20))
        pk = env.public_key
        legal = tuple(a for a in env.legal_actions if a is not None)
        ci = env.combo_index[tuple(sorted(int(c) for c in env.players[0].cards))]
        play = res.policy.strategy_for(pk, ci, legal)
        avg = res.average_policy.strategy_for(pk, ci, legal)
        for d in (play, avg):
            assert d.shape == (len(legal),)
            assert abs(d.sum() - 1.0) < 1e-5 and (d >= -1e-9).all()
        # a pinned actual-hand row is returned verbatim for play
        frozen = np.zeros(len(legal), np.float32)
        frozen[0] = 1.0
        res.state.frozen[(pk, ci)] = frozen
        np.testing.assert_allclose(
            res.policy.strategy_for(pk, ci, legal), frozen, atol=1e-6
        )

    def test_turn_subgame_conditions_river_betting(self):
        # The river-conditioned change (§6.5): a turn subgame builds 3-D river-stage
        # nodes whose average river-betting strategy genuinely **differs across the
        # river card** — the property a river-blind strategy could not represent.
        # The river is chance-*sampled* (one per iteration), but each river keeps its
        # own conditioned slice, so over iterations the per-river strategies diverge.
        env = _late_env(2, seed=6)
        ctx = _ctx(env)
        res = solve(env, ctx, _cfg(ctx, iters=250))
        river_nodes = [m for m in res.state.vstrat.values() if m.ndim == 3]
        assert river_nodes, "expected river-conditioned 3-D river-betting nodes"
        spread = 0.0
        for m in river_nodes:
            tot = m.sum(axis=-1, keepdims=True)
            avg = np.where(tot > 0, m / np.where(tot > 0, tot, 1.0), 0.0)
            mass = m.sum(axis=(1, 2)) > 0
            if mass.any():
                a0 = avg[mass, :, 0]  # action-0 prob per (combo, river)
                spread = max(spread, float((a0.max(1) - a0.min(1)).max()))
        assert spread > 0.0, "river betting did not condition on the river card"

    def test_turn_sampling_is_reproducible_given_seed(self):
        # The river chance node is sampled from ``ctx.rng`` (not the engine's
        # global RNG), so two solves with the same seed draw the same river
        # sequence and are bit-identical — reproducible despite the sampling.
        def run():
            env = _late_env(2, seed=2)
            ctx = _ctx(env, seed=7)
            return solve(env, ctx, _cfg(ctx, iters=50))

        a, b = run(), run()
        assert set(a.state.vregret) == set(b.state.vregret)
        for k in a.state.vregret:
            np.testing.assert_array_equal(a.state.vregret[k], b.state.vregret[k])

    def test_infeasible_combo_river_rows_are_masked(self):
        # A (combo, river) pair whose combo holds the river card is an impossible
        # deal: feasibility masking at the chance node must leave it zero strat-sum
        # (for a sampled river it is masked; an unsampled river stays all-zero).
        env = _late_env(2, seed=6)
        ctx = _ctx(env)
        res = solve(env, ctx, _cfg(ctx, iters=60))
        board = {int(c) for c in env.community_cards}
        rivers = sorted({int(x) for x in np.unique(env.combo_cards)} - board)
        cc = env.combo_cards
        checked = False
        for m in (m for m in res.state.vstrat.values() if m.ndim == 3):
            for k, r in enumerate(rivers):
                conflict = (cc[:, 0] == r) | (cc[:, 1] == r)
                if conflict.any():
                    assert np.all(m[conflict, k, :] == 0.0)
                    checked = True
        assert checked, "expected at least one infeasible (combo, river) row to check"

    def test_unequal_allin_stake_is_matched_not_max(self):
        # Heads-up unequal stacks: a short stack calls all-in for less against a
        # standing partial bet.  The env exposes the *final* contributions
        # (terminal_contributions) so the matched stake is unambiguously their
        # min (the bigger stack's excess is uncalled) — no parent reconstruction,
        # and a naive max() over the standing bet would overstate it.
        np.random.seed(1)  # a decisive call-all-in-for-less showdown
        start = [200, 2000]
        env = PokerEnv(players=[Player(0, start[0]), Player(1, start[1])],
                       low_card_rank=11, high_card_rank=14)
        _stub_lut(env)
        _advance_to(env, 3)
        standing = None
        while not env.is_terminal:
            legal = [a for a in env.legal_actions if a]
            actor = env.player_i
            pc = env.pot.capture()
            if actor == 1 and any(a.startswith("raise") for a in legal):
                env.step_in_place([a for a in legal if a.startswith("raise")][-1])
            elif actor == 0 and "all_in" in legal and pc[0] != pc[1]:
                standing = list(pc)  # pre-close standing bet (unequal contributions)
                env.step_in_place("all_in")  # short stack calls all-in for less
            else:
                env.step_in_place("call" if "call" in legal else legal[0])
        assert standing is not None and standing[0] != standing[1]
        assert env.is_terminal and not env.is_decision_free
        assert len([s for s in (0, 1) if env.players[s].is_active]) == 2  # showdown

        # The env's final contributions give the matched stake directly.
        tc = env.terminal_contributions
        matched = min(tc[0], tc[1])
        assert matched < max(standing)              # not the standing-bet level
        # winner's net == matched (per-player baseline; payout's single baseline is
        # invalid for unequal stacks)
        net = [env.players[s].n_chips - start[s] for s in (0, 1)]
        assert max(abs(net[0]), abs(net[1])) == matched

        # vector_payout uses that stake: one-hot opponent → ±matched per matchup.
        cc = env.combo_cards
        board = set(int(c) for c in env.community_cards)
        valid = [k for k in range(cc.shape[0])
                 if int(cc[k, 0]) not in board and int(cc[k, 1]) not in board]
        i, j = valid[0], next(k for k in valid
                              if not (set(cc[k].tolist()) & set(cc[valid[0]].tolist())))
        opp = np.zeros(env.n_combos)
        opp[j] = 1.0
        v = env.vector_payout(0, 1, opp, river=None)[i]
        assert abs(abs(v) - matched) < 1e-9

    def test_freezing_pins_actual_hand_row(self):
        # A pre-seeded frozen row for the bot's actual hand is never updated during
        # the search (its regret stays zero while other rows move), and the search
        # plays it back verbatim.  my_seat is the root actor so the root is a bot node.
        env = _late_env(3, seed=8)  # river subgame, deterministic
        actor = env.player_i
        my_hole = tuple(int(c) for c in env.players[actor].cards)
        ranges = {s: np.ones(env.n_combos, np.float32) / env.n_combos for s in range(2)}
        leaf = LeafConfig(policies=_policies(), n_rollouts=1)
        ctx = SubgameContext.from_runtime(
            env=env, my_seat=actor, my_hole=my_hole, ranges=ranges,
            folded_ranges={}, leaf=leaf, rng=np.random.default_rng(0),
        )
        pk = env.public_key
        legal = tuple(a for a in env.legal_actions if a is not None)
        ci = env.combo_index[tuple(sorted(my_hole))]
        pinned = np.zeros(len(legal), np.float64)
        pinned[0] = 1.0
        state = SolverState.empty()
        state.frozen[(pk, ci)] = pinned
        res = solve(env, ctx, _cfg(ctx, iters=30), warm_start=state)
        # the frozen actual-hand row accrued no regret...
        np.testing.assert_allclose(res.state.vregret[pk][ci], 0.0)
        # ...while other rows at the same node did move
        assert np.abs(res.state.vregret[pk]).sum() > 0.0
        np.testing.assert_allclose(res.policy.strategy_for(pk, ci, legal), pinned, atol=1e-6)

    def test_average_strategy_stabilises(self, _seeded):
        # A heads-up *river* subgame is fully deterministic (board complete, no
        # chance), so the vector regime's average strategy settles tightly.
        trial = _seeded

        def root_average(iters):
            env = _late_env(3, seed=20 + trial)
            ctx = _ctx(env, seed=4 + trial)
            pk = env.public_key
            width = len([a for a in env.legal_actions if a is not None])
            res = solve(env, ctx, _cfg(ctx, iters=iters))
            mat = res.state.vstrat.get(pk)
            if mat is None:
                return np.full(width, 1.0 / width)
            agg = mat.sum(axis=0)
            return agg / agg.sum() if agg.sum() > 0 else np.full(width, 1.0 / width)

        a_n = root_average(150)
        a_2n = root_average(300)
        assert np.abs(a_n - a_2n).max() < 0.25


# --------------------------------------------------------------------------- #
# Search-lifetime caches (§6.4.2, §6.7 Tier 1)
# --------------------------------------------------------------------------- #

class TestSearchLifetimeCaches:
    """The leaf-value and runout caches on ``SolverState``: a continuation value
    is computed once per ``(leaf public_key, holes, profile)`` and a decision-free
    runout integrated once per ``(holes, snapshot)`` — persisting across a
    warm-started re-search, and never breaking same-seed determinism."""

    def _preflop_env(self, low=11, high=14, stacks=(200, 200), seed=0) -> PokerEnv:
        # street_at_root == 0 → the depth limit makes the flop a continuation
        # meta-game leaf, so ``continuation_value`` is exercised.
        np.random.seed(seed)
        env = PokerEnv(
            players=[Player(i, s) for i, s in enumerate(stacks)],
            low_card_rank=low, high_card_rank=high,
        )
        _stub_lut(env)
        return env

    def _count_continuation(self, monkeypatch):
        """Patch the solver's ``continuation_value`` with a counting wrapper that
        records the same key ``_leaf_value`` memoises on; delegates to the real fn."""
        import poker_ai.search.mccfr as mod
        keys = []
        original = mod.continuation_value

        def wrapper(env, profile, ctx, runout_cache=None):
            n = env.n_players
            hk = tuple(tuple(int(c) for c in env.players[s].cards) for s in range(n))
            keys.append((env.public_key, hk, tuple(sorted(profile.items()))))
            return original(env, profile, ctx, runout_cache=runout_cache)

        monkeypatch.setattr(mod, "continuation_value", wrapper)
        return keys

    def test_leaf_value_computed_once_per_key(self, monkeypatch):
        keys = self._count_continuation(monkeypatch)
        env = self._preflop_env(seed=3)
        ctx = _ctx(env, seed=1)
        res = solve(env, ctx, _cfg(ctx, iters=40))
        assert keys, "preflop solve never reached a depth-limit leaf"
        # Each distinct (leaf pk, holes, profile) hits the rollouts exactly once;
        # every later visit is a cache hit (no duplicate keys recorded).
        assert len(keys) == len(set(keys))
        assert len(res.state.leaf_value_cache) == len(set(keys))

    def test_warm_start_reuses_cached_leaf_values(self, monkeypatch):
        env = self._preflop_env(seed=4)
        ctx = _ctx(env, seed=2)
        first = solve(env, ctx, _cfg(ctx, iters=30))
        cached_before = set(first.state.leaf_value_cache)
        assert cached_before, "no depth-limit leaf reached"
        # Re-search the same root with the carried state; record any recomputes.
        recomputed = self._count_continuation(monkeypatch)
        env2 = self._preflop_env(seed=4)
        ctx2 = _ctx(env2, seed=5)
        second = solve(env2, ctx2, _cfg(ctx2, iters=30), warm_start=first.state)
        assert second.state is first.state
        # No already-cached key is recomputed — cache hits skip the rollouts.
        assert set(recomputed).isdisjoint(cached_before)

    def test_caches_preserve_same_seed_determinism(self):
        def run():
            env = self._preflop_env(seed=6)
            ctx = _ctx(env, seed=7)
            np.random.seed(123)  # board-deal RNG (§6.4.1)
            return solve(env, ctx, _cfg(ctx, iters=25))

        r1, r2 = run(), run()
        assert set(r1.state.regret) == set(r2.state.regret)
        for k in r1.state.regret:
            np.testing.assert_allclose(r1.state.regret[k], r2.state.regret[k])
        # The caches themselves reproduce key-for-key under a fixed seed.
        assert set(r1.state.leaf_value_cache) == set(r2.state.leaf_value_cache)

    def test_forced_runout_terminal_memoises_once(self, monkeypatch):
        # The MCCFR forced-runout terminal integrates a given all-in once and
        # reuses it across seats / revisits via SolverState.runout_cache.
        env = _flop_env(seed=10)
        ctx = _ctx(env, seed=0)
        state = SolverState.empty()
        solver = _MCCFRSolver(env, state, ctx, _cfg(ctx), ctx.rng)
        term = copy.deepcopy(env)
        guard = 0
        while not term.is_terminal and guard < 12:
            legal = [a for a in term.legal_actions if a is not None]
            term.step_in_place("all_in" if "all_in" in legal else legal[0])
            guard += 1
        if not (term.is_decision_free and solver._use_equity):
            pytest.skip("flop line did not reach a decision-free runout")
        ref = copy.deepcopy(term).runout_equity()  # before patching the counter

        calls = {"n": 0}
        original = PokerEnv.runout_equity

        def spy(self, *a, **k):
            calls["n"] += 1
            return original(self, *a, **k)

        monkeypatch.setattr(PokerEnv, "runout_equity", spy)
        v0 = solver._terminal_value(term, 0)
        v0b = solver._terminal_value(term, 0)  # same key → hit
        v1 = solver._terminal_value(term, 1)   # other seat, same key → hit
        assert calls["n"] == 1
        assert v0b == v0
        assert abs(v0 - ref[0]) < 1e-9
        assert abs(v1 - ref[1]) < 1e-9
