"""Tests for the decision-free all-in runout evaluator (§6.4).

``PokerEnv.is_decision_free`` / ``PokerEnv.runout_equity`` replace the env's
single sampled all-in board with the exact expected value over **every** board
completion, reusing the recorded pre-runout snapshot (board prefix, frozen pot
contributions, active mask) and ``Pot.compute_utility`` for side pots.

The reference is an independent brute-force enumeration built from the same
snapshot, asserted equal to integer-chip exactness; plus invariant checks
(zero-sum, card removal, make/undo round-trip, the cap-fallback path) and the
``is_decision_free`` predicate across fold / showdown / non-terminal states.
"""

import collections
import copy
import itertools

import numpy as np
import pytest

import environment.dynamics as dynamics
from environment.player import Player
from environment.poker_env import PokerEnv
from environment.pot import Pot


def _env(stacks, low=11, high=14, seed=0):
    """Env over a configurable (small by default) deck.  A reduced rank range
    keeps full board enumeration tractable for the exact brute-force checks;
    the evaluator is shared with the env's own ``compute_winners`` so the
    reference and the unit are scored identically."""
    np.random.seed(seed)
    return PokerEnv(
        players=[Player(i, s) for i, s in enumerate(stacks)],
        low_card_rank=low,
        high_card_rank=high,
    )


def _drive_to_allin_runout(env, max_steps=12):
    """Step all-in / call until the hand force-resolves to showdown.  Returns
    True if it ended as a decision-free runout."""
    steps = 0
    while not env.is_terminal and steps < max_steps:
        legal = [a for a in env.legal_actions if a is not None]
        nxt = next((a for a in ("all_in", "call", "check") if a in legal), legal[-1])
        env.step_in_place(nxt)
        steps += 1
    return env.is_decision_free


def _shove_to_runout(env):
    """Drive a heads-up hand to a decision-free all-in runout under the corrected
    all-in contract: one player shoves and the opponent then calls all-in (a
    shove is no longer terminal on its own — the opponent must first respond).
    No-op if the shove already ended the hand (e.g. an all-in on a complete
    board, where the opponent's response leaves nothing to run out)."""
    env.step_in_place("all_in")
    if not env.is_terminal:
        env.step_in_place("all_in" if "all_in" in env.legal_actions else "call")
    return env


def _brute_runout_equity(env):
    """Independent reference: enumerate every board completion of the recorded
    prefix, score the showdown via the shared evaluator + ``Pot.compute_utility``,
    average, subtract each seat's contribution."""
    prefix, pot_chips, active = env._runout_info
    n = len(env.players)
    used = set(int(c) for c in prefix)
    for p in env.players:
        used.update(int(c) for c in p._cards)
    avail = [int(c) for c in env.deck._cards if int(c) not in used]
    k = 5 - len(prefix)
    scratch = Pot(n)
    scratch._chips = list(pot_chips)
    acc = [0.0] * n
    cnt = 0
    for comp in itertools.combinations(avail, k):
        board = list(prefix) + list(comp)
        groups = collections.defaultdict(list)
        for i in range(n):
            if active[i]:
                rank = dynamics._evaluator.evaluate(board, list(env.players[i]._cards))
                groups[rank].append(env.players[i])
        ranked = [groups[r] for r in sorted(groups)]
        winnings = scratch.compute_utility(env.players, ranked)
        for i in range(n):
            acc[i] += winnings[i]
        cnt += 1
    return {i: acc[i] / cnt - pot_chips[i] for i in range(n)}, cnt


class TestIsDecisionFree:

    def test_false_on_fresh_non_terminal_env(self):
        env = _env([10000, 10000])
        assert not env.is_decision_free
        assert not env.is_terminal

    def test_true_on_allin_runout(self):
        env = _env([10000, 10000])
        _shove_to_runout(env)  # shove + opponent calls all-in -> runout
        assert env.is_terminal
        assert env.is_decision_free
        assert env._runout_info is not None

    def test_false_on_fold_terminal(self):
        # A pre-flop fold ends the hand without a showdown runout.
        env = _env([10000, 10000])
        env.step_in_place("fold")
        assert env.is_terminal
        assert not env.is_decision_free

    def test_false_when_only_one_player_active_after_folds(self):
        # 3-way: folds reduce the field to a single active player.  The env
        # still force-resolves (deals a board), but there is no showdown, so it
        # is not a decision-free runout (the n_active >= 2 capture guard).
        env = _env([10000, 10000, 10000])
        guard = 0
        while not env.is_terminal and guard < 8:
            legal = [a for a in env.legal_actions if a is not None]
            env.step_in_place("fold" if "fold" in legal else legal[-1])
            guard += 1
        assert env.is_terminal
        assert not env.is_decision_free
        assert env._runout_info is None

    def test_false_when_board_already_complete(self):
        # Calldown to the river, then an all-in on a complete (5-card) board:
        # there is nothing left to run out, so it is not a decision-free runout.
        env = _env([10000, 10000])
        guard = 0
        while env.betting_round < 3 and not env.is_terminal and guard < 30:
            env.step_in_place("check" if "check" in env.legal_actions else "call")
            guard += 1
        assert env.betting_round == 3 and len(env.community_cards) == 5
        env.step_in_place("all_in")
        # However it ends, no incomplete-board runout was recorded.
        assert not env.is_decision_free


class TestRunoutEquityPreconditions:

    def test_raises_when_not_decision_free(self):
        env = _env([10000, 10000])
        with pytest.raises(ValueError, match="decision-free"):
            env.runout_equity()


class TestRunoutEquityExact:

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_headsup_matches_brute_force(self, seed):
        env = _env([10000, 10000], seed=seed)
        _shove_to_runout(env)
        assert env.is_decision_free
        eq = env.runout_equity()
        ref, _ = _brute_runout_equity(env)
        for i in range(2):
            assert abs(eq[i] - ref[i]) < 1e-9

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_three_way_side_pots_match_brute_force(self, seed):
        # Unequal stacks → a short all-in creates side pots; the equity must
        # respect the frozen side-pot structure.
        env = _env([3000, 8000, 8000], seed=seed)
        if not _drive_to_allin_runout(env):
            pytest.skip("sequence did not reach an all-in runout for this seed")
        _, pot_chips, _ = env._runout_info
        assert len(set(pot_chips)) > 1  # genuinely unequal contributions
        eq = env.runout_equity()
        ref, _ = _brute_runout_equity(env)
        for i in range(3):
            assert abs(eq[i] - ref[i]) < 1e-9

    def test_flop_allin_two_card_runout_full_deck(self):
        # In-search shape: all-in on the flop → 2 board cards to come, exact
        # path (C(45,2)=990 < cap).  Full deck so showdown ranks are real.
        np.random.seed(7)
        env = PokerEnv(players=[Player(i, 10000) for i in range(2)])
        guard = 0
        while env.betting_round < 1 and guard < 10:
            env.step_in_place("call" if "call" in env.legal_actions else "check")
            guard += 1
        assert env.betting_round == 1 and len(env.community_cards) == 3
        _shove_to_runout(env)
        assert env.is_decision_free
        prefix, _, _ = env._runout_info
        assert len(prefix) == 3
        eq = env.runout_equity()
        ref, cnt = _brute_runout_equity(env)
        assert cnt == 990
        for i in range(2):
            assert abs(eq[i] - ref[i]) < 1e-9


class TestRunoutEquityInvariants:

    def test_zero_sum(self):
        env = _env([10000, 10000], seed=2)
        _shove_to_runout(env)
        eq = env.runout_equity()
        assert abs(sum(eq.values())) < 1e-6

    def test_runout_equity_does_not_mutate_env(self):
        # Pure computation: the env must be field-identical before and after.
        env = _env([10000, 10000], seed=2)
        _shove_to_runout(env)
        snap = (
            tuple(env.community_cards),
            list(env.pot._chips),
            [(p.n_chips, p.is_active, tuple(p._cards)) for p in env.players],
            tuple(int(c) for c in env.deck._cards),
            env._runout_info,
        )
        _ = env.runout_equity()
        after = (
            tuple(env.community_cards),
            list(env.pot._chips),
            [(p.n_chips, p.is_active, tuple(p._cards)) for p in env.players],
            tuple(int(c) for c in env.deck._cards),
            env._runout_info,
        )
        assert snap == after

    def test_fallback_deterministic_under_fixed_rng(self):
        # Same rng seed → identical sampled-fallback estimate.
        np.random.seed(0)
        env = PokerEnv(players=[Player(i, 10000) for i in range(2)])
        _shove_to_runout(env)  # full-deck preflop → 5-card runout > cap
        a = env.runout_equity(rng=np.random.default_rng(11), cap=1000)
        b = env.runout_equity(rng=np.random.default_rng(11), cap=1000)
        assert a == b

    def test_card_removal_excludes_all_holes(self):
        # No board completion the enumeration scores may contain a dealt hole.
        env = _env([10000, 10000], seed=1)
        _shove_to_runout(env)
        prefix, _, _ = env._runout_info
        holes = set()
        for p in env.players:
            holes.update(int(c) for c in p._cards)
        used = set(int(c) for c in prefix) | holes
        avail = [int(c) for c in env.deck._cards if int(c) not in used]
        assert holes.isdisjoint(avail)
        # And the enumeration is over exactly the available cards.
        assert len(avail) == env.deck_size - len(holes) - len(prefix)

    def test_matches_sampled_payout_mean(self):
        # The exact runout equity equals the mean of the env's own single-board
        # resolution over many independent runouts (statistical sanity).  Small
        # stacks keep the contested pot — and thus the Monte-Carlo variance —
        # small enough for a tight tolerance; deterministic under np.seed(0).
        np.random.seed(0)
        base = PokerEnv(players=[Player(i, 200) for i in range(2)],
                        low_card_rank=10, high_card_rank=14)
        _shove_to_runout(base)
        eq = base.runout_equity()
        holes = [tuple(base.players[i]._cards) for i in range(2)]
        acc = np.zeros(2)
        N = 3000
        for _ in range(N):
            e = PokerEnv(players=[Player(i, 200) for i in range(2)],
                         low_card_rank=10, high_card_rank=14)
            e = e.with_hole_cards(holes)
            _shove_to_runout(e)
            for i in range(2):
                acc[i] += e.payout[i]
        mc = acc / N
        # Within Monte-Carlo tolerance for the (now full-stack) contested pot.
        assert abs(mc[0] - eq[0]) < 15.0


class TestMakeUndoRoundTrip:

    def test_undo_clears_runout_info(self):
        env = _env([10000, 10000], seed=3)
        env.step_in_place("all_in")  # shove: not terminal, no runout yet
        assert env._runout_info is None
        before = copy.deepcopy(env)
        # The opponent's all-in call is the step that reaches the runout.
        token = env.step_in_place(
            "all_in" if "all_in" in env.legal_actions else "call"
        )
        assert env._runout_info is not None
        env.undo(token)
        assert env._runout_info is None
        assert env._runout_info == before._runout_info

    def test_deepcopy_carries_runout_info(self):
        env = _env([10000, 10000], seed=3)
        _shove_to_runout(env)
        clone = copy.deepcopy(env)
        assert clone._runout_info == env._runout_info
        assert clone.runout_equity() == env.runout_equity()


class TestRunoutEquityCapFallback:

    def test_preflop_full_deck_samples_and_logs(self, caplog):
        # Pre-flop all-in on the full deck → 5 cards to come, C(48,5) ≫ cap, so
        # the sampling fallback runs: finite, zero-sum, and it warns.
        np.random.seed(0)
        env = PokerEnv(players=[Player(i, 10000) for i in range(2)])
        _shove_to_runout(env)
        assert env.is_decision_free
        with caplog.at_level("WARNING"):
            eq = env.runout_equity(rng=np.random.default_rng(0), cap=2000)
        assert any("exceed cap" in r.message for r in caplog.records)
        assert all(np.isfinite(v) for v in eq.values())
        assert abs(sum(eq.values())) < 1e-6

    def test_exact_path_below_cap_does_not_warn(self, caplog):
        env = _env([10000, 10000], seed=0)
        _shove_to_runout(env)
        with caplog.at_level("WARNING"):
            env.runout_equity()
        assert not any("exceed cap" in r.message for r in caplog.records)


class TestSinglePotFastPath:
    """The single-pot vectorised scoring path (winner-takes-pot) must match the
    scalar ``compute_utility`` reference exactly, and the rare board-tie
    completions must route through the scalar fallback."""

    def _tie_completions(self, env) -> int:
        """Number of board completions where >=2 active players tie for best."""
        prefix, _, active = env._runout_info
        used = set(int(c) for c in prefix)
        for p in env.players:
            used.update(int(c) for c in p._cards)
        avail = [int(c) for c in env.deck._cards if int(c) not in used]
        k = 5 - len(prefix)
        ties = 0
        for comp in itertools.combinations(avail, k):
            board = list(prefix) + list(comp)
            ranks = sorted(
                dynamics._evaluator.evaluate(board, list(env.players[i]._cards))
                for i in range(len(env.players))
                if active[i]
            )
            if len(ranks) >= 2 and ranks[0] == ranks[1]:
                ties += 1
        return ties

    def test_board_ties_exercise_scalar_fallback(self):
        # A small deck makes board-tie completions common; at least one seed must
        # hit the nwin>=2 fallback branch, and equity must still match exactly.
        seen_tie = False
        for seed in range(8):
            env = _env([10000, 10000], seed=seed)
            _shove_to_runout(env)
            if not env.is_decision_free:
                continue
            if self._tie_completions(env) > 0:
                seen_tie = True
                eq = env.runout_equity()
                ref, _ = _brute_runout_equity(env)
                for i in range(2):
                    assert abs(eq[i] - ref[i]) < 1e-9
        assert seen_tie, "no seed produced a board-tie completion"

    def test_multiway_side_pot_uses_scalar_path(self):
        # Unequal all-in stacks → >1 side pot → the fast path is *not* taken; the
        # untouched scalar path must still match the brute-force reference.
        found = False
        for seed in range(16):
            env = _env([3000, 8000, 8000], seed=seed)
            if not _drive_to_allin_runout(env):
                continue
            scratch = Pot(3)
            scratch._chips = list(env._runout_info[1])
            if len(scratch.side_pots) <= 1:
                continue
            found = True
            eq = env.runout_equity()
            ref, _ = _brute_runout_equity(env)
            for i in range(3):
                assert abs(eq[i] - ref[i]) < 1e-9
        assert found, "no seed produced a multiway side-pot runout"
