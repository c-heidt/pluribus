"""Tests for the vectorised range-vs-range showdown (§6.5, F2).

``showdown.showdown_cfv`` settles every acting combo against an opponent's
reach-weighted range in O(n log n) with exact card removal.  The reference here
is an independent brute-force O(n^2) all-pairs sum built from the *same* scalar
evaluator, asserted equal to floating tolerance, plus invariants (card removal,
ties net zero, zero-sum/antisymmetry, stake linearity, board incompatibility)
and consistency with the engine's own ``Pot.compute_utility`` on a concrete pair.
"""

import time

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv
from environment.pot import Pot
from environment import range_showdown as showdown
from environment.range_showdown import (
    rank_combos_on_board,
    removal_index,
    showdown_cfv,
    showdown_values,
)


# A small deck keeps the O(n^2) brute-force reference cheap: ranks 10-14 → a
# 20-card deck, C(20, 2) = 190 combos.
def _small_env():
    return PokerEnv(
        players=[Player(i, 10000) for i in range(2)],
        low_card_rank=10,
        high_card_rank=14,
    )


def _pick_board(combo_cards, rng, n_cards=5):
    """Five distinct deck cards as a board."""
    cards = np.unique(combo_cards)
    return [int(c) for c in rng.choice(cards, size=n_cards, replace=False)]


def _brute_cfv(ranks, valid, combo_cards, opp_reach, stake):
    """Independent O(n^2) reference with card removal.

    i beats j (gains) when j is weaker (rank[j] > rank[i]); loses when j is
    stronger.  Opponent combos sharing a card with i are excluded (removal).
    """
    n = len(ranks)
    sets = [frozenset((int(combo_cards[k, 0]), int(combo_cards[k, 1]))) for k in range(n)]
    v = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not valid[i]:
            continue
        tot = 0.0
        for j in range(n):
            if not valid[j] or sets[i] & sets[j]:
                continue
            if ranks[j] > ranks[i]:
                tot += opp_reach[j]
            elif ranks[j] < ranks[i]:
                tot -= opp_reach[j]
        v[i] = stake * tot
    return v


def _random_reach(n, valid, rng):
    w = rng.random(n)
    w[~valid] = 0.0
    return w


class TestRankCombosOnBoard:

    def test_board_cards_invalidate_sharing_combos(self):
        env = _small_env()
        cc = env.combo_cards
        rng = np.random.default_rng(0)
        board = _pick_board(cc, rng)
        ranks, valid = rank_combos_on_board(cc, board)
        board_set = set(board)
        for i in range(cc.shape[0]):
            shares = int(cc[i, 0]) in board_set or int(cc[i, 1]) in board_set
            assert valid[i] == (not shares)
        # Valid combos get a real rank; invalid ones the sentinel.
        assert (ranks[valid] < 7463).all()
        assert (ranks[~valid] == showdown._SENTINEL_RANK).all()

    def test_matches_evaluator_directly(self):
        env = _small_env()
        cc = env.combo_cards
        rng = np.random.default_rng(1)
        board = _pick_board(cc, rng)
        ranks, valid = rank_combos_on_board(cc, board)
        import environment.dynamics as dynamics

        for i in np.flatnonzero(valid)[:25]:
            expect = dynamics._evaluator.evaluate(
                [int(cc[i, 0]), int(cc[i, 1])], board
            )
            assert ranks[i] == expect


class TestShowdownCfvExact:

    @pytest.mark.parametrize("trial", range(6))
    def test_matches_brute_force(self, trial):
        env = _small_env()
        cc = env.combo_cards
        rng = np.random.default_rng(trial)
        board = _pick_board(cc, rng)
        ranks, valid = rank_combos_on_board(cc, board)
        opp = _random_reach(cc.shape[0], valid, rng)
        stake = 137.0
        got = showdown_cfv(ranks, valid, cc, opp, stake)
        ref = _brute_cfv(ranks, valid, cc, opp, stake)
        np.testing.assert_allclose(got, ref, atol=1e-7, rtol=0)

    def test_turn_board_four_cards(self):
        # Ranking works for a 4-card (turn) board too; settle still matches.
        env = _small_env()
        cc = env.combo_cards
        rng = np.random.default_rng(3)
        board = _pick_board(cc, rng, n_cards=4)
        ranks, valid = rank_combos_on_board(cc, board)
        opp = _random_reach(cc.shape[0], valid, rng)
        got = showdown_cfv(ranks, valid, cc, opp, 10.0)
        ref = _brute_cfv(ranks, valid, cc, opp, 10.0)
        np.testing.assert_allclose(got, ref, atol=1e-7, rtol=0)


class TestCardRemoval:

    def test_conflicting_opponent_mass_contributes_zero(self):
        # If the opponent's entire reach sits on combos that share a card with
        # the acting combo, card removal makes the acting CFV exactly zero.
        env = _small_env()
        cc = env.combo_cards
        rng = np.random.default_rng(5)
        board = _pick_board(cc, rng)
        ranks, valid = rank_combos_on_board(cc, board)
        i0 = int(np.flatnonzero(valid)[0])
        c0, c1 = int(cc[i0, 0]), int(cc[i0, 1])
        conflict = np.array(
            [
                valid[j] and (c0 in (int(cc[j, 0]), int(cc[j, 1]))
                              or c1 in (int(cc[j, 0]), int(cc[j, 1])))
                for j in range(cc.shape[0])
            ]
        )
        opp = np.zeros(cc.shape[0])
        opp[conflict] = rng.random(int(conflict.sum())) + 0.1
        cfv = showdown_cfv(ranks, valid, cc, opp, 50.0)
        assert abs(cfv[i0]) < 1e-9

    def test_ignoring_removal_would_differ(self):
        # Sanity that removal is actually doing something: a reference that
        # ignores removal disagrees with our result on at least some combo.
        env = _small_env()
        cc = env.combo_cards
        rng = np.random.default_rng(6)
        board = _pick_board(cc, rng)
        ranks, valid = rank_combos_on_board(cc, board)
        opp = _random_reach(cc.shape[0], valid, rng)
        got = showdown_cfv(ranks, valid, cc, opp, 1.0)

        # No-removal reference: full prefix/suffix sums, no per-card subtraction.
        order = np.argsort(ranks, kind="stable")
        r_sorted = ranks[order]
        w = np.where(valid, opp, 0.0)[order]
        no_rm = np.zeros(cc.shape[0])
        # beats = weaker mass, loses = stronger mass, ignoring shared cards.
        total = w.sum()
        # build strict prefix (stronger) and suffix (weaker) by rank groups
        change = np.ones(len(ranks), dtype=bool)
        change[1:] = r_sorted[1:] != r_sorted[:-1]
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], len(ranks))
        acc = 0.0
        loses_sorted = np.zeros(len(ranks))
        for a, b in zip(starts, ends):
            loses_sorted[a:b] = acc
            acc += w[a:b].sum()
        acc = 0.0
        beats_sorted = np.zeros(len(ranks))
        for a, b in zip(reversed(starts), reversed(ends)):
            beats_sorted[a:b] = acc
            acc += w[a:b].sum()
        no_rm[order] = beats_sorted - loses_sorted
        no_rm[~valid] = 0.0
        assert not np.allclose(got, no_rm, atol=1e-6)


class TestTies:

    def test_equal_rank_matchups_net_zero(self):
        # Find a card-disjoint pair of valid combos that tie on some board, put
        # all opponent reach on one, and assert the other's CFV is zero.
        env = _small_env()
        cc = env.combo_cards
        for seed in range(20):
            rng = np.random.default_rng(100 + seed)
            board = _pick_board(cc, rng)
            ranks, valid = rank_combos_on_board(cc, board)
            vidx = np.flatnonzero(valid)
            sets = {k: (int(cc[k, 0]), int(cc[k, 1])) for k in vidx}
            pair = None
            for a in vidx:
                for b in vidx:
                    if a < b and ranks[a] == ranks[b] and not (
                        set(sets[a]) & set(sets[b])
                    ):
                        pair = (int(a), int(b))
                        break
                if pair:
                    break
            if not pair:
                continue
            i, j = pair
            opp = np.zeros(cc.shape[0])
            opp[j] = 3.0
            cfv = showdown_cfv(ranks, valid, cc, opp, 25.0)
            assert abs(cfv[i]) < 1e-9
            return
        pytest.skip("no tied card-disjoint pair found across seeds")


class TestInvariants:

    def test_board_incompatible_acting_combos_are_zero(self):
        env = _small_env()
        cc = env.combo_cards
        rng = np.random.default_rng(7)
        board = _pick_board(cc, rng)
        ranks, valid = rank_combos_on_board(cc, board)
        opp = _random_reach(cc.shape[0], valid, rng)
        cfv = showdown_cfv(ranks, valid, cc, opp, 10.0)
        assert np.all(cfv[~valid] == 0.0)

    def test_zero_sum_antisymmetry(self):
        env = _small_env()
        cc = env.combo_cards
        rng = np.random.default_rng(8)
        board = _pick_board(cc, rng)
        reach_a = _random_reach(cc.shape[0], np.ones(cc.shape[0], bool), rng)
        reach_b = _random_reach(cc.shape[0], np.ones(cc.shape[0], bool), rng)
        cfv_a, cfv_b = showdown_values(cc, board, reach_a, reach_b, 1.0)
        joint = float(reach_a @ cfv_a + reach_b @ cfv_b)
        assert abs(joint) < 1e-6

    def test_stake_linearity(self):
        env = _small_env()
        cc = env.combo_cards
        rng = np.random.default_rng(9)
        board = _pick_board(cc, rng)
        ranks, valid = rank_combos_on_board(cc, board)
        opp = _random_reach(cc.shape[0], valid, rng)
        base = showdown_cfv(ranks, valid, cc, opp, 1.0)
        scaled = showdown_cfv(ranks, valid, cc, opp, 7.5)
        np.testing.assert_allclose(scaled, 7.5 * base, atol=1e-9, rtol=0)

    def test_deterministic(self):
        env = _small_env()
        cc = env.combo_cards
        rng = np.random.default_rng(10)
        board = _pick_board(cc, rng)
        ranks, valid = rank_combos_on_board(cc, board)
        opp = _random_reach(cc.shape[0], valid, rng)
        a = showdown_cfv(ranks, valid, cc, opp, 3.0)
        b = showdown_cfv(ranks, valid, cc, opp, 3.0)
        assert np.array_equal(a, b)


class TestEngineConsistency:

    def test_matches_compute_utility_for_concrete_pair(self):
        # One-hot reaches → a single concrete heads-up showdown.  The vectorised
        # CFV must equal the chip delta from Pot.compute_utility (gross − stake).
        env = _small_env()
        cc = env.combo_cards
        import environment.dynamics as dynamics

        for seed in range(30):
            rng = np.random.default_rng(200 + seed)
            board = _pick_board(cc, rng)
            ranks, valid = rank_combos_on_board(cc, board)
            vidx = np.flatnonzero(valid)
            i, j = int(vidx[0]), None
            si = (int(cc[i, 0]), int(cc[i, 1]))
            for k in vidx[1:]:
                sk = (int(cc[k, 0]), int(cc[k, 1]))
                if not (set(si) & set(sk)):
                    j = int(k)
                    break
            if j is None:
                continue

            stake = 80
            opp = np.zeros(cc.shape[0])
            opp[j] = 1.0
            cfv = showdown_cfv(ranks, valid, cc, opp, float(stake))

            # Reference via the engine's own side-pot / utility machinery.
            pa, pb = Player(0, 0), Player(1, 0)
            pa.order, pb.order = 0, 1
            pa._cards = (int(cc[i, 0]), int(cc[i, 1]))
            pb._cards = (int(cc[j, 0]), int(cc[j, 1]))
            pot = Pot(2)
            pot.add_chips(0, stake)
            pot.add_chips(1, stake)
            ri = dynamics._evaluator.evaluate(list(pa._cards), board)
            rj = dynamics._evaluator.evaluate(list(pb._cards), board)
            groups = {}
            for p, r in ((pa, ri), (pb, rj)):
                groups.setdefault(r, []).append(p)
            ranked = [groups[r] for r in sorted(groups)]
            payouts = pot.compute_utility([pa, pb], ranked)
            net_i = payouts[0] - stake  # gross winnings − contribution
            assert cfv[i] == pytest.approx(float(net_i), abs=1e-9)
            return
        pytest.skip("no card-disjoint pair found")


class TestRemovalParam:

    def test_precomputed_removal_matches_inline(self):
        # The optional precomputed removal index must give a bit-identical result
        # to deriving it inline from combo_cards.
        env = _small_env()
        cc = env.combo_cards
        rng = np.random.default_rng(11)
        board = _pick_board(cc, rng)
        ranks, valid = rank_combos_on_board(cc, board)
        opp = _random_reach(cc.shape[0], valid, rng)
        inline = showdown_cfv(ranks, valid, cc, opp, 5.0)
        pre = showdown_cfv(ranks, valid, cc, opp, 5.0, removal=removal_index(cc))
        assert np.array_equal(inline, pre)

    def test_removal_index_is_a_bijection_over_deck(self):
        env = _small_env()
        cc = env.combo_cards
        s0, s1, deck_size = removal_index(cc)
        assert deck_size == len(np.unique(cc))
        assert s0.min() >= 0 and s1.max() < deck_size
        # Slots round-trip: equal card int <=> equal slot.
        flat_cards = np.concatenate([cc[:, 0], cc[:, 1]])
        flat_slots = np.concatenate([s0, s1])
        for c in np.unique(flat_cards):
            slots = np.unique(flat_slots[flat_cards == c])
            assert slots.shape == (1,)


class TestDegenerate:

    def test_zero_reach_gives_zero(self):
        env = _small_env()
        cc = env.combo_cards
        rng = np.random.default_rng(12)
        board = _pick_board(cc, rng)
        ranks, valid = rank_combos_on_board(cc, board)
        cfv = showdown_cfv(ranks, valid, cc, np.zeros(cc.shape[0]), 100.0)
        assert np.all(cfv == 0.0)

    def test_tiny_three_rank_deck(self):
        # 3 ranks → 12 cards (the env's minimum for heads-up + a 5-card board),
        # C(12, 2) = 66 combos, few valid after the board.  Must not crash, must
        # match brute force, and stays zero-sum.
        env = PokerEnv(
            players=[Player(i, 10000) for i in range(2)],
            low_card_rank=12,
            high_card_rank=14,
        )
        cc = env.combo_cards
        rng = np.random.default_rng(0)
        board = _pick_board(cc, rng)
        ranks, valid = rank_combos_on_board(cc, board)
        full = np.ones(cc.shape[0], bool)
        ra = _random_reach(cc.shape[0], full, rng)
        rb = _random_reach(cc.shape[0], full, rng)
        ca, cb = showdown_values(cc, board, ra, rb, 1.0)
        # Brute force still agrees on this degenerate deck.
        np.testing.assert_allclose(
            ca, _brute_cfv(ranks, valid, cc, rb, 1.0), atol=1e-9
        )
        assert abs(float(ra @ ca + rb @ cb)) < 1e-6


class TestDirectional:

    def test_strictly_better_hand_wins_stake(self):
        # One-hot reaches on a card-disjoint pair where i is strictly stronger:
        # A (holding i) gains +stake, B (holding j) loses −stake.
        env = _small_env()
        cc = env.combo_cards
        for seed in range(30):
            rng = np.random.default_rng(300 + seed)
            board = _pick_board(cc, rng)
            ranks, valid = rank_combos_on_board(cc, board)
            vidx = np.flatnonzero(valid)
            pick = None
            for a in vidx:
                for b in vidx:
                    if (
                        ranks[a] < ranks[b]
                        and not (
                            {int(cc[a, 0]), int(cc[a, 1])}
                            & {int(cc[b, 0]), int(cc[b, 1])}
                        )
                    ):
                        pick = (int(a), int(b))
                        break
                if pick:
                    break
            if not pick:
                continue
            i, j = pick
            reach_a = np.zeros(cc.shape[0])
            reach_b = np.zeros(cc.shape[0])
            reach_a[i] = 1.0
            reach_b[j] = 1.0
            ca, cb = showdown_values(cc, board, reach_a, reach_b, 40.0)
            assert ca[i] == pytest.approx(40.0)
            assert cb[j] == pytest.approx(-40.0)
            return
        pytest.skip("no strict-order card-disjoint pair found")


class TestPerf:

    @pytest.mark.slow
    def test_full_deck_under_budget(self):
        # Full 52-card deck, C(52, 2) = 1326 combos: rank + settle stays well
        # under a per-call budget with no n^2 blow-up.
        env = PokerEnv(players=[Player(i, 10000) for i in range(2)])
        cc = env.combo_cards
        rng = np.random.default_rng(0)
        board = _pick_board(cc, rng)
        ranks, valid = rank_combos_on_board(cc, board)  # ranking cost (per board)
        opp = _random_reach(cc.shape[0], valid, rng)
        t0 = time.perf_counter()
        for _ in range(50):  # 50 settlement calls (many terminals per board)
            showdown_cfv(ranks, valid, cc, opp, 100.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"50 settles took {elapsed:.3f}s"
