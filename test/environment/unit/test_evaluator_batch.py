"""Correctness of the vectorised batch evaluator vs the scalar oracle.

``Evaluator.evaluate_batch`` must return exactly the same ranks as the scalar
``Evaluator.evaluate`` / ``_five`` for every hand.  The strongest guarantee is the
**exhaustive** five-card test — all C(52, 5) = 2,598,960 distinct hands compared
element-for-element — plus large random six/seven-card samples and structured
poker goldens.  The scalar evaluator (pinned by ``test_evaluator.py``) is the
reference; the batch tables are derived from the same ``HandRankTable``.
"""

import itertools

import numpy as np
import pytest

from environment.evaluator import Evaluator
from environment.utils import make_card, make_deck_arr


@pytest.fixture(scope="module")
def ev():
    return Evaluator()


def _full_deck():
    return make_deck_arr(2, 14).astype(np.int64)


@pytest.mark.slow
class TestExhaustiveFiveCard:

    def test_all_c52_5_hands_match_scalar(self, ev):
        # Every distinct 5-card hand: batch == scalar _five, exactly.
        deck = _full_deck()
        hands = np.array(list(itertools.combinations(deck.tolist(), 5)), dtype=np.int64)
        assert hands.shape == (2598960, 5)

        batch = ev.evaluate_batch(hands)
        # Scalar oracle over the same hands (chunked Python loop).
        scalar = np.fromiter(
            (ev._five(list(h)) for h in hands.tolist()),
            dtype=np.int64,
            count=hands.shape[0],
        )
        assert np.array_equal(batch, scalar)
        # Sanity: the full rank range is exercised.
        assert batch.min() == 1
        assert batch.max() == 7462


@pytest.mark.slow
class TestSevenCardSample:

    def test_random_seven_card_hands_match_scalar(self, ev):
        deck = _full_deck()
        rng = np.random.default_rng(0)
        n = 200_000
        hands = np.empty((n, 7), dtype=np.int64)
        for i in range(n):
            hands[i] = rng.choice(deck, size=7, replace=False)
        batch = ev.evaluate_batch(hands)
        scalar = np.fromiter(
            (ev.evaluate(list(h), []) for h in hands.tolist()),
            dtype=np.int64,
            count=n,
        )
        assert np.array_equal(batch, scalar)


@pytest.mark.slow
class TestSixCardSample:

    def test_random_six_card_hands_match_scalar(self, ev):
        deck = _full_deck()
        rng = np.random.default_rng(1)
        n = 100_000
        hands = np.empty((n, 6), dtype=np.int64)
        for i in range(n):
            hands[i] = rng.choice(deck, size=6, replace=False)
        batch = ev.evaluate_batch(hands)
        scalar = np.fromiter(
            (ev.evaluate(list(h), []) for h in hands.tolist()),
            dtype=np.int64,
            count=n,
        )
        assert np.array_equal(batch, scalar)


class TestStructuredGoldens:

    def test_royal_flush_is_rank_one(self, ev):
        royal = np.array(
            [[make_card(r, "spades") for r in (14, 13, 12, 11, 10)]], dtype=np.int64
        )
        assert ev.evaluate_batch(royal)[0] == 1

    def test_full_house_beats_flush_beats_straight(self, ev):
        fh = [make_card(14, "spades"), make_card(14, "hearts"), make_card(14, "diamonds"),
              make_card(13, "spades"), make_card(13, "hearts")]
        flush = [make_card(r, "spades") for r in (14, 12, 10, 8, 6)]
        straight = [make_card(14, "spades"), make_card(13, "hearts"),
                    make_card(12, "diamonds"), make_card(11, "clubs"), make_card(10, "spades")]
        ranks = ev.evaluate_batch(np.array([fh, flush, straight], dtype=np.int64))
        assert ranks[0] < ranks[1] < ranks[2]

    def test_quads_and_wheel_straight(self, ev):
        quads = [make_card(7, s) for s in ("spades", "hearts", "diamonds", "clubs")]
        quads.append(make_card(2, "spades"))
        wheel = [make_card(r, "spades") for r in (5, 4, 3, 2)]
        wheel.append(make_card(14, "hearts"))  # A-2-3-4-5 (non-flush)
        ranks = ev.evaluate_batch(np.array([quads, wheel], dtype=np.int64))
        # quads in [11, 166]; wheel is the worst straight (rank MAX_STRAIGHT=1609).
        assert 10 < ranks[0] <= 166
        assert ranks[1] == 1609

    def test_board_plays_tie(self, ev):
        # Two hole sets over a royal-flush board both play the board → identical rank.
        board = [make_card(r, "spades") for r in (14, 13, 12, 11, 10)]
        h1 = board + [make_card(2, "hearts"), make_card(3, "hearts")]
        h2 = board + [make_card(4, "diamonds"), make_card(5, "diamonds")]
        ranks = ev.evaluate_batch(np.array([h1, h2], dtype=np.int64))
        assert ranks[0] == ranks[1] == 1


class TestSmallDecks:

    def test_short_deck_combos_on_boards_match_scalar(self, ev):
        # Ranks 10-14 (20-card deck): every combo over a few boards matches scalar.
        deck = make_deck_arr(10, 14).astype(np.int64)
        rng = np.random.default_rng(3)
        for _ in range(5):
            board = rng.choice(deck, size=5, replace=False)
            board_set = set(int(c) for c in board)
            holes = [
                c for c in itertools.combinations(deck.tolist(), 2)
                if c[0] not in board_set and c[1] not in board_set
            ]
            hands = np.array([list(h) + board.tolist() for h in holes], dtype=np.int64)
            batch = ev.evaluate_batch(hands)
            scalar = np.fromiter(
                (ev.evaluate(list(h), board.tolist()) for h in holes),
                dtype=np.int64, count=len(holes),
            )
            assert np.array_equal(batch, scalar)


class TestApiContract:

    def test_rejects_unsupported_card_count(self, ev):
        with pytest.raises(ValueError, match="K in"):
            ev.evaluate_batch(np.zeros((3, 4), dtype=np.int64))

    def test_rejects_non_2d(self, ev):
        with pytest.raises(ValueError, match="2-D"):
            ev.evaluate_batch(np.zeros(5, dtype=np.int64))

    def test_empty_batch_returns_empty(self, ev):
        out = ev.evaluate_batch(np.zeros((0, 7), dtype=np.int64))
        assert out.shape == (0,)

    def test_deterministic(self, ev):
        deck = _full_deck()
        rng = np.random.default_rng(7)
        hands = np.array(
            [rng.choice(deck, size=7, replace=False) for _ in range(1000)],
            dtype=np.int64,
        )
        assert np.array_equal(ev.evaluate_batch(hands), ev.evaluate_batch(hands))
