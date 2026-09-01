"""``MemmapLookup.seat_batch_lookup`` must be bit-exact with the scalar path.

Sibling of ``test_batch_cluster_lookup.py`` (which covers ``clusters_for_board``,
the "thousands of combo rows against one board" shape used by the leaf-free
vector regime). This file covers the OTHER shape: ``seat_batch_lookup`` batches
a handful of seats' hole pairs against one board — the compiled search core's
leaf-rollout cluster refresh (``FastState.refresh_clusters``). It is backed by
a ``nogil`` Cython kernel (``poker_ai._core._cluster_lookup.batch_row_lookup``)
that reimplements the same combinadic row-index identity directly, rather than
calling ``clusters_for_board`` — a first attempt reusing ``clusters_for_board``
here was measured to be a net *regression* at realistic player counts (its
fixed per-call numpy overhead only pays off well above 8 players), so this
kernel exists specifically to be fast at small ``n``.
"""

import numpy as np
import pytest

from poker_ai import _core

pytestmark = pytest.mark.skipif(
    not _core.CORE_AVAILABLE, reason="compiled core extension not built"
)

from environment.player import Player
from environment.poker_env import PokerEnv
from information_abstraction.lookup import MemmapLookup, load_info_set_lut
from test.abstraction_helpers import passive_action

LUT_PATH = "data/20cards_exact"
_STREETS = {1: "flop", 2: "turn", 3: "river"}


@pytest.fixture(scope="module")
def lut():
    loaded = load_info_set_lut(LUT_PATH)
    if not loaded:
        pytest.skip(f"no LUT at {LUT_PATH}")
    return loaded


def _env_at(street: int, seed: int) -> PokerEnv:
    np.random.seed(seed)
    env = PokerEnv(
        players=[Player(i, 10_000) for i in range(2)],
        low_card_rank=10,
        high_card_rank=14,
    )
    guard = 0
    while env.betting_round < street and not env.is_terminal and guard < 60:
        env.step_in_place(passive_action(env))
        guard += 1
    return env


def _scalar_reference(stage_lut, holes, board):
    out = []
    for h in holes:
        key = tuple(sorted(int(c) for c in h) + sorted(int(c) for c in board))
        out.append(int(stage_lut[key]))
    return out


def _deck_for(stage_lut):
    return np.array(sorted(stage_lut._card_to_idx.keys()), dtype=np.int64)


@pytest.mark.parametrize("street", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_seat_batch_matches_scalar_getitem(lut, street, seed):
    stage_lut = lut[_STREETS[street]]
    if not isinstance(stage_lut, MemmapLookup):
        pytest.skip(f"{_STREETS[street]} is not a MemmapLookup in this LUT")
    blen = {1: 3, 2: 4, 3: 5}[street]
    deck = _deck_for(stage_lut)
    rng = np.random.RandomState(seed)
    for n_players in (2, 3, 4, 6, 9):
        need = 2 * n_players + blen
        if need > len(deck):
            continue
        cards = rng.choice(deck, size=need, replace=False)
        holes = np.array([cards[2 * i:2 * i + 2] for i in range(n_players)], dtype=np.int64)
        board = np.array(cards[2 * n_players:2 * n_players + blen], dtype=np.int64)

        expected = _scalar_reference(stage_lut, holes, board)
        got = stage_lut.seat_batch_lookup(holes, board)

        assert got.dtype == np.int64
        assert list(got) == expected, (
            f"street={_STREETS[street]} seed={seed} n_players={n_players}: "
            f"got={list(got)} expected={expected}"
        )
        assert all(c >= 0 for c in got), "valid disjoint input must never sentinel"


def test_seat_batch_lookup_empty_input(lut):
    stage_lut = lut["river"]
    if not isinstance(stage_lut, MemmapLookup):
        pytest.skip("river is not a MemmapLookup in this LUT")
    holes = np.empty((0, 2), dtype=np.int64)
    board = _deck_for(stage_lut)[:5]
    got = stage_lut.seat_batch_lookup(holes, board)
    assert got.shape == (0,)
    assert got.dtype == np.int64


def test_seat_batch_lookup_unknown_card_returns_sentinel(lut):
    """A card outside this street's deck must yield -1, not a crash or a
    garbage gathered row."""
    stage_lut = lut["river"]
    if not isinstance(stage_lut, MemmapLookup):
        pytest.skip("river is not a MemmapLookup in this LUT")
    deck = _deck_for(stage_lut)
    bogus_card = int(deck.max()) + 10_000  # guaranteed absent from _card_to_idx
    holes = np.array([[deck[0], bogus_card], [deck[1], deck[2]]], dtype=np.int64)
    board = deck[3:8]
    got = stage_lut.seat_batch_lookup(holes, board)
    assert got[0] == -1
    assert got[1] >= 0  # the other, fully-valid row is unaffected


def test_seat_batch_lookup_out_of_bounds_row_returns_sentinel():
    """Directly exercise the kernel's row-bounds guard with a fabricated
    (deliberately tiny) `mm`, rather than relying on natural card validity to
    ever produce an out-of-range row (it shouldn't, for real LUTs)."""
    from poker_ai._core._cluster_lookup import batch_row_lookup

    # Tiny synthetic 4-card deck: cards 0..3, indices 0..3.
    keys_sorted = np.array([0, 1, 2, 3], dtype=np.int64)
    vals_sorted = np.array([0, 1, 2, 3], dtype=np.int64)
    from information_abstraction.lookup import _comb_table
    C = _comb_table(4, 5)
    n_cards = 4
    # hole=(2,3) is the LAST hole-pair in lex order over a 4-card deck (rank
    # 5 of 6), board=(0,) the first of the 2 remaining -> row = 5*C[2,1]+0 =
    # 10, which overflows a deliberately undersized 1-row mm.
    mm = np.array([7], dtype=np.uint16)
    holes = np.array([[2, 3]], dtype=np.int64)
    board = np.array([0], dtype=np.int64)
    got = batch_row_lookup(holes, board, keys_sorted, vals_sorted, C, mm, n_cards)
    assert got[0] == -1


def test_seat_batch_lookup_survives_pickle_roundtrip(lut):
    """`_idx_cache` is dropped on pickle; seat_batch_lookup must rebuild it."""
    import pickle

    stage_lut = lut["river"]
    if not isinstance(stage_lut, MemmapLookup):
        pytest.skip("river is not a MemmapLookup in this LUT")
    deck = _deck_for(stage_lut)
    holes = np.array([[deck[0], deck[1]], [deck[2], deck[3]]], dtype=np.int64)
    board = deck[4:9]

    before = stage_lut.seat_batch_lookup(holes, board)
    revived = pickle.loads(pickle.dumps(stage_lut))
    after = revived.seat_batch_lookup(holes, board)
    assert np.array_equal(before, after)


def test_seat_batch_lookup_dtype_and_shape(lut):
    stage_lut = lut["turn"]
    if not isinstance(stage_lut, MemmapLookup):
        pytest.skip("turn is not a MemmapLookup in this LUT")
    deck = _deck_for(stage_lut)
    holes = np.array([[deck[0], deck[1]], [deck[2], deck[3]], [deck[4], deck[5]]], dtype=np.int64)
    board = deck[6:10]
    got = stage_lut.seat_batch_lookup(holes, board)
    assert got.shape == (3,)
    assert got.dtype == np.int64
