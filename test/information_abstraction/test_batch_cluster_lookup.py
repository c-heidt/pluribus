"""Batched combo->cluster lookup must be bit-exact with the scalar path.

``clusters_for_board`` collapses ``n_combos`` scalar ``cluster_for`` calls into a
handful of numpy ops plus one memmap gather, for a fixed board.  The vector
search regime keys its future-street tables on the result, so any disagreement
with the scalar path is a silent strategy corruption -- hence exact equality
(``==``, integer ids) against ``PokerEnv.cluster_for`` on every street.

Board-conflicting combos have no row on disk (the scalar path raises
``KeyError``); the batch path must report ``-1`` there rather than gathering a
garbage row.
"""

import itertools

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv
from information_abstraction.lookup import (
    MemmapLookup,
    clusters_for_board,
    load_info_set_lut,
)

LUT_PATH = "data/20cards_exact"
_STREETS = {1: "flop", 2: "turn", 3: "river"}


@pytest.fixture(scope="module")
def lut():
    loaded = load_info_set_lut(LUT_PATH)
    if not loaded:
        pytest.skip(f"no LUT at {LUT_PATH}")
    return loaded


def _env_at(street: int, seed: int) -> PokerEnv:
    """Heads-up 20-card env advanced to ``street`` (1=flop, 2=turn, 3=river)."""
    np.random.seed(seed)
    env = PokerEnv(
        players=[Player(i, 10_000) for i in range(2)],
        low_card_rank=10,
        high_card_rank=14,
    )
    guard = 0
    while env.betting_round < street and not env.is_terminal and guard < 60:
        env.step_in_place("call" if "call" in env.legal_actions else "check")
        guard += 1
    return env


def _scalar_reference(env, lut, valid_mask) -> np.ndarray:
    """Ground truth: PokerEnv.cluster_for per combo, -1 where board-conflicting."""
    env.card_info_lut = lut
    cc = env.combo_cards
    out = np.full(cc.shape[0], -1, dtype=np.int64)
    for i in np.flatnonzero(valid_mask):
        out[i] = env.cluster_for((int(cc[i, 0]), int(cc[i, 1])))
    return out


@pytest.mark.parametrize("street", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_batch_matches_scalar_cluster_for(lut, street, seed):
    env = _env_at(street, seed)
    assert env.betting_round == street
    cc = env.combo_cards
    board = np.asarray([int(c) for c in env.community_cards], dtype=np.int64)
    valid = ~(np.isin(cc[:, 0], board) | np.isin(cc[:, 1], board))

    expected = _scalar_reference(env, lut, valid)
    got = clusters_for_board(lut[_STREETS[street]], cc, board, valid)

    assert got.dtype == np.int64
    assert np.array_equal(got, expected), (
        f"street={_STREETS[street]} seed={seed}: "
        f"{int((got != expected).sum())} of {len(cc)} combos disagree"
    )
    # The test is only meaningful if real ids came back, not all sentinel.
    assert (got >= 0).sum() == int(valid.sum()) > 0


def test_conflicting_combos_get_sentinel(lut):
    """Combos sharing a card with the board must be -1, never a gathered row."""
    env = _env_at(3, seed=0)
    cc = env.combo_cards
    board = np.asarray([int(c) for c in env.community_cards], dtype=np.int64)

    got = clusters_for_board(lut["river"], cc, board)  # valid_mask derived

    conflict = np.isin(cc[:, 0], board) | np.isin(cc[:, 1], board)
    assert conflict.any(), "fixture must contain board-conflicting combos"
    assert np.all(got[conflict] == -1)
    assert np.all(got[~conflict] >= 0)


def test_derived_mask_matches_explicit_mask(lut):
    env = _env_at(2, seed=1)
    cc = env.combo_cards
    board = np.asarray([int(c) for c in env.community_cards], dtype=np.int64)
    valid = ~(np.isin(cc[:, 0], board) | np.isin(cc[:, 1], board))
    assert np.array_equal(
        clusters_for_board(lut["turn"], cc, board),
        clusters_for_board(lut["turn"], cc, board, valid),
    )


def test_board_folds_into_cluster_id(lut):
    """The premise of the redesign: a different runout => a different id.

    This is what lets future-street tables be keyed on the cluster alone, with
    no explicit river axis.  Asserted here so the property is pinned rather than
    assumed by the vector regime.
    """
    env = _env_at(2, seed=0)
    cc = env.combo_cards
    turn_board = [int(c) for c in env.community_cards]
    rivers = sorted({int(c) for c in np.unique(cc)} - set(turn_board))

    seen = set()
    for r in rivers:
        board = np.asarray(turn_board + [r], dtype=np.int64)
        ids = clusters_for_board(lut["river"], cc, board)
        seen.add(tuple(ids.tolist()))
    assert len(seen) > 1, "river card must change the cluster assignment"


def test_dict_street_dispatches_to_scalar_fallback(lut):
    """A plain-dict street (any LUT built without a memmap) must still work."""
    env = _env_at(1, seed=0)
    env.card_info_lut = lut
    cc = env.combo_cards
    board = np.asarray([int(c) for c in env.community_cards], dtype=np.int64)
    valid = ~(np.isin(cc[:, 0], board) | np.isin(cc[:, 1], board))

    entry = lut["flop"]
    if not isinstance(entry, MemmapLookup):
        pytest.skip("flop street is already a dict; covered by the parametrised test")

    # Materialise this street as a plain dict over the reachable keys, then
    # check the fallback branch agrees with the memmap branch.
    as_dict = {}
    for i in np.flatnonzero(valid):
        hole = sorted((int(cc[i, 0]), int(cc[i, 1])))
        key = tuple(hole + sorted(int(c) for c in board))
        as_dict[key] = entry[key]

    assert np.array_equal(
        clusters_for_board(as_dict, cc, board, valid),
        clusters_for_board(entry, cc, board, valid),
    )


def test_survives_pickle_roundtrip(lut):
    """_idx_cache is dropped on pickle; the batch path must rebuild it."""
    import pickle

    env = _env_at(3, seed=2)
    cc = env.combo_cards
    board = np.asarray([int(c) for c in env.community_cards], dtype=np.int64)
    entry = lut["river"]
    if not isinstance(entry, MemmapLookup):
        pytest.skip("river street is not a MemmapLookup in this LUT")

    before = clusters_for_board(entry, cc, board)
    revived = pickle.loads(pickle.dumps(entry))
    assert np.array_equal(clusters_for_board(revived, cc, board), before)
