"""Phase-3b gate: ``FastState.vector_payout`` vs ``PokerEnv.vector_payout``.

The vector regime's range-vs-range terminal settlement moved in-core (Phase 3).
``FastState.vector_payout`` must be **byte-for-byte identical** to
``PokerEnv.vector_payout`` at every terminal, for any (seat, opp, reach, river):
both derive stake / dead money / board and then call the *same* module-level
``range_showdown`` primitives, so the differential proves the field-source
derivation (contributions ← ``pot_chips``, board ← precomputed ``board``,
``terminal_board_len``, active flags) matches the env's.

Build a ``FastState`` from a **non-terminal** root (so ``pot_chips`` holds live
contributions, un-reset), drive it and the env down the same line to a terminal
with ``settle_winners=False`` (the vector regime's calling convention), then
compare across showdown / fold terminals, equal + unequal stacks, dead money
(a 3rd seat folded), and board-incompatible reach.  Skips if the core is absent.
"""

import collections
import copy

import numpy as np
import pytest

from poker_ai import _core

pytestmark = pytest.mark.skipif(
    not _core.CORE_AVAILABLE, reason="compiled core extension not built"
)

if _core.CORE_AVAILABLE:
    from environment.player import Player
    from environment.poker_env import (
        PokerEnv,
        _ACTION_BYTE,
        _STAGE_ID,
        RAISE_SIZES_BY_STAGE,
        max_raises_per_round,
    )
    from poker_ai._core import _state as _cystate

    # 4th arg is accepted for signature stability but no longer drives the cap
    # check — FastState derives max_raises_per_round(n_players) from its own
    # n_players (this module parametrizes over several player counts).
    _cystate.configure(
        _STAGE_ID, _ACTION_BYTE, RAISE_SIZES_BY_STAGE, max_raises_per_round(2)
    )
    FastState = _cystate.FastState


def _stub_lut(env):
    env.card_info_lut = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0)
    )


def _root(seed, stacks):
    np.random.seed(seed)
    env = PokerEnv(
        players=[Player(i, s) for i, s in enumerate(stacks)],
        low_card_rank=11, high_card_rank=14, small_blind=25, big_blind=50,
    )
    _stub_lut(env)
    return env


def _drive_to_terminal(env, cs, seed):
    """Step ``env`` (settle_winners=False, as the vector regime does) and ``cs``
    down one random legal line; return True iff a terminal was reached."""
    rng = np.random.RandomState(seed * 131 + 7)
    for _ in range(40):
        if env.is_terminal:
            return True
        legal = [a for a in env.legal_actions if a is not None]
        # Bias to all_in/call/fold so showdowns AND folds are reached often.
        weighted = []
        for a in legal:
            w = 3 if a in ("all_in", "call", "fold") else 1
            weighted += [a] * w
        action = weighted[rng.randint(len(weighted))]
        env.step_in_place(action, settle_winners=False)
        cs.step_in_place(action)
    return env.is_terminal


def _reach_variants(combo_cards, board, rng):
    n = combo_cards.shape[0]
    variants = [
        rng.random_sample(n),                       # dense random
        rng.random_sample(n) * (rng.random_sample(n) > 0.5),  # sparse
    ]
    # One-hot (a single concrete opponent hand) — the concrete-settlement case.
    oh = np.zeros(n)
    oh[rng.randint(n)] = 1.0
    variants.append(oh)
    # Board-masked (impossible combos zeroed) — card removal edge.
    bset = set(int(c) for c in board)
    mask = np.array(
        [0.0 if (int(a) in bset or int(b) in bset) else 1.0 for a, b in combo_cards]
    )
    variants.append(rng.random_sample(n) * mask)
    return variants


_STACKS = [
    (10000, 10000), (300, 300), (150, 400),      # HU equal + unequal
    (600, 600, 600), (150, 400, 900),            # 3-way (dead money possible)
]


@pytest.mark.parametrize("stacks", _STACKS)
def test_vector_payout_matches_env(stacks):
    """Over many random terminals, ``FastState.vector_payout`` equals
    ``PokerEnv.vector_payout`` byte-for-byte for random (seat, opp, reach, runout)."""
    n = len(stacks)
    compared = 0
    showdowns = 0
    folds = 0
    for seed in range(150):
        env = _root(seed, stacks)
        cs = FastState.from_poker_env(env)
        combo_cards = env.combo_cards
        if not _drive_to_terminal(env, cs, seed):
            continue
        assert cs.is_terminal
        rng = np.random.RandomState(seed * 997 + 1)
        board = [int(c) for c in list(env.community_cards)[:5]]
        cards = [int(c) for c in np.unique(combo_cards)]
        # The regime only samples rivers OFF the board (a fresh card completes it);
        # a river duplicating a board card is not a legal runout and crashes even
        # the env evaluator, so exclude them (as _VectorSolver._rivers does).
        board_set = set(board)
        non_board = [c for c in cards if c not in board_set]
        runouts = [None] + [
            non_board[rng.randint(len(non_board))]
            for _ in range(min(3, len(non_board)))
        ]
        # Two-card completions (a **flop** root's chance outcome): the pair must
        # be distinct and off-board, exactly as the regime's completion set is.
        if len(non_board) >= 2:
            for _ in range(2):
                pair = rng.choice(len(non_board), size=2, replace=False)
                runouts.append([non_board[int(pair[0])], non_board[int(pair[1])]])
        active = [s for s in range(n) if env.players[s].is_active]
        if len(active) == 2:
            showdowns += 1
        else:
            folds += 1
        # Any distinct (seat, opp) is a valid differential — vector_payout is a
        # pure function of the pair + reach + river; env vs FastState must agree.
        pairs = [(0, 1), (1, 0)]
        if n == 3:
            pairs += [(0, 2), (2, 1), (1, 2)]
        for seat, opp in pairs:
            for reach in _reach_variants(combo_cards, board, rng):
                for runout in runouts:
                    ev = env.vector_payout(seat, opp, reach, runout=runout)
                    cv = cs.vector_payout(seat, opp, reach, runout, combo_cards)
                    assert ev.dtype == cv.dtype == np.float64
                    assert np.array_equal(ev, cv), (
                        f"stacks={stacks} seed={seed} pair=({seat},{opp}) "
                        f"runout={runout}: {ev.tolist()} != {cv.tolist()}"
                    )
                    compared += 1
    assert compared > 0, "no terminals reached — test vacuous"
    assert showdowns > 0 and folds > 0, (
        f"need both showdown and fold terminals (showdowns={showdowns}, folds={folds})"
    )


def test_vector_payout_requires_terminal():
    """Mirrors the env precondition: not defined off a terminal node."""
    env = _root(0, (10000, 10000))
    cs = FastState.from_poker_env(env)
    assert not cs.is_terminal
    with pytest.raises(ValueError):
        cs.vector_payout(0, 1, np.ones(env.combo_cards.shape[0]), None, env.combo_cards)
