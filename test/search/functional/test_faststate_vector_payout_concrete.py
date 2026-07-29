"""Phase-2 gate: ``FastState.vector_payout_concrete`` vs ``PokerEnv`` counterpart.

The traverser-vectorized MCCFR terminal / leaf settlement moved in-core (search
Phase 2).  ``FastState.vector_payout_concrete`` must be **byte-for-byte identical**
to ``PokerEnv.vector_payout_concrete`` at every terminal: both derive per-combo
feasibility + rank the traverser's combos + rank each concrete opponent, then call
the *same* module-level ``_settle_traverser`` (core-backed under ``settle_concrete``).
The differential proves the FastState field-source derivation (holes ← ``hole``,
board ← ``board``, contributions ← ``pot_chips``, active flags, bet ``order``)
matches the env's.

Drive a ``FastState`` and the env down the same random line to a terminal with
``settle_winners=False``, then compare the per-combo vector for every seat, across
heads-up and 3-way (dead money + genuine side pots), showdown and fold terminals,
equal and unequal stacks.  Skips if the core is absent.
"""

import collections

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
    """Step ``env`` (settle_winners=False) and ``cs`` down one random legal line;
    biased to all_in/call/fold so showdowns AND folds are reached often."""
    rng = np.random.RandomState(seed * 131 + 7)
    for _ in range(40):
        if env.is_terminal:
            return True
        legal = [a for a in env.legal_actions if a is not None]
        weighted = []
        for a in legal:
            weighted += [a] * (3 if a in ("all_in", "call", "fold") else 1)
        action = weighted[rng.randint(len(weighted))]
        env.step_in_place(action, settle_winners=False)
        cs.step_in_place(action)
    return env.is_terminal


_STACKS = [
    (300, 300), (150, 400),                # HU equal + unequal
    (600, 600, 600), (150, 400, 900),      # 3-way: dead money + side pots
]


@pytest.mark.parametrize("stacks", _STACKS)
def test_vector_payout_concrete_matches_env(stacks):
    """Over many random terminals, ``FastState.vector_payout_concrete`` equals
    ``PokerEnv.vector_payout_concrete`` byte-for-byte, for every seat."""
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
        active = [s for s in range(n) if env.players[s].is_active]
        if len(active) >= 2:
            showdowns += 1
        else:
            folds += 1
        for seat in range(n):
            ev = env.vector_payout_concrete(seat)
            cv = cs.vector_payout_concrete(seat, combo_cards)
            assert ev.dtype == cv.dtype == np.float64
            assert ev.shape == cv.shape == (env.n_combos,)
            assert np.array_equal(ev, cv), (
                f"stacks={stacks} seed={seed} seat={seat}: "
                f"{ev.tolist()[:6]} != {cv.tolist()[:6]}"
            )
            compared += 1
    assert compared > 0, "no terminals reached — test vacuous"
    assert showdowns > 0 and folds > 0, (
        f"need both showdown and fold terminals (showdowns={showdowns}, folds={folds})"
    )


def test_vector_payout_concrete_requires_terminal():
    """Mirrors the env precondition: not defined off a terminal node."""
    env = _root(0, (300, 300))
    cs = FastState.from_poker_env(env)
    assert not cs.is_terminal
    with pytest.raises(ValueError):
        cs.vector_payout_concrete(0, env.combo_cards)
