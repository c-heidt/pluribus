"""Phase-3a gate: ``FastState`` public-state + depth surface vs ``PokerEnv``.

The subgame solver keys every ``SolverState`` node (and the golden digest) off
``env.public_key`` and classifies depth off ``env.betting_round`` /
``n_raises_this_round`` / ``n_players_started_round``.  Phase 3 drives the walk on
a ``FastState`` instead of a ``PokerEnv``, so these must be **byte-for-byte
identical** on canonical (on-tree) histories — the only ones a FastState-driven
solve ever sees (off-tree injected raises fall back to the Python walk).

Reuses the Phase-2 differential harness pattern (``test_faststate_differential``):
drive ``PokerEnv`` and ``FastState`` down the same random legal line and assert
agreement at every node, plus ``terminal_board_len`` at each terminal (the fold
path's board-removal length).  Skips cleanly when the core is not built.
"""

import random

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


class _ClusterStage:
    def __getitem__(self, key):
        acc = 0
        for c in key:
            acc = (acc * 1000003 + int(c)) & 0xFFFFFFFF
        return acc % 70000


class _StubLUT(dict):
    def __missing__(self, stage):
        s = _ClusterStage()
        self[stage] = s
        return s


def _env_legal(env):
    return [a for a in env.legal_actions if a is not None]


def _new_env(n_players, chips, seed, **kwargs):
    np.random.seed(seed)
    random.seed(seed)
    env = PokerEnv(players=[Player(i, chips) for i in range(n_players)], **kwargs)
    env.card_info_lut = _StubLUT()
    return env


def _assert_public_agrees(env, cs):
    """PokerEnv and FastState agree on the search public-state surface."""
    if env.is_terminal:
        assert cs.is_terminal
        # ``terminal_board_len`` is captured before the force-deal (fold path).
        assert env.terminal_board_len == cs.terminal_board_len, "terminal_board_len"
        return True
    assert not cs.is_terminal
    # public_key: equal AND repr-equal (the golden digest hashes ``repr(key)``);
    # equal tuples-of-strings always hash-equal, so this is the digest guarantee.
    epk, cpk = env.public_key, cs.public_key
    assert epk == cpk, f"public_key: {epk!r} != {cpk!r}"
    assert repr(epk) == repr(cpk), "public_key repr"
    assert hash(epk) == hash(cpk), "public_key hash"
    assert env.betting_round == cs.betting_round, "betting_round"
    assert env.n_raises_this_round == cs.n_raises_this_round, "n_raises_this_round"
    assert (
        env.n_players_started_round == cs.n_players_started_round
    ), "n_players_started_round"
    return False


_CONFIGS = [
    (2, 10000), (2, 300), (2, 150), (2, 100),
    (3, 10000), (3, 400), (3, 250),
    (4, 10000), (4, 500),
    (6, 10000), (6, 500),
]


@pytest.mark.parametrize("n_players,chips", _CONFIGS)
def test_public_surface_random_lines(n_players, chips):
    """Over many random legal lines, ``FastState``'s public_key / depth attrs /
    terminal_board_len match ``PokerEnv`` at every node."""
    for seed in range(120):
        env = _new_env(n_players, chips, seed)
        cs = FastState.from_poker_env(env)
        rng = random.Random(seed * 7 + 3)
        for _ in range(600):
            if _assert_public_agrees(env, cs):
                break
            le = _env_legal(env)
            action = le[rng.randrange(len(le))]
            env.step_in_place(action)
            cs.step_in_place(action)
        else:
            pytest.fail(f"runaway line (no terminal in 600 steps) seed={seed}")


def _dfs(env, cs, budget):
    if _assert_public_agrees(env, cs):
        return budget - 1
    for action in _env_legal(env):
        if budget <= 0:
            return budget
        et = env.step_in_place(action)
        ct = cs.step_in_place(action)
        budget = _dfs(env, cs, budget)
        env.undo(et)
        cs.undo(ct)
    return budget


@pytest.mark.parametrize(
    "n_players,chips,big_blind", [(2, 200, 100), (2, 300, 100), (3, 200, 100)]
)
def test_public_surface_full_subtree(n_players, chips, big_blind):
    """Exhaustive small-tree DFS: public surface matches at every node, and undo
    restores ``terminal_board_len`` / depth state so siblings compare correctly."""
    for seed in range(15):
        env = _new_env(n_players, chips, seed, big_blind=big_blind)
        cs = FastState.from_poker_env(env)
        remaining = _dfs(env, cs, budget=40000)
        assert remaining <= 40000


def test_terminal_board_len_undo_reverts():
    """After a terminal step + undo, a non-terminal sibling sees ``None`` again —
    ``terminal_board_len`` is snapshotted in the make/undo frame."""
    env = _new_env(2, 10000, seed=2, small_blind=50, big_blind=100)
    cs = FastState.from_poker_env(env)
    assert cs.terminal_board_len is None
    # A fold terminal at pre-flop → board len 0, then undo restores None.
    tok = cs.step_in_place("fold")
    assert cs.is_terminal and cs.terminal_board_len == 0
    cs.undo(tok)
    assert not cs.is_terminal and cs.terminal_board_len is None


def test_public_key_includes_skip_tokens():
    """A line where an all-in seat is skipped on a later street puts ``"skip"``
    tokens in the history — public_key must carry them identically.

    A shove-then-call preflop leaves the shover all-in; on the flop that seat is
    skipped (``skip_counter``), and the skip is prepended to the next actor's
    history entry — the exact padding ``PokerEnv`` records and FastState encodes.
    """
    found = False
    for seed in range(200):
        env = _new_env(6, 10000, seed, small_blind=25, big_blind=50)
        cs = FastState.from_poker_env(env)
        rng = random.Random(seed)
        steps = 0
        while not env.is_terminal and steps < 60:
            le = _env_legal(env)
            action = le[rng.randrange(len(le))]
            env.step_in_place(action)
            cs.step_in_place(action)
            steps += 1
            if env.is_terminal:
                break
            # Ground truth: when env's public_key carries a skip token, FastState
            # must reproduce it (== already asserted broadly; here we require the
            # skip case to actually occur so the check is non-vacuous).
            if any("skip" in acts for _, acts in env.public_key[1]):
                assert env.public_key == cs.public_key, (
                    f"skip-history mismatch: {env.public_key!r} != {cs.public_key!r}"
                )
                found = True
        if found:
            break
    assert found, "no skip token constructed — test would be vacuous"
