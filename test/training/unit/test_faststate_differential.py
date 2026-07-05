"""Phase-2 differential gate for the compiled betting engine (``_state.FastState``).

The Cython ``FastState`` must be byte-for-byte equivalent to ``PokerEnv`` on the
blueprint contract — ``player_i`` / ``legal_actions`` / ``is_terminal`` /
``info_set`` bytes / ``payout`` at **every** node — with make/undo exact under
arbitrary nesting.  These tests certify that three ways, all against the live
``PokerEnv`` as the source of truth:

1. **Three-way random-line walk** — drive ``PokerEnv``, the pure-Python spec
   ``FastStateRef``, and the Cython ``FastState`` down the *same* random legal
   action line, asserting all three agree at every node (and, incidentally,
   validating ``FastStateRef`` itself as the transliteration oracle).
2. **Recursive full-subtree DFS** — walk the entire (small) game tree the way
   ``_traverse`` does (step every legal action, recurse, undo), asserting the C
   POD undo stack restores an *identical* snapshot after deep nested recursion.
3. **Named edge fixtures** — heads-up post-flop order, over-the-top all-in
   response, fold terminals, and multiway side-pot / chop payouts.

Skips cleanly when the compiled extension is not built.
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
        MAX_RAISES_PER_ROUND,
    )
    from poker_ai._core._state_ref import FastStateRef
    from poker_ai._core import _state as _cystate

    # Dump the live alphabet + raise grid into the compiled engine (never
    # hard-coded — same anti-drift discipline as the Phase-1 kernels).
    _cystate.configure(_STAGE_ID, _ACTION_BYTE, RAISE_SIZES_BY_STAGE, MAX_RAISES_PER_ROUND)
    FastState = _cystate.FastState


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
class _ClusterStage:
    """Deterministic pseudo-cluster per (hole+board) key — varies the cluster
    varint width so the info-set encoding is exercised beyond a constant 0."""

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


def _ref_legal(ref):
    return [a for a in ref.legal_actions() if a is not None]


def _cy_legal(cs):
    return [a for a in cs.legal_actions() if a is not None]


def _new_env(n_players, chips, seed, **kwargs):
    np.random.seed(seed)
    random.seed(seed)
    env = PokerEnv(players=[Player(i, chips) for i in range(n_players)], **kwargs)
    env.card_info_lut = _StubLUT()
    return env


def _assert_node_agrees(env, ref, cs):
    """All three engines agree on the read-only contract at the current node."""
    term = env.is_terminal
    assert term == ref.is_terminal == cs.is_terminal, "is_terminal"
    if term:
        env_payout = dict(env.payout)
        assert env_payout == ref.payout() == cs.payout(), "payout"
        return True
    assert env.player_i == ref.player_i == cs.player_i, "player_i"
    assert env.betting_round == ref.betting_round == cs.betting_round, "betting_round"
    le = _env_legal(env)
    assert le == _ref_legal(ref) == _cy_legal(cs), "legal_actions"
    assert env.info_set == ref.info_set() == cs.info_set(), "info_set"
    return False


# ---------------------------------------------------------------------------
# 1. Three-way random-line walk (broad coverage + make/undo per node)
# ---------------------------------------------------------------------------
_CONFIGS = [
    (2, 10000), (2, 300), (2, 150), (2, 100),
    (3, 10000), (3, 400), (3, 250),
    (4, 10000), (4, 500),
    (6, 10000), (6, 500),
]


@pytest.mark.parametrize("n_players,chips", _CONFIGS)
def test_random_lines_three_way_match(n_players, chips):
    """PokerEnv, FastStateRef and FastState agree at every node of many random
    legal action lines, with a make/undo identity check at each step."""
    for seed in range(120):
        env = _new_env(n_players, chips, seed)
        ref = FastStateRef.from_poker_env(env)
        cs = FastState.from_poker_env(env)
        rng = random.Random(seed * 7 + 3)
        for _ in range(600):
            if _assert_node_agrees(env, ref, cs):
                break
            le = _env_legal(env)
            action = le[rng.randrange(len(le))]
            # make/undo is exact (snapshot before == after undo).
            snap = cs.snapshot()
            tok = cs.step_in_place(action)
            cs.undo(tok)
            assert cs.snapshot() == snap, (n_players, chips, seed, "undo identity")
            env.step_in_place(action)
            ref.step_in_place(action)
            cs.step_in_place(action)
        else:
            pytest.fail(f"runaway line (no terminal in 600 steps) seed={seed}")


# ---------------------------------------------------------------------------
# 2. Recursive full-subtree DFS (deep nested make/undo on the C POD stack)
# ---------------------------------------------------------------------------
def _dfs(env, ref, cs, budget):
    """Walk the full subtree the way ``_traverse`` does; assert undo after deep
    recursion restores an identical FastState snapshot.  ``budget`` bounds nodes."""
    if _assert_node_agrees(env, ref, cs):
        return budget - 1
    for action in _env_legal(env):
        if budget <= 0:
            return budget
        snap = cs.snapshot()
        et = env.step_in_place(action)
        rt = ref.step_in_place(action)
        ct = cs.step_in_place(action)
        budget = _dfs(env, ref, cs, budget)
        env.undo(et)
        ref.undo(rt)
        cs.undo(ct)
        assert cs.snapshot() == snap, "undo after recursion not identity"
    return budget


@pytest.mark.parametrize(
    "n_players,chips,big_blind",
    [(2, 200, 100), (2, 300, 100), (3, 200, 100)],
)
def test_full_subtree_dfs_make_undo(n_players, chips, big_blind):
    """Exhaustively walk a small game tree; the compiled engine matches the
    oracles at every node and the C undo stack unwinds deep recursion exactly."""
    for seed in range(15):
        env = _new_env(n_players, chips, seed, big_blind=big_blind)
        ref = FastStateRef.from_poker_env(env)
        cs = FastState.from_poker_env(env)
        remaining = _dfs(env, ref, cs, budget=40000)
        assert remaining <= 40000


# ---------------------------------------------------------------------------
# 3. Named edge fixtures
# ---------------------------------------------------------------------------
def _fs(env):
    return FastState.from_poker_env(env)


def test_heads_up_postflop_order_matches():
    """Heads-up: SB acts first pre-flop, BB leads post-flop — the compiled engine
    reproduces the corrected action order."""
    env = _new_env(2, 10000, seed=1)
    cs = _fs(env)
    assert env.player_i == cs.player_i == 0  # SB / button acts first pre-flop
    for _ in range(2):  # call down to the flop
        env.step_in_place("call")
        cs.step_in_place("call")
    assert env.betting_round == cs.betting_round == 1
    assert env.player_i == cs.player_i == 1  # BB leads post-flop


def test_over_the_top_all_in_response_matches():
    """An over-the-top all-in is not terminal until the opponent responds, and the
    responder may only call/fold (no raise) — engine mirrors the env."""
    env = _new_env(2, 10000, seed=2, small_blind=50, big_blind=100)
    cs = _fs(env)
    env.step_in_place("raise:1.0")
    cs.step_in_place("raise:1.0")
    env.step_in_place("all_in")
    cs.step_in_place("all_in")
    assert not env.is_terminal and not cs.is_terminal
    assert env.player_i == cs.player_i
    assert _env_legal(env) == _cy_legal(cs)
    assert not any(a.startswith("raise:") for a in _cy_legal(cs))


def test_fold_terminal_payout_matches():
    """Folding to a shove forfeits only the committed chips — payout matches."""
    env = _new_env(2, 10000, seed=3)
    cs = _fs(env)
    env.step_in_place("all_in")
    cs.step_in_place("all_in")
    env.step_in_place("fold")
    cs.step_in_place("fold")
    assert env.is_terminal and cs.is_terminal
    assert dict(env.payout) == cs.payout()


def test_called_all_in_full_stack_payout_matches():
    """A called all-in contests full stacks — zero-sum payout matches the env."""
    env = _new_env(2, 10000, seed=4)
    cs = _fs(env)
    env.step_in_place("all_in")
    cs.step_in_place("all_in")
    env.step_in_place("all_in")  # opponent calls the shove
    cs.step_in_place("all_in")
    assert env.is_terminal and cs.is_terminal
    assert dict(env.payout) == cs.payout()


@pytest.mark.parametrize("seed", range(40))
def test_multiway_allin_sidepot_payout_matches(seed):
    """Multiway all-in with UNEQUAL stacks exercises side pots / chops — the
    compiled ``payout`` matches the env chip-for-chip.

    Unequal starting stacks are what make side pots possible, and also what the
    ``payout`` per-seat-initial fix is about: ``PokerEnv.payout`` now nets each
    seat against its own start (zero-sum), so it is a valid ground truth here and
    this doubles as a regression for that env fix.
    """
    stacks = [150, 400, 900]
    np.random.seed(seed)
    random.seed(seed)
    env = PokerEnv(players=[Player(i, stacks[i]) for i in range(3)],
                   small_blind=25, big_blind=50)
    env.card_info_lut = _StubLUT()
    ref = FastStateRef.from_poker_env(env)
    cs = _fs(env)
    rng = random.Random(seed)
    guard = 0
    while not env.is_terminal and guard < 60:
        le = _env_legal(env)
        assert le == _ref_legal(ref) == _cy_legal(cs)
        # Bias toward all_in/call so we reach multiway all-in showdowns often.
        pool = [a for a in le if a in ("all_in", "call")] or le
        action = pool[rng.randrange(len(pool))]
        env.step_in_place(action)
        ref.step_in_place(action)
        cs.step_in_place(action)
        guard += 1
    assert env.is_terminal == cs.is_terminal
    if env.is_terminal:
        env_payout = dict(env.payout)
        assert sum(env_payout.values()) == 0  # per-seat netting is zero-sum
        assert cs.payout() == ref.payout() == env_payout


def test_genuine_sidepot_payout_matches():
    """A deterministic all-in line with unequal stacks reaches a REAL side pot
    (>=2 eligible seats committed for different amounts) and the compiled payout
    matches the env's concrete side-pot settlement chip-for-chip.

    Uses ``terminal_contributions`` (captured before ``compute_winners`` resets
    the pot) to *prove* a genuine side pot was reached — a plain post-terminal
    ``pot._chips`` read is all zeros and would make this assertion vacuous.
    """
    stacks = [150, 400, 900]
    reached = 0
    for seed in range(60):
        np.random.seed(seed)
        random.seed(seed)
        env = PokerEnv(players=[Player(i, stacks[i]) for i in range(3)],
                       small_blind=25, big_blind=50)
        env.card_info_lut = _StubLUT()
        ref = FastStateRef.from_poker_env(env)
        cs = _fs(env)
        guard = 0
        while not env.is_terminal and guard < 40:
            le = _env_legal(env)
            assert le == _ref_legal(ref) == _cy_legal(cs)
            # Full-stack commitments (not just calling the short shove) create
            # the unequal contributions a side pot needs.
            action = "all_in" if "all_in" in le else (
                "call" if "call" in le else le[0])
            env.step_in_place(action)
            ref.step_in_place(action)
            cs.step_in_place(action)
            guard += 1
        if not env.is_terminal:
            continue
        tc = env.terminal_contributions
        active = [i for i in range(3) if env.players[i].is_active]
        if tc and len(active) >= 2 and len({c for c in tc if c > 0}) > 1:
            reached += 1
        env_payout = dict(env.payout)
        assert sum(env_payout.values()) == 0
        assert cs.payout() == ref.payout() == env_payout
    assert reached > 0, "no genuine side pot reached — test would be vacuous"


def test_deep_line_undo_across_stack_realloc():
    """A long 6-max line grows the C undo stack past its initial capacity; the
    whole line then unwinds in LIFO order to a byte-identical snapshot at every
    frame, proving the ``realloc`` growth + restore path is correct."""
    for seed in range(20):
        np.random.seed(seed)
        random.seed(seed)
        env = PokerEnv(players=[Player(i, 100000) for i in range(6)],
                       small_blind=25, big_blind=50)
        env.card_info_lut = _StubLUT()
        cs = _fs(env)
        rng = random.Random(seed * 5 + 1)
        toks, snaps = [], []
        while not env.is_terminal and len(toks) < 200:
            le = _env_legal(env)
            assert le == _cy_legal(cs)
            action = le[rng.randrange(len(le))]
            snaps.append(cs.snapshot())
            toks.append(cs.step_in_place(action))
            env.step_in_place(action)
        # Unwind the entire line; each undo must restore its pre-step snapshot.
        for t in reversed(toks):
            cs.undo(t)
            assert cs.snapshot() == snaps[t], "undo across realloc not identity"


def test_invalid_action_raises_and_leaves_state_unchanged():
    """Guard rails must FAIL LOUD, not silently swallow (the ``except *`` fix):
    an unrecognised / malformed action raises and leaves the state untouched
    (no half-mutation), and ``None`` at an active seat asserts."""
    env = _new_env(2, 10000, seed=7)
    cs = _fs(env)
    for bad, exc in [("bogus", ValueError), ("raise:notafloat", ValueError),
                     (None, AssertionError)]:
        snap = cs.snapshot()
        with pytest.raises(exc):
            cs.step_in_place(bad)
        assert cs.snapshot() == snap, f"{bad!r} left state half-mutated"


def test_configure_is_installed():
    assert _cystate.is_configured()
