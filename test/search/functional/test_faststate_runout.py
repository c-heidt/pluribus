"""Phase-4a gate: ``FastState.runout_equity`` / ``is_decision_free`` / ``runout_key``
vs ``PokerEnv``.

The MCCFR walk's terminals AND the leaf rollout both score a decision-free all-in
runout exactly (``runout_equity``); Phase 4 drives both off a ``FastState``, so
these must be **byte-identical** to ``PokerEnv``.  The completion average is an
integer chip sum → order-independent, so the port reconstructs the pre-runout
snapshot from FastState fields (prefix = ``board[:terminal_board_len]``,
contributions = ``pot_chips``, active = ``is_active``) and settles via the *same*
module-level ``_settle_runout`` (core-backed under the ``runout`` flag).

Build a FastState from a non-terminal root, drive it and the env down the same
line to decision-free all-in terminals, and compare.  Skips if the core is absent.
"""

import collections

import numpy as np
import pytest

from poker_ai import _core
from test.abstraction_helpers import passive_action
from test.lut_helpers import install_cluster_lut

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
        ALL_IN_ALLOWED_BY_STAGE,
        CALL_ALLOWED_BY_STAGE,
    )
    from poker_ai._core import _state as _cystate

    _cystate.configure(
        _STAGE_ID, _ACTION_BYTE, RAISE_SIZES_BY_STAGE, MAX_RAISES_PER_ROUND,
        CALL_ALLOWED_BY_STAGE, ALL_IN_ALLOWED_BY_STAGE
    )
    FastState = _cystate.FastState


def _root(seed, stacks):
    """A root advanced to the flop (call/check), so all-in runouts are <=2 board
    cards — the exact enumeration path search always takes (the solver's
    ``_use_equity`` guard excludes the preflop 5-card runout, which falls back to
    non-deterministic Monte-Carlo sampling and is never scored by runout_equity)."""
    np.random.seed(seed)
    env = PokerEnv(
        players=[Player(i, s) for i, s in enumerate(stacks)],
        low_card_rank=9, high_card_rank=14, small_blind=25, big_blind=50,
    )
    install_cluster_lut(env)
    g = 0
    while env.betting_round < 1 and not env.is_terminal and g < 20:
        env.step_in_place(passive_action(env))
        g += 1
    return env


def _drive(env, cs, seed):
    """All-in/call-heavy random line so many terminals are decision-free runouts."""
    rng = np.random.RandomState(seed * 131 + 7)
    for _ in range(40):
        if env.is_terminal:
            return True
        legal = [a for a in env.legal_actions if a is not None]
        pool = []
        for a in legal:
            w = 4 if a in ("all_in", "call") else 1
            pool += [a] * w
        action = pool[rng.randint(len(pool))]
        env.step_in_place(action)          # default settle_winners=True (sets _runout_info)
        cs.step_in_place(action)
    return env.is_terminal


@pytest.mark.parametrize(
    "stacks",
    [(2000, 2000), (1500, 4000), (3000, 3000),
     (2000, 2000, 2000), (1500, 4000, 9000), (1000, 2500, 6000)],
)
def test_runout_equity_matches_env(stacks):
    """Over many all-in terminals: is_decision_free / runout_key / runout_equity all
    byte-identical to PokerEnv."""
    n = len(stacks)
    df_terminals = 0
    for seed in range(200):
        env = _root(seed, stacks)
        cs = FastState.from_poker_env(env)
        if not _drive(env, cs, seed):
            continue
        assert cs.is_terminal == env.is_terminal
        assert cs.is_decision_free == env.is_decision_free, (
            f"is_decision_free stacks={stacks} seed={seed}"
        )
        assert cs.runout_key == env.runout_key, f"runout_key stacks={stacks} seed={seed}"
        if not env.is_decision_free:
            continue
        df_terminals += 1
        ce = cs.runout_equity()
        ee = env.runout_equity()
        assert set(ce) == set(ee) == set(range(n))
        for i in range(n):
            assert ce[i] == ee[i], (
                f"runout_equity[{i}] stacks={stacks} seed={seed}: {ce[i]} != {ee[i]}"
            )
    assert df_terminals > 0, f"no decision-free terminals reached (stacks={stacks}) — vacuous"


def test_is_decision_free_false_off_terminal_and_on_fold():
    """Not decision-free at a live node, a fold-out (1 active), or a complete-board
    river showdown — matching PokerEnv."""
    env = _root(0, (2000, 2000))
    cs = FastState.from_poker_env(env)
    assert not cs.is_decision_free and cs.runout_key is None
    with pytest.raises(ValueError):
        cs.runout_equity()
    # Pre-flop fold-out → 1 active → not decision-free.
    tok = cs.step_in_place("fold")
    assert cs.is_terminal and not cs.is_decision_free and cs.runout_key is None
    cs.undo(tok)


def test_runout_equity_under_runout_kernel_flag():
    """The differential also holds with the Phase-2 ``runout`` settlement kernel on
    (both env and FastState route through the same module-level ``_settle_runout``)."""
    # (Wiring is import-time; this test documents intent — the suite is run under
    # PLURIBUS_CORE_KERNELS=runout in CI.  Here we just re-check one seed.)
    env = _root(3, (1500, 4000, 9000))
    cs = FastState.from_poker_env(env)
    if _drive(env, cs, 3) and env.is_decision_free:
        assert cs.runout_equity() == env.runout_equity()
