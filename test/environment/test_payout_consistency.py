"""The env's terminal-payout evaluators must agree with each other.

The environment owns three terminal payouts that share the one
``default_evaluator``: ``PokerEnv.payout`` (a single dealt hand),
``runout_equity`` (the board-average), and ``vector_payout`` (range-vs-range,
per-combo).  This test cross-validates the **vectorized** payout against the
**concrete** settlement to the chip: at a terminal, ``vector_payout`` against a
*one-hot* opponent reach must equal the engine's actual net chips when the two
concrete hands are dealt and the same betting line is replayed — across
showdown / fold / all-in terminals and equal *and* unequal stacks (the latter
exercises the matched-stake / uncalled-excess handling).
"""

import collections
import copy

import numpy as np
import pytest

from environment import range_showdown
from environment.player import Player
from environment.poker_env import PokerEnv


def _stub_lut(env):
    env.card_info_lut = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0)
    )


def _river_env(seed, stacks):
    np.random.seed(seed)
    env = PokerEnv(players=[Player(i, s) for i, s in enumerate(stacks)],
                   low_card_rank=11, high_card_rank=14)
    _stub_lut(env)
    g = 0
    while env.betting_round < 3 and not env.is_terminal and g < 60:
        env.step_in_place("call" if "call" in env.legal_actions else "check")
        g += 1
    return env


def _drive(env, policy):
    """Play a copy of ``env`` to a terminal under ``policy``; return the line."""
    e = copy.deepcopy(env)
    acts, g = [], 0
    while not e.is_terminal and g < 12:
        legal = [a for a in e.legal_actions if a]
        a = policy(legal)
        if a is None:
            return None
        acts.append(a)
        e.step_in_place(a)
        g += 1
    return acts if e.is_terminal else None


def _walk_line(env, line):
    e = copy.deepcopy(env)
    for a in line:
        if e.is_terminal:
            break
        if a not in [x for x in e.legal_actions if x]:
            return None
        e.step_in_place(a)
    return e if e.is_terminal else None


def _engine_net(root, line, i, j, p, stacks):
    e = copy.deepcopy(root)
    other = 1 - p
    holes = [None, None]
    holes[p] = (int(i[0]), int(i[1]))
    holes[other] = (int(j[0]), int(j[1]))
    e = e.with_hole_cards(holes)
    for a in line:
        if e.is_terminal:
            break
        e.step_in_place(a)
    return e.players[p].n_chips - stacks[p]


_LINES = {
    "passive": lambda la: "check" if "check" in la else ("call" if "call" in la else None),
    "fold": lambda la: "fold" if "fold" in la else ("check" if "check" in la else ("call" if "call" in la else None)),
    "allin": lambda la: "all_in" if "all_in" in la else ("call" if "call" in la else ("check" if "check" in la else None)),
}


@pytest.mark.parametrize("stacks", [(300, 300), (150, 600)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vector_payout_matches_concrete_settlement(stacks, seed):
    root = _river_env(seed, stacks)
    assert root.betting_round == 3
    board = set(int(c) for c in root.community_cards)
    cc = root.combo_cards
    valid = [k for k in range(cc.shape[0])
             if int(cc[k, 0]) not in board and int(cc[k, 1]) not in board]
    rng = np.random.default_rng(seed)
    acting = valid if len(valid) <= 12 else list(rng.choice(valid, 12, replace=False))

    checked_lines = 0
    for policy in _LINES.values():
        line = _drive(root, policy)
        if line is None:
            continue
        term = _walk_line(root, line)
        if term is None:
            continue
        checked_lines += 1
        opp = np.zeros(root.n_combos)
        for i_idx in acting:
            i = cc[i_idx]
            for j_idx in valid:
                j = cc[j_idx]
                if set((int(i[0]), int(i[1]))) & set((int(j[0]), int(j[1]))):
                    continue
                opp[:] = 0.0
                opp[j_idx] = 1.0
                vij = term.vector_payout(0, 1, opp, river=None)[i_idx]
                eng = _engine_net(root, line, i, j, 0, stacks)
                assert abs(float(vij) - float(eng)) < 1e-9, (
                    f"combo {i_idx} vs {j_idx} on line {line}: "
                    f"vector={vij} engine={eng}"
                )
    assert checked_lines >= 2  # exercised multiple terminal types


def _turn_fold_env(seed, stacks):
    """Drive a heads-up hand to a **turn-side fold** terminal.

    Returns ``(env, turn_board)`` where ``turn_board`` is the four community
    cards on the board at the moment the fold happened.  The engine force-deals
    the river out afterward (so ``len(community_cards) == 5`` at the terminal),
    which is exactly why ``terminal_board_len`` is needed to recover the board
    the fold actually saw.
    """
    np.random.seed(seed)
    env = PokerEnv(players=[Player(i, s) for i, s in enumerate(stacks)],
                   low_card_rank=11, high_card_rank=14)
    _stub_lut(env)
    g = 0
    while env.betting_round < 2 and not env.is_terminal and g < 60:
        env.step_in_place("call" if "call" in env.legal_actions else "check")
        g += 1
    assert env.betting_round == 2
    turn_board = list(env.community_cards)
    legal = [a for a in env.legal_actions if a]
    raise_act = next((a for a in legal if a.startswith("raise")), None)
    assert raise_act is not None
    env.step_in_place(raise_act)
    env.step_in_place("fold")
    assert env.is_terminal and not env.is_decision_free
    return env, turn_board


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_turn_fold_value_is_river_independent(seed):
    """A turn-side fold's vectorised payout must not depend on the sampled river.

    Regression for the bug where ``vector_payout`` masked a *turn-side* fold by
    the search's sampled river (because the engine force-deals the board to five,
    ``len(community) == 5`` always held).  A fold that ended on the turn never saw
    a river, so its per-combo value is river-independent and must equal the card
    removal computed on the **four-card turn board**.
    """
    env, turn_board = _turn_fold_env(seed, (300, 300))
    assert len(env.community_cards) == 5          # engine force-dealt the river
    assert env.terminal_board_len == 4            # but the fold saw only the turn

    cc = env.combo_cards
    low, high = env.low_card_rank, env.high_card_rank
    removal = range_showdown.removal_for(low, high)
    stake = float(min(env.terminal_contributions))
    winner = 0 if env.players[0].is_active else 1
    opp_seat = 1 - winner

    # Uniform opponent reach over turn-compatible combos.
    _, valid_turn = range_showdown.ranked_board(low, high, turn_board)
    opp = np.where(valid_turn, 1.0, 0.0)
    opp /= opp.sum()

    # Correct, river-independent fold value to the winner: stake * (opponent
    # reach on combos sharing no card with the acting combo), on the turn board.
    avail = range_showdown.reach_after_removal(
        cc, np.where(valid_turn, opp, 0.0), removal
    )
    correct = stake * np.where(valid_turn, avail, 0.0)

    board_cards = set(int(c) for c in turn_board)
    rivers = [int(c) for c in np.unique(cc) if int(c) not in board_cards]
    assert len(rivers) >= 2
    for r in rivers:
        v = env.vector_payout(winner, opp_seat, opp, river=r)
        assert np.allclose(v, correct, atol=1e-9), (
            f"seed {seed}: sampled river {r} changed the turn-fold value "
            f"(max |Δ|={np.abs(v - correct).max():.4f}, stake={stake})"
        )
    # And it agrees with the river-subgame convention (river=None → turn board).
    assert np.allclose(
        env.vector_payout(winner, opp_seat, opp, river=None), correct, atol=1e-9
    )


def test_terminal_board_len_round_trips_through_make_undo():
    """``terminal_board_len`` survives ``step_in_place`` / ``undo`` (LIFO)."""
    env, _ = _turn_fold_env(0, (300, 300))
    # Re-drive to just before the fold so we can step into the terminal and back.
    np.random.seed(0)
    fresh = PokerEnv(players=[Player(i, 300) for i in range(2)],
                     low_card_rank=11, high_card_rank=14)
    _stub_lut(fresh)
    g = 0
    while fresh.betting_round < 2 and not fresh.is_terminal and g < 60:
        fresh.step_in_place("call" if "call" in fresh.legal_actions else "check")
        g += 1
    raise_act = next(a for a in fresh.legal_actions if a and a.startswith("raise"))
    fresh.step_in_place(raise_act)
    assert fresh.terminal_board_len is None       # not yet terminal
    token = fresh.step_in_place("fold")           # into the terminal
    assert fresh.is_terminal and fresh.terminal_board_len == 4
    fresh.undo(token)                             # back out
    assert not fresh.is_terminal and fresh.terminal_board_len is None
