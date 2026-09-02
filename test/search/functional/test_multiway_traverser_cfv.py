"""Phase 0 gate: ``PokerEnv.vector_payout_concrete`` is bit-exact with the
concrete settlement.

The traverser-vectorized MCCFR terminal values every combo of one seat's hand
against **concrete** (single sampled) opponent holds at once.  This must agree to
the chip with the engine's own ``payout`` when that seat is actually dealt each
combo and the same betting line is replayed — across heads-up and multiway
showdowns, fold terminals, dead money (folded third seat), and a short-stack
all-in that forms a genuine side pot.

The board is frozen by rooting every scenario on the **river** (all five
community cards already dealt, so a hole swap cannot change the board), mirroring
``test_payout_consistency``'s ``_engine_net`` oracle.
"""

import collections
import copy

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv
from test.abstraction_helpers import passive_action
from test.lut_helpers import install_cluster_lut


def _river_env(seed, stacks, low=11, high=14):
    """Play to the river root (5-card board dealt) under passive calls."""
    np.random.seed(seed)
    env = PokerEnv(players=[Player(i, s) for i, s in enumerate(stacks)],
                   low_card_rank=low, high_card_rank=high)
    install_cluster_lut(env)
    g = 0
    while env.betting_round < 3 and not env.is_terminal and g < 80:
        env.step_in_place(passive_action(env))
        g += 1
    return env


def _river_env_folded_seat(seed, stacks=(6000, 6000, 6000), low=11, high=14):
    """3-player river root with seat 2 folded on the flop (dead money in the pot)."""
    np.random.seed(seed)
    env = PokerEnv(players=[Player(i, s) for i, s in enumerate(stacks)],
                   low_card_rank=low, high_card_rank=high)
    install_cluster_lut(env)
    g = 0
    while env.betting_round < 1 and not env.is_terminal and g < 80:
        env.step_in_place(passive_action(env))
        g += 1
    if env.betting_round != 1 or env.is_terminal:
        return None
    raised = False
    g = 0
    while env.betting_round == 1 and not env.is_terminal and g < 20:
        legal = [a for a in env.legal_actions if a]
        actor = env.player_i
        if actor == 2 and raised and "fold" in legal:
            env.step_in_place("fold")
        elif actor != 2 and not raised:
            raise_act = next((a for a in legal if a.startswith("raise")), None)
            if raise_act is None:
                return None
            env.step_in_place(raise_act)
            raised = True
        else:
            env.step_in_place(passive_action(env))
        g += 1
    if env.players[2].is_active:
        return None
    g = 0
    while env.betting_round < 3 and not env.is_terminal and g < 20:
        env.step_in_place(passive_action(env))
        g += 1
    if env.betting_round != 3 or env.is_terminal:
        return None
    return env


def _drive(env, policy):
    e = copy.deepcopy(env)
    acts, g = [], 0
    while not e.is_terminal and g < 16:
        legal = [a for a in e.legal_actions if a]
        a = policy(legal)
        if a is None:
            return None
        acts.append(a)
        e.step_in_place(a)
        g += 1
    return acts if e.is_terminal else None


def _terminal_via(root, line, holes):
    """Replay ``line`` from ``root`` with the given per-seat holes; env at terminal."""
    e = copy.deepcopy(root)
    e = e.with_hole_cards([holes[s] for s in range(root.n_players)])
    for a in line:
        if e.is_terminal:
            break
        if a not in [x for x in e.legal_actions if x]:
            return None
        e.step_in_place(a)
    return e if e.is_terminal else None


def _prefer(legal, *wanted):
    """First of ``wanted`` that is legal, else the cheapest raise, else ``None``.

    Keeps a scripted line playable whatever the abstraction offers at the node:
    a level with no ``call`` (pre-flop open) or no ``all_in`` still has a raise.
    """
    for a in wanted:
        if a in legal:
            return a
    raises = sorted(
        (a for a in legal if a.startswith("raise:")),
        key=lambda a: float(a.split(":", 1)[1]),
    )
    return raises[0] if raises else None


_LINES = {
    # Each policy names the actions it wants in preference order and falls back
    # to the cheapest raise, so a level that drops call/all_in from the
    # abstraction still yields a playable line instead of ``None``.
    "passive": lambda la: _prefer(la, "call"),
    "fold": lambda la: _prefer(la, "fold", "call"),
    "allin": lambda la: _prefer(la, "all_in", "call"),
}


def _feasible_combos(root, board, blockers):
    """Combo rows whose two cards avoid the board and all ``blockers`` cards."""
    excl = set(int(c) for c in board) | set(int(c) for c in blockers)
    cc = root.combo_cards
    return [k for k in range(cc.shape[0])
            if int(cc[k, 0]) not in excl and int(cc[k, 1]) not in excl]


def _pick_disjoint_holes(root, board, k):
    """``k`` mutually-disjoint, board-compatible concrete holes (as card tuples)."""
    cc = root.combo_cards
    used = set(int(c) for c in board)
    picks = []
    for row in range(cc.shape[0]):
        a, b = int(cc[row, 0]), int(cc[row, 1])
        if a in used or b in used:
            continue
        picks.append((a, b))
        used.update((a, b))
        if len(picks) == k:
            break
    assert len(picks) == k, "deck too small to pick disjoint holes"
    return picks


def _check_scenario(root, seat, opp_holes):
    """For every line to a terminal, assert vector_payout_concrete[c] == payout[c].

    ``opp_holes`` maps every non-``seat`` seat to its fixed concrete hole.
    Returns the number of distinct terminal lines actually checked.
    """
    board = list(root.community_cards)
    blockers = [c for h in opp_holes.values() for c in h]
    feasible = _feasible_combos(root, board, blockers)
    assert len(feasible) >= 3
    cc = root.combo_cards
    ref = tuple(int(x) for x in cc[feasible[0]])   # a feasible reference hole

    checked = 0
    for policy in _LINES.values():
        line = _drive(root, policy)
        if line is None:
            continue
        holes_ref = dict(opp_holes)
        holes_ref[seat] = ref
        term = _terminal_via(root, line, holes_ref)
        if term is None:
            continue
        vec = term.vector_payout_concrete(seat)
        assert vec.shape == (root.n_combos,)
        checked += 1
        # Infeasible combos must be exactly zero.
        infeasible = [k for k in range(cc.shape[0]) if k not in set(feasible)]
        assert np.allclose(vec[infeasible], 0.0)
        for c in feasible:
            holes_c = dict(opp_holes)
            holes_c[seat] = tuple(int(x) for x in cc[c])
            e = _terminal_via(root, line, holes_c)
            if e is None:
                continue
            eng = float(e.payout[seat])
            assert abs(float(vec[c]) - eng) < 1e-9, (
                f"seat {seat} combo {c} on line {line}: "
                f"vector={vec[c]} engine={eng}"
            )
    return checked


@pytest.mark.parametrize("stacks", [(3000, 3000), (1500, 6000)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_heads_up_matches_concrete(stacks, seed):
    """HU river: showdown / fold / all-in terminals, equal and unequal stacks."""
    root = _river_env(seed, stacks)
    assert root.betting_round == 3 and not root.is_terminal
    board = list(root.community_cards)
    (opp1,) = _pick_disjoint_holes(root, board, 1)
    checked = _check_scenario(root, seat=0, opp_holes={1: opp1})
    assert checked >= 2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multiway_dead_money_matches_concrete(seed):
    """3-way river with a folded seat: the folded seat's chips (dead money) must
    settle to the winner exactly as the concrete path pays them."""
    root = _river_env_folded_seat(seed)
    if root is None:
        pytest.skip("scenario not reachable on this seed")
    assert root.betting_round == 3 and not root.players[2].is_active
    board = list(root.community_cards)
    opp1, opp2 = _pick_disjoint_holes(root, board, 2)
    checked = _check_scenario(root, seat=0, opp_holes={1: opp1, 2: opp2})
    assert checked >= 2


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_side_pot_matches_concrete(seed):
    """Short-stack all-in forms a genuine side pot; per-combo settlement must
    still net the traverser correctly across both side pots."""
    from environment.pot import Pot

    root = _river_env(seed, stacks=(900, 4000, 4000))
    if root.betting_round != 3 or root.is_terminal:
        pytest.skip("river root not reached on this seed")
    board = list(root.community_cards)
    opp_a, opp_b = _pick_disjoint_holes(root, board, 2)
    line = _drive(root, _LINES["allin"])
    if line is None:
        pytest.skip("no all-in line to a terminal")

    saw_side_pot = False
    for seat in (0, 1):
        others = [s for s in range(3) if s != seat]
        opp_holes = {others[0]: opp_a, others[1]: opp_b}
        ref = _feasible_combos(root, board, list(opp_a) + list(opp_b))[0]
        holes_ref = {**opp_holes, seat: tuple(int(x) for x in root.combo_cards[ref])}
        term = _terminal_via(root, line, holes_ref)
        if term is None or term._terminal_contributions is None:
            continue
        scratch = Pot(3)
        scratch._chips = list(term._terminal_contributions)
        if len(scratch.side_pots) >= 2:
            saw_side_pot = True
        _check_scenario(root, seat=seat, opp_holes=opp_holes)
    assert saw_side_pot, "expected a genuine side pot from the short-stack all-in"
