"""Byte-identity gate for the Phase-2 runout settlement kernel (``_runout``).

``settle_runout`` replaces the per-completion side-pot scoring inside
``PokerEnv.runout_equity`` (the ``_settle_runout`` seam).  Accumulation there is
integer (chips won), so bit-identity needs only the same per-completion
settlement as ``Pot.compute_utility`` — asserted here by a differential of the
whole ``runout_equity`` (Python ``_settle_runout`` vs the core one swapped in)
over heads-up, unequal-stack, layered side-pot, folded-dead-money, and
already-complete-board runouts; plus the end-to-end golden digests (the MCCFR
solve hits runout 45×, so the gate is non-vacuous).

The existing brute-force reference in
``test/environment/unit/test_runout_equity.py`` and the leaf decision-free tests
are additionally run with ``PLURIBUS_CORE_KERNELS=runout`` in CI-style subprocess
checks (see the session notes); here we assert the kernel against the shipped
Python oracle directly.
"""

import numpy as np
import pytest

import environment.poker_env as pe
from environment.player import Player
from environment.poker_env import PokerEnv
from poker_ai._core._runout import settle_runout as _core_settle

from test.search.functional import test_search_core_golden as golden


def _core_adapter(active_players, all_players, pot_chips, rank_mat, count, n):
    """The production wiring's marshalling, isolated for the differential."""
    order = [0] * n
    for p in all_players:
        order[p.player_i] = p.order
    active_glob = [p.player_i for p in active_players]
    acc = _core_settle(rank_mat, list(pot_chips), order, active_glob, n)
    return [float(acc[i]) for i in range(n)]


def _env(stacks, seed, low=11, high=14):
    np.random.seed(seed)
    return PokerEnv(players=[Player(i, s) for i, s in enumerate(stacks)],
                    low_card_rank=low, high_card_rank=high)


def _drive_allin(env, steps=16):
    s = 0
    while not env.is_terminal and s < steps:
        legal = [a for a in env.legal_actions if a]
        nxt = next((a for a in ("all_in", "call", "check") if a in legal), legal[-1])
        env.step_in_place(nxt)
        s += 1
    return env.is_decision_free


def _assert_core_matches(env):
    """runout_equity with the Python settlement == with the core settlement (exact)."""
    py = env.runout_equity()
    pe._settle_runout = _core_adapter
    try:
        core = env.runout_equity()
    finally:
        pe._settle_runout = pe._settle_runout_py
    assert py.keys() == core.keys()
    for i in py:
        assert py[i] == core[i], f"seat {i}: py={py[i]!r} core={core[i]!r}"
    return env._runout_info


@pytest.mark.parametrize("stacks", [
    (200, 200),          # heads-up equal (dead=0, chops/odd chips on the small deck)
    (150, 600),          # heads-up unequal (matched stake, uncalled excess)
    (150, 400, 900),     # 3-way layered side pots (the multiway fixture)
    (100, 300, 700, 900),  # 4-way, three side-pot layers
])
def test_runout_settlement_bit_identical(stacks):
    checked = 0
    side_pots = 0
    for seed in range(80):
        env = _env(list(stacks), seed)
        if not _drive_allin(env):
            continue
        _, pot_chips, _ = _assert_core_matches(env)
        checked += 1
        if len({v for v in pot_chips if v > 0}) > 1:
            side_pots += 1
    assert checked >= 20, f"too few runouts reached for {stacks}: {checked}"
    if len(stacks) > 2:
        assert side_pots > 0, "multiway config never produced a genuine side pot"


def _oracle_accum(rank_mat, pot_chips, order, active_glob, n):
    """Independent reference: sum ``compute_utility_won`` (the ``_settle`` oracle)
    over every completion, grouping active seats by this completion's ranks."""
    from poker_ai._core._settle import compute_utility_won

    count, n_active = rank_mat.shape
    accum = [0] * n
    for ci in range(count):
        groups = {}
        for a in range(n_active):
            groups.setdefault(int(rank_mat[ci, a]), []).append(int(active_glob[a]))
        ranked = [groups[r] for r in sorted(groups)]
        won = compute_utility_won(list(pot_chips), ranked, list(order))
        for i in range(n):
            accum[i] += won[i]
    return accum


def test_settle_runout_kernel_differential():
    """``settle_runout`` == summed ``compute_utility_won`` over randomized inputs.

    Directly controls the paths env-driven play rarely reaches: dead-money
    members (contributors absent from the active set), layered side pots, chops
    and odd-chip remainders (a tiny rank range forces frequent ties), degenerate
    pots (no active member — left unassigned), and a single completion (count=1).
    """
    rng = np.random.RandomState(0)
    dead_money = chops = degenerate = single = 0
    for _ in range(3000):
        n = int(rng.randint(2, 7))
        n_active = int(rng.randint(2, n + 1))
        active_glob = sorted(rng.choice(n, size=n_active, replace=False).tolist())
        # Contributions for ALL seats (folded/inactive seats with chips = dead money).
        pot_chips = rng.randint(0, 5, size=n).tolist()
        for g in active_glob:                        # active seats must have skin in
            pot_chips[g] = int(rng.randint(1, 6))
        order = list(rng.permutation(n))
        count = int(rng.randint(1, 6))
        rank_mat = rng.randint(1, 4, size=(count, n_active)).astype(np.int64)  # ties

        core = _core_settle(rank_mat, list(pot_chips), order, active_glob, n).tolist()
        ref = _oracle_accum(rank_mat, pot_chips, order, active_glob, n)
        assert core == ref, (pot_chips, active_glob, order, rank_mat.tolist(), core, ref)

        if any(pot_chips[i] > 0 and i not in active_glob for i in range(n)):
            dead_money += 1
        if count == 1:
            single += 1
        # a chop = some completion where two active seats tie for a pot's best
        if n_active >= 2 and (rank_mat.min(axis=1, keepdims=True) == rank_mat).sum() > count:
            chops += 1
        inpots = set(active_glob)
        if any(pot_chips[i] > 0 and not (inpots & {i}) for i in range(n)):
            degenerate += 1
    assert dead_money > 100 and chops > 100 and single > 100, (
        dead_money, chops, single)


def test_mccfr_digest_unchanged_with_runout_kernel(monkeypatch):
    """The MCCFR golden digest is byte-unchanged with the runout kernel (hit 45×)."""
    monkeypatch.setattr(pe, "_settle_runout", _core_adapter)
    assert golden._mccfr_digest() == golden.GOLDEN_DIGEST_MCCFR


def test_vector_digest_unchanged_with_runout_kernel(monkeypatch):
    """The vector golden digest is likewise untouched by the runout kernel."""
    monkeypatch.setattr(pe, "_settle_runout", _core_adapter)
    assert golden._vector_digest() == golden.GOLDEN_DIGEST_VECTOR
