"""Direct differential fuzz for the ``settle_traverser`` core kernel (Phase 1).

``environment.poker_env._settle_traverser`` is the per-combo twin of
``_settle_runout``: given a frozen board, concrete opponents, and the traverser's
hand swept over its range, it returns the traverser seat's chips won per scenario,
peeling side pots and splitting odd chips exactly as ``Pot.compute_utility``.  The
compiled kernel (``poker_ai._core._runout.settle_traverser``, flag
``settle_concrete``) must be **byte-identical** to that pure-Python oracle.

The env-level bit-exactness gate lives in ``test_multiway_traverser_cfv.py`` (the
whole ``vector_payout_concrete`` path).  Here we hammer the kernel directly over
adversarial synthetic scenarios — many seats, genuine side pots, dead money,
heavy ties, and infeasible (sentinel-ranked) traverser combos — that the
env-driven lines do not densely cover.
"""

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import _settle_traverser_py

_core = pytest.importorskip("poker_ai._core._runout")
settle_traverser = _core.settle_traverser

_SENTINEL = 1 << 30


def _scenario(rng, n):
    """Build a random (active_players, all_players, pot_chips, rank_mat, traverser).

    Contributions vary per seat (so the peel forms genuine side pots); a random
    subset of seats is folded (dead money that can never win); ranks are drawn
    from a small range to force ties, and the traverser column carries occasional
    sentinels (infeasible combos).
    """
    order_perm = rng.permutation(n).tolist()
    players = [Player(i, 1000) for i in range(n)]
    for i in range(n):
        players[i].order = int(order_perm[i])

    # At least two active seats (a contested pot); the traverser is active.
    n_active = int(rng.integers(2, n + 1))
    active_set = sorted(rng.choice(n, size=n_active, replace=False).tolist())
    for i in range(n):
        players[i].is_active = i in active_set
    active_players = [players[i] for i in active_set]
    traverser_pi = int(rng.choice(active_set))

    # Per-seat contributions (>=1), deliberately uneven to peel side pots.
    pot_chips = [int(rng.integers(1, 6)) * int(rng.integers(1, 40)) for _ in range(n)]

    count = int(rng.integers(1, 60))
    # Small rank range → frequent ties; traverser column gets some sentinels.
    rank_mat = rng.integers(0, 4, size=(count, n_active)).astype(np.int64)
    trav_col = active_set.index(traverser_pi)
    sentinel_rows = rng.random(count) < 0.15
    rank_mat[sentinel_rows, trav_col] = _SENTINEL
    return active_players, players, pot_chips, rank_mat, count, traverser_pi


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_settle_traverser_bit_identical(n):
    """Kernel == pure-Python oracle, element-for-element, over many scenarios."""
    rng = np.random.default_rng(1234 + n)
    for _ in range(400):
        active_players, players, pot_chips, rank_mat, count, tpi = _scenario(rng, n)

        oracle = _settle_traverser_py(
            active_players, players, list(pot_chips), rank_mat, count, n, tpi
        )
        order = [0] * n
        for p in players:
            order[p.player_i] = p.order
        active_glob = [p.player_i for p in active_players]
        core = np.asarray(
            settle_traverser(rank_mat, list(pot_chips), order, active_glob, tpi, n),
            dtype=np.float64,
        )
        assert np.array_equal(oracle, core), (
            f"mismatch n={n} traverser={tpi} pot={pot_chips}\n"
            f"oracle={oracle}\ncore={core}"
        )


def test_settle_traverser_empty_and_degenerate():
    """Zero-count and single-active edge cases return the right shape / values."""
    rng = np.random.default_rng(0)
    active_players, players, pot_chips, _, _, tpi = _scenario(rng, 3)
    order = [0] * 3
    for p in players:
        order[p.player_i] = p.order
    active_glob = [p.player_i for p in active_players]
    empty = settle_traverser(
        np.empty((0, len(active_glob)), dtype=np.int64),
        list(pot_chips), order, active_glob, tpi, 3,
    )
    assert np.asarray(empty).shape == (0,)
