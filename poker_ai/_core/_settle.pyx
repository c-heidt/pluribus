# cython: language_level=3
"""Side-pot settlement kernel (Phase 1e) — Cython port of
``environment.pot.Pot.compute_utility`` (+ its ``side_pots`` peel and the
``_split_side_pot`` / ``_players_eligible_for_pot`` helpers).

At every showdown leaf the terminal payoff for player ``i`` is the pure function
``payout[i] = won[i] - contrib[i]`` (``environment.poker_env.PokerEnv.payout``
nets exactly this: ``add_chips(won)`` then ``n_chips - initial``).  ``contrib`` is
the per-seat pot contribution (``pot._chips``); ``won`` is what
``compute_utility`` distributes.  This kernel reproduces ``won`` **byte-identically**
and also exposes the netted ``payout`` for the Phase-3 in-core traversal.

Two order-sensitive details are replicated exactly (they decide single chips):

* **Side-pot peel** — repeatedly remove the minimum positive contribution across
  all still-contributing seats (in ``player_i`` order, matching the dict built by
  ``enumerate(self._chips)``); each peel forms one side pot whose members are the
  seats still in at that level and whose total is ``min * count``.  Folded seats
  that contributed are members (dead money in the total) but never win — they are
  absent from ``ranked_groups``, so a side pot no winner is eligible for is simply
  not awarded (chips stay unassigned), exactly as the Python does.
* **Odd-chip remainder** — a side pot splits ``total // n_winners`` to each
  winner; the ``total % n_winners`` leftover chips go one at a time to the first
  winners, where winners are the eligible group **stably sorted by ``player.order``**
  (``pot.py:137,147``).  The sort is stable so ties keep ranked-group order.

Inputs use ``player_i`` (0..n-1) as the seat index throughout: ``contrib`` and
``order`` are indexed by it, and ``ranked_groups`` is a best-first list of lists
of ``player_i``.  Purely integer arithmetic — no evaluator tables needed here
(the ranking is an input; the evaluator kernel produces it in Phase 1d/3).
"""

cimport cython

DEF MAX_PLAYERS = 32


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _settle_won(contrib, ranked_groups, order):
    """Return ``won`` as a length-n Python list (byte-identical to
    ``Pot.compute_utility`` restricted to the win amounts)."""
    cdef Py_ssize_t n = len(contrib)
    if n > MAX_PLAYERS:
        raise ValueError(
            "settlement kernel supports up to %d players, got %d"
            % (MAX_PLAYERS, n)
        )

    cdef long long remaining[MAX_PLAYERS]
    cdef long long won[MAX_PLAYERS]
    cdef int order_c[MAX_PLAYERS]
    cdef Py_ssize_t i, j, k

    for i in range(n):
        remaining[i] = contrib[i]
        won[i] = 0
        order_c[i] = order[i]

    # --- Build side pots by the exact min-peel (smallest pot first). ---
    cdef bint members[MAX_PLAYERS][MAX_PLAYERS]
    cdef long long pot_total[MAX_PLAYERS]
    cdef Py_ssize_t n_pots = 0
    cdef long long minv
    cdef bint found
    cdef Py_ssize_t count

    while True:
        found = False
        minv = 0
        for i in range(n):
            if remaining[i] > 0:
                if not found or remaining[i] < minv:
                    minv = remaining[i]
                    found = True
        if not found:
            break
        count = 0
        for i in range(n):
            if remaining[i] > 0:
                members[n_pots][i] = True
                count += 1
            else:
                members[n_pots][i] = False
        pot_total[n_pots] = minv * count
        for i in range(n):
            if remaining[i] > 0:
                remaining[i] -= minv
        n_pots += 1

    # --- Award each side pot to the best-ranked eligible group. ---
    cdef int eligible[MAX_PLAYERS]
    cdef Py_ssize_t n_elig, p
    cdef int pi, key
    cdef long long per, rem, n_total

    for p in range(n_pots):
        n_total = pot_total[p]
        for group in ranked_groups:            # best-ranked group first
            n_elig = 0
            for pi_obj in group:               # preserves group order
                pi = pi_obj
                if members[p][pi]:
                    eligible[n_elig] = pi
                    n_elig += 1
            if n_elig > 0:
                # Stable insertion sort by player.order (ties keep group order).
                for j in range(1, n_elig):
                    pi = eligible[j]
                    key = order_c[pi]
                    k = j - 1
                    while k >= 0 and order_c[eligible[k]] > key:
                        eligible[k + 1] = eligible[k]
                        k -= 1
                    eligible[k + 1] = pi
                per = n_total // n_elig
                rem = n_total - n_elig * per
                for j in range(n_elig):
                    won[eligible[j]] += per
                for j in range(rem):           # remainder to first winners
                    won[eligible[j]] += 1
                break                          # this side pot is settled

    return [won[i] for i in range(n)]


def compute_utility_won(contrib, ranked_groups, order):
    """Return ``won[player_i]`` (chips won, 0 for non-winners) as a list.

    Drop-in for the win amounts of ``Pot.compute_utility``: ``contrib`` is
    ``pot._chips`` (per ``player_i``), ``ranked_groups`` a best-first list of
    lists of ``player_i``, ``order`` the per-``player_i`` bet-order tiebreak.
    """
    return _settle_won(contrib, ranked_groups, order)


def settle_payout(contrib, ranked_groups, order):
    """Return the netted terminal payoff ``won[i] - contrib[i]`` per seat.

    This is exactly ``PokerEnv.payout`` — the quantity CFR reads at a terminal
    (``state.payout[i]``).
    """
    won = _settle_won(contrib, ranked_groups, order)
    return [won[i] - int(contrib[i]) for i in range(len(contrib))]
