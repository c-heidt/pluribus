# cython: language_level=3
"""Decision-free runout settlement kernel (Phase 2) — Cython port of the
per-completion side-pot settlement inside
``environment.poker_env.PokerEnv.runout_equity``.

``runout_equity`` averages the terminal payoff over **every** board completion of
an all-in runout.  Its cost (eval doc §6.7 / the search-bottlenecks profile) is
not the batched 7-card ranking (already a LUT via ``evaluate_batch``) but the
**settlement scaffolding**: per completion it re-peels the side pots and calls
``Pot.compute_utility`` (the scalar tie path), plus the numpy fast-path bincount
machinery.  This kernel replaces that whole block with one tight C loop.

**Bit-identity is easy here, unlike ``showdown_cfv``:** every quantity summed is
*integer* (chips won per seat per completion, or the fast path's ``count * pot``),
so ``accum[i]`` is an integer-valued float regardless of the summation order — the
kernel need only produce the *same integer settlement per completion* as
``Pot.compute_utility`` and leave the final ``accum[i] / count - pot_chips[i]``
float division to Python.  The settlement reproduces ``_settle.pyx``'s exact peel
+ odd-chip rules (the ranking is an input; ranks are complete-board integers, no
sentinel):

* **Side-pot peel** — done **once** (contributions are board-independent): repeatedly
  remove the minimum positive contribution across all still-contributing seats in
  ``player_i`` order; each peel is one side pot, members = seats still in, total =
  ``min * count``.  Folded seats that contributed are members (dead money) but can
  never win (they are not among the active columns).
* **Award per completion** — for each side pot, the winner group is the *best*
  (minimum) rank among the pot's **active member** columns; ties split
  ``total // n_win`` each with the ``total % n_win`` remainder to the first winners
  **sorted by ``order``** (stable).  A pot no active member is eligible for is left
  unassigned — exactly ``compute_utility``'s "no eligible group → not awarded".
"""

cimport cython
import numpy as np

DEF MAX_PLAYERS = 32


@cython.boundscheck(False)
@cython.wraparound(False)
def settle_runout(rank_mat, pot_chips, order, active_glob, int n_players):
    """Sum ``Pot.compute_utility`` over every completion; return ``accum`` int64[n].

    Parameters
    ----------
    rank_mat : (count, n_active) int array
        Best-5-of-7 rank of each active seat on each board completion (lower =
        stronger), as produced by ``evaluate_batch``.
    pot_chips : length-n int sequence
        Frozen per-``player_i`` pot contribution (the runout snapshot) — the
        settlement ``contrib``, over *all* seats incl. folded (dead money).
    order : length-n int sequence
        Per-``player_i`` bet-order tiebreak for the odd-chip remainder.
    active_glob : length-n_active int sequence
        Global ``player_i`` of each column of ``rank_mat`` (the still-active,
        ranked seats), in the order the ranks are laid out.
    n_players : int
        ``n`` — the number of seats.

    Returns
    -------
    numpy.ndarray
        int64[n]: summed chips won by each seat over all completions.  The caller
        forms the equity ``accum[i] / count - pot_chips[i]``.
    """
    cdef Py_ssize_t n = n_players
    if n > MAX_PLAYERS:
        raise ValueError(
            "runout kernel supports up to %d players, got %d" % (MAX_PLAYERS, n)
        )

    cdef const long[:, ::1] ranks = np.ascontiguousarray(rank_mat, dtype=np.int64)
    cdef const long[::1] contrib = np.ascontiguousarray(pot_chips, dtype=np.int64)
    cdef const long[::1] order_c = np.ascontiguousarray(order, dtype=np.int64)
    cdef const long[::1] aglob = np.ascontiguousarray(active_glob, dtype=np.int64)

    cdef Py_ssize_t count = ranks.shape[0]
    cdef Py_ssize_t n_active = ranks.shape[1]

    out = np.zeros(n, dtype=np.int64)
    cdef long[::1] accum = out
    if count == 0 or n_active == 0:
        return out

    # --- Peel side pots once (identical to _settle.pyx / Pot.side_pots). ---
    cdef long long remaining[MAX_PLAYERS]
    cdef bint members[MAX_PLAYERS][MAX_PLAYERS]
    cdef long long pot_total[MAX_PLAYERS]
    cdef Py_ssize_t n_pots = 0
    cdef long long minv
    cdef bint found
    cdef Py_ssize_t i, a, p, ci, cnt, j, k

    cdef long long won[MAX_PLAYERS]
    cdef int winners[MAX_PLAYERS]
    cdef long long best, r, tot, per, rem
    cdef int nwin, g, key, w

    with nogil:
        for i in range(n):
            remaining[i] = contrib[i]
        while True:
            found = False
            minv = 0
            for i in range(n):
                if remaining[i] > 0 and (not found or remaining[i] < minv):
                    minv = remaining[i]
                    found = True
            if not found:
                break
            cnt = 0
            for i in range(n):
                if remaining[i] > 0:
                    members[n_pots][i] = True
                    cnt += 1
                else:
                    members[n_pots][i] = False
            pot_total[n_pots] = minv * cnt
            for i in range(n):
                if remaining[i] > 0:
                    remaining[i] -= minv
            n_pots += 1

        # --- Award every side pot on every completion. ---
        for ci in range(count):
            for i in range(n):
                won[i] = 0
            for p in range(n_pots):
                tot = pot_total[p]
                # Best (minimum) rank among this pot's active member columns.
                found = False
                best = 0
                for a in range(n_active):
                    g = <int>aglob[a]
                    if members[p][g]:
                        r = ranks[ci, a]
                        if not found or r < best:
                            best = r
                            found = True
                if not found:
                    continue                      # unwinnable pot — unassigned
                # Winners = active member columns whose rank == best.
                nwin = 0
                for a in range(n_active):
                    g = <int>aglob[a]
                    if members[p][g] and ranks[ci, a] == best:
                        winners[nwin] = g
                        nwin += 1
                # Stable insertion sort of winners by ``order`` (ties keep column order).
                for j in range(1, nwin):
                    w = winners[j]
                    key = <int>order_c[w]
                    k = j - 1
                    while k >= 0 and <int>order_c[winners[k]] > key:
                        winners[k + 1] = winners[k]
                        k -= 1
                    winners[k + 1] = w
                per = tot // nwin
                rem = tot - nwin * per
                for j in range(nwin):
                    won[winners[j]] += per
                for j in range(rem):              # remainder to first winners
                    won[winners[j]] += 1
            for i in range(n):
                accum[i] += won[i]

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def settle_traverser(rank_mat, pot_chips, order, active_glob,
                     int traverser_pi, int n_players):
    """Chips won by seat ``traverser_pi`` in each of ``count`` scenarios.

    The per-combo twin of :func:`settle_runout`: the same one-shot side-pot peel
    and per-scenario award, but ``rank_mat``'s ``count`` axis is the traverser's
    hole combos (one board, concrete opponents, the traverser's hand swept over
    its range) rather than board completions, and the result is the traverser
    seat's chips won **per scenario** (not summed over the axis).  Byte-identical
    to ``environment.poker_env._settle_traverser``.

    Parameters
    ----------
    rank_mat : (count, n_active) int array
        Best-5-of-7 rank of each active seat in each scenario (lower = stronger);
        the traverser's column varies with its combo, each opponent column is that
        seat's fixed concrete rank.  Infeasible traverser combos are parked at the
        caller's sentinel (they lose every pot; the caller masks them to 0).
    pot_chips, order, active_glob, n_players
        As in :func:`settle_runout`.
    traverser_pi : int
        Global ``player_i`` whose per-scenario winnings are returned.

    Returns
    -------
    numpy.ndarray
        int64[count]: chips won by the traverser in each scenario.  The caller
        nets the traverser's contribution and zeroes infeasible combos.
    """
    cdef Py_ssize_t n = n_players
    if n > MAX_PLAYERS:
        raise ValueError(
            "runout kernel supports up to %d players, got %d" % (MAX_PLAYERS, n)
        )

    cdef const long[:, ::1] ranks = np.ascontiguousarray(rank_mat, dtype=np.int64)
    cdef const long[::1] contrib = np.ascontiguousarray(pot_chips, dtype=np.int64)
    cdef const long[::1] order_c = np.ascontiguousarray(order, dtype=np.int64)
    cdef const long[::1] aglob = np.ascontiguousarray(active_glob, dtype=np.int64)

    cdef Py_ssize_t count = ranks.shape[0]
    cdef Py_ssize_t n_active = ranks.shape[1]

    out = np.zeros(count, dtype=np.int64)
    cdef long[::1] trav_out = out
    if count == 0 or n_active == 0:
        return out

    # --- Peel side pots once (identical to settle_runout / _settle.pyx). ---
    cdef long long remaining[MAX_PLAYERS]
    cdef bint members[MAX_PLAYERS][MAX_PLAYERS]
    cdef long long pot_total[MAX_PLAYERS]
    cdef Py_ssize_t n_pots = 0
    cdef long long minv
    cdef bint found
    cdef Py_ssize_t i, a, p, ci, cnt, j, k

    cdef long long won[MAX_PLAYERS]
    cdef int winners[MAX_PLAYERS]
    cdef long long best, r, tot, per, rem
    cdef int nwin, g, key, w
    cdef int tpi = traverser_pi

    with nogil:
        for i in range(n):
            remaining[i] = contrib[i]
        while True:
            found = False
            minv = 0
            for i in range(n):
                if remaining[i] > 0 and (not found or remaining[i] < minv):
                    minv = remaining[i]
                    found = True
            if not found:
                break
            cnt = 0
            for i in range(n):
                if remaining[i] > 0:
                    members[n_pots][i] = True
                    cnt += 1
                else:
                    members[n_pots][i] = False
            pot_total[n_pots] = minv * cnt
            for i in range(n):
                if remaining[i] > 0:
                    remaining[i] -= minv
            n_pots += 1

        # --- Award every side pot in every scenario; emit the traverser's cell. ---
        for ci in range(count):
            for i in range(n):
                won[i] = 0
            for p in range(n_pots):
                tot = pot_total[p]
                found = False
                best = 0
                for a in range(n_active):
                    g = <int>aglob[a]
                    if members[p][g]:
                        r = ranks[ci, a]
                        if not found or r < best:
                            best = r
                            found = True
                if not found:
                    continue                      # unwinnable pot — unassigned
                nwin = 0
                for a in range(n_active):
                    g = <int>aglob[a]
                    if members[p][g] and ranks[ci, a] == best:
                        winners[nwin] = g
                        nwin += 1
                # Stable insertion sort of winners by ``order`` (ties keep column order).
                for j in range(1, nwin):
                    w = winners[j]
                    key = <int>order_c[w]
                    k = j - 1
                    while k >= 0 and <int>order_c[winners[k]] > key:
                        winners[k + 1] = winners[k]
                        k -= 1
                    winners[k + 1] = w
                per = tot // nwin
                rem = tot - nwin * per
                for j in range(nwin):
                    won[winners[j]] += per
                for j in range(rem):              # remainder to first winners
                    won[winners[j]] += 1
            trav_out[ci] = won[tpi]

    return out

