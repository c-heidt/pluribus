"""Scope + weighting of the calibration convergence metric (``_street_sigma`` / ``_hot_l1``).

The budget a calibration run emits is whatever ``mean_hot_l1`` first drops under the
tolerance, so what that number ranges over IS the calibration.  Two properties are
load-bearing and silently wrong-able:

- **scope** — the hero's decision nodes on the ROOT STREET, all of them and nothing else.
  One solve serves the whole street (the agent only re-searches at the street boundary or
  on an injected off-tree raise), so root-only under-measures and depth-limit-wide
  over-measures.  A leaked opponent node or a leaked next-street node both read as extra
  "unconvergence" and would inflate every budget.
- **weighting** — by reach mass, so a node the hero can barely reach cannot dominate the
  decisions it actually faces.

Both are asserted against hand-built states rather than a real solve, so a regression
points at the metric and not at the solver.
"""

import numpy as np
import pytest

from evaluation import calibrate


class _State:
    def __init__(self, legal_at, actor_at, vstrat):
        self.legal_at, self.actor_at, self.vstrat = legal_at, actor_at, vstrat


class _Policy:
    """SearchPolicy-shaped: ``strategy_for`` reads a per-node table."""

    def __init__(self, state, sigmas):
        self._state, self._sigmas = state, sigmas

    def strategy_for(self, pk, hr, legal):
        return np.asarray(self._sigmas[pk], dtype=np.float64)


class _Sample:
    def __init__(self, pk, hr=0):
        self.pk, self.hr = pk, hr


# Root street is "flop"; hero is seat 0.
ROOT = ("flop", ())
HERO_LATE = ("flop", (("flop", ("call", "raise")),))     # hero again, same street
OPP = ("flop", (("flop", ("call",)),))                   # opponent's node, same street
NEXT_STREET = ("turn", (("flop", ("call", "call")),))    # hero, but past the boundary


def _policy(sigmas, *, reach=None, actors=None):
    legal_at = {pk: ("fold", "call") for pk in sigmas}
    actor_at = {ROOT: 0, HERO_LATE: 0, OPP: 1, NEXT_STREET: 0}
    if actors:
        actor_at.update(actors)
    reach = reach or {}
    vstrat = {pk: np.full((1, 2), reach.get(pk, 1.0) / 2.0) for pk in sigmas}
    return _Policy(_State(legal_at, {k: v for k, v in actor_at.items() if k in sigmas},
                          vstrat), sigmas)


# --------------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------------- #
def test_sweeps_every_hero_node_on_the_root_street():
    """Not just the root — the later same-street nodes come out of the SAME solve."""
    pol = _policy({ROOT: [0.5, 0.5], HERO_LATE: [0.2, 0.8]})
    got = calibrate._street_sigma(pol, _Sample(ROOT))
    assert set(got) == {ROOT, HERO_LATE}


def test_opponent_nodes_are_excluded():
    """Only the hero's own decisions are the hero's convergence."""
    pol = _policy({ROOT: [0.5, 0.5], OPP: [0.1, 0.9]})
    assert set(calibrate._street_sigma(pol, _Sample(ROOT))) == {ROOT}


def test_nodes_past_the_street_boundary_are_excluded():
    """The next street is re-solved from scratch, so its convergence is not this budget's."""
    pol = _policy({ROOT: [0.5, 0.5], NEXT_STREET: [0.1, 0.9]})
    assert set(calibrate._street_sigma(pol, _Sample(ROOT))) == {ROOT}


def test_dbr_meta_game_nodes_are_excluded():
    """``(pk_base, "META", seat)`` rows share the table but are bias-class draws, not
    betting decisions — leaking them would skew the DBR arm alone."""
    meta = (ROOT, "META", 0)
    pol = _policy({ROOT: [0.5, 0.5], meta: [0.25, 0.75]}, actors={meta: 0})
    assert set(calibrate._street_sigma(pol, _Sample(ROOT))) == {ROOT}


def test_missing_root_is_a_miss_not_a_distribution():
    """An absent root must be NaN, never a (falsely converged) uniform read."""
    pol = _policy({OPP: [0.5, 0.5]})
    assert calibrate._street_sigma(pol, _Sample(ROOT)) is None
    assert np.isnan(calibrate._hot_l1(None, {ROOT: (np.array([0.5, 0.5]), 1.0)}))


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def test_identical_strategies_score_zero():
    snap = {ROOT: (np.array([0.3, 0.7]), 1.0), HERO_LATE: (np.array([0.5, 0.5]), 1.0)}
    assert calibrate._hot_l1(snap, snap) == pytest.approx(0.0)


def test_is_a_mean_not_a_sum_over_nodes():
    """Adding a perfectly-converged node must not shrink the score of a moving one, and
    adding nodes must not inflate the scale — otherwise the tolerance would silently
    depend on how big the street's betting tree happens to be."""
    moved = (np.array([0.0, 1.0]), 1.0)
    ref_moved = (np.array([1.0, 0.0]), 1.0)
    same = (np.array([0.5, 0.5]), 1.0)
    one = calibrate._hot_l1({ROOT: moved}, {ROOT: ref_moved})
    two = calibrate._hot_l1({ROOT: moved, HERO_LATE: same},
                            {ROOT: ref_moved, HERO_LATE: same})
    assert one == pytest.approx(2.0)          # full swap on a 2-action node
    assert two == pytest.approx(1.0)          # averaged with a settled node


def test_reach_weighting_discounts_the_barely_reachable_node():
    """A node the hero almost never reaches cannot dominate the one it always plays."""
    settled = np.array([0.5, 0.5])
    swapped_a, swapped_b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    snap = {ROOT: (settled, 1.0), HERO_LATE: (swapped_a, 0.001)}
    ref = {ROOT: (settled, 1.0), HERO_LATE: (swapped_b, 0.001)}
    assert calibrate._hot_l1(snap, ref) < 0.01


def test_node_absent_from_the_snapshot_is_scored_as_uniform():
    """``strategy_for`` returns uniform for an unvisited node — that is what the bot would
    play at that budget, so it must be charged, not skipped."""
    ref = {ROOT: (np.array([1.0, 0.0]), 1.0)}
    assert calibrate._hot_l1({}, ref) == pytest.approx(1.0)   # |.5-1| + |.5-0|


def test_reference_defines_the_node_set():
    """Rungs of one ladder must be scored on the same decisions, so extra nodes that only
    the snapshot has are ignored — the top-budget reference is the yardstick."""
    ref = {ROOT: (np.array([0.5, 0.5]), 1.0)}
    snap = {ROOT: (np.array([0.5, 0.5]), 1.0), HERO_LATE: (np.array([1.0, 0.0]), 1.0)}
    assert calibrate._hot_l1(snap, ref) == pytest.approx(0.0)


def test_cold_tail_mass_is_ignored_but_full_l1_still_sees_it():
    """The hot mask is the point of the metric; ``l1_to_ref`` keeps the unmasked view."""
    snap = {ROOT: (np.array([0.98, 0.02]), 1.0)}
    ref = {ROOT: (np.array([1.0, 0.0]), 1.0)}
    assert calibrate._hot_l1(snap, ref) == pytest.approx(0.02)     # only the hot action
    assert calibrate._full_l1(snap, ref) == pytest.approx(0.04)    # both entries


# --------------------------------------------------------------------------- #
# Per-regime tolerance + the measured noise floor
# --------------------------------------------------------------------------- #
def _rows(regime, hot_by_rung, *, cross=float("nan"), ladder=(100, 200, 400)):
    """One SweepRow per rung with a prescribed mean hot_l1 (top rung carries the floor)."""
    return [
        calibrate.SweepRow(
            condition="vanilla", regime=regime, street=1, n_live=2, per_replica=t,
            workers=1, pooled_iters=t, wall_seconds=1.0, stop_reason="iteration_cap",
            sample=0, rep=0, value_gap_mbb=0.0, root_value=0.0,
            hot_l1=hot_by_rung[t], argmax_match=1.0, l1_to_ref=hot_by_rung[t],
            hot_l1_cross_rep=(cross if t == ladder[-1] else float("nan")),
        )
        for t in ladder
    ]


def test_mccfr_gets_the_looser_bar_and_vector_the_tighter_one():
    """A residual of 0.15 is convergence for a SAMPLED cell and is not for a full-width
    one — the whole point of splitting the tolerance."""
    hot = {100: 0.40, 200: 0.15, 400: 0.0}
    mccfr = calibrate.summarize_cell(("vanilla", "mccfr", 1, 2), _rows("mccfr", hot))
    vector = calibrate.summarize_cell(("vanilla", "vector", 2, 2), _rows("vector", hot))
    assert mccfr.hot_l1_tol == calibrate.DEFAULT_HOT_L1_TOL_MCCFR == 0.20
    assert vector.hot_l1_tol == calibrate.DEFAULT_HOT_L1_TOL_VECTOR == 0.10
    assert mccfr.suggested_budget == 200        # 0.15 <= 0.20 → settled
    assert vector.suggested_budget is None      # 0.15 > 0.10 → still moving


def test_cross_rep_spread_is_measured_and_flags_an_over_precise_tolerance():
    """A tol under the cell's own cross-seed spread resolves the budget finer than two
    independent seeds of that cell agree to, so it is largely an artefact of which seeds
    ran. Not impossible (the self-distance is a nested comparison and its noise partly
    cancels) — but it must be surfaced, not silently emitted as a clean answer."""
    hot = {100: 0.40, 200: 0.30, 400: 0.0}
    s = calibrate.summarize_cell(("vanilla", "mccfr", 1, 2),
                                 _rows("mccfr", hot, cross=0.25))
    assert s.cross_rep_hot_l1 == pytest.approx(0.25)
    assert s.tol_below_cross_rep is True      # tol 0.20 <= spread 0.25
    s2 = calibrate.summarize_cell(("vanilla", "mccfr", 1, 2),
                                  _rows("mccfr", hot, cross=0.05))
    assert s2.tol_below_cross_rep is False


def test_cross_rep_spread_is_nan_without_repeats():
    """Deterministic cells run a single rep — there is no dispersion to measure, and a
    fabricated 0.0 would read as 'this cell is perfectly precise at any tolerance'."""
    hot = {100: 0.40, 200: 0.05, 400: 0.0}
    s = calibrate.summarize_cell(("vanilla", "vector", 3, 2), _rows("vector", hot))
    assert np.isnan(s.cross_rep_hot_l1)
    assert s.tol_below_cross_rep is False
