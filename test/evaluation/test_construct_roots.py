"""Deterministic root construction for calibration (``calibrate.construct_roots``).

Playing hands under-samples rare production spots (multiway, HU turn/river), so those
cells starve.  ``construct_roots`` instead BUILDS one root per ``(street, n_live)`` cell
directly — a fresh seeded deal driven down a passive line to the target street/live
count, with uniform board-masked beliefs.  These tests pin: full cell coverage, that
each root actually sits at its cell's street + live count (a real hero decision), the
regime routing (HU turn/river → vector, else mccfr), and byte-stable determinism.
"""

import collections
import dataclasses

import numpy as np
import pytest

from evaluation.calibrate import construct_roots
from evaluation.opponents import ModelSpec
from evaluation.runner import EvalConfig, EvalSession
from poker_ai.search.leaf import LeafConfig
from poker_ai.search.mccfr import _BIAS_CLASSES
from poker_ai.search.policy import Policy
from poker_ai.search.solver_state import SolverConfig
from test.lut_helpers import cluster_lut as _cluster_lut, install_cluster_lut


class _Uniform(Policy):
    def strategy(self, state, bias="none"):
        n = len(state.legal_actions)
        return np.full(n, 1.0 / n, np.float32) if n else np.array([], np.float32)


def _session(n_players, *, run_seed=7):
    leaf = LeafConfig(policies={c: _Uniform() for c in _BIAS_CLASSES})
    scfg = SolverConfig(leaf=leaf, max_iterations=4, max_wall_seconds=30.0,
                        discount_interval=20)
    lut = _cluster_lut()
    cfg = EvalConfig(
        run_id="t", run_seed=run_seed, table_policy="all_blueprint", fixed_seats=None,
        n_players=n_players, time_budget_hours=0.0, big_blind=100, small_blind=50,
        # 60 bb.  The abstraction has no pre-flop limp, so a stack that cannot
        # cover the ~2.6 bb open leaves fold-or-shove as the only pre-flop
        # lines and the deeper cells (river, 3 live) become unreachable.
        starting_stack=6000, low_card_rank=11, high_card_rank=14,
    )
    session = EvalSession(config=cfg, solver_cfg=scfg,
                          blueprint_policy=_Uniform(), card_info_lut=lut)
    return session, cfg, scfg


def _expected_cells(n_players):
    # POST-FLOP only: pre-flop is played from the blueprint, never searched, so it is
    # not calibrated (see ``calibrate._target_cells``).
    cells = set()
    for street in (1, 2, 3):
        for n_live in range(2, n_players + 1):
            cells.add((street, n_live))
    return cells


def _build(n_players, *, per_cell=2, run_seed=7, condition="vanilla", vector_only=False):
    session, cfg, scfg = _session(n_players, run_seed=run_seed)
    if condition != "vanilla":
        # The real arm dispatch, so model_spec / model_scope / k_beta are set the way
        # a run sets them — not hand-assembled here, which is how the OX arm's
        # 'belief_only' scope went unnoticed.
        arm = EvalConfig.for_condition(
            condition, model_spec=ModelSpec(p_max=1.0, error=0.2, seed=1),
            ox_k_beta=50.0, run_id="t",
        )
        cfg = dataclasses.replace(
            cfg, condition=arm.condition, model_spec=arm.model_spec,
            model_scope=arm.model_scope, k_beta=arm.k_beta,
        )
        session = dataclasses.replace(session, config=cfg)
    return construct_roots(session, cfg, scfg, condition,
                           per_cell=per_cell, run_seed=run_seed,
                           vector_only=vector_only)


def _digest(out):
    return [
        (cell, [(r.pk, r.hr, tuple(r.legal),
                 tuple(int(c) for c in r.env.community_cards)) for r in out[cell]])
        for cell in sorted(out, key=repr)
    ]


@pytest.mark.parametrize("n_players", [2, 3, 4])
def test_covers_every_street_and_live_count(n_players):
    out = _build(n_players)
    got = {(cell[2], cell[3]) for cell in out}          # (street, n_live)
    assert got == _expected_cells(n_players)


def test_per_cell_sample_count():
    out = _build(4, per_cell=3)
    assert out and all(len(v) == 3 for v in out.values())


def test_each_root_sits_at_its_cell():
    # The cell key must match what the root actually is: the ctx carries exactly
    # ``n_live`` live ranges, the env is on the target street, and the hero is to act.
    out = _build(4)
    for (_cond, _regime, street, n_live), roots in out.items():
        for r in roots:
            assert r.street == street
            assert len(r.ctx.ranges) == n_live
            assert r.env.betting_round == street
            assert not r.env.is_terminal
            assert r.env.player_i == r.ctx.my_seat     # a real hero decision node
            assert r.legal                              # non-empty legal set
            # Uniform belief is a normalised, board-masked distribution.
            for w in r.ctx.ranges.values():
                assert abs(float(np.sum(w)) - 1.0) < 1e-6


def test_regime_routing_hu_turn_river_is_vector():
    out = _build(4)
    by_cell = {(cell[2], cell[3]): cell[1] for cell in out}
    assert by_cell[(2, 2)] == "vector"                 # HU turn
    assert by_cell[(3, 2)] == "vector"                 # HU river
    assert by_cell[(1, 2)] == "mccfr"                  # HU flop stays MCCFR
    assert by_cell[(2, 3)] == "mccfr"                  # multiway turn
    assert by_cell[(1, 4)] == "mccfr"                  # multiway flop
    # Pre-flop is NOT calibrated (played from the blueprint, never searched).
    assert not any(street == 0 for street, _ in by_cell)


def test_vanilla_has_no_models():
    out = _build(4)
    for roots in out.values():
        for r in roots:
            assert not r.ctx.models                    # no model_spec ⇒ vanilla


def test_dbr_puts_models_in_the_subgame():
    """The premise of the OX test below: a 'full'-scope arm DOES carry models."""
    out = _build(4, condition="DBR(confidence=0.8)")
    assert any(r.ctx.models for roots in out.values() for r in roots)


def test_ox_roots_carry_no_models_into_the_subgame():
    """OX-Search is reach-only: its models shape beliefs, never the solve.

    An OX arm always carries a ``model_spec`` — that spec is what perturbs the
    beliefs — so ``model_spec is None`` does NOT distinguish it from DBR; only
    ``model_scope`` does.  Missing that gate here put models in every constructed
    root, and the vector solver refuses to build the gadget with a populated
    ``ctx.models`` ("requires ctx.models to be empty"), so an OX calibration died
    on its first solve — after the roots and the throughput probe were already paid
    for.
    """
    out = _build(4, condition="OX(k_beta=50)")
    assert out, "no roots constructed"
    for cell, roots in out.items():
        for r in roots:
            assert not r.ctx.models, f"OX root in {cell} carries models into the solve"


def test_deterministic_same_seed():
    assert _digest(_build(4, run_seed=11)) == _digest(_build(4, run_seed=11))


def test_different_seed_differs():
    # Different decks ⇒ the constructed roots differ (not a frozen constant).
    assert _digest(_build(4, run_seed=1)) != _digest(_build(4, run_seed=2))


def test_ox_builds_only_the_two_player_vector_cells():
    """An OX run's whole grid is the HU turn + river; the rest is discarded compute.

    The gadget exists only at the 2-player vector root.  Off it OX falls back to vanilla
    and takes vanilla's budget, so a root built for any other cell is solved, swept and
    then thrown away — on a production grid that is 7 of 9 cells.
    """
    out = _build(4, condition="OX(k_beta=50)", vector_only=True)
    assert set(out), "no roots constructed"
    for cell in out:
        _cond, regime, street, n_live = cell
        assert regime == "vector" and n_live == 2, f"OX built a non-gadget cell: {cell}"
    assert {(c[2], c[3]) for c in out} == {(2, 2), (3, 2)}      # turn + river, HU


def test_restricting_to_vector_cells_is_lossless():
    """The cells it does build are byte-identical to the full grid's — seeds are per cell."""
    full = _build(4)
    ox = _build(4, condition="OX(k_beta=50)", vector_only=True)
    for cell, roots in ox.items():
        twin = next(c for c in full if c[1:] == cell[1:])
        assert [(r.pk, r.hr, tuple(r.legal)) for r in roots] == \
               [(r.pk, r.hr, tuple(r.legal)) for r in full[twin]]
