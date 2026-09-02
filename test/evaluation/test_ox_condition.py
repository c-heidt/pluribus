"""OX-Search (Approach B) eval-wiring gates (§11.3 step 12b).

Covers the harness plumbing that turns the gadget solver into a runnable, logged
experiment arm: the ``OX(k_beta=X)`` condition (reach-only — search on, β set, and any
model confined to the BELIEF) and the per-decision ``ox_enter_prob``
opt-out-saturation column (round-trips through the v7 schema; NULL for every non-OX
row).  The arm-label grammar the model-error sweep is written in lives in
``test_condition_grammar.py``.
"""

import json

import pytest

from evaluation.runner import EvalConfig, _parse_ox_kbeta
from evaluation.sqlite_logging import DecisionRow, ExperimentLog, GameRow


# --------------------------------------------------------------------------- #
# The OX condition arm (label ↔ behaviour).
# --------------------------------------------------------------------------- #
def test_for_condition_ox_arm():
    ox = EvalConfig.for_condition("OX(k_beta=3.0)", run_id="r")
    assert ox.search_enabled is True          # OX searches
    assert ox.model_scope == "belief_only"    # reach-only: the model never reaches the solve
    assert ox.k_beta== 3.0                     # β threaded from the label

    # β variants parse; β=0 is a valid (naive-search) arm.
    assert _parse_ox_kbeta("OX(k_beta=0)") == 0.0
    assert _parse_ox_kbeta("ox(k_beta=1e2)") == 100.0

    # A BARE 'OX' (no explicit β) takes the code default (paper-derived gadget mix).
    from evaluation.runner import DEFAULT_OX_KBETA
    assert _parse_ox_kbeta("OX") == DEFAULT_OX_KBETA
    assert EvalConfig.for_condition("OX", run_id="r").k_beta== DEFAULT_OX_KBETA


def test_ox_arm_ignores_the_run_wide_dbr_knobs():
    from evaluation.opponents import ModelSpec

    # OX consumes no DBR machinery.  The run-wide defaults are shared by every arm of a
    # sweep, so a p_max/confidence in them must be *ignored* by an OX arm (not adopted,
    # which would run it as DBR) — while its error/seed, which shape σ̂ itself, are kept.
    defaults = ModelSpec(p_max=0.5, confidence=0.3, error=0.2, seed=7)
    ox = EvalConfig.for_condition("OX(k_beta=1)", model_spec=defaults, run_id="r")
    assert ox.model_scope == "belief_only"        # belief likelihood ONLY, no clamp
    assert (ox.model_spec.error, ox.model_spec.seed) == (0.2, 7)
    assert (ox.model_spec.p_max, ox.model_spec.confidence) == (1.0, 1.0)


def test_non_ox_arms_leave_k_beta_none():
    assert EvalConfig.for_condition("vanilla", run_id="r").k_beta is None
    assert EvalConfig.for_condition("blueprint_only", run_id="r").k_beta is None
    from evaluation.opponents import ModelSpec
    dbr = EvalConfig.for_condition(
        "DBR(p_max=1)", model_spec=ModelSpec(p_max=1.0), run_id="r"
    )
    assert dbr.k_beta is None and dbr.model_spec is not None
    assert dbr.model_scope == "full"           # DBR: belief AND clamp


def test_ox_k_beta_default_comes_from_the_run_when_the_label_omits_it():
    """β is held FIXED across an error sweep, so it belongs on the run, not each arm."""
    ox = EvalConfig.for_condition("OX(error=0.2)", ox_k_beta=0.02, run_id="r")
    assert ox.k_beta== 0.02
    # An explicit label still wins over the run-wide default.
    assert EvalConfig.for_condition("OX(k_beta=3)", ox_k_beta=0.02, run_id="r").k_beta== 3.0


# --------------------------------------------------------------------------- #
# The ox_enter_prob decision column (v7 schema round-trip).
# --------------------------------------------------------------------------- #
def _game(**kw):
    base = dict(
        run_id="run", hand_index=0, config_fingerprint="abc", table_label="t",
        table_config=json.dumps({"seats": ["bp", "bp"]}), hero_seat=0,
        button_seat=1, n_players=2, big_blind=100.0, starting_stack=20000.0,
        deck_seed=1,
    )
    base.update(kw)
    return GameRow(**base)


def test_ox_enter_prob_round_trips(tmp_path):
    """A searched OX decision persists its opt-out saturation; a non-OX row is NULL."""
    lg = ExperimentLog.open(tmp_path / "run.sqlite")
    try:
        with lg.game():
            gid = lg.log_game(_game())
            ox_id = lg.log_decision(
                gid,
                DecisionRow(
                    betting_stage="turn", regime="vector", searched=1,
                    num_live=2, n_live=2,
                    action_played="call", ox_enter_prob=0.42,
                ),
            )
            van_id = lg.log_decision(
                gid,
                DecisionRow(
                    betting_stage="turn", regime="vector", searched=1,
                    num_live=2, n_live=2,
                    action_played="call",            # vanilla: ox_enter_prob defaults None
                ),
            )
        con = lg._con
        got_ox = con.execute(
            "SELECT ox_enter_prob FROM decisions WHERE decision_id=?", (ox_id,)
        ).fetchone()[0]
        got_van = con.execute(
            "SELECT ox_enter_prob FROM decisions WHERE decision_id=?", (van_id,)
        ).fetchone()[0]
        assert got_ox == pytest.approx(0.42)
        assert got_van is None
    finally:
        lg.close()
