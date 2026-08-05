"""OX-Search (Approach B) eval-wiring gates (§11.3 step 12b).

Covers the harness plumbing that turns the gadget solver into a runnable, logged
experiment arm: the ``OX(beta=X)`` condition (reach-only — search on, β set, NO
model) and the per-decision ``ox_enter_prob`` opt-out-saturation column
(round-trips through the v7 schema; NULL for every non-OX row).
"""

import json

import pytest

from evaluation.runner import EvalConfig, _parse_ox_beta
from evaluation.sqlite_logging import DecisionRow, ExperimentLog, GameRow


# --------------------------------------------------------------------------- #
# The OX condition arm (label ↔ behaviour).
# --------------------------------------------------------------------------- #
def test_for_condition_ox_arm():
    ox = EvalConfig.for_condition("OX(beta=3.0)", run_id="r")
    assert ox.search_enabled is True          # OX searches
    assert ox.model_spec is None              # reach-only: NO opponent model
    assert ox.beta == 3.0                     # β threaded from the label

    # β variants parse; β=0 is a valid (naive-search) arm.
    assert _parse_ox_beta("OX(beta=0)") == 0.0
    assert _parse_ox_beta("ox(beta=1e2)") == 100.0

    # A BARE 'OX' (no explicit β) takes the code default (paper-derived gadget mix).
    from evaluation.runner import DEFAULT_OX_BETA
    assert _parse_ox_beta("OX") == DEFAULT_OX_BETA
    assert EvalConfig.for_condition("OX", run_id="r").beta == DEFAULT_OX_BETA


def test_ox_arm_rejects_model_and_malformed_beta():
    from evaluation.opponents import ModelSpec

    # OX consumes no DBR machinery — a model_spec on an OX arm is a wiring mistake.
    with pytest.raises(ValueError):
        EvalConfig.for_condition(
            "OX(beta=1)", model_spec=ModelSpec(p_max=1.0), run_id="r"
        )
    # Bare 'OX' now defaults, but a label that LOOKS like a (botched) β spec is a typo
    # and must fail loudly rather than silently defaulting.
    for bad in ("OX(beta=)", "OX(3.0)"):
        with pytest.raises(ValueError):
            EvalConfig.for_condition(bad, run_id="r")


def test_non_ox_arms_leave_beta_none():
    assert EvalConfig.for_condition("vanilla", run_id="r").beta is None
    assert EvalConfig.for_condition("blueprint_only", run_id="r").beta is None
    from evaluation.opponents import ModelSpec
    dbr = EvalConfig.for_condition(
        "DBR(p_max=1)", model_spec=ModelSpec(p_max=1.0), run_id="r"
    )
    assert dbr.beta is None and dbr.model_spec is not None


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
