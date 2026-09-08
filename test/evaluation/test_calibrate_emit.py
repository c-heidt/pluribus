"""``run_calibration``'s final emit must be JSON-encodable.

The suggested-config block is keyed the way production reads it —
``(street, n_live)`` tuples — and ``json.dumps`` refuses tuple keys outright.  That
failure lands on the LAST line of a calibration, after every solve is already paid
for, so it costs the whole run's wall-clock rather than just the file; it did exactly
that once, an hour in.  ``_jsonable`` is what stands between the two, and the shipped
budget tables alone are enough to trigger it, so this needs no measured summaries.
"""

import json

from poker_ai.search.solver_state import DBR, OX, VANILLA

from evaluation.calibrate import (
    CellSummary,
    _jsonable,
    _shipped_budget,
    suggest_config,
)


def test_suggested_config_encodes_with_no_measurements():
    """Even an empty sweep emits the shipped tables — tuple-keyed, and they must encode."""
    config = suggest_config([])
    # Guard the premise: if these ever stop being tuple-keyed, this test is testing air.
    assert any(isinstance(k, tuple) for k in config["mccfr_budget"]["vanilla"])

    text = json.dumps(_jsonable({"suggested_config": config}), indent=2)
    back = json.loads(text)["suggested_config"]

    # Keys survive as something a reader can act on, not as ``str(tuple)``.
    assert "flop/n2" in back["mccfr_budget"]["vanilla"]
    assert back["mccfr_budget"]["vanilla"]["flop/n2"] == (
        config["mccfr_budget"]["vanilla"][(1, 2)]
    )


def test_jsonable_leaves_already_stringified_keys_alone():
    """The ladders block stringifies its own keys; converting twice would mangle them."""
    assert _jsonable({"mccfr/1/n2": [10, 20]}) == {"mccfr/1/n2": [10, 20]}


def test_jsonable_recurses_through_lists():
    """Cells are a LIST of dicts, so the walk cannot stop at the first non-dict."""
    assert _jsonable([{(1, 2): 3}]) == [{"flop/n2": 3}]


# --------------------------------------------------------------------------- #
# OX-Search's emitted budget row
#
# OX is not a DBR variant.  Its gadget is live only in the 2-player VECTOR cells;
# everywhere else it falls back to vanilla and runs vanilla's code, so vanilla is what
# its row follows.  ``suggest_config`` used to copy DBR's whole vector row over OX's
# *after* folding in the measurements, which threw away everything an OX arm measured —
# a run that cost hours emitted the numbers it started with, no error, no warning.
# --------------------------------------------------------------------------- #


def _cell(cell, budget):
    ladder = [1000, 2000, 4000]
    return CellSummary(
        cell=cell, ladder=ladder,
        mean_value_gap_mbb={t: 0.1 for t in ladder},
        mean_hot_l1={t: 0.05 for t in ladder},
        argmax_stability={t: 0.9 for t in ladder},
        mean_l1={t: 0.1 for t in ladder},
        mean_wall={t: t / 100.0 for t in ladder},
        throughput_it_s=100.0, pooled_it_s=100.0,
        replica_spread_mbb=float("nan"), suggested_budget=budget, converged=True,
        below_ladder=False, hot_l1_tol=0.1, cross_rep_hot_l1=float("nan"),
        tol_below_cross_rep=False, n_samples=3, workers=1,
    )


def test_ox_run_emits_what_it_measured():
    """The 2-player vector cells an OX arm measures must reach the emitted block."""
    config = suggest_config([
        _cell(("OX(k_beta=50)", "vector", 2, 2), 2000),   # HU turn
        _cell(("OX(k_beta=50)", "vector", 3, 2), 1000),   # HU river
    ])
    assert config["vector_budget"][OX][(2, 2)] == 2000
    assert config["vector_budget"][OX][(3, 2)] == 1000


def test_ox_falls_back_to_vanilla_off_the_gadget_not_to_dbr():
    """Unmeasured cells take VANILLA's budget — OX plays vanilla wherever it falls back."""
    config = suggest_config([_cell(("OX(k_beta=50)", "vector", 2, 2), 2000)])
    vector, mccfr = config["vector_budget"], config["mccfr_budget"]
    # every cell this run did not measure
    assert vector[OX][(3, 2)] == vector[VANILLA][(3, 2)]
    assert mccfr[OX] == mccfr[VANILLA]
    # and the premise that makes this test meaningful: vanilla and DBR really differ,
    # so "OX == vanilla" cannot be passing by coincidence.
    assert mccfr[VANILLA] != mccfr[DBR]


def test_ox_tracks_a_freshly_measured_vanilla_not_the_shipped_one():
    """A vanilla arm's new numbers propagate into OX's row, not the stale table."""
    config = suggest_config([_cell(("vanilla", "mccfr", 1, 3), 7777)])
    assert config["mccfr_budget"][VANILLA][(1, 3)] == config["mccfr_budget"][OX][(1, 3)]
    assert config["mccfr_budget"][OX][(1, 3)] != _shipped_budget("mccfr")[VANILLA][(1, 3)]


def test_a_dbr_run_does_not_move_the_ox_row():
    """Nothing about DBR feeds OX — that coupling is what produced the silent discard."""
    config = suggest_config([_cell(("DBR", "vector", 2, 2), 9999)])
    assert config["vector_budget"][DBR][(2, 2)] == 10000       # measured, rounded up
    assert config["vector_budget"][OX][(2, 2)] == config["vector_budget"][VANILLA][(2, 2)]
