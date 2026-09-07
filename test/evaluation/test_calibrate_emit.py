"""``run_calibration``'s final emit must be JSON-encodable.

The suggested-config block is keyed the way production reads it —
``(street, n_live)`` tuples — and ``json.dumps`` refuses tuple keys outright.  That
failure lands on the LAST line of a calibration, after every solve is already paid
for, so it costs the whole run's wall-clock rather than just the file; it did exactly
that once, an hour in.  ``_jsonable`` is what stands between the two, and the shipped
budget tables alone are enough to trigger it, so this needs no measured summaries.
"""

import json

from evaluation.calibrate import _jsonable, suggest_config


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
