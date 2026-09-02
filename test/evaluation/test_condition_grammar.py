"""The experiment-arm label grammar — what a model-error sweep is written in (§10.1).

The eval's headline question is *how good must the opponent model be* for each
exploitation approach to gain, so a run is a list of arms that differ **only** in
model quality:

    vanilla ; DBR(confidence=0.8,error=0.2) ; DBR(confidence=0.8,error=0.3)
            ; OX(error=0.2) ; OX(error=0.3)

That requires two things this module gates: every knob is expressible *per arm* in the
label, and everything a label omits (β, ``p_max``, …) falls back to one run-wide
default so it is provably identical across the arms being compared.
"""

import pytest

from evaluation.opponents import ModelSpec
from evaluation.runner import (
    DEFAULT_DBR_P_MAX,
    DEFAULT_OX_KBETA,
    EvalConfig,
    _parse_condition,
    split_conditions,
)


# --------------------------------------------------------------------------- #
# Splitting a multi-arm string
# --------------------------------------------------------------------------- #

def test_split_conditions_does_not_tear_a_parameterised_label():
    """A label's own commas are its parameters — only a TOP-LEVEL comma separates arms.

    The naive ``split(',')`` turns 'DBR(confidence=0.8,error=0.2)' into two fragments,
    both nonsense, and the second would parse as a fresh (unparameterised) arm.
    """
    raw = "vanilla,DBR(confidence=0.8,error=0.2),OX(k_beta=0.05,error=0.3)"
    assert split_conditions(raw) == [
        "vanilla",
        "DBR(confidence=0.8,error=0.2)",
        "OX(k_beta=0.05,error=0.3)",
    ]


def test_split_conditions_accepts_semicolons_and_trims():
    assert split_conditions(" vanilla ; DBR(error=0.2) ;; OX ") == [
        "vanilla", "DBR(error=0.2)", "OX",
    ]
    assert split_conditions("") == []


# --------------------------------------------------------------------------- #
# Label parsing
# --------------------------------------------------------------------------- #

def test_parse_condition_kinds_and_params():
    assert _parse_condition("vanilla") == ("vanilla", {})
    assert _parse_condition(" OX( k_beta=0.05 , error=0.2 ) ") == (
        "ox", {"k_beta": 0.05, "error": 0.2},
    )
    assert _parse_condition("DBR")[0] == "dbr"
    assert _parse_condition("blueprint_only") == ("blueprint_only", {})


@pytest.mark.parametrize("bad_name", ["naive_br", "OXX(error=0.2)", "oxsearch", "typo"])
def test_an_unknown_arm_name_is_rejected_not_silently_run_as_dbr(bad_name):
    """A mistyped approach name must not become a different approach.

    The parameter vocabulary already stops a mistyped *knob*; a mistyped *name* is
    worse, because the arm still runs and the results table still reports it under the
    name that was asked for — so an approach comparison silently compares the wrong
    thing.  Only the four names in ``_ARM_PARAMS`` are arms.
    """
    with pytest.raises(ValueError, match="unknown arm"):
        _parse_condition(bad_name)


@pytest.mark.parametrize("bad", [
    "DBR(error=)",              # no value
    "DBR(0.2)",                 # no key
    "DBR(error=oops)",          # not a number
    "DBR(error=0.1,error=0.2)", # duplicated
    "DBR(erorr=0.2)",           # typo'd key
    "vanilla(error=0.2)",       # vanilla takes nothing
    "OX(k_beta=0.05",             # unbalanced
    "",
])
def test_malformed_labels_raise(bad):
    """A mistyped knob must not run a whole arm at the wrong model quality in silence."""
    with pytest.raises(ValueError):
        _parse_condition(bad)


# --------------------------------------------------------------------------- #
# Defaults vs per-arm overrides — the sweep itself
# --------------------------------------------------------------------------- #

def test_error_sweep_arms_differ_only_in_error():
    """The headline use: one defaults bundle, arms varying the model-quality axis."""
    defaults = ModelSpec(p_max=0.9, confidence=0.8, error=0.0, seed=3)
    arms = [
        EvalConfig.for_condition(c, model_spec=defaults, ox_k_beta=0.02, run_id="r")
        for c in split_conditions(
            "vanilla;DBR(error=0.2);DBR(error=0.3);OX(error=0.2);OX(error=0.3)"
        )
    ]
    vanilla, dbr2, dbr3, ox2, ox3 = arms

    assert vanilla.model_spec is None and vanilla.k_beta is None   # THE baseline
    # DBR arms: the label sets error; everything else comes from the shared defaults.
    for arm, err in ((dbr2, 0.2), (dbr3, 0.3)):
        assert arm.model_scope == "full"
        assert arm.model_spec.error == err
        assert arm.model_spec.p_max == 0.9
        assert arm.model_spec.confidence == 0.8
        assert arm.model_spec.seed == 3
    # OX arms: same error axis, same β, and the clamp knobs left inert (reach-only).
    for arm, err in ((ox2, 0.2), (ox3, 0.3)):
        assert arm.model_scope == "belief_only"
        assert arm.k_beta== 0.02
        assert arm.model_spec.error == err
        assert (arm.model_spec.p_max, arm.model_spec.confidence) == (1.0, 1.0)

    # Arms of one sweep must be distinguishable downstream, or the summary pools them.
    fps = [a.fingerprint_table_policy() for a in arms]
    assert len({repr(f) for f in fps}) == len(arms)


def test_dbr_p_max_and_ox_k_beta_have_defaults():
    """Neither knob has to be named — that is what makes the sweep list short."""
    dbr = EvalConfig.for_condition("DBR(error=0.2)", run_id="r")
    assert dbr.model_spec.p_max == DEFAULT_DBR_P_MAX
    assert EvalConfig.for_condition("OX", run_id="r").k_beta== DEFAULT_OX_KBETA


def test_label_scalar_and_run_schedule_conflict_is_rejected():
    """A schedule silently OVERRIDES the scalar in ``ModelSpec.resolve``.

    So a per-arm ``error=`` alongside a run-wide error schedule would drop the very
    value the sweep is built around, and every arm would run at the same quality —
    invisible in the results.  Refuse the combination instead.
    """
    sched = ModelSpec(error_schedule='{"kind":"street","by_round":{"0":0.05}}')
    with pytest.raises(ValueError):
        EvalConfig.for_condition("DBR(error=0.2)", model_spec=sched, run_id="r")
    with pytest.raises(ValueError):
        EvalConfig.for_condition("OX(error=0.2)", model_spec=sched, run_id="r")
    # Without a per-arm scalar the schedule is honoured, for OX's belief too.
    ox = EvalConfig.for_condition("OX", model_spec=sched, run_id="r")
    assert ox.model_spec is not None and ox.model_spec.error_schedule


def test_every_exploiting_arm_carries_a_model_at_zero_error():
    """The e=0 ceiling must mean the same thing on both curves.

    Both approaches need a model — they differ in how they *use* it (DBR clamps the
    solve, OX only shapes the beliefs).  So at error=0 both must hold the EXACT model,
    or the sweep's ceiling arm compares a perfect-model DBR against a no-model OX.

    Dropping the spec for OX would not even give a *neutral* arm: it falls back to the
    agent's unmodeled likelihood, which reads the blueprint at bias "none" (never the
    seat's actual bp_fold/bp_call/bp_raise bias) and prefers the last search's average
    policy whenever a search ran that round — a DIFFERENT distribution, not a perfect
    one, so the OX curve's ceiling would not mean what DBR's does.
    """
    exact = ModelSpec(error=0.0)
    for label in ("DBR", "DBR(error=0)", "OX", "OX(error=0)"):
        arm = EvalConfig.for_condition(label, model_spec=exact, run_id="r")
        assert arm.model_spec is not None, label
        assert arm.model_spec.error == 0.0, label
    # ...and only the baseline is genuinely model-free.
    assert EvalConfig.for_condition("vanilla", model_spec=exact, run_id="r").model_spec is None
