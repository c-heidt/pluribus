"""One calibration, every arm — each solving under its OWN solver config.

The arms differ in exactly one solver knob: OX-Search sets ``kβ`` (its gadget root),
vanilla and DBR leave it unset.  Calibrate used to build a single shared ``SolverConfig``
from ``conditions[0]``, so mixing an OX arm with vanilla/DBR was refused outright and OX
had to run as a second invocation writing a second output directory.  Now the config
travels on the job, so all arms land in one ``calibration_rows.csv`` / summary.

What has to hold, and is pinned here:

- a job solves under its own arm's config — an OX job with the gadget, a vanilla job
  beside it without.  Getting this wrong is silent: every arm would be measured as
  whichever one happened to be first in ``--conditions``;
- a job that carries no config falls back to the run-wide one (the shape
  ``test_multiway_cap`` builds);
- the rows the sweep returns keep their condition, so the shared CSV separates the arms.
"""

import dataclasses

import numpy as np

from evaluation import calibrate
from evaluation.calibrate import _ladder_key
from test.evaluation.test_multiway_cap import _Cfg, _FakePolicy, _Res, _Sample


@dataclasses.dataclass
class _OxCfg(_Cfg):
    """A solver-config stub carrying the one knob the arms differ in."""

    ox_kbeta: float = None


_SEEN: dict = {}


def _recording_solve(env, ctx, cfg, regime_override=None, snapshot_at=None,
                     on_snapshot=None):
    """Record the kβ each solve actually ran with, keyed by the root it solved."""
    _SEEN.setdefault(env["arm"], set()).add(getattr(cfg, "ox_kbeta", None))
    pk = ("pk", 1, 2)
    if on_snapshot is not None:
        for t in (snapshot_at or []):
            on_snapshot(int(t), _FakePolicy(pk), 0.01)
    return _Res(pk)


def _job(arm, cfg, *, n_roots=1):
    samples = [_Sample(arm, "vector", 2, 2) for _ in range(n_roots)]
    for s in samples:
        s.env = {"regime": "vector", "n_live": 2, "arm": arm}
    job = dict(cell=samples[0].cell, samples=samples, force_regime=None,
               ladder=[100, 200], ref_from=None)
    if cfg is not None:
        job["cfg"] = cfg
    return job


def _sweep(jobs, monkeypatch, default_cfg):
    _SEEN.clear()
    monkeypatch.setattr(calibrate, "solve", _recording_solve)
    return calibrate.sweep_jobs(jobs, default_cfg, pool_workers=1, reps=1,
                                base_seed=0, big_blind=100)


def test_each_arm_solves_under_its_own_config(monkeypatch):
    """The OX job gets the gadget; the vanilla job next to it does not."""
    vanilla_cfg = _OxCfg(ox_kbeta=None)
    ox_cfg = _OxCfg(ox_kbeta=50.0)
    rows = _sweep([_job("vanilla", vanilla_cfg), _job("OX(k_beta=50)", ox_cfg)],
                  monkeypatch, vanilla_cfg)

    assert _SEEN["vanilla"] == {None}, "a vanilla job was solved WITH the OX gadget"
    assert _SEEN["OX(k_beta=50)"] == {50.0}, "an OX job was solved without its gadget"
    # Both arms are present in one result set — the shared CSV/summary depends on it.
    assert {r.condition for r in rows} == {"vanilla", "OX(k_beta=50)"}


def test_a_job_without_a_config_uses_the_run_wide_one(monkeypatch):
    """The pre-existing job shape keeps working — no config means the default."""
    default = _OxCfg(ox_kbeta=7.0)
    _sweep([_job("vanilla", None)], monkeypatch, default)
    assert _SEEN["vanilla"] == {7.0}


def test_rows_from_one_sweep_keep_their_arm(monkeypatch):
    """Rows carry the condition, so one CSV can hold every arm without ambiguity."""
    cfg = _OxCfg()
    rows = _sweep([_job("vanilla", cfg), _job("DBR", cfg), _job("OX(k_beta=50)", cfg)],
                  monkeypatch, cfg)
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r.condition, []).append(r)
    assert set(by_arm) == {"vanilla", "DBR", "OX(k_beta=50)"}
    assert all(rs for rs in by_arm.values())


# --------------------------------------------------------------------------- #
# Wall-anchored ladders are per (cell, approach)
#
# A ladder's top rung is the deepest solve that fits the cell's wall budget at the
# measured throughput, so it is only correct for the arm it was probed under — and the
# arms do not run at the same rate (DBR carries the model clamp and VR-MCCFR; OX solves a
# larger gadget root).  Keying on the cell alone handed every arm the ladder probed for
# whichever condition came FIRST in --conditions.
# --------------------------------------------------------------------------- #
def _cell_job(condition, regime="vector", street=2, n_live=2, force_regime=None):
    return {"cell": (condition, regime, street, n_live), "force_regime": force_regime}


def test_each_approach_gets_its_own_ladder():
    """vanilla / DBR / OX on the SAME cell must not share a wall anchor."""
    keys = {c: _ladder_key(_cell_job(c))
            for c in ("vanilla", "DBR", "OX(k_beta=50)")}
    assert len(set(keys.values())) == 3, f"approaches collapsed onto one ladder: {keys}"


def test_arms_of_one_approach_still_share_a_ladder():
    """A DBR error sweep differs in model noise, not in what a solve costs."""
    assert _ladder_key(_cell_job("DBR(error=0.2)")) == \
           _ladder_key(_cell_job("DBR(error=0.3)"))


def test_ladder_key_still_separates_cells_and_forced_regimes():
    """The cell identity it always carried must survive the added approach axis."""
    assert _ladder_key(_cell_job("vanilla", street=2)) != \
           _ladder_key(_cell_job("vanilla", street=3))
    assert _ladder_key(_cell_job("vanilla", n_live=2)) != \
           _ladder_key(_cell_job("vanilla", n_live=3))
    # the regime A/B solves one cell under both regimes against a shared reference
    assert _ladder_key(_cell_job("vanilla", force_regime="vector")) != \
           _ladder_key(_cell_job("vanilla", force_regime="mccfr"))
