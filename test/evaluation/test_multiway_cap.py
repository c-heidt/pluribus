"""Peak-RAM cap on concurrent multiway solves (``sweep_jobs(max_concurrent_multiway=K)``).

The sweep hands out the most expensive solves FIRST (longest-processing-time), so without
a cap every worker grabs a multiway MCCFR solve at once — the RAM spike that OOMs the node.
Two things must hold, and both are pinned here against the real code:

- **classification** — a 3-way (``n_live == 3``) MCCFR cell counts as multiway/heavy, as
  does 4-way; heads-up MCCFR and the vector cells do not (they hold small tables);
- **the limit is actually held** — under a real forked pool, the number of heavy solves
  running *simultaneously* never exceeds ``K``.

``solve`` is monkeypatched to a cheap sleeper, so the REAL ``_sweep_process`` gating and
the REAL ``sweep_jobs`` scheduling/interleaving run without paying for actual CFR.  The
concurrency tally lives in shared memory (``mp.Value``) so the forked workers all update
the same counter.
"""

import dataclasses
import multiprocessing as mp
import time
from typing import Optional

import numpy as np
import pytest

from evaluation import calibrate


# --------------------------------------------------------------------------- #
# Minimal stand-ins: enough shape for _sweep_process, none of the cost
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class _Ctx:
    rng: object = None


@dataclasses.dataclass
class _Cfg:
    auto_budget: bool = True
    max_iterations: int = 100
    max_wall_seconds: float = 1.0


class _Sample:
    """A RootSample-shaped stub (``sweep_jobs`` only reads these fields)."""

    def __init__(self, condition, regime, street, n_live):
        self.condition, self.regime = condition, regime
        self.street, self.n_live = street, n_live
        self.env = {"regime": regime, "n_live": n_live}   # deepcopy-able
        self.ctx = _Ctx()
        self.pk, self.hr, self.legal = ("pk", street, n_live), 0, ["fold", "call"]

    @property
    def cell(self):
        return (self.condition, self.regime, int(self.street), int(self.n_live))


class _Res:
    """A SearchResult-shaped stub; ``_root_sigma`` reads legal_at/average_policy."""

    def __init__(self, pk):
        self.root_value = 1.0
        self.iterations_run = 10
        self.wall_seconds = 0.01
        self.stop_reason = "iteration_cap"
        self.state = type("S", (), {"legal_at": {pk: True}})()
        self.average_policy = type(
            "P", (), {"strategy_for": lambda self, pk, hr, legal: np.array([0.5, 0.5])}
        )()


# Shared across the forked pool workers (module globals are inherited by fork).
_TRACK: dict = {}


class _FakePolicy:
    """Stands in for the live ``SearchPolicy`` handed to an ``on_snapshot`` hook."""

    def __init__(self, pk):
        self._state = type("S", (), {"legal_at": {pk: True}})()

    def strategy_for(self, pk, hr, legal):
        return np.array([0.5, 0.5])


def _fake_solve(env, ctx, cfg, regime_override=None, snapshot_at=None, on_snapshot=None):
    """Cheap stand-in for ``solve`` that records heavy-solve concurrency.

    Called from INSIDE ``_sweep_process``'s semaphore-guarded block, so the peak it
    records is exactly the quantity the cap is supposed to bound.  Fires ``on_snapshot``
    once per requested rung, as the real ``solve`` does.
    """
    regime = regime_override or env["regime"]
    heavy = (regime == "mccfr" and int(env["n_live"]) >= 3)
    if heavy:
        with _TRACK["lock"]:
            _TRACK["cur"].value += 1
            if _TRACK["cur"].value > _TRACK["peak"].value:
                _TRACK["peak"].value = _TRACK["cur"].value
        time.sleep(0.05)          # long enough for workers to genuinely overlap
        with _TRACK["lock"]:
            _TRACK["cur"].value -= 1
    if on_snapshot is not None:
        for t in (snapshot_at or ()):
            on_snapshot(int(t), _FakePolicy(("pk", 0, 0)), 0.01)
    return _Res(("pk", 0, 0))


def _jobs():
    """One job per cell of a 4p post-flop grid (3 heavy + 3 light), 4 roots each."""
    specs = [
        ("mccfr", 1, 2), ("mccfr", 1, 3), ("mccfr", 1, 4),      # flop: n3/n4 heavy
        ("mccfr", 2, 3), ("vector", 2, 2), ("vector", 3, 2),    # turn n3 heavy; vector light
    ]
    out = []
    for regime, street, n_live in specs:
        samples = [_Sample("DBR", regime, street, n_live) for _ in range(4)]
        out.append(dict(cell=samples[0].cell, samples=samples,
                        force_regime=None, ladder=[100, 200], ref_from=None))
    return out


def _run(max_concurrent_multiway: Optional[int], workers: int, monkeypatch):
    monkeypatch.setattr(calibrate, "solve", _fake_solve)
    # _root_sigma needs the sample's pk in legal_at; give every _Res the right key.
    monkeypatch.setattr(calibrate, "_root_sigma",
                        lambda res, s: np.array([0.5, 0.5]))
    ctx = mp.get_context("fork")
    _TRACK["lock"] = ctx.Lock()
    _TRACK["cur"] = ctx.Value("i", 0)
    _TRACK["peak"] = ctx.Value("i", 0)
    rows = calibrate.sweep_jobs(
        _jobs(), _Cfg(), pool_workers=workers, reps=2, base_seed=0, big_blind=100,
        max_concurrent_multiway=max_concurrent_multiway,
    )
    return rows, int(_TRACK["peak"].value)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("regime,n_live,heavy", [
    ("mccfr", 2, False),    # heads-up MCCFR — small tables
    ("mccfr", 3, True),     # 3-way IS multiway
    ("mccfr", 4, True),     # 4-way
    ("vector", 2, False),   # vector cells are enumeration, not the RAM hog
])
def test_multiway_classification(regime, n_live, heavy):
    """The predicate ``sweep_jobs`` gates on: MCCFR with >= 3 live players.

    Mirrors the inline ``_is_multiway``; guards the cell-tuple indices
    (``cell = (condition, regime, street, n_live)`` ⇒ regime is [1], n_live is [3]).
    """
    cell = ("DBR", regime, 1, n_live)
    assert (cell[1] == "mccfr" and int(cell[3]) >= 3) is heavy


# --------------------------------------------------------------------------- #
# The cap is actually enforced
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 2, 3])
def test_concurrency_never_exceeds_cap(k, monkeypatch):
    rows, peak = _run(k, workers=8, monkeypatch=monkeypatch)
    assert peak <= k, f"{peak} heavy solves ran at once, cap was {k}"
    assert rows, "the sweep produced no rows"


def test_uncapped_actually_overlaps(monkeypatch):
    """Sanity: without a cap the heavy solves DO run concurrently.

    Without this, the cap tests above could pass trivially (e.g. if the pool serialised
    everything anyway) and prove nothing.
    """
    _, peak = _run(None, workers=8, monkeypatch=monkeypatch)
    assert peak >= 2, f"expected overlapping heavy solves uncapped, saw peak={peak}"


def test_cap_preserves_every_spec(monkeypatch):
    """Capping/interleaving must not drop or duplicate work — same rows either way."""
    capped, _ = _run(2, workers=8, monkeypatch=monkeypatch)
    uncapped, _ = _run(None, workers=8, monkeypatch=monkeypatch)
    key = lambda r: (r.cell, r.sample, r.rep, r.per_replica)
    assert sorted(map(key, capped)) == sorted(map(key, uncapped))
