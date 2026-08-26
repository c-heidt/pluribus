"""The parallel eval path must be bit-reproducible and worker-count independent.

``evaluation.runner.run_evaluation_parallel`` (doc §10.1) forks ``n_workers``, each
pulling hand indices off a shared dynamic counter and logging to its own node-local
DB, which the parent then merges.  Its contract is that **every hand is fully
determined by ``(run_seed, hand_index)``** — deal, seating, and all five seed
sub-streams — so the merged result is identical to a serial run and identical for
any worker count, and the §10.1 CRN pairing across conditions still holds.

That contract is load-bearing for every number the evaluation reports, and until now
nothing exercised it: no test referenced ``run_evaluation_parallel`` or
``hand_pool``.  It is also exactly the property that silently breaks if any hand
starts depending on process-level state — a module-global RNG read outside the
per-hand reseed, a cached solver keyed across hands, a fork-inherited handle.  Such
a break is invisible in a serial run and shows up only as irreproducible cluster
results, so it is pinned here rather than assumed.

⚠️ **The contract is conditional, and this module gates only the half that holds.**
Reproducibility survives the fork because no hand depends on execution order — but a
solve stopped by ``SolverConfig.max_wall_seconds`` completes a *load-dependent*
number of iterations, so a wall-capped run reproduces neither across worker counts
nor across repeat runs (measured: 32/40 identical hands, with the batch's wall-cap
share swinging 16→42 of 55 searches between two runs of one script).  The fixture
here is deliberately iteration-bound — ``_stub_session`` solves at
``max_iterations=4`` — and :meth:`TestWorkerCountIndependence.test_fixture_is_iteration_bound`
asserts it, so if the fixture ever gains a binding wall cap the suite says *why*
these tests stopped meaning anything instead of flaking on a busy machine.

Hand-level determinism is the *reason* the pool can hand indices out dynamically:
because no hand depends on which worker ran it or on what ran before it, cores can
be filled greedily (most hands fold pre-flop; a few are long turn solves).

The arms here are the real experiment **conditions** (``vanilla`` — search, no model,
THE baseline — and a ``DBR`` arm — search *with* an opponent model), configured the
way the runner configures them: ``search_enabled`` + ``model_spec``, not some proxy
knob.  What is asserted across them is only the *deal side* — same cards, same
seating. Play is expected to differ; that difference is the measurement.

Fast: stub session, small deck, tiny solver budget — a handful of seconds.
"""

import dataclasses
import os

import pytest

from evaluation.opponents import ModelSpec
from evaluation.runner import run_evaluation, run_evaluation_parallel
from evaluation.sqlite_logging import ExperimentLog
from test.evaluation.test_aivat import _NonUniformPolicy
from test.evaluation.test_runner import _stub_session

_N_HANDS = 20
_SEED = 17
_DBR_SPEC = ModelSpec(p_max=0.9, error=0.1, seed=3)


def _session(run_id, *, condition="vanilla", model_spec=None, aivat=False):
    """A stub session wearing one experiment condition.

    Mirrors ``EvalConfig.for_condition``: ``vanilla`` is search with **no** model,
    a ``DBR(...)`` arm is search **with** one.  Both fields are set together so the
    label can never disagree with the behaviour.
    """
    session = _stub_session(run_id=run_id, run_seed=_SEED, n_players=4,
                            starting_stack=10000, blueprint=_NonUniformPolicy())
    session.config.condition = condition
    session.config.search_enabled = condition != "blueprint_only"
    session.config.model_spec = model_spec
    session.config.aivat = aivat
    session.config.aivat_rollouts = 3
    return session


def _read(path):
    """Everything a hand produces, keyed by ``hand_index``.

    Deliberately wide — outcome, cards, seating, the played action sequence and
    ``aivat_value`` — so a divergence anywhere is caught, not just one in the final
    chip count (two different lines can net the same chips).
    """
    log = ExperimentLog.open(os.fspath(path))
    try:
        rows = log._con.execute(
            "SELECT g.hand_index, g.hero_chips_delta, g.aivat_value, g.hero_hole, "
            "g.final_board, g.deck_seed, g.hero_seat, g.button_seat, "
            "(SELECT group_concat(d.action_played) FROM decisions d "
            "  WHERE d.game_id = g.game_id) AS acts, "
            "(SELECT group_concat(s.seat || ':' || s.agent_label, '|') "
            "  FROM (SELECT seat, agent_label FROM game_seats "
            "        WHERE game_id = g.game_id ORDER BY seat) s) AS seats, "
            "(SELECT COUNT(*) FROM decisions d "
            "  WHERE d.game_id = g.game_id AND d.modeled_decision = 1) AS n_modeled "
            "FROM games g ORDER BY g.hand_index"
        ).fetchall()
    finally:
        log.close()
    assert rows, "run produced no games"
    return {r[0]: tuple(r)[1:] for r in rows}


def _serial(tmp_path, run_id, **kw):
    path = tmp_path / f"{run_id}.sqlite"
    log = ExperimentLog.open(os.fspath(path))
    try:
        run_evaluation(log=log, session=_session(run_id, **kw), max_hands=_N_HANDS)
    finally:
        log.close()
    return _read(path)


def _parallel(tmp_path, run_id, workers, **kw):
    path = tmp_path / f"{run_id}.sqlite"
    ExperimentLog.open(os.fspath(path)).close()          # create the merge target
    run_evaluation_parallel(_session(run_id, **kw), path,
                            n_workers=workers, max_hands=_N_HANDS)
    return _read(path)


# Column offsets into the _read tuple, i.e. the SELECT order with hand_index stripped:
#   0 hero_chips_delta  1 aivat_value  2 hero_hole  3 final_board  4 deck_seed
#   5 hero_seat         6 button_seat  7 acts       8 seats        9 n_modeled
_CARDS = slice(2, 5)        # hero_hole, final_board, deck_seed
_SEATING = slice(5, 7)      # hero_seat, button_seat (the seats map is _SEATS)
_ACTS = 7
_SEATS = 8
_N_MODELED = 9


class TestWorkerCountIndependence:
    """A hand's result must not depend on how many workers ran the batch.

    Valid only while the solves are iteration-bound; see the module docstring and
    :meth:`test_fixture_is_iteration_bound`.
    """

    def test_fixture_is_iteration_bound(self, tmp_path):
        """No solve here may stop on the wall cap.

        The precondition for everything else in this class.  A wall-capped solve is
        load-dependent, so if the fixture ever became wall-bound these tests would
        start failing for a reason that has nothing to do with the pool — or worse,
        pass on a fast machine and fail on a busy one.
        """
        path = tmp_path / "wallcheck.sqlite"
        log = ExperimentLog.open(os.fspath(path))
        try:
            run_evaluation(log=log, session=_session("wallcheck"),
                           max_hands=_N_HANDS)
            stops = log._con.execute(
                "SELECT stop_reason, COUNT(*) FROM decisions WHERE searched = 1 "
                "GROUP BY stop_reason"
            ).fetchall()
        finally:
            log.close()
        by_reason = {r[0]: r[1] for r in stops}
        assert by_reason, "no searches ran — the reproducibility tests are vacuous"
        assert by_reason.get("wall_cap", 0) == 0, (
            f"fixture became wall-capped ({by_reason}); a wall-bound solve is "
            "load-dependent, so the worker-count tests below no longer test the pool"
        )

    def test_parallel_matches_serial(self, tmp_path):
        serial = _serial(tmp_path, "R_serial")
        par = _parallel(tmp_path, "R_w3", workers=3)
        assert set(serial) == set(par)
        assert serial == par

    def test_worker_counts_agree_with_each_other(self, tmp_path):
        assert _parallel(tmp_path, "R_w2", workers=2) == \
               _parallel(tmp_path, "R_w5", workers=5)

    def test_reproducible_across_repeat_runs(self, tmp_path):
        assert _parallel(tmp_path, "R_rep1", workers=4) == \
               _parallel(tmp_path, "R_rep2", workers=4)

    def test_holds_for_a_modeled_arm(self, tmp_path):
        """Determinism must survive the DBR machinery too — the models are rebuilt
        per hand from the just-drawn seat labels, so a stray RNG draw in there would
        make results worker-dependent."""
        assert _serial(tmp_path, "R_dbr_s", condition="DBR(p_max=0.9)",
                       model_spec=_DBR_SPEC) == \
               _parallel(tmp_path, "R_dbr_p", workers=4,
                         condition="DBR(p_max=0.9)", model_spec=_DBR_SPEC)

    def test_aivat_value_also_reproduces(self, tmp_path):
        """AIVAT runs on its own seed sub-stream (a 5th ``derive_seeds`` child), so
        the estimate must be worker-count independent too — otherwise the metric
        the headline is computed from would depend on the cluster's core count."""
        serial = _serial(tmp_path, "R_aiv_s", aivat=True)
        par = _parallel(tmp_path, "R_aiv_p", workers=4, aivat=True)
        assert all(v[1] is not None for v in serial.values())   # column populated
        assert serial == par


class TestCrnAcrossConditions:
    """The §10.1 pairing must survive the pool: two **conditions** sharing a
    ``run_seed`` see the same deal and the same seating on every ``hand_index``,
    whichever worker happens to pick it up.

    Only the deal side is asserted. The arms play differently — that is the whole
    point of comparing them — so nothing here constrains the action sequence.

    Note the deliberate absence of a "where the model never fired, play must match
    vanilla" assertion. It is *nearly* true and tempting, but the predicate is
    subtle: ``modeled_decision`` marks a decision that consulted the model, while a
    hand can still have had its solve shaped by the model without any logged played
    decision carrying the flag. Measured on this fixture the naive version holds on
    15 of 16 quiet hands — i.e. it would be a flaky assertion dressed as an exact
    one, so the exact deal-side invariants are gated instead.
    """

    def _arms(self, tmp_path):
        return (_parallel(tmp_path, "R_vanilla", workers=4, condition="vanilla"),
                _parallel(tmp_path, "R_model", workers=4,
                          condition="DBR(p_max=0.9)", model_spec=_DBR_SPEC))

    def test_same_cards_and_deck_seed(self, tmp_path):
        vanilla, dbr = self._arms(tmp_path)
        for k in sorted(set(vanilla) & set(dbr)):
            assert vanilla[k][_CARDS] == dbr[k][_CARDS], f"deal differs at hand {k}"

    def test_same_seating(self, tmp_path):
        """Seating is drawn from ``table_ss`` *before* the hero agent is built (see
        the CRN guard in ``_play_and_log_one``), and ``build_models`` is documented
        to add no RNG draw — so turning a model on must not move a single seat."""
        vanilla, dbr = self._arms(tmp_path)
        for k in sorted(set(vanilla) & set(dbr)):
            assert vanilla[k][_SEATING] == dbr[k][_SEATING], f"seats differ at {k}"
            assert vanilla[k][_SEATS] == dbr[k][_SEATS], f"seat labels differ at {k}"

    def test_deal_is_identical_with_aivat_on(self, tmp_path):
        """Cross-condition CRN must survive AIVAT being switched on.

        AIVAT evaluates its value function by re-dealing hypothetical boards, so it
        is the one component with an obvious route to disturbing the played deal.
        It does not: it works on ``with_hole_cards`` copies and owns its own streams
        (``poker_ai.search.rng``).  Gated for both arms at once because a per-arm
        passivity test would not catch a disturbance that happened to be identical
        within an arm but differed between them.
        """
        vanilla = _parallel(tmp_path, "R_van_aiv", workers=4,
                            condition="vanilla", aivat=True)
        dbr = _parallel(tmp_path, "R_dbr_aiv", workers=4,
                        condition="DBR(p_max=0.9)", model_spec=_DBR_SPEC,
                        aivat=True)
        for k in sorted(set(vanilla) & set(dbr)):
            assert vanilla[k][_CARDS] == dbr[k][_CARDS], f"deal differs at hand {k}"
            assert vanilla[k][_SEATING] == dbr[k][_SEATING], f"seats differ at {k}"
        # Non-vacuity: the deals must actually vary hand to hand.
        assert len({v[_CARDS] for v in vanilla.values()}) > 1

    def test_the_model_arm_is_actually_modeled(self, tmp_path):
        """Guard against a vacuous comparison.

        If ``model_spec`` failed to reach the hero, the 'DBR' arm would silently be
        a second vanilla run and every assertion above would still pass while
        comparing an arm against itself. Checking ``modeled_decision`` directly is
        far more precise than checking that play differs.
        """
        vanilla, dbr = self._arms(tmp_path)
        assert sum(v[_N_MODELED] for v in vanilla.values()) == 0
        assert sum(v[_N_MODELED] for v in dbr.values()) > 0
