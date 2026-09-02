"""Unit tests for :mod:`poker_ai.blueprint.offline_average`.

Covers the three layers of the offline post-flop average-strategy tool:

* ``sigma_from_regret_chunk`` — the vectorised regret-match kernel, pinned to
  the training-side oracle ``calculate_strategy_from_row``.
* ``average_street`` — the per-row snapshot mean, including the presence-count
  divisor for rows that only some snapshots contain and the integer scaling.
* ``build_final_blueprint`` — the end-to-end assembly, verified by restoring the
  output through the real ``CFRTables`` / warm-start path and reading the
  averaged post-flop rows and the carried-through pre-flop average back out.
"""

import inspect
import os

import joblib
import numpy as np
import pytest

from environment.action_space import MAX_ACTIONS_PER_STREET
from environment.poker_env import INFO_SET_ENCODING
from poker_ai.blueprint.offline_average import (
    SIGMA_SCALE_DEFAULT,
    average_chunk,
    average_street,
    build_final_blueprint,
    sigma_from_regret_chunk,
)
from poker_ai.blueprint.tree_utils import calculate_strategy_from_row
from poker_ai.tables.cfr_tables import CFRTables


# ---------------------------------------------------------------------------
# sigma_from_regret_chunk — golden vs the training-side oracle
# ---------------------------------------------------------------------------
class TestSigmaFromRegretChunk:
    def test_matches_oracle_row_for_row(self):
        rng = np.random.default_rng(0)
        # Mix of all-positive, all-negative, mixed, and all-zero rows.
        chunk = rng.integers(-50, 50, size=(64, 5)).astype(np.int32)
        chunk[0] = 0                      # all-zero → uniform fallback
        chunk[1] = [-3, -1, -7, -2, -9]   # all-negative → uniform fallback
        chunk[2] = [10, 0, 0, 0, 0]       # single positive

        got = sigma_from_regret_chunk(chunk)

        for i in range(chunk.shape[0]):
            oracle = calculate_strategy_from_row(chunk[i])  # maskless
            np.testing.assert_allclose(got[i], oracle, rtol=0, atol=1e-6)

    def test_rows_sum_to_one(self):
        chunk = np.array([[5, 5, 0, 0, 0], [-1, -1, -1, -1, -1]], dtype=np.int32)
        sigma = sigma_from_regret_chunk(chunk)
        np.testing.assert_allclose(sigma.sum(axis=1), [1.0, 1.0], atol=1e-9)
        # All-negative row is uniform over all columns.
        np.testing.assert_allclose(sigma[1], np.full(5, 0.2), atol=1e-9)


# ---------------------------------------------------------------------------
# average_street — presence-count divisor, scaling, empty rows
# ---------------------------------------------------------------------------
class TestAverageStreet:
    def _write_regret(self, d, r, arr):
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"regret_{r}_chunk_000000.npy", arr.astype(np.int32))

    def test_per_row_divisor_and_scale(self, tmp_path):
        r = 2
        # snap_full has 3 rows, snap_short has 2 rows → row 2 is present in one
        # snapshot only, so its divisor is 1 while rows 0-1 divide by 2.
        full = np.array([[10, 0, 0, 0, 0],
                         [0, 10, 0, 0, 0],
                         [3, 1, 0, 0, 0]], dtype=np.int32)
        short = np.array([[0, 10, 0, 0, 0],
                          [10, 0, 0, 0, 0]], dtype=np.int32)
        snap_full = tmp_path / "checkpoint_2000"
        snap_short = tmp_path / "checkpoint_1000"
        self._write_regret(snap_full, r, full)
        self._write_regret(snap_short, r, short)

        out = tmp_path / "out"
        out.mkdir()
        scale = 1_000_000
        # min_confirming_snapshots=1: this test is about the presence-count
        # divisor, not the confirming-snapshots publish gate, and row 2 is
        # deliberately confirmed by only one snapshot.
        # final_dir defines the shape → the 3-row snapshot.
        n = average_street(
            [snap_short, snap_full], snap_full, r, out, scale,
            min_confirming_snapshots=1,
        )
        assert n == 1

        got = np.load(out / f"strategy_{r}_chunk_000000.npy")
        assert got.dtype == np.int32
        assert got.shape == (3, 5)

        s_full = sigma_from_regret_chunk(full)
        s_short = sigma_from_regret_chunk(short)
        # Rows 0,1 averaged over both snapshots; row 2 only in the full one.
        exp0 = np.rint(scale * (s_full[0] + s_short[0]) / 2).astype(np.int32)
        exp1 = np.rint(scale * (s_full[1] + s_short[1]) / 2).astype(np.int32)
        exp2 = np.rint(scale * s_full[2]).astype(np.int32)
        np.testing.assert_array_equal(got[0], exp0)
        np.testing.assert_array_equal(got[1], exp1)
        np.testing.assert_array_equal(got[2], exp2)

    def test_workers_match_serial(self, tmp_path):
        """Parallel averaging must produce byte-identical output to serial.

        --workers only changes how many chunks are in flight (each is
        self-contained), so it must never change the result.
        """
        r = 2
        rng = np.random.default_rng(7)
        snaps = []
        for s in range(3):
            d = tmp_path / f"checkpoint_{1000 * (s + 1)}"
            # Several chunks so there is real work to spread across workers.
            for chunk_id in range(4):
                d.mkdir(parents=True, exist_ok=True)
                arr = rng.integers(-40, 40, size=(50, 5)).astype(np.int32)
                np.save(d / f"regret_{r}_chunk_{chunk_id:06d}.npy", arr)
            snaps.append(d)
        final = snaps[-1]

        serial = tmp_path / "serial"
        serial.mkdir()
        n_serial = average_street(snaps, final, r, serial, 1_000_000, workers=1)

        par = tmp_path / "par"
        par.mkdir()
        n_par = average_street(snaps, final, r, par, 1_000_000, workers=4)

        assert n_serial == n_par == 4
        for chunk_id in range(4):
            name = f"strategy_{r}_chunk_{chunk_id:06d}.npy"
            np.testing.assert_array_equal(
                np.load(serial / name), np.load(par / name)
            )

    def test_resume_skips_complete_chunks(self, tmp_path):
        """With resume, an existing output is skipped and left untouched."""
        r = 3
        snap = tmp_path / "checkpoint_1000"
        self._write_regret(snap, r, np.array([[7, 0, 0, 0, 0]], dtype=np.int32))
        out = tmp_path / "out"
        out.mkdir()

        assert average_chunk([snap], snap, r, 0, out, 1_000_000) is True
        original = np.load(out / f"strategy_{r}_chunk_000000.npy").copy()

        # Poison the existing output: resume must skip it (not recompute),
        # proving the skip is real rather than an accidental rewrite.
        poisoned = np.full((1, 5), 123, dtype=np.int32)
        np.save(out / f"strategy_{r}_chunk_000000.npy", poisoned)
        assert average_chunk([snap], snap, r, 0, out, 1_000_000, resume=True) is False
        np.testing.assert_array_equal(
            np.load(out / f"strategy_{r}_chunk_000000.npy"), poisoned
        )

        # Without resume it is recomputed, restoring the correct values.
        assert average_chunk([snap], snap, r, 0, out, 1_000_000, resume=False) is True
        np.testing.assert_array_equal(
            np.load(out / f"strategy_{r}_chunk_000000.npy"), original
        )

    def test_output_is_written_atomically(self, tmp_path):
        """No temp/partial file survives a completed chunk write.

        Resume trusts "file exists" == "work finished", which is only sound if
        writes are atomic (temp + rename), never in-place truncation.
        """
        r = 3
        snap = tmp_path / "checkpoint_1000"
        self._write_regret(snap, r, np.array([[5, 5, 0, 0, 0]], dtype=np.int32))
        out = tmp_path / "out"
        out.mkdir()
        average_chunk([snap], snap, r, 0, out, 1_000_000)
        names = sorted(p.name for p in out.iterdir())
        assert names == [f"strategy_{r}_chunk_000000.npy"], names

    def test_row_absent_from_all_snapshots_stays_zero(self, tmp_path):
        r = 3
        # final_dir has 2 rows but the only averaged snapshot has 1 row, so the
        # second row is present in no averaged snapshot → all-zero output.
        final = np.array([[5, 0, 0, 0, 0], [0, 5, 0, 0, 0]], dtype=np.int32)
        snap = np.array([[5, 0, 0, 0, 0]], dtype=np.int32)
        final_dir = tmp_path / "checkpoint_9"
        snap_dir = tmp_path / "checkpoint_5"
        self._write_regret(final_dir, r, final)
        self._write_regret(snap_dir, r, snap)

        out = tmp_path / "out"
        out.mkdir()
        # min_confirming_snapshots=1: this test is about absence-from-file vs
        # presence, not the confirming-snapshots publish gate, and there is
        # only one snapshot to confirm anything at all.
        average_street(
            [snap_dir], final_dir, r, out, 1_000_000, min_confirming_snapshots=1
        )
        got = np.load(out / f"strategy_{r}_chunk_000000.npy")
        assert got.shape == (2, 5)
        np.testing.assert_array_equal(got[1], np.zeros(5, dtype=np.int32))
        assert got[0].sum() > 0

    def test_no_positive_regret_row_stays_zero_even_if_present(self, tmp_path):
        """A row present in every snapshot but never showing positive regret
        in any of them must stay all-zero (defer to the live regret-match
        fallback), not get the maskless-uniform placeholder baked in."""
        r = 1
        early = np.array([[-3, -1, -7, -2, -9]], dtype=np.int32)   # all-negative
        late = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)         # all-zero
        early_dir = tmp_path / "checkpoint_1000"
        late_dir = tmp_path / "checkpoint_2000"
        self._write_regret(early_dir, r, early)
        self._write_regret(late_dir, r, late)

        out = tmp_path / "out"
        out.mkdir()
        average_street([early_dir, late_dir], late_dir, r, out, 1_000_000)
        got = np.load(out / f"strategy_{r}_chunk_000000.npy")
        np.testing.assert_array_equal(got[0], np.zeros(5, dtype=np.int32))

    def test_placeholder_snapshots_excluded_not_just_zeroed(self, tmp_path):
        """A row with real signal in one snapshot and no signal in another
        must average over ONLY the real-signal snapshot — not include the
        no-signal snapshot's placeholder (whether uniform or zero) in the
        divisor, which would needlessly dilute a row that did converge."""
        r = 1
        no_signal = np.array([[-3, -1, -7, -2, -9]], dtype=np.int32)  # all-negative
        real = np.array([[10, 0, 0, 0, 0]], dtype=np.int32)           # pure fold
        d_no_signal = tmp_path / "checkpoint_1000"
        d_real = tmp_path / "checkpoint_2000"
        self._write_regret(d_no_signal, r, no_signal)
        self._write_regret(d_real, r, real)

        out = tmp_path / "out"
        out.mkdir()
        # min_confirming_snapshots=1: this test is about dilution by a
        # no-signal snapshot, not the confirming-snapshots publish gate, and
        # the row is deliberately confirmed by only one real-signal snapshot.
        average_street(
            [d_no_signal, d_real], d_real, r, out, 1_000_000,
            min_confirming_snapshots=1,
        )
        got = np.load(out / f"strategy_{r}_chunk_000000.npy")

        # Averaged over the ONE real-signal snapshot alone: exactly its
        # regret-matched row (pure fold), not diluted by the no-signal one.
        expected = np.rint(1_000_000 * sigma_from_regret_chunk(real)[0]).astype(np.int32)
        np.testing.assert_array_equal(got[0], expected)
        assert got[0][0] == 1_000_000  # pure fold, not watered down toward uniform

    def test_default_defers_row_confirmed_by_only_one_snapshot(self, tmp_path):
        """The MIN_CONFIRMING_SNAPSHOTS_DEFAULT (2) gate: a row with real
        signal in exactly one snapshot is NOT published by default — it's a
        single categorical sample, same standard BlueprintPolicy already
        applies to a once-visited pre-flop row. Deferred to zero (live
        regret-match fallback) rather than a falsely-confident average."""
        r = 1
        only_signal = np.array([[10, 0, 0, 0, 0]], dtype=np.int32)  # pure fold
        snap = tmp_path / "checkpoint_1000"
        self._write_regret(snap, r, only_signal)

        out = tmp_path / "out"
        out.mkdir()
        average_street([snap], snap, r, out, 1_000_000)  # default gate
        got = np.load(out / f"strategy_{r}_chunk_000000.npy")
        np.testing.assert_array_equal(got[0], np.zeros(5, dtype=np.int32))

    def test_default_publishes_row_confirmed_by_two_snapshots(self, tmp_path):
        """The same row, but independently confirmed by a second snapshot,
        clears the default gate and is published with full mass."""
        r = 1
        signal_a = np.array([[10, 0, 0, 0, 0]], dtype=np.int32)  # pure fold
        signal_b = np.array([[8, 0, 0, 0, 0]], dtype=np.int32)   # also fold-leaning
        snap_a = tmp_path / "checkpoint_1000"
        snap_b = tmp_path / "checkpoint_2000"
        self._write_regret(snap_a, r, signal_a)
        self._write_regret(snap_b, r, signal_b)

        out = tmp_path / "out"
        out.mkdir()
        average_street([snap_a, snap_b], snap_b, r, out, 1_000_000)  # default gate
        got = np.load(out / f"strategy_{r}_chunk_000000.npy")
        expected = np.rint(
            1_000_000 * (sigma_from_regret_chunk(signal_a)[0]
                         + sigma_from_regret_chunk(signal_b)[0]) / 2
        ).astype(np.int32)
        np.testing.assert_array_equal(got[0], expected)
        assert got[0].sum() >= 10  # clears min_strategy_mass at read time

    def test_confirming_fraction_dominates_absolute_floor_at_scale(self, tmp_path):
        """On a run with many retained snapshots, 'confirmed by 2' is a very
        low bar (the reported real-world symptom: ~99% visited even after the
        MIN_CONFIRMING_SNAPSHOTS=2 fix, on an 81-snapshot run) — the default
        min_confirming_fraction=0.5 must dominate and reject a row confirmed
        by only 2 of, say, 10 snapshots."""
        r = 1
        n_snaps = 10
        confirmed_by = 2  # well past the absolute floor of 2... exactly at it
        snap_dirs = []
        for s in range(n_snaps):
            d = tmp_path / f"checkpoint_{1000 * (s + 1)}"
            regret = np.array(
                [[10, 0, 0, 0, 0]] if s < confirmed_by else [[0, 0, 0, 0, 0]],
                dtype=np.int32,
            )
            self._write_regret(d, r, regret)
            snap_dirs.append(d)

        out = tmp_path / "out"
        out.mkdir()
        average_street(snap_dirs, snap_dirs[-1], r, out, 1_000_000)  # default gates
        got = np.load(out / f"strategy_{r}_chunk_000000.npy")
        # 2/10 confirmations clears the absolute floor (2) but not the
        # fraction floor (ceil(0.5*10)=5) -> still deferred to zero.
        np.testing.assert_array_equal(got[0], np.zeros(5, dtype=np.int32))

    def test_confirming_fraction_publishes_when_majority_confirms(self, tmp_path):
        r = 1
        n_snaps = 10
        confirmed_by = 6  # majority of 10 -> clears ceil(0.5*10)=5
        snap_dirs = []
        for s in range(n_snaps):
            d = tmp_path / f"checkpoint_{1000 * (s + 1)}"
            regret = np.array(
                [[10, 0, 0, 0, 0]] if s < confirmed_by else [[0, 0, 0, 0, 0]],
                dtype=np.int32,
            )
            self._write_regret(d, r, regret)
            snap_dirs.append(d)

        out = tmp_path / "out"
        out.mkdir()
        average_street(snap_dirs, snap_dirs[-1], r, out, 1_000_000)  # default gates
        got = np.load(out / f"strategy_{r}_chunk_000000.npy")
        assert got[0].sum() >= 10  # published, clears min_strategy_mass
        assert got[0][0] == 1_000_000  # pure fold, averaged only over confirmers

    def test_min_snapshot_regret_magnitude_excludes_weak_snapshots(self, tmp_path):
        """A snapshot with a tiny positive-regret blip (the single-touch noise
        case) must not count toward confirmation when a magnitude floor is
        set, even though it would under the plain 'any positive' test."""
        r = 1
        weak = np.array([[1, 0, 0, 0, 0]], dtype=np.int32)     # barely positive
        strong = np.array([[500, 0, 0, 0, 0]], dtype=np.int32)  # clearly positive
        d_weak = tmp_path / "checkpoint_1000"
        d_strong = tmp_path / "checkpoint_2000"
        self._write_regret(d_weak, r, weak)
        self._write_regret(d_strong, r, strong)

        out = tmp_path / "out"
        out.mkdir()
        # Without a magnitude floor: both count (any positive) -> 2/2 confirms,
        # clears both the floor (2) and the fraction (ceil(0.5*2)=1).
        average_street(
            [d_weak, d_strong], d_strong, r, out, 1_000_000,
        )
        got_without = np.load(out / f"strategy_{r}_chunk_000000.npy")
        assert got_without[0].sum() > 0

        # With a magnitude floor above `weak`'s regret sum (1) but below
        # `strong`'s (500): only `strong` counts -> 1/2 confirms, below the
        # (still-active) floor of 2 -> deferred to zero.
        out2 = tmp_path / "out2"
        out2.mkdir()
        average_street(
            [d_weak, d_strong], d_strong, r, out2, 1_000_000,
            min_confirming_snapshots=1, min_confirming_fraction=0.0,
            min_snapshot_regret_magnitude=100,
        )
        got_with = np.load(out2 / f"strategy_{r}_chunk_000000.npy")
        # Only `strong` confirms (1 snapshot) -> clears min_confirming_snapshots=1
        # -> published as exactly `strong`'s own regret-matched row, not an
        # average that included the weak snapshot.
        expected = np.rint(1_000_000 * sigma_from_regret_chunk(strong)[0]).astype(np.int32)
        np.testing.assert_array_equal(got_with[0], expected)


# ---------------------------------------------------------------------------
# build_final_blueprint — end-to-end, restore through the real tables
# ---------------------------------------------------------------------------
def _new_tables(index_path, shm_name):
    shm_dir = str(index_path.parent / shm_name)
    os.makedirs(shm_dir, exist_ok=True)
    return CFRTables(
        index_path=index_path,
        shm_dir=shm_dir,
        actions_per_street=MAX_ACTIONS_PER_STREET,
    )


def _save_all_streets(tables, cp_dir):
    cp_dir.mkdir(parents=True, exist_ok=True)
    for r in range(4):
        n = tables._indexes[r].n_allocated_rows
        tables.regret[r].store.save_all(cp_dir, n, f"regret_{r}")
        tables.strategy[r].store.save_all(cp_dir, n, f"strategy_{r}")


def _write_state(cp_dir, tables, t):
    state = {
        "t": t,
        "n_players": 2,
        "chunk_size": 4_000_000,
        # The live tag, not a placeholder: the restore path validates it (a
        # blueprint written under a different action abstraction has both
        # differently-coded keys and differently-shaped rows).
        "info_set_encoding": INFO_SET_ENCODING,
        "sync_interval": 1000,
        "checkpoint_start_cycles": 0,
        "n_chunks_per_street": tables.n_chunks_per_street(),
    }
    joblib.dump(state, cp_dir / "server_state.pkl")


class TestBuildFinalBlueprint:
    def test_average_restores_and_reads_back(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        index_path = train_dir / "lmdb_index"

        # Turn (street 2) rows, and a pre-flop (street 0) average-strategy row.
        turn_keys = ["t0", "t1", "t2"]
        preflop_key = "pf0"
        # Rows are sized from the live action abstraction, never a literal width.
        preflop_phi = np.zeros(MAX_ACTIONS_PER_STREET[0], dtype=np.int32)
        preflop_phi[:2] = (7, 3)

        tables = _new_tables(index_path, "shm_w")
        try:
            for k in turn_keys:
                tables.regret[2].update_row(k, 0, 1)  # allocate the rows in order
            for col, v in enumerate(preflop_phi):
                if v:
                    tables.strategy[0].update_row(preflop_key, col, int(v))
            tables.regret[0].update_row(preflop_key, 0, 1)  # keep a regret fallback row

            # Two snapshots sharing the one index.
            cp1 = train_dir / "checkpoint_1000"
            cp2 = train_dir / "checkpoint_2000"
            _save_all_streets(tables, cp1)
            _write_state(cp1, tables, t=1000)
            _save_all_streets(tables, cp2)
            _write_state(cp2, tables, t=2000)
        finally:
            tables.close()

        # Craft distinct turn regrets per snapshot (rows in allocation order).
        w2 = MAX_ACTIONS_PER_STREET[2]
        r1 = np.zeros((3, w2), dtype=np.int32)
        r1[0, :2] = (9, 1)
        r1[1, 1:3] = (4, 4)
        r1[2, :4] = 1
        r2 = np.zeros((3, w2), dtype=np.int32)
        r2[0, :2] = (1, 9)
        r2[1, 0] = 8
        r2[2, 4] = 5
        np.save(cp1 / "regret_2_chunk_000000.npy", r1)
        np.save(cp2 / "regret_2_chunk_000000.npy", r2)

        out_dir = tmp_path / "final_bp"
        # No snapshot_weighting → the DEFAULT, which is "linear" (t-weighted).
        build_final_blueprint(train_dir, out_dir, scale=SIGMA_SCALE_DEFAULT)

        # Output is a self-contained blueprint dir.
        assert (out_dir / "lmdb_index").is_dir()
        cps = list(out_dir.glob("checkpoint_[0-9]*"))
        assert len(cps) == 1 and cps[0].name == "checkpoint_2000"

        # Restore through the real warm-start path and read rows back.
        from poker_ai.tables.warm_start import apply_warm_start_to_tables

        restored = _new_tables(out_dir / "lmdb_index", "shm_r")
        try:
            apply_warm_start_to_tables(restored, out_dir, expected_n_players=2)

            # Pre-flop φ carried through verbatim.
            got_phi = restored.strategy[0].get_row_if_exists(preflop_key)
            np.testing.assert_array_equal(got_phi, preflop_phi)

            # Post-flop strategy = averaged regret-matched σ, scaled.  Under the
            # default "linear" weighting the snapshots at t=1000 and t=2000 carry
            # weights 0.5 and 1.0 (normalised by the largest t), so the mean is
            # (0.5·s1 + 1.0·s2) / 1.5 — the later snapshot counts double.
            s1 = sigma_from_regret_chunk(r1)
            s2 = sigma_from_regret_chunk(r2)
            expected = np.rint(
                SIGMA_SCALE_DEFAULT * (0.5 * s1 + 1.0 * s2) / 1.5
            ).astype(np.int32)
            for row, key in enumerate(turn_keys):
                got = restored.strategy[2].get_row_if_exists(key)
                np.testing.assert_array_equal(got, expected[row])
                assert int(got.sum()) >= 10  # clears min_strategy_mass
        finally:
            restored.close()

        # …and the "equal" path still reproduces the plain unweighted mean, which
        # is what regenerating a pre-2026-09-02 blueprint depends on.
        eq_dir = tmp_path / "final_bp_equal"
        build_final_blueprint(train_dir, eq_dir, scale=SIGMA_SCALE_DEFAULT,
                              snapshot_weighting="equal")
        restored_eq = _new_tables(eq_dir / "lmdb_index", "shm_req")
        try:
            apply_warm_start_to_tables(restored_eq, eq_dir, expected_n_players=2)
            expected_eq = np.rint(
                SIGMA_SCALE_DEFAULT * (s1 + s2) / 2
            ).astype(np.int32)
            for row, key in enumerate(turn_keys):
                got = restored_eq.strategy[2].get_row_if_exists(key)
                np.testing.assert_array_equal(got, expected_eq[row])
        finally:
            restored_eq.close()

    def test_refuses_inconsistent_snapshots(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        (train_dir / "lmdb_index").mkdir()
        for t, enc in ((1000, "enc-a"), (2000, "enc-b")):
            cp = train_dir / f"checkpoint_{t}"
            cp.mkdir()
            np.save(cp / "regret_0_chunk_000000.npy", np.zeros((1, 6), np.int32))
            joblib.dump(
                {"t": t, "n_players": 2, "chunk_size": 4_000_000,
                 "info_set_encoding": enc, "sync_interval": 1000,
                 "checkpoint_start_cycles": 0, "n_chunks_per_street": {0: 1}},
                cp / "server_state.pkl",
            )
        with pytest.raises(ValueError, match="info_set_encoding"):
            build_final_blueprint(train_dir, tmp_path / "out")

    def test_resume_continues_interrupted_build(self, tmp_path):
        """A second run with resume=True completes a partial output dir.

        Simulates a killed job: the first build is interrupted after the index
        copy, leaving a partial output that the default guard would reject.
        """
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        tables = _new_tables(train_dir / "lmdb_index", "shm_w3")
        try:
            tables.regret[2].update_row("k0", 0, 1)
            cp = train_dir / "checkpoint_1000"
            _save_all_streets(tables, cp)
            _write_state(cp, tables, t=1000)
        finally:
            tables.close()

        out_dir = tmp_path / "out"
        # First (complete) build, so we know the expected result.
        build_final_blueprint(train_dir, out_dir)
        expected = np.load(out_dir / "checkpoint_1000" / "strategy_2_chunk_000000.npy")

        # Simulate an interruption: drop the averaged post-flop chunks but keep
        # the copied index (exactly the state a killed job leaves).
        for r in (1, 2, 3):
            p = out_dir / "checkpoint_1000" / f"strategy_{r}_chunk_000000.npy"
            if p.exists():
                p.unlink()

        # Without resume the existing lmdb_index is refused...
        with pytest.raises(FileExistsError):
            build_final_blueprint(train_dir, out_dir)
        # ...with resume the build completes in place.
        build_final_blueprint(train_dir, out_dir, resume=True)
        np.testing.assert_array_equal(
            np.load(out_dir / "checkpoint_1000" / "strategy_2_chunk_000000.npy"),
            expected,
        )

    def test_min_t_excludes_early_snapshots(self, tmp_path):
        # Two snapshots; min_t drops the earlier from the average but the later
        # (also the final) still defines the output. Verify the average equals
        # the single surviving snapshot's regret-matched strategy.
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        index_path = train_dir / "lmdb_index"
        tables = _new_tables(index_path, "shm_w2")
        try:
            tables.regret[2].update_row("k0", 0, 1)
            for t in (1000, 5000):
                cp = train_dir / f"checkpoint_{t}"
                _save_all_streets(tables, cp)
                _write_state(cp, tables, t=t)
        finally:
            tables.close()

        early = np.array([[10, 0, 0, 0, 0]], dtype=np.int32)
        late = np.array([[0, 0, 10, 0, 0]], dtype=np.int32)
        np.save(train_dir / "checkpoint_1000" / "regret_2_chunk_000000.npy", early)
        np.save(train_dir / "checkpoint_5000" / "regret_2_chunk_000000.npy", late)

        out_dir = tmp_path / "out"
        # min_confirming_snapshots=1: this test is about min_t filtering, not
        # the confirming-snapshots publish gate, and only one snapshot
        # survives the min_t=2000 cutoff by construction.
        build_final_blueprint(
            train_dir, out_dir, min_t=2000, min_confirming_snapshots=1
        )

        got = np.load(out_dir / "checkpoint_5000" / "strategy_2_chunk_000000.npy")
        expected = np.rint(
            SIGMA_SCALE_DEFAULT * sigma_from_regret_chunk(late)
        ).astype(np.int32)
        np.testing.assert_array_equal(got, expected)


class TestSnapshotWeighting:
    """``snapshot_weights`` makes the per-row mean weighted (offline_average)."""

    @staticmethod
    def _snap(tmp_path, name, regret_rows):
        d = tmp_path / name
        (d).mkdir(parents=True, exist_ok=True)
        np.save(d / "regret_1_chunk_000000.npy", np.asarray(regret_rows, dtype=np.int32))
        return d

    def test_default_is_unweighted_and_linear_favours_late_snapshots(self, tmp_path):
        from poker_ai.blueprint.offline_average import average_chunk

        # Two snapshots, one row, two actions.  Early prefers action 0, late
        # prefers action 1 — both with positive regret, so both confirm the row.
        early = self._snap(tmp_path, "checkpoint_1", [[100, 0]])
        late = self._snap(tmp_path, "checkpoint_2", [[0, 100]])
        snaps = [early, late]

        out_eq = tmp_path / "eq"
        out_eq.mkdir()
        average_chunk(snaps, late, 1, 0, out_eq, 1_000_000,
                      min_confirming_snapshots=2, min_confirming_fraction=0.0,
                      min_snapshot_regret_magnitude=None)
        eq = np.load(out_eq / "strategy_1_chunk_000000.npy")[0]
        # Unweighted mean of (1,0) and (0,1) is (0.5, 0.5).
        assert eq[0] == eq[1]

        out_lin = tmp_path / "lin"
        out_lin.mkdir()
        average_chunk(snaps, late, 1, 0, out_lin, 1_000_000,
                      min_confirming_snapshots=2, min_confirming_fraction=0.0,
                      min_snapshot_regret_magnitude=None,
                      snapshot_weights=[1.0, 3.0])
        lin = np.load(out_lin / "strategy_1_chunk_000000.npy")[0]
        # Weighted 1:3 → (0.25, 0.75); the late snapshot dominates.
        assert lin[1] > lin[0]
        assert lin[0] == pytest.approx(250_000, abs=2)
        assert lin[1] == pytest.approx(750_000, abs=2)

    def test_weights_do_not_relax_the_confirmation_gate(self, tmp_path):
        """A heavy snapshot must not publish a row on its own — the tally is a
        plain count of confirming snapshots, not a weighted sum."""
        from poker_ai.blueprint.offline_average import average_chunk

        confirming = self._snap(tmp_path, "checkpoint_1", [[100, 0]])
        silent = self._snap(tmp_path, "checkpoint_2", [[0, 0]])  # no positive regret
        out = tmp_path / "out"
        out.mkdir()
        average_chunk([confirming, silent], silent, 1, 0, out, 1_000_000,
                      min_confirming_snapshots=2, min_confirming_fraction=0.0,
                      min_snapshot_regret_magnitude=None,
                      snapshot_weights=[1000.0, 1.0])
        row = np.load(out / "strategy_1_chunk_000000.npy")[0]
        assert row.sum() == 0  # only 1 of 2 confirmed → unpublished

    def test_mismatched_weight_length_is_rejected(self, tmp_path):
        from poker_ai.blueprint.offline_average import average_chunk

        d = self._snap(tmp_path, "checkpoint_1", [[100, 0]])
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(ValueError, match="align positionally"):
            average_chunk([d], d, 1, 0, out, 1_000_000, snapshot_weights=[1.0, 2.0])

    def test_default_weighting_is_linear_but_average_chunk_stays_unweighted(self):
        """Two different layers, deliberately not the same default.

        ``SNAPSHOT_WEIGHTING_DEFAULT`` is what a *build* uses (flipped to
        ``linear`` on measured evidence).  ``average_chunk``'s own
        ``snapshot_weights=None`` stays the plain unweighted mean, so the
        low-level primitive is unchanged and old blueprints stay reproducible
        via ``--snapshot-weighting equal``.
        """
        from poker_ai.blueprint import offline_average as oa

        assert oa.SNAPSHOT_WEIGHTING_DEFAULT == "linear"
        assert set(oa.SNAPSHOT_WEIGHTINGS) == {"equal", "linear"}
        sig = inspect.signature(oa.average_chunk)
        assert sig.parameters["snapshot_weights"].default is None
