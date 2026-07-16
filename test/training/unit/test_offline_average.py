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

import os

import joblib
import numpy as np
import pytest

from environment.action_space import MAX_ACTIONS_PER_STREET
from poker_ai.blueprint.offline_average import (
    SIGMA_SCALE_DEFAULT,
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
        # final_dir defines the shape → the 3-row snapshot.
        n = average_street([snap_short, snap_full], snap_full, r, out, scale)
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
        average_street([snap_dir], final_dir, r, out, 1_000_000)
        got = np.load(out / f"strategy_{r}_chunk_000000.npy")
        assert got.shape == (2, 5)
        np.testing.assert_array_equal(got[1], np.zeros(5, dtype=np.int32))
        assert got[0].sum() > 0


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
        "info_set_encoding": "test-enc",
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
        preflop_phi = np.array([7, 3, 0, 0, 0, 0], dtype=np.int32)  # street 0 width 6

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
        r1 = np.array([[9, 1, 0, 0, 0],
                       [0, 4, 4, 0, 0],
                       [1, 1, 1, 1, 0]], dtype=np.int32)
        r2 = np.array([[1, 9, 0, 0, 0],
                       [8, 0, 0, 0, 0],
                       [0, 0, 0, 0, 5]], dtype=np.int32)
        np.save(cp1 / "regret_2_chunk_000000.npy", r1)
        np.save(cp2 / "regret_2_chunk_000000.npy", r2)

        out_dir = tmp_path / "final_bp"
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

            # Post-flop strategy = averaged regret-matched σ, scaled.
            s1 = sigma_from_regret_chunk(r1)
            s2 = sigma_from_regret_chunk(r2)
            expected = np.rint(SIGMA_SCALE_DEFAULT * (s1 + s2) / 2).astype(np.int32)
            for row, key in enumerate(turn_keys):
                got = restored.strategy[2].get_row_if_exists(key)
                np.testing.assert_array_equal(got, expected[row])
                assert int(got.sum()) >= 10  # clears min_strategy_mass
        finally:
            restored.close()

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
        build_final_blueprint(train_dir, out_dir, min_t=2000)

        got = np.load(out_dir / "checkpoint_5000" / "strategy_2_chunk_000000.npy")
        expected = np.rint(
            SIGMA_SCALE_DEFAULT * sigma_from_regret_chunk(late)
        ).astype(np.int32)
        np.testing.assert_array_equal(got, expected)
