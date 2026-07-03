"""Tests for the blueprint metrics report (evaluation/blueprint_metrics.py).

Two styles:

- **Hand-built checkpoint** — a tiny ``checkpoint_*`` directory written with
  joblib + numpy exactly as :meth:`CFRTables.save_chunks` lays it out (trimmed
  ``strategy_{r}_chunk_*.npy`` / ``regret_{r}_chunk_*.npy`` + ``server_state.pkl``),
  with counts chosen so every aggregate is asserted against arithmetic we control.
- **Pure-function** unit tests for the histogram-percentile helper and the
  label fallback.
"""

import json

import joblib
import numpy as np
import pytest

from evaluation.blueprint_metrics import (
    PRUNE_THRESHOLD,
    _action_labels,
    _hist_percentile,
    analyze,
    build_report,
    compute_regret_metrics,
    compute_street_metrics,
    main,
)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

# 4 action columns: fold, call, all_in, one raise.  Chosen narrower than the
# live preflop action space so the generic-label fallback path is exercised.
WIDTH = 4

# Rows with hand-checkable aggregates.  Row 2 is unvisited (all-zero) and must
# be excluded from visited counts, mean strategy, entropy, and purity stats.
STRAT_ROWS = np.array(
    [
        [2, 2, 0, 0],      # 50/50 fold-call mix, mass 4
        [0, 0, 0, 4],      # pure raise, mass 4
        [0, 0, 0, 0],      # unvisited
        [0, 8, 0, 0],      # pure call, mass 8
    ],
    dtype=np.int32,
)
# Column mass = [2, 10, 0, 8], total 16 → play freq [1/8, 5/8, 0, 1/2 - ...]
EXPECTED_PLAY_FREQ = np.array([2, 10, 0, 4]) / 16.0
# Per-row distributions: [.5,.5,0,0], [0,0,0,1], [0,1,0,0] → mean over 3 rows.
EXPECTED_MEAN_STRAT = np.array([0.5 / 3, 1.5 / 3, 0.0, 1.0 / 3])

REGRET_ROWS = np.array(
    [
        [100, -50, 0, 0],
        [-310_000_000, PRUNE_THRESHOLD - 1, 0, 300],
    ],
    dtype=np.int32,
)


def _write_checkpoint(tmp_path, *, streets=(0,), n_chunks=None, t=42, n_players=2):
    """Write a minimal blueprint dir with one checkpoint; return its root."""
    bp = tmp_path / "blueprint"
    cp = bp / f"checkpoint_{t:012d}"
    cp.mkdir(parents=True)
    ncs = {r: 0 for r in range(4)}
    for r in streets:
        np.save(cp / f"strategy_{r}_chunk_000000.npy", STRAT_ROWS)
        np.save(cp / f"regret_{r}_chunk_000000.npy", REGRET_ROWS)
        ncs[r] = 1
    if n_chunks is not None:
        ncs = n_chunks
    joblib.dump(
        {"t": t, "n_players": n_players, "n_chunks_per_street": ncs},
        cp / "server_state.pkl",
    )
    return bp


# --------------------------------------------------------------------------- #
# Street metrics
# --------------------------------------------------------------------------- #


class TestStreetMetrics:
    def test_play_freq_is_visit_weighted_column_mass(self, tmp_path):
        bp = _write_checkpoint(tmp_path)
        files = sorted((bp / "checkpoint_000000000042").glob("strategy_0_*.npy"))
        m = compute_street_metrics(files, street=0)
        got = np.array(list(m["play_freq"].values()))
        np.testing.assert_allclose(got, EXPECTED_PLAY_FREQ)

    def test_mean_strategy_weights_infosets_equally(self, tmp_path):
        bp = _write_checkpoint(tmp_path)
        files = sorted((bp / "checkpoint_000000000042").glob("strategy_0_*.npy"))
        m = compute_street_metrics(files, street=0)
        got = np.array(list(m["mean_strategy"].values()))
        np.testing.assert_allclose(got, EXPECTED_MEAN_STRAT)

    def test_visited_and_row_counts(self, tmp_path):
        bp = _write_checkpoint(tmp_path)
        files = sorted((bp / "checkpoint_000000000042").glob("strategy_0_*.npy"))
        m = compute_street_metrics(files, street=0)
        assert m["n_infosets_total"] == 4
        assert m["n_infosets_scanned"] == 4
        assert m["sampled"] is False
        assert m["n_chunks_total"] == m["n_chunks_read"] == 1
        assert m["n_visited"] == 3
        assert m["visited_frac"] == pytest.approx(0.75)
        assert m["total_visit_mass"] == 16.0

    def test_buckets_fold_call_allin_raise(self, tmp_path):
        bp = _write_checkpoint(tmp_path)
        files = sorted((bp / "checkpoint_000000000042").glob("strategy_0_*.npy"))
        b = compute_street_metrics(files, street=0)["play_freq_buckets"]
        assert b["fold"] == pytest.approx(2 / 16)
        assert b["call"] == pytest.approx(10 / 16)
        assert b["all_in"] == pytest.approx(0.0)
        assert b["raise"] == pytest.approx(4 / 16)

    def test_determinism_stats(self, tmp_path):
        bp = _write_checkpoint(tmp_path)
        files = sorted((bp / "checkpoint_000000000042").glob("strategy_0_*.npy"))
        d = compute_street_metrics(files, street=0)["determinism"]
        # Two of three visited rows are pure (max prob 1.0), one is a 50/50 mix.
        assert d["frac_maxp_gt_99"] == pytest.approx(2 / 3)
        assert d["frac_maxp_gt_50"] == pytest.approx(2 / 3)   # 0.5 is not > 0.5
        # Mixed row: H = ln 2 / ln 4 = 0.5 normalised; pure rows: 0.
        assert d["entropy_p90"] == pytest.approx(0.5, abs=0.02)
        assert d["entropy_p10"] == pytest.approx(0.0, abs=0.02)

    def test_no_files_returns_none(self):
        assert compute_street_metrics([], street=2) is None

    def test_batching_matches_single_pass(self, tmp_path):
        bp = _write_checkpoint(tmp_path)
        files = sorted((bp / "checkpoint_000000000042").glob("strategy_0_*.npy"))
        whole = compute_street_metrics(files, street=0)
        batched = compute_street_metrics(files, street=0, batch_rows=1)
        assert whole["play_freq"] == batched["play_freq"]
        assert whole["determinism"] == batched["determinism"]

    def test_multiple_chunk_files_accumulate(self, tmp_path):
        cp = tmp_path / "cp"
        cp.mkdir()
        np.save(cp / "strategy_0_chunk_000000.npy", STRAT_ROWS)
        np.save(cp / "strategy_0_chunk_000001.npy", STRAT_ROWS)
        m = compute_street_metrics(sorted(cp.glob("strategy_0_*.npy")), street=0)
        assert m["n_infosets_total"] == 8
        assert m["n_infosets_scanned"] == 8
        assert m["total_visit_mass"] == 32.0
        got = np.array(list(m["play_freq"].values()))
        np.testing.assert_allclose(got, EXPECTED_PLAY_FREQ)   # doubling cancels

    def test_inconsistent_width_raises(self, tmp_path):
        cp = tmp_path / "cp"
        cp.mkdir()
        np.save(cp / "strategy_0_chunk_000000.npy", STRAT_ROWS)
        np.save(cp / "strategy_0_chunk_000001.npy", np.zeros((2, 5), dtype=np.int32))
        with pytest.raises(ValueError, match="Inconsistent action width"):
            compute_street_metrics(sorted(cp.glob("strategy_0_*.npy")), street=0)

    def test_sampling_reads_subset_but_exact_total(self, tmp_path):
        cp = tmp_path / "cp"
        cp.mkdir()
        # 5 full chunks of 4 rows each + a partial last chunk of 2 rows = 22 rows.
        for i in range(5):
            np.save(cp / f"strategy_0_chunk_{i:06d}.npy", STRAT_ROWS)
        np.save(cp / "strategy_0_chunk_000005.npy", STRAT_ROWS[:2])
        files = sorted(cp.glob("strategy_0_*.npy"))
        m = compute_street_metrics(files, street=0, sample_chunks=3)
        assert m["sampled"] is True
        assert m["n_chunks_total"] == 6
        assert m["n_chunks_read"] == 3
        # Exact total from chunk-0 (full=4) and last (partial=2): 5*4 + 2 = 22.
        assert m["n_infosets_total"] == 22
        # Scanned only the 3 selected chunks (endpoints always included).
        assert m["n_infosets_scanned"] < 22

    def test_sample_chunks_none_reads_all(self, tmp_path):
        cp = tmp_path / "cp"
        cp.mkdir()
        for i in range(3):
            np.save(cp / f"strategy_0_chunk_{i:06d}.npy", STRAT_ROWS)
        files = sorted(cp.glob("strategy_0_*.npy"))
        m = compute_street_metrics(files, street=0, sample_chunks=None)
        assert m["sampled"] is False
        assert m["n_chunks_read"] == m["n_chunks_total"] == 3
        assert m["n_infosets_scanned"] == m["n_infosets_total"] == 12

    def test_canonical_labels_used_when_width_matches(self, tmp_path):
        from environment.action_space import CANONICAL_ACTIONS

        cp = tmp_path / "cp"
        cp.mkdir()
        width = len(CANONICAL_ACTIONS[3])
        rows = np.zeros((2, width), dtype=np.int32)
        rows[0, 0] = 1                       # one fold visit
        np.save(cp / "strategy_3_chunk_000000.npy", rows)
        m = compute_street_metrics(sorted(cp.glob("*.npy")), street=3)
        assert m["action_labels_canonical"] is True
        assert list(m["play_freq"]) == CANONICAL_ACTIONS[3]


# --------------------------------------------------------------------------- #
# Regret metrics
# --------------------------------------------------------------------------- #


class TestRegretMetrics:
    def test_fractions(self, tmp_path):
        cp = tmp_path / "cp"
        cp.mkdir()
        np.save(cp / "regret_0_chunk_000000.npy", REGRET_ROWS)
        m = compute_regret_metrics(sorted(cp.glob("*.npy")))
        assert m["n_entries"] == 8
        assert m["frac_positive"] == pytest.approx(2 / 8)     # 100 and 300
        assert m["mean_positive_regret"] == pytest.approx(200.0)
        assert m["frac_at_floor"] == pytest.approx(1 / 8)     # -310M entry
        # -310M, PRUNE_THRESHOLD-1 are both < the prune threshold.
        assert m["frac_prunable"] == pytest.approx(2 / 8)

    def test_no_files_returns_none(self):
        assert compute_regret_metrics([]) is None


# --------------------------------------------------------------------------- #
# Report + entry point
# --------------------------------------------------------------------------- #


class TestReport:
    def test_build_report_meta_and_streets(self, tmp_path):
        bp = _write_checkpoint(tmp_path, streets=(0, 1))
        report = build_report(bp, include_regret=True)
        assert report["meta"]["t"] == 42
        assert report["meta"]["n_players"] == 2
        assert report["meta"]["checkpoint"] == "checkpoint_000000000042"
        assert report["meta"]["sampled"] is False
        assert set(report["streets"]) == {"preflop", "flop"}
        assert report["meta"]["n_infosets_total"] == 8
        assert set(report["regret"]) == {"preflop", "flop"}
        assert report["warnings"] == []

    def test_regret_off_by_default(self, tmp_path):
        bp = _write_checkpoint(tmp_path)
        assert build_report(bp)["regret"] == {}

    def test_latest_checkpoint_wins(self, tmp_path):
        bp = _write_checkpoint(tmp_path, t=42)
        old = bp / "checkpoint_000000000007"
        old.mkdir()
        joblib.dump(
            {"t": 7, "n_players": 2, "n_chunks_per_street": {r: 0 for r in range(4)}},
            old / "server_state.pkl",
        )
        assert build_report(bp)["meta"]["t"] == 42

    def test_explicit_checkpoint_and_bare_checkpoint_dir(self, tmp_path):
        bp = _write_checkpoint(tmp_path, t=42)
        by_name = build_report(bp, checkpoint="checkpoint_000000000042")
        direct = build_report(bp / "checkpoint_000000000042")
        assert by_name["meta"]["t"] == direct["meta"]["t"] == 42

    def test_missing_chunk_warns_but_completes(self, tmp_path):
        bp = _write_checkpoint(tmp_path, n_chunks={0: 2, 1: 0, 2: 0, 3: 0})
        report = build_report(bp)
        assert len(report["warnings"]) == 1
        assert "expects 2" in report["warnings"][0]
        assert "preflop" in report["streets"]

    def test_sampled_report_meta(self, tmp_path):
        # Two chunks per street; sample 1 → sampled=True, exact total preserved.
        cp = tmp_path / "bp" / "checkpoint_000000000042"
        cp.mkdir(parents=True)
        for i in range(2):
            np.save(cp / f"strategy_0_chunk_{i:06d}.npy", STRAT_ROWS)
        joblib.dump(
            {"t": 42, "n_players": 2, "n_chunks_per_street": {0: 2, 1: 0, 2: 0, 3: 0}},
            cp / "server_state.pkl",
        )
        report = build_report(tmp_path / "bp", sample_chunks=1)
        assert report["meta"]["sampled"] is True
        assert report["meta"]["sample_chunks"] == 1
        assert report["streets"]["preflop"]["n_infosets_total"] == 8

    def test_missing_checkpoint_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_report(tmp_path)

    def test_analyze_writes_artifacts(self, tmp_path, capsys):
        bp = _write_checkpoint(tmp_path)
        out = tmp_path / "out"
        report = analyze(bp, out_dir=out)
        with open(out / "blueprint_metrics.json") as fh:
            on_disk = json.load(fh)
        assert on_disk["meta"]["t"] == report["meta"]["t"] == 42
        md = (out / "blueprint_metrics.md").read_text()
        assert "# Blueprint metrics" in md
        assert "## Preflop" in md
        assert "| action | play freq | mean strategy |" in md
        human = capsys.readouterr().out
        assert "PREFLOP" in human
        assert "play freq" in human

    def test_main_cli(self, tmp_path, capsys):
        bp = _write_checkpoint(tmp_path)
        assert main([str(bp), "--out", str(tmp_path / "o"), "--full"]) == 0
        assert (tmp_path / "o" / "blueprint_metrics.json").exists()
        assert "PREFLOP" in capsys.readouterr().out

    def test_main_cli_regret_flag(self, tmp_path):
        bp = _write_checkpoint(tmp_path)
        assert main([str(bp), "--out", str(tmp_path / "o"), "--regret"]) == 0
        report = json.load(open(tmp_path / "o" / "blueprint_metrics.json"))
        assert "preflop" in report["regret"]


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


class TestHelpers:
    def test_hist_percentile_uniform(self):
        hist = np.ones(100, dtype=np.int64)
        assert _hist_percentile(hist, 50, 0.0, 1.0) == pytest.approx(0.5, abs=0.01)
        assert _hist_percentile(hist, 90, 0.0, 1.0) == pytest.approx(0.9, abs=0.01)

    def test_hist_percentile_empty(self):
        assert _hist_percentile(np.zeros(10, dtype=np.int64), 50, 0.0, 1.0) is None

    def test_label_fallback_prefix_is_positional(self):
        labels, canonical = _action_labels(6, street=2)   # turn is width 5
        assert canonical is False
        assert labels[:3] == ["fold", "call", "all_in"]
        assert labels[3:] == ["raise_#1", "raise_#2", "raise_#3"]
