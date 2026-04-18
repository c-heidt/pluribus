"""Unit tests for MemmapLookup and load_info_set_lut."""
import pickle

import joblib
import numpy as np
import pytest

from information_abstraction import load_info_set_lut
from information_abstraction.lookup import MemmapLookup


def _make_memmap_lookup(tmp_path, combos, cluster_ids_by_row):
    """Write a cluster_ids.dat and return a MemmapLookup pointing at it.

    ``combos`` is a reference CardCombos for card_to_idx/n_cards.
    """
    path = tmp_path / "cluster_ids.dat"
    mm = np.memmap(
        path, dtype=np.uint16, mode="w+",
        shape=(len(cluster_ids_by_row),),
    )
    mm[:] = cluster_ids_by_row.astype(np.uint16)
    mm.flush()
    return MemmapLookup(
        ids_path=path,
        card_to_idx=combos._card_to_idx,
        n_cards=combos._n_cards,
        n_rows=len(cluster_ids_by_row),
    )


class TestMemmapLookup:
    def test_round_trip_every_combo(self, tmp_path, tiny_combos):
        # Assign a distinct cluster to every row so a lookup mismatch surfaces.
        river = tiny_combos.river
        cluster_ids = np.arange(len(river), dtype=np.uint16)
        lookup = _make_memmap_lookup(tmp_path, tiny_combos, cluster_ids)
        for row, combo in enumerate(river):
            assert lookup[tuple(int(c) for c in combo)] == row

    def test_pickle_preserves_rows(self, tmp_path, tiny_combos):
        cluster_ids = np.arange(len(tiny_combos.flop), dtype=np.uint16)
        original = _make_memmap_lookup(tmp_path, tiny_combos, cluster_ids)
        reloaded = pickle.loads(pickle.dumps(original))
        for combo in tiny_combos.flop[:50]:
            key = tuple(int(c) for c in combo)
            assert reloaded[key] == original[key]

    def test_pickle_drops_memmap_handle(self, tmp_path, tiny_combos):
        cluster_ids = np.zeros(len(tiny_combos.flop), dtype=np.uint16)
        lookup = _make_memmap_lookup(tmp_path, tiny_combos, cluster_ids)
        _ = lookup[tuple(int(c) for c in tiny_combos.flop[0])]
        assert lookup._mm is not None
        reloaded = pickle.loads(pickle.dumps(lookup))
        assert reloaded._mm is None  # lazy-load on first access

    def test_rebind_redirects_reads(self, tmp_path, tiny_combos):
        cluster_ids_a = np.full(len(tiny_combos.flop), 1, dtype=np.uint16)
        cluster_ids_b = np.full(len(tiny_combos.flop), 2, dtype=np.uint16)
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        lookup = _make_memmap_lookup(dir_a, tiny_combos, cluster_ids_a)
        sample = tuple(int(c) for c in tiny_combos.flop[0])
        assert lookup[sample] == 1
        _make_memmap_lookup(dir_b, tiny_combos, cluster_ids_b)
        lookup.rebind(dir_b / "cluster_ids.dat")
        assert lookup[sample] == 2


class TestLoadInfoSetLut:
    def test_empty_lut_path_returns_empty_dict(self):
        assert load_info_set_lut("", pickle_dir=False) == {}
        assert load_info_set_lut(None, pickle_dir=False) == {}

    def test_joblib_path_round_trip(self, tmp_path):
        fixture = {
            "pre_flop": {(1, 2): 0},
            "flop": {(1, 2, 3, 4, 5): 7},
        }
        joblib.dump(fixture, tmp_path / "card_info_lut.joblib")
        loaded = load_info_set_lut(tmp_path)
        assert loaded == fixture

    def test_missing_joblib_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_info_set_lut(tmp_path)

    def test_legacy_pickle_dir_path_round_trip(self, tmp_path):
        pieces = {
            "preflop_lossless.pkl": {(1, 2): 0},
            "flop_lossy_2.pkl": {(1, 2, 3, 4, 5): 1},
            "turn_lossy_2.pkl": {(1, 2, 3, 4, 5, 6): 2},
            "river_lossy_2.pkl": {(1, 2, 3, 4, 5, 6, 7): 3},
        }
        for name, payload in pieces.items():
            joblib.dump(payload, tmp_path / name)
        loaded = load_info_set_lut(tmp_path, pickle_dir=True)
        assert loaded["pre_flop"] == pieces["preflop_lossless.pkl"]
        assert loaded["flop"] == pieces["flop_lossy_2.pkl"]
        assert loaded["turn"] == pieces["turn_lossy_2.pkl"]
        assert loaded["river"] == pieces["river_lossy_2.pkl"]

    def test_legacy_pickle_dir_missing_file_raises(self, tmp_path):
        # Only write three of the four legacy files.
        joblib.dump({}, tmp_path / "preflop_lossless.pkl")
        joblib.dump({}, tmp_path / "flop_lossy_2.pkl")
        joblib.dump({}, tmp_path / "turn_lossy_2.pkl")
        with pytest.raises(ValueError, match="File not found"):
            load_info_set_lut(tmp_path, pickle_dir=True)
