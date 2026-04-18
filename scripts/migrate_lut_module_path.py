"""One-time migration: rewrite card_info_lut.joblib so pickled MemmapLookup
instances point at poker_ai.information_abstraction.lookup instead of the
removed poker_ai.clustering.unified_lut_builder.

The information-abstraction refactor moved MemmapLookup to a new package.
Pre-refactor LUTs embed the old module path in their pickle metadata and
fail to unpickle with ``ModuleNotFoundError: No module named
'poker_ai.clustering'`` under the new code.  Re-pickling through a short
sys.modules stub is enough: pickle stamps the live class's __module__ on
dump, so after round-tripping the file references the new path and loads
natively.

Usage
-----
    python scripts/migrate_lut_module_path.py [lut_dir]

where lut_dir contains card_info_lut.joblib (default:
data/clustering/20cards_exact).  The original is backed up to
card_info_lut.joblib.bak and the migrated version overwrites
card_info_lut.joblib.
"""
import shutil
import sys
import types
from pathlib import Path

import joblib


# ---------------------------------------------------------------------------
# Install compat stubs so pickle can resolve the old module path
# ---------------------------------------------------------------------------

def _install_stubs():
    """Register sys.modules entries for the removed clustering paths."""
    from poker_ai.information_abstraction import lookup as _new_lookup

    old_pkg = types.ModuleType("poker_ai.clustering")
    old_mod = types.ModuleType("poker_ai.clustering.unified_lut_builder")
    old_mod.MemmapLookup = _new_lookup.MemmapLookup
    old_pkg.unified_lut_builder = old_mod

    sys.modules.setdefault("poker_ai.clustering", old_pkg)
    sys.modules.setdefault(
        "poker_ai.clustering.unified_lut_builder", old_mod,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def migrate(lut_dir: str) -> None:
    lut_path = Path(lut_dir) / "card_info_lut.joblib"
    backup_path = lut_path.with_suffix(".joblib.bak")

    if not lut_path.exists():
        print(f"ERROR: {lut_path} not found.")
        sys.exit(1)

    if backup_path.exists():
        print(f"Backup already exists at {backup_path} — skipping backup step.")
    else:
        shutil.copy2(str(lut_path), str(backup_path))
        print(f"Backed up original to {backup_path}")

    _install_stubs()

    print(f"Loading {lut_path} ...")
    lut = joblib.load(str(lut_path))

    # Report structure so the user can sanity-check what was migrated
    for street, value in lut.items():
        print(f"  {street}: {type(value).__name__}")

    print(f"Saving migrated LUT to {lut_path} ...")
    joblib.dump(lut, str(lut_path))
    print("Done.")


if __name__ == "__main__":
    lut_dir = (
        sys.argv[1] if len(sys.argv) > 1
        else "data/clustering/20cards_exact"
    )
    migrate(lut_dir)
