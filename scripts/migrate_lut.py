"""One-time migration script: convert card_info_lut.joblib from old Card-object
keys to plain int keys.

The LUT was saved when poker hands were represented as poker_ai.poker.card.Card
objects.  The new code uses plain Cactus Kev 32-bit integers throughout.  The
Card object's _eval_card attribute holds the identical integer, so conversion
is lossless.

The river entry is a MemmapLookup whose __getitem__ already calls int() on
each card, so it already accepts both representations.  Only the pre-flop,
flop, and turn dicts need their keys rewritten.

Usage
-----
    python scripts/migrate_lut.py [lut_dir]

where lut_dir contains card_info_lut.joblib (default: data/clustering/20cards_exact).
The original file is kept as card_info_lut.joblib.bak and the migrated version
overwrites card_info_lut.joblib.
"""
import shutil
import sys
import types
from pathlib import Path

import joblib


# ---------------------------------------------------------------------------
# Install compat stubs so pickle can deserialise old Card objects
# ---------------------------------------------------------------------------

def _install_stubs():
    class Card:
        """Stub matching the old poker_ai.poker.card.Card pickle layout."""
        def __setstate__(self, state):
            self.__dict__.update(state)

        def __int__(self):
            return int(self._eval_card)

        def __repr__(self):
            return f"Card(_rank={self._rank}, _suit={self._suit!r})"

    poker_mod = types.ModuleType("poker_ai.poker")
    card_mod = types.ModuleType("poker_ai.poker.card")
    card_mod.Card = Card
    poker_mod.card = card_mod

    # Register under every name pickle might look up
    sys.modules.setdefault("poker_ai.poker", poker_mod)
    sys.modules.setdefault("poker_ai.poker.card", card_mod)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _convert_street(street_lut):
    """Convert a {tuple-of-Cards: cluster_id} dict to {tuple-of-ints: cluster_id}.

    Skips conversion for MemmapLookup objects (river) since they already
    handle plain ints internally.
    """
    # MemmapLookup and other non-dict objects are left as-is
    if not isinstance(street_lut, dict):
        return street_lut

    new = {}
    for combo, cluster_id in street_lut.items():
        # Each element of combo may be a Card object or already an int
        new_key = tuple(int(c) for c in combo)
        new[new_key] = cluster_id
    return new


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
    old_lut = joblib.load(str(lut_path))

    print("Converting keys ...")
    new_lut = {}
    for street, street_lut in old_lut.items():
        if isinstance(street_lut, dict):
            before = len(street_lut)
            new_lut[street] = _convert_street(street_lut)
            after = len(new_lut[street])
            print(f"  {street}: {before} entries converted (keys: {after} unique)")
        else:
            new_lut[street] = street_lut
            print(f"  {street}: MemmapLookup — kept as-is")

    # Validate that a sample key is now a tuple of ints
    for street, slut in new_lut.items():
        if isinstance(slut, dict) and slut:
            sample_key = next(iter(slut))
            assert all(isinstance(c, int) for c in sample_key), (
                f"{street}: key {sample_key} still contains non-int elements"
            )
            print(f"  {street}: key type OK -> {type(sample_key[0])}")

    print(f"Saving migrated LUT to {lut_path} ...")
    joblib.dump(new_lut, str(lut_path))
    print("Done.")


if __name__ == "__main__":
    lut_dir = sys.argv[1] if len(sys.argv) > 1 else "data/clustering/20cards_exact"
    migrate(lut_dir)
