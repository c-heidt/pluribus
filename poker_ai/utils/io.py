import copy
import datetime
import hashlib
import json
import logging
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np

log = logging.getLogger("poker_ai.utils.io")


class NumpyJSONEncoder(json.JSONEncoder):
    """Handle those pesky numpy arrays on serialisation."""

    def default(self, obj):
        """Method to handle the conversion of numpy types to Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJSONEncoder, self).default(obj)


def to_dict(**kwargs) -> Dict[str, Any]:
    """Hacky method to convert weird collections dicts to regular dicts."""
    return json.loads(json.dumps(copy.deepcopy(kwargs)))


def print_strategy(strategy: Dict[str, Dict[str, int]]):
    """Print strategy."""
    for info_set, action_to_probabilities in sorted(strategy.items()):
        norm = sum(list(action_to_probabilities.values()))
        log.info(f"{info_set}")
        for action, probability in action_to_probabilities.items():
            log.info(f"  - {action}: {probability / norm:.2f}")


def create_dir(dir_name: str = "results") -> Path:
    """Create (or reuse) a directory by name for saving training output."""
    path: Path = Path(f"./{dir_name}")
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Atomic save utilities
# ---------------------------------------------------------------------------

def atomic_joblib_dump(obj: Any, path: Union[str, Path]) -> None:
    """Save *obj* with joblib atomically via a temp file then rename.

    Using a temp file in the same directory guarantees the rename is atomic on
    POSIX (rename syscall) even across NFS when the tmp and target are on the
    same mount point.  On failure the temp file is cleaned up and the original
    path is left untouched.
    """
    path = Path(path)
    # Place the temp file beside the target so os.rename is atomic (same fs).
    tmp_fd, tmp_str = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp.joblib", prefix=path.stem + "_"
    )
    tmp_path = Path(tmp_str)
    try:
        os.close(tmp_fd)
        joblib.dump(obj, tmp_path)
        shutil.move(str(tmp_path), str(path))
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"atomic_joblib_dump failed for {path}: {exc}") from exc


def atomic_numpy_save(arr: np.ndarray, path: Union[str, Path]) -> None:
    """Save a numpy array atomically to *path* (*.npy* format).

    Integrity is preserved: if the write fails the original file is untouched.
    The temp file is written to the same directory as the target so that
    ``os.rename`` is always on the same filesystem.
    """
    path = Path(path)
    tmp_fd, tmp_str = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp.npy", prefix=path.stem + "_"
    )
    tmp_path = Path(tmp_str)
    try:
        os.close(tmp_fd)
        np.save(tmp_path, arr, allow_pickle=False)
        shutil.move(str(tmp_path), str(path))
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"atomic_numpy_save failed for {path}: {exc}") from exc


def atomic_numpy_load(
    path: Union[str, Path],
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Load a numpy array from *path* with optional integrity checks.

    Parameters
    ----------
    path:
        File path to read (*.npy* format).
    expected_shape:
        If provided, raises ``ValueError`` when the loaded array shape does not
        match.  Pass a tuple with ``-1`` for any dimension you don't want to
        constrain (e.g. ``(-1, 5)`` to require exactly 5 columns).
    expected_dtype:
        If provided, raises ``ValueError`` when the loaded dtype does not
        match.

    Returns
    -------
    np.ndarray
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"numpy array not found at {path}")

    arr: np.ndarray = np.load(path, allow_pickle=False)

    if expected_shape is not None:
        for dim_got, dim_want in zip(arr.shape, expected_shape):
            if dim_want != -1 and dim_got != dim_want:
                raise ValueError(
                    f"Shape mismatch loading {path}: expected {expected_shape}, "
                    f"got {arr.shape}"
                )
        if len(arr.shape) != len(expected_shape):
            raise ValueError(
                f"Rank mismatch loading {path}: expected {len(expected_shape)}D, "
                f"got {len(arr.shape)}D"
            )

    if expected_dtype is not None and arr.dtype != np.dtype(expected_dtype):
        raise ValueError(
            f"dtype mismatch loading {path}: expected {expected_dtype}, got {arr.dtype}"
        )

    return arr


# ---------------------------------------------------------------------------
# 128-bit infoset hashing (Phase 1.2)
# ---------------------------------------------------------------------------

def hash_info_set_128(info_set: str) -> Tuple[int, int]:
    """Return a 128-bit hash of *info_set* as a pair of unsigned 64-bit ints.

    Uses xxhash.xxh3_128 when available (fast path, ~10x faster than blake2b)
    and falls back to hashlib.blake2b (16-byte digest) otherwise.

    Returns
    -------
    (high_64, low_64) — both unsigned 64-bit integers.

    Notes
    -----
    xxhash is non-cryptographic but has excellent collision resistance for
    string keys at this digest length.  blake2b is cryptographic and has
    negligible collision probability.  Both are acceptable for infoset
    indexing.
    """
    try:
        import xxhash  # fast path
        digest_int: int = xxhash.xxh3_128(info_set).intdigest()
        high = digest_int >> 64
        low = digest_int & 0xFFFF_FFFF_FFFF_FFFF
        return high, low
    except ImportError:
        digest: bytes = hashlib.blake2b(
            info_set.encode("utf-8"), digest_size=16
        ).digest()
        high, low = struct.unpack("<QQ", digest)
        return high, low


def hash_info_set_bytes(info_set: str) -> bytes:
    """Return the 16-byte (128-bit) raw digest for *info_set*.

    This is the canonical key format used when storing hashes in LMDB.
    """
    high, low = hash_info_set_128(info_set)
    return struct.pack("<QQ", high, low)

