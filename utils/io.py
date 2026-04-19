"""Atomic filesystem-write primitives.

Every writer in the codebase should persist through one of these helpers so
the target file is always either the previous contents or the complete new
payload — never a partial write from a killed process or full disk.  The
``mkstemp`` call places the temp file in the *same* directory as the target
so ``shutil.move`` collapses to a POSIX ``rename`` syscall (atomic even on
NFS when both sides live on the same mount point).
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import joblib
import numpy as np


def atomic_joblib_dump(obj: Any, path: Union[str, Path]) -> None:
    """Save *obj* with joblib atomically via a temp file then rename.

    On failure the temp file is cleaned up and *path* is left untouched.
    """
    path = Path(path)
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
        raise RuntimeError(
            f"atomic_joblib_dump failed for {path}: {exc}"
        ) from exc


def atomic_numpy_save(arr: np.ndarray, path: Union[str, Path]) -> None:
    """Save a numpy array atomically to *path* in ``.npy`` format.

    Uses ``allow_pickle=False`` so the file is a pure array dump with no
    executable pickle payload — safe to load in untrusted contexts.
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
        raise RuntimeError(
            f"atomic_numpy_save failed for {path}: {exc}"
        ) from exc


def atomic_numpy_load(
    path: Union[str, Path],
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Load a numpy ``.npy`` file with optional integrity checks.

    Parameters
    ----------
    path:
        File to read.
    expected_shape:
        If provided, raises ``ValueError`` when the loaded array shape
        does not match.  Pass ``-1`` for any dimension you don't want to
        constrain (e.g. ``(-1, 5)`` for "any rows, exactly 5 columns").
    expected_dtype:
        If provided, raises ``ValueError`` when the loaded dtype does not
        match.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"numpy array not found at {path}")

    arr: np.ndarray = np.load(path, allow_pickle=False)

    if expected_shape is not None:
        if len(arr.shape) != len(expected_shape):
            raise ValueError(
                f"Rank mismatch loading {path}: "
                f"expected {len(expected_shape)}D, got {len(arr.shape)}D"
            )
        for dim_got, dim_want in zip(arr.shape, expected_shape):
            if dim_want != -1 and dim_got != dim_want:
                raise ValueError(
                    f"Shape mismatch loading {path}: "
                    f"expected {expected_shape}, got {arr.shape}"
                )

    if expected_dtype is not None and arr.dtype != np.dtype(expected_dtype):
        raise ValueError(
            f"dtype mismatch loading {path}: "
            f"expected {expected_dtype}, got {arr.dtype}"
        )
    return arr
