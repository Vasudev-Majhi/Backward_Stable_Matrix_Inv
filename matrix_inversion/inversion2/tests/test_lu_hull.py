"""LU pipeline parity tests under HullKVCache.

Mirrors test_lu.py but does NOT force the standard cache, so it actually
exercises the Hull C++ extension.  All Hull-touching tests are skipped if the
extension can't be built locally.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
INV2 = os.path.dirname(HERE)
if INV2 not in sys.path:
    sys.path.insert(0, INV2)

# Make sure no prior test has stuck the standard-cache override.
os.environ.pop("MINV_USE_STANDARD_CACHE", None)

import _bootstrap  # noqa: F401, E402


def _hull_available():
    try:
        from transformer_vm.attention.hull_cache import HullKVCache
        _ = HullKVCache(1, 1)
        return True
    except Exception:
        return False


HULL = _hull_available()
pytestmark = pytest.mark.skipif(
    not HULL, reason="hull_ext build not available; run on server"
)


def _well_conditioned(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return 5.0 * np.eye(n) + 0.3 * rng.randn(n, n)


def _fresh_runner_lu():
    """Re-import runner_lu in a Hull-only env."""
    for mod in ("runner_lu", "build_lu"):
        sys.modules.pop(mod, None)
    import runner_lu  # noqa: WPS433
    import build_lu  # noqa: WPS433
    assert runner_lu.USING_HULL, "expected Hull cache active"
    return runner_lu, build_lu


@pytest.mark.parametrize("n", [4, 5])
def test_lu_solve_matches_numpy_hull(n):
    runner_lu, build_lu = _fresh_runner_lu()
    A = _well_conditioned(n)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = build_lu.build_for_matrix(A, model_dir=tmpdir)
        X = runner_lu.invert(A, result["model_path"])

    Xref = np.linalg.inv(A)
    max_err = float(np.max(np.abs(X - Xref)))
    assert max_err < 1e-9, f"n={n}: max_err {max_err:.2e}"


def test_lu_full_inversion_n5_hull():
    runner_lu, build_lu = _fresh_runner_lu()
    A = _well_conditioned(5)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = build_lu.build_for_matrix(A, model_dir=tmpdir)
        X = runner_lu.invert(A, result["model_path"])
    Xref = np.linalg.inv(A)
    max_err = float(np.max(np.abs(X - Xref)))
    assert max_err < 1e-9, f"max_err {max_err:.2e}"


def test_lu_full_inversion_n10_hull():
    """Larger n where the V-cache patch fires across many tokens."""
    runner_lu, build_lu = _fresh_runner_lu()
    A = _well_conditioned(10)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = build_lu.build_for_matrix(A, model_dir=tmpdir)
        X = runner_lu.invert(A, result["model_path"])
    Xref = np.linalg.inv(A)
    max_err = float(np.max(np.abs(X - Xref)))
    assert max_err < 1e-9, f"max_err {max_err:.2e}"
