"""Tests for the LU-decomposition matrix inversion path."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

# Ensure inversion2/ is on the path so we can import lu_factor, build_lu, runner_lu.
HERE = os.path.dirname(os.path.abspath(__file__))
INV2 = os.path.dirname(HERE)
if INV2 not in sys.path:
    sys.path.insert(0, INV2)

# Force standard cache for tests so we don't depend on the Hull C++ extension.
os.environ.setdefault("MINV_USE_STANDARD_CACHE", "1")

import _bootstrap  # noqa: F401, E402
from lu_factor import doolittle  # noqa: E402


def _well_conditioned(n: int, seed: int = 42) -> np.ndarray:
    """Generate a 5I + small-noise matrix that admits Doolittle LU."""
    rng = np.random.RandomState(seed)
    return 5.0 * np.eye(n) + 0.3 * rng.randn(n, n)


def test_doolittle_correct():
    """Factor random well-conditioned matrices and verify L*U == A, L unit lower-tri."""
    for n in (5, 10):
        A = _well_conditioned(n)
        L, U = doolittle(A)
        assert np.allclose(L @ U, A, atol=1e-10)
        # L is unit lower-triangular: diagonal = 1, strict upper = 0.
        assert np.allclose(np.diag(L), np.ones(n))
        assert np.allclose(np.triu(L, k=1), 0.0)
        # U is upper-triangular.
        assert np.allclose(np.tril(U, k=-1), 0.0)


def test_doolittle_raises_on_zero_pivot():
    """A[0,0]=0 (with no row below to swap) must raise ValueError."""
    A = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    with pytest.raises(ValueError, match="[Zz]ero pivot"):
        doolittle(A)


@pytest.mark.parametrize("n", [4, 5])
def test_lu_solve_matches_numpy(n):
    """Build an LU model for a random well-conditioned matrix; solve A x = b for
    several random b's and assert results match np.linalg.solve."""
    from build_lu import build_for_matrix
    from runner_lu import invert as _  # ensure module imports cleanly

    A = _well_conditioned(n)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = build_for_matrix(A, model_dir=tmpdir)
        model_path = result["model_path"]
        assert os.path.exists(model_path)
        assert os.path.exists(model_path + ".slots.json")

        # Load and run inversion to get all columns at once (= solve for I).
        from runner_lu import invert
        X = invert(A, model_path)

        Xref = np.linalg.inv(A)
        max_err = float(np.max(np.abs(X - Xref)))
        assert max_err < 1e-9, f"n={n}: max_err {max_err:.2e}"


def test_lu_full_inversion_n5():
    """End-to-end inversion for n=5 with target max_abs_err < 1e-9."""
    from build_lu import build_for_matrix
    from runner_lu import invert

    A = _well_conditioned(5)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = build_for_matrix(A, model_dir=tmpdir)
        X = invert(A, result["model_path"])

    Xref = np.linalg.inv(A)
    max_err = float(np.max(np.abs(X - Xref)))
    assert max_err < 1e-9, f"max_err {max_err:.2e}"
