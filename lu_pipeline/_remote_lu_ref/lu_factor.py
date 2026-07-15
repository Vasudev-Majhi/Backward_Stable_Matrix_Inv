"""Doolittle LU factorization (no pivoting). Used by build_lu.py."""
from __future__ import annotations

import numpy as np


def doolittle(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Factor A = L*U where L is unit lower-tri, U is upper-tri.

    Raises ValueError if a pivot vanishes (|U[i,i]| < 1e-12).
    Returns (L, U) as float64.
    """
    A = A.astype(np.float64)
    n = A.shape[0]
    L = np.eye(n, dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]
        if abs(U[i, i]) < 1e-12:
            raise ValueError(
                f"Zero pivot at U[{i},{i}]={U[i,i]:.2e}; matrix needs pivoting (Phase 2)"
            )
        for k in range(i + 1, n):
            L[k, i] = (A[k, i] - L[k, :i] @ U[:i, i]) / U[i, i]
    return L, U
