"""Doolittle LU factorization (no pivoting) and triangular solves.

Used by lu_direct_reference.py to verify the same numpy math the merged
transformer carries out internally. The build pipeline calls doolittle()
to bake L and U into per-token weights; nothing else here runs at inference.
"""
from __future__ import annotations

import numpy as np


def doolittle(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Factor A = L*U, L unit lower-tri, U upper-tri. Raises on zero pivot."""
    A = A.astype(np.float64)
    n = A.shape[0]
    L = np.eye(n, dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]
        if abs(U[i, i]) < 1e-12:
            raise ValueError(f"Zero pivot at U[{i},{i}]={U[i,i]:.2e}; needs pivoting")
        for k in range(i + 1, n):
            L[k, i] = (A[k, i] - L[k, :i] @ U[:i, i]) / U[i, i]
    return L, U


def forward_sub(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve L y = b for unit lower-triangular L."""
    n = L.shape[0]
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y


def back_sub(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve U x = y for upper-triangular U."""
    n = U.shape[0]
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x
