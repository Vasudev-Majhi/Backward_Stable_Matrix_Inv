"""Pure numpy Jacobi reference implementation for A·x = b.

No vocabulary quantization — full float64 precision throughout.
Used to verify transformer output against a known-good solver.

Jacobi update:
    x_new[i] = (b[i] - sum_{k!=i} A[i,k]*x[k]) / A[i,i]

Convergence requires A to be strictly diagonally dominant.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np


def jacobi_solve(A: np.ndarray, b: np.ndarray, T: int) -> np.ndarray:
    """Run T Jacobi sweeps for A·x = b, starting from x=0. Returns x."""
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)
    for _ in range(T):
        x_new = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = b[i]
            for k in range(n):
                if k != i:
                    s -= A[i, k] * x[k]
            x_new[i] = s / A[i, i]
        x = x_new
    return x


def jacobi_invert(A: np.ndarray, T: int) -> np.ndarray:
    """Invert A by running Jacobi on each column of the identity."""
    n = A.shape[0]
    X = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        e_j = np.zeros(n, dtype=np.float64)
        e_j[j] = 1.0
        X[:, j] = jacobi_solve(A, e_j, T)
    return X


def auto_T(n: int, A: np.ndarray | None = None) -> int:
    """Default iteration budget: enough for diagonally-dominant matrices.

    For matrices with high spectral radius rho (> 0.9), caller should override
    with T = ceil(log(target_eps) / log(rho)).
    """
    return max(20, 5 * n)


if __name__ == "__main__":
    A = np.array([[5., 1., 0.], [1., 5., 1.], [0., 1., 5.]])
    T = auto_T(3)
    X = jacobi_invert(A, T)
    Xref = np.linalg.inv(A)
    print("Jacobi X:")
    print(X)
    print("numpy ref:")
    print(Xref)
    print("max abs err:", np.max(np.abs(X - Xref)))
