"""Quick check: build+invert at a single n with current cache."""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa

from build_lu import build_for_matrix
from runner_lu import invert, USING_HULL


def main(n: int = 5):
    os.makedirs("/tmp/lu_test", exist_ok=True)
    np.random.seed(42)
    A = 5.0 * np.eye(n) + 0.3 * np.random.randn(n, n)
    r = build_for_matrix(A, model_dir="/tmp/lu_test")
    X = invert(A, r["model_path"])
    err = float(np.max(np.abs(X - np.linalg.inv(A))))
    print(f"n={n}  using_hull={USING_HULL}  max_err={err:.6e}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(n)
