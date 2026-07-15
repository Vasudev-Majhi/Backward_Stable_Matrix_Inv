
import _bootstrap, numpy as np, time, sys, os
print('python', sys.version.split()[0], flush=True)
os.makedirs("/tmp/hull_smoke", exist_ok=True)
from build_lu import build_for_matrix
import runner_lu
print('USING_HULL =', runner_lu.USING_HULL, flush=True)
assert runner_lu.USING_HULL, 'expected Hull cache active'
rng = np.random.RandomState(42)
A = 5.0*np.eye(10) + 0.3*rng.randn(10,10)
t0=time.time(); r=build_for_matrix(A, model_dir="/tmp/hull_smoke"); t_build=time.time()-t0
t0=time.time(); X=runner_lu.invert(A, r["model_path"]); t_run=time.time()-t0
err=float(np.max(np.abs(X-np.linalg.inv(A))))
print(f"n=10 build={t_build:.2f}s infer={t_run:.2f}s max_err={err:.3e}", flush=True)
