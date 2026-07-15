"""Minimal harness for the Hull V-cache patch (insert_v / update_v_at).

Pins down whether the C++ HullKVCache replaces V at an existing seq instead of
appending a new entry, for both LATEST and AVERAGE tiebreaks.

If the Hull C++ extension can't build locally (e.g. Windows without cl), the
Hull-touching tests are skipped and only the BruteHullCache parity assertions
run.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
INV2 = os.path.dirname(HERE)
if INV2 not in sys.path:
    sys.path.insert(0, INV2)

import _bootstrap  # noqa: F401, E402

from brute_hull_cache import BruteHullCache  # noqa: E402


def _try_import_hull():
    try:
        from transformer_vm.attention.hull_cache import HullKVCache
        # Force-load the extension here so a build failure is caught now.
        _ = HullKVCache(1, 1)
        return HullKVCache
    except Exception as e:  # noqa: BLE001
        return None


HullKVCache = _try_import_hull()
HULL_AVAILABLE = HullKVCache is not None


def _t(arr):
    return torch.tensor(arr, dtype=torch.float64)


def _both_caches():
    """Yield (name, factory) for both Hull (if available) and Brute caches."""
    yield ("brute", lambda: BruteHullCache(1, 1))
    if HULL_AVAILABLE:
        yield ("hull", lambda: HullKVCache(1, 1))


# ─────────────────────────────────────────────────────────────────────────
# Test 1: update_v_at replaces V at the existing entry, no append
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tiebreak_latest", [True, False])
@pytest.mark.parametrize("name,factory", list(_both_caches()))
def test_insert_v_replaces_value_for_same_position(name, factory, tiebreak_latest):
    cache = factory()
    cache.set_tiebreak(0, 0, tiebreak_latest)

    # 5 entries with distinct (kx, ky) so each position is the unique
    # argmax of its own key.  Mimics the LATEST-mode 2D-key shape from
    # _to_2d_key (kx linear in position, ky quadratic + inv_log_pos jitter).
    keys = [(2.0 * i, -(i * i) + (0.3 / max(np.log1p(i + 1), 1e-9)))
            for i in range(5)]
    vals = [(i + 0.5, i + 1.5) for i in range(5)]

    for k, v in zip(keys, vals):
        cache.layer_step(0, _t(k), _t(k), _t(v))

    # Replace V at seq=2 (position 2 in our setup).
    new_v = (99.0, 100.0)
    cache.update_v_at(0, 2, _t(new_v))

    # Cross-check against brute oracle.  The probe inserts a 6th entry at
    # seq=5 with V=(0,0) and queries with K close to K_2 but distinct, so
    # K_2 is still the unique argmax for that query.
    brute = BruteHullCache(1, 1)
    brute.set_tiebreak(0, 0, tiebreak_latest)
    for k, v in zip(keys, vals):
        brute.layer_step(0, _t(k), _t(k), _t(v))
    brute.update_v_at(0, 2, _t(new_v))

    probe_k = (4.001, -3.999)
    probe_q = (4.0, -3.9)  # picks position 2 (kx=4, ky≈-3.9) as argmax
    out_test = cache.layer_step(0, _t(probe_k), _t(probe_q), _t((0.0, 0.0)))
    out_oracle = brute.layer_step(0, _t(probe_k), _t(probe_q), _t((0.0, 0.0)))

    assert torch.allclose(out_test, out_oracle, atol=1e-10), (
        f"{name} tb={tiebreak_latest}: out_test={out_test} vs "
        f"oracle={out_oracle}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Test 2: clear() resets all state
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,factory", list(_both_caches()))
def test_clear_then_insert_round_trip(name, factory):
    cache = factory()
    cache.set_tiebreak(0, 0, True)

    for i in range(5):
        cache.layer_step(0, _t((2.0 * i, -i * i + 0.1)),
                         _t((2.0 * i, -i * i + 0.1)),
                         _t((float(i), float(i + 1))))

    cache.clear()
    cache.set_tiebreak(0, 0, True)

    # Single insert + query -> must return that V exactly.
    k = _t((10.0, -5.0))
    v = _t((42.0, 43.0))
    out = cache.layer_step(0, k, k, v)
    assert torch.allclose(out, v, atol=1e-12), (
        f"{name}: clear() did not reset state — got {out}, expected {v}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Test 3: end-to-end LU n=4 with Hull matches numpy
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HULL_AVAILABLE,
                    reason="hull_ext build not available; run on server")
def test_runner_vcache_patch_minimal_lu(tmp_path):
    # Make sure runner_lu picks Hull, not the standard fallback.
    os.environ.pop("MINV_USE_STANDARD_CACHE", None)
    # Re-import runner_lu to pick up the env change.
    for mod in ("runner_lu", "build_lu"):
        sys.modules.pop(mod, None)

    import runner_lu  # noqa: WPS433
    import build_lu  # noqa: WPS433

    assert runner_lu.USING_HULL, "expected Hull cache active"

    rng = np.random.RandomState(7)
    n = 4
    A = 5.0 * np.eye(n) + 0.3 * rng.randn(n, n)

    r = build_lu.build_for_matrix(A, model_dir=str(tmp_path))
    X = runner_lu.invert(A, r["model_path"])
    Xref = np.linalg.inv(A)
    max_err = float(np.max(np.abs(X - Xref)))
    assert max_err < 1e-9, f"Hull LU n={n}: max_err={max_err:.2e}"


# ─────────────────────────────────────────────────────────────────────────
# Test 4: update_v_at finds the entry by seq even if K snapshot would differ
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,factory", list(_both_caches()))
def test_update_v_at_uses_recorded_seq_not_passed_keys(name, factory):
    """update_v_at_seq looks up the entry by seq. The new V is applied
    regardless of any K passed in; insert_v's `keys` parameter is unused.
    """
    cache = factory()
    cache.set_tiebreak(0, 0, True)

    keys = [(2.0 * i, -(i * i) + 0.1) for i in range(3)]
    vals = [(i + 0.5, i + 1.5) for i in range(3)]
    for k, v in zip(keys, vals):
        cache.layer_step(0, _t(k), _t(k), _t(v))

    new_v = (123.0, 456.0)
    # Pass arbitrary garbage for keys; insert_v should ignore it.
    cache.insert_v(0, _t((9999.0, -9999.0)), _t(new_v))

    # Now query at K_2: the latest-tiebreak resolution must return new_v
    # because we updated the entry inserted at seq=2.
    out = cache.layer_step(0, _t(keys[2]), _t(keys[2]), _t((0.0, 0.0)))

    # Cross-check with brute oracle
    brute = BruteHullCache(1, 1)
    brute.set_tiebreak(0, 0, True)
    for k, v in zip(keys, vals):
        brute.layer_step(0, _t(k), _t(k), _t(v))
    brute.insert_v(0, _t((9999.0, -9999.0)), _t(new_v))
    out_oracle = brute.layer_step(0, _t(keys[2]), _t(keys[2]), _t((0.0, 0.0)))

    assert torch.allclose(out, out_oracle, atol=1e-10), (
        f"{name}: out={out} vs oracle={out_oracle}"
    )
