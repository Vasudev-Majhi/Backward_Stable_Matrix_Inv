"""Shared harness for the revision experiments.

Two capabilities the shipped runner does not have:

  1. `solve()` drives Transformer 1 with an ARBITRARY run-time right-hand side
     b, supplied by the runner at each rhs_i token, with NO recompilation.
     The compiled model is unchanged; only the value the runner hands in at
     rhs_i changes. This is the same contract under which the runner already
     hands y_i back at bck_i, so it is inside the paper's declared
     host/runner/transformer boundary (Table 6).

  2. `RecordingCache` observes every attention lookup: the selected (argmax)
     position, the softmax one-hotness, the true Shannon entropy computed
     WITHOUT the 1e-30 regulariser, and output finiteness. These are the
     behavioural proxies whose diagnosticity the revision measures.

Everything runs in whatever dtype is requested; float64 is the production
reference trace against which perturbed configurations are scored.
"""
from __future__ import annotations

import json
import os
import sys
import math

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "lu_pipeline"))
sys.path.insert(0, os.path.join(_ROOT, "craft"))
sys.path.insert(0, os.path.join(_ROOT, "craft", "cadj"))
sys.path.insert(0, os.path.join(_ROOT, "transformer-vm"))

DATASET = os.environ.get(
    "CRAFT_DATASET", os.path.join(_ROOT, "dataset", "circuit_dataset_rv.jsonl")
)

import _path  # noqa: F401,E402
from parse import parse_netlist  # type: ignore  # noqa: E402
from cadj_reference import K_LEVELS, SCALE, V_STEP  # type: ignore  # noqa: E402
from lu_direct_tokenize import PREDICTED, tokenize_lu_direct  # noqa: E402
from lu_direct_reference import build_partition  # noqa: E402
from lu_factor import back_sub, doolittle, forward_sub  # noqa: E402


def load_circuits(path: str | None = None) -> dict:
    path = path or DATASET
    out = {}
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            out[c["ID"]] = c
    return out


class RecordingCache:
    """StandardKVCache plus per-lookup instrumentation.

    Records, for every (layer, head, step): the argmax position, whether the
    softmax row is exactly one-hot, the true Shannon entropy (no regulariser),
    and whether the output is finite.
    """

    def __init__(self, n_layers, n_heads, dtype=torch.float64, record=True):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dtype = dtype
        self.record = record
        self._keys = [[] for _ in range(n_layers)]
        self._vals = [[] for _ in range(n_layers)]
        self.sel = []          # argmax position per lookup, flat
        self.onehot = []       # bool per lookup
        self.entropy = []      # float per lookup (nats, true)
        self.finite = []       # bool per lookup
        self.tied = []         # bool per lookup: max attained more than once

    def clear(self):
        self._keys = [[] for _ in range(self.n_layers)]
        self._vals = [[] for _ in range(self.n_layers)]

    def layer_step(self, layer, keys, queries, values):
        self._keys[layer].append(keys.clone())
        self._vals[layer].append(values.clone())
        d = keys.shape[0] // self.n_heads
        K = torch.stack(self._keys[layer]).reshape(-1, self.n_heads, d)
        V = torch.stack(self._vals[layer]).reshape(-1, self.n_heads, d)
        Q = queries.reshape(self.n_heads, -1)
        scores = torch.einsum("thi,hi->th", K, Q)
        weights = F.softmax(scores, dim=0)
        out = torch.einsum("th,thi->hi", weights, V)

        if self.record:
            s64 = scores.double()
            w = weights.double()
            mx = s64.max(dim=0).values
            am = s64.argmax(dim=0)
            ties = (s64 == mx.unsqueeze(0)).sum(dim=0)
            # True Shannon entropy: 0*log0 := 0, no 1e-30 regulariser.
            wl = torch.where(w > 0, w * torch.log(w), torch.zeros_like(w))
            ent = -wl.sum(dim=0)
            oh = (w.max(dim=0).values == 1.0)
            fin = torch.isfinite(out.double()).all(dim=-1)
            for h in range(self.n_heads):
                self.sel.append(int(am[h]))
                self.onehot.append(bool(oh[h]))
                self.entropy.append(float(ent[h]))
                self.finite.append(bool(fin[h]))
                self.tied.append(int(ties[h]) > 1)
        return out.flatten()


def _forward_step(model, cache, x_initial):
    x = x_initial.clone()
    for li, (attn, ff_in, ff_out) in enumerate(
        zip(model.attn, model.ff_in, model.ff_out, strict=True)
    ):
        q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
        out = cache.layer_step(li, k, q, v)
        x = x + attn.out_proj(out)
        gate, val = ff_in(x).chunk(2, dim=-1)
        x = x + ff_out(F.relu(gate) * val)
    return x


def _build_x(model, tok_idx, pos):
    from transformer_vm.model.transformer import add_position_encoding  # type: ignore
    x = model.tok.weight[tok_idx].clone()
    add_position_encoding(x, pos)
    return x


def _vcache_patch(model, cache, x_init, n_layers, primary_value, primary_slots):
    x_corrected = x_init.clone()
    for slot in primary_slots:
        x_corrected[slot] = x_corrected.new_tensor(primary_value)
    for li in range(n_layers):
        new_kqv = model.attn[li].in_proj_weight @ x_corrected
        new_k, _, new_v = new_kqv.chunk(3, dim=-1)
        if hasattr(cache, "_vals"):
            cache._vals[li][-1] = new_v.clone()
        elif hasattr(cache, "insert_v"):
            cache.insert_v(li, new_k, new_v)


class CompiledSolver:
    """A compiled CRAFT model, callable on arbitrary right-hand sides."""

    def __init__(self, model_path: str, netlist: str, dtype=torch.float64,
                 prune_frac: float = 0.0, prune_nonzero_only: bool = False,
                 quantize: str | None = None):
        from transformer_vm.model.weights import load_weights  # type: ignore
        with open(model_path + ".slots.json") as f:
            self.side = json.load(f)
        self.model, self.all_tokens, self.tok_to_idx = load_weights(model_path)
        self.model.eval()
        self.tokens, self.pc, self.layout = tokenize_lu_direct(netlist)
        self.n_free = int(self.side["n_free"])
        self.dtype = dtype
        self.prune_frac = prune_frac
        self.prune_nonzero_only = prune_nonzero_only
        self.quantize = quantize
        self._apply_perturbations()
        self.model = self.model.to(dtype)

    # -- weight-space perturbations (pruning / quantization) ----------------
    def _apply_perturbations(self):
        if self.prune_frac > 0:
            with torch.no_grad():
                allw = torch.cat([p.detach().abs().flatten()
                                  for p in self.model.parameters()])
                if self.prune_nonzero_only:
                    allw = allw[allw > 0]
                if allw.numel():
                    k = int(self.prune_frac * allw.numel())
                    thr = (torch.kthvalue(allw, max(k, 1)).values
                           if k > 0 else torch.tensor(-1.0, dtype=allw.dtype))
                    for p in self.model.parameters():
                        p.mul_((p.detach().abs() > thr).to(p.dtype))
        if self.quantize:
            with torch.no_grad():
                bits = int(self.quantize.split("int")[-1].split("_")[0])
                qmax = 2 ** (bits - 1) - 1
                if self.quantize.startswith("per_tensor"):
                    allw = torch.cat([p.detach().flatten()
                                      for p in self.model.parameters()])
                    s = allw.abs().max().item() / qmax
                    for p in self.model.parameters():
                        if s > 0:
                            p.copy_(torch.round(p / s).clamp(-qmax, qmax) * s)
                else:  # per_channel: per-parameter-tensor, per-output-row scale
                    for p in self.model.parameters():
                        pv = p.detach()
                        if pv.dim() >= 2:
                            s = pv.abs().amax(dim=tuple(range(1, pv.dim())),
                                              keepdim=True) / qmax
                        else:
                            s = pv.abs().max() / qmax
                        s = torch.where(s > 0, s, torch.ones_like(s))
                        p.copy_(torch.round(p / s).clamp(-qmax, qmax) * s)

    # -- the solve ----------------------------------------------------------
    def solve(self, b_ext=None, record=True):
        """Execute the compiled program. If b_ext is given, the runner supplies
        those right-hand-side entries at rhs_i instead of the model-computed
        source-driven b. Returns dict with x (solution), b_used, cache, pred_v.
        """
        s = self.side
        slot_x_value = int(s["slot_x_value"])
        slot_y_input = int(s["slot_y_input"])
        slot_b_value = int(s["slot_b_value"])
        slot_x_new = int(s["slot_x_new"])
        n_layers = len(self.model.attn)
        n_heads = self.model.attn[0].num_heads
        cache = RecordingCache(n_layers, n_heads, dtype=self.dtype, record=record)

        y_buf = [0.0] * max(self.n_free, 1)
        x_buf = [0.0] * max(self.n_free, 1)
        b_used = [0.0] * max(self.n_free, 1)
        pred_v = 0.0
        x = None
        pos = 0
        with torch.no_grad():
            for tok in self.tokens:
                if tok == PREDICTED:
                    logits = self.model.head(x)
                    best_name, best = "v_0", -math.inf
                    for name, idx in self.tok_to_idx.items():
                        if name.startswith("v_"):
                            sc = float(logits[idx])
                            if sc > best:
                                best, best_name = sc, name
                    pred_v = int(best_name.split("_")[1]) * V_STEP / SCALE
                    x = _forward_step(self.model, cache,
                                      _build_x(self.model, self.tok_to_idx[best_name], pos))
                    pos += 1
                    continue
                if tok not in self.tok_to_idx:
                    tok = "start"
                x_init = _build_x(self.model, self.tok_to_idx[tok], pos)
                if tok.startswith("bck_"):
                    i = int(tok.split("_", 1)[1])
                    x_init[slot_y_input] = x_init[slot_y_input] + x_init.new_tensor(y_buf[i])
                x = _forward_step(self.model, cache, x_init)

                if tok.startswith("rhs_"):
                    i = int(tok.split("_", 1)[1])
                    b_i = (float(b_ext[i]) if b_ext is not None
                           else float(x[slot_b_value]))
                    b_used[i] = b_i
                    _vcache_patch(self.model, cache, x_init, n_layers, b_i,
                                  [slot_x_value, slot_b_value, slot_x_new])
                elif tok.startswith("fwd_"):
                    i = int(tok.split("_", 1)[1])
                    y_i = float(x[slot_x_new])
                    y_buf[i] = y_i
                    _vcache_patch(self.model, cache, x_init, n_layers, y_i,
                                  [slot_x_value, slot_x_new])
                elif tok.startswith("bck_"):
                    i = int(tok.split("_", 1)[1])
                    x_i = float(x[slot_x_new])
                    x_buf[i] = x_i
                    _vcache_patch(self.model, cache, x_init, n_layers, x_i,
                                  [slot_x_value, slot_x_new])
                pos += 1
        return {
            "x": np.array(x_buf[: self.n_free], dtype=np.float64),
            "y": np.array(y_buf[: self.n_free], dtype=np.float64),
            "b_used": np.array(b_used[: self.n_free], dtype=np.float64),
            "pred_v": pred_v,
            "cache": cache,
        }


# ---------------------------------------------------------------------------
# Numerical-analysis helpers
# ---------------------------------------------------------------------------

def eta_rigal_gaches(A, x, b):
    """Normwise joint backward error (Rigal-Gaches 1967; Higham Thm 7.1).

        eta = ||r||_inf / (||A||_inf ||x||_inf + ||b||_inf),  r = b - A x

    This is the quantity the paper calls the per-column joint normwise backward
    error. It is CLASSICAL, not new; we report it as calibration.
    """
    A = np.asarray(A, float); x = np.asarray(x, float); b = np.asarray(b, float)
    if A.size == 0:
        return 0.0
    r = b - A @ x
    den = np.abs(A).sum(axis=1).max() * np.abs(x).max() + np.abs(b).max()
    if den == 0:
        return 0.0
    return float(np.abs(r).max() / den)


def growth_factor(A):
    """||L||inf ||U||inf / ||A||inf for the Doolittle factors actually used."""
    A = np.asarray(A, float)
    if A.size == 0:
        return float("nan")
    L, U = doolittle(A)
    na = np.abs(A).sum(axis=1).max()
    if na == 0:
        return float("nan")
    return float(np.abs(L).sum(axis=1).max() * np.abs(U).sum(axis=1).max() / na)


def reference_solve(A, b, dps=160):
    """High-precision reference solve (mpmath, 160 digits) -> float64 array."""
    from mpmath import mp, matrix, lu_solve
    if np.asarray(A).size == 0:
        return np.zeros(0)
    old = mp.dps
    mp.dps = dps
    try:
        Am = matrix([[mp.mpf(float(v)) for v in row] for row in np.asarray(A)])
        bm = matrix([mp.mpf(float(v)) for v in np.asarray(b)])
        xs = lu_solve(Am, bm)
        return np.array([float(v) for v in xs], dtype=np.float64)
    finally:
        mp.dps = old


def host_lu_solve(A, b):
    """The binary64 recurrence the compiler emits: Doolittle + fwd/back sub."""
    if np.asarray(A).size == 0:
        return np.zeros(0)
    L, U = doolittle(np.asarray(A, float))
    return back_sub(U, forward_sub(L, np.asarray(b, float)))


def circuit_system(netlist: str):
    """Return (A_FF, b_source, free, fixed, pc) for a netlist."""
    pc = parse_netlist(netlist)
    free, fixed, A_FF, A_FP = build_partition(pc)
    v_P = np.array([pc.fixed_voltage[fn] for fn in fixed], dtype=np.float64)
    b = -A_FP @ v_P if len(fixed) else np.zeros(len(free))
    return A_FF, b, free, fixed, pc


def build_model_for(cid: str, out_dir: str, circuits=None, force=False):
    """Build (or reuse) the compiled model for one circuit. Returns model path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"model_{cid}_lu_direct.bin")
    if os.path.exists(path) and os.path.exists(path + ".slots.json") and not force:
        return path
    os.environ.setdefault("MILP_TIME_LIMIT", "60")
    from build_lu_direct import build_for_circuit
    build_for_circuit(cid, out_path=path)
    return path
