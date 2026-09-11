"""Compiler ground truth, independent of any execution trace.

Fixes two defects in the earlier analysis:

W3  The one-hotness proxy pooled LOOKUP heads with PASSTHROUGH heads. Passthrough
    heads use the erase query and legitimately attend broadly, so pooling them
    made `frac_onehot` top out at 0.22 even on provably correct runs. We now read
    the compiler's own per-head labels (`head_map` -> allocation.yaml) and score
    proxies on lookup heads only.

W4  Correctness was defined as "selects what the float64 run selected", which is
    circular for the precision arm: any dtype change is broken by construction.
    We now define the intended target position from the PROGRAM:

      exact_argmax(p) = argmax_p <k_p, q_t>   evaluated in exact rational
                        arithmetic over the unperturbed binary64 weights.

    In practice we certify the float64 argmax instead of computing it exactly:
    if the winning margin exceeds a conservative bound on the accumulated
    rounding error, the float64 argmax is provably the exact argmax. Only when
    the certificate fails do we fall back to exact Fraction arithmetic. We report
    the certified fraction, which is itself the selector-exactness claim.

The oracle is a property of the compiled program, not of a run, so a perturbed
configuration is scored against what the program MEANS, not against what one
particular execution did.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from craft_harness import (  # noqa: E402
    CompiledSolver, _build_x, _forward_step, _vcache_patch, PREDICTED, V_STEP, SCALE,
)


# ---------------------------------------------------------------------------
# Compiler head labels
# ---------------------------------------------------------------------------

def head_labels_for(model_path):
    """Return {(layer, head): 'lookup'|'passthrough'} from the compiler's own
    allocation dump, written next to the model at build time."""
    p = model_path + ".alloc.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        doc = json.load(f)
    out = {}
    for layer in doc.get("layers", []):
        li = int(layer["layer"])
        for h in layer.get("attention_heads", []):
            out[(li, int(h["head"]))] = h.get("type", "unknown")
    return out


def capture_allocation(model_path, cwd_alloc="allocation.yaml"):
    """Convert the allocation.yaml the builder just wrote into a JSON sidecar
    pinned to this model. Called immediately after a build."""
    if not os.path.exists(cwd_alloc):
        return False
    try:
        import yaml
        with open(cwd_alloc) as f:
            doc = yaml.safe_load(f)
    except Exception:
        return False
    with open(model_path + ".alloc.json", "w") as f:
        json.dump(doc, f)
    return True


# ---------------------------------------------------------------------------
# Certified-exact selector oracle
# ---------------------------------------------------------------------------

class OracleCache:
    """Records, per attention lookup, the certified-exact argmax position.

    The certificate: with scores s_i computed in binary64 from 2-term dot
    products, each |s_i - fl(s_i)| <= 3u|s_i| (two products and one add, plus
    slack). If s_(1) - s_(2) > 3u(|s_(1)| + |s_(2)|) then the computed argmax is
    the exact argmax. Otherwise we recompute that row in exact rational
    arithmetic.
    """

    N_TERMS_SLACK = 8.0  # generous: 2 mults + 1 add is 3u; we allow 8u

    def __init__(self, n_layers, n_heads, labels=None, exact_only_for="lookup"):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self._keys = [[] for _ in range(n_layers)]
        self._vals = [[] for _ in range(n_layers)]
        self.records = []   # (layer, head, step, argmax, certified, exact_used)
        self._step = [0] * n_layers
        # Passthrough heads read the most recent position via the erase query and
        # routinely produce exact ties, for which "the argmax" is resolved by the
        # head's tie-break rule rather than by arithmetic. Certifying them is
        # meaningless and the exact fallback is expensive, so we compute the
        # oracle only where the compiler says a real lookup happens.
        self.labels = labels
        self.exact_only_for = exact_only_for

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

        s = scores.double().numpy()
        u = np.finfo(np.float64).eps / 2
        step = self._step[layer]
        for h in range(self.n_heads):
            col = s[:, h]
            am = int(np.argmax(col))
            top = col[am]
            if col.size > 1:
                rest = np.delete(col, am)
                second = float(np.max(rest))
                arg2 = int(np.argmax(rest))
            else:
                second, arg2 = -np.inf, am
            gap = top - second
            bound = self.N_TERMS_SLACK * u * (abs(top) + abs(second))
            certified = bool(np.isfinite(gap) and gap > bound)
            exact_used = False
            is_target = (self.labels is None
                         or self.labels.get((layer, h)) == self.exact_only_for)
            if not certified and is_target:
                am = self._exact_argmax(K[:, h, :], Q[h, :])
                exact_used = True
            self.records.append((layer, h, step, am, certified, exact_used))
        self._step[layer] += 1
        return out.flatten()

    @staticmethod
    def _exact_argmax(Kh, qh):
        q = [Fraction(float(v)) for v in qh]
        best_i, best_v = 0, None
        for i in range(Kh.shape[0]):
            v = sum(Fraction(float(Kh[i, j])) * q[j] for j in range(len(q)))
            if best_v is None or v > best_v:
                best_v, best_i = v, i
        return best_i


def build_oracle(solver: CompiledSolver, b_ext=None, labels=None):
    """Run the UNPERTURBED float64 program and return the certified target
    position for every attention lookup, in execution order."""
    s = solver.side
    slot_x_value = int(s["slot_x_value"]); slot_y_input = int(s["slot_y_input"])
    slot_b_value = int(s["slot_b_value"]); slot_x_new = int(s["slot_x_new"])
    n_layers = len(solver.model.attn)
    n_heads = solver.model.attn[0].num_heads
    cache = OracleCache(n_layers, n_heads, labels=labels)
    y_buf = [0.0] * max(solver.n_free, 1)
    pos = 0
    x = None
    with torch.no_grad():
        for tok in solver.tokens:
            if tok == PREDICTED:
                logits = solver.model.head(x)
                best_name, best = "v_0", -float("inf")
                for name, idx in solver.tok_to_idx.items():
                    if name.startswith("v_"):
                        sc = float(logits[idx])
                        if sc > best:
                            best, best_name = sc, name
                x = _forward_step(solver.model, cache,
                                  _build_x(solver.model, solver.tok_to_idx[best_name], pos))
                pos += 1
                continue
            if tok not in solver.tok_to_idx:
                tok = "start"
            x_init = _build_x(solver.model, solver.tok_to_idx[tok], pos)
            if tok.startswith("bck_"):
                i = int(tok.split("_", 1)[1])
                x_init[slot_y_input] = x_init[slot_y_input] + x_init.new_tensor(y_buf[i])
            x = _forward_step(solver.model, cache, x_init)
            if tok.startswith("rhs_"):
                i = int(tok.split("_", 1)[1])
                b_i = float(b_ext[i]) if b_ext is not None else float(x[slot_b_value])
                _vcache_patch(solver.model, cache, x_init, n_layers, b_i,
                              [slot_x_value, slot_b_value, slot_x_new])
            elif tok.startswith("fwd_"):
                i = int(tok.split("_", 1)[1])
                y_buf[i] = float(x[slot_x_new])
                _vcache_patch(solver.model, cache, x_init, n_layers, y_buf[i],
                              [slot_x_value, slot_x_new])
            elif tok.startswith("bck_"):
                _vcache_patch(solver.model, cache, x_init, n_layers,
                              float(x[slot_x_new]), [slot_x_value, slot_x_new])
            pos += 1
    recs = cache.records
    lk = [r for r in recs
          if labels is None or labels.get((r[0], r[1])) == "lookup"]
    return {
        "frac_certified_lookup": (float(np.mean([r[4] for r in lk])) if lk else float("nan")),
        "n_lookup_heads_scored": len(lk),
        "targets": [r[3] for r in recs],
        "layer_head": [(r[0], r[1]) for r in recs],
        "certified": [r[4] for r in recs],
        "exact_used": [r[5] for r in recs],
        "n_lookups": len(recs),
        "frac_certified": float(np.mean([r[4] for r in recs])) if recs else float("nan"),
        "n_exact_fallback": int(sum(r[5] for r in recs)),
    }


def score_against_oracle(cache_records_sel, layer_head, oracle, labels,
                         restrict_to="lookup"):
    """Hit rate of a run's selections against the program oracle.

    restrict_to='lookup' scores only compiler-labelled lookup heads (W3 fix).
    """
    tgt = oracle["targets"]
    n = min(len(cache_records_sel), len(tgt))
    idx = range(n)
    if restrict_to and labels:
        idx = [i for i in range(n)
               if labels.get(layer_head[i], "unknown") == restrict_to]
    if not len(list(idx)):
        return float("nan"), 0
    idx = [i for i in range(n)
           if (not restrict_to or not labels
               or labels.get(layer_head[i], "unknown") == restrict_to)]
    hits = sum(1 for i in idx if cache_records_sel[i] == tgt[i])
    return hits / len(idx), len(idx)
