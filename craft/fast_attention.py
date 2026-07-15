"""Fast O(1) attention specialized for Jacobi DSL fetches.

All of the circuit-transformer's fetches share the structure:
    fetch(value=V, query=position + off, key=position,
          clear_key=1 - <kind_flag>, tie_break="latest")

So for a fetch whose key is `position`, the score surface
    score(k, q) = -(k - q)² + clear_penalty(k)
has a unique maximum at k = q (among non-cleared positions). With integer
positions and integer queries, this is just a dict lookup.

Drop-in replacement for BruteAttention when key_exprs_2d encodes `key=position`.
For lookups with other key structures, falls back to brute-force.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
from transformer_vm.graph import core as _graph
from transformer_vm.graph.core import Expression


class FastPositionAttention:
    """O(1) attention cache for lookups whose 1-D key equals `position`."""

    def __init__(self, lookup):
        self.lookup = lookup
        # {position_int: (cleared_bool, [val0, val1, ...])}
        self.table: dict[int, tuple[bool, list[float]]] = {}
        self.entries: list[tuple[int, float, float, list[float]]] = []  # for fallback
        self.fast_mode = self._can_use_fast(lookup)

    @staticmethod
    def _can_use_fast(lookup) -> bool:
        """Return True if key_exprs_2d matches the shape produced by
        _to_2d_key when the 1D key is `position` (our integer sequential key).

        We just check that the 1D key implied by key_exprs_2d[0] / [1] is
        the `position` InputDim. Relies on KEY_OFFSET==0 and the standard
        quadratic encoding.
        """
        try:
            kx = lookup.key_exprs_2d[0]
            ky = lookup.key_exprs_2d[1]
            # kx should be 2 * position (maybe + small LATEST_ALPHA on ky).
            # Check kx has single term with coefficient 2 on `position`.
            pos_dim = _graph.position
            if not isinstance(kx, Expression):
                return False
            if pos_dim not in kx.terms:
                return False
            if abs(kx.terms[pos_dim] - 2.0) > 1e-9:
                return False
            # ky should have -position_sq term (coefficient -1), plus
            # clear_key penalty and LATEST_ALPHA·inv_log_pos.
            psq = _graph.position_sq
            if not isinstance(ky, Expression):
                return False
            if psq not in ky.terms:
                return False
            if abs(ky.terms[psq] - (-1.0)) > 1e-9:
                return False
            return True
        except Exception:
            return False

    def clear(self):
        self.table.clear()
        self.entries.clear()

    def insert_and_query(self, vals, seq):
        lu = self.lookup
        if self.fast_mode:
            return self._fast_insert_query(vals, seq)
        return self._brute_insert_query(vals, seq)

    # ------------------------------------------------------------------
    def _compute_clear_penalty(self, vals) -> float:
        """Approximate clear_key contribution to ky. Sign is what matters:
        a large negative value means "cleared"."""
        ky_expr = self.lookup.key_exprs_2d[1]
        # In ky_expr, the clear-key contribution appears as `-BIG * clear_val`.
        # Evaluate the whole ky at current vals and subtract the position_sq
        # portion; what's left is the clear penalty + inv_log_pos terms.
        pos = vals.get(_graph.position, 0.0)
        psq_val = pos * pos
        expected_pos_part = -1.0 * psq_val
        actual = ky_expr.evaluate(vals)
        residual = actual - expected_pos_part
        return residual  # includes LATEST_ALPHA*inv_log_pos, negligible

    def _fast_insert_query(self, vals, seq):
        lu = self.lookup
        # Evaluate value exprs at this token and record.
        raw_vals = [v.evaluate(vals) for v in lu.value_exprs]
        # Determine if this token is "cleared" (clear_key==1, very negative ky).
        # A simple test: evaluate ky, check magnitude.
        ky_val = lu.key_exprs_2d[1].evaluate(vals)
        # If ky is strongly negative (|ky| >> pos²), token is cleared.
        pos = vals.get(_graph.position, float(seq))
        expected_uncleared = -pos * pos
        cleared = ky_val < expected_uncleared - 1e15  # BIG threshold

        self.table[int(seq)] = (cleared, raw_vals)

        # Query side: target position = query_val.
        qx_expr = lu.query_exprs_2d[0]
        qx = qx_expr.evaluate(vals)
        # qx = query_val (with KEY_OFFSET=0). For "key=position" encoding:
        # qx = 2*q*k equals... actually we have query_exprs_2d[0] = q
        # (since _to_2d_query returns [q - KEY_OFFSET, 1] with KEY_OFFSET=0).
        # Score peaks at k == qx. So target_pos = int(round(qx)).
        target_pos = int(round(qx))

        if target_pos in self.table and not self.table[target_pos][0]:
            return list(self.table[target_pos][1])
        # No match → for "latest" tie_break with no legal match, fall back to
        # all-zero values (consistent with how clear_key isolates).
        return [0.0 for _ in lu.value_exprs]

    # ------------------------------------------------------------------
    def _brute_insert_query(self, vals, seq):
        """Fallback: generic brute-force attention."""
        lu = self.lookup
        kx = lu.key_exprs_2d[0].evaluate(vals)
        ky = lu.key_exprs_2d[1].evaluate(vals)
        raw = [v.evaluate(vals) for v in lu.value_exprs]
        self.entries.append((seq, kx, ky, raw))

        qx = lu.query_exprs_2d[0].evaluate(vals)
        qy = lu.query_exprs_2d[1].evaluate(vals)

        best_score = -1e300
        for _s, ekx, eky, _ev in self.entries:
            score = qx * ekx + qy * eky
            if score > best_score + 1e-9:
                best_score = score

        if lu.tie_break == "average":
            total = [0.0] * len(lu.value_exprs)
            count = 0
            for _s, ekx, eky, ev in self.entries:
                score = qx * ekx + qy * eky
                if abs(score - best_score) <= 1e-9:
                    for j in range(len(ev)):
                        total[j] += ev[j]
                    count += 1
            return [t / count for t in total] if count > 0 else total
        else:
            best_seq = -1
            best_vals = None
            for s, _ekx, _eky, ev in self.entries:
                score = qx * _ekx + qy * _eky
                if abs(score - best_score) <= 1e-9 and s > best_seq:
                    best_seq = s
                    best_vals = ev
            return list(best_vals) if best_vals is not None else [0.0] * len(lu.value_exprs)


def make_fast_runtime(program_graph):
    """Build a Runtime that uses FastPositionAttention where possible."""
    from transformer_vm.evaluator import Runtime
    rt = Runtime(use_hull=False, program_graph=program_graph)
    # Replace each lookup's attention cache with FastPositionAttention.
    for lu in rt.all_lookups:
        rt.attention[lu.id] = FastPositionAttention(lu)
    return rt
