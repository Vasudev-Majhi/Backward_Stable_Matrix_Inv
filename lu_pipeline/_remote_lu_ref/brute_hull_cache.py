"""Python brute-force hard-max KV cache — same interface as HullKVCache."""
from __future__ import annotations
import torch
import numpy as np


class BruteHullCache:
    """O(n^2) exact hard-max attention. Same interface as HullKVCache."""

    def __init__(self, n_layers, n_heads):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self._entries = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self._tiebreak = [[False] * n_heads for _ in range(n_layers)]
        self._seq = -1
        self._last_seq_per_layer = [-1] * n_layers

    def clear(self):
        self._entries = [[[] for _ in range(self.n_heads)] for _ in range(self.n_layers)]
        self._seq = -1
        self._last_seq_per_layer = [-1] * self.n_layers

    def set_tiebreak(self, layer, head, latest):
        self._tiebreak[layer][head] = bool(latest)

    def layer_step(self, layer, keys, queries, values):
        self._seq += 1
        self._last_seq_per_layer[layer] = self._seq
        k_np = keys.reshape(-1, 2).detach().numpy()    # (n_heads, 2)
        q_np = queries.reshape(-1, 2).detach().numpy()  # (n_heads, 2)
        v_np = values.reshape(-1, 2).detach().numpy()   # (n_heads, 2)

        out = np.zeros((self.n_heads, 2), dtype=np.float64)

        for h in range(self.n_heads):
            # Insert current KV
            self._entries[layer][h].append((k_np[h], v_np[h], self._seq))

            # Brute-force argmax
            entries = self._entries[layer][h]
            best_score = -1e300
            best_seq = -1
            vsum = np.zeros(2)
            vcount = 0

            for kk, vv, ss in entries:
                score = float(q_np[h, 0] * kk[0] + q_np[h, 1] * kk[1])
                if score > best_score + 1e-12:
                    best_score = score
                    if self._tiebreak[layer][h]:
                        best_seq = ss
                        out[h] = vv.copy()
                    else:
                        vsum = vv.copy()
                        vcount = 1
                elif abs(score - best_score) <= 1e-12:
                    if self._tiebreak[layer][h]:
                        if ss > best_seq:
                            best_seq = ss
                            out[h] = vv.copy()
                    else:
                        vsum += vv
                        vcount += 1

            if not self._tiebreak[layer][h] and vcount > 0:
                out[h] = vsum / vcount

        return torch.from_numpy(out.flatten()).to(keys.dtype)

    def update_v_at(self, layer, seq, values):
        """Replace V for the entry inserted at this seq (every head)."""
        v_np = values.reshape(-1, 2).detach().numpy()
        for h in range(self.n_heads):
            for idx, (kk, vv, ss) in enumerate(self._entries[layer][h]):
                if ss == seq:
                    self._entries[layer][h][idx] = (kk, v_np[h].copy(), ss)
                    break

    def insert_v(self, layer, keys, values):
        """V-cache patch: replace the V of the most recently inserted entry."""
        del keys
        seq = self._last_seq_per_layer[layer]
        if seq < 0:
            return
        self.update_v_at(layer, seq, values)
