"""O(log n) hull-based KV cache wrapper using pybind11 convex hull extension."""

import os

import torch

_hull_ext = None


def _load_ext():
    global _hull_ext
    if _hull_ext is None:
        from torch.utils.cpp_extension import load

        root = os.path.dirname(os.path.abspath(__file__))
        _hull_ext = load(
            name="hull_ext",
            sources=[os.path.join(root, "hull_ext.cpp")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_include_paths=[root],
            verbose=False,
        )
    return _hull_ext


class HullKVCache:
    """O(log n) hard-attention KV cache using 2D convex hulls."""

    def __init__(self, n_layers, n_heads):
        ext = _load_ext()
        self._cache = ext.HullKVCache(n_layers, n_heads)
        self.n_layers = n_layers
        self._seq = -1
        # Records the seq used for the most recent layer_step on each layer.
        # Used by insert_v to find the entry to update.
        self._last_seq_per_layer = [-1] * n_layers

    def clear(self):
        """Reset all hull state and rewind the sequence counter."""
        self._cache.clear()
        self._seq = -1
        self._last_seq_per_layer = [-1] * self.n_layers

    def set_tiebreak(self, layer, head, latest):
        """Set tiebreak mode for a head: True for latest, False for average."""
        self._cache.set_tiebreak(layer, head, 1 if latest else 0)

    def layer_step(self, layer, keys, queries, values):
        """Insert KV pair and query all heads for one layer, return attention output."""
        self._seq += 1
        self._last_seq_per_layer[layer] = self._seq
        out_np = self._cache.layer_step(
            layer,
            keys.reshape(-1, 2).numpy(),
            queries.reshape(-1, 2).numpy(),
            values.reshape(-1, 2).numpy(),
            self._seq,
        )
        return torch.from_numpy(out_np).flatten()

    def update_v_at(self, layer, seq, values):
        """Replace V at the entry inserted with this seq for every head in
        this layer.  Same K, no append — the metadata aggregates are rewritten
        in-place inside the C++ extension.
        """
        self._cache.update_v_at_seq(
            int(layer),
            int(seq),
            values.reshape(-1, 2).numpy(),
        )

    def insert_v(self, layer, keys, values):
        """V-cache patch: replace the value of the most recently inserted entry
        for this layer.  `keys` is accepted for backwards compatibility with
        callers that pass new K alongside V; it is not needed because the
        entry is identified by its seq.
        """
        del keys  # unused — entry is identified by recorded seq
        seq = self._last_seq_per_layer[layer]
        if seq < 0:
            return  # no entry yet, nothing to patch
        self.update_v_at(layer, seq, values)
