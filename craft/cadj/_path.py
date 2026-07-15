"""Sys.path shim for the cadj/ folder.

Adds craft/ (parent) so we can import _bootstrap, parse, interpreter,
experiments.spectrum, etc., AND adds craft/cadj/ so siblings can use
plain `from cadj_reference import ...` style without a package install.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PARENT = _os.path.dirname(_HERE)

for _p in (_PARENT, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import _bootstrap  # noqa: F401 — pulls transformer-vm into sys.path

# Fallback if the shared _bootstrap path is stale (transformer-vm moved).
def _ensure_transformer_vm() -> None:
    try:
        import transformer_vm  # noqa: F401
        return
    except ImportError:
        pass
    _CANDIDATES = [
        _os.path.join(_os.path.dirname(_PARENT), "transformer-vm"),
        _os.path.join(_os.path.dirname(_PARENT), "nohull_nouniversal_jacob", "transformer-vm"),
    ]
    for cand in _CANDIDATES:
        if _os.path.isdir(_os.path.join(cand, "transformer_vm")):
            if cand not in _sys.path:
                _sys.path.insert(0, cand)
            return

_ensure_transformer_vm()
del _ensure_transformer_vm
