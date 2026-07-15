"""sys.path shim for the lu_pipeline/ folder.

Adds:
  - lu_pipeline/                        (siblings: lu_factor, lu_direct_reference, ...)
  - craft/                   (parse, _bootstrap, runner, build)
  - craft/cadj/              (direct_reference, cadj_reference, direct_*)
  - transformer-vm path (via _bootstrap fallback)
"""
from __future__ import annotations

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PROJECT = _os.path.dirname(_HERE)
_CLAUDE_FILES = _os.path.join(_PROJECT, "craft")
_CADJ = _os.path.join(_CLAUDE_FILES, "cadj")

for _p in (_HERE, _CLAUDE_FILES, _CADJ):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

# Pull transformer_vm into sys.path. craft/_bootstrap probes for
# transformer-vm under the project root.
import _bootstrap  # noqa: F401, E402
