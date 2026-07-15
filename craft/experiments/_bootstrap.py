"""Import shim for scripts under craft/experiments/.

Adds both `transformer-vm/` (for `transformer_vm.*` imports) and the parent
`craft/` folder (for `parse`, `interpreter`, `tokenize_netlist`,
`fast_attention`, `jacobi_reference`, `build`, `runner` imports) to sys.path.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CLAUDE_FILES = _HERE.parent
_TRANSFORMER_VM = _CLAUDE_FILES.parent / "transformer-vm"

for p in (_TRANSFORMER_VM, _CLAUDE_FILES):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
