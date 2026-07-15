"""sys.path shim for the inversion2/ working tree.

Adds:
  - ../transformer-vm  -> for `import transformer_vm.*`
  - ../craft     -> for shared helpers

We APPEND (not prepend) so the importing script's own directory stays
ahead in sys.path.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

for sub in ("transformer-vm", "craft"):
    p = _ROOT / sub
    if p.exists() and str(p) not in sys.path:
        sys.path.append(str(p))
