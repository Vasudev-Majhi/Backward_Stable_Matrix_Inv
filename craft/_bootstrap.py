"""Import shim for standalone scripts under craft/.

Every script imports this first so `transformer_vm.*` resolves without
needing to pip-install the sibling transformer-vm package.
"""
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
_CANDIDATES = [
    _PROJECT / "transformer-vm",
    _PROJECT / "nohull_nouniversal_jacob" / "transformer-vm",
]
for _cand in _CANDIDATES:
    if (_cand / "transformer_vm").is_dir():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break
