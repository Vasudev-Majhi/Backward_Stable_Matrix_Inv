import sys
from pathlib import Path

_TRANSFORMER_VM = Path(__file__).resolve().parent.parent / "transformer-vm"
if str(_TRANSFORMER_VM) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMER_VM))
