"""Root adapter package exposing the final/ core engine."""
from __future__ import annotations
import sys
from pathlib import Path

# Add final to sys.path so final.src is accessible
_FINAL_PATH = Path(__file__).resolve().parent.parent / "final"
if str(_FINAL_PATH) not in sys.path:
    sys.path.insert(0, str(_FINAL_PATH))
