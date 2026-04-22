"""
Root conftest.py — configures sys.path for pytest.

The Atwater project uses two import root conventions simultaneously:
- ``atwater.src.knowledge.*`` — requires the PARENT of this directory on sys.path
- ``src.memory.*``, ``src.optimization.*`` — requires THIS directory on sys.path

This conftest adds both paths so all imports resolve correctly when running:
    pytest tests/
from within the atwater/ directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This file lives at atwater/conftest.py
_HERE = Path(__file__).parent               # .../atwater/
_WORKSPACE = _HERE.parent                   # .../joshua_augustine_workspace/

# Add atwater/ so `src.memory`, `src.optimization`, `src.orchestrator` resolve
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Add workspace/ so `atwater.src.knowledge.*` resolves
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
