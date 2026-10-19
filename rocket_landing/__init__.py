"""Learn to land a rocket, then investigate what its reward teaches it."""

import os
from pathlib import Path

# Keep Matplotlib's writable font cache inside this project in restricted hosts.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"))

__version__ = "2.0.0"
