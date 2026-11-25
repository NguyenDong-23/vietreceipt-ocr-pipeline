"""Integration-only bootstrap: bundled source, caller-owned Paddle runtime/cache."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent

def configure():
    source = str(ROOT / "source-goc")
    if source not in sys.path:
        sys.path.insert(0, source)
    return {}
