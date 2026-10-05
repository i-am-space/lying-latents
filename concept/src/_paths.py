"""Put <repo>/src and <repo>/concept/src on sys.path. Nothing else."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "concept" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
