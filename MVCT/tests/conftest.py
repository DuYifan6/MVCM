"""Make the flat research modules under ``src`` visible to test subprocesses."""

import os
import sys
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
source = str(SOURCE_DIR)

if source not in sys.path:
    sys.path.insert(0, source)

existing = os.environ.get("PYTHONPATH")
paths = existing.split(os.pathsep) if existing else []
if source not in paths:
    os.environ["PYTHONPATH"] = os.pathsep.join([source, *paths])
