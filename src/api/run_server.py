"""Launch the NBA Spread Prediction API with uvicorn."""

import os
import sys
from pathlib import Path

# Ensure the project root is on PYTHONPATH before uvicorn spawns its reload
# subprocess.  sys.path manipulations are process-local, but os.environ changes
# are inherited by child processes, so the reload worker can import 'src.*'.
_PROJECT_ROOT = str(Path(__file__).parents[2])
os.environ.setdefault("PYTHONPATH", _PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import uvicorn  # noqa: E402 — must come after sys.path is set

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
