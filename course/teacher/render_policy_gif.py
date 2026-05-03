"""教師/学生共通入口: saved model rollout GIF rendering."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / "scripts" / "render_policy_gif.py"), run_name="__main__")
