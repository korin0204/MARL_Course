"""学生用入口: Overcooked MAPPO training."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / "scripts" / "train_coop_mappo.py"), run_name="__main__")
