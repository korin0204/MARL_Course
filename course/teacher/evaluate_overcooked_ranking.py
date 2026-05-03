"""教師用入口: Overcooked score ranking evaluation."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / "scripts" / "evaluate_coop_submissions.py"), run_name="__main__")
