"""学生用入口: Bomber Actor-Critic vs baseline opponents."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / "scripts" / "train_bomber_actor_critic.py"), run_name="__main__")
