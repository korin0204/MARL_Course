"""Tiny terminal renderer helper for quick live debugging."""
from __future__ import annotations

import time
from typing import Any


def print_live(env: Any, sleep: float = 0.05) -> None:
    """Clear terminal, print current frame, and wait briefly."""
    print("\033[2J\033[H" + env.render())
    time.sleep(sleep)
