"""Small scheduling helpers for training hyperparameters."""
from __future__ import annotations

import math


def exponential_decay(start: float, end: float, step: int, total_steps: int, decay_rate: float = 8.0) -> float:
    """Exponentially decay a scalar from `start` toward `end`.

    `decay_rate` controls how quickly the value approaches `end`.
    Larger values reduce exploration earlier. For example, with rate=8.0 the
    value is already close to `end` by the latter half of training.
    """

    if total_steps <= 0:
        return end
    progress = max(0.0, min(1.0, step / max(1, total_steps)))
    return end + (start - end) * math.exp(-decay_rate * progress)
