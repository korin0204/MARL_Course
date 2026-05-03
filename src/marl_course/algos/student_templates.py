"""Student-facing policy templates used in onboarding and smoke tests."""
from __future__ import annotations

import math
import random
from typing import Any

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


def torch_available() -> bool:
    """Return True when PyTorch is installed in the active environment."""
    return torch is not None and nn is not None


class LinearPolicy:
    """Dependency-free linear baseline for Colab explanations and smoke tests."""

    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = n_actions
        self.rng = random.Random(seed)
        self.bias = [self.rng.uniform(-0.01, 0.01) for _ in range(n_actions)]

    def act(self, obs: dict[str, Any], action_mask: list[int] | None = None, deterministic: bool = True) -> int:
        """Select legal action via deterministic argmax or stochastic softmax."""
        mask = action_mask or obs.get("action_mask", [1] * self.n_actions)
        legal = [idx for idx, ok in enumerate(mask) if ok]
        if not legal:
            return 0
        if deterministic:
            return max(legal, key=lambda idx: self.bias[idx])
        weights = [math.exp(self.bias[idx]) for idx in legal]
        total = sum(weights)
        pick = self.rng.random() * total
        acc = 0.0
        for idx, weight in zip(legal, weights):
            acc += weight
            if acc >= pick:
                return idx
        return legal[-1]


if torch_available():

    class TorchMLPPolicy(nn.Module):  # type: ignore[misc]
        """Small MLP template students can extend for function approximation."""

        def __init__(self, input_dim: int, n_actions: int, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x)

else:

    class TorchMLPPolicy:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise RuntimeError("torch is not installed. Run .venv/bin/python -m pip install -e '.[dev]'")
