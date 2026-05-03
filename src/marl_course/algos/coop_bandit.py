"""Lightweight team-strategy learner for cooperative kitchen tasks.

Instead of directly learning low-level actions, this policy learns which
high-level role assignment strategy works best for the current training setup.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from marl_course.envs.coop_kitchen import CoopGreedyTeamPolicy


STRATEGIES = {
    "balanced": {"onion_workers": (0, 2), "dish_workers": (1, 3), "cook_dish_workers": (1, 3)},
    "single_chef": {"onion_workers": (0,), "dish_workers": (3,), "cook_dish_workers": (3,)},
    "fast_onions": {"onion_workers": (0, 1, 2), "dish_workers": (3,), "cook_dish_workers": (3,)},
    "fast_dishes": {"onion_workers": (0,), "dish_workers": (1, 2, 3), "cook_dish_workers": (1, 2, 3)},
}


@dataclass
class CoopStrategyBanditPolicy:
    """Epsilon-greedy bandit over predefined strategy templates."""

    values: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in STRATEGIES})
    counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in STRATEGIES})
    seed: int = 0
    current_strategy: str = "balanced"

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self._delegate = self._make_delegate(self.current_strategy)

    def reset_episode(self, layout_metadata: dict[str, Any] | None = None, epsilon: float = 0.0) -> None:
        """Pick strategy for this episode and reset delegated team policy."""
        self.current_strategy = self.select_strategy(epsilon=epsilon)
        self._delegate = self._make_delegate(self.current_strategy)
        self._delegate.reset_episode(layout_metadata)

    def select_strategy(self, epsilon: float = 0.0) -> str:
        """Choose strategy via epsilon-greedy over learned strategy values."""
        if self.rng.random() < epsilon:
            return self.rng.choice(list(STRATEGIES))
        return max(STRATEGIES, key=lambda name: self.values.get(name, 0.0))

    def act(self, obs: dict[str, Any], action_mask: list[list[int]] | None = None, deterministic: bool = True) -> list[int]:
        return self._delegate.act(obs, action_mask=action_mask, deterministic=deterministic)

    def update(self, reward: float, alpha: float = 0.2) -> float:
        """Incrementally update chosen strategy value from episode return."""
        old = self.values.get(self.current_strategy, 0.0)
        self.counts[self.current_strategy] = self.counts.get(self.current_strategy, 0) + 1
        self.values[self.current_strategy] = old + alpha * (reward - old)
        return math.fabs(reward - old)

    def _make_delegate(self, strategy: str) -> CoopGreedyTeamPolicy:
        """Instantiate concrete low-level team policy for one strategy name."""
        kwargs = STRATEGIES[strategy]
        return CoopGreedyTeamPolicy(**kwargs)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "values": self.values,
                    "counts": self.counts,
                    "seed": self.seed,
                    "current_strategy": max(STRATEGIES, key=lambda name: self.values.get(name, 0.0)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "CoopStrategyBanditPolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            values={k: float(v) for k, v in payload.get("values", {}).items()} or {name: 0.0 for name in STRATEGIES},
            counts={k: int(v) for k, v in payload.get("counts", {}).items()} or {name: 0 for name in STRATEGIES},
            seed=int(payload.get("seed", 0)),
            current_strategy=payload.get("current_strategy", "balanced"),
        )
