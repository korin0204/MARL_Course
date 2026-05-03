"""Tabular Q-learning components for Bomber student baseline."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from marl_course.common.grid import DIRS, Position, add_pos, manhattan


def bomber_features(obs: dict[str, Any]) -> str:
    """Convert rich Bomber observation into a compact discrete feature key.

    This key defines the tabular state for Q-learning.
    """
    stats = obs["stats"]
    self_id = obs["self_id"]
    agent = f"agent_{self_id}"
    pos = tuple(stats[agent]["position"])
    grid = obs["grid"]
    danger = _positions(grid, "danger")
    wood = _positions(grid, "wood")
    power = _positions(grid, "bomb_up") | _positions(grid, "fire_up")
    enemies = [
        tuple(data["position"])
        for other, data in stats.items()
        if other != agent and data["alive"]
    ]
    nearest_enemy = min((manhattan(pos, enemy) for enemy in enemies), default=9)
    nearest_power = min((manhattan(pos, item) for item in power), default=9)
    adjacent_wood = any(add_pos(pos, d) in wood for d in DIRS.values())
    safe_moves = 0
    for action, direction in DIRS.items():
        nxt = add_pos(pos, direction)
        if obs["action_mask"][action] and nxt not in danger:
            safe_moves += 1
    return "|".join(
        [
            f"danger={int(pos in danger)}",
            f"safe={min(safe_moves, 3)}",
            f"wood={int(adjacent_wood)}",
            f"enemy={min(nearest_enemy // 2, 4)}",
            f"power={min(nearest_power // 2, 4)}",
            f"bomb={int(bool(obs['action_mask'][5]))}",
        ]
    )


def _positions(grid: dict[str, list[list[float]]], key: str) -> set[Position]:
    """Collect active cell positions from one grid plane."""
    plane = grid[key]
    return {(r, c) for r, row in enumerate(plane) for c, value in enumerate(row) if value > 0}


@dataclass
class BomberQLearningPolicy:
    """Simple epsilon-greedy tabular Q-learning policy."""

    q_table: dict[str, list[float]] = field(default_factory=dict)
    n_actions: int = 6
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def q_values(self, obs: dict[str, Any]) -> list[float]:
        """Return mutable Q-value vector for the current feature state."""
        key = bomber_features(obs)
        if key not in self.q_table:
            self.q_table[key] = [0.0 for _ in range(self.n_actions)]
        return self.q_table[key]

    def act(self, obs: dict[str, Any], action_mask: list[int] | None = None, deterministic: bool = True, epsilon: float = 0.0) -> int:
        """Choose action with legal-mask-aware epsilon-greedy policy."""
        mask = action_mask or obs.get("action_mask", [1] * self.n_actions)
        legal = [idx for idx, ok in enumerate(mask) if ok]
        if not legal:
            return 0
        if not deterministic and self.rng.random() < epsilon:
            return self.rng.choice(legal)
        q_values = self.q_values(obs)
        return max(legal, key=lambda idx: q_values[idx])

    def update(self, obs: dict[str, Any], action: int, reward: float, next_obs: dict[str, Any], done: bool, alpha: float, gamma: float) -> float:
        """Apply one-step TD update and return absolute TD error."""
        q_values = self.q_values(obs)
        old = q_values[action]
        next_q = 0.0 if done else max(self.q_values(next_obs))
        target = reward + gamma * next_q
        q_values[action] += alpha * (target - old)
        return abs(target - old)

    def save(self, path: Path) -> None:
        """Save Q-table policy as JSON for easy inspection in class."""
        path.write_text(json.dumps({"q_table": self.q_table, "n_actions": self.n_actions, "seed": self.seed}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BomberQLearningPolicy":
        """Load Q-table policy from JSON checkpoint."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(q_table={k: list(v) for k, v in payload["q_table"].items()}, n_actions=payload.get("n_actions", 6), seed=payload.get("seed", 0))
