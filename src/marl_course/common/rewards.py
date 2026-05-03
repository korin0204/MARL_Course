"""Reward/event abstractions for student-customizable training rewards."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .api import AgentID


@dataclass(slots=True)
class Event:
    """Game event emitted by environments for reward shaping.

    Example events: "winner", "block_destroyed", "delivery".
    Students can map these events to custom dense rewards during training.
    """

    name: str
    actor: AgentID | None = None
    target: AgentID | None = None
    value: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


class RewardFn:
    """Student-facing reward hook.

    Environments emit game events and reference scores. Training code can plug
    in arbitrary reward functions without changing teacher evaluation.
    """

    def reset(self, env: Any, initial_obs: dict[AgentID, Any]) -> None:
        return None

    def __call__(
        self,
        transition: Any,
        events: list[Event],
        info: dict[str, Any],
    ) -> dict[AgentID, float]:
        raise NotImplementedError


class ZeroReward(RewardFn):
    """Default reward function that returns zero for all agents."""

    def __call__(
        self,
        transition: Any,
        events: list[Event],
        info: dict[str, Any],
    ) -> dict[AgentID, float]:
        agents = info.get("agents", [])
        return {agent: 0.0 for agent in agents}
