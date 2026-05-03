"""Shared API types for all environments in this project.

These types make training/evaluation code environment-agnostic as long as the
environment follows the minimal parallel multi-agent interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

AgentID = str
ObsDict = dict[str, Any]
ActionDict = dict[AgentID, int]


@dataclass(slots=True)
class StepResult:
    """Container returned by `env.step(actions)`.

    Fields mirror common MARL interfaces:
    - observations: next observation for each agent
    - rewards: scalar reward per agent
    - terminations/truncations: done flags (terminal vs time-limit)
    - infos: extra debug/event data
    """

    observations: dict[AgentID, ObsDict]
    rewards: dict[AgentID, float]
    terminations: dict[AgentID, bool]
    truncations: dict[AgentID, bool]
    infos: dict[AgentID, dict[str, Any]]


class ParallelEnv(Protocol):
    """Small PettingZoo Parallel API subset used by the course code."""

    agents: list[AgentID]
    possible_agents: list[AgentID]

    def reset(self, seed: int | None = None) -> tuple[dict[AgentID, ObsDict], dict[AgentID, dict[str, Any]]]:
        ...

    def step(self, actions: ActionDict) -> StepResult:
        ...

    def state(self) -> dict[str, Any]:
        ...

    def render(self, mode: str = "ansi") -> str | list[list[tuple[int, int, int]]]:
        ...
