"""Reference reward-shaping classes students can customize."""
from __future__ import annotations

from typing import Any

from marl_course.common.rewards import Event, RewardFn


class BomberShapingReward(RewardFn):
    """Event-based dense reward for Bomber training loops."""

    def __init__(
        self,
        win: float = 1.0,
        block: float = 0.03,
        powerup: float = 0.05,
        self_elim: float = -0.5,
        eliminated: float = -0.4,
        eliminate: float = 0.4,
    ):
        self.win = win
        self.block = block
        self.powerup = powerup
        self.self_elim = self_elim
        self.eliminated = eliminated
        self.eliminate = eliminate

    def __call__(self, transition: Any, events: list[Event], info: dict[str, Any]) -> dict[str, float]:
        """Map emitted events to per-agent shaped rewards."""
        agents = info.get("agents", [f"agent_{idx}" for idx in range(4)])
        rewards = {agent: 0.0 for agent in agents}
        for event in events:
            if event.name == "winner" and event.actor:
                rewards[event.actor] += self.win
            elif event.name == "block_destroyed" and event.actor:
                rewards[event.actor] += self.block
            elif event.name == "powerup_collected" and event.actor:
                rewards[event.actor] += self.powerup
            elif event.name == "self_eliminated" and event.actor:
                rewards[event.actor] += self.self_elim
            elif event.name == "agent_eliminated" and event.target in rewards:
                rewards[event.target] += self.eliminated
            elif event.name == "enemy_eliminated" and event.actor:
                rewards[event.actor] += self.eliminate
        return rewards


class CoopKitchenShapingReward(RewardFn):
    """Event-based team shaping reward for cooperative kitchen."""

    def __init__(self, onion: float = 0.5, dish: float = 0.2, soup_pick: float = 1.0, delivery: float = 20.0, collision: float = -0.05):
        self.onion = onion
        self.dish = dish
        self.soup_pick = soup_pick
        self.delivery = delivery
        self.collision = collision

    def __call__(self, transition: Any, events: list[Event], info: dict[str, Any]) -> dict[str, float]:
        """Convert kitchen events into dense role-learning feedback."""
        agents = info.get("agents", [f"agent_{idx}" for idx in range(4)])
        rewards = {agent: 0.0 for agent in agents}
        for event in events:
            if event.name in {"onion_added", "pot_started"} and event.actor:
                rewards[event.actor] += self.onion
            elif event.name == "dish_picked" and event.actor:
                rewards[event.actor] += self.dish
            elif event.name == "soup_picked" and event.actor:
                rewards[event.actor] += self.soup_pick
            elif event.name == "soup_delivered":
                for agent in agents:
                    rewards[agent] += self.delivery
            elif event.name == "collision" and event.actor:
                rewards[event.actor] += self.collision
        return rewards
