"""Overcooked-inspired cooperative kitchen environment for 4 agents.

Agents must coordinate onion pickup, pot filling, dish handling, and soup
delivery under a fixed horizon.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from marl_course.common.api import AgentID, ActionDict, StepResult
from marl_course.common.grid import Position, add_pos
from marl_course.common.rewards import Event
from .layouts import Layout, builtin_layouts


@dataclass(slots=True)
class CoopKitchenConfig:
    """Core environment timing/reward and observation sizing settings."""
    layout_name: str = "open_kitchen_4p"
    horizon: int = 400
    cook_time: int = 8
    delivery_reward: float = 20.0
    max_height: int = 11
    max_width: int = 15


@dataclass(slots=True)
class Pot:
    """State of one cooking pot."""
    position: Position
    onions: int = 0
    cook_remaining: int = 0
    ready: bool = False


class CoopKitchenEnv:
    """4-player cooperative cooking environment with shared team score.

    Action ids:
    - 0/1/2/3: move up/down/left/right
    - 4: stay
    - 5: interact (pickup / place / deliver depending on context)
    """
    ACTIONS = ["up", "down", "left", "right", "stay", "interact"]
    MOVE_DIRS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    TERRAIN = ["wall", "floor", "onion", "dish", "pot", "delivery", "counter", "out_of_bounds"]

    def __init__(self, config: CoopKitchenConfig | None = None, layout: Layout | None = None):
        self.config = config or CoopKitchenConfig()
        self.layout = layout or builtin_layouts()[self.config.layout_name]
        self.possible_agents = [f"agent_{idx}" for idx in range(4)]
        self.agents = list(self.possible_agents)
        self.rng = random.Random()
        self.step_count = 0
        self.positions: dict[AgentID, Position] = {}
        self.facing: dict[AgentID, int] = {}
        self.holding: dict[AgentID, str | None] = {}
        self.pots: dict[Position, Pot] = {}
        self.counter_items: dict[Position, str] = {}
        self.score = 0.0
        self.delivered_soups = 0
        self.collisions = 0
        self.invalid_interacts = 0
        self.events: list[Event] = []

    def reset(self, seed: int | None = None) -> tuple[dict[AgentID, dict[str, Any]], dict[AgentID, dict[str, Any]]]:
        if seed is not None:
            self.rng.seed(seed)
        self.agents = list(self.possible_agents)
        self.step_count = 0
        self.positions = {agent: self.layout.starts[idx] for idx, agent in enumerate(self.possible_agents)}
        self.facing = {agent: 4 for agent in self.possible_agents}
        self.holding = {agent: None for agent in self.possible_agents}
        self.pots = {pos: Pot(pos) for pos in self._terrain_positions("P")}
        self.counter_items = {}
        self.score = 0.0
        self.delivered_soups = 0
        self.collisions = 0
        self.invalid_interacts = 0
        self.events = []
        obs = self._observations()
        infos = {agent: {"action_mask": obs[agent]["action_mask"]} for agent in self.possible_agents}
        return obs, infos

    def step(self, actions: ActionDict) -> StepResult:
        self.events = []
        self.step_count += 1
        self._move_agents(actions)
        for agent in self.possible_agents:
            if actions.get(agent, 4) == 5:
                self._interact(agent)
        self._tick_pots()
        truncated = self.step_count >= self.config.horizon
        rewards = {agent: 0.0 for agent in self.possible_agents}
        for event in self.events:
            if event.name == "soup_delivered":
                for agent in self.possible_agents:
                    rewards[agent] += self.config.delivery_reward
        obs = self._observations()
        infos = {
            agent: {
                "events": list(self.events),
                "score": self.score,
                "delivered_soups": self.delivered_soups,
                "collisions": self.collisions,
                "invalid_interacts": self.invalid_interacts,
                "action_mask": obs[agent]["action_mask"],
            }
            for agent in self.possible_agents
        }
        dones = {agent: False for agent in self.possible_agents}
        truncs = {agent: truncated for agent in self.possible_agents}
        return StepResult(obs, rewards, dones, truncs, infos)

    def state(self) -> dict[str, Any]:
        return {
            "layout": self.layout.name,
            "grid": self.layout.grid,
            "positions": dict(self.positions),
            "facing": dict(self.facing),
            "holding": dict(self.holding),
            "pots": {pos: {"onions": pot.onions, "cook_remaining": pot.cook_remaining, "ready": pot.ready} for pos, pot in self.pots.items()},
            "counter_items": dict(self.counter_items),
            "score": self.score,
            "step": self.step_count,
        }

    def render(self, mode: str = "ansi") -> str:
        grid = [list(row) for row in self.layout.grid]
        for pos, item in self.counter_items.items():
            r, c = pos
            grid[r][c] = item[0].lower()
        for pos, pot in self.pots.items():
            r, c = pos
            grid[r][c] = "R" if pot.ready else str(pot.onions)
        for idx, agent in enumerate(self.possible_agents):
            r, c = self.positions[agent]
            suffix = {"onion": "o", "dish": "d", "soup": "s", None: ""}[self.holding[agent]]
            # Render-only symbols: digits are reserved for pot progress.
            # Free agents use letters; carrying agents use item letters.
            grid[r][c] = "ABEG"[idx] if not suffix else suffix
        header = f"CoopKitchen layout={self.layout.name} step={self.step_count} score={self.score} soups={self.delivered_soups}\n"
        return header + "\n".join("".join(row) for row in grid)

    def terrain_at(self, pos: Position) -> str:
        r, c = pos
        if r < 0 or c < 0 or r >= self.layout.height or c >= self.layout.width:
            return "X"
        return self.layout.grid[r][c]

    def passable_positions(self) -> set[Position]:
        cells = set()
        for r, row in enumerate(self.layout.grid):
            for c, char in enumerate(row):
                if char in {" ", "C"}:
                    cells.add((r, c))
        return cells

    def _move_agents(self, actions: ActionDict) -> None:
        passable = self.passable_positions()
        occupied = {pos: agent for agent, pos in self.positions.items()}
        proposals = {}
        for agent in self.possible_agents:
            action = actions.get(agent, 4)
            if action in self.MOVE_DIRS:
                self.facing[agent] = action
                nxt = add_pos(self.positions[agent], self.MOVE_DIRS[action])
                proposals[agent] = nxt if nxt in passable else self.positions[agent]
            else:
                proposals[agent] = self.positions[agent]
        counts: dict[Position, int] = {}
        for pos in proposals.values():
            counts[pos] = counts.get(pos, 0) + 1
        for agent, nxt in proposals.items():
            cur = self.positions[agent]
            other = occupied.get(nxt)
            swap = other is not None and proposals.get(other) == cur
            if counts[nxt] > 1 or swap:
                if nxt != cur:
                    self.collisions += 1
                    self.events.append(Event("collision", actor=agent, data={"to": nxt}))
                continue
            self.positions[agent] = nxt

    def _interact(self, agent: AgentID) -> None:
        targets = self._interaction_targets(agent)
        for target in targets:
            terrain = self.terrain_at(target)
            held = self.holding[agent]
            if terrain == "O" and held is None:
                self.holding[agent] = "onion"
                self.events.append(Event("onion_picked", actor=agent))
                return
            if terrain == "D" and held is None:
                self.holding[agent] = "dish"
                self.events.append(Event("dish_picked", actor=agent))
                return
            if terrain == "P" and target in self.pots:
                pot = self.pots[target]
                if held == "onion" and pot.onions < 3 and not pot.ready and pot.cook_remaining == 0:
                    pot.onions += 1
                    self.holding[agent] = None
                    if pot.onions == 3:
                        pot.cook_remaining = self.config.cook_time
                        self.events.append(Event("pot_started", actor=agent, data={"pot": target}))
                    else:
                        self.events.append(Event("onion_added", actor=agent, data={"pot": target, "onions": pot.onions}))
                    return
                if held == "dish" and pot.ready:
                    pot.ready = False
                    pot.onions = 0
                    self.holding[agent] = "soup"
                    self.events.append(Event("soup_picked", actor=agent))
                    return
            if terrain == "S" and held == "soup":
                self.holding[agent] = None
                self.score += self.config.delivery_reward
                self.delivered_soups += 1
                self.events.append(Event("soup_delivered", actor=agent, value=self.config.delivery_reward))
                return
            if terrain == "C":
                if held is not None and target not in self.counter_items:
                    self.counter_items[target] = held
                    self.holding[agent] = None
                    self.events.append(Event("counter_drop", actor=agent, data={"item": self.counter_items[target]}))
                    return
                if held is None and target in self.counter_items:
                    self.holding[agent] = self.counter_items.pop(target)
                    self.events.append(Event("counter_pick", actor=agent, data={"item": self.holding[agent]}))
                    return
        self.invalid_interacts += 1
        self.events.append(Event("invalid_interact", actor=agent))

    def _interaction_targets(self, agent: AgentID) -> list[Position]:
        pos = self.positions[agent]
        facing = self.facing.get(agent, 4)
        ordered = []
        if facing in self.MOVE_DIRS:
            ordered.append(add_pos(pos, self.MOVE_DIRS[facing]))
        ordered.extend(add_pos(pos, delta) for delta in self.MOVE_DIRS.values())
        ordered.append(pos)
        dedup = []
        seen = set()
        for item in ordered:
            if item not in seen:
                dedup.append(item)
                seen.add(item)
        return dedup

    def _tick_pots(self) -> None:
        for pot in self.pots.values():
            if pot.cook_remaining > 0:
                pot.cook_remaining -= 1
                if pot.cook_remaining == 0:
                    pot.ready = True
                    self.events.append(Event("soup_ready", data={"pot": pot.position}))

    def _terrain_positions(self, char: str) -> set[Position]:
        return {
            (r, c)
            for r, row in enumerate(self.layout.grid)
            for c, value in enumerate(row)
            if value == char
        }

    def _observations(self) -> dict[AgentID, dict[str, Any]]:
        shared = {
            "global_state": self.state(),
            "layout_id": self.layout.name,
            "valid_cell_mask": self._valid_cell_mask(),
        }
        obs = {}
        for agent in self.possible_agents:
            obs[agent] = {
                **shared,
                "agent_obs": self._planes(agent),
                "agent_features": {
                    other: {
                        "position": self.positions[other],
                        "holding": self.holding[other],
                        "facing": self.facing[other],
                    }
                    for other in self.possible_agents
                },
                "self_id": int(agent.split("_")[-1]),
                "action_mask": self._action_mask(agent),
            }
        return obs

    def _planes(self, agent: AgentID) -> dict[str, list[list[float]]]:
        h, w = self.config.max_height, self.config.max_width

        def zeros() -> list[list[float]]:
            return [[0.0 for _ in range(w)] for _ in range(h)]

        planes = {name: zeros() for name in self.TERRAIN}
        for item in ["onion_item", "dish_item", "soup_item", "pot_onions", "pot_ready", "pot_cooking"]:
            planes[item] = zeros()
        for idx in range(4):
            planes[f"agent_{idx}"] = zeros()
        for r in range(h):
            for c in range(w):
                if r >= self.layout.height or c >= self.layout.width:
                    planes["out_of_bounds"][r][c] = 1.0
                    continue
                char = self.layout.grid[r][c]
                key = {
                    "X": "wall",
                    " ": "floor",
                    "O": "onion",
                    "D": "dish",
                    "P": "pot",
                    "S": "delivery",
                    "C": "counter",
                }[char]
                planes[key][r][c] = 1.0
        for pos, pot in self.pots.items():
            r, c = pos
            planes["pot_onions"][r][c] = pot.onions / 3.0
            planes["pot_ready"][r][c] = 1.0 if pot.ready else 0.0
            planes["pot_cooking"][r][c] = pot.cook_remaining / max(1, self.config.cook_time)
        for pos, item in self.counter_items.items():
            r, c = pos
            planes[f"{item}_item"][r][c] = 1.0
        for idx, other in enumerate(self.possible_agents):
            r, c = self.positions[other]
            planes[f"agent_{idx}"][r][c] = 1.0
            held = self.holding[other]
            if held:
                planes[f"{held}_item"][r][c] = 1.0
        return planes

    def _valid_cell_mask(self) -> list[list[int]]:
        h, w = self.config.max_height, self.config.max_width
        return [
            [1 if r < self.layout.height and c < self.layout.width else 0 for c in range(w)]
            for r in range(h)
        ]

    def _action_mask(self, agent: AgentID) -> list[int]:
        pos = self.positions[agent]
        occupied = {p for a, p in self.positions.items() if a != agent}
        passable = self.passable_positions()
        mask = [0, 0, 0, 0, 1, 1]
        for action, delta in self.MOVE_DIRS.items():
            nxt = add_pos(pos, delta)
            mask[action] = int(nxt in passable and nxt not in occupied)
        return mask
