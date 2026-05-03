"""Baseline opponent policies for Bomber Arena.

These are used for student training opponents and teacher-side benchmark
matches. The rule-based policy is intentionally non-trivial.
"""
from __future__ import annotations

import random
from typing import Any

from marl_course.common.grid import DIRS, Position, add_pos, first_step_action, manhattan, shortest_path


class BomberStayPolicy:
    """Always selects action 0 (stay)."""

    def act(self, obs: dict[str, Any], action_mask: list[int] | None = None, deterministic: bool = True) -> int:
        return 0


class BomberRandomPolicy:
    """Uniform random legal-action policy."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, obs: dict[str, Any], action_mask: list[int] | None = None, deterministic: bool = True) -> int:
        mask = action_mask or obs.get("action_mask", [1, 1, 1, 1, 1, 1])
        choices = [idx for idx, ok in enumerate(mask) if ok]
        return self.rng.choice(choices or [0])


class BomberRuleBasedPolicy:
    """Heuristic baseline with hazard avoidance and tactical bombing.

    High-level behavior:
    1) escape imminent blast danger
    2) pursue power-ups / safe pressure opportunities
    3) place bombs when escape route exists
    """
    """A reasonably competent baseline.

    It avoids predicted blast zones, places bombs near crates or enemies only
    when an escape route exists, collects powerups, and otherwise moves toward
    useful board areas.
    """

    def __init__(self, aggression: float = 0.35, seed: int | None = None):
        self.aggression = aggression
        self.rng = random.Random(seed)

    def act(self, obs: dict[str, Any], action_mask: list[int] | None = None, deterministic: bool = True) -> int:
        mask = action_mask or obs.get("action_mask", [1, 1, 1, 1, 1, 1])
        stats = obs["stats"]
        self_agent = f"agent_{obs['self_id']}"
        if not stats[self_agent]["alive"]:
            return 0
        pos = tuple(stats[self_agent]["position"])
        grid = obs["grid"]
        passable = self._passable(grid, include_bombs=False)
        danger = self._danger(grid)

        if pos in danger:
            escape = self._move_to_safe(pos, passable, danger, mask)
            if escape is not None:
                return escape
            return self._masked_random(mask)

        if mask[5] and self._bomb_would_help(pos, obs) and self._has_escape_after_bomb(pos, passable, obs):
            return 5

        powerups = self._positions_where(grid, "bomb_up") | self._positions_where(grid, "fire_up")
        action = self._path_action(pos, powerups, passable, danger, mask)
        if action is not None:
            return action

        enemies = [
            tuple(data["position"])
            for agent, data in stats.items()
            if agent != self_agent and data["alive"]
        ]
        if enemies and (deterministic or self.rng.random() < self.aggression):
            adjacent_to_enemy = {
                add_pos(enemy, delta)
                for enemy in enemies
                for delta in DIRS.values()
                if add_pos(enemy, delta) in passable
            }
            action = self._path_action(pos, adjacent_to_enemy, passable, danger, mask)
            if action is not None:
                return action

        frontier = self._crate_frontier(grid, passable)
        action = self._path_action(pos, frontier, passable, danger, mask)
        if action is not None:
            return action
        return self._safe_random(pos, mask, danger)

    def _passable(self, grid: dict[str, list[list[float]]], include_bombs: bool) -> set[Position]:
        h, w = len(grid["wall"]), len(grid["wall"][0])
        cells = set()
        for r in range(h):
            for c in range(w):
                if grid["wall"][r][c] or grid["wood"][r][c]:
                    continue
                if not include_bombs and grid["bomb_timer"][r][c] > 0:
                    continue
                cells.add((r, c))
        return cells

    def _danger(self, grid: dict[str, list[list[float]]]) -> set[Position]:
        return self._positions_where(grid, "danger") | self._positions_where(grid, "flame")

    def _positions_where(self, grid: dict[str, list[list[float]]], key: str) -> set[Position]:
        h, w = len(grid[key]), len(grid[key][0])
        return {(r, c) for r in range(h) for c in range(w) if grid[key][r][c] > 0}

    def _move_to_safe(self, pos: Position, passable: set[Position], danger: set[Position], mask: list[int]) -> int | None:
        safe = passable - danger
        action = self._path_action(pos, safe, passable, danger=set(), mask=mask)
        return action

    def _path_action(
        self,
        pos: Position,
        goals: set[Position],
        passable: set[Position],
        danger: set[Position],
        mask: list[int],
    ) -> int | None:
        goals = {goal for goal in goals if goal in passable and goal not in danger}
        if not goals:
            return None
        path = shortest_path(pos, goals, passable - danger)
        action = first_step_action(path)
        if action and mask[action]:
            return action
        return None

    def _bomb_would_help(self, pos: Position, obs: dict[str, Any]) -> bool:
        grid = obs["grid"]
        stats = obs["stats"]
        blast = stats[f"agent_{obs['self_id']}"]["blast"]
        for direction in DIRS.values():
            cur = pos
            for _ in range(blast):
                cur = add_pos(cur, direction)
                r, c = cur
                if grid["wall"][r][c]:
                    break
                if grid["wood"][r][c]:
                    return True
                for idx in range(4):
                    if idx != obs["self_id"] and grid[f"agent_{idx}"][r][c]:
                        return True
        return False

    def _has_escape_after_bomb(self, pos: Position, passable: set[Position], obs: dict[str, Any]) -> bool:
        stats = obs["stats"]
        blast = stats[f"agent_{obs['self_id']}"]["blast"]
        danger = set(self._danger(obs["grid"]))
        danger.add(pos)
        for direction in DIRS.values():
            cur = pos
            for _ in range(blast):
                cur = add_pos(cur, direction)
                danger.add(cur)
                r, c = cur
                if obs["grid"]["wall"][r][c] or obs["grid"]["wood"][r][c]:
                    break
        safe = passable - danger
        return bool(shortest_path(pos, safe, passable))

    def _crate_frontier(self, grid: dict[str, list[list[float]]], passable: set[Position]) -> set[Position]:
        crates = self._positions_where(grid, "wood")
        frontier = set()
        for crate in crates:
            for direction in DIRS.values():
                pos = add_pos(crate, direction)
                if pos in passable:
                    frontier.add(pos)
        return frontier

    def _safe_random(self, pos: Position, mask: list[int], danger: set[Position]) -> int:
        choices = []
        for action, delta in DIRS.items():
            if mask[action] and add_pos(pos, delta) not in danger:
                choices.append(action)
        if mask[0] and pos not in danger:
            choices.append(0)
        return self.rng.choice(choices or [idx for idx, ok in enumerate(mask) if ok] or [0])

    def _masked_random(self, mask: list[int]) -> int:
        return self.rng.choice([idx for idx, ok in enumerate(mask) if ok] or [0])
