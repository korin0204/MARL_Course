"""Baseline 4-agent team policies for Coop Kitchen."""
from __future__ import annotations

import random
from typing import Any

from marl_course.common.grid import Position, add_pos, shortest_path

MOVE_DIRS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


class CoopStayTeamPolicy:
    """All agents wait in place every step."""

    def reset_episode(self, layout_metadata: dict[str, Any] | None = None) -> None:
        return None

    def act(self, obs: dict[str, Any], action_mask: list[list[int]] | None = None, deterministic: bool = True) -> list[int]:
        return [4, 4, 4, 4]


class CoopRandomTeamPolicy:
    """Each agent samples a legal action independently."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def reset_episode(self, layout_metadata: dict[str, Any] | None = None) -> None:
        return None

    def act(self, obs: dict[str, Any], action_mask: list[list[int]] | None = None, deterministic: bool = True) -> list[int]:
        masks = action_mask or obs["action_mask"]
        return [self.rng.choice([idx for idx, ok in enumerate(mask) if ok] or [4]) for mask in masks]


class CoopGreedyTeamPolicy:
    """Heuristic cooperative baseline.

    It assigns coarse roles (onion runner, dish helper, cooker) and chooses
    shortest-path moves to context-dependent targets.
    """
    """Scripted team with dynamic role switching.

    Agents prefer different jobs, but switch when the pot is ready, ingredients
    are missing, or delivery is available.
    """

    def __init__(
        self,
        onion_workers: tuple[int, ...] = (0, 2),
        dish_workers: tuple[int, ...] = (1, 2, 3),
        cook_dish_workers: tuple[int, ...] = (1, 3),
    ):
        self.onion_workers = set(onion_workers)
        self.dish_workers = set(dish_workers)
        self.cook_dish_workers = set(cook_dish_workers)

    def reset_episode(self, layout_metadata: dict[str, Any] | None = None) -> None:
        return None

    def act(self, obs: dict[str, Any], action_mask: list[list[int]] | None = None, deterministic: bool = True) -> list[int]:
        masks = action_mask or obs["action_mask"]
        actions = []
        for idx in range(4):
            actions.append(self._act_agent(obs, idx, masks[idx]))
        return actions

    def _act_agent(self, obs: dict[str, Any], idx: int, mask: list[int]) -> int:
        state = obs["global_state"]
        agent = f"agent_{idx}"
        pos = tuple(state["positions"][agent])
        held = state["holding"][agent]
        grid = state["grid"]
        passable = self._passable(grid)
        pots = state["pots"]
        any_onion_carrier = any(item == "onion" for item in state["holding"].values())
        ready_pots = [tuple(pos) for pos, pot in pots.items() if pot["ready"]]
        needy_pots = [tuple(pos) for pos, pot in pots.items() if pot["onions"] < 3 and not pot["ready"] and pot["cook_remaining"] == 0]
        cooking_pots = [tuple(pos) for pos, pot in pots.items() if pot["cook_remaining"] > 0]

        if held == "soup":
            return self._go_interact(pos, self._terrain(grid, "S"), passable, mask)
        if held == "dish" and ready_pots:
            return self._go_interact(pos, set(ready_pots), passable, mask)
        if held == "onion" and needy_pots:
            return self._go_interact(pos, set(needy_pots), passable, mask)
        if held is None:
            if ready_pots and idx in self.dish_workers:
                return self._go_interact(pos, self._terrain(grid, "D"), passable, mask)
            if needy_pots and idx in self.onion_workers and not any_onion_carrier:
                return self._go_interact(pos, self._terrain(grid, "O"), passable, mask)
            if ready_pots:
                return self._go_interact(pos, self._terrain(grid, "D"), passable, mask)
            if needy_pots and idx in self.onion_workers and not any_onion_carrier:
                return self._go_interact(pos, self._terrain(grid, "O"), passable, mask)
            if cooking_pots and idx in self.cook_dish_workers:
                return self._go_interact(pos, self._terrain(grid, "D"), passable, mask)
            wait = self._wait_goals(grid, idx, passable)
            action = self._move_to(pos, wait, passable, mask)
            if action is not None:
                return action
        return 4 if mask[4] else self._first_legal(mask)

    def _go_interact(self, pos: Position, targets: set[Position], passable: set[Position], mask: list[int]) -> int:
        if self._adjacent(pos, targets):
            return 5 if mask[5] else 4
        goals = {add_pos(t, delta) for t in targets for delta in MOVE_DIRS.values() if add_pos(t, delta) in passable}
        path = shortest_path(pos, goals, passable)
        action = self._first_step_action(path)
        if action is not None and mask[action]:
            return action
        return 4 if mask[4] else self._first_legal(mask)

    def _move_to(self, pos: Position, goals: set[Position], passable: set[Position], mask: list[int]) -> int | None:
        path = shortest_path(pos, goals, passable)
        action = self._first_step_action(path)
        if action is not None and mask[action]:
            return action
        return None

    def _wait_goals(self, grid: tuple[str, ...], idx: int, passable: set[Position]) -> set[Position]:
        h, w = len(grid), len(grid[0])
        anchors = {
            0: (1, 1),
            1: (h - 2, 1),
            2: (1, w - 2),
            3: (h - 2, w - 2),
        }
        anchor = anchors[idx]
        return set(sorted(passable, key=lambda p: abs(p[0] - anchor[0]) + abs(p[1] - anchor[1]))[:3])

    def _terrain(self, grid: tuple[str, ...], char: str) -> set[Position]:
        return {(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == char}

    def _passable(self, grid: tuple[str, ...]) -> set[Position]:
        return {(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value in {" ", "C"}}

    def _adjacent(self, pos: Position, targets: set[Position]) -> bool:
        return any(add_pos(pos, delta) in targets for delta in MOVE_DIRS.values()) or pos in targets

    def _first_step_action(self, path: list[Position]) -> int | None:
        if len(path) < 2:
            return None
        sr, sc = path[0]
        nr, nc = path[1]
        delta = (nr - sr, nc - sc)
        for action, direction in MOVE_DIRS.items():
            if direction == delta:
                return action
        return None

    def _first_legal(self, mask: list[int]) -> int:
        for idx, ok in enumerate(mask):
            if ok:
                return idx
        return 4
