"""Grid-world helper functions used by both game environments.

These helpers centralize common movement/path logic so policy and environment
code can stay focused on game-specific rules.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable

Position = tuple[int, int]

DIRS: dict[int, Position] = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


def add_pos(a: Position, b: Position) -> Position:
    """Add two (row, col) positions."""
    return (a[0] + b[0], a[1] + b[1])


def manhattan(a: Position, b: Position) -> int:
    """Compute Manhattan distance |dr| + |dc|."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def shortest_path(
    start: Position,
    goals: Iterable[Position],
    passable: set[Position],
) -> list[Position]:
    """Breadth-first shortest path from start to any goal over passable cells."""
    goal_set = set(goals)
    if start in goal_set:
        return [start]
    queue: deque[Position] = deque([start])
    parent: dict[Position, Position | None] = {start: None}
    while queue:
        current = queue.popleft()
        for delta in DIRS.values():
            nxt = add_pos(current, delta)
            if nxt not in passable or nxt in parent:
                continue
            parent[nxt] = current
            if nxt in goal_set:
                path = [nxt]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])  # type: ignore[arg-type]
                path.reverse()
                return path
            queue.append(nxt)
    return []


def first_step_action(path: list[Position]) -> int:
    """Convert the first edge of a path into the environment action id."""
    if len(path) < 2:
        return 0
    sr, sc = path[0]
    nr, nc = path[1]
    delta = (nr - sr, nc - sc)
    for action, direction in DIRS.items():
        if direction == delta:
            return action
    return 0
