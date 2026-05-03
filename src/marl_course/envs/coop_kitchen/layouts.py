"""Static and generated kitchen layouts for train/eval."""
from __future__ import annotations

import random
from dataclasses import dataclass

from marl_course.common.grid import Position


@dataclass(frozen=True, slots=True)
class Layout:
    """Grid layout definition including initial spawn positions."""
    name: str
    grid: tuple[str, ...]
    starts: tuple[Position, Position, Position, Position]

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0])


def builtin_layouts() -> dict[str, Layout]:
    """Return named classroom layouts used for standard evaluation."""
    return {
        "open_kitchen_4p": Layout(
            "open_kitchen_4p",
            (
                "XXXXXXXXXXX",
                "X O   P   X",
                "X         X",
                "X    C    X",
                "X         X",
                "X  D   S  X",
                "XXXXXXXXXXX",
            ),
            ((2, 2), (2, 8), (4, 2), (4, 8)),
        ),
        "corridor_4p": Layout(
            "corridor_4p",
            (
                "XXXXXXXXXXX",
                "XO   X   PX",
                "X    X    X",
                "X         X",
                "X    X    X",
                "XD   X   SX",
                "XXXXXXXXXXX",
            ),
            ((3, 2), (3, 4), (3, 6), (3, 8)),
        ),
        "split_station_4p": Layout(
            "split_station_4p",
            (
                "XXXXXXXXXXX",
                "XO      P X",
                "X XXX XXX X",
                "X    C    X",
                "X XXX XXX X",
                "X D      SX",
                "XXXXXXXXXXX",
            ),
            ((3, 2), (3, 4), (3, 6), (3, 8)),
        ),
    }


def generate_layout(seed: int, family: str = "open", width: int = 13, height: int = 9) -> Layout:
    """Generate a procedural layout for zero-shot generalization experiments."""
    """Generate a reproducible held-out kitchen layout.

    The recipe remains onion soup; only station placement and obstacles vary.
    """

    rng = random.Random(f"{seed}:{family}:{width}:{height}")
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for r in range(height):
        for c in range(width):
            if r == 0 or c == 0 or r == height - 1 or c == width - 1:
                grid[r][c] = "X"
    if family in {"corridor", "bottleneck", "heldout_bottleneck"}:
        mid = width // 2
        for r in range(1, height - 1):
            if r not in {height // 2, height // 2 - 1}:
                grid[r][mid] = "X"
    if family in {"island", "heldout_island"}:
        for r in range(2, height - 2):
            for c in range(3, width - 3):
                if r in {2, height - 3} or c in {3, width - 4}:
                    grid[r][c] = "X"
        grid[height // 2][3] = " "
        grid[height // 2][width - 4] = " "
    if family in {"long_delivery", "heldout_zigzag"}:
        for c in range(2, width - 2, 2):
            for r in range(1, height - 1):
                if r != (c % (height - 2)) + 1:
                    grid[r][c] = "X"

    floor = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1) if grid[r][c] == " "]
    rng.shuffle(floor)
    stations = {"O": floor.pop(), "D": floor.pop(), "P": floor.pop(), "S": floor.pop()}
    for char, pos in stations.items():
        r, c = pos
        grid[r][c] = char
    if floor:
        r, c = floor.pop()
        grid[r][c] = "C"
    starts = tuple(floor[:4]) if len(floor) >= 4 else ((1, 1), (1, width - 2), (height - 2, 1), (height - 2, width - 2))
    return Layout(f"{family}_{seed}", tuple("".join(row) for row in grid), starts)  # type: ignore[arg-type]
