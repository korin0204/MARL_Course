"""Bomber-style multi-agent battle environment used in the course.

This implementation is intentionally simpler than commercial Bomberman while
keeping core mechanics: movement, bomb placement, blast propagation, and
last-survivor win condition.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from marl_course.common.api import AgentID, ActionDict, StepResult
from marl_course.common.grid import DIRS, Position, add_pos
from marl_course.common.rewards import Event


@dataclass(slots=True)
class BomberConfig:
    """Environment constants (map size, bomb timing, and power-up rates)."""
    height: int = 11
    width: int = 13
    max_steps: int = 300
    bomb_timer: int = 4
    flame_life: int = 2
    initial_blast: int = 2
    initial_ammo: int = 1
    wood_density: float = 0.55
    powerup_prob: float = 0.25


@dataclass(slots=True)
class Bomb:
    """Runtime bomb state tracked on the grid."""
    owner: AgentID
    position: Position
    timer: int
    blast: int


class BomberArenaEnv:
    """4-agent fully-observed Bomber arena environment.

    Observation contains board planes + per-agent stats + legal action mask.
    Action ids:
    - 0: stay
    - 1/2/3/4: move up/down/left/right
    - 5: place bomb
    """
    """A compact fully-observed 4-player bomber arena."""

    ACTIONS = ["stay", "up", "down", "left", "right", "bomb"]
    WALL = "#"
    WOOD = "+"
    EMPTY = " "

    def __init__(self, config: BomberConfig | None = None):
        self.config = config or BomberConfig()
        self.possible_agents = [f"agent_{idx}" for idx in range(4)]
        self.agents = list(self.possible_agents)
        self.rng = random.Random()
        self.step_count = 0
        self.board: list[list[str]] = []
        self.positions: dict[AgentID, Position] = {}
        self.alive: dict[AgentID, bool] = {}
        self.ammo: dict[AgentID, int] = {}
        self.blast: dict[AgentID, int] = {}
        self.bombs: list[Bomb] = []
        self.flames: dict[Position, int] = {}
        self.flame_owners: dict[Position, AgentID] = {}
        self.powerups: dict[Position, str] = {}
        self.events: list[Event] = []
        self.last_winner: AgentID | None = None
        self.rankings: list[AgentID] = []

    def reset(self, seed: int | None = None) -> tuple[dict[AgentID, dict[str, Any]], dict[AgentID, dict[str, Any]]]:
        if seed is not None:
            self.rng.seed(seed)
        self.step_count = 0
        self.agents = list(self.possible_agents)
        self.bombs = []
        self.flames = {}
        self.flame_owners = {}
        self.powerups = {}
        self.events = []
        self.last_winner = None
        self.rankings = []
        self.board = self._make_board()
        spawns = self._spawn_positions()
        self.positions = {agent: spawns[idx] for idx, agent in enumerate(self.possible_agents)}
        self.alive = {agent: True for agent in self.possible_agents}
        self.ammo = {agent: self.config.initial_ammo for agent in self.possible_agents}
        self.blast = {agent: self.config.initial_blast for agent in self.possible_agents}
        obs = self._observations()
        infos = {agent: {"action_mask": obs[agent]["action_mask"]} for agent in self.possible_agents}
        return obs, infos

    def step(self, actions: ActionDict) -> StepResult:
        self.events = []
        self.step_count += 1
        live_agents = [agent for agent in self.possible_agents if self.alive.get(agent, False)]

        for agent in live_agents:
            action = actions.get(agent, 0)
            if action == 5 and self._can_place_bomb(agent):
                pos = self.positions[agent]
                self.bombs.append(Bomb(agent, pos, self.config.bomb_timer, self.blast[agent]))
                self.ammo[agent] -= 1
                self.events.append(Event("bomb_placed", actor=agent, data={"pos": pos}))

        self._move_agents(actions, live_agents)
        self._tick_flames()
        self._tick_bombs()
        self._kill_agents_in_flames()

        alive_agents = [agent for agent in self.possible_agents if self.alive.get(agent, False)]
        terminated = len(alive_agents) <= 1
        truncated = self.step_count >= self.config.max_steps
        if terminated and alive_agents:
            self.last_winner = alive_agents[0]
            self.rankings.insert(0, alive_agents[0])
            self.events.append(Event("winner", actor=alive_agents[0], value=1.0))
        elif truncated:
            self.events.append(Event("timeout"))

        rewards = {agent: 0.0 for agent in self.possible_agents}
        if self.last_winner is not None:
            rewards[self.last_winner] = 1.0
        obs = self._observations()
        infos = {
            agent: {
                "events": list(self.events),
                "winner_id": self.last_winner,
                "rankings": list(self.rankings),
                "survival_steps": self.step_count if self.alive.get(agent, False) else None,
                "action_mask": obs[agent]["action_mask"],
            }
            for agent in self.possible_agents
        }
        dones = {agent: terminated for agent in self.possible_agents}
        truncs = {agent: truncated for agent in self.possible_agents}
        return StepResult(obs, rewards, dones, truncs, infos)

    def state(self) -> dict[str, Any]:
        return {
            "board": [row[:] for row in self.board],
            "positions": dict(self.positions),
            "alive": dict(self.alive),
            "ammo": dict(self.ammo),
            "blast": dict(self.blast),
            "bombs": [
                {"owner": bomb.owner, "position": bomb.position, "timer": bomb.timer, "blast": bomb.blast}
                for bomb in self.bombs
            ],
            "flames": dict(self.flames),
            "flame_owners": dict(self.flame_owners),
            "powerups": dict(self.powerups),
            "step": self.step_count,
        }

    def render(self, mode: str = "ansi") -> str:
        grid = [row[:] for row in self.board]
        for pos, item in self.powerups.items():
            r, c = pos
            # Render-only symbols: U=extra bomb, F=larger flame range.
            # Agents use letters below, so bomb timers can safely use digits.
            grid[r][c] = "U" if item == "bomb_up" else "F"
        for bomb in self.bombs:
            r, c = bomb.position
            grid[r][c] = str(max(0, bomb.timer))
        for (r, c), _life in self.flames.items():
            grid[r][c] = "*"
        for idx, agent in enumerate(self.possible_agents):
            if self.alive.get(agent, False):
                r, c = self.positions[agent]
                grid[r][c] = "ABCD"[idx]
        header = f"BomberArena step={self.step_count} winner={self.last_winner}\n"
        return header + "\n".join("".join(row) for row in grid)

    def danger_map(self, include_future: bool = True) -> set[Position]:
        danger = set(self.flames)
        if include_future:
            for bomb in self.bombs:
                # 観測用の危険マップは「もし爆発したら届くマス」を計算するだけ。
                # ここで木箱破壊やpowerup生成などの副作用を起こすと、
                # 観測を作るだけで環境が変わってしまい、DQNの学習信号が壊れる。
                danger.update(self._blast_cells(bomb.position, bomb.blast, apply_effects=False))
        return danger

    def passable_positions(self, ignore_bombs: bool = False) -> set[Position]:
        blocked_bombs = set() if ignore_bombs else {bomb.position for bomb in self.bombs}
        cells = set()
        for r in range(self.config.height):
            for c in range(self.config.width):
                pos = (r, c)
                if self.board[r][c] == self.EMPTY and pos not in blocked_bombs:
                    cells.add(pos)
        return cells

    def _make_board(self) -> list[list[str]]:
        h, w = self.config.height, self.config.width
        board = [[self.EMPTY for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if r == 0 or c == 0 or r == h - 1 or c == w - 1 or (r % 2 == 0 and c % 2 == 0):
                    board[r][c] = self.WALL
        safe = set()
        for spawn in self._spawn_positions():
            safe.add(spawn)
            for direction in DIRS.values():
                safe.add(add_pos(spawn, direction))
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if board[r][c] == self.EMPTY and (r, c) not in safe and self.rng.random() < self.config.wood_density:
                    board[r][c] = self.WOOD
        return board

    def _spawn_positions(self) -> list[Position]:
        h, w = self.config.height, self.config.width
        return [(1, 1), (1, w - 2), (h - 2, 1), (h - 2, w - 2)]

    def _can_place_bomb(self, agent: AgentID) -> bool:
        return self.alive.get(agent, False) and self.ammo[agent] > 0 and all(
            bomb.position != self.positions[agent] for bomb in self.bombs
        )

    def _move_agents(self, actions: ActionDict, live_agents: list[AgentID]) -> None:
        occupied = {self.positions[agent]: agent for agent in live_agents}
        bomb_positions = {bomb.position for bomb in self.bombs}
        proposals: dict[AgentID, Position] = {}
        for agent in live_agents:
            action = actions.get(agent, 0)
            if action in DIRS:
                nxt = add_pos(self.positions[agent], DIRS[action])
                if self._is_walkable(nxt) and nxt not in bomb_positions:
                    proposals[agent] = nxt
                else:
                    proposals[agent] = self.positions[agent]
            else:
                proposals[agent] = self.positions[agent]

        counts: dict[Position, int] = {}
        for pos in proposals.values():
            counts[pos] = counts.get(pos, 0) + 1

        for agent, nxt in proposals.items():
            cur = self.positions[agent]
            swap = False
            other = occupied.get(nxt)
            if other is not None and proposals.get(other) == cur:
                swap = True
            if counts[nxt] > 1 or swap:
                continue
            self.positions[agent] = nxt
            if nxt in self.powerups:
                item = self.powerups.pop(nxt)
                if item == "bomb_up":
                    self.ammo[agent] += 1
                else:
                    self.blast[agent] += 1
                self.events.append(Event("powerup_collected", actor=agent, data={"item": item}))

    def _is_walkable(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.config.height and 0 <= c < self.config.width and self.board[r][c] == self.EMPTY

    def _tick_flames(self) -> None:
        expired = []
        for pos, life in self.flames.items():
            self.flames[pos] = life - 1
            if self.flames[pos] <= 0:
                expired.append(pos)
        for pos in expired:
            del self.flames[pos]
            self.flame_owners.pop(pos, None)

    def _tick_bombs(self) -> None:
        to_explode: list[Bomb] = []
        for bomb in self.bombs:
            bomb.timer -= 1
            if bomb.timer <= 0:
                to_explode.append(bomb)
        exploded: set[Position] = set()
        while to_explode:
            bomb = to_explode.pop()
            if bomb.position in exploded:
                continue
            exploded.add(bomb.position)
            if bomb in self.bombs:
                self.bombs.remove(bomb)
            self.ammo[bomb.owner] += 1
            cells = self._blast_cells(bomb.position, bomb.blast, apply_effects=True, owner=bomb.owner)
            for cell in cells:
                self.flames[cell] = self.config.flame_life
                self.flame_owners[cell] = bomb.owner
            self.events.append(Event("bomb_exploded", actor=bomb.owner, data={"pos": bomb.position}))
            for other in list(self.bombs):
                if other.position in cells and other.position not in exploded:
                    to_explode.append(other)

    def _blast_cells(
        self,
        origin: Position,
        blast: int,
        apply_effects: bool = True,
        owner: AgentID | None = None,
    ) -> set[Position]:
        """爆風が届くマスを返す。

        `apply_effects=True` は実際の爆発処理用で、木箱破壊・powerup生成・
        event発行を行う。`False` は観測のdanger plane用で、副作用なしに
        blast範囲だけを見積もる。
        """

        cells = {origin}
        for direction in DIRS.values():
            current = origin
            for _ in range(blast):
                current = add_pos(current, direction)
                r, c = current
                if not (0 <= r < self.config.height and 0 <= c < self.config.width):
                    break
                if self.board[r][c] == self.WALL:
                    break
                cells.add(current)
                if self.board[r][c] == self.WOOD:
                    if apply_effects:
                        self.board[r][c] = self.EMPTY
                        self.events.append(Event("block_destroyed", actor=owner, data={"pos": current}))
                        if self.rng.random() < self.config.powerup_prob:
                            self.powerups[current] = "bomb_up" if self.rng.random() < 0.5 else "fire_up"
                    break
        return cells

    def _kill_agents_in_flames(self) -> None:
        for agent, pos in list(self.positions.items()):
            if self.alive.get(agent, False) and pos in self.flames:
                self.alive[agent] = False
                self.rankings.append(agent)
                flame_owner = self.flame_owners.get(pos)
                if flame_owner == agent:
                    self.events.append(Event("self_eliminated", actor=agent, target=agent, data={"pos": pos}))
                elif flame_owner is not None:
                    # enemy_eliminated is the event students should use for
                    # kill rewards: actor=killer, target=victim.
                    self.events.append(Event("enemy_eliminated", actor=flame_owner, target=agent, value=1.0, data={"pos": pos}))
                    self.events.append(Event("agent_eliminated", actor=flame_owner, target=agent, value=1.0, data={"pos": pos}))
                else:
                    self.events.append(Event("agent_eliminated", actor=None, target=agent, data={"pos": pos}))

    def _observations(self) -> dict[AgentID, dict[str, Any]]:
        danger = self.danger_map(include_future=True)
        bomb_map = {bomb.position: bomb for bomb in self.bombs}
        obs = {}
        for agent in self.possible_agents:
            obs[agent] = {
                "grid": self._grid_planes(agent, danger, bomb_map),
                "stats": {
                    other: {
                        "alive": self.alive[other],
                        "ammo": self.ammo[other],
                        "blast": self.blast[other],
                        "position": self.positions[other],
                    }
                    for other in self.possible_agents
                },
                "self_id": int(agent.split("_")[-1]),
                "alive_mask": [1.0 if self.alive[a] else 0.0 for a in self.possible_agents],
                "action_mask": self._action_mask(agent),
            }
        return obs

    def _grid_planes(self, agent: AgentID, danger: set[Position], bomb_map: dict[Position, Bomb]) -> dict[str, list[list[float]]]:
        h, w = self.config.height, self.config.width

        def zeros() -> list[list[float]]:
            return [[0.0 for _ in range(w)] for _ in range(h)]

        planes = {
            "wall": zeros(),
            "wood": zeros(),
            "bomb_timer": zeros(),
            "bomb_blast": zeros(),
            "flame": zeros(),
            "bomb_up": zeros(),
            "fire_up": zeros(),
            "danger": zeros(),
        }
        for idx in range(4):
            planes[f"agent_{idx}"] = zeros()
        for r in range(h):
            for c in range(w):
                if self.board[r][c] == self.WALL:
                    planes["wall"][r][c] = 1.0
                elif self.board[r][c] == self.WOOD:
                    planes["wood"][r][c] = 1.0
                if (r, c) in danger:
                    planes["danger"][r][c] = 1.0
        for pos, bomb in bomb_map.items():
            r, c = pos
            planes["bomb_timer"][r][c] = bomb.timer / max(1, self.config.bomb_timer)
            planes["bomb_blast"][r][c] = bomb.blast / 8.0
        for pos, life in self.flames.items():
            r, c = pos
            planes["flame"][r][c] = life / max(1, self.config.flame_life)
        for pos, item in self.powerups.items():
            r, c = pos
            planes[item][r][c] = 1.0
        for idx, other in enumerate(self.possible_agents):
            if self.alive[other]:
                r, c = self.positions[other]
                planes[f"agent_{idx}"][r][c] = 1.0
        return planes

    def _action_mask(self, agent: AgentID) -> list[int]:
        if not self.alive.get(agent, False):
            return [1, 0, 0, 0, 0, 0]
        mask = [1, 0, 0, 0, 0, 0]
        bomb_positions = {bomb.position for bomb in self.bombs}
        for action, direction in DIRS.items():
            nxt = add_pos(self.positions[agent], direction)
            mask[action] = int(self._is_walkable(nxt) and nxt not in bomb_positions)
        mask[5] = int(self._can_place_bomb(agent))
        return mask
