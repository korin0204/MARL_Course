"""Teacher-side Bomber tournament evaluation utilities."""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any

from marl_course.envs.bomber_arena import BomberArenaEnv, BomberRuleBasedPolicy


@dataclass(slots=True)
class BomberPlayer:
    """One tournament entry (student_id + policy object)."""

    student_id: str
    policy: Any
    model_name: str | None = None


def default_bomber_players() -> list[BomberPlayer]:
    """Reference baseline pool used when no submissions are provided."""
    return [
        BomberPlayer(f"rule_{idx}", BomberRuleBasedPolicy(aggression=0.25 + idx * 0.15, seed=idx), model_name=f"rule_based_{idx}")
        for idx in range(4)
    ]


def run_bomber_tournament(
    players: list[BomberPlayer] | None = None,
    episodes: int = 8,
    seed: int = 0,
    live_ascii: bool = False,
    live: bool = False,
    sleep: float = 0.05,
    live_tile_size: int = 48,
) -> dict[str, Any]:
    """Run round-robin Bomber matches and aggregate leaderboard metrics.

    Key output fields:
    - leaderboard: sorted by points/wins
    - standings: per-player raw stats
    - elapsed_sec: runtime for classroom ops visibility
    """
    roster = players or default_bomber_players()
    scores = {player.student_id: {"wins": 0, "games": 0, "survival_steps": 0, "invalid_actions": 0} for player in roster}
    games = []
    viewer = None
    if live:
        from marl_course.visualization.pygame_viewer import PygameGridViewer

        viewer = PygameGridViewer(tile_size=live_tile_size)
    groups = list(itertools.combinations(range(len(roster)), 4)) or [tuple(range(len(roster)))]
    for group_idx, group in enumerate(groups):
        selected = [roster[idx] for idx in group]
        agent_labels = [
            ("ABCD"[env_idx], player.model_name or player.student_id)
            for env_idx, player in enumerate(selected)
        ]
        for ep in range(episodes):
            env = BomberArenaEnv()
            obs, _infos = env.reset(seed=seed + group_idx * 1000 + ep)
            done = False
            while not done:
                actions = {}
                for env_idx, player in enumerate(selected):
                    agent = f"agent_{env_idx}"
                    agent_obs = obs[agent]
                    mask = agent_obs["action_mask"]
                    try:
                        action = int(player.policy.act(agent_obs, action_mask=mask, deterministic=True))
                    except Exception:
                        action = 0
                    if action < 0 or action >= len(mask) or not mask[action]:
                        scores[player.student_id]["invalid_actions"] += 1
                        action = 0
                    actions[agent] = action
                result = env.step(actions)
                obs = result.observations
                done = any(result.terminations.values()) or any(result.truncations.values())
                if live_ascii:
                    print("\033[2J\033[H" + env.render())
                    time.sleep(sleep)
                if viewer is not None:
                    viewer.draw_from_ansi(env.render(), "Bomber Arena Tournament", agent_labels=agent_labels)
                    time.sleep(sleep)
            winner = env.last_winner
            winner_student = None
            if winner is not None:
                winner_student = selected[int(winner.split("_")[-1])].student_id
                scores[winner_student]["wins"] += 1
            for env_idx, player in enumerate(selected):
                scores[player.student_id]["games"] += 1
                scores[player.student_id]["survival_steps"] += env.step_count
            games.append({"players": [p.student_id for p in selected], "winner": winner_student, "steps": env.step_count})
    leaderboard = sorted(
        (
            {
                "student_id": student_id,
                "wins": data["wins"],
                "games": data["games"],
                "win_rate": data["wins"] / max(1, data["games"]),
                "avg_survival_steps": data["survival_steps"] / max(1, data["games"]),
                "invalid_actions": data["invalid_actions"],
            }
            for student_id, data in scores.items()
        ),
        key=lambda row: (row["win_rate"], row["avg_survival_steps"]),
        reverse=True,
    )
    if viewer is not None:
        viewer.close()
    return {"leaderboard": leaderboard, "games": games}
