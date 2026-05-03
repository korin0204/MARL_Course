"""Evaluation helpers for cooperative kitchen tasks (seen + zero-shot)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from marl_course.envs.coop_kitchen import (
    CoopGreedyTeamPolicy,
    CoopKitchenConfig,
    CoopKitchenEnv,
    Layout,
    builtin_layouts,
    generate_layout,
)


@dataclass(slots=True)
class CoopTeam:
    """One submitted team policy entry."""

    student_id: str
    policy: Any
    model_name: str | None = None


def default_coop_teams() -> list[CoopTeam]:
    """Return baseline team(s) when no submissions are provided."""
    return [CoopTeam("greedy_team", CoopGreedyTeamPolicy(), model_name="greedy_baseline")]


def make_team_obs(observations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-agent dicts into one centralized team observation."""
    first = observations["agent_0"]
    return {
        "agent_obs": [observations[f"agent_{idx}"]["agent_obs"] for idx in range(4)],
        "global_state": first["global_state"],
        "agent_features": [observations[f"agent_{idx}"]["agent_features"][f"agent_{idx}"] for idx in range(4)],
        "action_mask": [observations[f"agent_{idx}"]["action_mask"] for idx in range(4)],
        "layout_id": first["layout_id"],
        "valid_cell_mask": first["valid_cell_mask"],
    }


def run_coop_evaluation(
    teams: list[CoopTeam] | None = None,
    episodes: int = 3,
    seed: int = 0,
    layouts: list[Layout] | None = None,
    live_ascii: bool = False,
    live: bool = False,
    sleep: float = 0.05,
    live_tile_size: int = 48,
) -> dict[str, Any]:
    """Evaluate teams on known layouts and return average-score leaderboard."""
    roster = teams or default_coop_teams()
    layout_list = layouts or list(builtin_layouts().values())
    rows = []
    viewer = None
    if live:
        from marl_course.visualization.pygame_viewer import PygameGridViewer

        viewer = PygameGridViewer(tile_size=live_tile_size)
    for team in roster:
        model_name = team.model_name or team.student_id
        agent_labels = [("A", model_name), ("B", model_name), ("E", model_name), ("G", model_name)]
        totals = {"score": 0.0, "soups": 0, "collisions": 0, "invalid_interacts": 0, "episodes": 0}
        for layout_idx, layout in enumerate(layout_list):
            for ep in range(episodes):
                env = CoopKitchenEnv(CoopKitchenConfig(layout_name=layout.name), layout=layout)
                obs, _infos = env.reset(seed=seed + layout_idx * 1000 + ep)
                if hasattr(team.policy, "reset_episode"):
                    team.policy.reset_episode({"layout": layout.name})
                done = False
                while not done:
                    team_obs = make_team_obs(obs)
                    masks = team_obs["action_mask"]
                    try:
                        action_list = list(team.policy.act(team_obs, action_mask=masks, deterministic=True))
                    except Exception:
                        action_list = [4, 4, 4, 4]
                    actions = {}
                    for idx, action in enumerate(action_list[:4]):
                        mask = masks[idx]
                        if action < 0 or action >= len(mask) or not mask[action]:
                            action = 4
                        actions[f"agent_{idx}"] = int(action)
                    result = env.step(actions)
                    obs = result.observations
                    done = any(result.truncations.values())
                    if live_ascii:
                        print("\033[2J\033[H" + env.render())
                        time.sleep(sleep)
                    if viewer is not None:
                        viewer.draw_from_ansi(env.render(), "Coop Kitchen Evaluation", agent_labels=agent_labels)
                        time.sleep(sleep)
                totals["score"] += env.score
                totals["soups"] += env.delivered_soups
                totals["collisions"] += env.collisions
                totals["invalid_interacts"] += env.invalid_interacts
                totals["episodes"] += 1
        rows.append(
            {
                "student_id": team.student_id,
                "avg_score": totals["score"] / max(1, totals["episodes"]),
                "avg_soups": totals["soups"] / max(1, totals["episodes"]),
                "avg_collisions": totals["collisions"] / max(1, totals["episodes"]),
                "avg_invalid_interacts": totals["invalid_interacts"] / max(1, totals["episodes"]),
                "episodes": totals["episodes"],
            }
        )
    rows.sort(key=lambda row: row["avg_score"], reverse=True)
    if viewer is not None:
        viewer.close()
    return {"leaderboard": rows}


def run_coop_zero_shot_evaluation(
    teams: list[CoopTeam] | None = None,
    episodes: int = 2,
    seed: int = 0,
    live_ascii: bool = False,
    live: bool = False,
    sleep: float = 0.05,
    live_tile_size: int = 48,
) -> dict[str, Any]:
    """Evaluate zero-shot generalization across seen/unseen/heldout buckets."""
    seen = list(builtin_layouts().values())
    unseen = [generate_layout(seed + idx, family="corridor") for idx in range(2)]
    heldout = [generate_layout(seed + idx + 100, family="heldout_island") for idx in range(2)]
    buckets = {"seen": seen, "unseen_same_family": unseen, "heldout_family": heldout}
    result = {}
    for name, layouts in buckets.items():
        result[name] = run_coop_evaluation(
            teams,
            episodes=episodes,
            seed=seed,
            layouts=layouts,
            live_ascii=live_ascii,
            live=live,
            sleep=sleep,
            live_tile_size=live_tile_size,
        )["leaderboard"]
    return result
