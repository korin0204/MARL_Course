"""Post-training rollout recording helpers.

These functions generate a deterministic replay GIF so students and teachers
can inspect policy behavior and attach media to W&B runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from marl_course.envs.bomber_arena import BomberArenaEnv, BomberRuleBasedPolicy
from marl_course.envs.coop_kitchen import CoopKitchenEnv
from marl_course.evaluation.coop import make_team_obs
from marl_course.visualization.gif import ansi_frames_to_gif


def record_bomber_policy_gif(
    policy: Any,
    out_dir: Path,
    seed: int,
    max_steps: int,
    fps: int,
    tile_size: int,
    model_name: str = "student_model",
) -> Path | None:
    """Record one deterministic post-training Bomber rollout."""

    env = BomberArenaEnv()
    obs, _ = env.reset(seed=seed)
    opponents = [BomberRuleBasedPolicy(seed=seed + idx + 1) for idx in range(3)]
    frames = [env.render()]
    done = False
    while not done and len(frames) < max_steps:
        action = policy.act(obs["agent_0"], obs["agent_0"]["action_mask"], deterministic=True)
        actions = {"agent_0": action}
        for idx in range(1, 4):
            actions[f"agent_{idx}"] = opponents[idx - 1].act(obs[f"agent_{idx}"], obs[f"agent_{idx}"]["action_mask"])
        result = env.step(actions)
        obs = result.observations
        frames.append(env.render())
        done = any(result.terminations.values()) or any(result.truncations.values())
    agent_labels = [
        ("A", f"student {model_name}"),
        ("B", "rule_based"),
        ("C", "rule_based"),
        ("D", "rule_based"),
    ]
    return ansi_frames_to_gif(frames, out_dir / "post_training_episode.gif", tile_size=tile_size, fps=fps, agent_labels=agent_labels)


def record_coop_policy_gif(
    policy: Any,
    out_dir: Path,
    seed: int,
    max_steps: int,
    fps: int,
    tile_size: int,
    model_name: str = "student_model",
) -> Path | None:
    """Record one deterministic post-training Coop Kitchen rollout."""

    env = CoopKitchenEnv()
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset_episode"):
        policy.reset_episode({"layout": env.layout.name})
    frames = [env.render()]
    done = False
    while not done and len(frames) < max_steps:
        team_obs = make_team_obs(obs)
        actions = policy.act(team_obs, team_obs["action_mask"], deterministic=True)
        result = env.step({f"agent_{idx}": int(actions[idx]) for idx in range(4)})
        obs = result.observations
        frames.append(env.render())
        done = any(result.truncations.values())
    agent_labels = [
        ("A", f"team {model_name}"),
        ("B", f"team {model_name}"),
        ("E", f"team {model_name}"),
        ("G", f"team {model_name}"),
    ]
    return ansi_frames_to_gif(frames, out_dir / "post_training_episode.gif", tile_size=tile_size, fps=fps, agent_labels=agent_labels)
