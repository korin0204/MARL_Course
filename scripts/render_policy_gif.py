"""Regenerate GIFs from trained student policies.

Students can use this script after training to inspect a saved model against
chosen opponents without retraining.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from marl_course.common.submission import load_policy_from_dir
from marl_course.envs.bomber_arena import BomberArenaEnv, BomberRandomPolicy, BomberRuleBasedPolicy, BomberStayPolicy
from marl_course.envs.coop_kitchen import CoopKitchenEnv
from marl_course.evaluation.coop import make_team_obs
from marl_course.visualization.gif import ansi_frames_to_gif


def main() -> None:
    """Load a trained policy, run one rollout, and save a legend-rich GIF."""
    parser = argparse.ArgumentParser(description="Render trained policy rollout as GIF.")
    parser.add_argument("--env", choices=["bomber", "coop"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing policy.py, policy.pt, metadata.json.")
    parser.add_argument("--out", type=Path, help="Output GIF path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=24)
    parser.add_argument(
        "--opponents",
        default="rule,rule,rule",
        help="Bomber opponents for agent_1..3: rule, random, stay, or submission:/path/to/dir.",
    )
    args = parser.parse_args()

    loaded = load_policy_from_dir(args.model_dir)
    out = args.out or args.model_dir / f"{args.env}_custom_rollout.gif"
    if args.env == "bomber":
        opponents, opponent_labels = _make_bomber_opponents(args.opponents, seed=args.seed)
        model_name = str(loaded.metadata.get("model_name", loaded.student_id))
        gif_path = render_bomber_gif(
            student_policy=loaded.policy,
            opponents=opponents,
            agent_labels=[("A", f"student {model_name}")] + opponent_labels,
            out=out,
            seed=args.seed,
            steps=args.steps,
            fps=args.fps,
            tile_size=args.tile_size,
        )
    else:
        model_name = str(loaded.metadata.get("model_name", loaded.student_id))
        gif_path = render_coop_gif(
            team_policy=loaded.policy,
            agent_labels=[("A", f"team {model_name}"), ("B", f"team {model_name}"), ("E", f"team {model_name}"), ("G", f"team {model_name}")],
            out=out,
            seed=args.seed,
            steps=args.steps,
            fps=args.fps,
            tile_size=args.tile_size,
        )
    print({"gif": str(gif_path), "student_id": loaded.student_id})


def render_bomber_gif(
    student_policy: Any,
    opponents: list[Any],
    agent_labels: list[tuple[str, str]],
    out: Path,
    seed: int,
    steps: int,
    fps: int,
    tile_size: int,
) -> Path | None:
    """Render Bomber rollout with trained model as agent_0."""
    env = BomberArenaEnv()
    obs, _infos = env.reset(seed=seed)
    frames = [env.render()]
    done = False
    while not done and len(frames) < steps:
        actions = {
            "agent_0": _safe_act(student_policy, obs["agent_0"], obs["agent_0"]["action_mask"]),
        }
        for idx in range(1, 4):
            opponent = opponents[idx - 1]
            actions[f"agent_{idx}"] = _safe_act(opponent, obs[f"agent_{idx}"], obs[f"agent_{idx}"]["action_mask"])
        result = env.step(actions)
        obs = result.observations
        frames.append(env.render())
        done = any(result.terminations.values()) or any(result.truncations.values())
    return ansi_frames_to_gif(frames, out, tile_size=tile_size, fps=fps, agent_labels=agent_labels)


def render_coop_gif(
    team_policy: Any,
    agent_labels: list[tuple[str, str]],
    out: Path,
    seed: int,
    steps: int,
    fps: int,
    tile_size: int,
) -> Path | None:
    """Render Coop rollout with trained 4-agent team policy."""
    env = CoopKitchenEnv()
    obs, _infos = env.reset(seed=seed)
    if hasattr(team_policy, "reset_episode"):
        team_policy.reset_episode({"layout": env.layout.name})
    frames = [env.render()]
    done = False
    while not done and len(frames) < steps:
        team_obs = make_team_obs(obs)
        try:
            actions = list(team_policy.act(team_obs, team_obs["action_mask"], deterministic=True))
        except TypeError:
            actions = list(team_policy.act(team_obs, action_mask=team_obs["action_mask"], deterministic=True))
        result = env.step({f"agent_{idx}": int(actions[idx]) for idx in range(4)})
        obs = result.observations
        frames.append(env.render())
        done = any(result.truncations.values())
    return ansi_frames_to_gif(frames, out, tile_size=tile_size, fps=fps, agent_labels=agent_labels)


def _make_bomber_opponents(spec: str, seed: int) -> tuple[list[Any], list[tuple[str, str]]]:
    """Build three Bomber opponents from a comma-separated spec."""
    names = [item.strip() for item in spec.split(",") if item.strip()]
    if not names:
        names = ["rule"]
    while len(names) < 3:
        names.append(names[-1])
    opponents = []
    labels = []
    for idx, name in enumerate(names[:3]):
        agent_char = "BCD"[idx]
        if name == "rule":
            opponents.append(BomberRuleBasedPolicy(seed=seed + idx + 1))
            labels.append((agent_char, "rule_based"))
        elif name == "random":
            opponents.append(BomberRandomPolicy(seed=seed + idx + 1))
            labels.append((agent_char, "random_baseline"))
        elif name == "stay":
            opponents.append(BomberStayPolicy())
            labels.append((agent_char, "stay_baseline"))
        elif name.startswith("submission:"):
            loaded = load_policy_from_dir(Path(name.removeprefix("submission:")))
            opponents.append(loaded.policy)
            labels.append((agent_char, str(loaded.metadata.get("model_name", loaded.student_id))))
        else:
            raise ValueError(f"Unknown opponent spec: {name}")
    return opponents, labels


def _safe_act(policy: Any, obs: dict[str, Any], action_mask: list[int]) -> int:
    """Call a policy and fall back to stay if it returns an illegal action."""
    try:
        action = int(policy.act(obs, action_mask, deterministic=True))
    except TypeError:
        action = int(policy.act(obs, action_mask=action_mask, deterministic=True))
    except Exception:
        action = 0
    if action < 0 or action >= len(action_mask) or not action_mask[action]:
        return 0
    return action


if __name__ == "__main__":
    main()
