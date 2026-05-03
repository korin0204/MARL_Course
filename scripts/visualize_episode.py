"""Preview one episode for Bomber or Coop with ASCII / pygame rendering.

Useful in class: instructors can project live behavior, and students can
inspect policy behavior step-by-step with configurable delay.
"""
from __future__ import annotations

import argparse
import time

from marl_course.envs.bomber_arena import BomberArenaEnv, BomberRuleBasedPolicy
from marl_course.envs.coop_kitchen import CoopGreedyTeamPolicy, CoopKitchenEnv
from marl_course.evaluation.coop import make_team_obs
from marl_course.visualization.pygame_viewer import PygameGridViewer


def show_bomber(steps: int, seed: int, ascii_only: bool) -> None:
    """Render a Bomber match with rule-based policies.

    Args:
        steps: Maximum simulation steps to show.
        seed: Random seed for deterministic replay.
        ascii_only: If True, render in terminal only (no pygame window).
    """
    env = BomberArenaEnv()
    obs, _ = env.reset(seed=seed)
    policies = [BomberRuleBasedPolicy(seed=idx) for idx in range(4)]
    viewer = None if ascii_only else PygameGridViewer()
    for _ in range(steps):
        actions = {
            f"agent_{idx}": policies[idx].act(obs[f"agent_{idx}"], obs[f"agent_{idx}"]["action_mask"])
            for idx in range(4)
        }
        result = env.step(actions)
        obs = result.observations
        frame = env.render()
        if ascii_only:
            print("\033[2J\033[H" + frame)
        else:
            viewer.draw_from_ansi(frame, "Bomber Arena")
        time.sleep(0.08)
        if any(result.terminations.values()) or any(result.truncations.values()):
            break
    if viewer:
        viewer.close()


def show_coop(steps: int, seed: int, ascii_only: bool) -> None:
    """Render a Coop Kitchen rollout with the greedy baseline team."""
    env = CoopKitchenEnv()
    obs, _ = env.reset(seed=seed)
    policy = CoopGreedyTeamPolicy()
    viewer = None if ascii_only else PygameGridViewer()
    for _ in range(steps):
        team_obs = make_team_obs(obs)
        acts = policy.act(team_obs, team_obs["action_mask"])
        result = env.step({f"agent_{idx}": acts[idx] for idx in range(4)})
        obs = result.observations
        frame = env.render()
        if ascii_only:
            print("\033[2J\033[H" + frame)
        else:
            viewer.draw_from_ansi(frame, "Coop Kitchen")
        time.sleep(0.08)
    if viewer:
        viewer.close()


def main() -> None:
    """Select environment and rendering mode from CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["bomber", "coop"], default="bomber")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ascii", action="store_true")
    args = parser.parse_args()
    if args.env == "bomber":
        show_bomber(args.steps, args.seed, args.ascii)
    else:
        show_coop(args.steps, args.seed, args.ascii)


if __name__ == "__main__":
    main()
