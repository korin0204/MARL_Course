"""Evaluate cooperative kitchen submissions on known layouts.

Instructor-facing script that loads teams from submission folders and computes
scores on standard layouts for grading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from marl_course.common.config import cli_or_config, load_json_config
from marl_course.common.submission import load_policy_from_dir
from marl_course.evaluation.coop import CoopTeam, default_coop_teams, run_coop_evaluation


def load_teams(submissions: Path | None) -> list[CoopTeam] | None:
    """Load coop teams from submission folders if they match coop env metadata."""
    if submissions is None or not submissions.exists():
        return None
    teams = []
    for child in sorted(submissions.iterdir()):
        if child.is_dir() and (child / "policy.py").exists():
            loaded = load_policy_from_dir(child)
            if str(loaded.metadata.get("env_id", "")).startswith("coop_kitchen"):
                teams.append(CoopTeam(loaded.student_id, loaded.policy, model_name=str(loaded.metadata.get("model_name", loaded.student_id))))
    return teams or None


def main() -> None:
    """Run cooperative evaluation and write a leaderboard report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/evaluate_coop.json"))
    parser.add_argument("--submissions", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--live", action=argparse.BooleanOptionalAction)
    parser.add_argument("--live-ascii", action=argparse.BooleanOptionalAction)
    parser.add_argument("--live-sleep", type=float)
    parser.add_argument("--live-tile-size", type=int)
    args = parser.parse_args()
    config = load_json_config(args.config)
    submissions = cli_or_config(args, config, "submissions", None)
    teams = load_teams(Path(submissions) if submissions else None) or default_coop_teams()
    result = run_coop_evaluation(
        teams,
        episodes=int(cli_or_config(args, config, "episodes", 2)),
        seed=int(cli_or_config(args, config, "seed", 0)),
        live_ascii=bool(cli_or_config(args, config, "live_ascii", False)),
        live=bool(cli_or_config(args, config, "live", False)),
        sleep=float(cli_or_config(args, config, "live_sleep", 0.05)),
        live_tile_size=int(cli_or_config(args, config, "live_tile_size", 48)),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
