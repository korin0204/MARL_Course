"""Evaluate submitted Bomber policies in a round-robin tournament.

This script is for instructors. It loads each student submission directory,
runs matches against baseline agents and other submissions, and writes a JSON
report that can be used for grading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from marl_course.common.config import cli_or_config, load_json_config
from marl_course.common.submission import load_policy_from_dir
from marl_course.evaluation.bomber import BomberPlayer, default_bomber_players, run_bomber_tournament


def load_players(submissions: Path | None) -> list[BomberPlayer] | None:
    """Load bomber players from submission directories.

    Each valid directory must contain:
    - `policy.py`: exposes `load_policy(model_path, device=...)`
    - `metadata.json`: includes `env_id` compatible with bomber arena
    """
    if submissions is None or not submissions.exists():
        return None
    players = []
    for child in sorted(submissions.iterdir()):
        if child.is_dir() and (child / "policy.py").exists():
            loaded = load_policy_from_dir(child)
            if str(loaded.metadata.get("env_id", "")).startswith("bomber_arena"):
                players.append(BomberPlayer(loaded.student_id, loaded.policy, model_name=str(loaded.metadata.get("model_name", loaded.student_id))))
    return players or None


def main() -> None:
    """Parse config/CLI, run tournament, and save leaderboard JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/evaluate_bomber.json"))
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
    players = load_players(Path(submissions) if submissions else None) or default_bomber_players()
    result = run_bomber_tournament(
        players,
        episodes=int(cli_or_config(args, config, "episodes", 4)),
        seed=int(cli_or_config(args, config, "seed", 0)),
        live_ascii=bool(cli_or_config(args, config, "live_ascii", False)),
        live=bool(cli_or_config(args, config, "live", False)),
        sleep=float(cli_or_config(args, config, "live_sleep", 0.05)),
        live_tile_size=int(cli_or_config(args, config, "live_tile_size", 48)),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
