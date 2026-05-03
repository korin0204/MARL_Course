"""Train neural MAPPO across multiple Coop Kitchen layouts for zero-shot generalization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from marl_course.algos.artifacts import record_coop_policy_gif
from marl_course.algos.coop_torch import MAPPOActorCritic, MAPPOPolicy, infer_coop_shapes
from marl_course.algos.logging import MetricLogger
from marl_course.common.config import cli_or_config, dump_effective_config, load_json_config
from marl_course.common.model_naming import default_model_name, validate_model_name
from marl_course.common.torch_device import configure_torch_runtime, resolve_torch_device, torch_device_summary
from marl_course.envs.coop_kitchen import CoopKitchenConfig, CoopKitchenEnv, builtin_layouts, generate_layout
from marl_course.evaluation.coop import make_team_obs
from scripts.train_coop_mappo import collect_rollout, update_mappo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural MAPPO on mixed generated layouts for zero-shot evaluation.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_coop_zero_shot.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--ppo-epochs", type=int)
    parser.add_argument("--clip-eps", type=float)
    parser.add_argument("--value-coef", type=float)
    parser.add_argument("--entropy-coef", type=float)
    parser.add_argument("--device")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb-project")
    parser.add_argument("--gif-steps", type=int)
    parser.add_argument("--gif-fps", type=int)
    parser.add_argument("--gif-tile-size", type=int)
    args = parser.parse_args()

    cfg = _effective_config(args, load_json_config(args.config))
    device = configure_torch_runtime(resolve_torch_device(cfg["device"]))
    cfg["device"] = device
    cfg["torch_device"] = torch_device_summary(device)
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    dump_effective_config(out, cfg)
    torch.manual_seed(cfg["seed"])

    base_layout = next(iter(builtin_layouts().values()))
    probe_env = CoopKitchenEnv(CoopKitchenConfig(layout_name=base_layout.name), layout=base_layout)
    probe_obs, _ = probe_env.reset(seed=cfg["seed"])
    obs_shape, _ = infer_coop_shapes(make_team_obs(probe_obs))
    model = MAPPOActorCritic(obs_shape=obs_shape, hidden_dim=cfg["hidden_dim"]).to(device)
    if cfg["resume_from"]:
        payload = torch.load(str(_policy_path(cfg["resume_from"])), map_location=device)
        model.load_state_dict(payload["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    logger = MetricLogger(out, use_wandb=cfg["wandb"], project=cfg["wandb_project"], run_name="overcooked-mappo-zero-shot", config=cfg)
    best_score = 0.0
    seen_layouts = list(builtin_layouts().values())
    families = list(cfg["families"])

    for ep in range(cfg["episodes"]):
        if ep % 4 == 0:
            layout = seen_layouts[(ep // 4) % len(seen_layouts)]
            bucket = "seen"
        else:
            family = families[ep % len(families)]
            layout = generate_layout(cfg["seed"] + ep, family=family)
            bucket = family
        rollout = collect_rollout(model, layout, cfg["seed"] + ep, device)
        loss_info = update_mappo(model, optimizer, rollout, cfg)
        best_score = max(best_score, rollout["score"])
        logger.log(
            {
                "episode": ep,
                "layout_bucket": bucket,
                "layout_name": layout.name,
                "score": rollout["score"],
                "soups": rollout["soups"],
                "best_score": best_score,
                "steps": len(rollout["rewards"]),
                "loss_total": loss_info["loss_total"],
                "loss_policy": loss_info["loss_policy"],
                "loss_value": loss_info["loss_value"],
                "entropy": loss_info["entropy"],
                "collisions": rollout["collisions"],
                "invalid_interacts": rollout["invalid_interacts"],
            },
            step=ep,
        )

    torch.save({"algo": "mappo", "obs_shape": list(obs_shape), "hidden_dim": cfg["hidden_dim"], "n_actions": 6, "model_state_dict": model.state_dict()}, out / "policy.pt")
    (out / "metadata.json").write_text(json.dumps({"student_id": cfg["student_id"], "model_name": cfg["model_name"], "env_id": "coop_kitchen_zero_shot_v1", "algo": "mappo_zero_shot"}, indent=2), encoding="utf-8")
    (out / "policy.py").write_text(
        "from marl_course.algos.coop_torch import load_mappo_policy\n\n"
        "def load_policy(model_path, device='cpu'):\n"
        "    return load_mappo_policy(model_path, device=device)\n",
        encoding="utf-8",
    )
    policy = MAPPOPolicy(model=model.eval(), device=device)
    gif_path = record_coop_policy_gif(policy, out, seed=cfg["seed"] + 9999, max_steps=cfg["gif_steps"], fps=cfg["gif_fps"], tile_size=cfg["gif_tile_size"], model_name=cfg["model_name"])
    if gif_path is not None:
        logger.log_gif(gif_path, fps=cfg["gif_fps"])
    logger.close()
    print({"episodes": cfg["episodes"], "best_score": best_score, "out": str(out), "gif": str(gif_path) if gif_path else None})


def _effective_config(args: argparse.Namespace, file_config: dict[str, object]) -> dict[str, Any]:
    student_id = file_config.get("student_id", "coop_zero_shot_student")
    episodes = int(cli_or_config(args, file_config, "episodes", 160))
    model_name = validate_model_name(cli_or_config(args, file_config, "model_name", default_model_name(student_id, "mappozeroshot", episodes)))
    return {
        "student_id": student_id,
        "model_name": model_name,
        "out": str(cli_or_config(args, file_config, "out", "outputs/coop_mappo_zero_shot_student")),
        "resume_from": str(cli_or_config(args, file_config, "resume_from", "")),
        "episodes": episodes,
        "seed": int(cli_or_config(args, file_config, "seed", 2)),
        "gamma": float(cli_or_config(args, file_config, "gamma", 0.99)),
        "gae_lambda": float(cli_or_config(args, file_config, "gae_lambda", 0.95)),
        "lr": float(cli_or_config(args, file_config, "lr", 3.0e-4)),
        "hidden_dim": int(cli_or_config(args, file_config, "hidden_dim", 128)),
        "ppo_epochs": int(cli_or_config(args, file_config, "ppo_epochs", 4)),
        "clip_eps": float(cli_or_config(args, file_config, "clip_eps", 0.2)),
        "value_coef": float(cli_or_config(args, file_config, "value_coef", 0.5)),
        "entropy_coef": float(cli_or_config(args, file_config, "entropy_coef", 0.01)),
        "families": file_config.get("families", ["open", "corridor", "bottleneck", "island", "long_delivery"]),
        "device": str(cli_or_config(args, file_config, "device", "auto")),
        "wandb": bool(cli_or_config(args, file_config, "wandb", False)),
        "wandb_project": str(cli_or_config(args, file_config, "wandb_project", "marl-course-games")),
        "gif_steps": int(cli_or_config(args, file_config, "gif_steps", 120)),
        "gif_fps": int(cli_or_config(args, file_config, "gif_fps", 8)),
        "gif_tile_size": int(cli_or_config(args, file_config, "gif_tile_size", 16)),
    }


def _policy_path(value: object) -> Path:
    path = Path(str(value))
    return path / "policy.pt" if path.is_dir() else path


if __name__ == "__main__":
    main()
