"""Train neural QMIX across multiple Coop Kitchen layouts for zero-shot generalization."""
from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path
from typing import Any

import torch

from marl_course.algos.artifacts import record_coop_policy_gif
from marl_course.algos.coop_torch import QMIXNet, QMIXPolicy, infer_coop_shapes
from marl_course.algos.logging import MetricLogger
from marl_course.common.config import cli_or_config, dump_effective_config, load_json_config
from marl_course.common.model_naming import default_model_name, validate_model_name
from marl_course.common.schedules import exponential_decay
from marl_course.common.torch_device import configure_torch_runtime, resolve_torch_device, torch_device_summary
from marl_course.envs.coop_kitchen import CoopKitchenConfig, CoopKitchenEnv, builtin_layouts, generate_layout
from marl_course.evaluation.coop import make_team_obs
from scripts.train_coop_qmix import select_qmix_actions, train_qmix_step
from marl_course.algos.coop_torch import coop_team_obs_to_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural QMIX on mixed generated layouts for zero-shot evaluation.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_coop_zero_shot.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--mixing-dim", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--replay-size", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--train-every", type=int)
    parser.add_argument("--target-update", type=int)
    parser.add_argument("--epsilon-start", type=float)
    parser.add_argument("--epsilon-end", type=float)
    parser.add_argument("--epsilon-decay-rate", type=float)
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
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    base_layout = next(iter(builtin_layouts().values()))
    probe_env = CoopKitchenEnv(CoopKitchenConfig(layout_name=base_layout.name), layout=base_layout)
    probe_obs, _ = probe_env.reset(seed=cfg["seed"])
    obs_shape, state_dim = infer_coop_shapes(make_team_obs(probe_obs))
    online = QMIXNet(obs_shape, state_dim, hidden_dim=cfg["hidden_dim"], mixing_dim=cfg["mixing_dim"]).to(device)
    target = QMIXNet(obs_shape, state_dim, hidden_dim=cfg["hidden_dim"], mixing_dim=cfg["mixing_dim"]).to(device)
    if cfg["resume_from"]:
        payload = torch.load(str(_policy_path(cfg["resume_from"])), map_location=device)
        online.load_state_dict(payload["model_state_dict"])
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=cfg["lr"])
    replay = deque(maxlen=cfg["replay_size"])
    logger = MetricLogger(out, use_wandb=cfg["wandb"], project=cfg["wandb_project"], run_name="overcooked-qmix-zero-shot", config=cfg)
    best_score = 0.0
    total_steps = 0
    seen_layouts = list(builtin_layouts().values())
    families = list(cfg["families"])

    for ep in range(cfg["episodes"]):
        epsilon = exponential_decay(cfg["epsilon_start"], cfg["epsilon_end"], ep, cfg["episodes"], cfg["epsilon_decay_rate"])
        if ep % 4 == 0:
            layout = seen_layouts[(ep // 4) % len(seen_layouts)]
            bucket = "seen"
        else:
            family = families[ep % len(families)]
            layout = generate_layout(cfg["seed"] + ep, family=family)
            bucket = family
        env = CoopKitchenEnv(CoopKitchenConfig(layout_name=layout.name), layout=layout)
        obs, _ = env.reset(seed=cfg["seed"] + ep)
        done = False
        losses: list[float] = []
        prev_score = 0.0
        prev_collisions = 0
        prev_invalid = 0
        while not done:
            team_obs = make_team_obs(obs)
            obs_tensor = coop_team_obs_to_tensor(team_obs, device="cpu")
            masks = torch.tensor(team_obs["action_mask"], dtype=torch.float32)
            actions = select_qmix_actions(online, obs_tensor, masks, epsilon, device)
            result = env.step({f"agent_{idx}": int(actions[idx]) for idx in range(4)})
            next_team_obs = make_team_obs(result.observations)
            reward = (env.score - prev_score) - 0.02 * (env.collisions - prev_collisions) - 0.01 * (env.invalid_interacts - prev_invalid)
            prev_score = env.score
            prev_collisions = env.collisions
            prev_invalid = env.invalid_interacts
            done = any(result.truncations.values())
            replay.append((obs_tensor, torch.tensor(actions, dtype=torch.long), float(reward), coop_team_obs_to_tensor(next_team_obs, device="cpu"), torch.tensor(next_team_obs["action_mask"], dtype=torch.float32), float(done)))
            obs = result.observations
            total_steps += 1
            if len(replay) >= cfg["warmup_steps"] and total_steps % cfg["train_every"] == 0:
                loss = train_qmix_step(online, target, optimizer, replay, cfg, device)
                losses.append(loss)
                if total_steps % cfg["target_update"] == 0:
                    target.load_state_dict(online.state_dict())
        best_score = max(best_score, env.score)
        logger.log(
            {
                "episode": ep,
                "layout_bucket": bucket,
                "layout_name": layout.name,
                "score": env.score,
                "soups": env.delivered_soups,
                "best_score": best_score,
                "epsilon": epsilon,
                "loss": sum(losses) / max(1, len(losses)),
                "replay_size": len(replay),
                "collisions": env.collisions,
                "invalid_interacts": env.invalid_interacts,
            },
            step=ep,
        )

    torch.save({"algo": "qmix", "obs_shape": list(obs_shape), "state_dim": state_dim, "hidden_dim": cfg["hidden_dim"], "mixing_dim": cfg["mixing_dim"], "n_actions": 6, "model_state_dict": online.state_dict()}, out / "policy.pt")
    (out / "metadata.json").write_text(json.dumps({"student_id": cfg["student_id"], "model_name": cfg["model_name"], "env_id": "coop_kitchen_zero_shot_v1", "algo": "qmix_zero_shot"}, indent=2), encoding="utf-8")
    (out / "policy.py").write_text(
        "from marl_course.algos.coop_torch import load_qmix_policy\n\n"
        "def load_policy(model_path, device='cpu'):\n"
        "    return load_qmix_policy(model_path, device=device)\n",
        encoding="utf-8",
    )
    policy = QMIXPolicy(model=online.eval(), device=device)
    gif_path = record_coop_policy_gif(policy, out, seed=cfg["seed"] + 9999, max_steps=cfg["gif_steps"], fps=cfg["gif_fps"], tile_size=cfg["gif_tile_size"], model_name=cfg["model_name"])
    if gif_path is not None:
        logger.log_gif(gif_path, fps=cfg["gif_fps"])
    logger.close()
    print({"episodes": cfg["episodes"], "best_score": best_score, "out": str(out), "gif": str(gif_path) if gif_path else None})


def _effective_config(args: argparse.Namespace, file_config: dict[str, object]) -> dict[str, Any]:
    student_id = file_config.get("student_id", "coop_qmix_zero_shot_student")
    episodes = int(cli_or_config(args, file_config, "episodes", 160))
    model_name = validate_model_name(cli_or_config(args, file_config, "model_name", default_model_name(student_id, "qmixzeroshot", episodes)))
    return {
        "student_id": student_id,
        "model_name": model_name,
        "out": str(cli_or_config(args, file_config, "out", "outputs/coop_qmix_zero_shot_student")),
        "resume_from": str(cli_or_config(args, file_config, "resume_from", "")),
        "episodes": episodes,
        "seed": int(cli_or_config(args, file_config, "seed", 2)),
        "gamma": float(cli_or_config(args, file_config, "gamma", 0.99)),
        "lr": float(cli_or_config(args, file_config, "lr", 5.0e-4)),
        "hidden_dim": int(cli_or_config(args, file_config, "hidden_dim", 128)),
        "mixing_dim": int(cli_or_config(args, file_config, "mixing_dim", 32)),
        "batch_size": int(cli_or_config(args, file_config, "batch_size", 64)),
        "replay_size": int(cli_or_config(args, file_config, "replay_size", 50000)),
        "warmup_steps": int(cli_or_config(args, file_config, "warmup_steps", 256)),
        "train_every": int(cli_or_config(args, file_config, "train_every", 1)),
        "target_update": int(cli_or_config(args, file_config, "target_update", 500)),
        "epsilon_start": float(cli_or_config(args, file_config, "epsilon_start", 0.8)),
        "epsilon_end": float(cli_or_config(args, file_config, "epsilon_end", 0.05)),
        "epsilon_decay_rate": float(cli_or_config(args, file_config, "epsilon_decay_rate", 5.0)),
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
