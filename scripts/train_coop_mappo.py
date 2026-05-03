"""Train Overcooked/Coop Kitchen with neural MAPPO.

完全協調タスクとして、4エージェントが共有actorを使い、criticは4人分の
観測をまとめて見る centralized critic にする。旧MAPPO-liteは
`_old/scripts/train_coop_mappo_lite.py` に退避済み。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from marl_course.algos.artifacts import record_coop_policy_gif
from marl_course.algos.coop_torch import MAPPOActorCritic, MAPPOPolicy, coop_team_obs_to_tensor, infer_coop_shapes, masked_logits
from marl_course.algos.logging import MetricLogger
from marl_course.common.config import cli_or_config, dump_effective_config, load_json_config
from marl_course.common.model_naming import default_model_name, validate_model_name
from marl_course.common.torch_device import configure_torch_runtime, resolve_torch_device, torch_device_summary
from marl_course.envs.coop_kitchen import CoopKitchenConfig, CoopKitchenEnv, builtin_layouts
from marl_course.evaluation.coop import make_team_obs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural MAPPO on one Coop Kitchen map.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_coop_mappo.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--layout-name")
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

    layouts = builtin_layouts()
    layout = layouts[cfg["layout_name"]]
    probe_env = CoopKitchenEnv(CoopKitchenConfig(layout_name=layout.name), layout=layout)
    probe_obs, _ = probe_env.reset(seed=cfg["seed"])
    obs_shape, _state_dim = infer_coop_shapes(make_team_obs(probe_obs))
    model = MAPPOActorCritic(obs_shape=obs_shape, hidden_dim=cfg["hidden_dim"]).to(device)
    if cfg["resume_from"]:
        payload = torch.load(str(_policy_path(cfg["resume_from"])), map_location=device)
        model.load_state_dict(payload["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    logger = MetricLogger(out, use_wandb=cfg["wandb"], project=cfg["wandb_project"], run_name="overcooked-mappo", config=cfg)
    best_score = 0.0

    for ep in range(cfg["episodes"]):
        rollout = collect_rollout(model, layout, cfg["seed"] + ep, device)
        loss_info = update_mappo(model, optimizer, rollout, cfg)
        best_score = max(best_score, rollout["score"])
        logger.log(
            {
                "episode": ep,
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
    (out / "metadata.json").write_text(json.dumps({"student_id": cfg["student_id"], "model_name": cfg["model_name"], "env_id": "coop_kitchen_v1", "algo": "mappo"}, indent=2), encoding="utf-8")
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


def collect_rollout(model: MAPPOActorCritic, layout: Any, seed: int, device: str) -> dict[str, Any]:
    """1エピソード分のMAPPO軌跡を収集する。"""

    env = CoopKitchenEnv(CoopKitchenConfig(layout_name=layout.name), layout=layout)
    obs, _ = env.reset(seed=seed)
    data = {key: [] for key in ["obs", "masks", "actions", "log_probs", "values", "rewards", "dones"]}
    done = False
    prev_score = 0.0
    prev_collisions = 0
    prev_invalid = 0
    while not done:
        team_obs = make_team_obs(obs)
        obs_tensor = coop_team_obs_to_tensor(team_obs, device=device)
        masks = torch.tensor(team_obs["action_mask"], dtype=torch.float32, device=device)
        logits = masked_logits(model.forward_actor(obs_tensor), masks)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        value = model.forward_value(obs_tensor)
        result = env.step({f"agent_{idx}": int(actions[idx].item()) for idx in range(4)})
        reward = (env.score - prev_score) - 0.02 * (env.collisions - prev_collisions) - 0.01 * (env.invalid_interacts - prev_invalid)
        prev_score = env.score
        prev_collisions = env.collisions
        prev_invalid = env.invalid_interacts
        done = any(result.truncations.values())
        data["obs"].append(obs_tensor.detach().cpu())
        data["masks"].append(masks.detach().cpu())
        data["actions"].append(actions.detach().cpu())
        data["log_probs"].append(dist.log_prob(actions).sum().detach().cpu())
        data["values"].append(value.detach().cpu())
        data["rewards"].append(float(reward))
        data["dones"].append(float(done))
        obs = result.observations
    data["score"] = env.score
    data["soups"] = env.delivered_soups
    data["collisions"] = env.collisions
    data["invalid_interacts"] = env.invalid_interacts
    return data


def update_mappo(model: MAPPOActorCritic, optimizer: torch.optim.Optimizer, rollout: dict[str, Any], cfg: dict[str, Any]) -> dict[str, float]:
    """PPO clippingで共有actorと集中criticを更新する。"""

    device = cfg["device"]
    obs = torch.stack(rollout["obs"]).to(device)
    masks = torch.stack(rollout["masks"]).to(device)
    actions = torch.stack(rollout["actions"]).to(device)
    old_log_probs = torch.stack(rollout["log_probs"]).to(device)
    old_values = torch.stack(rollout["values"]).to(device)
    rewards = torch.tensor(rollout["rewards"], dtype=torch.float32, device=device)
    returns, advantages = generalized_advantage(rewards, old_values, cfg["gamma"], cfg["gae_lambda"])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
    last_info = {"loss_total": 0.0, "loss_policy": 0.0, "loss_value": 0.0, "entropy": 0.0}
    for _ in range(cfg["ppo_epochs"]):
        logits = masked_logits(model.forward_actor(obs), masks)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1).mean()
        values = model.forward_value(obs)
        ratio = torch.exp(log_probs - old_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - cfg["clip_eps"], 1.0 + cfg["clip_eps"]) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns)
        loss = policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        last_info = {"loss_total": float(loss.item()), "loss_policy": float(policy_loss.item()), "loss_value": float(value_loss.item()), "entropy": float(entropy.item())}
    return last_info


def generalized_advantage(rewards: torch.Tensor, values: torch.Tensor, gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    next_value = torch.tensor(0.0, device=rewards.device)
    gae = torch.tensor(0.0, device=rewards.device)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
        next_value = values[t]
    return returns, advantages


def _effective_config(args: argparse.Namespace, file_config: dict[str, object]) -> dict[str, Any]:
    student_id = file_config.get("student_id", "coop_mappo_student")
    episodes = int(cli_or_config(args, file_config, "episodes", 120))
    model_name = validate_model_name(cli_or_config(args, file_config, "model_name", default_model_name(student_id, "mappo", episodes)))
    return {
        "student_id": student_id,
        "model_name": model_name,
        "out": str(cli_or_config(args, file_config, "out", "outputs/coop_mappo_student")),
        "resume_from": str(cli_or_config(args, file_config, "resume_from", "")),
        "layout_name": str(cli_or_config(args, file_config, "layout_name", "open_kitchen_4p")),
        "episodes": episodes,
        "seed": int(cli_or_config(args, file_config, "seed", 0)),
        "gamma": float(cli_or_config(args, file_config, "gamma", 0.99)),
        "gae_lambda": float(cli_or_config(args, file_config, "gae_lambda", 0.95)),
        "lr": float(cli_or_config(args, file_config, "lr", 3.0e-4)),
        "hidden_dim": int(cli_or_config(args, file_config, "hidden_dim", 128)),
        "ppo_epochs": int(cli_or_config(args, file_config, "ppo_epochs", 4)),
        "clip_eps": float(cli_or_config(args, file_config, "clip_eps", 0.2)),
        "value_coef": float(cli_or_config(args, file_config, "value_coef", 0.5)),
        "entropy_coef": float(cli_or_config(args, file_config, "entropy_coef", 0.01)),
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
