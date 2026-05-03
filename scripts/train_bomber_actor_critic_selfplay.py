"""Train Bomber Actor-Critic with fixed-opponent self-play.

Actor-Criticはon-policy更新なので、固定相手の世代交代だけを行い、
学習中の相手モデルには勾配を流さない。
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from marl_course.algos.artifacts import record_bomber_policy_gif
from marl_course.algos.bomber_torch import ActorCriticNet, ActorCriticTorchPolicy, infer_obs_shapes, load_actor_critic_policy
from marl_course.algos.logging import MetricLogger
from marl_course.common.config import cli_or_config, dump_effective_config, load_json_config
from marl_course.common.model_naming import default_model_name, validate_model_name
from marl_course.common.schedules import exponential_decay
from marl_course.common.submission import load_policy_from_dir
from marl_course.common.torch_device import configure_torch_runtime, resolve_torch_device, torch_device_summary
from marl_course.envs.bomber_arena import BomberArenaEnv, BomberRuleBasedPolicy
from scripts.train_bomber_actor_critic import discounted_returns, select_action_actor_critic, student_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Bomber Actor-Critic with automatic fixed-opponent self-play.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_bomber_actor_critic.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--opponent-from", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--value-coef", type=float)
    parser.add_argument("--entropy-coef", type=float)
    parser.add_argument("--epsilon-start", type=float)
    parser.add_argument("--epsilon-end", type=float)
    parser.add_argument("--epsilon-decay-rate", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--use-cnn", action=argparse.BooleanOptionalAction)
    parser.add_argument("--device")
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--promotion-win-rate", type=float)
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

    env = BomberArenaEnv()
    obs, _ = env.reset(seed=cfg["seed"])
    obs_dim, grid_shape, stats_dim = infer_obs_shapes(obs["agent_0"])
    payload = _load_payload_or_none(cfg["resume_from"], device)
    use_cnn = bool(payload.get("use_cnn", cfg["use_cnn"])) if payload else bool(cfg["use_cnn"])
    hidden_dim = int(payload.get("hidden_dim", cfg["hidden_dim"])) if payload else int(cfg["hidden_dim"])
    model = ActorCriticNet(obs_dim, hidden_dim=hidden_dim, use_cnn=use_cnn, grid_shape=grid_shape, stats_dim=stats_dim).to(device)
    if payload:
        model.load_state_dict(payload["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    opponent_policy = _initial_opponent(cfg["opponent_from"], model, device, cfg["seed"])
    opponent_generation = 0
    logger = MetricLogger(out, use_wandb=cfg["wandb"], project=cfg["wandb_project"], run_name="bomber-actor-critic-selfplay", config=cfg)
    wins = 0
    moving_win = 0.0

    for ep in range(cfg["episodes"]):
        epsilon = exponential_decay(cfg["epsilon_start"], cfg["epsilon_end"], ep, cfg["episodes"], cfg["epsilon_decay_rate"])
        obs, _ = env.reset(seed=cfg["seed"] + ep)
        done = False
        ep_return = 0.0
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        rewards: list[float] = []
        while not done:
            agent_obs = obs["agent_0"]
            action, log_prob, entropy, value = select_action_actor_critic(model, agent_obs, epsilon=epsilon, device=device)
            actions = {"agent_0": action}
            for idx in range(1, 4):
                actions[f"agent_{idx}"] = _safe_act(opponent_policy, obs[f"agent_{idx}"], obs[f"agent_{idx}"]["action_mask"])
            result = env.step(actions)
            done = any(result.terminations.values()) or any(result.truncations.values())
            reward = student_reward(env, result.infos["agent_0"]["events"])
            ep_return += reward
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
            obs = result.observations

        returns = discounted_returns(rewards, gamma=cfg["gamma"], device=device)
        values_tensor = torch.stack(values)
        log_probs_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)
        advantages = returns - values_tensor
        policy_loss = -(log_probs_tensor * advantages.detach()).mean()
        value_loss = torch.nn.functional.mse_loss(values_tensor, returns)
        entropy_loss = -entropy_tensor.mean()
        total_loss = policy_loss + cfg["value_coef"] * value_loss + cfg["entropy_coef"] * entropy_loss
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        win = int(env.last_winner == "agent_0")
        wins += win
        moving_win = 0.95 * moving_win + 0.05 * win
        eval_win_rate = None
        promoted = False
        if (ep + 1) % cfg["eval_interval"] == 0:
            eval_win_rate = _evaluate(model, opponent_policy, cfg["eval_episodes"], cfg["seed"] + 100000 + ep, device)
            if eval_win_rate >= cfg["promotion_win_rate"]:
                opponent_policy = _frozen_policy(model, device)
                opponent_generation += 1
                promoted = True
                _save_checkpoint(out / f"selfplay_opponent_gen{opponent_generation}.pt", model, obs_dim, hidden_dim, use_cnn, grid_shape, stats_dim)
        logger.log(
            {
                "episode": ep,
                "return": ep_return,
                "win": win,
                "moving_win_rate": moving_win,
                "epsilon": epsilon,
                "loss_total": float(total_loss.item()),
                "entropy": float(entropy_tensor.mean().item()),
                "opponent_generation": opponent_generation,
                "selfplay_eval_win_rate": eval_win_rate if eval_win_rate is not None else -1.0,
                "promoted": int(promoted),
            },
            step=ep,
        )

    _save_checkpoint(out / "policy.pt", model, obs_dim, hidden_dim, use_cnn, grid_shape, stats_dim)
    (out / "metadata.json").write_text(json.dumps({"student_id": cfg["student_id"], "model_name": cfg["model_name"], "env_id": "bomber_arena_v1", "algo": "actor_critic_selfplay", "use_cnn": use_cnn}, indent=2), encoding="utf-8")
    (out / "policy.py").write_text(
        "from marl_course.algos.bomber_torch import load_actor_critic_policy\n\n"
        "def load_policy(model_path, device='cpu'):\n"
        "    return load_actor_critic_policy(model_path, device=device)\n",
        encoding="utf-8",
    )
    gif_path = record_bomber_policy_gif(ActorCriticTorchPolicy(model.eval(), device=device), out, seed=cfg["seed"] + 9999, max_steps=cfg["gif_steps"], fps=cfg["gif_fps"], tile_size=cfg["gif_tile_size"], model_name=cfg["model_name"])
    if gif_path:
        logger.log_gif(gif_path, fps=cfg["gif_fps"])
    logger.close()
    print({"episodes": cfg["episodes"], "wins": wins, "win_rate": wins / max(1, cfg["episodes"]), "opponent_generation": opponent_generation, "out": str(out)})


def _effective_config(args: argparse.Namespace, file_config: dict[str, object]) -> dict[str, Any]:
    student_id = file_config.get("student_id", "bomber_actor_critic_selfplay_student")
    episodes = int(cli_or_config(args, file_config, "episodes", 10000))
    model_name = validate_model_name(cli_or_config(args, file_config, "model_name", default_model_name(student_id, "actorcriticselfplay", episodes)))
    return {
        "student_id": student_id,
        "model_name": model_name,
        "out": str(cli_or_config(args, file_config, "out", "outputs/bomber_actor_critic_selfplay_student")),
        "resume_from": str(cli_or_config(args, file_config, "resume_from", "")),
        "opponent_from": str(cli_or_config(args, file_config, "opponent_from", "")),
        "episodes": episodes,
        "seed": int(cli_or_config(args, file_config, "seed", 0)),
        "gamma": float(cli_or_config(args, file_config, "gamma", 0.99)),
        "lr": float(cli_or_config(args, file_config, "lr", 3.0e-4)),
        "value_coef": float(cli_or_config(args, file_config, "value_coef", 0.5)),
        "entropy_coef": float(cli_or_config(args, file_config, "entropy_coef", 0.01)),
        "epsilon_start": float(cli_or_config(args, file_config, "epsilon_start", 0.0)),
        "epsilon_end": float(cli_or_config(args, file_config, "epsilon_end", 0.0)),
        "epsilon_decay_rate": float(cli_or_config(args, file_config, "epsilon_decay_rate", 8.0)),
        "hidden_dim": int(cli_or_config(args, file_config, "hidden_dim", 256)),
        "use_cnn": bool(cli_or_config(args, file_config, "use_cnn", True)),
        "device": str(cli_or_config(args, file_config, "device", "auto")),
        "eval_interval": int(cli_or_config(args, file_config, "eval_interval", 1000)),
        "eval_episodes": int(cli_or_config(args, file_config, "eval_episodes", 32)),
        "promotion_win_rate": float(cli_or_config(args, file_config, "promotion_win_rate", 0.60)),
        "wandb": bool(cli_or_config(args, file_config, "wandb", False)),
        "wandb_project": str(cli_or_config(args, file_config, "wandb_project", "marl-course-games")),
        "gif_steps": int(cli_or_config(args, file_config, "gif_steps", 200)),
        "gif_fps": int(cli_or_config(args, file_config, "gif_fps", 16)),
        "gif_tile_size": int(cli_or_config(args, file_config, "gif_tile_size", 16)),
    }


def _load_payload_or_none(path_value: object, device: str) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    return torch.load(str(path / "policy.pt" if path.is_dir() else path), map_location=device)


def _initial_opponent(path_value: object, model: ActorCriticNet, device: str, seed: int) -> Any:
    if path_value:
        path = Path(str(path_value))
        if path.is_dir() and (path / "policy.py").exists():
            return load_policy_from_dir(path, device=device).policy
        return load_actor_critic_policy(path / "policy.pt" if path.is_dir() else path, device=device)
    return BomberRuleBasedPolicy(seed=seed + 999)


def _frozen_policy(model: ActorCriticNet, device: str) -> ActorCriticTorchPolicy:
    frozen = copy.deepcopy(model).to(device).eval()
    for param in frozen.parameters():
        param.requires_grad_(False)
    return ActorCriticTorchPolicy(frozen, device=device)


def _evaluate(model: ActorCriticNet, opponent: Any, episodes: int, seed: int, device: str) -> float:
    policy = ActorCriticTorchPolicy(model.eval(), device=device)
    wins = 0
    for ep in range(episodes):
        env = BomberArenaEnv()
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            actions = {"agent_0": _safe_act(policy, obs["agent_0"], obs["agent_0"]["action_mask"])}
            for idx in range(1, 4):
                actions[f"agent_{idx}"] = _safe_act(opponent, obs[f"agent_{idx}"], obs[f"agent_{idx}"]["action_mask"])
            result = env.step(actions)
            obs = result.observations
            done = any(result.terminations.values()) or any(result.truncations.values())
        wins += int(env.last_winner == "agent_0")
    return wins / max(1, episodes)


def _safe_act(policy: Any, obs: dict[str, Any], action_mask: list[int]) -> int:
    try:
        action = int(policy.act(obs, action_mask, deterministic=True))
    except TypeError:
        action = int(policy.act(obs, action_mask=action_mask, deterministic=True))
    except Exception:
        action = 0
    if action < 0 or action >= len(action_mask) or not action_mask[action]:
        return 0
    return action


def _save_checkpoint(path: Path, model: ActorCriticNet, obs_dim: int, hidden_dim: int, use_cnn: bool, grid_shape: tuple[int, int, int], stats_dim: int) -> None:
    torch.save(
        {
            "algo": "actor_critic",
            "obs_dim": obs_dim,
            "n_actions": 6,
            "hidden_dim": hidden_dim,
            "use_cnn": use_cnn,
            "grid_shape": list(grid_shape),
            "stats_dim": stats_dim,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


if __name__ == "__main__":
    main()
