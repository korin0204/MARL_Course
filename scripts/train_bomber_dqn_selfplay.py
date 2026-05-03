"""Train Bomber DQN with fixed-opponent self-play.

通常のDQN学習はルールベース相手で始める。self-play版では、一定間隔で
現在モデルを評価し、勝率が閾値を超えたら「次の固定対戦相手」として昇格する。
相手は学習中に勾配更新されないため、Independent Learnerの自己対戦として扱いやすい。
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import deque
from pathlib import Path
from typing import Any

import torch

from marl_course.algos.artifacts import record_bomber_policy_gif
from marl_course.algos.bomber_torch import DQNNet, DQNTorchPolicy, bomber_obs_to_tensor, infer_obs_shapes, load_dqn_policy
from marl_course.algos.logging import MetricLogger
from marl_course.common.config import cli_or_config, dump_effective_config, load_json_config
from marl_course.common.model_naming import default_model_name, validate_model_name
from marl_course.common.schedules import exponential_decay
from marl_course.common.submission import load_policy_from_dir
from marl_course.common.torch_device import configure_torch_runtime, resolve_torch_device, torch_device_summary
from marl_course.envs.bomber_arena import BomberArenaEnv, BomberRandomPolicy, BomberRuleBasedPolicy
from scripts.train_bomber_dqn import select_action_dqn, student_reward, train_dqn_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Bomber DQN with automatic fixed-opponent self-play.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_bomber_dqn.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--opponent-from", type=Path, help="既存モデル/提出ディレクトリを初期固定相手にする。")
    parser.add_argument("--model-name")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--replay-size", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--train-every", type=int)
    parser.add_argument("--target-update", type=int)
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

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    env = BomberArenaEnv()
    obs, _ = env.reset(seed=cfg["seed"])
    obs_dim, grid_shape, stats_dim = infer_obs_shapes(obs["agent_0"])

    resume_payload = _load_payload_or_none(cfg["resume_from"], device)
    use_cnn = bool(resume_payload.get("use_cnn", cfg["use_cnn"])) if resume_payload else bool(cfg["use_cnn"])
    hidden_dim = int(resume_payload.get("hidden_dim", cfg["hidden_dim"])) if resume_payload else int(cfg["hidden_dim"])
    if resume_payload and resume_payload.get("grid_shape"):
        grid_shape = tuple(resume_payload["grid_shape"])
    if resume_payload and resume_payload.get("stats_dim") is not None:
        stats_dim = int(resume_payload["stats_dim"])

    online = DQNNet(obs_dim, hidden_dim=hidden_dim, use_cnn=use_cnn, grid_shape=grid_shape, stats_dim=stats_dim).to(device)
    target = DQNNet(obs_dim, hidden_dim=hidden_dim, use_cnn=use_cnn, grid_shape=grid_shape, stats_dim=stats_dim).to(device)
    if resume_payload:
        online.load_state_dict(resume_payload["model_state_dict"])
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=cfg["lr"])
    replay = deque(maxlen=cfg["replay_size"])

    opponent_policy = _initial_opponent(cfg["opponent_from"], online, device, seed=cfg["seed"])
    opponent_generation = 0
    logger = MetricLogger(out, use_wandb=cfg["wandb"], project=cfg["wandb_project"], run_name="bomber-dqn-selfplay", config=cfg)
    wins = 0
    moving_win = 0.0
    total_steps = 0

    for ep in range(cfg["episodes"]):
        epsilon = exponential_decay(cfg["epsilon_start"], cfg["epsilon_end"], ep, cfg["episodes"], cfg["epsilon_decay_rate"])
        obs, _ = env.reset(seed=cfg["seed"] + ep)
        done = False
        ep_return = 0.0
        losses: list[float] = []
        while not done:
            agent_obs = obs["agent_0"]
            action = select_action_dqn(online, agent_obs, agent_obs["action_mask"], epsilon=epsilon, device=device)
            actions = {"agent_0": action}
            for idx in range(1, 4):
                actions[f"agent_{idx}"] = _safe_act(opponent_policy, obs[f"agent_{idx}"], obs[f"agent_{idx}"]["action_mask"])
            result = env.step(actions)
            done = any(result.terminations.values()) or any(result.truncations.values())
            reward = student_reward(env, result.infos["agent_0"]["events"])
            ep_return += reward
            replay.append(
                (
                    bomber_obs_to_tensor(agent_obs, device="cpu"),
                    int(action),
                    float(reward),
                    bomber_obs_to_tensor(result.observations["agent_0"], device="cpu"),
                    float(done),
                    torch.tensor(result.observations["agent_0"]["action_mask"], dtype=torch.float32),
                )
            )
            obs = result.observations
            total_steps += 1
            if len(replay) >= cfg["warmup_steps"] and total_steps % cfg["train_every"] == 0:
                loss = train_dqn_step(online, target, optimizer, replay, cfg["batch_size"], cfg["gamma"], device)
                losses.append(loss)
                if total_steps % cfg["target_update"] == 0:
                    target.load_state_dict(online.state_dict())

        win = int(env.last_winner == "agent_0")
        wins += win
        moving_win = 0.95 * moving_win + 0.05 * win
        eval_win_rate = None
        promoted = False
        if (ep + 1) % cfg["eval_interval"] == 0:
            eval_win_rate = _evaluate_dqn(online, opponent_policy, cfg["eval_episodes"], cfg["seed"] + 100000 + ep, device)
            if eval_win_rate >= cfg["promotion_win_rate"]:
                opponent_policy = _frozen_dqn_policy(online, device)
                opponent_generation += 1
                promoted = True
                _save_checkpoint(out / f"selfplay_opponent_gen{opponent_generation}.pt", online, obs_dim, hidden_dim, use_cnn, grid_shape, stats_dim)
        logger.log(
            {
                "episode": ep,
                "return": ep_return,
                "win": win,
                "moving_win_rate": moving_win,
                "epsilon": epsilon,
                "loss": sum(losses) / max(1, len(losses)),
                "replay_size": len(replay),
                "opponent_generation": opponent_generation,
                "selfplay_eval_win_rate": eval_win_rate if eval_win_rate is not None else -1.0,
                "promoted": int(promoted),
            },
            step=ep,
        )

    _save_checkpoint(out / "policy.pt", online, obs_dim, hidden_dim, use_cnn, grid_shape, stats_dim)
    (out / "metadata.json").write_text(
        json.dumps({"student_id": cfg["student_id"], "model_name": cfg["model_name"], "env_id": "bomber_arena_v1", "algo": "dqn_selfplay", "use_cnn": use_cnn}, indent=2),
        encoding="utf-8",
    )
    (out / "policy.py").write_text(
        "from marl_course.algos.bomber_torch import load_dqn_policy\n\n"
        "def load_policy(model_path, device='cpu'):\n"
        "    return load_dqn_policy(model_path, device=device)\n",
        encoding="utf-8",
    )
    gif_path = record_bomber_policy_gif(DQNTorchPolicy(online.eval(), device=device), out, seed=cfg["seed"] + 9999, max_steps=cfg["gif_steps"], fps=cfg["gif_fps"], tile_size=cfg["gif_tile_size"], model_name=cfg["model_name"])
    if gif_path is not None:
        logger.log_gif(gif_path, fps=cfg["gif_fps"])
    logger.close()
    print({"episodes": cfg["episodes"], "wins": wins, "win_rate": wins / max(1, cfg["episodes"]), "opponent_generation": opponent_generation, "out": str(out)})


def _effective_config(args: argparse.Namespace, file_config: dict[str, object]) -> dict[str, Any]:
    student_id = file_config.get("student_id", "bomber_dqn_selfplay_student")
    episodes = int(cli_or_config(args, file_config, "episodes", 10000))
    model_name = validate_model_name(cli_or_config(args, file_config, "model_name", default_model_name(student_id, "dqnselfplay", episodes)))
    return {
        "student_id": student_id,
        "model_name": model_name,
        "out": str(cli_or_config(args, file_config, "out", "outputs/bomber_dqn_selfplay_student")),
        "resume_from": str(cli_or_config(args, file_config, "resume_from", "")),
        "opponent_from": str(cli_or_config(args, file_config, "opponent_from", "")),
        "episodes": episodes,
        "seed": int(cli_or_config(args, file_config, "seed", 0)),
        "gamma": float(cli_or_config(args, file_config, "gamma", 0.99)),
        "lr": float(cli_or_config(args, file_config, "lr", 1.0e-3)),
        "batch_size": int(cli_or_config(args, file_config, "batch_size", 64)),
        "replay_size": int(cli_or_config(args, file_config, "replay_size", 50000)),
        "warmup_steps": int(cli_or_config(args, file_config, "warmup_steps", 1000)),
        "train_every": int(cli_or_config(args, file_config, "train_every", 1)),
        "target_update": int(cli_or_config(args, file_config, "target_update", 500)),
        "epsilon_start": float(cli_or_config(args, file_config, "epsilon_start", 0.4)),
        "epsilon_end": float(cli_or_config(args, file_config, "epsilon_end", 0.02)),
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
    model_path = path / "policy.pt" if path.is_dir() else path
    return torch.load(str(model_path), map_location=device)


def _initial_opponent(path_value: object, online: DQNNet, device: str, seed: int) -> Any:
    if path_value:
        path = Path(str(path_value))
        if path.is_dir() and (path / "policy.py").exists():
            return load_policy_from_dir(path, device=device).policy
        return load_dqn_policy(path / "policy.pt" if path.is_dir() else path, device=device)
    if seed % 2 == 0:
        return BomberRuleBasedPolicy(seed=seed + 999)
    return BomberRandomPolicy(seed=seed + 999)


def _frozen_dqn_policy(model: DQNNet, device: str) -> DQNTorchPolicy:
    frozen = copy.deepcopy(model).to(device).eval()
    for param in frozen.parameters():
        param.requires_grad_(False)
    return DQNTorchPolicy(model=frozen, device=device)


def _evaluate_dqn(model: DQNNet, opponent: Any, episodes: int, seed: int, device: str) -> float:
    policy = DQNTorchPolicy(model=model.eval(), device=device)
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


def _save_checkpoint(path: Path, model: DQNNet, obs_dim: int, hidden_dim: int, use_cnn: bool, grid_shape: tuple[int, int, int], stats_dim: int) -> None:
    torch.save(
        {
            "algo": "dqn",
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
