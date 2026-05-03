"""Train Bomber student policy with actor-critic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from marl_course.algos.artifacts import record_bomber_policy_gif
from marl_course.algos.bomber_torch import ActorCriticNet, ActorCriticTorchPolicy, bomber_obs_to_tensor, infer_obs_shapes, masked_logits
from marl_course.algos.logging import MetricLogger
from marl_course.common.config import cli_or_config, dump_effective_config, load_json_config
from marl_course.common.model_naming import default_model_name, validate_model_name
from marl_course.common.schedules import exponential_decay
from marl_course.common.torch_device import configure_torch_runtime, resolve_torch_device, torch_device_summary
from marl_course.envs.bomber_arena import BomberArenaEnv, BomberRuleBasedPolicy


def main() -> None:
    # Actor-Critic版の学習スクリプト:
    # 方策(actor)と価値関数(critic)を同時に学習する。
    parser = argparse.ArgumentParser(description="Train a Bomber Arena Actor-Critic student.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_bomber_actor_critic.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume-from", type=Path)
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
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb-project")
    parser.add_argument("--gif-steps", type=int)
    parser.add_argument("--gif-fps", type=int)
    parser.add_argument("--gif-tile-size", type=int)
    args = parser.parse_args()

    cfg = _effective_config(args, load_json_config(args.config))
    # device=autoなら、Apple Silicon MacではMPS、Colab等ではCUDA、なければCPUを選ぶ。
    # 実際に選ばれたdeviceはeffective_config/W&Bにも保存される。
    device = configure_torch_runtime(resolve_torch_device(cfg["device"]))
    cfg["device"] = device
    cfg["torch_device"] = torch_device_summary(device)
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    dump_effective_config(out, cfg)

    torch.manual_seed(cfg["seed"])

    # 観測次元を調べてネットワーク初期化。
    env = BomberArenaEnv()
    obs, _ = env.reset(seed=cfg["seed"])
    obs_dim, grid_shape, stats_dim = infer_obs_shapes(obs["agent_0"])
    resume_from = policy_path_or_none(cfg["resume_from"])
    resume_payload = torch.load(str(resume_from), map_location=device) if resume_from is not None else None
    use_cnn = bool(resume_payload.get("use_cnn", cfg["use_cnn"])) if resume_payload else bool(cfg["use_cnn"])
    hidden_dim = int(resume_payload.get("hidden_dim", cfg["hidden_dim"])) if resume_payload else int(cfg["hidden_dim"])
    if resume_payload and resume_payload.get("grid_shape"):
        grid_shape = tuple(resume_payload["grid_shape"])
    if resume_payload and resume_payload.get("stats_dim") is not None:
        stats_dim = int(resume_payload["stats_dim"])
    model = ActorCriticNet(obs_dim=obs_dim, hidden_dim=hidden_dim, use_cnn=use_cnn, grid_shape=grid_shape, stats_dim=stats_dim).to(device)
    if resume_from is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    logger = MetricLogger(out, use_wandb=cfg["wandb"], project=cfg["wandb_project"], run_name="bomber-actor-critic", config=cfg)
    wins = 0
    moving_win = 0.0

    for ep in range(cfg["episodes"]):
        # Actor-Criticは方策分布からサンプルするので、それ自体が探索になる。
        # epsilonは追加の一様混合で、通常は0のままentropy_coefで調整する。
        epsilon = exponential_decay(cfg["epsilon_start"], cfg["epsilon_end"], ep, cfg["episodes"], cfg["epsilon_decay_rate"])
        opponents = [BomberRuleBasedPolicy(seed=cfg["seed"] + ep * 17 + idx) for idx in range(3)]
        obs, _ = env.reset(seed=cfg["seed"] + ep)
        done = False
        ep_return = 0.0

        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        rewards: list[float] = []

        while not done:
            agent_obs = obs["agent_0"]
            # 1ステップ分の行動サンプルと、学習に必要な統計量を取得。
            action, log_prob, entropy, value = select_action_actor_critic(model, agent_obs, epsilon=epsilon, device=device)

            actions = {"agent_0": action}
            for idx in range(1, 4):
                actions[f"agent_{idx}"] = opponents[idx - 1].act(obs[f"agent_{idx}"], obs[f"agent_{idx}"]["action_mask"])
            result = env.step(actions)
            done = any(result.terminations.values()) or any(result.truncations.values())

            reward = student_reward(env, result.infos["agent_0"]["events"])
            ep_return += reward
            # エピソード全体を後でまとめて更新するため系列で保持。
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
            obs = result.observations

        returns = discounted_returns(rewards, gamma=cfg["gamma"], device=device)
        values_tensor = torch.stack(values)
        log_probs_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)
        # advantage = 実リターン - 価値予測
        # actorはadvantageが高い行動の確率を上げる。
        advantages = returns - values_tensor

        policy_loss = -(log_probs_tensor * advantages.detach()).mean()
        # criticは価値予測をreturnsに近づける。
        value_loss = F.mse_loss(values_tensor, returns)
        # entropyは方策の偏り過ぎを抑える正則化。
        entropy_loss = -entropy_tensor.mean()
        total_loss = policy_loss + cfg["value_coef"] * value_loss + cfg["entropy_coef"] * entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        # 勾配爆発を防ぐためのクリップ。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        win = int(env.last_winner == "agent_0")
        wins += win
        moving_win = 0.95 * moving_win + 0.05 * win
        logger.log(
            {
                "episode": ep,
                "return": ep_return,
                "win": win,
                "moving_win_rate": moving_win,
                "epsilon": epsilon,
                "steps": env.step_count,
                "loss_total": float(total_loss.item()),
                "loss_policy": float(policy_loss.item()),
                "loss_value": float(value_loss.item()),
                "entropy": float(entropy_tensor.mean().item()),
            },
            step=ep,
        )

    checkpoint = {
        "algo": "actor_critic",
        "obs_dim": obs_dim,
        "n_actions": 6,
        "hidden_dim": hidden_dim,
        "use_cnn": use_cnn,
        "grid_shape": list(grid_shape),
        "stats_dim": stats_dim,
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, out / "policy.pt")
    (out / "metadata.json").write_text(
        json.dumps({"student_id": cfg["student_id"], "model_name": cfg["model_name"], "env_id": "bomber_arena_v1", "algo": "actor_critic", "use_cnn": use_cnn}, indent=2),
        encoding="utf-8",
    )
    (out / "policy.py").write_text(
        "from marl_course.algos.bomber_torch import load_actor_critic_policy\n\n"
        "def load_policy(model_path, device='cpu'):\n"
        "    return load_actor_critic_policy(model_path, device=device)\n",
        encoding="utf-8",
    )

    inference_policy = ActorCriticTorchPolicy(model=model.eval(), device=device)
    gif_path = record_bomber_policy_gif(
        inference_policy,
        out,
        seed=cfg["seed"] + 9999,
        max_steps=cfg["gif_steps"],
        fps=cfg["gif_fps"],
        tile_size=cfg["gif_tile_size"],
        model_name=cfg["model_name"],
    )
    if gif_path is not None:
        logger.log_gif(gif_path, fps=cfg["gif_fps"])
    logger.close()
    print(
        {
            "episodes": cfg["episodes"],
            "wins": wins,
            "win_rate": wins / max(1, cfg["episodes"]),
            "out": str(out),
            "gif": str(gif_path) if gif_path else None,
        }
    )


def select_action_actor_critic(
    model: ActorCriticNet,
    obs: dict[str, object],
    epsilon: float,
    device: str,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    # 1観測をネットワークに通して (logits, value) を得る。
    obs_vec = bomber_obs_to_tensor(obs, device=device).unsqueeze(0)
    logits, value = model(obs_vec)
    mask_tensor = torch.tensor(obs["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
    logits = masked_logits(logits, mask_tensor)
    probs = torch.softmax(logits, dim=-1)
    if epsilon > 0.0:
        # 追加探索が必要な場合だけ、一様合法手分布を少し混ぜる。
        probs = (1.0 - epsilon) * probs + epsilon * (mask_tensor / torch.clamp(mask_tensor.sum(dim=1, keepdim=True), min=1.0))
    dist = torch.distributions.Categorical(probs=probs)
    action = dist.sample()
    return int(action.item()), dist.log_prob(action).squeeze(0), dist.entropy().squeeze(0), value.squeeze(0)


def discounted_returns(rewards: list[float], gamma: float, device: str) -> torch.Tensor:
    returns: list[float] = []
    running = 0.0
    # 末尾から畳み込むと G_t = r_t + gamma*r_{t+1} + ... を簡潔に計算できる。
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32, device=device)


def student_reward(env: BomberArenaEnv, events: list[object]) -> float:
    reward = 0.0
    for event in events:
        if getattr(event, "name", "") == "winner" and getattr(event, "actor", None) == "agent_0":
            reward += 1.0
        elif getattr(event, "name", "") == "powerup_collected" and getattr(event, "actor", None) == "agent_0":
            reward += 0.05
        elif getattr(event, "name", "") == "self_eliminated" and getattr(event, "actor", None) == "agent_0":
            reward -= 0.5
        elif getattr(event, "name", "") == "agent_eliminated" and getattr(event, "target", None) == "agent_0":
            # 敵の爆風や連鎖爆発で倒された場合も、終局報酬を待たず即時に減点する。
            reward -= 0.4
        elif getattr(event, "name", "") == "enemy_eliminated" and getattr(event, "actor", None) == "agent_0":
            reward += 0.4
        elif getattr(event, "name", "") == "block_destroyed" and getattr(event, "actor", None) == "agent_0":
            reward += 0.01
    if env.last_winner is not None and env.last_winner != "agent_0":
        reward -= 0.2
    return reward


def _effective_config(args: argparse.Namespace, file_config: dict[str, object]) -> dict[str, object]:
    """Actor-Criticの設定値を確定する (CLI > JSON > 既定値)。"""
    student_id = file_config.get("student_id", "bomber_actor_critic_student")
    episodes = int(cli_or_config(args, file_config, "episodes", 300))
    model_name = validate_model_name(cli_or_config(args, file_config, "model_name", default_model_name(student_id, "actorcritic", episodes)))
    return {
        # 提出管理 / 出力先
        "student_id": student_id,
        # モデル名は `<name>_<algo>_<episodes>` 形式。GIF凡例とmetadataに出る。
        "model_name": model_name,
        "out": str(cli_or_config(args, file_config, "out", "outputs/bomber_actor_critic_student")),
        # 追加学習の初期値にする既存policy.pt。Noneなら新規学習。
        "resume_from": str(cli_or_config(args, file_config, "resume_from", "")),
        # 学習長と乱数
        "episodes": episodes,
        "seed": int(cli_or_config(args, file_config, "seed", 0)),
        # 基本ハイパーパラメータ
        "gamma": float(cli_or_config(args, file_config, "gamma", 0.99)),
        "lr": float(cli_or_config(args, file_config, "lr", 3.0e-4)),
        # 損失の重み: total = policy + value_coef*value + entropy_coef*entropy
        "value_coef": float(cli_or_config(args, file_config, "value_coef", 0.5)),
        "entropy_coef": float(cli_or_config(args, file_config, "entropy_coef", 0.01)),
        # Actor-Criticは方策サンプリングとentropyが探索を担うため、デフォルトは0。
        "epsilon_start": float(cli_or_config(args, file_config, "epsilon_start", 0.0)),
        "epsilon_end": float(cli_or_config(args, file_config, "epsilon_end", 0.0)),
        "epsilon_decay_rate": float(cli_or_config(args, file_config, "epsilon_decay_rate", 8.0)),
        # ネットワーク規模 / 実行デバイス
        "hidden_dim": int(cli_or_config(args, file_config, "hidden_dim", 256)),
        # Trueなら盤面チャネルをCNN、statsをMLPで処理して結合する。
        "use_cnn": bool(cli_or_config(args, file_config, "use_cnn", False)),
        # `auto` はMacならmps、GPU付きColabならcuda、なければcpuへ自動解決。
        "device": str(cli_or_config(args, file_config, "device", "auto")),
        # 実験ログ
        "wandb": bool(cli_or_config(args, file_config, "wandb", False)),
        "wandb_project": str(cli_or_config(args, file_config, "wandb_project", "marl-course-games")),
        # 学習後GIF
        "gif_steps": int(cli_or_config(args, file_config, "gif_steps", 200)),
        "gif_fps": int(cli_or_config(args, file_config, "gif_fps", 8)),
        "gif_tile_size": int(cli_or_config(args, file_config, "gif_tile_size", 16)),
    }


def policy_path_or_none(value: object) -> Path | None:
    """追加学習用のpolicy.ptパスを解決する。"""
    if not value:
        return None
    path = Path(str(value))
    return path / "policy.pt" if path.is_dir() else path


if __name__ == "__main__":
    main()
