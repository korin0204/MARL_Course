"""Train Bomber student policy with Deep Q-Network (DQN)."""
from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

from marl_course.algos.artifacts import record_bomber_policy_gif
from marl_course.algos.bomber_torch import DQNNet, DQNTorchPolicy, bomber_obs_to_tensor, infer_obs_shapes
from marl_course.algos.logging import MetricLogger
from marl_course.common.config import cli_or_config, dump_effective_config, load_json_config
from marl_course.common.model_naming import default_model_name, validate_model_name
from marl_course.common.schedules import exponential_decay
from marl_course.common.torch_device import configure_torch_runtime, resolve_torch_device, torch_device_summary
from marl_course.envs.bomber_arena import BomberArenaEnv, BomberRuleBasedPolicy


def main() -> None:
    # CLIで受け取る値: ほとんどはconfig JSONで管理し、必要時のみ上書き。
    parser = argparse.ArgumentParser(description="Train a Bomber Arena DQN student.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_bomber_dqn.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume-from", type=Path)
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

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # 観測次元を初回resetから推定してネットワークを構築。
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
    online = DQNNet(obs_dim=obs_dim, hidden_dim=hidden_dim, use_cnn=use_cnn, grid_shape=grid_shape, stats_dim=stats_dim).to(device)
    target = DQNNet(obs_dim=obs_dim, hidden_dim=hidden_dim, use_cnn=use_cnn, grid_shape=grid_shape, stats_dim=stats_dim).to(device)
    if resume_from is not None:
        online.load_state_dict(resume_payload["model_state_dict"])
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=cfg["lr"])
    replay = deque(maxlen=cfg["replay_size"])

    logger = MetricLogger(out, use_wandb=cfg["wandb"], project=cfg["wandb_project"], run_name="bomber-dqn", config=cfg)
    total_steps = 0
    wins = 0
    moving_win = 0.0

    for ep in range(cfg["episodes"]):
        # DQNはargmax方策になりやすいためepsilon-greedyを使う。
        # decay_rateを大きくするとランダム行動期間を短くできる。
        epsilon = exponential_decay(cfg["epsilon_start"], cfg["epsilon_end"], ep, cfg["episodes"], cfg["epsilon_decay_rate"])
        opponents = [BomberRuleBasedPolicy(seed=cfg["seed"] + ep * 13 + idx) for idx in range(3)]
        obs, _ = env.reset(seed=cfg["seed"] + ep)
        done = False
        ep_return = 0.0
        losses: list[float] = []

        while not done:
            agent_obs = obs["agent_0"]
            mask = agent_obs["action_mask"]
            # DQNの行動選択: epsilon-greedy + 非合法手のmask。
            action = select_action_dqn(online, agent_obs, mask, epsilon=epsilon, device=device)

            actions = {"agent_0": action}
            for idx in range(1, 4):
                actions[f"agent_{idx}"] = opponents[idx - 1].act(obs[f"agent_{idx}"], obs[f"agent_{idx}"]["action_mask"])
            result = env.step(actions)
            done = any(result.terminations.values()) or any(result.truncations.values())

            reward = student_reward(env, result.infos["agent_0"]["events"])
            ep_return += reward
            # replay bufferにはCPU tensorで遷移を保存し、学習時にdeviceへ移す。
            transition = (
                bomber_obs_to_tensor(agent_obs, device="cpu"),
                int(action),
                float(reward),
                bomber_obs_to_tensor(result.observations["agent_0"], device="cpu"),
                float(done),
                torch.tensor(result.observations["agent_0"]["action_mask"], dtype=torch.float32),
            )
            replay.append(transition)
            obs = result.observations
            total_steps += 1

            if len(replay) >= cfg["warmup_steps"] and total_steps % cfg["train_every"] == 0:
                # onlineで学習、targetでブートストラップ先を計算 (DQNの基本構成)。
                loss_value = train_dqn_step(
                    online=online,
                    target=target,
                    optimizer=optimizer,
                    replay=replay,
                    batch_size=cfg["batch_size"],
                    gamma=cfg["gamma"],
                    device=device,
                )
                losses.append(loss_value)
                if total_steps % cfg["target_update"] == 0:
                    # target networkを周期同期して学習を安定化。
                    target.load_state_dict(online.state_dict())

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
                "loss": sum(losses) / max(1, len(losses)),
                "replay_size": len(replay),
            },
            step=ep,
        )

    checkpoint = {
        # 教員側評価ローダーが使う最低限の構造。
        "algo": "dqn",
        "obs_dim": obs_dim,
        "n_actions": 6,
        "hidden_dim": hidden_dim,
        "use_cnn": use_cnn,
        "grid_shape": list(grid_shape),
        "stats_dim": stats_dim,
        "model_state_dict": online.state_dict(),
    }
    torch.save(checkpoint, out / "policy.pt")
    (out / "metadata.json").write_text(
        json.dumps({"student_id": cfg["student_id"], "model_name": cfg["model_name"], "env_id": "bomber_arena_v1", "algo": "dqn", "use_cnn": use_cnn}, indent=2),
        encoding="utf-8",
    )
    (out / "policy.py").write_text(
        "from marl_course.algos.bomber_torch import load_dqn_policy\n\n"
        "def load_policy(model_path, device='cpu'):\n"
        "    return load_dqn_policy(model_path, device=device)\n",
        encoding="utf-8",
    )

    inference_policy = DQNTorchPolicy(model=online.eval(), device=device)
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


def select_action_dqn(model: DQNNet, obs: dict[str, object], action_mask: list[int], epsilon: float, device: str) -> int:
    # action_mask=1 の行動だけを候補にする。
    legal = [idx for idx, ok in enumerate(action_mask) if ok]
    if not legal:
        return 0
    if random.random() < epsilon:
        return random.choice(legal)
    with torch.no_grad():
        obs_vec = bomber_obs_to_tensor(obs, device=device).unsqueeze(0)
        q_values = model(obs_vec).squeeze(0)
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device)
        # 非合法行動は非常に小さい値で潰してargmax対象から除外。
        q_values = q_values.masked_fill(mask_tensor <= 0, -1.0e9)
        return int(torch.argmax(q_values).item())


def train_dqn_step(
    online: DQNNet,
    target: DQNNet,
    optimizer: torch.optim.Optimizer,
    replay: deque,
    batch_size: int,
    gamma: float,
    device: str,
) -> float:
    # ミニバッチをリプレイバッファから一様サンプル。
    batch = random.sample(replay, batch_size)
    obs_vec = torch.stack([item[0] for item in batch]).to(device)
    actions = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
    next_obs_vec = torch.stack([item[3] for item in batch]).to(device)
    dones = torch.tensor([item[4] for item in batch], dtype=torch.float32, device=device)
    next_masks = torch.stack([item[5] for item in batch]).to(device)

    q_values = online(obs_vec).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # TD target: r + gamma * max_a' Q_target(s', a') * (1-done)
        next_q = target(next_obs_vec)
        next_q = next_q.masked_fill(next_masks <= 0, -1.0e9)
        max_next_q = torch.max(next_q, dim=1).values
        targets = rewards + gamma * (1.0 - dones) * max_next_q

    loss = F.smooth_l1_loss(q_values, targets)
    # 逆伝播 + 勾配クリップ (発散しにくくする)。
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


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
    """DQNの設定値を確定する (CLI > JSON > 既定値)。"""
    student_id = file_config.get("student_id", "bomber_dqn_student")
    episodes = int(cli_or_config(args, file_config, "episodes", 300))
    model_name = validate_model_name(cli_or_config(args, file_config, "model_name", default_model_name(student_id, "dqn", episodes)))
    return {
        # 提出管理 / 出力先
        "student_id": student_id,
        # モデル名は `<name>_<algo>_<episodes>` 形式。GIF凡例とmetadataに出る。
        "model_name": model_name,
        "out": str(cli_or_config(args, file_config, "out", "outputs/bomber_dqn_student")),
        # 追加学習の初期値にする既存policy.pt。Noneなら新規学習。
        "resume_from": str(cli_or_config(args, file_config, "resume_from", "")),
        # 学習長と乱数
        "episodes": episodes,
        "seed": int(cli_or_config(args, file_config, "seed", 0)),
        # 最適化と割引率
        "gamma": float(cli_or_config(args, file_config, "gamma", 0.99)),
        "lr": float(cli_or_config(args, file_config, "lr", 1.0e-3)),
        # ミニバッチ学習 / リプレイバッファ
        "batch_size": int(cli_or_config(args, file_config, "batch_size", 64)),
        "replay_size": int(cli_or_config(args, file_config, "replay_size", 20000)),
        "warmup_steps": int(cli_or_config(args, file_config, "warmup_steps", 1000)),
        # 何ステップごとに学習するか / target同期間隔
        "train_every": int(cli_or_config(args, file_config, "train_every", 1)),
        "target_update": int(cli_or_config(args, file_config, "target_update", 500)),
        # 探索率
        "epsilon_start": float(cli_or_config(args, file_config, "epsilon_start", 0.4)),
        "epsilon_end": float(cli_or_config(args, file_config, "epsilon_end", 0.02)),
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
