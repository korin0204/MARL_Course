"""Train Bomber student policy with tabular Q-learning.

Students can edit config JSON and reward shaping to study learning behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from marl_course.algos.artifacts import record_bomber_policy_gif
from marl_course.algos.bomber_qlearning import BomberQLearningPolicy
from marl_course.algos.logging import MetricLogger
from marl_course.common.config import cli_or_config, dump_effective_config, load_json_config
from marl_course.common.model_naming import default_model_name, validate_model_name
from marl_course.common.schedules import exponential_decay
from marl_course.envs.bomber_arena import BomberArenaEnv, BomberRuleBasedPolicy


def main() -> None:
    # 1) CLI引数を定義: config JSONを基本に、必要ならCLIで上書きできる。
    parser = argparse.ArgumentParser(description="Train a Bomber Arena Q-learning student.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_bomber.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--epsilon-start", type=float)
    parser.add_argument("--epsilon-end", type=float)
    parser.add_argument("--epsilon-decay-rate", type=float)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb-project")
    parser.add_argument("--gif-steps", type=int)
    parser.add_argument("--gif-fps", type=int)
    parser.add_argument("--gif-tile-size", type=int)
    args = parser.parse_args()
    file_config = load_json_config(args.config)
    # 2) 実際に使う設定値を確定して出力ディレクトリへ保存する。
    #    (授業で再現実験しやすくするため)
    cfg = _effective_config(args, file_config)
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    dump_effective_config(out, cfg)
    resume_from = _policy_path_or_none(cfg["resume_from"])
    policy = BomberQLearningPolicy.load(resume_from) if resume_from else BomberQLearningPolicy(seed=cfg["seed"])
    logger = MetricLogger(
        out,
        use_wandb=cfg["wandb"],
        project=cfg["wandb_project"],
        run_name="bomber-qlearning",
        config=cfg,
    )
    wins = 0
    moving_win = 0.0
    # 3) エピソード学習ループ
    for ep in range(cfg["episodes"]):
        # epsilonは指数減衰。decay_rateを大きくすると早くgreedy寄りになる。
        epsilon = exponential_decay(cfg["epsilon_start"], cfg["epsilon_end"], ep, cfg["episodes"], cfg["epsilon_decay_rate"])
        env = BomberArenaEnv()
        # 対戦相手は強めのルールベース3体 (studentはagent_0)。
        opponents = [BomberRuleBasedPolicy(seed=cfg["seed"] + ep * 10 + idx) for idx in range(3)]
        obs, _ = env.reset(seed=cfg["seed"] + ep)
        done = False
        total_reward = 0.0
        td_error = 0.0
        updates = 0
        while not done:
            # 現在状態からstudentの行動を選択 (epsilon-greedy)。
            agent_obs = obs["agent_0"]
            action = policy.act(agent_obs, agent_obs["action_mask"], deterministic=False, epsilon=epsilon)
            actions = {"agent_0": action}
            for idx in range(1, 4):
                actions[f"agent_{idx}"] = opponents[idx - 1].act(obs[f"agent_{idx}"], obs[f"agent_{idx}"]["action_mask"])
            result = env.step(actions)
            done = any(result.terminations.values()) or any(result.truncations.values())
            # 報酬設計は関数に切り出してあり、学生が編集しやすい。
            reward = _student_reward(env, result.infos["agent_0"]["events"])
            total_reward += reward
            # 1-step TD更新: Q(s,a) <- Q(s,a)+alpha*(r+gamma*maxQ(s')-Q(s,a))
            td_error += policy.update(agent_obs, action, reward, result.observations["agent_0"], done, cfg["alpha"], cfg["gamma"])
            updates += 1
            obs = result.observations
        win = int(env.last_winner == "agent_0")
        wins += win
        moving_win = 0.95 * moving_win + 0.05 * win
        logger.log(
            {
                # return: shaped rewardの合計, win: 勝敗(0/1), moving_win_rate: 指数移動平均
                "episode": ep,
                "return": total_reward,
                "win": win,
                "moving_win_rate": moving_win,
                "epsilon": epsilon,
                "steps": env.step_count,
                "td_error": td_error / max(1, updates),
                "q_states": len(policy.q_table),
            },
            step=ep,
        )
    policy.save(out / "policy.pt")
    # 提出用メタ情報: 教員側トーナメントで環境互換を検証するために使う。
    (out / "metadata.json").write_text(
        json.dumps({"student_id": cfg["student_id"], "model_name": cfg["model_name"], "env_id": "bomber_arena_v1", "algo": "qlearning"}, indent=2),
        encoding="utf-8",
    )
    (out / "policy.py").write_text(
        "from pathlib import Path\n"
        "from marl_course.algos.bomber_qlearning import BomberQLearningPolicy\n\n"
        "def load_policy(model_path, device='cpu'):\n"
        "    return BomberQLearningPolicy.load(Path(model_path))\n",
        encoding="utf-8",
    )
    gif_path = record_bomber_policy_gif(
        policy,
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
    print({"episodes": cfg["episodes"], "wins": wins, "win_rate": wins / max(1, cfg["episodes"]), "out": str(out), "gif": str(gif_path) if gif_path else None})


def _effective_config(args: argparse.Namespace, file_config: dict[str, object]) -> dict[str, object]:
    """実行時設定を確定する。

    優先順位: CLI引数 > config JSON > 既定値
    """
    student_id = file_config.get("student_id", "bomber_student")
    episodes = int(cli_or_config(args, file_config, "episodes", 200))
    model_name = validate_model_name(cli_or_config(args, file_config, "model_name", default_model_name(student_id, "qlearning", episodes)))
    return {
        # 提出管理
        "student_id": student_id,
        # モデル名は `<name>_<algo>_<episodes>` 形式。GIF凡例とmetadataに出る。
        "model_name": model_name,
        # 出力先 (学習済みモデル / ログ / effective_config.json を保存)
        "out": str(cli_or_config(args, file_config, "out", "outputs/bomber_student")),
        # 追加学習の初期値にする既存policy.pt。Noneなら新規学習。
        "resume_from": str(cli_or_config(args, file_config, "resume_from", "")),
        # 学習の長さと乱数
        "episodes": episodes,
        "seed": int(cli_or_config(args, file_config, "seed", 0)),
        # Q-learning更新式の係数
        "alpha": float(cli_or_config(args, file_config, "alpha", 0.25)),
        "gamma": float(cli_or_config(args, file_config, "gamma", 0.97)),
        # epsilon-greedy探索率 (開始値→終了値へ減衰)
        "epsilon_start": float(cli_or_config(args, file_config, "epsilon_start", 0.4)),
        "epsilon_end": float(cli_or_config(args, file_config, "epsilon_end", 0.02)),
        "epsilon_decay_rate": float(cli_or_config(args, file_config, "epsilon_decay_rate", 8.0)),
        # 実験ログ
        "wandb": bool(cli_or_config(args, file_config, "wandb", False)),
        "wandb_project": str(cli_or_config(args, file_config, "wandb_project", "marl-course-games")),
        # 学習後GIFの描画設定
        "gif_steps": int(cli_or_config(args, file_config, "gif_steps", 160)),
        "gif_fps": int(cli_or_config(args, file_config, "gif_fps", 8)),
        "gif_tile_size": int(cli_or_config(args, file_config, "gif_tile_size", 16)),
    }


def _student_reward(env: BomberArenaEnv, events: list[object]) -> float:
    """Default shaped reward used during training (not teacher grading)."""
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


def _policy_path_or_none(value: object) -> Path | None:
    """追加学習用のpolicy.ptパスを解決する。"""
    if not value:
        return None
    path = Path(str(value))
    return path / "policy.pt" if path.is_dir() else path


if __name__ == "__main__":
    main()
