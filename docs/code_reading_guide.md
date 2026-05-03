# コード読解ガイド（授業用）

このドキュメントは、実行コマンドではなく「どのファイルをどう読むと理解が進むか」をまとめたものです。

## 1. まず全体像を掴む

- `scripts/`: 実行入口（train / evaluate / visualize）
- `src/marl_course/envs/`: ゲーム環境本体
- `src/marl_course/algos/`: 学習アルゴリズムと補助
- `src/marl_course/evaluation/`: 教師評価ロジック
- `src/marl_course/common/`: config・提出物ロードなどの共通部
- `configs/`: 実験設定（JSON）

最初は `scripts/train_*.py` を1本開いて、そこから import 先に降りていく読み方が最短です。

## 2. 学習ループの読み方（共通）

各 `train_*.py` で共通の流れ:

1. `load_json_config` で JSON を読む  
2. `_effective_config(...)` で「CLI > JSON > default」で最終設定を確定  
3. `env.reset()` で初期化  
4. `while not done:` で行動選択 → `env.step(actions)`  
5. 報酬計算（`student_reward` など）  
6. 方策更新（Q-learning / DQN / Actor-Critic / bandit）  
7. `MetricLogger` で `metrics.csv` と W&B へ記録  
8. `policy.pt`, `policy.py`, `metadata.json` を出力  

## 3. Bomber の重要ファイル

- `scripts/train_bomber.py`  
  - 表形式 Q-learning の最小例。更新式の追跡が最もしやすい。
- `scripts/train_bomber_dqn.py`  
  - リプレイバッファ、target network、ミニバッチ更新の実装。
- `scripts/train_bomber_actor_critic.py`  
  - return/advantage 計算、方策損失・価値損失・entropy 正則化。
- `src/marl_course/envs/bomber_arena/env.py`  
  - 爆弾・爆風・勝敗ルール、観測生成。
- `src/marl_course/envs/bomber_arena/policies.py`  
  - ルールベース対戦相手（授業デフォルト敵）。

## 4. Coop / Zero-shot の重要ファイル

- `scripts/train_coop_mappo.py` / `scripts/train_coop_qmix.py`  
  - 軽量戦略学習（bandit）として、4人の役割方針を更新。
- `scripts/train_coop_zero_shot_league.py`  
  - 既知 + 生成マップで学習し、汎化を狙う。
- `src/marl_course/envs/coop_kitchen/env.py`  
  - 料理進行（食材投入、調理、皿、配膳）とスコア定義。
- `src/marl_course/envs/coop_kitchen/layouts.py`  
  - 固定マップと生成マップ family。
- `src/marl_course/evaluation/coop.py`  
  - `seen / unseen_same_family / heldout_family` の評価導線。

## 5. 教師評価の読みどころ

- `scripts/evaluate_bomber_tournament.py`
- `scripts/evaluate_coop_submissions.py`
- `scripts/evaluate_coop_zero_shot.py`
- `src/marl_course/common/submission.py`

ポイント:

- 評価は提出モデルの推論再生で行う（学習時 reward は採点に直結しない）
- `metadata.json` の `env_id` で互換性チェック
- `policy.py` の `load_policy(...)` を共通インターフェースとして扱う

## 6. 可視化関連

- `scripts/visualize_episode.py`: 手元デバッグ用
- `src/marl_course/visualization/pygame_viewer.py`: 授業投影用
- `src/marl_course/algos/artifacts.py`: 学習後GIF保存

## 7. 読解のおすすめ順

1. `scripts/train_bomber.py`
2. `src/marl_course/envs/bomber_arena/env.py`
3. `scripts/evaluate_bomber_tournament.py`
4. `scripts/train_bomber_dqn.py`
5. `scripts/train_coop_zero_shot_league.py`
6. `src/marl_course/evaluation/coop.py`

この順だと「単純 → 複雑」「学習 → 評価 → 汎化」と自然に理解が進みます。
