# 設定ファイル（config JSON）リファレンス

この資料は、`train_*.py` の `_effective_config` に対応した説明です。
値の優先順位はすべて共通で、**CLI引数 > config JSON > 既定値** です。

## 共通キー（ほぼ全 train スクリプト）

- `student_id`: 提出識別子。`metadata.json` に保存される。
- `model_name`: モデル識別名。命名規則は `<name>_<algo>_<episodes>`。例: `alice_dqn_10000`。
- `out`: 出力ディレクトリ。`policy.pt`, `metrics.csv`, `effective_config.json` などが出る。
- `resume_from`: 追加学習の初期値にする既存 `policy.pt` または出力ディレクトリ。
- `episodes`: 学習エピソード数。
- `seed`: 乱数シード。
- `wandb`: `true` なら W&B へメトリクス送信。
- `wandb_project`: W&B プロジェクト名。
- `gif_steps`: 学習後GIFの最大ステップ数。
- `gif_fps`: GIF再生速度。
- `gif_tile_size`: GIFタイル描画サイズ。

## Bomber Q-learning (`configs/train_bomber.json`)

- `alpha`: 学習率（TD誤差をどれだけ反映するか）。
- `gamma`: 割引率（将来報酬の重み）。
- `epsilon_start`: 探索率の初期値。
- `epsilon_end`: 探索率の最終値。
- `epsilon_decay_rate`: 指数減衰の速さ。大きいほど早く `epsilon_end` に近づく。

## Bomber DQN (`configs/train_bomber_dqn.json`)

- `gamma`: 割引率。
- `lr`: Adam 学習率。
- `batch_size`: ミニバッチサイズ。
- `replay_size`: リプレイバッファ容量。
- `warmup_steps`: 学習開始前にバッファを貯めるステップ数。
- `train_every`: 何ステップごとに1回更新するか。
- `target_update`: target network 同期間隔。
- `epsilon_start`, `epsilon_end`: epsilon-greedy探索率。
- `epsilon_decay_rate`: 指数減衰の速さ。DQNでは高めにしてランダム行動期間を短くする。
- `hidden_dim`: MLP中間層次元。
- `use_cnn`: `true` の場合、盤面チャネルをCNN、stats/self_idをMLPで処理して結合する。
- `device`: `auto` / `mps` / `cuda` / `cpu`。`auto` はApple Silicon Macなら `mps`、それ以外は `cpu` を選ぶ。CUDAは明示指定時だけ使う。
- `torch_device`: 実行時に自動追加されるdevice情報。`effective_config.json` とW&B configに保存される。

## Bomber Actor-Critic (`configs/train_bomber_actor_critic.json`)

- `gamma`: 割引率。
- `lr`: Adam 学習率。
- `value_coef`: 価値損失の重み。
- `entropy_coef`: entropy 項の重み（探索維持）。
- `epsilon_start`, `epsilon_end`: 方策分布に追加で混ぜる一様探索率。通常は `0.0` 推奨。
- `epsilon_decay_rate`: 追加探索を使う場合の指数減衰速度。
- `hidden_dim`: MLP中間層次元。
- `use_cnn`: `true` の場合、盤面チャネルをCNN、stats/self_idをMLPで処理して結合する。
- `device`: `auto` / `mps` / `cuda` / `cpu`。Mac学生は通常 `auto` 推奨。Mac以外の `auto` は `cpu`。
- `torch_device`: 実行時に自動追加されるdevice情報。MPS利用可否やPyTorch version確認に使う。

## Coop MAPPO (`configs/train_coop_mappo.json`)

- `layout_name`: 単一MAP学習で使う既知レイアウト。
- `gamma`: 割引率。
- `gae_lambda`: GAEでadvantageを平滑化する係数。
- `lr`: Adam学習率。
- `hidden_dim`: CNN encoder / actor / critic の中間次元。
- `ppo_epochs`: 1 rolloutを何回PPO更新に使うか。
- `clip_eps`: PPO ratio clipping幅。
- `value_coef`: critic損失の重み。
- `entropy_coef`: 探索維持のentropy重み。
- `device`: `auto` / `mps` / `cpu` / 明示`cuda`。

## Coop QMIX (`configs/train_coop_qmix.json`)

- `layout_name`: 単一MAP学習で使う既知レイアウト。
- `gamma`: 割引率。
- `lr`: Adam学習率。
- `hidden_dim`: agent Q network の中間次元。
- `mixing_dim`: QMIX mixer の隠れ次元。
- `batch_size`: replay bufferからサンプルする遷移数。
- `replay_size`: replay buffer容量。
- `warmup_steps`: 学習更新を始める前に貯める遷移数。
- `train_every`: 何環境stepごとに更新するか。
- `target_update`: target network同期間隔。
- `epsilon_start`, `epsilon_end`: 戦略探索率。
- `epsilon_decay_rate`: 戦略探索率の指数減衰速度。

## Coop Zero-Shot (`configs/train_coop_zero_shot.json`)

- MAPPO zero-shotではMAPPO系の `gamma`, `gae_lambda`, `ppo_epochs` などを使う。
- QMIX zero-shotではQMIX系の `batch_size`, `replay_size`, `mixing_dim` などを使う。
- `epsilon_start`, `epsilon_end`: 戦略探索率。
- `epsilon_decay_rate`: 戦略探索率の指数減衰速度。
- `families`: 生成マップの family 候補（汎化学習分布）。

## どの設定が実際に使われたか確認する方法

各実行の `out` 以下に `effective_config.json` が出力される。  
採点時・再現時はこのファイルを正として扱う。
