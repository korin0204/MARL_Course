# 学生向け 学習コードの使い方

実行だけでなくコード理解を進める場合は、先に次を読むのがおすすめです。

- `docs/code_reading_guide.md`
- `docs/config_reference.md`

## 1. ローカル環境

このプロジェクトでは `pyenv + venv` を使います。

```bash
pyenv local 3.11.0
python -m venv .venv
.venv/bin/python -m pip install -e ".[train]"
```

W&B を使う場合:

```bash
.venv/bin/python -m wandb login
```

W&B を使わない場合でも、各学習スクリプトは `outputs/.../metrics.csv` に学習曲線を保存します。

Apple Silicon Macを使う学生は、追加で次のガイドも見てください。

```text
docs/mac_acceleration_guide.md
```

MPS確認:

```bash
./run_mac_check_device.sh
```

## 2. Bomber Arena

py 版:

```bash
.venv/bin/python scripts/train_bomber.py --config configs/train_bomber.json
```

Q-learning が弱いときは、DQN と Actor-Critic を使えます。

```bash
.venv/bin/python scripts/train_bomber_dqn.py --config configs/train_bomber_dqn.json
.venv/bin/python scripts/train_bomber_actor_critic.py --config configs/train_bomber_actor_critic.json
```

MacでCNN付きモデルを速く回す場合:

```bash
./run_mac_bomber_dqn_fast.sh
./run_mac_bomber_actor_critic_fast.sh
```

W&B に学習曲線を出す場合:

```bash
.venv/bin/python scripts/train_bomber.py --config configs/train_bomber.json --wandb
.venv/bin/python scripts/train_bomber_dqn.py --config configs/train_bomber_dqn.json --wandb
.venv/bin/python scripts/train_bomber_actor_critic.py --config configs/train_bomber_actor_critic.json --wandb
```

Colab 版:

```text
notebooks/01_bomber_arena_student_colab.ipynb
```

変更してよい場所:

- `configs/train_bomber.json`
- `configs/train_bomber_dqn.json`
- `configs/train_bomber_actor_critic.json`
- `model_name`（例: `alice_dqn_10000`。GIF凡例とmetadataに表示される）
- `student_reward(...)`
- `episodes`
- `alpha`, `gamma`
- `epsilon_start`, `epsilon_end`, `epsilon_decay_rate`
- `gif_steps`, `gif_fps`, `gif_tile_size`
- `BomberQLearningPolicy` を自作方策に置き換える部分

提出物は `outputs/my_bomber/` に作られます。

```text
metadata.json
policy.py
policy.pt
metrics.csv
effective_config.json
post_training_episode.gif
```

教師評価では `student_reward(...)` は使われません。提出モデルを決定的に再生し、勝敗で採点されます。
敵を倒したときは `enemy_eliminated` イベントが出ます。`event.actor` が倒した側、`event.target` が倒された側なので、報酬設計で撃破ボーナスを加算できます。

### 2.1 コードの対応関係（Bomber）

- 実行入口: `scripts/train_bomber.py`
- 環境本体: `src/marl_course/envs/bomber_arena/env.py`
- ルールベース敵: `src/marl_course/envs/bomber_arena/policies.py`
- Qテーブル方策: `src/marl_course/algos/bomber_qlearning.py`
- DQN/ACモデル: `src/marl_course/algos/bomber_torch.py`

`configs/train_bomber_dqn.json` と `configs/train_bomber_actor_critic.json` では `use_cnn` を `true` にできます。  
`true` の場合、盤面チャネル（壁、木箱、爆弾、炎、敵位置など）はCNNで処理し、ammo/blast/座標/self_idのような非空間情報はMLPで処理してから結合します。Bomberは空間構造が重要なので、MLPのみより学習しやすくなる可能性があります。

`device` は通常 `auto` にしてください。MacならMPS、ColabならCUDA、どちらもなければCPUを選び、実際に選ばれた値は `effective_config.json` に保存されます。

探索率は指数減衰で、`epsilon_decay_rate` が大きいほど早く `epsilon_end` に近づきます。DQN/Q-learningでは `epsilon_start=0.4`, `epsilon_end=0.02`, `epsilon_decay_rate=8.0` を初期値にしています。Actor-Criticは方策分布からサンプルするため、追加epsilonはデフォルト `0.0` です。

## 3. Coop Kitchen MAPPO-lite

py 版:

```bash
.venv/bin/python scripts/train_coop_mappo.py --config configs/train_coop_mappo.json
```

W&B:

```bash
.venv/bin/python scripts/train_coop_mappo.py --config configs/train_coop_mappo.json --wandb
```

Colab 版:

```text
notebooks/02_coop_kitchen_mappo_colab.ipynb
```

このループは、共有actorとcentralized criticを使うニューラル MAPPO です。各agentの盤面チャネルをCNNで処理し、criticは4人分の観測をまとめてチーム価値を予測します。

## 4. Coop Kitchen QMIX-lite

py 版:

```bash
.venv/bin/python scripts/train_coop_qmix.py --config configs/train_coop_qmix.json
```

Colab 版:

```text
notebooks/03_coop_kitchen_qmix_colab.ipynb
```

このループは、共有agent Q networkとmonotonic mixing networkを使うニューラル QMIX です。各agentのQ値をmixerでチームQ値へ合成して、完全協調報酬を学習します。

## 5. Overcooked Zero-Shot League

py 版:

```bash
.venv/bin/python scripts/train_coop_zero_shot_league.py --config configs/train_coop_zero_shot.json
```

Colab 版:

```text
notebooks/04_coop_kitchen_zero_shot_league_colab.ipynb
```

zero-shot 版では、固定マップだけでなく生成マップも使って学習します。未知マップで動くように、観測の `valid_cell_mask` と padding 済み `agent_obs` を利用してください。

### 5.1 コードの対応関係（Coop）

- 実行入口: `scripts/train_coop_mappo.py`, `scripts/train_coop_qmix.py`, `scripts/train_coop_zero_shot_league.py`
- 環境本体: `src/marl_course/envs/coop_kitchen/env.py`
- マップ定義/生成: `src/marl_course/envs/coop_kitchen/layouts.py`
- ベースライン方策: `src/marl_course/envs/coop_kitchen/policies.py`
- チーム観測変換: `src/marl_course/evaluation/coop.py` の `make_team_obs`

## 6. 学習曲線

W&B を使う場合は `--wandb` を付けます。

```bash
.venv/bin/python scripts/train_bomber.py --config configs/train_bomber.json --wandb
```

`config.json` の内容は W&B の config として保存されます。学習後の `post_training_episode.gif` も W&B に送られます。ローカルでは同じ GIF が出力ディレクトリに残ります。

W&B を使わない場合:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/my_bomber/metrics.csv")
print(df.tail())
PY
```

Colab notebook では最後のセルで `metrics.csv` を plot します。

## 7. 学習済みモデルからGIFを作り直す

学習済みの `outputs/...` を使って、相手を変えたGIFを再生成できます。

Bomber:

```bash
.venv/bin/python scripts/render_policy_gif.py \
  --env bomber \
  --model-dir outputs/my_bomber \
  --opponents rule,random,stay \
  --steps 200 \
  --tile-size 24
```

`--opponents` は `rule`, `random`, `stay`, `submission:/path/to/other_student` を3つ並べます。

Coop:

```bash
.venv/bin/python scripts/render_policy_gif.py \
  --env coop \
  --model-dir outputs/my_coop \
  --steps 200 \
  --tile-size 24
```

GIFと教師用 `--live` 表示は同じスプライト描画を使うので、凡例・色・進捗表示が一致します。
凡例には `A -> alice_dqn_10000`, `B -> rule_based` のように、どのエージェントがどのモデルかも表示されます。

## 8. 追加学習

すべてのtrainスクリプトで `--resume-from` が使えます。

```bash
.venv/bin/python scripts/train_bomber_dqn.py \
  --config configs/train_bomber_dqn.json \
  --resume-from outputs/my_bomber_dqn \
  --episodes 1000 \
  --out outputs/my_bomber_dqn_more
```

`--resume-from` には `policy.pt` そのもの、または `policy.pt` を含むディレクトリを指定できます。

## 9. run.shで簡単に実行する

```bash
./run_train_examples.sh
./run_continue_examples.sh
ENV=bomber MODEL_DIR=outputs/demo_bomber_dqn ./run_render_gifs.sh
```

## 10. 読み進め方（最短）

1. `scripts/train_bomber.py` を読む（学習ループの型を掴む）  
2. `_effective_config` のコメントと `docs/config_reference.md` を照合  
3. `env.step(...)` の実体として `env.py` を読む  
4. 評価スクリプト (`scripts/evaluate_*.py`) を見て提出形式を確認  
