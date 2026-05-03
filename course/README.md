# MARL授業コード 正規入口

このディレクトリを授業中に見る入口にします。旧 `scripts/` は互換用・内部実装用として残します。

## 学生が使う学習コード

### Bomber-man

- `course/student/bomber/train_dqn_vs_baselines.py`
  - DQN vs ルールベース / random など
- `course/student/bomber/train_dqn_selfplay.py`
  - DQN vs 学習済みDQN。相手を固定し、評価勝率が上がったら自動で相手を入れ替える。
- `course/student/bomber/train_actor_critic_vs_baselines.py`
  - Actor-Critic vs ルールベース / random など
- `course/student/bomber/train_actor_critic_selfplay.py`
  - Actor-Critic vs 学習済みActor-Critic。相手入れ替え付き。

### Overcooked

- `course/student/overcooked/train_mappo.py`
  - ニューラルMAPPO 単一マップ学習
- `course/student/overcooked/train_qmix.py`
  - ニューラルQ-MIX 単一マップ学習
- `course/student/overcooked/train_mappo_zero_shot.py`
  - 複数マップ学習、未知マップ評価
- `course/student/overcooked/train_qmix_zero_shot.py`
  - 複数マップ学習、未知マップ評価

## 教師が使う評価コード

- `course/teacher/evaluate_bomber_tournament.py`
  - Bomber提出モデルの総当たり/トーナメントランキング
- `course/teacher/evaluate_overcooked_ranking.py`
  - Overcooked提出モデルのスコアランキング
- `course/teacher/evaluate_overcooked_zero_shot.py`
  - 未知マップ汎化ランキング
- `course/teacher/render_policy_gif.py`
  - 学習済みモデルからGIFを再生成

## シェルスクリプト

`course/run/` 以下に授業中にそのまま実行するスクリプトを置きます。

```bash
./course/run/bomber/dqn_vs_baselines.sh
./course/run/bomber/dqn_selfplay.sh
./course/run/bomber/actor_critic_vs_baselines.sh
./course/run/bomber/actor_critic_selfplay.sh
./course/run/overcooked/mappo.sh
./course/run/overcooked/qmix.sh
./course/run/overcooked/mappo_zero_shot.sh
./course/run/overcooked/qmix_zero_shot.sh
./course/run/teacher/evaluate_bomber_tournament.sh
./course/run/teacher/evaluate_overcooked_ranking.sh
```

## device方針

- Apple Silicon Mac: `device=auto` で `mps`
- それ以外: `cpu`

CUDAは教室環境の再現性と小規模モデルでの転送コストを考え、`auto` では選びません。明示的に `--device cuda` を指定した場合だけ使います。
