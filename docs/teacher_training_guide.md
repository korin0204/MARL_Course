# 教師向け 学習・評価コードの使い方

授業準備でコード全体の意図を確認したい場合は、次を併読してください。

- `docs/code_reading_guide.md`
- `docs/config_reference.md`

## 1. 環境準備

```bash
pyenv local 3.11.0
python -m venv .venv
.venv/bin/python -m pip install -e ".[train,dev]"
```

W&B を授業で使う場合は、学生に project 名を指定させると回収しやすくなります。

```bash
.venv/bin/python scripts/train_bomber.py --config configs/train_bomber.json --wandb --wandb-project marl-course-2026
.venv/bin/python scripts/train_bomber_dqn.py --config configs/train_bomber_dqn.json --wandb --wandb-project marl-course-2026
.venv/bin/python scripts/train_bomber_actor_critic.py --config configs/train_bomber_actor_critic.json --wandb --wandb-project marl-course-2026
```

W&B を使わない授業では、各提出ディレクトリの `metrics.csv` を回収してください。

## 2. 学生に配るもの

- `notebooks/01_bomber_arena_student_colab.ipynb`
- `notebooks/02_coop_kitchen_mappo_colab.ipynb`
- `notebooks/03_coop_kitchen_qmix_colab.ipynb`
- `notebooks/04_coop_kitchen_zero_shot_league_colab.ipynb`
- `docs/student_training_guide.md`
- `docs/assignment_bomber_arena.md`
- `docs/assignment_coop_kitchen.md`
- `docs/code_reading_guide.md`（発展学習用）
- `docs/config_reference.md`（ハイパラ意味確認用）

学生が変更してよいのは、学習報酬、学習率、探索率、モデル構造、学習 episode 数、方策実装です。

## 3. 提出物の回収

各学生の提出ディレクトリは次の形式にします。

```text
submissions/student_id/
├── metadata.json
├── policy.py
└── policy.pt
```

`metrics.csv` は成績評価に必須ではありませんが、レポート確認や不正な偶然勝ちの検出に役立ちます。

## 4. Bomber Arena 評価

通常評価:

```bash
.venv/bin/python scripts/evaluate_bomber_tournament.py --config configs/evaluate_bomber.json --submissions submissions --episodes 8
```

教室スクリーン表示:

```bash
.venv/bin/python scripts/evaluate_bomber_tournament.py --config configs/evaluate_bomber.json --submissions submissions --episodes 1 --live
```

描画が速すぎる場合は `configs/evaluate_bomber.json` の `live_sleep` を大きくします。例: `0.12` から `0.25`。画面の大きさは `live_tile_size` で調整します。

pygame を使えない環境:

```bash
.venv/bin/python scripts/evaluate_bomber_tournament.py --config configs/evaluate_bomber.json --submissions submissions --episodes 1 --live-ascii
```

Bomber は報酬合計ではなく、対戦結果をスコア化します。`policy.act(..., deterministic=True)` の再生結果が評価対象です。

### 4.1 評価コードの見どころ

- `scripts/evaluate_bomber_tournament.py`  
  提出ディレクトリ探索とJSON結果出力
- `src/marl_course/common/submission.py`  
  `policy.py` + `policy.pt` のロード規約
- `src/marl_course/evaluation/bomber.py`  
  リーグ戦ループ、順位計算

## 5. Coop Kitchen 評価

固定マップ評価:

```bash
.venv/bin/python scripts/evaluate_coop_submissions.py --config configs/evaluate_coop.json --submissions submissions --episodes 3
```

Zero-shot league:

```bash
.venv/bin/python scripts/evaluate_coop_zero_shot.py --config configs/evaluate_coop_zero_shot.json --submissions submissions --episodes 2
```

主な指標:

- `avg_score`
- `avg_soups`
- `avg_collisions`
- `avg_invalid_interacts`
- zero-shot の `seen`, `unseen_same_family`, `heldout_family`

## 6. 簡易動作チェック

授業前に次を実行します。

```bash
.venv/bin/python scripts/smoke_check.py
.venv/bin/python scripts/train_bomber.py --config configs/train_bomber.json --episodes 2 --out outputs/check_bomber
.venv/bin/python scripts/train_bomber_dqn.py --config configs/train_bomber_dqn.json --episodes 2 --out outputs/check_bomber_dqn
.venv/bin/python scripts/train_bomber_actor_critic.py --config configs/train_bomber_actor_critic.json --episodes 2 --out outputs/check_bomber_ac
.venv/bin/python scripts/train_coop_mappo.py --config configs/train_coop_mappo.json --episodes 2 --out outputs/check_mappo
.venv/bin/python scripts/train_coop_qmix.py --config configs/train_coop_qmix.json --episodes 2 --out outputs/check_qmix
.venv/bin/python scripts/train_coop_zero_shot_league.py --config configs/train_coop_zero_shot.json --episodes 2 --out outputs/check_zero
```

`--wandb` なしでも `metrics.csv` と `post_training_episode.gif` が作られていれば、学習曲線と実行例の記録はできています。

シェルスクリプトでまとめて確認する場合:

```bash
./run_smoke_teacher.sh
EPISODES=2 ./run_train_examples.sh
EPISODES=2 ./run_continue_examples.sh
```

## 7. Config 管理

学習用 config:

- `configs/train_bomber.json`
- `configs/train_bomber_dqn.json`
- `configs/train_bomber_actor_critic.json`
- `configs/train_coop_mappo.json`
- `configs/train_coop_qmix.json`
- `configs/train_coop_zero_shot.json`

評価用 config:

- `configs/evaluate_bomber.json`
- `configs/evaluate_coop.json`
- `configs/evaluate_coop_zero_shot.json`

各 run の出力先には `effective_config.json` が保存されます。これは CLI 上書き後の最終設定なので、W&B の config と照合できます。

## 8. 授業運用向けコード導線

- 学生向け入口: `scripts/train_*.py`
- 教師向け入口: `scripts/evaluate_*.py`
- 可視化: `scripts/visualize_episode.py`, `src/marl_course/visualization/pygame_viewer.py`
- 汎化評価: `scripts/evaluate_coop_zero_shot.py`

授業での説明は「train → evaluate → visualize」の順にすると、実装と採点の関係が伝わりやすいです。

## 9. 教師用評価・表示の自動化

提出物フォルダを指定してまとめて評価できます。

```bash
SUBMISSIONS=submissions EPISODES=3 ./run_evaluate_teacher.sh
```

教室で見せる `--live` 表示と、学生が生成するGIFは共通レンダラーを使います。  
そのため、凡例・色・爆弾カウント・鍋進捗の見た目は一致します。
また、`metadata.json` の `model_name` を読み、`A -> alice_dqn_10000` のようにエージェントとモデル名の対応も表示します。
