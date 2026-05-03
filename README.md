# MARL Course Games

授業用の MARL 教材プロジェクトです。競争型の `Bomber Arena` と、4エージェント協調型の `Coop Kitchen` を提供します。

## ローカル開発

このプロジェクトではローカル開発でも `pyenv + venv` を必須にします。

```bash
pyenv local 3.11.0
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
```

Apple Silicon MacでGPU(MPS)を使った学習を確認する場合:

```bash
.venv/bin/python -m pip install -e ".[train,mac]"
./course/run/check_device.sh
./course/run/bomber/dqn_vs_baselines.sh
```

依存を入れなくても、環境と評価の smoke path は標準ライブラリだけで動くようにしてあります。

```bash
.venv/bin/python scripts/smoke_check.py
```

## 主なコマンド

授業中に学生・教師へ案内する正規入口は `course/` です。

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

詳細な対応関係は `course/README.md` にあります。旧 `scripts/` は内部実装・互換用として残しています。

```bash
.venv/bin/python scripts/train_bomber.py --config configs/train_bomber.json
.venv/bin/python scripts/train_bomber_dqn.py --config configs/train_bomber_dqn.json
.venv/bin/python scripts/train_bomber_actor_critic.py --config configs/train_bomber_actor_critic.json
.venv/bin/python scripts/train_coop_mappo.py --config configs/train_coop_mappo.json
.venv/bin/python scripts/train_coop_qmix.py --config configs/train_coop_qmix.json
.venv/bin/python scripts/train_coop_zero_shot_league.py --config configs/train_coop_zero_shot.json
.venv/bin/python scripts/evaluate_bomber_tournament.py --config configs/evaluate_bomber.json --live-ascii
.venv/bin/python scripts/evaluate_coop_submissions.py --config configs/evaluate_coop.json --live-ascii
.venv/bin/python scripts/evaluate_coop_zero_shot.py --config configs/evaluate_coop_zero_shot.json --live-ascii
```

`--live` は pygame が入っている環境で教室スクリーン表示に使います。依存なしの確認には `--live-ascii` を使います。
GIF再生成や一括実行には次も使えます。

```bash
./course/run/teacher/evaluate_bomber_tournament.sh --live-ascii
./course/run/teacher/evaluate_overcooked_ranking.sh --live-ascii
```

詳しい使い方:

- 学生向け: `docs/student_training_guide.md`
- 教師向け: `docs/teacher_training_guide.md`
- コード読解: `docs/code_reading_guide.md`
- config解説: `docs/config_reference.md`
- Mac高速化: `docs/mac_acceleration_guide.md`
