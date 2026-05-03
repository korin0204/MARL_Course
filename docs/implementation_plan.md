# MARL 授業用ゲーム教材 実装計画書

作成日: 2026-05-03

## 目的

学生が MARL を「学習して強くなる」体験として理解できるように、対戦型の Bomberman 系環境と協調型の Overcooked 系環境を同一リポジトリで提供する。成果物は、Google Colab で動く入門コード、高度な学生向けの pyenv + venv 前提の Python プロジェクト、教師が提出モデルを集計・評価する実行環境、学習・評価の様子を見られるビジュアライザーで構成する。

本教材はゲームルール・体験を参考にした教育用再実装とする。公式アセット、ロゴ、キャラクター画像、商標を含む素材は同梱しない。パッケージ名や画面表示も `bomber_arena` / `coop_kitchen` のような教育用名称を使い、ドキュメント内でのみ参照元ゲーム名を説明する。

## 調査メモ

- KONAMI 公式の Super Bomberman R 2 では、Standard は複数セット制で「各セットとも最後まで生き残ることができれば勝者」と説明されている。Battle 64 は最後の1人になるまでの生き残り、Grand Prix はチーム戦で Basic Bomber と Crystals がある。授業用 Bomberman は Standard の生存勝利を主軸にする。
- Super Bomberman R Online 公式マニュアルでは、Standard は最大16人・ソロまたはチーム戦で、勝者は最後まで生き残ったプレイヤー/チームとされる。授業要件は最大4エージェントなので、4人対戦に縮約する。
- Overcooked の公式 Steam ページでは、1から4人の協力料理ゲームで、チームで注文を準備・調理・提供するゲームとして説明されている。授業用 Overcooked は4エージェント協調を標準にする。
- HumanCompatibleAI の Overcooked-AI は、スープをできるだけ速く届ける完全協調ベンチマークで、3個までの材料を鍋に入れ、調理後に皿で取り出して提供する。既存コードは Overcooked の MDP、Gym 風環境、可視化、プランナーを持つが、標準の研究設定は2エージェント寄りで、README 上でも旧 DRL/BC 実装は deprecated とされている。そのため、本教材では API と状態設計を参考にしつつ、4エージェント教育用の軽量環境を自前実装する。
- MAPPO は centralized training / decentralized execution の PPO 系手法として、共同報酬タスクで強いベースラインになる。授業では「共有 actor + agent id + centralized critic」を標準実装にする。
- QMIX は各エージェントの局所 Q を単調な mixing network で統合し、集中学習・分散実行を可能にする。授業では MAPPO と対比しやすい value-based 協調手法として実装する。

## 全体成果物

リポジトリ構成案:

```text
.
├── pyproject.toml
├── .python-version
├── README.md
├── docs/
│   ├── implementation_plan.md
│   ├── assignment_bomber_arena.md
│   ├── assignment_coop_kitchen.md
│   └── grading_guide.md
├── notebooks/
│   ├── 01_bomber_arena_student_colab.ipynb
│   ├── 02_coop_kitchen_mappo_colab.ipynb
│   ├── 03_coop_kitchen_qmix_colab.ipynb
│   └── 04_coop_kitchen_zero_shot_league_colab.ipynb
├── src/marl_course/
│   ├── common/
│   ├── envs/bomber_arena/
│   ├── envs/coop_kitchen/
│   ├── algos/
│   ├── policies/
│   ├── evaluation/
│   └── visualization/
├── scripts/
│   ├── train_bomber.py
│   ├── train_coop_mappo.py
│   ├── train_coop_qmix.py
│   ├── train_coop_zero_shot_league.py
│   ├── evaluate_bomber_tournament.py
│   ├── evaluate_coop_submissions.py
│   ├── evaluate_coop_zero_shot.py
│   ├── visualize_episode.py
│   └── package_submission.py
├── submissions/
│   └── README.md
└── tests/
```

Python は 3.11 系を標準にする。Colab では `pip install -e .`、ローカルではこの開発作業自体も含めて必ず `pyenv local 3.11.x` と `python -m venv .venv` を使う。本番環境や system Python を汚染しないことを開発ルールにする。主要依存は `numpy`, `gymnasium`, `pettingzoo`, `torch`, `pygame`, `matplotlib`, `imageio`, `tqdm`, `pydantic`, `pyyaml`, `tensorboard`, `pytest` に抑える。分散学習フレームワークは初期版では入れず、学習の見通しを優先する。

ローカル開発コマンドは `.venv/bin/python` / `.venv/bin/pip` または venv を activate した状態の `python` / `pip` に限定する。依存追加、テスト、notebook 変換、可視化確認も venv の中で行う。新しい shell セッションでは、作業開始時に `pyenv version` と `which python` を確認する。

## 共通 API

PettingZoo Parallel API 互換の環境 API を共通化する。同時行動ゲームなので、内部標準は agent id ごとの辞書形式にする。学習コード側では、これを batched tensor に変換する adapter を用意する。

```python
observations, infos = env.reset(seed=seed)
observations, rewards, terminations, truncations, infos = env.step(actions)
global_state = env.state()
frame = env.render(mode="rgb_array")
```

提出モデルは PyTorch の `state_dict` と `metadata.json` に統一する。

Bomberman 提出:

```text
student_id/
├── policy.pt
├── metadata.json
└── policy.py
```

`policy.py` は次の関数またはクラスを必須にする。

```python
def load_policy(model_path: str, device: str):
    ...

class Policy:
    def act(self, obs, action_mask=None, deterministic=True) -> int:
        ...
```

Overcooked 提出:

```python
class TeamPolicy:
    def act(self, obs, action_mask=None, deterministic=True) -> list[int]:
        # 4エージェント分の action を返す
        ...
```

教師環境は学生コードを隔離ロードし、観測次元・行動数・metadata のバージョンを検証してから評価する。授業期間中に環境バージョンを変える場合は `env_spec_version` を上げ、旧提出物と混ぜない。

初期ベースラインとして、両環境に `RandomPolicy`, `StayPolicy`, `Greedy/ScriptedPolicy` を同梱する。これにより、学生は学習前に環境と可視化を確認でき、教師は評価環境の smoke test をすぐ実行できる。

### 学習用報酬カスタマイズ

Colab とローカルプロジェクトの両方で、学生が学習用報酬を自由に設計できるようにする。環境本体はイベントログと標準スコアを返し、報酬は `RewardFn` / `RewardConfig` で後段計算する。

```python
class RewardFn:
    def reset(self, env, initial_obs):
        ...

    def __call__(self, transition, events, info) -> dict[str, float]:
        # agent_id ごとの学習用 reward を返す
        ...
```

学生は Colab 上で、勝敗、破壊、距離、危険マス、配達、役割分担、衝突、停滞など任意の特徴から報酬を作れる。提出評価では学生の `RewardFn` はロードしない。教師評価は提出ポリシーを決定的に実行し、ゲーム結果だけを採点する。

Bomberman 系と Overcooked 系で観測次元や state 次元を揃える必要はない。各環境の API バージョン、観測仕様、提出インターフェースだけを固定する。

## Bomberman 環境計画

### 採用ルール

- 完全観測、最大4エージェント対戦。
- Standard ルール相当の「最後まで生き残ったエージェントが勝利」。
- グリッドは classic 風の初期値 `11 x 13`。外周と市松状の固定ブロック、ランダムな破壊可能ブロック、四隅の安全な初期スポーン。初学者向けに `7 x 7` / `9 x 9`、研究環境比較向けに `11 x 11` も layout preset として用意する。
- 同時行動。行動は `stay`, `up`, `down`, `left`, `right`, `bomb` の6種類。
- 爆弾はタイマー後に十字方向へ爆風を出す。固定ブロックで止まり、破壊可能ブロックを壊す。爆弾の誘爆あり。
- 各エージェントは初期爆弾数1、初期火力2。破壊可能ブロックから `bomb_up`, `fire_up` が一定確率で出る。
- 複雑化しすぎる要素は初期版に入れない。具体的には、キック、パンチ、投げ、速度差、乗り物、復讐カート、突然死、キャラクター固有能力は入れない。

### 学習報酬と評価スコア

Bomber Arena の教師評価は、提出モデルを `argmax` または `deterministic=True` で再生し、実際の対戦結果から勝者・順位をスコア化する。したがって評価時には環境報酬そのものは本質的に不要であり、学生が学習中に使った reward design も採点には関与しない。

環境は学習の便宜として標準イベントと reference reward を返す。

- reference sparse reward: 勝利 `+1.0`、それ以外 `0.0`
- event log: `block_destroyed`, `powerup_collected`, `agent_eliminated`, `self_eliminated`, `entered_danger`, `survived_step` など
- score event: `winner_id`, `rankings`, `survival_steps`, `elimination_causes`

Colab では学生が高い自由度で reward を変えられるように、次の例をテンプレートとして提供する。

- 勝利・順位ベース reward
- 破壊可能ブロックや powerup を使った探索 reward
- 危険マップや自爆回避 reward
- 相手撃破や盤面制圧 reward
- 生存時間、接敵距離、爆弾設置頻度への reward/penalty

ただし教師評価の成績は reward 合計ではなく、対戦の勝敗・順位・安定性だけで計算する。

### 観測・出力固定

観測は `dict` とし、ニューラルネットで扱いやすい固定テンソルにする。

```python
obs = {
    "grid": np.ndarray(shape=(C, 11, 13), dtype=np.float32),
    "stats": np.ndarray(shape=(4, S), dtype=np.float32),
    "self_id": int,
    "alive_mask": np.ndarray(shape=(4,), dtype=np.float32),
    "action_mask": np.ndarray(shape=(6,), dtype=np.float32),
}
```

`grid` の初期チャンネル案:

- 固定ブロック
- 破壊可能ブロック
- 爆弾タイマー
- 爆弾火力
- 爆風残り時間
- `bomb_up`
- `fire_up`
- エージェント0から3の位置
- 危険マップ

出力は6 action の logits または action index。教師評価では action mask 違反時に `stay` へフォールバックし、違反率も成績レポートに記録する。環境自体は PettingZoo 互換として `observations[agent_id]` に同じ構造を返し、提出用 adapter が上記の単一 agent obs に整形する。

### 学生用学習コード

入門版:

- ルールベース相手3体を固定。
- DQN または PPO の単一エージェント学習。
- replay buffer、epsilon-greedy、学習曲線、評価動画生成までを Colab で完結。

発展版:

- independent learners による自己対戦。
- 共有 replay buffer による share-exp。
- 過去 checkpoint を相手 pool に入れる league self-play。
- 学生が architecture を差し替えられる `StudentNet` テンプレート。

### 教師用評価

- 提出モデルを全員分ロードし、4人対戦の round-robin tournament を実行。
- 各モデルは `deterministic=True` で action logits の argmax 再生を行う。確率サンプリングを許す場合は、別枠の stochastic exhibition として扱う。
- 参加者数が多い場合は予選グループ + 決勝、または Swiss 方式。
- 各組み合わせは spawn 位置 permutation と複数 seed で評価。
- スコアは `平均勝率`, `平均順位点`, `直接対戦勝率`, `引き分け率`, `自爆率`, `平均生存ターン`, `action_mask違反率`, `クラッシュ率` を CSV/JSON に出力。
- モデルのロード失敗、例外、時間超過は自動で失格扱いまたは rule-based fallback とし、ログを残す。
- 授業中に教室前方のスクリーンで見せることを想定し、評価 CLI は `--live` で pygame viewer を開き、試合名、参加者名、現在順位、残り時間、勝者、主要イベントをオーバーレイ表示する。

## Overcooked 系環境計画

### 採用ルール

4エージェント完全協調を標準にする。公式 Overcooked の体験に合わせ、注文を満たすために材料取得、鍋投入、調理待ち、皿取得、提供、カウンター受け渡しを行う。

初期版の料理は onion soup のみとする。

- 鍋に onion を3個入れる。
- 一定ステップ調理する。
- 皿を持ったエージェントが soup を取り出す。
- delivery マスへ持って行くと得点。

行動は Overcooked-AI と同じ6種類に揃える。

```text
up, down, left, right, stay, interact
```

4人協調で学習を安定させるため、初期マップは3種類に限定する。

- `open_kitchen_4p`: 役割分担が自然に起きる広めの基本マップ。
- `corridor_4p`: 通路混雑と受け渡しが必要なマップ。
- `split_station_4p`: 材料・皿・鍋・提出先が分かれ、協調が必要なマップ。

ランダム性は seed で完全再現可能にする。採点では公開 seed と非公開 seed を分ける。

衝突解決は決定的かつ公平にする。同じマスへの複数移動は全員 stay、位置交換は禁止、壁・カウンター・鍋・配達口など進入不可 terrain への移動は stay とする。固定優先順位は学習バイアスになるため使わない。

### 報酬・スコア

環境スコアは提出された soup 数を主指標にする。

- soup delivery: `+20`
- episode horizon: 400 step を初期値
- 採点スコア: 複数レイアウト・複数 seed の平均 delivery score

学習補助として shaped reward を提供する。

- onion を鍋に入れる
- 皿を取る
- 調理済み soup を取る
- soup を提供する
- 不要な interact や長時間停滞への軽いペナルティ

採点では sparse score を主指標にし、shaped reward は参考ログに留める。

### 観測・出力固定

学生は1つの TeamPolicy で4エージェント分の方策を出力する。

```python
obs = {
    "agent_obs": np.ndarray(shape=(4, C, H, W), dtype=np.float32),
    "global_state": np.ndarray(shape=(G,), dtype=np.float32),
    "agent_features": np.ndarray(shape=(4, F), dtype=np.float32),
    "action_mask": np.ndarray(shape=(4, 6), dtype=np.float32),
    "layout_id": int,
}
```

MAPPO では `agent_obs + agent_id` を actor に入れ、`global_state` を centralized critic に入れる。QMIX では各 agent の局所観測から `Q_i(a_i)` を出し、`global_state` を mixing network の hypernetwork に入れる。

提出時は `TeamPolicy.act(...) -> list[int]` を固定し、MAPPO でも QMIX でも評価側から見たインターフェースを同じにする。

固定3レイアウト課題では、レイアウトごとに `H, W` が同じである必要はない。ただし batch 学習を簡単にするため、標準レイアウトは同じ `H x W` に揃える。異なる形のレイアウトを扱う場合は padding と mask を使い、実マップ外を `out_of_bounds` channel として明示する。

### Overcooked Zero-Shot League

既存の固定3レイアウト評価は残しつつ、発展課題として `overcooked-zero-shot-league` を追加する。目的は、訓練中に見たレイアウトだけに過適合したモデルではなく、未知形状の厨房でも役割分担・探索・運搬・提供を成立させる汎化性能を測ることである。

構成:

- 訓練用 layout family: `open`, `corridor`, `split_station`, `bottleneck`, `island`, `long_delivery` などを procedural generator で多数生成する。
- 検証用 layout family: 訓練と同じ family だが seed 非公開。
- ゼロショット評価 layout family: 訓練に出ない形状、通路幅、station 配置、鍋数、皿置き場、delivery 位置を含む。
- 料理ルールは初期版と同じ onion soup に固定し、未知要素を「マップ形状と station 配置」に集中させる。
- 学生には訓練 generator と公開 validation layouts を渡し、教師は非公開 generator seed と held-out family で評価する。

観測設計:

- zero-shot league では Overcooked 内の観測次元を揃える。可変サイズ map は `max_h x max_w` に padding し、`valid_cell_mask` と `terrain/channel mask` を付ける。
- `layout_embedding` は教師から直接与えない。モデルが盤面チャンネルから構造を読むことを基本にする。
- 局所観測だけでなく、完全観測版と履歴付き版を用意する。履歴は frame stack、前回 action、前回所持品、直近イベントを選べるようにする。
- RNN/Transformer policy を発展実装として許容し、未知マップでの探索・役割維持を学習できるようにする。

学習設計:

- `MultiLayoutSampler` が episode ごとにレイアウトをサンプルする。
- curriculum として、最初は固定3レイアウト、その後 procedural layouts、最後に held-out validation で評価する。
- MAPPO は shared actor + centralized critic のまま、critic には padded global state と valid mask を渡す。
- QMIX は episode replay に `layout_id`, `valid_cell_mask`, `avail_actions`, `state` を保存する。
- 評価指標は平均 score に加えて、seen layout score、unseen same-family score、held-out family score、score drop、seed 分散を出す。

提出インターフェース:

```python
class TeamPolicy:
    def act(self, obs, action_mask=None, deterministic=True) -> list[int]:
        ...

    def reset_episode(self, layout_metadata=None):
        # RNN hidden state や履歴バッファを初期化する任意 hook
        ...
```

`reset_episode` は任意だが、zero-shot league では履歴モデルを自然に扱えるように評価側が呼び出す。未実装の場合は no-op とする。

### 学生用学習コード

入門 MAPPO:

- 共有 actor、centralized critic。
- GAE、clip objective、entropy bonus、value normalization。
- notebook では1レイアウト短時間学習、プロジェクト版では複数 parallel env。

発展 QMIX:

- recurrent agent Q network は発展扱い。初期版は MLP/CNN + frame stack で開始。
- mixing network は非負重み制約を実装。
- replay buffer は episode 単位で保存し、action mask と terminated/truncated を含める。

追加課題:

- MAPPO と QMIX の比較レポート。
- レイアウト汎化評価。
- shaped reward の有無による学習曲線比較。

### 教師用評価

- 各提出 TeamPolicy を単独で4エージェントチームとして実行。
- 公開3レイアウト + 非公開追加レイアウトで評価可能にする。
- seed ごとの `score`, `delivered_soups`, `collisions`, `invalid_interacts`, `invalid_actions`, `agent_event_counts`, `timeout/crash`, `episode_length` を保存。
- 評価結果は CSV、JSON、HTML 簡易レポートに出す。
- 上位モデルは replay と mp4/gif を自動生成する。
- 授業中のスクリーン表示用に、`--live` で pygame viewer を開き、現在の seed、layout 名、提出者名、score、delivery 数、衝突数、主要イベントをリアルタイム表示する。
- zero-shot league では、通常評価とは別に seen/unseen/held-out の leaderboard を作り、未知マップでの性能低下も成績レポートに含める。

## ビジュアライザー

共通 replay 形式を定義する。

```json
{
  "env_id": "bomber_arena_v1",
  "seed": 123,
  "layout": "...",
  "steps": [
    {"obs_summary": {}, "actions": [], "rewards": [], "events": []}
  ],
  "final_score": {}
}
```

可視化は3層で提供する。

- Colab: `matplotlib.animation` / `imageio` で GIF/MP4 生成。
- ローカル: `pygame` で再生、一時停止、ステップ送り、エージェント別 action 表示。
- 教師評価: 上位試合・異常試合を一括レンダリングする `scripts/visualize_episode.py`。

アセットは単純なタイル・色・アイコンを自前描画する。公式画像は使わない。

## 実装分担とワークツリー計画

初期コミット後、次の branch/worktree に分ける。

- `codex/env-core`: 共通 API、型、seed 管理、提出モデル loader、replay schema。
- `codex/bomber-arena`: Bomberman 系環境、ルールベース bot、単体テスト。
- `codex/coop-kitchen`: 4人 Overcooked 系環境、レイアウト、単体テスト。
- `codex/coop-zero-shot-league`: procedural layout generator、padding/mask 観測、zero-shot 評価。
- `codex/algorithms`: DQN/PPO/MAPPO/QMIX の教育用実装。
- `codex/evaluation`: 教師用 tournament/evaluation CLI、CSV/HTML レポート。
- `codex/visualization`: pygame viewer、notebook render helper、動画出力。
- `codex/notebooks-docs`: Colab notebooks、課題文、採点ガイド。

相互依存を減らすため、最初に `env-core` を作り、環境2本は同じ `MultiAgentEnv` protocol に合わせて実装する。アルゴリズム・評価・可視化はその protocol にだけ依存させる。

## 実装フェーズ

### Phase 0: プロジェクト基盤

- `pyproject.toml`, `.python-version`, `.gitignore`, `README.md`。
- pyenv + venv 手順。
- pytest と最小 CI 相当のローカルコマンド。
- 乱数 seed の共通ユーティリティ。

完了条件:

- `python -m pytest` が空または smoke test で通る。
- Colab から `pip install -e .` できる。

### Phase 1: 共通環境 API と提出 API

- `MultiAgentEnv` protocol。
- `Policy` / `TeamPolicy` loader。
- action mask 検証。
- replay writer/reader。

完了条件:

- ダミー環境とダミーポリシーで評価・replay 保存が動く。

### Phase 2: Bomber Arena

- ルールエンジン、爆弾・爆風・誘爆・ブロック破壊。
- 完全観測 encoding。
- ルールベース bot。
- DQN/PPO 学習サンプル。
- round-robin tournament。
- pygame/Colab 可視化。

完了条件:

- 1000 seed 分の deterministic reset/step テスト。
- 爆風、誘爆、勝敗、action mask の単体テスト。
- ルールベース bot 同士の評価が最後まで落ちずに走る。

### Phase 3: Coop Kitchen 4P

- グリッド、材料、鍋、調理、皿、提供、カウンター。
- 3レイアウト。
- 完全協調 reward と shaped reward。
- MAPPO と QMIX の baseline。
- evaluation CLI。
- pygame/Colab 可視化。

完了条件:

- 手書き scripted policy が soup を提供できる。
- MAPPO/QMIX の短時間 smoke training が loss を更新し、checkpoint を保存できる。
- 提出 TeamPolicy が複数 seed 評価でスコア化される。

### Phase 4: Overcooked Zero-Shot League

- procedural layout generator。
- padded observation と valid mask。
- `MultiLayoutSampler`。
- 履歴付き/RNN対応 policy adapter。
- zero-shot evaluation CLI と leaderboard。
- Colab notebook `04_coop_kitchen_zero_shot_league_colab.ipynb`。

完了条件:

- 訓練 layout と held-out layout が seed で再現できる。
- 同じ提出モデルを seen/unseen/held-out に分けて評価できる。
- live viewer で未知マップ評価を教室スクリーンに表示できる。

### Phase 5: 授業パッケージ化

- Colab notebook 4本。
- 学生向け課題文。
- 教師向け採点手順。
- サンプル提出 zip。
- よくあるエラーと復旧方法。

完了条件:

- 初学者が notebook を上から実行して提出物を作れる。
- 教師が `submissions/` に zip を置き、1コマンドで成績 CSV を作れる。

## テスト方針

- ルール単体テスト: 移動、衝突、interaction、爆風、調理、提供。
- seed 再現性テスト: 同一 seed と同一 action sequence で完全一致。
- observation shape テスト: バージョン固定。
- action mask テスト: 不正行動を環境が一貫処理。
- submission loader テスト: 正常、metadata 不一致、例外、タイムアウト。
- smoke training: 100から1000 step 程度で checkpoint 保存まで確認。
- renderer テスト: `rgb_array` が空でなく、想定サイズになる。

## リスクと対策

- Sparse reward だけでは学習が進みにくい: 学習用 reward hook と scripted/baseline bot を提供し、採点はゲーム結果のスコアリングに固定する。
- Overcooked 4人版が複雑になりすぎる: 初期料理を onion soup のみにし、汚れ皿・投げ・動的地形は拡張扱いにする。
- Zero-shot league が難しすぎる: 固定3レイアウト課題を標準課題として残し、zero-shot は発展課題または加点評価にする。
- 学生提出コードの実行が不安定: loader で time limit、例外処理、metadata 検証、CPU 評価モードを徹底する。
- Colab の実行時間が足りない: notebook は短時間で動く baseline を優先し、強化はローカル発展版に誘導する。
- 公式ゲームとの混同: 教材名、アセット、README に教育用再実装であることを明記する。

## 参照 URL

- KONAMI, Super Bomberman R 2 Battle: https://www.konami.com/games/bomberman/r2/jp/ja/battle/
- KONAMI, Super Bomberman R 2 Battle English: https://www.konami.com/games/bomberman/r2/eu/en/battle/
- KONAMI, Super Bomberman R Online manual Standard: https://dds.konami.com/games/manual/sbro/multi/pt/pc/page04.html
- Steam, Overcooked official store page: https://store.steampowered.com/app/448510/overcooked/
- HumanCompatibleAI, Overcooked-AI: https://github.com/HumanCompatibleAI/overcooked_ai
- PettingZoo Parallel API: https://pettingzoo.farama.org/api/parallel/
- MultiAgentLearning, Pommerman playground: https://github.com/MultiAgentLearning/playground
- NeurIPS 2022, The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games: https://papers.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html
- Oxford CS, QMIX publication page: https://www.cs.ox.ac.uk/publications/publication12036-abstract.html
