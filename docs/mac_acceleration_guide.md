# Mac学生向け 高速学習ガイド

Apple Silicon Mac はCPU/GPUがユニファイドメモリを共有するため、PyTorchの
MPSバックエンドを使うと、NN計算のdevice転送コストを抑えながら学習できます。
この教材では `device=auto` を指定すると、Macでは自動で `mps` を選びます。
Mac以外では再現性を優先して `cpu` を選びます。CUDAは明示指定した場合だけ使います。

## 1. セットアップ

ローカル開発ルールは通常と同じく `pyenv + venv` です。

```bash
pyenv local 3.11.0
python -m venv .venv
.venv/bin/python -m pip install -e ".[train,mac]"
```

`mac` extra はdevice確認時にメモリ容量を表示するための `psutil` を追加します。
PyTorch本体のMPS対応は通常の `torch` wheel に含まれます。

## 2. MPSが使えるか確認する

```bash
./run_mac_check_device.sh
```

出力例:

```json
{
  "resolved_device": "mps",
  "mps_available": true,
  "cuda_available": false,
  "system_memory_gb": 24.0
}
```

`resolved_device` が `mps` ならMac GPUでNN計算が走ります。`cpu` の場合は、
PyTorchやmacOSの組み合わせでMPSが使えない状態です。

## 3. Mac向けの実行例

DQN:

```bash
./run_mac_bomber_dqn_fast.sh
```

Actor-Critic:

```bash
./run_mac_bomber_actor_critic_fast.sh
```

学習長だけ変える:

```bash
EPISODES=50000 ./run_mac_bomber_dqn_fast.sh
```

出力先やモデル名も変えられます。

```bash
EPISODES=50000 \
OUT=outputs/alice_mac_dqn \
MODEL_NAME=alice_dqn_50000 \
./run_mac_bomber_dqn_fast.sh
```

## 4. configで指定する場合

`configs/train_bomber_dqn.json` と `configs/train_bomber_actor_critic.json` では、
次の値がMac向けの基本設定です。

```json
{
  "use_cnn": true,
  "device": "auto"
}
```

- `device: "auto"`: MPS、CUDA、CPUの順に自動選択する。
- `device: "mps"`: Mac GPUを明示指定する。MPSが使えない環境では失敗する。
- `device: "cpu"`: GPUを使わずCPUだけで動かす。
- `use_cnn: true`: 盤面の空間構造をCNNで処理する。GPUの恩恵を受けやすい。

学習開始時に `effective_config.json` へ次のような情報も保存されます。

```json
{
  "device": "mps",
  "torch_device": {
    "resolved_device": "mps",
    "mps_available": true,
    "torch": "2.x.x"
  }
}
```

W&Bを有効にしている場合、この情報もconfigとして残るので、教師は学生が
どの実行環境で学習したかを後から確認できます。

## 5. 速くなる部分と限界

速くなりやすい部分:

- DQN/Actor-CriticのCNN・MLP順伝播
- DQNのミニバッチ更新
- 大きめの `batch_size` や `hidden_dim` を使う実験

速くなりにくい部分:

- Pythonで書かれた環境step
- 1ステップごとの細かい方策呼び出し
- 小さすぎるMLPだけのモデル

そのため、Macで速度を活かす場合は `use_cnn=true` を基本にし、DQNでは
`batch_size` を `64` から `128` に上げる実験も候補になります。ただし、
大きすぎるbatchは初期学習の反応を鈍くすることがあるので、学習曲線を見ながら
調整してください。
