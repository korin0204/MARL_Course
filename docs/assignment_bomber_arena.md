# Bomber Arena 課題

目的は、最大4エージェント対戦で勝率の高い `Policy` を提出することです。

学生は学習時の報酬を自由に設計できます。教師評価では報酬関数は読み込まず、提出モデルを `deterministic=True` で実行し、勝敗と順位だけをスコア化します。

提出物:

```text
metadata.json
policy.py
policy.pt
```

`policy.py`:

```python
def load_policy(model_path: str, device: str = "cpu"):
    ...
```

返される policy は `act(obs, action_mask=None, deterministic=True) -> int` を持つ必要があります。
