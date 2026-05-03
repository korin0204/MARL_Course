# Coop Kitchen 課題

目的は、4エージェントチームとしてできるだけ多くの soup を提供する `TeamPolicy` を作ることです。

標準課題では固定3レイアウトを使います。発展課題 `overcooked-zero-shot-league` では、訓練時に見ないマップ形状に対する汎化性能を評価します。

提出 policy:

```python
class TeamPolicy:
    def act(self, obs, action_mask=None, deterministic=True) -> list[int]:
        ...

    def reset_episode(self, layout_metadata=None):
        ...
```

`reset_episode` は任意です。RNN や履歴バッファを使う場合に実装してください。
