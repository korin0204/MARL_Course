"""PyTorchの実行デバイスを安全に選ぶための補助関数。

授業では学生のPCがばらばらになりやすいため、学習スクリプト側では
`device=auto` を指定できるようにしておく。Apple Silicon MacではMPS、
それ以外ではCPUへ自動的に落とす。CUDAは転送コストや授業環境差を避けるため、
明示指定された場合だけ使う。
"""
from __future__ import annotations

import platform
from typing import Any


def resolve_torch_device(requested: str | None = "auto") -> str:
    """`auto` / `mps` / `cuda` / `cpu` などの指定を実際のdevice名へ解決する。

    `auto` は授業用の推奨値。Macのユニファイドメモリ環境ではPyTorchの
    MPSバックエンドが使える場合に `mps` を選ぶ。
    """

    import torch

    requested_device = (requested or "auto").lower()
    if requested_device == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("device='mps' was requested, but PyTorch MPS is not available in this environment.")
        return "mps"
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested, but CUDA is not available in this environment.")
        return requested_device
    if requested_device != "auto":
        return requested_device
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_torch_runtime(device: str) -> str:
    """選ばれたdeviceに合わせてPyTorchの軽い高速化設定を行う。

    現在のコードは授業で壊れにくいことを優先し、MPS非対応になりがちな
    実験的最適化は避けている。行列計算精度だけはPyTorch推奨の範囲で
    `high` にして、DQN/Actor-Criticの全結合層を少し速くしやすくする。
    """

    import torch

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device == "cpu":
        # 小規模環境ではCPUスレッドを増やし過ぎるとかえって遅くなることがある。
        # 既定値が極端に小さい場合だけ、授業用PCで妥当な範囲へ補正する。
        torch.set_num_threads(max(1, torch.get_num_threads()))
    return device


def torch_device_summary(device: str) -> dict[str, Any]:
    """W&B/effective_configに残すためのdevice情報を作る。

    `psutil` が入っていればメモリ容量も記録する。入っていない環境でも
    学習自体は止めないよう、追加情報はoptionalにしている。
    """

    import torch

    summary: dict[str, Any] = {
        "requested_runtime": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "resolved_device": device,
        "mps_built": bool(getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_built()),
        "mps_available": bool(getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device == "cuda" and torch.cuda.is_available():
        summary["cuda_name"] = torch.cuda.get_device_name(0)
    try:
        import psutil

        memory = psutil.virtual_memory()
        summary["system_memory_gb"] = round(memory.total / (1024**3), 2)
    except Exception:
        summary["system_memory_gb"] = None
    return summary
