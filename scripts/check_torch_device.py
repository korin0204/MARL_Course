"""Mac/Colab/CPU環境でPyTorchがどのdeviceを使うか確認するスクリプト。"""
from __future__ import annotations

import argparse
import json

import torch

from marl_course.common.torch_device import configure_torch_runtime, resolve_torch_device, torch_device_summary


def main() -> None:
    """`--device auto` の解決結果と、簡単なテンソル計算の成否を表示する。"""

    parser = argparse.ArgumentParser(description="Check PyTorch device for MARL course training.")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, cpu など。通常はauto推奨。")
    args = parser.parse_args()

    device = configure_torch_runtime(resolve_torch_device(args.device))
    summary = torch_device_summary(device)
    x = torch.randn(256, 256, device=device)
    y = torch.mm(x, x)
    summary["matmul_mean"] = float(y.mean().detach().cpu().item())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
