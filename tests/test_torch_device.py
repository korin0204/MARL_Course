from __future__ import annotations

from marl_course.common.torch_device import configure_torch_runtime, resolve_torch_device, torch_device_summary


def test_resolve_torch_device_accepts_explicit_cpu() -> None:
    """明示指定したCPUは、GPU有無に関係なくそのまま使える。"""

    assert resolve_torch_device("cpu") == "cpu"


def test_auto_device_has_summary() -> None:
    """auto解決結果はconfig/W&Bへ保存できるdictとして説明できる。"""

    device = configure_torch_runtime(resolve_torch_device("auto"))
    summary = torch_device_summary(device)
    assert summary["resolved_device"] in {"cpu", "cuda", "mps"}
    assert "mps_available" in summary
    assert "torch" in summary
