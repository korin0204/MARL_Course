#!/usr/bin/env bash
# MacならMPS、その他ならCPUに解決されるかを確認する。
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" scripts/check_torch_device.py --device auto
