#!/usr/bin/env bash
# 教師用: Overcooked提出モデルを未知マップ汎化でランキングする。
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" course/teacher/evaluate_overcooked_zero_shot.py --config configs/evaluate_coop_zero_shot.json "$@"
