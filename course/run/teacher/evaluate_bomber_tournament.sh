#!/usr/bin/env bash
# 教師用: Bomber-man提出モデルをトーナメント評価する。
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" course/teacher/evaluate_bomber_tournament.py --config configs/evaluate_bomber.json "$@"
