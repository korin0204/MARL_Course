#!/usr/bin/env bash
# 教師用: Overcooked提出モデルを平均スコアでランキングする。
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" course/teacher/evaluate_overcooked_ranking.py --config configs/evaluate_coop.json "$@"
