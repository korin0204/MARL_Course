#!/usr/bin/env bash
# Overcooked: MAPPO single-map training entry.
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
EPISODES="${EPISODES:-120}"
OUT="${OUT:-outputs/course_overcooked_mappo}"
MODEL_NAME="${MODEL_NAME:-student_mappo_${EPISODES}}"

"$PYTHON" course/student/overcooked/train_mappo.py \
  --config configs/train_coop_mappo.json \
  --episodes "$EPISODES" \
  --out "$OUT" \
  --model-name "$MODEL_NAME"
