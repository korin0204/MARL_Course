#!/usr/bin/env bash
# Overcooked: MAPPO zero-shot multi-map training entry.
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
EPISODES="${EPISODES:-160}"
OUT="${OUT:-outputs/course_overcooked_mappo_zero_shot}"
MODEL_NAME="${MODEL_NAME:-student_mappozeroshot_${EPISODES}}"

"$PYTHON" course/student/overcooked/train_mappo_zero_shot.py \
  --config configs/train_coop_zero_shot.json \
  --episodes "$EPISODES" \
  --out "$OUT" \
  --model-name "$MODEL_NAME"
