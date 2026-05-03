#!/usr/bin/env bash
# Overcooked: Q-MIX single-map training entry.
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
EPISODES="${EPISODES:-120}"
OUT="${OUT:-outputs/course_overcooked_qmix}"
MODEL_NAME="${MODEL_NAME:-student_qmix_${EPISODES}}"

"$PYTHON" course/student/overcooked/train_qmix.py \
  --config configs/train_coop_qmix.json \
  --episodes "$EPISODES" \
  --out "$OUT" \
  --model-name "$MODEL_NAME"
