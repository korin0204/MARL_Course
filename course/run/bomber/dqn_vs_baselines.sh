#!/usr/bin/env bash
# Bomber-man: DQN vs rule/random baselines.
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
EPISODES="${EPISODES:-10000}"
OUT="${OUT:-outputs/course_bomber_dqn_baseline}"
MODEL_NAME="${MODEL_NAME:-student_dqn_${EPISODES}}"

"$PYTHON" course/student/bomber/train_dqn_vs_baselines.py \
  --config configs/train_bomber_dqn.json \
  --episodes "$EPISODES" \
  --out "$OUT" \
  --model-name "$MODEL_NAME" \
  --device auto \
  --use-cnn
