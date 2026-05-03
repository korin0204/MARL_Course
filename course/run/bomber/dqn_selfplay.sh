#!/usr/bin/env bash
# Bomber-man: DQN self-play with fixed opponent promotion.
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
EPISODES="${EPISODES:-10000}"
OUT="${OUT:-outputs/course_bomber_dqn_selfplay}"
MODEL_NAME="${MODEL_NAME:-student_dqnselfplay_${EPISODES}}"
OPPONENT_FROM="${OPPONENT_FROM:-}"

args=(
  course/student/bomber/train_dqn_selfplay.py
  --config configs/train_bomber_dqn_selfplay.json
  --episodes "$EPISODES"
  --out "$OUT"
  --model-name "$MODEL_NAME"
  --device auto
  --use-cnn
)
if [[ -n "$OPPONENT_FROM" ]]; then
  args+=(--opponent-from "$OPPONENT_FROM")
fi
"$PYTHON" "${args[@]}"
