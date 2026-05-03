#!/usr/bin/env bash
# Bomber-man: Actor-Critic self-play with fixed opponent promotion.
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
EPISODES="${EPISODES:-10000}"
OUT="${OUT:-outputs/course_bomber_actor_critic_selfplay}"
MODEL_NAME="${MODEL_NAME:-student_actorcriticselfplay_${EPISODES}}"
OPPONENT_FROM="${OPPONENT_FROM:-}"

args=(
  course/student/bomber/train_actor_critic_selfplay.py
  --config configs/train_bomber_actor_critic_selfplay.json
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
