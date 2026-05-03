#!/usr/bin/env bash
# Bomber-man: Actor-Critic vs rule/random baselines.
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
EPISODES="${EPISODES:-10000}"
OUT="${OUT:-outputs/course_bomber_actor_critic_baseline}"
MODEL_NAME="${MODEL_NAME:-student_actorcritic_${EPISODES}}"

"$PYTHON" course/student/bomber/train_actor_critic_vs_baselines.py \
  --config configs/train_bomber_actor_critic.json \
  --episodes "$EPISODES" \
  --out "$OUT" \
  --model-name "$MODEL_NAME" \
  --device auto \
  --use-cnn
