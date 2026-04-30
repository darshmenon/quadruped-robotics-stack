#!/bin/bash
# Train a quadruped walking policy using Unitree RL Gym (Isaac Gym backend)
#
# Usage:
#   ./scripts/train_policy.sh go2          # train Unitree Go2 (default)
#   ./scripts/train_policy.sh h1           # train Unitree H1
#   ./scripts/train_policy.sh g1           # train Unitree G1
#   ./scripts/train_policy.sh go2 --headless  # no GUI (faster)
#
# Registered tasks: go2, h1, h1_2, g1
#
# Requirements:
#   - Isaac Gym installed (https://developer.nvidia.com/isaac-gym)
#   - pip install -e training/

set -e

TASK=${1:-go2}
HEADLESS=${2:-""}
PYTHON_BIN=${PYTHON:-python3}

echo "Training task: $TASK"

cd training
"$PYTHON_BIN" legged_gym/scripts/train.py --task="$TASK" $HEADLESS
