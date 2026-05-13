#!/bin/bash
# Train a quadruped walking policy.
#
# Backends:
#   mujoco  - MuJoCo + Stable-Baselines3 PPO (headless, no Isaac Gym needed)
#   gazebo  - Gazebo headless + ROS2 + SB3 PPO (requires sourced ROS2)
#   isaac   - Isaac Gym + rsl_rl (requires Isaac Gym installed)
#
# Usage:
#   ./scripts/train_policy.sh                      # mujoco, go2, headless
#   ./scripts/train_policy.sh mujoco               # MuJoCo backend (default)
#   ./scripts/train_policy.sh mujoco --timesteps 5000000
#   ./scripts/train_policy.sh gazebo               # Gazebo backend
#   ./scripts/train_policy.sh isaac go2            # Isaac Gym backend

set -e

BACKEND=${1:-mujoco}
PYTHON_BIN=${PYTHON:-python3}
shift || true  # remaining args forwarded to training script

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

case "$BACKEND" in
  mujoco)
    echo "Backend: MuJoCo (headless)"
    "$PYTHON_BIN" "$ROOT/training/train_mujoco.py" "$@"
    ;;
  gazebo|gz)
    echo "Backend: Gazebo (headless)"
    source /opt/ros/humble/setup.bash 2>/dev/null || true
    source "$ROOT/ros2/install/setup.bash" 2>/dev/null || true
    "$PYTHON_BIN" "$ROOT/training/train_gazebo.py" "$@"
    ;;
  isaac)
    echo "Backend: Isaac Gym"
    TASK=${1:-go2}
    shift || true
    cd "$ROOT/training"
    "$PYTHON_BIN" legged_gym/scripts/train.py --task="$TASK" "$@"
    ;;
  *)
    echo "Unknown backend: $BACKEND. Use mujoco | gazebo | isaac"
    exit 1
    ;;
esac
