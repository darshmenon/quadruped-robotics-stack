#!/usr/bin/env bash
# Run scripts/benchmark_gazebo.py with ROS, the workspace and Quad-SDK's local libs sourced.
#   scripts/benchmark_gazebo.sh --controller both --trials 3 --goal 5.0
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
source "$REPO_ROOT/ros2/install/setup.bash"
source "$REPO_ROOT/ros2/quad_sdk_external/setup_env.sh"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
exec /usr/bin/python3 "$REPO_ROOT/scripts/benchmark_gazebo.py" "$@"
