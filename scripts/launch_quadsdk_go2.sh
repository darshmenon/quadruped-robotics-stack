#!/usr/bin/env bash
# Launch the Quad-SDK NMPC locomotion backend for Go2 in Gazebo Harmonic.
# Prereqs (one-time): ./scripts/setup_quadsdk_apt_deps.sh (needs your sudo password)
#                      ./scripts/build_quadsdk_local_libs.sh (no sudo)
#                      cd ros2 && colcon build --symlink-install --cmake-args \
#                        -DQUADSDK_DEPS_PREFIX=$(pwd)/quad_sdk_external/local
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/humble/setup.bash
source "$REPO_ROOT/ros2/install/setup.bash"
source "$REPO_ROOT/ros2/quad_sdk_external/setup_env.sh"

WORLD="${1:-flat.sdf}"
ros2 launch quad_utils quad_gazebo.py world:="$WORLD"
