#!/usr/bin/env bash
# One-command ROS2 (Humble) Quad-SDK Go2 walking demo: launches Gazebo
# Harmonic (optionally on a terrain world, not just flat ground) via
# `ros2 launch`, spawns Go2, stands it up, then starts the NMPC planning
# stack toward a goal.
#
# Prereqs (one-time, see README "Quad-SDK (NMPC locomotion)" section):
#   ./scripts/setup_quadsdk_apt_deps.sh   (needs your sudo password)
#   ./scripts/build_quadsdk_local_libs.sh (no sudo)
#   cd ros2 && colcon build --symlink-install && cd ..
#
# Usage:
#   ./scripts/walk_quadsdk_go2.sh                          # flat ground, goal (5, 0)
#   ./scripts/walk_quadsdk_go2.sh 8.0 0.0                   # flat ground, custom goal
#   ./scripts/walk_quadsdk_go2.sh 8.0 0.0 gui                # also show the Gazebo GUI
#   ./scripts/walk_quadsdk_go2.sh 8.0 0.0 gui step_20cm.sdf  # walk over a terrain world
#
# Terrain worlds available under
# ros2/quad_sdk/quad_simulator/quad_sim_scripts/worlds/ (verified: the sim
# loads each mesh correctly, robot stands, and NMPC solves with zero
# failures given a *reachable* goal):
#   flat.sdf (small patch, ~5m radius -- goals past that report "Invalid goal
#   state" since they're outside the terrain mesh), big_flat.sdf (use this
#   for goals farther than ~5m on flat ground),
#   step_10cm.sdf / step_15cm.sdf / step_20cm.sdf / step_25cm.sdf / step_30cm.sdf,
#   gap_20cm.sdf / gap_40cm.sdf / gap_80cm.sdf,
#   slope_20.sdf, slope_20_hole.sdf,
#   rough_25cm.sdf, rough_40cm_huge.sdf,
#   parkour_local_min.sdf, and *_local_min.sdf variants (recovery-from-stuck tests).
# NOTE: a goal placed ON TOP of an obstacle (e.g. [3.0, 0.0] on step_20cm.sdf,
# which happens to coincide with the step itself) correctly reports "Invalid
# goal state" -- that's the planner's collision/clearance check working as
# intended, not a bug. Pick a goal before/around the feature, not on it.
set -eo pipefail
# Not `set -u`: ROS2's own setup.bash scripts reference unset variables
# internally (e.g. AMENT_TRACE_SETUP_FILES) and abort under nounset.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GOAL_X="${1:-5.0}"
GOAL_Y="${2:-0.0}"
SHOW_GUI="${3:-headless}"
WORLD="${4:-flat.sdf}"
GUI_FLAG="false"
if [ "$SHOW_GUI" = "gui" ]; then
  GUI_FLAG="true"
fi

source /opt/ros/humble/setup.bash
source "$REPO_ROOT/ros2/install/setup.bash"
source "$REPO_ROOT/ros2/quad_sdk_external/setup_env.sh"

cleanup() {
  echo "Shutting down..."
  kill "${GAZEBO_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Launching Gazebo + Go2 on world: $WORLD ..."
ros2 launch quad_utils quad_gazebo.py gui:="$GUI_FLAG" rviz:=false world:="$WORLD" &
GAZEBO_PID=$!

echo "==> Waiting for the sim to come up..."
sleep 15

echo "==> Standing Go2 up (holding the stand command for 2s -- a single"
echo "    one-shot publish can be lost to a ROS2 discovery race)..."
timeout 5 ros2 topic pub /robot_1/control/mode std_msgs/msg/UInt8 "data: 1" --rate 10 >/dev/null 2>&1 || true
sleep 2

echo "==> Standing height check:"
timeout 3 ros2 topic echo /robot_1/state/ground_truth --once 2>/dev/null | grep -m1 "z:" || true

echo "==> Launching the NMPC planning stack toward goal ($GOAL_X, $GOAL_Y)..."
ros2 launch quad_utils quad_plan.py goal_state:="[$GOAL_X, $GOAL_Y]"
