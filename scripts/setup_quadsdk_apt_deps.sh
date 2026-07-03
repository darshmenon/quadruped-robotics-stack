#!/usr/bin/env bash
# One-time, privileged setup for the Quad-SDK locomotion backend (ros2/quad_sdk/).
# Run this yourself in a real terminal -- it needs your sudo password, which
# an agent session cannot supply. Everything here is a plain `apt install`;
# nothing is added to bashrc and no third-party apt repos are added (this
# environment already has Gazebo Harmonic / gz-sim8 installed via
# ros-humble-ros-gzharmonic-*, so we reuse that instead of the upstream
# Quad-SDK script's OSRF gazebo-stable repo).
set -euo pipefail

sudo apt-get update

sudo apt-get install -y \
  ros-humble-grid-map-core \
  ros-humble-grid-map-ros \
  ros-humble-grid-map-msgs \
  ros-humble-grid-map-pcl \
  ros-humble-plotjuggler-ros \
  ros-humble-cv-bridge \
  ros-humble-vision-opencv \
  ros-humble-ros-gz \
  ros-humble-gz-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-topic-tools \
  ros-humble-rmw-cyclonedds-cpp \
  ros-humble-pinocchio \
  ros-humble-teleop-twist-joy \
  ros-humble-rosbag2 \
  ros-humble-rosbag2-storage \
  ros-humble-tf2 \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs \
  tmux cpplint chrony iw libmetis-dev libglfw3-dev

# ros-humble-mujoco-vendor is intentionally NOT installed here: it's not
# available for Humble (introduced on newer distros). quad_utils's CMakeLists
# already treats it as optional (find_package(... QUIET)) -- without it, the
# mujoco_recorder_node target is skipped and everything else builds fine,
# since this repo already has its own separate MuJoCo RL pipeline.

echo "Done. Next: ./scripts/build_quadsdk_local_libs.sh (no sudo needed) to build RBDL + IPOPT."
