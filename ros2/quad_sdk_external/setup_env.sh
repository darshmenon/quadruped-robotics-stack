#!/usr/bin/env bash
# Source this AFTER `source ros2/install/setup.bash`, e.g.:
#   source ros2/install/setup.bash
#   source ros2/quad_sdk_external/setup_env.sh
#
# Points Gazebo at the Quad-SDK Go2 model/world assets and the local
# (non-sudo) RBDL/IPOPT install prefix used by ros2/quad_sdk_external/local.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALL_DIR="$REPO_ROOT/ros2/install"
LOCAL_DEPS="$REPO_ROOT/ros2/quad_sdk_external/local"

export GZ_SIM_RESOURCE_PATH="$GZ_SIM_RESOURCE_PATH:\
$INSTALL_DIR/quad_sim_scripts/share/quad_sim_scripts/models:\
$INSTALL_DIR/quad_sim_scripts/share/quad_sim_scripts/worlds:\
$INSTALL_DIR/go2_description/share/go2_description/models:\
$INSTALL_DIR/objects_description/share/objects_description/models:\
$INSTALL_DIR/sensor_description/share/sensor_description/models"

export GZ_SIM_SYSTEM_PLUGIN_PATH="$GZ_SIM_SYSTEM_PLUGIN_PATH:$INSTALL_DIR/gazebo_plugins/lib"
export LD_LIBRARY_PATH="$LOCAL_DEPS/lib:$LD_LIBRARY_PATH"
