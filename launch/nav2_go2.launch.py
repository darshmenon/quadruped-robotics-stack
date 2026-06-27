"""Launch Nav2 for the Go2 Gazebo simulation.

This starts Nav2 against the existing CHAMP map/config and runs a cmd_vel to
Go2 joint-target adapter. The Gazebo world must already be running.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_launch_description():
    champ_config = get_package_share_directory("champ_config")
    nav2_bringup = get_package_share_directory("nav2_bringup")

    default_map = os.path.join(champ_config, "maps", "map.yaml")
    default_params = os.path.join(champ_config, "config", "autonomy", "navigation.yaml")

    return LaunchDescription([
        DeclareLaunchArgument("map", default_value=default_map),
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup, "launch", "bringup_launch.py")
            ),
            launch_arguments={
                "map": LaunchConfiguration("map"),
                "params_file": LaunchConfiguration("params_file"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }.items(),
        ),
    ])
