"""Launch Go2 Gazebo with CHAMP producing Go2 joint targets.

Usage:
    source /opt/ros/humble/setup.bash
    source ros2/install/setup.bash
    ros2 launch launch/champ_go2_gazebo.launch.py
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


REPO = Path(__file__).resolve().parents[1]
GO2_URDF = REPO / "urdf" / "go2_unitree" / "urdf" / "go2_gz.urdf"
GO2_CHAMP_CONFIG = REPO / "config" / "go2_champ"
ADAPTER = REPO / "scripts" / "champ_joint_trajectory_to_go2_gz.py"


def generate_launch_description():
    headless = LaunchConfiguration("headless")
    rviz = LaunchConfiguration("rviz")
    world = LaunchConfiguration("world")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(REPO / "training" / "launch" / "gazebo_rl.launch.py")
        ),
        launch_arguments={
            "headless": headless,
            "world": world,
            "stand_duration": "2.0",
        }.items(),
    )

    champ = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("champ_bringup"),
                "launch",
                "bringup.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": "true",
            "description_path": str(GO2_URDF),
            "joints_map_path": str(GO2_CHAMP_CONFIG / "joints.yaml"),
            "links_map_path": str(GO2_CHAMP_CONFIG / "links.yaml"),
            "gait_config_path": str(GO2_CHAMP_CONFIG / "gait.yaml"),
            "robot_name": "go2",
            "base_link_frame": "base",
            "gazebo": "true",
            "rviz": rviz,
            "joint_controller_topic": "/champ/joint_trajectory",
            "publish_joint_states": "false",
            "publish_foot_contacts": "false",
            "publish_odom_tf": "false",
            "close_loop_odom": "true",
            "state_estimation": "false",
        }.items(),
    )

    adapter = ExecuteProcess(
        cmd=[
            "python3",
            str(ADAPTER),
            "--ros-args",
            "-p",
            "use_sim_time:=true",
            "-p",
            "input_topic:=/champ/joint_trajectory",
            "-p",
            "output_prefix:=/go2/cmd",
        ],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument(
            "world",
            default_value=str(REPO / "training" / "envs" / "go2_gz_world.sdf"),
        ),
        gazebo,
        TimerAction(period=10.0, actions=[champ, adapter]),
    ])
