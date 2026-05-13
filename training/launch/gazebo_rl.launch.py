"""
Launch Gazebo headlessly for RL training.

Usage:
    ros2 launch training/launch/gazebo_rl.launch.py headless:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    headless = DeclareLaunchArgument("headless", default_value="true")
    gui = DeclareLaunchArgument("gui", default_value="false")
    robot = DeclareLaunchArgument("robot", default_value="go1")

    champ_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("champ_gazebo"), "launch", "gazebo.launch.py"
            ])
        ]),
        launch_arguments={
            "headless": LaunchConfiguration("headless"),
            "gui": LaunchConfiguration("gui"),
            "robot_name": LaunchConfiguration("robot"),
            "paused": "false",
        }.items(),
    )

    return LaunchDescription([headless, gui, robot, champ_launch])
