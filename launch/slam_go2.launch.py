"""Launch SLAM Toolbox for the Go2 Gazebo simulation."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


REPO = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = REPO / "config" / "go2_slam.yaml"


def generate_launch_description():
    slam_toolbox = get_package_share_directory("slam_toolbox")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("slam_params_file", default_value=str(DEFAULT_PARAMS)),
        DeclareLaunchArgument("rviz", default_value="false"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(Path(slam_toolbox) / "launch" / "online_async_launch.py")
            ),
            launch_arguments={
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "slam_params_file": LaunchConfiguration("slam_params_file"),
            }.items(),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", str(REPO / "ros2" / "champ_navigation" / "rviz" / "slam.rviz")],
            parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
