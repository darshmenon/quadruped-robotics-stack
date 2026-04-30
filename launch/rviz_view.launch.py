"""
Launch file: Visualize quadruped robot in RViz2.

Usage:
    ros2 launch launch/rviz_view.launch.py robot:=go2
    ros2 launch launch/rviz_view.launch.py robot:=go1
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


URDF_MAP = {
    'go2': os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'urdf', 'go2_unitree', 'urdf', 'go2.urdf'
    ),
}


def _launch_setup(context, *args, **kwargs):
    robot = LaunchConfiguration('robot').perform(context)
    urdf_path = URDF_MAP.get(robot)

    if urdf_path is None or not os.path.exists(urdf_path):
        raise RuntimeError(
            f"No URDF found for robot '{robot}'. "
            f"Available: {list(URDF_MAP.keys())}"
        )

    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
    )

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
    )

    return [robot_state_publisher, joint_state_publisher_gui, rviz2]


def generate_launch_description():
    robot_arg = DeclareLaunchArgument(
        'robot',
        default_value='go2',
        description='Robot model to visualize. Supported: go2',
    )

    return LaunchDescription([
        robot_arg,
        OpaqueFunction(function=_launch_setup),
    ])
