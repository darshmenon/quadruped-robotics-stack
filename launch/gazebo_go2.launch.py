"""
Launch file: Spawn Unitree Go2 in Gazebo Garden (gz-sim7) with ROS2 bridge.

Usage:
    source /opt/ros/humble/setup.bash
    ros2 launch ./launch/gazebo_go2.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


URDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'urdf', 'go2_unitree', 'urdf', 'go2.urdf'
)


def _load_robot_description():
    with open(URDF_PATH, 'r', encoding='utf-8') as urdf_file:
        return urdf_file.read()


def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    robot_description = _load_robot_description()
    use_rviz = LaunchConfiguration('rviz')
    spawn_z = LaunchConfiguration('spawn_z')

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz alongside Gazebo.',
    )
    spawn_z_arg = DeclareLaunchArgument(
        'spawn_z',
        default_value='0.5',
        description='Initial Z height when spawning the Go2 model.',
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }]
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_go2',
        output='screen',
        arguments=[
            '-name', 'go2',
            '-topic', 'robot_description',
            '-z', spawn_z,
        ]
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_ros2_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/world/empty/model/go2/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        ],
        remappings=[
            ('/world/empty/model/go2/joint_state', 'joint_states'),
        ]
    )

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        rviz_arg,
        spawn_z_arg,
        gazebo,
        robot_state_publisher,
        spawn_robot,
        bridge,
        rviz2,
    ])
