"""
Launch Go2 in Gazebo Harmonic (gz sim 8) headlessly for RL training.

Uses native Gazebo joint control (no ros2_control) + ros_gz_bridge.

Usage:
    source /opt/ros/humble/setup.bash
    ros2 launch training/launch/gazebo_rl.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

URDF = "/home/asimov/quadruped-dog-rl/urdf/go2_unitree/urdf/go2_gz.urdf"
WORLD = "/home/asimov/quadruped-dog-rl/training/envs/go2_gz_world.sdf"


def generate_launch_description():
    headless_arg = DeclareLaunchArgument("headless", default_value="true")

    # Gazebo Harmonic — server-only headless (-s), run immediately (-r)
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
        ]),
        launch_arguments={
            "gz_args": f"-s -r {WORLD}",
            "on_exit_shutdown": "true",
        }.items(),
    )

    # Robot state publisher (TF from URDF)
    with open(URDF, "r") as f:
        robot_description = f.read()

    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
        output="screen",
    )

    # Spawn go2 into running Gazebo world
    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "go2",
            "-string", robot_description,
            "-x", "0", "-y", "0", "-z", "0.42",
        ],
        output="screen",
    )

    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]

    bridge_args = [
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        "/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU",
        "/world/go2_rl/model/go2/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model",
    ]
    # per-joint position command bridges: ROS2 Float64 → Gazebo Double
    for jname in joint_names:
        bridge_args.append(f"/go2/cmd/{jname}@std_msgs/msg/Float64]gz.msgs.Double")

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=bridge_args,
        remappings=[("/world/go2_rl/model/go2/joint_state", "/joint_states")],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    return LaunchDescription([
        headless_arg,
        gz_sim,
        robot_state_pub,
        TimerAction(period=2.0, actions=[spawn]),
        bridge,
    ])
