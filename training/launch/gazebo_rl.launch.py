"""
Launch Go2 in Gazebo Harmonic (gz sim 8) headlessly for RL training.

Uses native Gazebo joint control (no ros2_control) + ros_gz_bridge.

Usage:
    source /opt/ros/humble/setup.bash
    ros2 launch training/launch/gazebo_rl.launch.py
"""

import os
import subprocess
import tempfile
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

REPO = Path(__file__).resolve().parents[2]
URDF = REPO / "urdf" / "go2_unitree" / "urdf" / "go2_gz.urdf"
DEFAULT_WORLD = REPO / "training" / "envs" / "go2_gz_world.sdf"
STAND_SDF = Path(tempfile.gettempdir()) / "go2_stand.sdf"
STAND_SDF_SCRIPT = REPO / "scripts" / "make_go2_stand.py"
STAND_NODE = REPO / "scripts" / "stand_go2_gz.py"
ODOM_NODE = REPO / "scripts" / "gz_pose_to_odom.py"


def generate_launch_description():
    headless_arg = DeclareLaunchArgument("headless", default_value="false")
    world_arg = DeclareLaunchArgument("world", default_value=str(DEFAULT_WORLD))
    stand_duration_arg = DeclareLaunchArgument("stand_duration", default_value="-1.0")

    return LaunchDescription([
        headless_arg,
        world_arg,
        stand_duration_arg,
        OpaqueFunction(function=_launch_setup),
    ])


def _launch_setup(context, *args, **kwargs):
    headless = LaunchConfiguration("headless").perform(context).lower() == "true"
    world = LaunchConfiguration("world").perform(context)
    stand_duration = LaunchConfiguration("stand_duration").perform(context)
    gz_args = f"{'-s ' if headless else ''}{world}"
    subprocess.run([
        "python3",
        str(STAND_SDF_SCRIPT),
        "--urdf",
        str(URDF),
        "--out",
        str(STAND_SDF),
    ], check=True)

    # Gazebo Harmonic. Use server-only mode when headless, otherwise open GUI.
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
        ]),
        launch_arguments={
            "gz_args": gz_args,
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
            "-file", str(STAND_SDF),
            "-x", "0", "-y", "0", "-z", "0.32",
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
        "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
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

    stand = ExecuteProcess(
        cmd=[
            "python3",
            str(STAND_NODE),
            "--reset-upright",
            "--unpause",
            "--duration-seconds",
            stand_duration,
        ],
        output="screen",
    )

    odom = ExecuteProcess(
        cmd=[
            "python3",
            str(ODOM_NODE),
            "--world", "go2_rl",
            "--model", "go2",
            "--odom-frame", "odom",
            "--base-frame", "base",
            "--ros-args", "-p", "use_sim_time:=true",
        ],
        output="screen",
    )

    return [
        gz_sim,
        robot_state_pub,
        # Delay spawn past world load: the multi-terrain world has ~65 extra
        # static models, which takes longer to settle than the flat world
        # under software rendering. Spawning too early drops the Go2 onto a
        # not-yet-settled world and it faceplants immediately.
        TimerAction(period=5.0, actions=[spawn]),
        bridge,
        # Extra buffer past spawn before touching the entity: the heavier
        # multi-terrain world can still be inserting the ~65 static terrain
        # models into the ECS a couple seconds after "OK creation of entity"
        # comes back, so an early set_pose call (from --reset-upright) can
        # silently target a not-yet-registered entity (id:0).
        TimerAction(period=9.0, actions=[odom]),
        TimerAction(period=9.0, actions=[stand]),
    ]
