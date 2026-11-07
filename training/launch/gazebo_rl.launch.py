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
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable, TimerAction
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
ARM_REACH_NODE = REPO / "intelligence" / "manipulation" / "arm_reach_node.py"


def generate_launch_description():
    headless_arg = DeclareLaunchArgument("headless", default_value="false")
    urdf_arg = DeclareLaunchArgument(
        "urdf", default_value=str(URDF),
        description="Robot URDF; pass the output of scripts/make_go2_noarm_urdf.py for an arm-free Go2.")
    world_arg = DeclareLaunchArgument("world", default_value=str(DEFAULT_WORLD))
    stand_duration_arg = DeclareLaunchArgument("stand_duration", default_value="-1.0")
    p_gain_arg = DeclareLaunchArgument("p_gain", default_value="300", description="Leg JointPositionController p_gain")
    d_gain_arg = DeclareLaunchArgument("d_gain", default_value="30", description="Leg JointPositionController d_gain")
    mu_arg = DeclareLaunchArgument("mu", default_value="2.5", description="Foot/ground friction coefficient")
    enable_arm_reach_arg = DeclareLaunchArgument("enable_arm_reach", default_value="false")
    enable_lidar3d_arg = DeclareLaunchArgument(
        "enable_lidar3d", default_value="true",
        description="Bridge the lidar3d gpu_lidar's PointCloud2 to /points, for Nav2's 3D "
                    "obstacle costmap source (see config/go2_navigation.yaml) and rtabmap_slam. "
                    "Off by default only makes sense if something else is already bridging "
                    "/points (e.g. launch/slam3d_go2.launch.py does its own).")
    enable_octomap_arg = DeclareLaunchArgument(
        "enable_octomap", default_value="true",
        description="Run octomap_server on /points, building a persistent 3D occupancy map "
                    "and publishing /projected_map for Nav2's octomap_layer (StaticLayer) in "
                    "config/go2_navigation.yaml. Requires enable_lidar3d and a 'map' TF (Nav2's "
                    "AMCL) to actually integrate scans -- harmless no-op without either.")
    # go2_gz.urdf's manipulator arm links (see
    # ros2/champ_description/urdf/arm.urdf.xacro) use package://champ_description
    # mesh URIs. Gazebo resolves those against GZ_SIM_RESOURCE_PATH, not ROS's
    # package:// mechanism (that's only wired up for robot_state_publisher/RViz),
    # so without this the arm mesh visuals silently fail to load while the rest
    # of the body (which uses absolute file:// mesh paths) renders fine.
    champ_description_share = get_package_share_directory("champ_description")
    gz_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.path.dirname(champ_description_share)
        + (os.pathsep + os.environ["GZ_SIM_RESOURCE_PATH"]
           if os.environ.get("GZ_SIM_RESOURCE_PATH") else ""),
    )

    return LaunchDescription([
        headless_arg,
        world_arg,
        stand_duration_arg,
        urdf_arg,
        p_gain_arg,
        mu_arg,
        d_gain_arg,
        enable_arm_reach_arg,
        enable_lidar3d_arg,
        enable_octomap_arg,
        gz_resource_path,
        OpaqueFunction(function=_launch_setup),
    ])


def _launch_setup(context, *args, **kwargs):
    headless = LaunchConfiguration("headless").perform(context).lower() == "true"
    world = LaunchConfiguration("world").perform(context)
    urdf = LaunchConfiguration("urdf").perform(context)
    stand_duration = LaunchConfiguration("stand_duration").perform(context)
    enable_arm_reach = LaunchConfiguration("enable_arm_reach").perform(context).lower() == "true"
    enable_lidar3d = LaunchConfiguration("enable_lidar3d").perform(context).lower() == "true"
    enable_octomap = LaunchConfiguration("enable_octomap").perform(context).lower() == "true"
    gz_args = f"{'-s ' if headless else ''}{world}"
    subprocess.run([
        "python3",
        str(STAND_SDF_SCRIPT),
        "--urdf",
        urdf,
        "--mu",
        LaunchConfiguration("mu").perform(context),
        "--p-gain",
        LaunchConfiguration("p_gain").perform(context),
        "--d-gain",
        LaunchConfiguration("d_gain").perform(context),
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
    with open(urdf, "r") as f:
        robot_description = f.read()
    # go2_gz.urdf bakes in a __REPO_ROOT__ placeholder for the ros2_control
    # params file path (see scripts/make_go2_stand.py), resolved here.
    robot_description = robot_description.replace("__REPO_ROOT__", str(REPO))

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
    # make_go2_stand.py adds a native JointPositionController (topic
    # /go2/cmd/<joint>) for every joint in STANDING_POSE, arm included -- but
    # until now only the leg joints were bridged to ROS2, so the arm had no
    # way to be commanded at runtime.
    arm_joint_names = [
        "base_joint", "lower_arm_joint", "upper_arm_joint",
        "wrist1_joint", "wrist2_joint",
        # Both bridged independently -- the URDF's right_finger_joint <mimic>
        # tag isn't enforced by gz sdf -p's URDF->SDF conversion, see
        # ros2/champ_description/urdf/arm.urdf.xacro.
        "left_finger_joint", "right_finger_joint",
    ]

    bridge_args = [
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        "/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU",
        "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        "/world/go2_rl/model/go2/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model",
    ]
    # per-joint position command bridges: ROS2 Float64 → Gazebo Double
    for jname in joint_names + arm_joint_names:
        bridge_args.append(f"/go2/cmd/{jname}@std_msgs/msg/Float64]gz.msgs.Double")

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=bridge_args,
        remappings=[
            ("/world/go2_rl/model/go2/joint_state", "/joint_states"),
            ("/scan", "/scan_raw"),
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    # Filters raw lidar noise (near-range/shadow points) before AMCL, SLAM
    # Toolbox, and the Nav2 costmaps see it. See config/laser_filters.yaml.
    laser_filter = Node(
        package="laser_filters",
        executable="scan_to_scan_filter_chain",
        name="scan_to_scan_filter_chain",
        parameters=[str(REPO / "config" / "laser_filters.yaml"), {"use_sim_time": True}],
        remappings=[("scan", "/scan_raw"), ("scan_filtered", "/scan")],
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

    actions = [
        gz_sim,
        robot_state_pub,
        # Delay spawn past world load: the multi-terrain world has ~65 extra
        # static models, which takes longer to settle than the flat world
        # under software rendering. Spawning too early drops the Go2 onto a
        # not-yet-settled world and it faceplants immediately.
        TimerAction(period=5.0, actions=[spawn]),
        bridge,
        laser_filter,
        # Extra buffer past spawn before touching the entity: the heavier
        # multi-terrain world can still be inserting the ~65 static terrain
        # models into the ECS a couple seconds after "OK creation of entity"
        # comes back, so an early set_pose call (from --reset-upright) can
        # silently target a not-yet-registered entity (id:0).
        TimerAction(period=9.0, actions=[odom]),
        TimerAction(period=9.0, actions=[stand]),
    ]

    if enable_arm_reach:
        arm_reach = ExecuteProcess(cmd=["python3", str(ARM_REACH_NODE)], output="screen")
        # Start after `stand` (period=9.0) has issued its own arm command
        # (STOW_POSE) so arm_reach_node's first /arm/target message wins.
        actions.append(TimerAction(period=11.0, actions=[arm_reach]))

    if enable_lidar3d:
        # gpu_lidar always publishes LaserScan on its own <topic> and the
        # actual PointCloud2 on a nested "<topic>/points" -- see the lidar3d
        # sensor comment in go2_gz.urdf. Same bridge launch/slam3d_go2.launch.py
        # uses for its own (quad_sdk-path) points_bridge.
        points_bridge = Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="points_bridge",
            arguments=["/points/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"],
            remappings=[("/points/points", "/points")],
            parameters=[{"use_sim_time": True}],
            output="screen",
        )
        actions.append(TimerAction(period=5.0, actions=[points_bridge]))

        if enable_octomap:
            # Builds a persistent 3D occupancy map from /points and publishes
            # /projected_map (2D OccupancyGrid) for Nav2's octomap_layer
            # (StaticLayer, see config/go2_navigation.yaml). frame_id is
            # "map" per octomap_server's own guidance ("set to 'map' if SLAM
            # or localization running") -- needs AMCL's map->odom TF, so this
            # is inert (no error, just no integration) until nav2_go2.launch.py
            # is also up.
            octomap_server = Node(
                package="octomap_server",
                executable="octomap_server_node",
                name="octomap_server",
                parameters=[{
                    "use_sim_time": True,
                    "resolution": 0.05,
                    "frame_id": "map",
                    "base_frame_id": "base",
                    "sensor_model.max_range": 8.0,
                }],
                remappings=[("cloud_in", "/points")],
                output="screen",
            )
            # A few seconds after points_bridge (5.0) so /points is already
            # flowing before octomap_server's first cloud callback.
            actions.append(TimerAction(period=8.0, actions=[octomap_server]))

    return actions
