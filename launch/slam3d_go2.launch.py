"""3D LiDAR SLAM (RTAB-Map) + optional frontier exploration for the Go2.

Two selectable locomotion backends (locomotion:=champ|nmpc), both wearing
the same 16-channel gpu_lidar and feeding the same RTAB-Map ICP SLAM setup
-- a real 3D map (and a 2D occupancy grid projected from it) instead of
slam_toolbox's single-plane /scan mapping (see launch/slam_go2.launch.py
for that 2D path):

- champ (default): launch/champ_go2_gazebo.launch.py (native gz-sim +
  CHAMP gait engine driving Go2 from /cmd_vel). Lidar sensor lives on
  urdf/go2_unitree/urdf/go2_gz.urdf.xacro ("lidar3d"). The frontier
  explorer drives it directly with Twist commands.
- nmpc: Quad-SDK's quad_gazebo.py + quad_plan.py (global/local planner +
  NMPC controller, the backend verified end-to-end in the README's
  "Quad-SDK (NMPC locomotion)" section). Lidar sensor lives on
  ros2/quad_sdk/quad_simulator/go2_description/models/go2/go2.sdf.xacro
  (also "lidar3d"). Quad-SDK has no odom publisher at all, so
  scripts/quadsdk_ground_truth_to_odom.py republishes its ground-truth
  robot state as odom + TF. The frontier explorer doesn't drive this one
  itself -- it just publishes goals to Quad-SDK's live goal_state topic
  and lets global_body_planner/nmpc_controller do the walking.

Usage:
    source /opt/ros/humble/setup.bash
    source ros2/install/setup.bash
    ros2 launch launch/slam3d_go2.launch.py
    ros2 launch launch/slam3d_go2.launch.py headless:=true explore:=true
    ros2 launch launch/slam3d_go2.launch.py locomotion:=nmpc headless:=true explore:=true
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


REPO = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = REPO / "training" / "envs" / "go2_gz_world_outdoor.sdf"
FRONTIER_EXPLORER = REPO / "scripts" / "frontier_explorer_go2.py"
OBSTACLE_TRACKER = REPO / "scripts" / "obstacle_tracker_go2.py"
GROUND_TRUTH_TO_ODOM = REPO / "scripts" / "quadsdk_ground_truth_to_odom.py"
QUAD_SDK_INSTALL = REPO / "ros2" / "install"
QUAD_SDK_LOCAL_DEPS = REPO / "ros2" / "quad_sdk_external" / "local"

RTABMAP_COMMON_PARAMS = {
    "use_sim_time": True,
    "subscribe_depth": False,
    "subscribe_rgb": False,
    "subscribe_scan_cloud": True,
    "approx_sync": True,
    "wait_for_transform": 0.3,
    # RTAB-Map params are strings.
    "Reg/Strategy": "1",           # ICP -- no camera for Vis registration
    "Icp/PointToPlane": "true",
    "Grid/Sensor": "0",            # occupancy grid from the lidar cloud
    "Grid/3D": "false",            # projected 2D grid for the frontier explorer
    "Grid/CellSize": "0.05",
    "Grid/RangeMax": "20.0",
    # lidar3d sits at body height looking mostly outward, so very few beams
    # hit the ground near the robot -- without ray tracing, RTAB-Map only
    # marks a cell free when a ground point lands in it, which leaves most
    # driven-over cells occupied/unknown (Magi's magi_slam hit the same
    # thing: 51%->100% of driven cells correctly freed once enabled).
    "Grid/RayTracing": "true",
    # 16-channel vertical resolution is sparse enough that normal-based
    # ground segmentation sprinkles spurious "obstacle" cells on flat
    # ground -- filter isolated points before classification (same fix
    # rosnav's slam_nav.launch.py lidar_type:=3d path uses).
    "Grid/NoiseFilteringRadius": "0.1",
    "Grid/NoiseFilteringMinNeighbors": "5",
    "Mem/IncrementalMemory": "true",
}


def _champ_actions(headless, world, explore, track_obstacles):
    champ_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(REPO / "launch" / "champ_go2_gazebo.launch.py")),
        # Do NOT pass rviz:=false here. This GroupAction is scoped=False (see
        # generate_launch_description) so an included launch_argument would
        # clobber the parent LaunchConfiguration("rviz") and prevent this
        # file's own slam.rviz Node from starting. champ_go2_gazebo already
        # defaults rviz to false.
        launch_arguments={
            "headless": headless,
            "world": world,
        }.items(),
    )

    # gpu_lidar always publishes LaserScan on its own <topic> and the actual
    # PointCloud2 on a nested "<topic>/points" -- see the comment on the
    # lidar3d sensor block in go2_gz.urdf.xacro.
    points_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="points_bridge",
        output="screen",
        arguments=["/points/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"],
        remappings=[("/points/points", "/points")],
        parameters=[{"use_sim_time": True}],
    )

    # RGB camera feed for the RViz picture-in-picture inset (see
    # go2_gz.urdf.xacro's "camera" sensor). Unlike lidar3d's gpu_lidar,
    # gz-sim's camera sensor publishes Image directly on its own <topic>,
    # no nested "/points" subtopic.
    camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="camera_bridge",
        output="screen",
        arguments=["/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image"],
        parameters=[{"use_sim_time": True}],
    )

    # Started after CHAMP/the joint-trajectory adapter are up (t=13s in
    # champ_go2_gazebo.launch.py) and ground-truth odom has been publishing
    # for a few seconds (t=9s), so RTAB-Map's TF lookups don't race startup.
    # database_path under /tmp: ~/.ros/rtabmap.db is often root-owned or
    # unwritable when launches are started from mixed-privilege sessions,
    # which makes rtabmap FATAL on open and leaves map->base unpublished.
    rtabmap_slam = TimerAction(
        period=16.0,
        actions=[Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[{
                **RTABMAP_COMMON_PARAMS,
                # base_footprint (go2_gz.urdf.xacro), not "base" -- keeps the
                # map/grid z=0 origin at the ground instead of body height.
                "frame_id": "base_footprint",
                "odom_frame_id": "odom",
                "map_frame_id": "map",
                "database_path": "/tmp/go2_rtabmap/rtabmap_champ.db",
            }],
            remappings=[("odom", "/odom"), ("scan_cloud", "/points")],
            arguments=["-d"],  # fresh database each run
        )],
    )

    frontier_explorer = TimerAction(
        period=20.0,
        condition=IfCondition(explore),
        actions=[ExecuteProcess(
            cmd=["python3", str(FRONTIER_EXPLORER), "--ros-args",
                 "-p", "use_sim_time:=true", "-p", "control_mode:=cmd_vel"],
            output="screen",
        )],
    )

    # Started after rtabmap_slam (t=16s) so map->base TF is already
    # available, same reasoning as frontier_explorer's own 20s delay.
    obstacle_tracker = TimerAction(
        period=18.0,
        condition=IfCondition(track_obstacles),
        actions=[ExecuteProcess(
            cmd=["python3", str(OBSTACLE_TRACKER), "--ros-args",
                 "-p", "use_sim_time:=true"],
            output="screen",
        )],
    )

    return [champ_gazebo, points_bridge, camera_bridge, rtabmap_slam, frontier_explorer, obstacle_tracker]


def _nmpc_env_actions():
    # Mirrors ros2/quad_sdk_external/setup_env.sh, which walk_quadsdk_go2.sh
    # sources by hand before its own `ros2 launch quad_utils quad_gazebo.py`.
    # Without these, gz-sim can't resolve the go2 model's model://imu (and
    # other quad_sdk_external submodel) URIs -- UserCommands fails spawn
    # with "Unable to find uri[model://imu]" for any locomotion:=nmpc launch
    # that goes through this file instead of walk_quadsdk_go2.sh.
    resource_dirs = [
        str(QUAD_SDK_INSTALL / "quad_sim_scripts" / "share" / "quad_sim_scripts" / "models"),
        str(QUAD_SDK_INSTALL / "quad_sim_scripts" / "share" / "quad_sim_scripts" / "worlds"),
        str(QUAD_SDK_INSTALL / "go2_description" / "share" / "go2_description" / "models"),
        str(QUAD_SDK_INSTALL / "objects_description" / "share" / "objects_description" / "models"),
        str(QUAD_SDK_INSTALL / "sensor_description" / "share" / "sensor_description" / "models"),
    ]
    existing_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    resource_path = ":".join(resource_dirs + ([existing_resource_path] if existing_resource_path else []))

    plugin_dir = str(QUAD_SDK_INSTALL / "gazebo_plugins" / "lib")
    existing_plugin_path = os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")
    plugin_path = f"{existing_plugin_path}:{plugin_dir}" if existing_plugin_path else plugin_dir

    lib_dir = str(QUAD_SDK_LOCAL_DEPS / "lib")
    existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_path = f"{lib_dir}:{existing_ld_path}" if existing_ld_path else lib_dir

    return [
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", resource_path),
        SetEnvironmentVariable("GZ_SIM_SYSTEM_PLUGIN_PATH", plugin_path),
        SetEnvironmentVariable("LD_LIBRARY_PATH", ld_path),
    ]


def _nmpc_actions(nmpc_gui_flag, nmpc_world, explore, track_obstacles):
    quad_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(REPO / "ros2" / "quad_sdk" / "quad_utils" / "launch" / "quad_gazebo.py")),
        launch_arguments={
            "gui": nmpc_gui_flag,
            "rviz": "false",
            "world": nmpc_world,
        }.items(),
    )

    # Same stand sequence walk_quadsdk_go2.sh uses: a single one-shot publish
    # can be lost to a ROS2 discovery race, so hold it for a few seconds.
    stand = TimerAction(
        period=15.0,
        actions=[ExecuteProcess(
            cmd=["timeout", "5", "ros2", "topic", "pub", "/robot_1/control/mode",
                 "std_msgs/msg/UInt8", "data: 1", "--rate", "10"],
            output="screen",
        )],
    )

    # goal_state:="[1.0, 0.0]" just gets the planner moving immediately on
    # startup; the frontier explorer takes over with real goals once RTAB-Map
    # has a map (see global_body_planner.cpp's live goal_state_sub_).
    nmpc_plan = TimerAction(
        period=18.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(REPO / "ros2" / "quad_sdk" / "quad_utils" / "launch" / "quad_plan.py")),
            launch_arguments={"goal_state": "[1.0, 0.0]"}.items(),
        )],
    )

    # See the lidar3d sensor comment in go2.sdf.xacro -- world name is
    # "default" (gz-sim's own default, not a name this repo chose) and
    # "robot_1" is quad_gazebo.py's default robot_configs entry name.
    points_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="points_bridge",
        output="screen",
        arguments=[
            "/world/default/model/robot_1/link/body/sensor/lidar3d/scan/points"
            "@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ],
        remappings=[
            ("/world/default/model/robot_1/link/body/sensor/lidar3d/scan/points", "/robot_1/points"),
        ],
        parameters=[{"use_sim_time": True}],
    )

    # gz-sim auto-derives the point cloud's frame_id from the sensor's scoped
    # name (confirmed empirically: "robot_1/body/lidar3d") since plain SDF
    # <sensor> blocks have no frame-override tag (unlike the URDF path's
    # <gz_frame_id>base</gz_frame_id>, which sdformat_urdf's converter does
    # honor -- go2_gz.urdf.xacro's cloud really does come through as "base").
    # Without this, RTAB-Map can't place the cloud relative to "body" at all
    # ("TF of received scan cloud ... is not set, aborting"). Static because
    # it's the sensor's fixed mount offset, matching go2.sdf.xacro's lidar3d
    # <pose>0 0 0.16 0 0 0</pose>.
    lidar_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="lidar3d_static_tf",
        output="screen",
        arguments=["0", "0", "0.16", "0", "0", "0", "body", "robot_1/body/lidar3d"],
        parameters=[{"use_sim_time": True}],
    )

    ground_truth_to_odom = TimerAction(
        period=10.0,
        actions=[ExecuteProcess(
            cmd=["python3", str(GROUND_TRUTH_TO_ODOM), "--ros-args",
                 "-p", "use_sim_time:=true",
                 "-p", "ground_truth_topic:=/robot_1/state/ground_truth",
                 "-p", "odom_topic:=/robot_1/odom",
                 "-p", "base_frame:=body"],
            output="screen",
        )],
    )

    rtabmap_slam = TimerAction(
        period=22.0,
        actions=[Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[{
                **RTABMAP_COMMON_PARAMS,
                "frame_id": "body",
                "odom_frame_id": "odom",
                "map_frame_id": "map",
                "database_path": "/tmp/go2_rtabmap/rtabmap_nmpc.db",
            }],
            remappings=[("odom", "/robot_1/odom"), ("scan_cloud", "/robot_1/points")],
            arguments=["-d"],
        )],
    )

    frontier_explorer = TimerAction(
        period=26.0,
        condition=IfCondition(explore),
        actions=[ExecuteProcess(
            cmd=["python3", str(FRONTIER_EXPLORER), "--ros-args",
                 "-p", "use_sim_time:=true",
                 "-p", "control_mode:=nmpc_goal",
                 "-p", "base_frame:=body",
                 "-p", "goal_state_topic:=/robot_1/goal_state"],
            output="screen",
        )],
    )

    # Started after rtabmap_slam (t=22s) so map->body TF is already
    # available, same reasoning as frontier_explorer's own 26s delay.
    obstacle_tracker = TimerAction(
        period=24.0,
        condition=IfCondition(track_obstacles),
        actions=[ExecuteProcess(
            cmd=["python3", str(OBSTACLE_TRACKER), "--ros-args",
                 "-p", "use_sim_time:=true",
                 "-p", "points_topic:=/robot_1/points",
                 "-p", "base_frame:=body"],
            output="screen",
        )],
    )

    return _nmpc_env_actions() + [
        quad_gazebo, stand, nmpc_plan, points_bridge, lidar_static_tf,
        ground_truth_to_odom, rtabmap_slam, frontier_explorer, obstacle_tracker,
    ]


def generate_launch_description():
    headless = LaunchConfiguration("headless")
    rviz = LaunchConfiguration("rviz")
    world = LaunchConfiguration("world")
    nmpc_world = LaunchConfiguration("nmpc_world")
    explore = LaunchConfiguration("explore")
    track_obstacles = LaunchConfiguration("track_obstacles")

    # quad_gazebo.py's `gui` arg is the inverse of this launch's `headless`
    # (same true/false semantics as champ_go2_gazebo.launch.py, just spelled
    # the opposite way upstream).
    nmpc_gui_flag = PythonExpression(["'false' if '", headless, "' == 'true' else 'true'"])

    champ_branch = GroupAction(
        condition=LaunchConfigurationEquals("locomotion", "champ"),
        # Not scoped: champ_go2_gazebo.launch.py declares "state_estimation" and
        # reads it back inside a TimerAction (fires 13s later, after this
        # group's synchronous action list -- including PopLaunchConfigurations --
        # has already been visited). A scoped group pops that configuration out
        # of context before the timer fires, so the deferred lookup throws
        # "launch configuration 'state_estimation' does not exist".
        scoped=False,
        actions=_champ_actions(headless, world, explore, track_obstacles),
    )

    nmpc_branch = GroupAction(
        condition=LaunchConfigurationEquals("locomotion", "nmpc"),
        actions=_nmpc_actions(nmpc_gui_flag, nmpc_world, explore, track_obstacles),
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", str(REPO / "ros2" / "champ_navigation" / "rviz" / "slam.rviz")],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument("headless", default_value="false",
                               description="Skip Gazebo/RViz GUIs (server + mapping only)"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("locomotion", default_value="champ",
                               description="Locomotion backend: 'champ' (default, CHAMP gait "
                                           "engine via /cmd_vel) or 'nmpc' (Quad-SDK global/local "
                                           "planner + NMPC controller, goal-driven)."),
        DeclareLaunchArgument("world", default_value=str(DEFAULT_WORLD),
                               description="SDF world path, locomotion:=champ only -- defaults "
                                           "to the outdoor demo world (trees/rocks on open ground)."),
        DeclareLaunchArgument("nmpc_world", default_value="big_flat.sdf",
                               description="World filename from quad_sim_scripts/worlds/, "
                                           "locomotion:=nmpc only (flat.sdf's mesh only spans "
                                           "~5m -- too small to explore)."),
        DeclareLaunchArgument("explore", default_value="false",
                               description="Auto-start scripts/frontier_explorer_go2.py"),
        DeclareLaunchArgument("track_obstacles", default_value="false",
                               description="Auto-start scripts/obstacle_tracker_go2.py"),
        # rtabmap's database_path parents below must exist before the node
        # opens them -- ~/.ros is often root-owned after mixed-privilege
        # sessions, so we keep the DBs under /tmp/go2_rtabmap instead.
        ExecuteProcess(cmd=["mkdir", "-p", "/tmp/go2_rtabmap"], output="screen"),
        champ_branch,
        nmpc_branch,
        rviz2,
    ])
