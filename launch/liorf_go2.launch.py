"""LIORF sidecar for the Go2 sim cloud (item 14: only after KISS-ICP loses).

Same adapter and frame split as launch/lio_sam_go2.launch.py. No feature-extraction
node: that is the LIORF change. Estimate: /liorf/mapping/odometry.
Needs `source ~/ws_liorf/install_clean/setup.bash`.
"""
import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

REPO = Path(__file__).resolve().parents[1]


def generate_launch_description():
    cfg = os.environ.get("LIORF_CONFIG") or str(REPO / "config" / "liorf_go2.yaml")
    sim = {"use_sim_time": True}
    remap = [("/tf", "/lio_tf"), ("/tf_static", "/lio_tf_static")]
    nodes = [
        ExecuteProcess(cmd=["python3", str(REPO / "scripts" / "lidar_ring_time_adapter.py"), "--ros-args",
                            "-p", "use_sim_time:=true", "-p", "in_topic:=/points", "-p", "out_topic:=/points_lio",
                            "-p", "n_ring:=16", "-p", "vfov_deg:=10.0", "-p", "period:=0.1", "-p", "stamp_shift_s:=-0.1"],
                       output="screen"),
        Node(package="tf2_ros", executable="static_transform_publisher", name="lio_base_to_base",
             arguments=["0", "0", "0", "0", "0", "0", "lio_base", "base"], remappings=remap, parameters=[sim]),
    ]
    for exe in ("liorf_imuPreintegration", "liorf_imageProjection", "liorf_mapOptmization"):
        nodes.append(Node(package="liorf", executable=exe, name=exe, output="screen",
                          parameters=[cfg, sim], remappings=remap))
    return LaunchDescription(nodes)
