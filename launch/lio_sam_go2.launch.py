"""LIO-SAM on the Go2 sim cloud + IMU (experimental sidecar; offline replay is the reference).

LIO-SAM needs a Velodyne-style cloud (ring + per-point time), which the sim's /points lacks, so
scripts/lidar_ring_time_adapter.py (n_ring 16, vfov_deg 10, stamp_shift_s -0.1: the sim stamps a sweep at its end)
writes /points_lio. LIO-SAM's own TF goes to /lio_tf (its /tf is remapped) so the robot's TF tree is untouched; the
estimate is /lio_sam/mapping/odometry (frame lio_odom). Needs `source ~/lio_ws/install/setup.bash`.
Config: config/lio_sam_go2.yaml (rosnav session draft; IMU noise values are guesses, the sim IMU has no noise model).
"""
import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

REPO = Path(__file__).resolve().parents[1]


def generate_launch_description():
    cfg = os.environ.get("LIO_SAM_CONFIG") or str(REPO / "config" / "lio_sam_go2.yaml")   # LIO_SAM_CONFIG: try a variant
    sim = {"use_sim_time": True}
    remap = [("/tf", "/lio_tf"), ("/tf_static", "/lio_tf_static")]
    nodes = [
        ExecuteProcess(cmd=["python3", str(REPO / "scripts" / "lidar_ring_time_adapter.py"), "--ros-args",
                            "-p", "use_sim_time:=true", "-p", "in_topic:=/points", "-p", "out_topic:=/points_lio",
                            "-p", "n_ring:=16", "-p", "vfov_deg:=10.0", "-p", "period:=0.1", "-p", "stamp_shift_s:=-0.1"],
                       output="screen"),
        # lidarFrame is "base" (the cloud header); baselinkFrame lio_base is a separate frame at the same pose, so LIO-SAM
        # never publishes a TF with an empty frame_id. The real lidar mount is carried by extrinsicTrans in the yaml.
        Node(package="tf2_ros", executable="static_transform_publisher", name="lio_base_to_base",
             arguments=["0", "0", "0", "0", "0", "0", "lio_base", "base"], remappings=remap, parameters=[sim]),
    ]
    for exe in ("lio_sam_imuPreintegration", "lio_sam_imageProjection", "lio_sam_featureExtraction", "lio_sam_mapOptimization"):
        nodes.append(Node(package="lio_sam", executable=exe, name=exe, output="screen", parameters=[cfg, sim], remappings=remap))
    return LaunchDescription(nodes)
