"""RKO-LIO (LiDAR-inertial odometry) sidecar for the Go2 Gazebo sim.

Runs rko_lio on /points + /imu/data and publishes /rko_lio/odom in its own odom frame (rko_odom).
It never owns odom->base TF, so it is safe next to SLAM/Nav2; scripts/lio_drift_log.py scores it against
the simulator ground-truth /odom.

rko_lio's launch file only reads parameters given on the command line (context.argv), so including it from
another launch file silently drops them; it is therefore started as a subprocess with param:=value args.

    ros2 launch launch/slam3d_go2.launch.py headless:=true rviz:=false explore:=true   # sim + lidar
    ros2 launch launch/lio_go2.launch.py                                               # this sidecar
"""
from launch import LaunchDescription
from launch.actions import ExecuteProcess

PARAMS = {
    "imu_topic": "/imu/data", "lidar_topic": "/points",
    "imu_frame": "base", "lidar_frame": "base", "base_frame": "base",   # go2_gz.urdf.xacro: all sensors gz_frame_id=base
    "odom_frame": "rko_odom", "autodetect": "false",
    "deskew": "false",           # sim cloud has no per-point timestamps
    "voxel_size": "0.25",        # room scale; rko default 1.0 is for outdoor
    "min_range": "0.4", "max_range": "30.0",   # drop inf (no-return) beams and far noise
    "max_correspondence_distance": "1.0",      # a walking gait moves the body ~0.1 m per scan
    "double_downsample": "false",              # 16-beam lidar is sparse
    "reset_on_registration_error": "true",     # keep running instead of aborting on a bad scan
    "use_sim_time": "true",
}


def generate_launch_description():
    cmd = ["ros2", "launch", "rko_lio", "odometry.launch.py"] + [f"{k}:={v}" for k, v in PARAMS.items()]
    return LaunchDescription([ExecuteProcess(cmd=cmd, output="screen")])
