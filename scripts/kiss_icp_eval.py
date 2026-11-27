#!/usr/bin/env python3
"""Score KISS-ICP (lidar only) on a Go2 bag against simulator /odom.

Reads /points straight out of the bag. No IMU, no extrinsics, no ros2 bag play.
The cloud has no per-point time, so deskew is off (config/kiss_icp_go2.yaml).
Source ROS 2 first (`source /opt/ros/humble/setup.bash`).

  python3 scripts/kiss_icp_eval.py training/logs/bags/go2_arena_walk1 --repeats 3
  python3 scripts/kiss_icp_eval.py training/logs/bags/go2_outdoor_weave2 --repeats 3
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

REPO = Path(__file__).resolve().parents[1]
FUS = Path.home() / "fusion_tools"
CFG = REPO / "config" / "kiss_icp_go2.yaml"


def _poses_tum(poses, stamps, path):
    with open(path, "w") as f:
        for T, t in zip(poses, stamps):
            tx, ty, tz = T[:3, 3]
            qw, qx, qy, qz = Quaternion(matrix=T, atol=0.01).elements
            f.write(f"{t:.9f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


def run_bag(bag, topic, n_scans):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from sensor_msgs_py import point_cloud2
    from kiss_icp.config import load_config
    from kiss_icp.kiss_icp import KissICP

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=""), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in types:
        sys.exit(f"topic {topic} not in bag. available: {list(types)}")
    msg_type = get_message(types[topic])
    odom = KissICP(load_config(CFG))
    poses, stamps = [], []
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name != topic:
            continue
        msg = deserialize_message(data, msg_type)
        raw = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        frame = np.column_stack((raw["x"], raw["y"], raw["z"])).astype(np.float64)
        frame = frame[np.isfinite(frame).all(axis=1)]
        frame = frame[np.isfinite(frame).all(axis=1)]
        ts = np.zeros(len(frame))  # sim scan is one instant; config keeps deskew off
        odom.register_frame(frame, ts)
        poses.append(odom.last_pose.copy())
        stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        if len(poses) == 1:
            print(f"first cloud: {len(frame)} points", flush=True)
        elif len(poses) % 200 == 0:
            print(f"  {len(poses)} scans", flush=True)
        if n_scans > 0 and len(poses) >= n_scans:
            break
    return poses, stamps, len(poses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--topic", default="/points")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--n-scans", type=int, default=-1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    bag = Path(a.bag)
    out = Path(a.out) if a.out else REPO / "training" / "logs" / "kiss_icp" / bag.name
    out.mkdir(parents=True, exist_ok=True)
    src = "source /opt/ros/humble/setup.bash >/dev/null 2>&1; "
    gt = out / "gt.tum"
    if not gt.exists():
        subprocess.run(
            ["bash", "-c", src + f"python3 {FUS}/odom_to_tum.py {bag} /odom {gt}"],
            check=True,
        )
    digests = []
    for k in range(1, a.repeats + 1):
        poses, stamps, n = run_bag(bag, a.topic, a.n_scans)
        tum = out / f"est_{k}.tum"
        _poses_tum(poses, stamps, tum)
        digest = subprocess.check_output(["sha256sum", str(tum)], text=True).split()[0]
        digests.append(digest)
        print(f"== {bag.name} repeat {k}: {n} scans, tum {digest[:12]}", flush=True)
        scored = subprocess.run(
            [str(FUS / "eval_traj.sh"), str(gt), str(tum)], capture_output=True, text=True
        )
        sys.stdout.write(scored.stdout)
        if scored.returncode != 0:
            sys.stderr.write(scored.stderr)
            if "APE" not in scored.stdout:
                sys.exit(scored.returncode)
    print(f"identical repeats: {len(set(digests)) == 1}")


if __name__ == "__main__":
    main()
