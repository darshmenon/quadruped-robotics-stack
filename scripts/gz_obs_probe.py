#!/usr/bin/env python3
"""Record the signals rl_cmd_vel_go2_gz.py feeds the policy (joint pos/vel, IMU quaternion + gyro) at
50 Hz sim time, to compare against a MuJoCo replay of the same targets.

    python3 scripts/gz_obs_probe.py --seconds 20 --out /tmp/gz_obs.npz
"""
import argparse

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState

JOINTS = [f"{leg}_{j}_joint" for leg in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]


class Probe(Node):
    def __init__(self, seconds):
        super().__init__("gz_obs_probe", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.n_end, self.rows = int(seconds * 50), []
        self.q, self.qd = np.zeros(12), np.zeros(12)
        self.quat, self.gyro = np.array([1.0, 0, 0, 0]), np.zeros(3)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, "/joint_states", self.js, qos)
        self.create_subscription(Imu, "/imu/data", self.imu, qos)
        self.create_timer(0.02, self.tick)

    def js(self, m):
        i = {n: k for k, n in enumerate(m.name)}
        if all(n in i for n in JOINTS):
            self.q = np.array([m.position[i[n]] for n in JOINTS])
            self.qd = np.array([m.velocity[i[n]] for n in JOINTS])

    def imu(self, m):
        o, a = m.orientation, m.angular_velocity
        self.quat, self.gyro = np.array([o.w, o.x, o.y, o.z]), np.array([a.x, a.y, a.z])

    def tick(self):
        self.rows.append(np.concatenate([self.q, self.qd, self.quat, self.gyro]))
        if len(self.rows) >= self.n_end:
            raise SystemExit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    args, ros = ap.parse_known_args()
    rclpy.init(args=ros)
    node = Probe(args.seconds)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    d = np.array(node.rows)
    np.savez(args.out, q=d[:, :12], qd=d[:, 12:24], quat=d[:, 24:28], gyro=d[:, 28:31])
    print("saved", args.out, d.shape)


if __name__ == "__main__":
    main()
