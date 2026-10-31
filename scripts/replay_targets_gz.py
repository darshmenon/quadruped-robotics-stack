#!/usr/bin/env python3
"""Replay a (T, 12) joint-target array into native Gazebo open-loop, on sim time (50 Hz).

Targets are in the policy's actuator order [hips, thighs, calves] x [FL, FR, RL, RR] (same as
rl_cmd_vel_go2_gz.ACT_ORDER). Used to test whether the Gazebo plant reproduces a MuJoCo walk:
record targets from a MuJoCo rollout (data.ctrl[:12]), play them here, and read /odom.

    python3 scripts/replay_targets_gz.py --targets /tmp/mujoco_targets.npy --hold 3
"""
import argparse

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float64

ACT_ORDER = [f"{leg}_{j}_joint" for j in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]


class Replay(Node):
    def __init__(self, targets, hold_s):
        super().__init__("replay_targets_gz", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.t, self.i, self.hold_ticks, self.n = targets, 0, int(hold_s * 50), 0
        self.pubs = [self.create_publisher(Float64, f"/go2/cmd/{n}", 10) for n in ACT_ORDER]
        self.create_timer(0.02, self.tick)

    def tick(self):
        self.n += 1
        row = self.t[0] if self.n <= self.hold_ticks else self.t[min(self.i, len(self.t) - 1)]
        if self.n > self.hold_ticks:
            self.i += 1
        for p, v in zip(self.pubs, row):
            p.publish(Float64(data=float(v)))
        if self.i >= len(self.t):
            self.get_logger().info("replay finished")
            raise SystemExit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to hold the first target before replaying")
    args, ros = ap.parse_known_args()
    rclpy.init(args=ros)
    try:
        rclpy.spin(Replay(np.load(args.targets), args.hold))
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
