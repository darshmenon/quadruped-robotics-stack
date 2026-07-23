#!/usr/bin/env python3
"""Convert geometry_msgs/Twist velocity commands into Go2 joint targets."""

import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intelligence.gait.gait_scheduler import GaitScheduler
from teleop_go2_gz import JOINTS, CTRL_DT, GAITS, Gait, _joint_targets


class CmdVelGo2(Node):
    def __init__(self):
        super().__init__("cmd_vel_go2_gz")
        self._cmd = np.zeros(3, dtype=np.float64)
        self._scheduler = GaitScheduler()
        self._cycle_phase = 0.0
        self._pubs = [self.create_publisher(Float64, f"/go2/cmd/{name}", 10) for name in JOINTS]
        self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, 10)

    def _cmd_cb(self, msg):
        self._cmd[:] = [msg.linear.x, msg.linear.y, msg.angular.z]

    def publish_targets(self):
        speed = float(np.hypot(self._cmd[0], self._cmd[2] * 0.3))
        gait = self._scheduler.get_gait_params(speed)
        self._cycle_phase = (
            self._cycle_phase + 2.0 * np.pi * gait.frequency * CTRL_DT
        ) % (2.0 * np.pi)
        leg_phases = (
            self._cycle_phase + np.array(gait.phase_offsets) * 2.0 * np.pi
        ) % (2.0 * np.pi)
        targets = np.clip(_joint_targets(leg_phases, self._cmd, gait), -2.7, 2.7)
        for pub, value in zip(self._pubs, targets):
            msg = Float64()
            msg.data = float(value)
            pub.publish(msg)


def main():
    rclpy.init()
    node = CmdVelGo2()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            node.publish_targets()
            time.sleep(CTRL_DT)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
