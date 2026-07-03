#!/usr/bin/env python3
"""Bridge CHAMP JointTrajectory commands to Go2 Gazebo joint topics."""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory


JOINT_LIMITS = {
    "FL_hip_joint": (-1.0472, 1.0472),
    "FL_thigh_joint": (-1.5708, 3.4907),
    "FL_calf_joint": (-2.7227, -0.83776),
    "FR_hip_joint": (-1.0472, 1.0472),
    "FR_thigh_joint": (-1.5708, 3.4907),
    "FR_calf_joint": (-2.7227, -0.83776),
    "RL_hip_joint": (-1.0472, 1.0472),
    "RL_thigh_joint": (-0.5236, 4.5379),
    "RL_calf_joint": (-2.7227, -0.83776),
    "RR_hip_joint": (-1.0472, 1.0472),
    "RR_thigh_joint": (-0.5236, 4.5379),
    "RR_calf_joint": (-2.7227, -0.83776),
}

CHAMP_NEUTRAL = {
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 1.0281174182891846,
    "FL_calf_joint": -2.056234836578369,
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 1.0281174182891846,
    "FR_calf_joint": -2.056234836578369,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 1.0281174182891846,
    "RL_calf_joint": -2.056234836578369,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 1.0281174182891846,
    "RR_calf_joint": -2.056234836578369,
}

GO2_STAND = {
    "FL_hip_joint": 0.10,
    "FL_thigh_joint": 0.80,
    "FL_calf_joint": -1.50,
    "FR_hip_joint": -0.10,
    "FR_thigh_joint": 0.80,
    "FR_calf_joint": -1.50,
    "RL_hip_joint": 0.10,
    "RL_thigh_joint": 1.00,
    "RL_calf_joint": -1.50,
    "RR_hip_joint": -0.10,
    "RR_thigh_joint": 1.00,
    "RR_calf_joint": -1.50,
}


class ChampToGo2Gazebo(Node):
    def __init__(self):
        super().__init__("champ_joint_trajectory_to_go2_gz")
        self.declare_parameter("input_topic", "/champ/joint_trajectory")
        self.declare_parameter("output_prefix", "/go2/cmd")
        self.declare_parameter("publish_rate_limit_hz", 200.0)
        self.declare_parameter("map_neutral_to_stand", True)
        self.declare_parameter("delta_scale", 0.6)

        output_prefix = self.get_parameter("output_prefix").value.rstrip("/")
        input_topic = self.get_parameter("input_topic").value
        self._min_period = 1.0 / float(self.get_parameter("publish_rate_limit_hz").value)
        self._map_neutral_to_stand = bool(self.get_parameter("map_neutral_to_stand").value)
        self._delta_scale = float(self.get_parameter("delta_scale").value)
        self._last_publish_time = 0.0
        self._pubs = {
            joint: self.create_publisher(Float64, f"{output_prefix}/{joint}", 10)
            for joint in JOINT_LIMITS
        }

        self.create_subscription(JointTrajectory, input_topic, self._trajectory_cb, 10)
        self.get_logger().info(f"bridging {input_topic} to {output_prefix}/<joint>")

    def _trajectory_cb(self, msg):
        now = time.monotonic()
        if now - self._last_publish_time < self._min_period:
            return
        self._last_publish_time = now

        if not msg.points:
            return
        point = msg.points[0]
        if len(point.positions) < len(msg.joint_names):
            self.get_logger().warn("ignoring trajectory point with missing positions")
            return

        for index, joint in enumerate(msg.joint_names):
            pub = self._pubs.get(joint)
            if pub is None:
                continue

            value = float(point.positions[index])
            if not math.isfinite(value):
                continue
            if self._map_neutral_to_stand:
                value = GO2_STAND[joint] + self._delta_scale * (value - CHAMP_NEUTRAL[joint])
            lower, upper = JOINT_LIMITS[joint]
            clamped = min(max(value, lower), upper)
            out = Float64()
            out.data = clamped
            pub.publish(out)


def main():
    rclpy.init()
    node = ChampToGo2Gazebo()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
