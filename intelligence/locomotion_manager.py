"""
Locomotion Manager — ROS2 node that fuses IMU + foot contact forces + raw
velocity commands through TerrainEstimator + GaitScheduler + AdaptiveController
to publish safe, terrain-adapted velocity commands and gait status.

Subscriptions:
    /cmd_vel_raw          geometry_msgs/Twist          desired velocity (nav/teleop)
    /imu                  sensor_msgs/Imu              base orientation
    /foot_forces          std_msgs/Float32MultiArray   [FL, FR, RL, RR] contact forces (N)

Publications:
    /cmd_vel              geometry_msgs/Twist   safe adapted velocity command
    /locomotion_status    std_msgs/String       JSON: terrain, gait, speed, slope_deg

Parameters:
    update_rate   (float, default 50.0)   control loop Hz
    max_speed     (float, default 1.5)    hard cap on linear velocity (m/s)
    contact_scale (float, default 1.0)    multiplier applied to /foot_forces values

Usage:
    python3 intelligence/locomotion_manager.py

    ros2 run quadruped_dog_rl locomotion_manager --ros-args \
        -p update_rate:=50.0 -p max_speed:=1.2
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray, String

from intelligence.terrain.adaptive_controller import AdaptiveController


class LocomotionManager(Node):
    def __init__(self):
        super().__init__("locomotion_manager")

        self.declare_parameter("update_rate",   50.0)
        self.declare_parameter("max_speed",      1.5)
        self.declare_parameter("contact_scale",  1.0)

        rate                 = float(self.get_parameter("update_rate").value)
        self._max_speed      = float(self.get_parameter("max_speed").value)
        self._contact_scale  = float(self.get_parameter("contact_scale").value)

        # latest state from subscribers
        self._desired_vx  = 0.0
        self._desired_wz  = 0.0
        self._imu_roll    = 0.0
        self._imu_pitch   = 0.0
        self._contacts    = [100.0, 100.0, 100.0, 100.0]  # nominal standing force

        self._ctrl = AdaptiveController()

        self.create_subscription(Twist,             "/cmd_vel_raw",  self._cmd_cb,     10)
        self.create_subscription(Imu,               "/imu",          self._imu_cb,     10)
        self.create_subscription(Float32MultiArray, "/foot_forces",  self._contact_cb, 10)

        self._cmd_pub    = self.create_publisher(Twist,  "/cmd_vel",           10)
        self._status_pub = self.create_publisher(String, "/locomotion_status", 10)

        self.create_timer(1.0 / rate, self._loop)
        self.get_logger().info(
            f"Locomotion manager ready — {rate} Hz, max_speed={self._max_speed} m/s"
        )

    # ------------------------------------------------------------------ #

    def _cmd_cb(self, msg: Twist) -> None:
        self._desired_vx = float(msg.linear.x)
        self._desired_wz = float(msg.angular.z)

    def _imu_cb(self, msg: Imu) -> None:
        q = msg.orientation
        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        self._imu_roll = math.atan2(sinr, cosr)

        sinp = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
        self._imu_pitch = math.asin(sinp)

    def _contact_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 4:
            self._contacts = [float(v) * self._contact_scale for v in msg.data[:4]]

    def _loop(self) -> None:
        adapted = self._ctrl.adapt(
            desired_speed=min(self._desired_vx, self._max_speed),
            desired_angular=self._desired_wz,
            imu_roll=self._imu_roll,
            imu_pitch=self._imu_pitch,
            contacts=self._contacts,
        )

        cmd = Twist()
        cmd.linear.x  = adapted.linear_x
        cmd.angular.z = adapted.angular_z
        self._cmd_pub.publish(cmd)

        status = String()
        status.data = json.dumps({
            "terrain":        adapted.terrain,
            "gait":           adapted.gait,
            "speed":          round(adapted.linear_x, 3),
            "angular":        round(adapted.angular_z, 3),
            "slope_deg":      adapted.slope_deg,
            "foot_clearance": adapted.foot_clearance,
        })
        self._status_pub.publish(status)


def main():
    rclpy.init()
    node = LocomotionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
