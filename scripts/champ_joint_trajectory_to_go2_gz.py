#!/usr/bin/env python3
"""Bridge CHAMP JointTrajectory commands to Go2 Gazebo joint topics."""

import math
import re
import subprocess
import time

import rclpy
import numpy as np
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Imu
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

COMMAND_TIMEOUT_S = 0.35
ROLL_KP = 0.18
PITCH_KP = 0.16
ROLL_KD = 0.0
PITCH_KD = 0.0
POSE_CHECK_PERIOD_S = 0.2
RESET_COOLDOWN_S = 1.0


class ChampToGo2Gazebo(Node):
    def __init__(self):
        super().__init__("champ_joint_trajectory_to_go2_gz")
        self.declare_parameter("input_topic", "/champ/joint_trajectory")
        self.declare_parameter("output_prefix", "/go2/cmd")
        self.declare_parameter("publish_rate_limit_hz", 200.0)
        self.declare_parameter("map_neutral_to_stand", True)
        self.declare_parameter("delta_scale", 0.35)
        self.declare_parameter("enable_idle_recovery", True)
        self.declare_parameter("max_idle_drift_m", 0.12)
        self.declare_parameter("min_base_z", 0.20)
        self.declare_parameter("max_tilt_rad", 0.65)

        output_prefix = self.get_parameter("output_prefix").value.rstrip("/")
        input_topic = self.get_parameter("input_topic").value
        self._min_period = 1.0 / float(self.get_parameter("publish_rate_limit_hz").value)
        self._map_neutral_to_stand = bool(self.get_parameter("map_neutral_to_stand").value)
        self._delta_scale = float(self.get_parameter("delta_scale").value)
        self._enable_idle_recovery = bool(self.get_parameter("enable_idle_recovery").value)
        self._max_idle_drift_m = float(self.get_parameter("max_idle_drift_m").value)
        self._min_base_z = float(self.get_parameter("min_base_z").value)
        self._max_tilt_rad = float(self.get_parameter("max_tilt_rad").value)
        self._last_publish_time = 0.0
        self._last_cmd_time = 0.0
        self._last_pose_check_time = 0.0
        self._last_reset_time = 0.0
        self._cmd = np.zeros(3, dtype=np.float64)
        self._roll = 0.0
        self._pitch = 0.0
        self._roll_rate = 0.0
        self._pitch_rate = 0.0
        self._latest_champ_targets = dict(CHAMP_NEUTRAL)
        self._pubs = {
            joint: self.create_publisher(Float64, f"{output_prefix}/{joint}", 10)
            for joint in JOINT_LIMITS
        }

        self.create_subscription(JointTrajectory, input_topic, self._trajectory_cb, 10)
        self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, 10)
        self.create_subscription(Imu, "/imu/data", self._imu_cb, 10)
        self.create_timer(self._min_period, self._publish_targets)
        self.get_logger().info(f"bridging {input_topic} to {output_prefix}/<joint>")

    def _cmd_cb(self, msg):
        self._cmd[:] = [msg.linear.x, msg.linear.y, msg.angular.z]
        self._last_cmd_time = time.monotonic()

    def _cmd_is_active(self):
        if time.monotonic() - self._last_cmd_time > COMMAND_TIMEOUT_S:
            self._cmd[:] = 0.0
            return False
        return float(np.linalg.norm(self._cmd)) >= 1.0e-4

    def _imu_cb(self, msg):
        q = msg.orientation
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        self._roll = float(math.atan2(sinr_cosp, cosr_cosp))

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        self._pitch = float(math.asin(max(-1.0, min(1.0, sinp))))

        self._roll_rate = float(msg.angular_velocity.x)
        self._pitch_rate = float(msg.angular_velocity.y)

    def _apply_posture_feedback(self, targets):
        corrected = dict(targets)
        roll_corr = float(np.clip(
            ROLL_KP * self._roll + ROLL_KD * self._roll_rate, -0.10, 0.10
        ))
        pitch_corr = float(np.clip(
            PITCH_KP * self._pitch + PITCH_KD * self._pitch_rate, -0.12, 0.12
        ))

        corrected["FL_hip_joint"] -= roll_corr
        corrected["RL_hip_joint"] -= roll_corr
        corrected["FR_hip_joint"] += roll_corr
        corrected["RR_hip_joint"] += roll_corr

        corrected["FL_thigh_joint"] += pitch_corr
        corrected["FR_thigh_joint"] += pitch_corr
        corrected["RL_thigh_joint"] -= pitch_corr
        corrected["RR_thigh_joint"] -= pitch_corr
        return corrected

    def _trajectory_cb(self, msg):
        if not msg.points:
            return
        point = msg.points[0]
        if len(point.positions) < len(msg.joint_names):
            self.get_logger().warn("ignoring trajectory point with missing positions")
            return

        for index, joint in enumerate(msg.joint_names):
            if joint not in self._latest_champ_targets:
                continue

            value = float(point.positions[index])
            if not math.isfinite(value):
                continue
            self._latest_champ_targets[joint] = value

    def _build_targets(self):
        targets = dict(GO2_STAND)
        use_champ_delta = self._cmd_is_active()
        for joint, value in self._latest_champ_targets.items():
            if self._map_neutral_to_stand:
                if use_champ_delta:
                    value = GO2_STAND[joint] + self._delta_scale * (value - CHAMP_NEUTRAL[joint])
                else:
                    value = GO2_STAND[joint]
            targets[joint] = value
        return self._apply_posture_feedback(targets)

    def _publish_targets(self):
        now = time.monotonic()
        if now - self._last_publish_time < self._min_period:
            return
        self._last_publish_time = now

        if self._enable_idle_recovery and not self._cmd_is_active():
            self._maybe_reset_idle()

        targets = self._build_targets()
        for joint, value in targets.items():
            pub = self._pubs[joint]
            lower, upper = JOINT_LIMITS[joint]
            clamped = min(max(value, lower), upper)
            out = Float64()
            out.data = clamped
            pub.publish(out)

    def _maybe_reset_idle(self):
        now = time.monotonic()
        if now - self._last_pose_check_time < POSE_CHECK_PERIOD_S:
            return
        self._last_pose_check_time = now

        pose = self._read_go2_pose()
        if pose is None:
            return
        x, y, z, roll, pitch = pose
        drift = math.hypot(x, y)
        fallen = z < self._min_base_z or max(abs(roll), abs(pitch)) > self._max_tilt_rad
        drifted = drift > self._max_idle_drift_m
        if not fallen and not drifted:
            return
        if now - self._last_reset_time < RESET_COOLDOWN_S:
            return
        self._last_reset_time = now
        reason = "fallen" if fallen else "idle drift"
        self.get_logger().warn(
            f"{reason} detected x={x:.2f} y={y:.2f} z={z:.2f}; resetting upright"
        )
        self._reset_upright()

    def _read_go2_pose(self):
        try:
            result = subprocess.run(
                ["gz", "topic", "-e", "-n", "1", "-t", "/world/go2_rl/pose/info"],
                check=False,
                text=True,
                capture_output=True,
                timeout=1.0,
            )
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0:
            return None
        match = re.search(
            r'name: "go2".*?position \{(.*?)\}.*?orientation \{(.*?)\}',
            result.stdout,
            re.S,
        )
        if not match:
            return None

        def value(block, key, default=0.0):
            item = re.search(rf"\b{key}: ([^\s]+)", block)
            return float(item.group(1)) if item else default

        pos, ori = match.groups()
        x = value(pos, "x")
        y = value(pos, "y")
        z = value(pos, "z")
        qx = value(ori, "x")
        qy = value(ori, "y")
        qz = value(ori, "z")
        qw = value(ori, "w", 1.0)
        roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
        return x, y, z, roll, pitch

    def _reset_upright(self):
        subprocess.run(
            [
                "gz", "service",
                "-s", "/world/go2_rl/control",
                "--reqtype", "gz.msgs.WorldControl",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "3000",
                "--req", "pause: true",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "gz", "service",
                "-s", "/world/go2_rl/set_pose",
                "--reqtype", "gz.msgs.Pose",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "3000",
                "--req", "name: 'go2' position: {x: 0 y: 0 z: 0.32} orientation: {w: 1 x: 0 y: 0 z: 0}",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        for _ in range(25):
            targets = self._apply_posture_feedback(GO2_STAND)
            for joint, value in targets.items():
                msg = Float64()
                msg.data = float(value)
                self._pubs[joint].publish(msg)
            time.sleep(0.01)
        subprocess.run(
            [
                "gz", "service",
                "-s", "/world/go2_rl/control",
                "--reqtype", "gz.msgs.WorldControl",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "3000",
                "--req", "pause: false",
            ],
            check=False,
            text=True,
            capture_output=True,
        )


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
