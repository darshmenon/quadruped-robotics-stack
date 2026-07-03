#!/usr/bin/env python3
"""Hold the Go2 in a standing pose in Gazebo Sim."""

import argparse
import os
import sys
import subprocess
import time
import re

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intelligence.gait.gait_scheduler import GaitScheduler
from teleop_go2_gz import CTRL_DT, GAITS, Gait, _joint_targets


JOINT_TARGETS = {
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
TILT_RESET_RAD = 0.75
RESET_COOLDOWN_S = 2.0
POSE_CHECK_PERIOD_S = 0.5
MIN_BASE_Z = 0.18
ROLL_KP = 0.18
PITCH_KP = 0.16


def _gz_service(service, reqtype, reptype, request, timeout=3000):
    return subprocess.run(
        [
            "gz", "service",
            "-s", service,
            "--reqtype", reqtype,
            "--reptype", reptype,
            "--timeout", str(timeout),
            "--req", request,
        ],
        check=False,
        text=True,
        capture_output=True,
    )


class StandPublisher(Node):
    def __init__(self):
        super().__init__("go2_stand_publisher")
        self._cmd = np.zeros(3, dtype=np.float64)
        self._last_cmd_time = 0.0
        self._last_reset_time = 0.0
        self._last_pose_check_time = 0.0
        self._roll = 0.0
        self._pitch = 0.0
        self._scheduler = GaitScheduler()
        self._leg_phases = np.array(GAITS[Gait.STAND].phase_offsets) * 2.0 * np.pi
        self._pubs = {
            joint: self.create_publisher(Float64, f"/go2/cmd/{joint}", 10)
            for joint in JOINT_TARGETS
        }
        self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, 10)
        self.create_subscription(Imu, "/imu/data", self._imu_cb, 10)

    def _cmd_cb(self, msg):
        self._cmd[:] = [
            float(np.clip(msg.linear.x, -0.15, 0.25)),
            float(np.clip(msg.linear.y, -0.12, 0.12)),
            float(np.clip(msg.angular.z, -0.25, 0.25)),
        ]
        self._last_cmd_time = time.monotonic()

    def _imu_cb(self, msg):
        q = msg.orientation
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        self._roll = float(np.arctan2(sinr_cosp, cosr_cosp))

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        self._pitch = float(np.arcsin(np.clip(sinp, -1.0, 1.0)))

    def _cmd_is_active(self):
        if time.monotonic() - self._last_cmd_time > COMMAND_TIMEOUT_S:
            self._cmd[:] = 0.0
            return False
        return float(np.linalg.norm(self._cmd)) >= 1.0e-4

    def _apply_posture_feedback(self, targets):
        corrected = np.array(targets, dtype=np.float64)
        roll_corr = float(np.clip(ROLL_KP * self._roll, -0.10, 0.10))
        pitch_corr = float(np.clip(PITCH_KP * self._pitch, -0.12, 0.12))

        # Hip ab/adduction for roll support: left legs are FL/RL, right are FR/RR.
        corrected[0] -= roll_corr
        corrected[6] -= roll_corr
        corrected[3] += roll_corr
        corrected[9] += roll_corr

        # Thigh offset for pitch support: front legs are FL/FR, rear are RL/RR.
        corrected[1] += pitch_corr
        corrected[4] += pitch_corr
        corrected[7] -= pitch_corr
        corrected[10] -= pitch_corr
        return np.clip(corrected, -2.7, 2.7)

    def maybe_recover(self):
        imu_fallen = max(abs(self._roll), abs(self._pitch)) >= TILT_RESET_RAD
        pose_fallen = self._model_pose_fallen()
        if not imu_fallen and not pose_fallen:
            return
        now = time.monotonic()
        if now - self._last_reset_time < RESET_COOLDOWN_S:
            return
        self._last_reset_time = now
        self._cmd[:] = 0.0
        self.get_logger().warn(
            f"fall detected roll={self._roll:.2f} pitch={self._pitch:.2f}; resetting upright"
        )
        _gz_service(
            "/world/go2_rl/control",
            "gz.msgs.WorldControl",
            "gz.msgs.Boolean",
            "pause: true",
        )
        _gz_service(
            "/world/go2_rl/set_pose",
            "gz.msgs.Pose",
            "gz.msgs.Boolean",
            "name: 'go2' position: {x: 0 y: 0 z: 0.45} orientation: {w: 1 x: 0 y: 0 z: 0}",
        )
        for _ in range(25):
            self.publish_once()
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.01)
        _gz_service(
            "/world/go2_rl/control",
            "gz.msgs.WorldControl",
            "gz.msgs.Boolean",
            "pause: false",
        )

    def _model_pose_fallen(self):
        now = time.monotonic()
        if now - self._last_pose_check_time < POSE_CHECK_PERIOD_S:
            return False
        self._last_pose_check_time = now

        result = subprocess.run(
            ["gz", "topic", "-e", "-n", "1", "-t", "/world/go2_rl/pose/info"],
            check=False,
            text=True,
            capture_output=True,
            timeout=1.0,
        )
        if result.returncode != 0:
            return False

        match = re.search(
            r'name: "go2".*?position \{.*?z: ([^\s]+).*?\}.*?orientation \{(.*?)\}',
            result.stdout,
            re.S,
        )
        if not match:
            return False

        try:
            base_z = float(match.group(1))
        except ValueError:
            return False
        return base_z < MIN_BASE_Z

    def publish_once(self):
        if not self._cmd_is_active():
            targets = list(JOINT_TARGETS.values())
        else:
            speed = float(np.hypot(self._cmd[0], self._cmd[2] * 0.3))
            gait = self._scheduler.get_gait_params(speed)
            self._leg_phases = (
                self._leg_phases + 2.0 * np.pi * gait.frequency * CTRL_DT
            ) % (2.0 * np.pi)
            targets = np.clip(_joint_targets(self._leg_phases, self._cmd, gait), -2.7, 2.7)

        targets = self._apply_posture_feedback(targets)
        for pub, target in zip(self._pubs.values(), targets):
            msg = Float64()
            msg.data = float(target)
            pub.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-upright", action="store_true")
    parser.add_argument("--unpause", action="store_true")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=-1.0,
        help="Exit after this many seconds. Negative values run until stopped.",
    )
    args = parser.parse_args()

    if args.reset_upright:
        _gz_service(
            "/world/go2_rl/control",
            "gz.msgs.WorldControl",
            "gz.msgs.Boolean",
            "pause: true",
        )
        _gz_service(
            "/world/go2_rl/set_pose",
            "gz.msgs.Pose",
            "gz.msgs.Boolean",
            "name: 'go2' position: {x: 0 y: 0 z: 0.45} orientation: {w: 1 x: 0 y: 0 z: 0}",
        )

    rclpy.init()
    node = StandPublisher()

    # Prime controllers before physics resumes.
    for _ in range(50):
        node.publish_once()
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(0.01)

    if args.unpause:
        _gz_service(
            "/world/go2_rl/control",
            "gz.msgs.WorldControl",
            "gz.msgs.Boolean",
            "pause: false",
        )

    period = 1.0 / args.rate
    end_time = None
    if args.duration_seconds >= 0.0:
        end_time = time.monotonic() + args.duration_seconds

    try:
        while rclpy.ok():
            if end_time is not None and time.monotonic() >= end_time:
                break
            node.maybe_recover()
            node.publish_once()
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
