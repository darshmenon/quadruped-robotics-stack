#!/usr/bin/env python3
"""Publish odom and odom->base TF from the Gazebo model pose."""

import argparse
import math
import re
import subprocess

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def _field(block, name, default):
    match = re.search(rf"\b{name}:\s*([-+0-9.eE]+)", block)
    return float(match.group(1)) if match else default


def _section(text, marker):
    start = text.find(marker)
    if start < 0:
        return ""
    next_pose = text.find("\npose {", start + len(marker))
    return text[start:] if next_pose < 0 else text[start:next_pose]


def _parse_pose(text, model_name):
    block = _section(text, f'name: "{model_name}"')
    if not block:
        return None

    position = re.search(r"position\s*\{([^}]*)\}", block, re.S)
    orientation = re.search(r"orientation\s*\{([^}]*)\}", block, re.S)
    pos = position.group(1) if position else ""
    quat = orientation.group(1) if orientation else ""

    return {
        "x": _field(pos, "x", 0.0),
        "y": _field(pos, "y", 0.0),
        "z": _field(pos, "z", 0.0),
        "qx": _field(quat, "x", 0.0),
        "qy": _field(quat, "y", 0.0),
        "qz": _field(quat, "z", 0.0),
        "qw": _field(quat, "w", 1.0),
    }


def _yaw_from_quat(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _angle_delta(current, previous):
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


class GazeboPoseOdom(Node):
    def __init__(self, world, model, odom_frame, base_frame, rate):
        super().__init__("gz_pose_to_odom")
        self.topic = f"/world/{world}/pose/info"
        self.model = model
        self.odom_frame = odom_frame
        self.base_frame = base_frame
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_pose = None
        self.last_time = None
        self.timer = self.create_timer(1.0 / rate, self.publish_pose)

    def publish_pose(self):
        try:
            result = subprocess.run(
                ["gz", "topic", "-e", "-n", "1", "-t", self.topic],
                capture_output=True,
                text=True,
                timeout=0.25,
            )
        except subprocess.TimeoutExpired:
            return

        pose = _parse_pose(result.stdout, self.model)
        if pose is None:
            return

        now = self.get_clock().now()
        stamp = now.to_msg()
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = pose["x"]
        odom.pose.pose.position.y = pose["y"]
        odom.pose.pose.position.z = pose["z"]
        odom.pose.pose.orientation.x = pose["qx"]
        odom.pose.pose.orientation.y = pose["qy"]
        odom.pose.pose.orientation.z = pose["qz"]
        odom.pose.pose.orientation.w = pose["qw"]

        if self.last_pose is not None and self.last_time is not None:
            dt = (now - self.last_time).nanoseconds / 1e9
            if dt > 0.0:
                odom.twist.twist.linear.x = (pose["x"] - self.last_pose["x"]) / dt
                odom.twist.twist.linear.y = (pose["y"] - self.last_pose["y"]) / dt
                yaw = _yaw_from_quat(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
                last_yaw = _yaw_from_quat(
                    self.last_pose["qx"],
                    self.last_pose["qy"],
                    self.last_pose["qz"],
                    self.last_pose["qw"],
                )
                odom.twist.twist.angular.z = _angle_delta(yaw, last_yaw) / dt

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = pose["x"]
        transform.transform.translation.y = pose["y"]
        transform.transform.translation.z = pose["z"]
        transform.transform.rotation.x = pose["qx"]
        transform.transform.rotation.y = pose["qy"]
        transform.transform.rotation.z = pose["qz"]
        transform.transform.rotation.w = pose["qw"]

        self.odom_pub.publish(odom)
        self.tf_broadcaster.sendTransform(transform)
        self.last_pose = pose
        self.last_time = now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", default="go2_rl")
    parser.add_argument("--model", default="go2")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base")
    parser.add_argument("--rate", type=float, default=10.0)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = GazeboPoseOdom(args.world, args.model, args.odom_frame, args.base_frame, args.rate)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
