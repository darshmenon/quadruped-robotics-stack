#!/usr/bin/env python3
"""Publish odom and odom->base TF from a live Gazebo pose subscription."""

import argparse
import math
import os
import threading

# gz.msgs10's generated _pb2 modules were built against a protobuf version
# newer than what upb-based rclpy/system protobuf expects here -- importing
# them under the default (C++) protobuf implementation raises "Descriptors
# cannot be created directly". Force the pure-Python implementation before
# the gz.msgs10/gz.transport13 imports below pull protobuf in.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import gz.transport13 as gz_transport
import rclpy
from geometry_msgs.msg import TransformStamped
from gz.msgs10.pose_v_pb2 import Pose_V
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def _yaw_from_quat(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _angle_delta(current, previous):
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


class GazeboPoseOdom(Node):
    def __init__(self, world, model, odom_frame, base_frame, rate):
        super().__init__("gz_pose_to_odom")
        self.model = model
        self.odom_frame = odom_frame
        self.base_frame = base_frame
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_pose = None
        self.last_time = None

        # gz-transport delivers messages on its own internal thread -- guard
        # the handoff to the ROS timer below with a lock instead of spawning
        # a `gz topic -e -n1` subprocess per tick (the old approach), which
        # jittered badly under load and produced stale odom->base TF that
        # blew RTAB-Map's wait_for_transform and octomap_server's message
        # filter queue.
        self._pose_lock = threading.Lock()
        self._latest_pose = None

        self._gz_node = gz_transport.Node()
        topic = f"/world/{world}/pose/info"
        if not self._gz_node.subscribe(Pose_V, topic, self._on_pose_v):
            self.get_logger().error(f"failed to subscribe to {topic}")

        self.timer = self.create_timer(1.0 / rate, self.publish_pose)

    def _on_pose_v(self, msg):
        for pose in msg.pose:
            if pose.name == self.model:
                with self._pose_lock:
                    self._latest_pose = {
                        # time the pose was valid in the sim: stamping with the (later) timer tick made /odom
                        # lag its own stamp by the delivery latency, ~55 ms against the IMU
                        "stamp_ns": msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nsec,
                        "x": pose.position.x,
                        "y": pose.position.y,
                        "z": pose.position.z,
                        "qx": pose.orientation.x,
                        "qy": pose.orientation.y,
                        "qz": pose.orientation.z,
                        "qw": pose.orientation.w,
                    }
                return

    def publish_pose(self):
        with self._pose_lock:
            pose = self._latest_pose

        if pose is None:
            return

        now = self.get_clock().now()
        if pose.get("stamp_ns"):
            from rclpy.time import Time
            now = Time(nanoseconds=pose["stamp_ns"], clock_type=self.get_clock().clock_type)  # the pose's own sim time
        if self.last_time is not None and now <= self.last_time:
            return                                         # same pose message already published
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
