#!/usr/bin/env python3
"""Check the core Go2 autonomy topics and TF links."""

import argparse
import math
import sys
import time
from collections import deque

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener

from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


DEFAULT_TOPICS = {
    "/odom": Odometry,
    "/points": PointCloud2,
    "/map": OccupancyGrid,
    "/obstacle_tracker/state": String,
}


# Minimum message rates (Hz) for topics where a stalled-but-fresh stream means a broken stack.
MIN_RATES = {"/odom": 5.0, "/points": 1.0}
# Fall detection from /odom: base this low or tilted this far (rad) counts as fallen.
FALL_MIN_Z = 0.15
FALL_MAX_TILT = math.radians(60.0)


def tilt_from_quat(x, y, z, w):
    """Angle between body z-axis and world z (rad)."""
    cos_tilt = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(max(-1.0, min(1.0, cos_tilt)))


class StackCheck(Node):
    def __init__(self, topics):
        super().__init__("go2_stack_check")
        self.seen = {name: None for name in topics}
        self.stamps = {name: deque(maxlen=50) for name in topics}
        self.pose = None  # (z, tilt) from the latest /odom
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        for name, msg_type in topics.items():
            self.create_subscription(msg_type, name, self._callback(name), qos)

    def _callback(self, topic):
        def cb(msg):
            now = time.monotonic()
            self.seen[topic] = now
            self.stamps[topic].append(now)
            if topic == "/odom":
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                self.pose = (p.z, tilt_from_quat(q.x, q.y, q.z, q.w))
        return cb

    def rate_hz(self, topic):
        t = self.stamps[topic]
        if len(t) < 2 or t[-1] == t[0]:
            return 0.0
        return (len(t) - 1) / (t[-1] - t[0])

    def fall_report(self):
        if self.pose is None:
            return [(False, "fall check", "no /odom pose")]
        z, tilt = self.pose
        fell = z < FALL_MIN_Z or tilt > FALL_MAX_TILT
        return [(not fell, "fall check", f"base z={z:.2f} m, tilt={math.degrees(tilt):.0f} deg")]

    def topic_report(self, max_age):
        now = time.monotonic()
        rows = []
        ok = True
        for topic, stamp in self.seen.items():
            if stamp is None:
                rows.append((False, topic, "no messages"))
                ok = False
                continue
            age = now - stamp
            fresh = age <= max_age
            rate = self.rate_hz(topic)
            fast_enough = rate >= MIN_RATES.get(topic, 0.0)
            detail = f"last message {age:.1f}s ago, {rate:.1f} Hz"
            if not fast_enough:
                detail += f" (< {MIN_RATES[topic]:g} Hz)"
            rows.append((fresh and fast_enough, topic, detail))
            ok = ok and fresh and fast_enough
        return ok, rows

    def tf_report(self, pairs):
        rows = []
        ok = True
        for target, source in pairs:
            try:
                self.tf_buffer.lookup_transform(target, source, rclpy.time.Time(), timeout=Duration(seconds=0.2))
                rows.append((True, f"{target}->{source}", "available"))
            except TransformException as exc:
                rows.append((False, f"{target}->{source}", str(exc).split("\n", 1)[0]))
                ok = False
        return ok, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=8.0,
                        help="seconds to wait for first messages")
    parser.add_argument("--max-age", type=float, default=3.0,
                        help="maximum allowed topic age in seconds")
    parser.add_argument("--base-frame", default="base")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--skip-obstacles", action="store_true",
                        help="do not require /obstacle_tracker/state")
    args = parser.parse_args()

    topics = dict(DEFAULT_TOPICS)
    if args.skip_obstacles:
        topics.pop("/obstacle_tracker/state")

    rclpy.init()
    node = StackCheck(topics)
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline and any(stamp is None for stamp in node.seen.values()):
            rclpy.spin_once(node, timeout_sec=0.1)

        # Keep spinning ~2 s so the rate estimate has several samples.
        settle = time.monotonic() + 2.0
        while time.monotonic() < settle:
            rclpy.spin_once(node, timeout_sec=0.1)

        topics_ok, topic_rows = node.topic_report(args.max_age)
        fall_rows = node.fall_report()
        tf_ok, tf_rows = node.tf_report([
            (args.odom_frame, args.base_frame),
            (args.map_frame, args.base_frame),
        ])
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print("Go2 stack check")
    for good, name, detail in topic_rows:
        print(f"[{'OK' if good else 'FAIL'}] topic {name}: {detail}")
    for good, name, detail in tf_rows:
        print(f"[{'OK' if good else 'FAIL'}] tf {name}: {detail}")
    for good, name, detail in fall_rows:
        print(f"[{'OK' if good else 'FAIL'}] {name}: {detail}")

    return 0 if topics_ok and tf_ok and all(g for g, _, _ in fall_rows) else 1


if __name__ == "__main__":
    sys.exit(main())
