#!/usr/bin/env python3
"""Send the Go2 around a list of map-frame waypoints via Nav2.

Usage:
    python3 scripts/patrol_go2.py --waypoints 1,0 1,1 0,1 0,0 --loops 2
"""

import argparse
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


def parse_waypoints(items):
    pts = []
    for item in items:
        x, y = item.split(",")
        pts.append((float(x), float(y)))
    return pts


class Patrol(Node):
    def __init__(self):
        super().__init__("patrol_go2")
        self.client = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def go(self, x, y, yaw=0.0):
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x, pose.pose.position.y = x, y
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)
        goal.pose = pose
        handle_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, handle_future)
        handle = handle_future.result()
        if not handle.accepted:
            self.get_logger().warn(f"goal ({x}, {y}) rejected")
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        ok = result_future.result().status == 4  # STATUS_SUCCEEDED
        self.get_logger().info(f"goal ({x}, {y}) {'reached' if ok else 'failed'}")
        return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waypoints", nargs="+", required=True, help="x,y pairs")
    ap.add_argument("--loops", type=int, default=1)
    args = ap.parse_args()
    pts = parse_waypoints(args.waypoints)
    rclpy.init()
    node = Patrol()
    node.client.wait_for_server()
    for _ in range(args.loops):
        for x, y in pts:
            node.go(x, y)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
