"""
Waypoint Navigator — autonomous point-to-point navigation for the quadruped.

Publishes velocity commands (/cmd_vel) to drive the robot through a list
of (x, y) waypoints using a simple pure pursuit controller.

Usage:
    ros2 run quadruped_dog_rl waypoint_navigator --ros-args \
        -p waypoints:="[[2.0,0.0],[2.0,2.0],[0.0,2.0],[0.0,0.0]]"
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import json


class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')

        self.declare_parameter('waypoints', [2.0, 0.0, 0.0, 0.0])
        self.declare_parameter('linear_speed', 0.5)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('angular_gain', 2.0)

        raw = self.get_parameter('waypoints').value
        self.waypoints = [[raw[i], raw[i+1]] for i in range(0, len(raw), 2)]
        self.linear_speed = self.get_parameter('linear_speed').get_parameter_value().double_value
        self.goal_tol = self.get_parameter('goal_tolerance').get_parameter_value().double_value
        self.ang_gain = self.get_parameter('angular_gain').get_parameter_value().double_value

        self.current_wp = 0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info(f'Waypoint navigator started. {len(self.waypoints)} waypoints.')

    def odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

    def control_loop(self):
        if self.current_wp >= len(self.waypoints):
            self.stop()
            self.get_logger().info('All waypoints reached.')
            return

        wx, wy = self.waypoints[self.current_wp]
        dx = wx - self.x
        dy = wy - self.y
        dist = math.sqrt(dx**2 + dy**2)

        if dist < self.goal_tol:
            self.get_logger().info(f'Reached waypoint {self.current_wp}: ({wx}, {wy})')
            self.current_wp += 1
            return

        target_yaw = math.atan2(dy, dx)
        yaw_error = math.atan2(math.sin(target_yaw - self.yaw), math.cos(target_yaw - self.yaw))

        cmd = Twist()
        cmd.linear.x = self.linear_speed * max(0.0, math.cos(yaw_error))
        cmd.angular.z = self.ang_gain * yaw_error
        self.cmd_pub.publish(cmd)

    def stop(self):
        self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = WaypointNavigator()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
