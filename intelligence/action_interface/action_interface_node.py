"""
Action Interface Node -- translates a small, fixed vocabulary of discrete
commands (walk_forward, stand, wave_hand, stop, ...) into the low-level
topics that already drive the robot: /cmd_vel and /cmd_pose for CHAMP
locomotion, /arm/target and /gripper/command for the manipulator arm.

This is the bridge everything downstream (voice pipeline, LLM commander,
web UI) plugs into: a caller never needs to know what a "wave" looks like
in joint space, only that "wave_hand" is a valid action name.

Subscriptions:
    /action_cmd    std_msgs/String   JSON: {"action": <str>, "duration": <float, optional>,
                                             "id": <str, optional>}

Publications:
    /action_status std_msgs/String   JSON: {"id":..., "action":..., "state":..., "message":...}
                                      state in: accepted, running, succeeded, failed, rejected
    /cmd_vel       geometry_msgs/Twist   drives champ_base quadruped_controller_node
    /cmd_pose      geometry_msgs/Pose    body height/tilt (champ_base)
    /arm/target    geometry_msgs/Point   consumed by arm_reach_node.py
    /gripper/command std_msgs/Float64    consumed by arm_reach_node.py

Actions (v1, all safe -- bounded duration, one active action at a time,
a new command preempts whatever is running):
    stand, crouch, sit          hold a body height, cmd_vel zeroed
    stop                        zero cmd_vel, return to stand height
    walk_forward, walk_backward linear.x for `duration` seconds, then auto-stop
    turn_left, turn_right       angular.z for `duration` seconds, then auto-stop
    wave_hand                   scripted arm waypoint sequence, then re-stow

Requires champ_base's quadruped_controller_node (and, for wave_hand,
arm_reach_node.py) already running -- this node only ever publishes
commands, it does not launch anything.

Usage:
    python3 intelligence/action_interface/action_interface_node.py

    ros2 topic pub /action_cmd std_msgs/msg/String '{data: "{\\"action\\": \\"stand\\"}"}'
    ros2 topic echo /action_status
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, Point, Vector3
from std_msgs.msg import String, Float64

MAX_DURATION = 10.0  # hard cap (s) on any timed action, regardless of request

# height_offset values match intelligence/gait/gait_scheduler.py's GAITS table
POSE_ACTIONS = {
    "stand":  0.0,
    "crouch": -0.10,
    "sit":    -0.16,
}

VELOCITY_ACTIONS = {
    # name: (linear_x, angular_z, default_duration_s)
    "walk_forward":  (0.4, 0.0, 3.0),
    "walk_backward": (-0.3, 0.0, 3.0),
    "turn_left":     (0.0, 0.6, 2.0),
    "turn_right":    (0.0, -0.6, 2.0),
}

# (x, y, z) arm target, gripper width, hold seconds -- a friendly side-to-side
# wave, waypoints reused from arm_reach_demo.py's verified-reachable set.
WAVE_SEQUENCE = [
    (0.42, 0.0, 0.15, 0.0, 0.4),
    (0.35, -0.24, 0.15, 0.02, 0.35),
    (0.35, 0.24, 0.15, 0.02, 0.35),
    (0.35, -0.24, 0.15, 0.02, 0.35),
    (0.35, 0.24, 0.15, 0.02, 0.35),
]
ARM_STOW_TARGET = (0.2, 0.0, -0.05)  # roughly matches arm_reach_node.STOW_POSE

VALID_ACTIONS = sorted(set(POSE_ACTIONS) | set(VELOCITY_ACTIONS) | {"stop", "wave_hand"})

CONTROL_RATE_HZ = 20.0


class ActionInterfaceNode(Node):
    def __init__(self):
        super().__init__("action_interface_node")

        self._cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._cmd_pose_pub = self.create_publisher(Pose, "/cmd_pose", 10)
        self._arm_target_pub = self.create_publisher(Point, "/arm/target", 10)
        self._gripper_pub = self.create_publisher(Float64, "/gripper/command", 10)
        self._status_pub = self.create_publisher(String, "/action_status", 10)

        self.create_subscription(String, "/action_cmd", self._cmd_cb, 10)

        self._pose_z = 0.0            # persistent body-height target
        self._velocity_until = None   # rclpy Time when a timed velocity action ends
        self._velocity_twist = Twist()

        self._wave_steps = []         # remaining (target, gripper, hold) steps
        self._wave_deadline = None
        self._active = None           # {"id", "action"} of the in-flight action

        self.create_timer(1.0 / CONTROL_RATE_HZ, self._control_loop)
        self.get_logger().info(
            f"action_interface_node ready. Valid actions: {VALID_ACTIONS}"
        )

    # ------------------------------------------------------------------ #

    def _publish_status(self, req_id, action, state, message=""):
        self._status_pub.publish(String(data=json.dumps({
            "id": req_id, "action": action, "state": state, "message": message,
        })))

    def _cmd_cb(self, msg: String):
        try:
            payload = json.loads(msg.data)
            action = str(payload.get("action", "")).strip().lower()
        except (json.JSONDecodeError, AttributeError) as e:
            self.get_logger().warn(f"bad /action_cmd payload: {e}")
            return

        req_id = payload.get("id")
        duration = payload.get("duration")

        if action not in VALID_ACTIONS:
            self._publish_status(req_id, action, "rejected", f"unknown action, choose from {VALID_ACTIONS}")
            return

        # A new command preempts whatever is currently running.
        self._stop_all_motion()
        self._active = {"id": req_id, "action": action}
        self._publish_status(req_id, action, "accepted")
        self.get_logger().info(f'executing action "{action}" (id={req_id})')

        if action == "stop":
            self._pose_z = 0.0
            self._publish_status(req_id, action, "succeeded")
            self._active = None
        elif action in POSE_ACTIONS:
            self._pose_z = POSE_ACTIONS[action]
            self._publish_status(req_id, action, "succeeded")
            self._active = None
        elif action in VELOCITY_ACTIONS:
            vx, wz, default_s = VELOCITY_ACTIONS[action]
            seconds = min(float(duration), MAX_DURATION) if duration else default_s
            seconds = max(0.0, seconds)
            self._velocity_twist = Twist(linear=Vector3(x=vx, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=wz))
            self._velocity_until = self.get_clock().now().nanoseconds / 1e9 + seconds
            self._publish_status(req_id, action, "running")
        elif action == "wave_hand":
            self._wave_steps = list(WAVE_SEQUENCE)
            self._wave_deadline = None
            self._publish_status(req_id, action, "running")

    def _stop_all_motion(self):
        self._velocity_until = None
        self._velocity_twist = Twist()
        self._wave_steps = []
        self._wave_deadline = None
        self._cmd_vel_pub.publish(Twist())

    # ------------------------------------------------------------------ #

    def _control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9

        # timed velocity action
        if self._velocity_until is not None:
            if now < self._velocity_until:
                self._cmd_vel_pub.publish(self._velocity_twist)
            else:
                self._cmd_vel_pub.publish(Twist())
                self._velocity_until = None
                if self._active:
                    self._publish_status(self._active["id"], self._active["action"], "succeeded")
                    self._active = None

        # scripted wave_hand sequence
        if self._wave_steps or self._wave_deadline is not None:
            if self._wave_deadline is None:
                x, y, z, grip, hold = self._wave_steps.pop(0)
                self._arm_target_pub.publish(Point(x=x, y=y, z=z))
                self._gripper_pub.publish(Float64(data=grip))
                self._wave_deadline = now + hold
            elif now >= self._wave_deadline:
                self._wave_deadline = None
                if not self._wave_steps:
                    sx, sy, sz = ARM_STOW_TARGET
                    self._arm_target_pub.publish(Point(x=sx, y=sy, z=sz))
                    self._gripper_pub.publish(Float64(data=0.0))
                    if self._active:
                        self._publish_status(self._active["id"], self._active["action"], "succeeded")
                        self._active = None

        # persistent body-height hold (stand/crouch/sit)
        self._cmd_pose_pub.publish(Pose(position=Point(x=0.0, y=0.0, z=self._pose_z)))


def main():
    rclpy.init()
    node = ActionInterfaceNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node._cmd_vel_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
