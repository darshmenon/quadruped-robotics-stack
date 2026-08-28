"""
Manual test client for action_interface_node.py -- publishes one action
command and prints status updates until a terminal state arrives.

Usage:
    python3 intelligence/action_interface/send_action.py stand
    python3 intelligence/action_interface/send_action.py walk_forward --duration 2
    python3 intelligence/action_interface/send_action.py wave_hand
"""

import argparse
import json
import time
import uuid

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from intelligence.action_interface.action_interface_node import VALID_ACTIONS


class SendAction(Node):
    def __init__(self, action, duration, timeout):
        super().__init__("send_action_client")
        self._req_id = str(uuid.uuid4())[:8]
        self._acked = False   # stop re-publishing once the node has seen the request
        self._done = False
        self._timeout = timeout
        self._start = time.time()

        self._cmd_pub = self.create_publisher(String, "/action_cmd", 10)
        self.create_subscription(String, "/action_status", self._status_cb, 10)

        payload = {"action": action, "id": self._req_id}
        if duration is not None:
            payload["duration"] = duration
        # republish until acked -- covers ROS2 discovery latency on first send
        self.create_timer(0.3, lambda: self._publish_once(payload))

    def _publish_once(self, payload):
        if not self._acked:
            self._cmd_pub.publish(String(data=json.dumps(payload)))

    def _status_cb(self, msg: String):
        status = json.loads(msg.data)
        if status.get("id") != self._req_id:
            return
        self._acked = True
        print(f"[{status['state']}] {status['action']}: {status.get('message', '')}")
        if status["state"] in ("succeeded", "failed", "rejected"):
            self._done = True

    def timed_out(self):
        return time.time() - self._start > self._timeout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=VALID_ACTIONS)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    rclpy.init()
    node = SendAction(args.action, args.duration, args.timeout)
    try:
        while rclpy.ok() and not node._done and not node.timed_out():
            rclpy.spin_once(node, timeout_sec=0.2)
        if node.timed_out():
            print("timed out waiting for a terminal status")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
